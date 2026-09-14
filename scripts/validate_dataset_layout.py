#!/usr/bin/env python3
"""Validate a canonical directory release and its portable metadata.

By default checks manifest integrity, schema, actual file counts/sizes, full
identity/AU pairing, AU inventory, exclusions and any supplied split assignments.
Use --checksums for a full content rehash and --images for PNG integrity checks.
Missing historical metadata is reported separately from layout validity.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dataset.hf_assets import atomic_json, load_manifest, relative_path, sha256_file
from dataset.layout import COMPONENTS, LAYOUT_VERSION, MANIFEST_NAME, assert_dataset_ready, canonical_identity


def validate_shards(root: Path, *, checksums=False, images=False, workers=8) -> dict:
    """Check a release in place without materializing another full dataset copy."""
    manifest, version = load_manifest(root)
    layout = json.loads((root / 'metadata/layout.json').read_text())
    inventory_path = root / 'metadata/manifest.jsonl'
    if sha256_file(inventory_path) != layout['manifest_sha256']:
        raise ValueError('Manifest checksum mismatch')
    inventory = [json.loads(line) for line in inventory_path.read_text().splitlines()]
    indexed = {row['path']: row for row in manifest['files']}
    if len(inventory) != len(indexed) or {r['path'] for r in inventory} != set(indexed):
        raise ValueError('Shard inventory differs from canonical metadata')
    for row in inventory:
        if any(indexed[row['path']][key] != row[key]
               for key in ('identity', 'action_unit_id', 'size', 'sha256')):
            raise ValueError(f'Shard entry differs from canonical metadata: {row["path"]}')
    expected = {}
    for entry in manifest['files']:
        expected.setdefault(entry['shard'], {})[entry['member']] = entry
    actual = {p.relative_to(root).as_posix() for c in COMPONENTS for p in (root / c).glob('*.tar')}
    declared = {shard['path'] for shard in manifest['shards']}
    if actual != declared:
        raise ValueError(f'Shard inventory differs: missing={len(declared-actual)}, extra={len(actual-declared)}')

    def check(shard):
        path = root / shard['path']
        if path.stat().st_size != shard['size']:
            raise ValueError(f'Shard size mismatch: {path}')
        if checksums and sha256_file(path) != shard['sha256']:
            raise ValueError(f'Shard checksum mismatch: {path}')
        entries = expected.get(shard['path'], {})
        with tarfile.open(path, 'r:') as archive:
            members = archive.getmembers()
            by_name = {member.name: member for member in members}
            if len(by_name) != len(members) or any(not m.isfile() for m in members):
                raise ValueError(f'Duplicate or non-regular archive members: {path}')
            for member in members:
                relative_path(member.name)
            for name, entry in entries.items():
                if name not in by_name or by_name[name].size != entry['size']:
                    raise ValueError(f'Missing/invalid archive member: {path}:{name}')
                if images and entry['path'].endswith('.png'):
                    from PIL import Image
                    with archive.extractfile(by_name[name]) as handle, Image.open(handle) as image:
                        if image.format != 'PNG':
                            raise ValueError(f'Expected PNG: {name}')
                        image.verify()
        print(f'Validated {shard["path"]}', flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(check, manifest['shards']):
            pass
    for entry in manifest.get('ancillary_files', []):
        path = root / relative_path(entry['path'])
        if not path.is_file() or path.stat().st_size != entry['size']:
            raise ValueError(f'Ancillary file size mismatch: {path}')
        if checksums and sha256_file(path) != entry['sha256']:
            raise ValueError(f'Ancillary file checksum mismatch: {path}')
    return {'status': 'valid', 'format': manifest['format'], 'manifest_sha256': version,
            'files': len(inventory), 'shards': len(declared),
            'counts': dict(Counter(row['path'].split('/')[0] for row in inventory)),
            'sha256_verified': checksums, 'png_integrity_verified': images,
            'missing_split_files': [name for name in ('ict_split.csv', 'custom_split.csv')
                                    if not (root / 'metadata' / name).is_file()],
            'missing_sidecars': {c: len(ids) for c, ids in layout['missing_sidecars'].items()},
            'note': 'Integrity validation does not certify historical metadata or runtime dependencies.'}


def validate(root: Path, *, checksums: bool = False, images: bool = False, workers: int = 8) -> dict:
    root = root.resolve()
    assert_dataset_ready(root)
    if (root / MANIFEST_NAME).is_file():
        return validate_shards(root, checksums=checksums, images=images, workers=workers)
    metadata = root / 'metadata'
    layout = json.loads((metadata / 'layout.json').read_text())
    if layout.get('layout_version') != LAYOUT_VERSION or layout.get('status') != 'complete':
        raise ValueError('Unsupported/incomplete directory layout')
    manifest_path = metadata / 'manifest.jsonl'
    if sha256_file(manifest_path) != layout['manifest_sha256']:
        raise ValueError('Manifest checksum mismatch')
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    seen, inventory, by_component = set(), {}, {c: set() for c in COMPONENTS}
    for row in rows:
        path = relative_path(row['path'])
        component, name = path.split('/')
        identity, au = row['identity'], row['action_unit_id']
        if component not in COMPONENTS or component != row['component'] or canonical_identity(identity) != identity:
            raise ValueError(f'Invalid identity/source: {row}')
        suffix = Path(name).suffix
        if component == 'expressed_imgs':
            if type(au) is not int or au not in range(53):
                raise ValueError(f'Invalid AU: {row}')
            expected = f'{identity}-{au:02d}.png'
        else:
            if au is not None or suffix not in ({'.png'} if component == 'neutral_imgs' else {'.fbx', '.json'}):
                raise ValueError(f'Invalid asset extension/AU: {row}')
            expected = identity + suffix
        if name != expected or path in seen or not re.fullmatch('[0-9a-f]{64}', row['sha256']):
            raise ValueError(f'Duplicate/invalid manifest record: {row}')
        seen.add(path)
        by_component[component].add(identity)
        item = inventory.setdefault(identity, {'identity': identity, 'action_units': [], 'collections': []})
        if component not in item['collections']:
            item['collections'].append(component)
        if au is not None:
            item['action_units'].append(au)
    actual = set()
    for component in COMPONENTS:
        directory = root / component
        if directory.is_symlink():
            raise ValueError(f'Symlink collection: {directory}')
        for path in directory.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError(f'Non-data entry in canonical collection: {path}')
            actual.add(path.relative_to(root).as_posix())
    if seen != actual:
        raise ValueError(f'File inventory differs: missing={len(seen-actual)}, extra={len(actual-seen)}')
    counts = dict(Counter(row['component'] for row in rows))
    if len(rows) != layout['file_count'] or counts != layout['counts']:
        raise ValueError('Layout counts disagree with manifest')
    for row in rows:
        if row['identity'] not in by_component['neutral_imgs']:
            raise ValueError(f'Asset without neutral identity: {row}')
        if row['path'].endswith('.json') and row['path'][:-5] + '.fbx' not in seen:
            raise ValueError(f'Orphan sidecar: {row}')
    recorded = [json.loads(line) for line in (metadata / 'availability.jsonl').read_text().splitlines()]
    if len(recorded) != len(inventory) or {x['identity']: x for x in recorded} != inventory:
        raise ValueError('Availability/AU inventory differs from files')
    aus = json.loads((metadata / 'action_units.json').read_text())
    if set(aus) != {str(i) for i in range(53)}:
        raise ValueError('Expected the original 53 AU controls')
    exclusions = {}
    for component, name in [('meshes_ict', 'bad_ict_source_landmark_meshes.txt'),
                            ('meshes_custom', 'bad_blendshape_landmark_meshes.txt'),
                            ('expressed_imgs', 'bad_waft_blink_left_alignment_person_ids.txt')]:
        keys = [line.split('#', 1)[0].strip() for line in (metadata / name).read_text().splitlines()]
        keys = [x for x in keys if x]
        if len(keys) != len(set(keys)) or any(canonical_identity(x) != x for x in keys):
            raise ValueError(f'Noncanonical/duplicate exclusion: {name}')
        if set(keys) - by_component['neutral_imgs']:
            raise ValueError(f'Unknown excluded identity: {name}')
        exclusions[component] = len(keys)
    missing_splits, validated_splits = [], {}
    for component, name, column in [('meshes_ict', 'ict_split.csv', 'mesh_id'),
                                    ('meshes_custom', 'custom_split.csv', 'mesh_id'),
                                    ('neutral_imgs', 'image_split.csv', 'person_id')]:
        path = metadata / name
        if not path.is_file():
            missing_splits.append(name)
            continue
        identities = set()
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                key = row[column]
                if canonical_identity(key) != key or key in identities or key not in by_component[component] or row['split'] not in {'train', 'val', 'test'}:
                    raise ValueError(f'Invalid split assignment in {name}: {row}')
                identities.add(key)
        validated_splits[name] = len(identities)

    def check(row):
        path = root / row['path']
        if path.stat().st_size != row['size']:
            raise ValueError(f'Size mismatch: {path}')
        if checksums and sha256_file(path) != row['sha256']:
            raise ValueError(f'Content checksum mismatch: {path}')
        if images and path.suffix == '.png':
            from PIL import Image
            with Image.open(path) as image:
                if image.format != 'PNG':
                    raise ValueError(f'Expected PNG: {path}')
                image.verify()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, _ in enumerate(pool.map(check, rows), 1):
            if index % 10000 == 0:
                print(f'Validated {index}/{len(rows)} assets', flush=True)
    missing_sidecars = {c: sorted(Path(p).stem for p in seen if p.startswith(c + '/') and p.endswith('.fbx') and p[:-4] + '.json' not in seen)
                        for c in ('meshes_ict', 'meshes_custom')}
    if missing_sidecars != layout['missing_sidecars']:
        raise ValueError('Missing sidecar inventory disagrees with layout')
    return {'status': 'valid', 'files': len(rows), 'counts': counts, 'identities': len(inventory),
            'sha256_verified': checksums, 'png_integrity_verified': images,
            'exclusions': exclusions, 'missing_sidecars': {k: len(v) for k, v in missing_sidecars.items()},
            'missing_split_files': missing_splits, 'validated_split_files': validated_splits,
            'training_ready': not missing_splits and not any(missing_sidecars.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--checksums', action='store_true')
    parser.add_argument('--images', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    result = validate(args.data_root, checksums=args.checksums, images=args.images, workers=args.workers)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
