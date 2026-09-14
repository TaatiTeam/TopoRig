"""TopoRig's portable identity names and manifest-indexed WebDataset assets.

The training datasets remain map-style PyTorch datasets. This adapter materializes
the required source files in a separate cache because Blender/FBX and the existing
WAFT pipeline need real filenames. It never extracts into a Hub snapshot.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import tempfile
from typing import Iterable, Iterator, Optional


from dataset.layout import (FORMAT, COMPONENTS, MANIFEST_NAME, NEUTRAL_IMAGE_RE,
                            EXPRESSED_IMAGE_RE, IDENTITY_RE, canonical_identity, person_id,
                            assert_dataset_ready)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Serialize cache/build publication across workers and DDP ranks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Invalid relative asset path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in {".", ".."} for p in value.split("/")):
        raise ValueError(f"Unsafe relative asset path: {value!r}")
    if str(path) != value:
        raise ValueError(f"Non-canonical relative asset path: {value!r}")
    return value


def _checked_path(root: Path, name: str) -> Path:
    path = root / relative_path(name)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Asset path escapes its root: {name!r}")
    return path


def load_manifest(root: Path) -> tuple[dict, str]:
    raw = (root / MANIFEST_NAME).read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ValueError(f"Unsupported TopoRig asset format in {root / MANIFEST_NAME}")
    shards = {}
    for shard in manifest["shards"]:
        path = relative_path(shard["path"])
        if path in shards or shard["component"] not in COMPONENTS:
            raise ValueError(f"Duplicate/invalid shard: {path}")
        if not re.fullmatch(r"[0-9a-f]{64}", shard["sha256"]):
            raise ValueError(f"Invalid shard checksum: {path}")
        if not isinstance(shard["size"], int) or shard["size"] < 0:
            raise ValueError(f"Invalid shard size: {path}")
        shards[path] = shard
    seen = set()
    members = set()
    for entry in manifest["files"]:
        path = relative_path(entry["path"])
        member = relative_path(entry["member"])
        shard = relative_path(entry["shard"])
        component = path.split("/", 1)[0]
        if component not in COMPONENTS or shard not in shards:
            raise ValueError(f"Unknown component/shard for {path}")
        if component != shards[shard]["component"]:
            raise ValueError(f"Mismatched component/shard for {path}")
        if path in seen or (shard, member) in members:
            raise ValueError(f"Duplicate asset path/member: {path}")
        if not isinstance(entry["size"], int) or entry["size"] < 0:
            raise ValueError(f"Invalid size for {path}")
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError(f"Invalid checksum for {path}")
        identity = canonical_identity(entry["identity"])
        if identity != entry["identity"]:
            raise ValueError(f"Non-canonical identity for {path}")
        au = entry.get("action_unit_id")
        if au is not None and (type(au) is not int or not 0 <= au < 53):
            raise ValueError(f"Invalid action unit for {path}")
        if component == "expressed_imgs":
            expected = f"{component}/{identity}-{au:02d}.png" if au is not None else ""
        else:
            suffix = Path(path).suffix
            allowed = {".png"} if component == "neutral_imgs" else {".fbx", ".json"}
            expected = f"{component}/{identity}{suffix}" if suffix in allowed and au is None else ""
        if path != expected:
            raise ValueError(f"Asset path disagrees with identity/AU metadata: {path}")
        seen.add(path)
        members.add((shard, member))
    return manifest, hashlib.sha256(raw).hexdigest()


def prepare_assets(
    data_root: str | Path,
    cache_root: str | Path,
    components: Iterable[str],
    *,
    identities: Optional[Iterable[str]] = None,
    action_units: Optional[Iterable[int]] = None,
) -> Path:
    """Return canonical source directories, materializing selected shards once.

    A plain renamed directory also works. Sharded data must first be downloaded
    (``download_assets`` below, or ``snapshot_download``). Content-verified files
    are reused; cache publication is atomic and protected by a process lock.
    """
    source = Path(data_root).expanduser().resolve()
    assert_dataset_ready(source)
    components = set(components)
    if not components or not components.issubset(COMPONENTS):
        raise ValueError(f"components must be selected from {COMPONENTS}")
    if not (source / MANIFEST_NAME).is_file():
        for component in components:
            if not (source / component).is_dir():
                raise FileNotFoundError(
                    f"Missing {source / component}; supply a renamed dataset root or "
                    f"a downloaded shard release containing {MANIFEST_NAME}."
                )
        return source
    manifest, version = load_manifest(source)
    identity_set = None if identities is None else {canonical_identity(x) for x in identities}
    au_set = None if action_units is None else set(action_units)
    entries = [
        item for item in manifest["files"]
        if item["path"].split("/", 1)[0] in components
        and (identity_set is None or item["identity"] in identity_set)
        and (au_set is None or item.get("action_unit_id") is None
             or item["action_unit_id"] in au_set)
    ]
    destination = Path(cache_root).expanduser().resolve() / FORMAT / version
    if destination.is_relative_to(source):
        raise ValueError("asset_cache_dir must be outside the downloaded dataset root")
    destination.mkdir(parents=True, exist_ok=True)
    with file_lock(destination / ".extract.lock"):
        stamp_path = destination / ".verified.json"
        stamps = json.loads(stamp_path.read_text()) if stamp_path.exists() else {}
        pending: dict[str, list[dict]] = {}
        for entry in entries:
            path = _checked_path(destination, entry["path"])
            stat = path.stat() if path.exists() else None
            stamp = [stat.st_size, stat.st_mtime_ns, entry["sha256"]] if stat else None
            if stamp is not None and stamp[0] == entry["size"] and stamps.get(entry["path"]) == stamp:
                continue
            pending.setdefault(entry["shard"], []).append(entry)
        shards = {s["path"]: s for s in manifest["shards"]}
        for shard_name, selected in pending.items():
            # Hub snapshot shard files can be symlinks into the Hub blob cache.
            shard_path = source / relative_path(shard_name)
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing shard {shard_path}; download the required components first")
            if shard_path.stat().st_size != shards[shard_name]["size"]:
                raise ValueError(f"Shard size mismatch: {shard_path}")
            if sha256_file(shard_path) != shards[shard_name]["sha256"]:
                raise ValueError(f"Shard checksum mismatch: {shard_path}")
            print(f"[assets] Preparing {len(selected)} files from {shard_name}", file=sys.stderr, flush=True)
            with tarfile.open(shard_path, "r:") as archive:
                # getmembers/getmember do not extract links or arbitrary paths.
                names = [m.name for m in archive.getmembers()]
                if len(names) != len(set(names)):
                    raise ValueError(f"Duplicate archive members in {shard_name}")
                for entry in selected:
                    member = archive.getmember(entry["member"])
                    if not member.isfile() or member.size != entry["size"]:
                        raise ValueError(f"Invalid archive member {entry['member']}")
                    path = _checked_path(destination, entry["path"])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    fd, temporary = tempfile.mkstemp(prefix=".asset-", dir=path.parent)
                    try:
                        with os.fdopen(fd, "wb") as output, archive.extractfile(member) as handle:
                            digest = hashlib.sha256()
                            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                                digest.update(block)
                                output.write(block)
                        if digest.hexdigest() != entry["sha256"]:
                            raise ValueError(f"Asset checksum mismatch: {entry['path']}")
                        os.replace(temporary, path)
                    finally:
                        Path(temporary).unlink(missing_ok=True)
                    stat = path.stat()
                    stamps[entry["path"]] = [stat.st_size, stat.st_mtime_ns, entry["sha256"]]
            atomic_json(stamp_path, stamps)
        for component in components:
            (destination / component).mkdir(exist_ok=True)
    return destination


def asset_cache_namespace(data_root: str | Path) -> str:
    root = Path(data_root).expanduser().resolve()
    if (root / MANIFEST_NAME).is_file():
        return sha256_file(root / MANIFEST_NAME)[:20]
    from dataset.layout import LAYOUT_VERSION
    digest = hashlib.sha256(f"{LAYOUT_VERSION}:{root}".encode())
    layout = root / "metadata/layout.json"
    if layout.is_file():
        digest.update(layout.read_bytes())
    return digest.hexdigest()[:20]


def versioned_cache_dir(cache_root: str | Path, data_root: str | Path, component: str) -> Path:
    """Keep derived tensors outside assets and separate layouts and sources."""
    cache = Path(cache_root).expanduser().resolve()
    source = Path(data_root).expanduser().resolve()
    if cache.is_relative_to(source):
        raise ValueError("cache_dir must be outside the read-only dataset root")
    if component not in {*COMPONENTS, "image"}:
        raise ValueError(f"Invalid cache component: {component}")
    return cache / "portable" / asset_cache_namespace(source) / component


def download_assets(repo_id: str, *, revision: str, local_dir: Optional[str] = None,
                    components: Iterable[str] = COMPONENTS) -> Path:
    from huggingface_hub import HfApi, snapshot_download

    components = set(components)
    if not components or not components.issubset(COMPONENTS):
        raise ValueError(f"components must be selected from {COMPONENTS}")
    # Resolve moving refs once so metadata and shards always share one revision.
    commit = HfApi().repo_info(repo_id=repo_id, repo_type="dataset", revision=revision).sha
    root = Path(snapshot_download(repo_id, repo_type="dataset", revision=commit,
                                  local_dir=local_dir,
                                  allow_patterns=[MANIFEST_NAME, "README.md", "LICENSE*",
                                                  "REPRODUCING.md", "metadata/**", "reference_assets/**"]))
    manifest, _ = load_manifest(root)
    selected = [s["path"] for s in manifest["shards"] if s["component"] in components]
    if selected:
        snapshot_download(repo_id, repo_type="dataset", revision=commit,
                          local_dir=local_dir, allow_patterns=selected)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", help="Hugging Face dataset repository to download")
    parser.add_argument("--revision", default="main", help="Commit, tag, or branch; resolved to a pinned commit")
    parser.add_argument("--data-root", type=Path, help="Already downloaded dataset root")
    parser.add_argument("--local-dir", help="Optional download directory")
    parser.add_argument("--cache-dir", type=Path, help="Materialize source files into this separate cache")
    parser.add_argument("--components", nargs="+", choices=COMPONENTS, default=list(COMPONENTS))
    args = parser.parse_args()
    if bool(args.repo_id) == bool(args.data_root):
        parser.error("Supply exactly one of --repo-id and --data-root")
    root = download_assets(args.repo_id, revision=args.revision, local_dir=args.local_dir,
                           components=args.components) if args.repo_id else args.data_root
    if args.cache_dir:
        root = prepare_assets(root, args.cache_dir, args.components)
    print(root)


if __name__ == "__main__":
    main()
