"""Canonical TopoRig identities and dataset layout (no tensor or Hub dependencies)."""
from __future__ import annotations

import os
from pathlib import Path
import re

LAYOUT_VERSION = 1

FORMAT = "toporig-webdataset-v1"
COMPONENTS = ("neutral_imgs", "expressed_imgs", "meshes_ict", "meshes_custom")
MANIFEST_NAME = "toporig_manifest.json"
MIGRATION_MARKER = ".toporig-migration-in-progress.json"
PACKAGE_MARKER = ".toporig-packaging-in-progress.json"
NEUTRAL_IMAGE_RE = re.compile(r"^((?:v2_)?img_\d{5})(?:_2)?\.png$")
EXPRESSED_IMAGE_RE = re.compile(r"^((?:v2_)?img_\d{5})(?:_4)?-(\d{2})\.png$")
IDENTITY_RE = re.compile(
    r"^(v2_)?(?:img_)?(\d{5})(?:_2(?:_fit(?:_100k_faces)?)?)?"
    r"(?:\.(?:png|fbx|json))?$"
)


def canonical_identity(value: object) -> str:
    """Normalize only known identity forms; never discard the v2 namespace."""
    value = str(value).strip()
    if value.isdigit() and len(value) <= 5:
        value = value.zfill(5)
    match = IDENTITY_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid TopoRig identity: {value!r}")
    return f"{match.group(1) or ''}img_{match.group(2)}"


def assert_dataset_ready(path: str | Path) -> None:
    """Fail closed while a source-tree migration or rollback is incomplete."""
    path = Path(path).expanduser().resolve()
    for root in (path, *path.parents):
        if (root / PACKAGE_MARKER).exists():
            raise ValueError(f'TopoRig dataset packaging is incomplete: {root}')
        if (root / MIGRATION_MARKER).exists():
            raise ValueError(f'TopoRig dataset migration is incomplete: {root}; resume or roll back its journal')


def person_id(value: object) -> str:
    """Keep legacy numeric person IDs, with an explicit namespace for v2 IDs."""
    return canonical_identity(value).replace("img_", "", 1)


def reference_assets_root() -> Path:
    """Prefer an explicit root, then references included in the dataset release."""
    project = Path(__file__).resolve().parents[1]
    data = Path(os.environ.get("TOPORIG_DATA_ROOT", str(project / "data"))).expanduser()
    if not data.is_absolute():
        data = project / data
    default = data / "reference_assets"
    if not default.is_dir():
        default = project / "reference_assets"
    path = Path(os.environ.get("TOPORIG_REFERENCE_ROOT", str(default))).expanduser()
    return (path if path.is_absolute() else project / path).resolve()


def mesh_path(root: Path, source: str, identity: str, suffix: str = ".fbx") -> Path:
    if source not in {"ict", "custom"} or suffix not in {".fbx", ".json"}:
        raise ValueError("Expected an ict/custom mesh source and FBX/JSON suffix")
    return Path(root) / f"meshes_{source}" / f"{canonical_identity(identity)}{suffix}"
