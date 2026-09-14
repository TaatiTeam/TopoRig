"""Shared YAML loading for training, cache tools and historical checkpoint configs."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1

def load_config(
    path: Path,
    _stack: Optional[tuple[Path, ...]] = None,
    *, phase: Optional[int] = None,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    stack = _stack or ()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Cyclic config inheritance: {chain}")

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected {path} to contain a YAML mapping.")
    if loaded.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported config schema version: {loaded['schema_version']}")
    if loaded.get("recipe") in {"final", "stage1", "stage2"} and "data" not in loaded and not loaded.get("extends"):
        from utils.training_config import expand_training_config
        return expand_training_config(loaded, phase=phase)

    parent_value = loaded.pop("extends", None)
    if parent_value in (None, ""):
        return resolve_config_paths(loaded, path.parent)
    parent_path = Path(str(parent_value)).expanduser()
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    parent = load_config(parent_path, (*stack, path), phase=phase)
    resolved = resolve_config_paths(
        deep_merge_config(parent, loaded),
        path.parent,
    )
    if resolved.get("schema_version") == SCHEMA_VERSION:
        from utils.training_config import resolve_model_dimensions
        resolve_model_dimensions(resolved)
    return resolved


def deep_merge_config(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge_config(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_config_paths(config: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    project_root = ROOT

    def walk(value: Any, key: str = "") -> Any:
        if isinstance(value, MutableMapping):
            return {name: walk(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [walk(item, key) for item in value]
        if not isinstance(value, str) or not value:
            return value
        if not key.endswith(("_dir", "_csv", "_path", "_root", "_file", "output_dir", "_checkpoint")):
            return value
        value = os.path.expandvars(value)
        if "$" in value:
            raise ValueError(f"Unresolved environment variable in {key}: {value}")
        path = Path(value).expanduser()
        if path.is_absolute():
            return str(path)
        if value.startswith("./") or value.startswith("../"):
            return str((config_dir / path).resolve())
        return str((project_root / path).resolve())

    return walk(resolved)


def save_config(config: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)
