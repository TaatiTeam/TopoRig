"""Shared sample selection and sharding for training cache preparation."""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, TypeVar

from dataset.image_dataset import (
    NEUTRAL_IMAGE_RE,
    _normalize_action_unit_ids,
    _normalize_person_id,
    _normalize_person_id_filter,
)
from dataset.hf_assets import prepare_assets, versioned_cache_dir


T = TypeVar("T")


def env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


def resolve_shard_args(args: argparse.Namespace) -> None:
    task_id = env_int("SLURM_ARRAY_TASK_ID")
    task_min = env_int("SLURM_ARRAY_TASK_MIN")
    task_max = env_int("SLURM_ARRAY_TASK_MAX")
    task_count = env_int("SLURM_ARRAY_TASK_COUNT")

    if args.i is None:
        if task_id is None:
            args.i = 1
        elif task_min is not None and task_min != 1:
            args.i = task_id - task_min + 1
        else:
            args.i = task_id

    if args.n is None:
        if task_count is not None:
            args.n = task_count
        elif task_min is not None and task_max is not None:
            args.n = task_max - task_min + 1
        else:
            args.n = 1


def validate_shard_args(shard_index: int, shard_count: int) -> None:
    if shard_count < 1:
        raise ValueError("--n must be at least 1.")
    if shard_index < 1 or shard_index > shard_count:
        raise ValueError("--i must be between 1 and --n.")


def shard_items(items: Sequence[T], shard_index: int, shard_count: int) -> list[T]:
    validate_shard_args(shard_index, shard_count)
    start = (len(items) * (shard_index - 1)) // shard_count
    end = (len(items) * shard_index) // shard_count
    return list(items[start:end])


def load_config(path: Path) -> dict[str, Any]:
    from utils.config import load_config as load_training_config

    return load_training_config(path)


def image_args_from_config(config: dict[str, Any], split: str) -> dict[str, Any]:
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("Config is missing a data mapping.")
    split_config = data.get(split)
    if not isinstance(split_config, dict):
        raise ValueError(f"Config is missing data.{split}.")
    image_args = split_config.get("image_args")
    if not isinstance(image_args, dict):
        raise ValueError(f"Config is missing data.{split}.image_args.")
    image_args = dict(image_args)
    data_root = image_args.get("data_root", data.get("data_root"))
    if data_root is not None:
        cache_dir = Path(image_args.get("cache_dir", ".cache/image_dataset")).expanduser()
        root = prepare_assets(
            data_root,
            image_args.get("asset_cache_dir", data.get("asset_cache_dir"))
            or cache_dir.parent / "hf_assets",
            ("neutral_imgs", "expressed_imgs", "meshes_custom"),
            identities=_normalize_person_id_filter(image_args.get("person_ids")),
            action_units=_normalize_action_unit_ids(image_args.get("action_units")),
        )
        image_args.update(
            neutral_dir=str(root / "neutral_imgs"), expressed_dir=str(root / "expressed_imgs"),
            neutral_mesh_dir=str(root / "meshes_custom"), neutral_mesh_filename_template="{identity}.fbx",
            neutral_mesh_filename_fallback_template=None,
            mediapipe_landmark_dir=str(root / "meshes_custom"),
            mediapipe_landmark_filename_template="{identity}.json",
            mediapipe_landmark_filename_fallback_template=None,
            cache_dir=str(versioned_cache_dir(cache_dir, data_root, "image")),
        )
    return image_args


def list_neutral_images(neutral_dir: Path) -> dict[str, Path]:
    if not neutral_dir.is_dir():
        raise FileNotFoundError(neutral_dir)
    neutral_images: dict[str, Path] = {}
    for path in sorted(neutral_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith("v2_") and path.name.endswith("_2.png"):
            continue  # Preserve legacy recipes; portable releases use clean names.
        match = NEUTRAL_IMAGE_RE.match(path.name)
        if match is not None:
            identity = _normalize_person_id(match.group(1))
            if identity in neutral_images:
                raise ValueError(f"Duplicate neutral identity: {identity}")
            neutral_images[identity] = path
    return neutral_images


def split_person_ids(split_csv: Optional[Path], split: str) -> Optional[set[str]]:
    if split.lower() == "all":
        return None
    if split_csv is None:
        raise ValueError("split_csv is required unless split='all'.")
    if not split_csv.is_file():
        raise FileNotFoundError(split_csv)

    with split_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{split_csv} is empty.")
        missing = {"person_id", "split"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{split_csv} is missing columns: {sorted(missing)}")
        return {
            _normalize_person_id(row["person_id"])
            for row in reader
            if row["split"].strip().lower() == split.lower()
        }
