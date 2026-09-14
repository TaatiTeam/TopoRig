from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Optional, TypeVar

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.mesh_dataset import (
    DEFAULT_LANDMARK_MAPPER,
    MeshCacheTask,
    _cache_process_start_method,
    _cache_rigged_mesh_task,
    _cache_worker_count,
    _normalize_cache_format,
    _normalize_cache_compression,
    _normalize_cache_compression_level,
    _normalize_hdf5_blendshape_chunk_vertices,
    _normalize_blendshape_name_mapping,
    _normalize_mesh_id,
    _load_excluded_mesh_ids,
    _rigged_mesh_cache_path,
)
from dataset.hf_assets import prepare_assets, asset_cache_namespace, versioned_cache_dir
from dataset.layout import canonical_identity


DEFAULT_CONFIG = ROOT / "config" / "stage2.yaml"
T = TypeVar("T")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-cache TopoRig mesh targets for one shard.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Training config to read mesh args from.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Config split whose mesh args should be cached.",
    )
    parser.add_argument(
        "--mesh-source",
        default="blendshape_transfer",
        help=(
            "Mesh source to cache: 'mesh_args'/'primary' for the main mesh stream, "
            "or an extra_mesh_args key such as 'blendshape_transfer'."
        ),
    )
    parser.add_argument(
        "--i",
        type=int,
        default=None,
        help="1-based shard index. Defaults to SLURM_ARRAY_TASK_ID when present.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Total shard count. Defaults to SLURM_ARRAY_TASK_COUNT when present.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override cache_workers from the selected mesh args.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally process only this many meshes from the selected shard.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print progress every N completed meshes.",
    )
    return parser


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


def mesh_args_from_config(
    config: dict[str, Any],
    split: str,
    mesh_source: str,
) -> dict[str, Any]:
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("Config is missing a data mapping.")
    split_config = data.get(split)
    if not isinstance(split_config, Mapping):
        raise ValueError(f"Config is missing data.{split}.")

    source = mesh_source.strip()
    if source in {"mesh", "primary", "mesh_args"}:
        mesh_args = split_config.get("mesh_args")
    else:
        mesh_args = extra_mesh_args_from_config(split_config, source)

    if not isinstance(mesh_args, Mapping):
        raise ValueError(f"Could not find mesh source {mesh_source!r}.")
    if mesh_args.get("enabled", True) is False:
        raise ValueError(f"Mesh source {mesh_source!r} is disabled in the config.")
    result = dict(mesh_args)
    result.setdefault("split", split_config.get("split", "all"))
    result.setdefault("split_csv", split_config.get("split_csv"))
    data_root = result.get("data_root", data.get("data_root"))
    if data_root:
        component = f"meshes_{result.get('mesh_source', 'ict')}"
        cache = Path(result["cache_dir"]).expanduser()
        root = prepare_assets(data_root, result.get("asset_cache_dir", data.get("asset_cache_dir")) or cache.parent / "hf_assets", (component,))
        result.update(data_root=str(data_root), mesh_dir=str(root / component),
                      cache_dir=str(versioned_cache_dir(cache, data_root, component)))
    return result


def extra_mesh_args_from_config(
    split_config: Mapping[str, Any],
    source: str,
) -> Optional[Mapping[str, Any]]:
    extra_mesh_args = split_config.get("extra_mesh_args")
    if isinstance(extra_mesh_args, Mapping):
        if "mesh_dir" in extra_mesh_args:
            if source in {"extra", "extra_mesh", "0"}:
                return extra_mesh_args
            return None
        candidate = extra_mesh_args.get(source)
        return candidate if isinstance(candidate, Mapping) else None

    if isinstance(extra_mesh_args, Sequence) and not isinstance(
        extra_mesh_args,
        (str, bytes),
    ):
        for index, candidate in enumerate(extra_mesh_args):
            if not isinstance(candidate, Mapping):
                continue
            name = str(candidate.get("name", f"extra_mesh_{index}"))
            if source in {name, str(index)}:
                return candidate
    return None


def list_mesh_paths(mesh_dir: Path) -> dict[str, Path]:
    if mesh_dir.is_file():
        if mesh_dir.suffix.lower() != ".fbx":
            raise ValueError(
                "mesh_dir must be a directory of .fbx files or one .fbx file."
            )
        return {mesh_dir.stem: mesh_dir}

    if not mesh_dir.is_dir():
        raise FileNotFoundError(mesh_dir)

    mesh_paths: dict[str, Path] = {}
    for path in sorted(mesh_dir.iterdir()):
        if path.is_file() and path.suffix.lower() == ".fbx":
            mesh_paths[path.stem] = path
    return mesh_paths


def split_mesh_ids(split_csv: Optional[Path], split: str) -> Optional[set[str]]:
    normalized_split = split.strip().lower()
    if normalized_split == "all":
        return None
    if split_csv is None:
        raise ValueError("split_csv is required unless split='all'.")
    if not split_csv.is_file():
        raise FileNotFoundError(split_csv)

    with split_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{split_csv} is empty.")
        missing = {"mesh_id", "split"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{split_csv} is missing columns: {sorted(missing)}")
        return {
            _normalize_mesh_id(row["mesh_id"])
            for row in reader
            if row["split"].strip().lower() == normalized_split
        }


def build_mesh_paths(mesh_args: dict[str, Any]) -> dict[str, Path]:
    mesh_dir = Path(mesh_args["mesh_dir"]).expanduser()
    mesh_paths = list_mesh_paths(mesh_dir)
    allowed_mesh_ids = split_mesh_ids(
        Path(mesh_args["split_csv"]).expanduser()
        if mesh_args.get("split_csv") not in (None, "")
        else None,
        str(mesh_args.get("split", "all")),
    )
    if allowed_mesh_ids is not None:
        if mesh_args.get("data_root"):
            allowed_mesh_ids = {canonical_identity(value) for value in allowed_mesh_ids}
        mesh_paths = {
            mesh_id: path
            for mesh_id, path in mesh_paths.items()
            if mesh_id in allowed_mesh_ids
        }
    excluded_mesh_ids = _load_excluded_mesh_ids(
        mesh_args.get("exclude_mesh_ids"),
        optional_path(mesh_args.get("exclude_mesh_ids_file")),
    )
    if mesh_args.get("data_root"):
        excluded_mesh_ids = {canonical_identity(value) for value in excluded_mesh_ids}
    if excluded_mesh_ids:
        mesh_paths = {
            mesh_id: path
            for mesh_id, path in mesh_paths.items()
            if mesh_id not in excluded_mesh_ids
        }
    return {mesh_id: mesh_paths[mesh_id] for mesh_id in sorted(mesh_paths)}


def optional_path(value: Any) -> Optional[Path]:
    if value in (None, ""):
        return None
    return Path(value).expanduser()


def build_tasks(
    mesh_args: dict[str, Any],
    mesh_ids: Sequence[str],
    mesh_paths: Mapping[str, Path],
) -> list[MeshCacheTask]:
    cache_dir = Path(mesh_args["cache_dir"]).expanduser()
    rigged_mesh_dir = cache_dir / "rigged_meshes"
    rigged_mesh_dir.mkdir(parents=True, exist_ok=True)

    mediapipe_landmark_dir = optional_path(mesh_args.get("mediapipe_landmark_dir"))
    if mediapipe_landmark_dir is not None:
        mediapipe_landmark_dir.mkdir(parents=True, exist_ok=True)

    drop_mouth_probability = mesh_args.get("drop_mouth_probability")
    if drop_mouth_probability is None:
        drop_mouth_probability = mesh_args.get("drop_teeth_probability", 0.0)

    cache_compression = _normalize_cache_compression(
        mesh_args.get("cache_compression")
    )
    cache_compression_level = _normalize_cache_compression_level(
        int(mesh_args.get("cache_compression_level", 1))
    )
    cache_format = _normalize_cache_format(str(mesh_args.get("cache_format", "torch")))
    hdf5_blendshape_chunk_vertices = _normalize_hdf5_blendshape_chunk_vertices(
        int(mesh_args.get("hdf5_blendshape_chunk_vertices", 8192))
    )

    return [
        MeshCacheTask(
            mesh_id=mesh_id,
            mesh_path=mesh_paths[mesh_id],
            cache_path=_rigged_mesh_cache_path(
                rigged_mesh_dir,
                mesh_id,
                cache_format,
            ),
            use_mediapipe_landmarks=bool(mesh_args.get("use_mediapipe_landmarks", True)),
            generate_mediapipe_landmarks=bool(
                mesh_args.get("generate_mediapipe_landmarks", False)
            ),
            mediapipe_landmark_dir=mediapipe_landmark_dir,
            mediapipe_mapper_path=Path(
                mesh_args.get("mediapipe_mapper_path", DEFAULT_LANDMARK_MAPPER)
            ).expanduser(),
            mediapipe_mapper_args=tuple(
                str(arg) for arg in (mesh_args.get("mediapipe_mapper_args") or ())
            ),
            drop_mouth_probability=float(drop_mouth_probability),
            cut_eyeballs_probability=float(
                mesh_args.get("cut_eyeballs_probability", 0.0)
            ),
            cache_random_seed=int(mesh_args.get("cache_random_seed", 0)),
            cache_format=cache_format,
            cache_compression=cache_compression,
            cache_compression_level=cache_compression_level,
            hdf5_blendshape_chunk_vertices=hdf5_blendshape_chunk_vertices,
            blendshape_name_mapping=_normalize_blendshape_name_mapping(
                str(mesh_args.get("blendshape_name_mapping", "metahuman"))
            ),
        )
        for mesh_id in mesh_ids
    ]


def cache_tasks(
    tasks: Sequence[MeshCacheTask],
    *,
    workers: int,
    log_every: int,
) -> None:
    worker_count = _cache_worker_count(workers, len(tasks))
    total = len(tasks)
    started_at = time.time()
    print(
        f"[INFO] Caching {total} mesh cache file(s) with "
        f"{worker_count} worker process(es)."
    )
    if total == 0:
        return

    def log_progress(processed: int) -> None:
        if log_every <= 0 and processed < total:
            return
        if log_every > 0 and processed % log_every != 0 and processed < total:
            return
        elapsed = max(time.time() - started_at, 1.0e-6)
        rate = processed / elapsed
        print(
            f"[INFO] {processed}/{total} cached or verified "
            f"({rate:.2f} meshes/s, {rate * 60.0:.1f} meshes/min).",
            flush=True,
        )

    if worker_count == 1:
        for processed, task in enumerate(tasks, start=1):
            _cache_rigged_mesh_task(task)
            log_progress(processed)
        return

    context = multiprocessing.get_context(_cache_process_start_method())
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        ) as executor:
            for processed, _result in enumerate(
                executor.map(_cache_rigged_mesh_task, tasks),
                start=1,
            ):
                log_progress(processed)
    except BrokenProcessPool as exc:
        raise RuntimeError(
            "A mesh cache worker process crashed while importing FBX files. "
            "Retry with fewer workers, for example --workers 2."
        ) from exc


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    resolve_shard_args(args)
    validate_shard_args(int(args.i), int(args.n))
    if args.workers is not None and args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")

    config = load_config(args.config)
    mesh_args = mesh_args_from_config(config, args.split, args.mesh_source)
    mesh_paths = build_mesh_paths(mesh_args)
    mesh_ids = sorted(mesh_paths)
    shard = shard_items(mesh_ids, int(args.i), int(args.n))
    if args.limit is not None:
        shard = shard[: args.limit]
    print(
        f"[INFO] Selected shard {args.i}/{args.n}: {len(shard)} of "
        f"{len(mesh_ids)} total mesh(es) for {args.split}.{args.mesh_source}."
    )
    workers = int(args.workers or mesh_args.get("cache_workers", 1))
    tasks = build_tasks(mesh_args, shard, mesh_paths)
    cache_tasks(tasks, workers=workers, log_every=int(args.log_every))


if __name__ == "__main__":
    main()
