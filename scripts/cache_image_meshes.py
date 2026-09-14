from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.image_dataset import (
    EXPRESSED_IMAGE_RE,
    _format_person_id_template,
    _load_neutral_mesh_for_model,
    _load_person_id_filter_file,
    _normalize_action_unit_ids,
    _normalize_person_id_filter,
    _normalize_person_id,
    _weld_neutral_mesh_payload,
)
from utils.cache_preparation import (
    image_args_from_config,
    list_neutral_images,
    load_config,
    resolve_shard_args,
    shard_items,
    split_person_ids,
    validate_shard_args,
)


DEFAULT_CONFIG = (
    ROOT
    / "config"
    / "stage2.yaml"
)


@dataclass(frozen=True)
class ImageMeshCacheTask:
    person_id: str
    mesh_path: Path
    cache_path: Path
    mesh_up_axis: str
    mesh_front_axis: str
    normalize_on_get: bool
    normalized_extent: float
    landmark_path: Optional[Path] = None
    min_landmarks: int = 1
    welded_cache_path: Optional[Path] = None
    weld_tolerance: float = 1.0e-6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-cache TopoRig image-branch neutral mesh tensors for one shard.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Training config to read data.<split>.image_args from.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Config split whose image mesh tensors should be cached.",
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
        default=1,
        help="FBX conversion worker processes for this shard.",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=32,
        help="Recycle each worker after this many meshes to release parser memory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally inspect only this many identities from the selected shard.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print progress every N newly cached meshes.",
    )
    parser.add_argument(
        "--skip-landmark-validation",
        action="store_true",
        help="Skip validating required image landmark maps after mesh caching.",
    )
    return parser


def eligible_person_ids(image_args: dict[str, Any]) -> list[str]:
    neutral_images = list_neutral_images(Path(image_args["neutral_dir"]).expanduser())
    if image_args.get("identity_profile") == "historical":
        neutral_images = {key: path for key, path in neutral_images.items() if not key.startswith("v2_")}

    person_filter = _normalize_person_id_filter(image_args.get("person_ids"))
    if person_filter is not None:
        neutral_images = {
            person_id: path
            for person_id, path in neutral_images.items()
            if person_id in person_filter
        }

    excluded = _normalize_person_id_filter(image_args.get("exclude_person_ids")) or set()
    exclusion_file = image_args.get("exclude_person_ids_file")
    if exclusion_file not in (None, ""):
        excluded.update(
            _load_person_id_filter_file(Path(str(exclusion_file)).expanduser())
        )
    if excluded:
        neutral_images = {
            person_id: path
            for person_id, path in neutral_images.items()
            if person_id not in excluded
        }

    allowed_split_ids = split_person_ids(
        Path(str(image_args["split_csv"])).expanduser()
        if image_args.get("split_csv") not in (None, "")
        else None,
        str(image_args.get("split", "all")),
    )
    if allowed_split_ids is not None:
        neutral_images = {
            person_id: path
            for person_id, path in neutral_images.items()
            if person_id in allowed_split_ids
        }

    expressed_dir = Path(image_args["expressed_dir"]).expanduser()
    if not expressed_dir.is_dir():
        raise FileNotFoundError(expressed_dir)
    requested_aus = set(_normalize_action_unit_ids(image_args.get("action_units")))
    eligible: set[str] = set()
    for path in expressed_dir.iterdir():
        if not path.is_file():
            continue
        match = EXPRESSED_IMAGE_RE.match(path.name)
        if match is None:
            continue
        person_id = _normalize_person_id(match.group(1))
        if person_id not in neutral_images:
            continue
        if int(match.group(2)) in requested_aus:
            eligible.add(person_id)
    return sorted(eligible)


def image_landmark_path(
    image_args: dict[str, Any],
    person_id: str,
    using_fallback: bool,
) -> Optional[Path]:
    if not bool(image_args.get("use_mediapipe_landmarks", False)):
        return None
    landmark_dir = Path(
        str(
            image_args.get(
                "mediapipe_landmark_dir",
                Path(str(image_args["cache_dir"])).expanduser()
                / "mediapipe_landmarks",
            )
        )
    ).expanduser()
    template_key = "mediapipe_landmark_filename_template"
    template = image_args.get(template_key, "mesh_{person_id}.json")
    if using_fallback and image_args.get(
        "mediapipe_landmark_filename_fallback_template"
    ) not in (None, ""):
        template_key = "mediapipe_landmark_filename_fallback_template"
        template = image_args[template_key]
    filename = _format_person_id_template(str(template), person_id, template_key)
    path = Path(filename).expanduser()
    return path if path.is_absolute() else landmark_dir / path


def image_mesh_path_and_variant(
    image_args: dict[str, Any],
    person_id: str,
) -> tuple[Path, bool]:
    exact_path = image_args.get("neutral_mesh_path")
    if exact_path not in (None, ""):
        return Path(str(exact_path)).expanduser(), False

    mesh_dir = Path(image_args["neutral_mesh_dir"]).expanduser()
    primary_name = _format_person_id_template(
        str(image_args.get("neutral_mesh_filename_template", "mesh_{person_id}.glb")),
        person_id,
        "neutral_mesh_filename_template",
    )
    primary_path = Path(primary_name).expanduser()
    if not primary_path.is_absolute():
        primary_path = mesh_dir / primary_path
    fallback_template = image_args.get("neutral_mesh_filename_fallback_template")
    if fallback_template not in (None, ""):
        fallback_name = _format_person_id_template(
            str(fallback_template),
            person_id,
            "neutral_mesh_filename_fallback_template",
        )
        fallback_path = Path(fallback_name).expanduser()
        if not fallback_path.is_absolute():
            fallback_path = mesh_dir / fallback_path
        primary_landmarks = image_landmark_path(image_args, person_id, False)
        fallback_landmarks = image_landmark_path(image_args, person_id, True)
        if (
            primary_path.is_file()
            and fallback_path.is_file()
            and primary_landmarks is not None
            and fallback_landmarks is not None
            and not primary_landmarks.is_file()
            and fallback_landmarks.is_file()
        ):
            return fallback_path, True
        if primary_path.is_file():
            return primary_path, False
        if fallback_path.is_file():
            return fallback_path, True
    return primary_path, False


def image_mesh_path(image_args: dict[str, Any], person_id: str) -> Path:
    return image_mesh_path_and_variant(image_args, person_id)[0]


def image_mesh_tensor_dir(image_args: dict[str, Any]) -> Path:
    configured = image_args.get("neutral_mesh_tensor_dir")
    if configured not in (None, ""):
        return Path(str(configured)).expanduser()
    return Path(image_args["cache_dir"]).expanduser() / "neutral_mesh_tensors"


def build_tasks(
    image_args: dict[str, Any],
    person_ids: Sequence[str],
) -> list[ImageMeshCacheTask]:
    tensor_dir = image_mesh_tensor_dir(image_args)
    tensor_dir.mkdir(parents=True, exist_ok=True)
    weld_enabled = bool(image_args.get("weld_coincident_vertices", False))
    welded_dir = Path(
        str(
            image_args.get(
                "welded_neutral_mesh_tensor_dir",
                tensor_dir / "welded",
            )
        )
    ).expanduser()
    if weld_enabled:
        welded_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[ImageMeshCacheTask] = []
    missing_sources: list[tuple[str, Path]] = []
    for person_id in person_ids:
        mesh_path, using_fallback = image_mesh_path_and_variant(
            image_args,
            person_id,
        )
        if not mesh_path.is_file():
            missing_sources.append((person_id, mesh_path))
            continue
        cache_filename = (
            f"mesh_{person_id}{'_fallback' if using_fallback else ''}.pt"
        )
        tasks.append(
            ImageMeshCacheTask(
                person_id=person_id,
                mesh_path=mesh_path,
                cache_path=tensor_dir / cache_filename,
                mesh_up_axis=str(image_args.get("mesh_up_axis", "z")),
                mesh_front_axis=str(image_args.get("mesh_front_axis", "-y")),
                normalize_on_get=bool(image_args.get("normalize_on_get", True)),
                normalized_extent=float(image_args.get("normalized_extent", 2.0)),
                landmark_path=image_landmark_path(
                    image_args,
                    person_id,
                    using_fallback,
                ),
                min_landmarks=int(image_args.get("min_mediapipe_landmarks", 1)),
                welded_cache_path=(
                    welded_dir / cache_filename if weld_enabled else None
                ),
                weld_tolerance=float(image_args.get("weld_tolerance", 1.0e-6)),
            )
        )
    if missing_sources:
        shown = ", ".join(
            f"{person_id} ({path})" for person_id, path in missing_sources[:10]
        )
        suffix = "..." if len(missing_sources) > 10 else ""
        raise FileNotFoundError(f"Missing image mesh source(s): {shown}{suffix}")
    return tasks


def cache_image_mesh_task(task: ImageMeshCacheTask) -> tuple[str, int, int]:
    if image_mesh_cache_task_complete(task):
        return "existing", 0, 0
    if task.cache_path.is_file():
        mesh = torch.load(task.cache_path, map_location="cpu", weights_only=True)
    else:
        mesh = _load_neutral_mesh_for_model(
            task.mesh_path,
            mesh_up_axis=task.mesh_up_axis,
            mesh_front_axis=task.mesh_front_axis,
            normalize_on_get=task.normalize_on_get,
            normalized_extent=task.normalized_extent,
        )
    vertices = mesh.get("vertices")
    faces = mesh.get("faces")
    if not isinstance(vertices, torch.Tensor) or vertices.ndim != 2:
        raise ValueError(f"Invalid vertices loaded from {task.mesh_path}.")
    if not isinstance(faces, torch.Tensor) or faces.ndim != 2:
        raise ValueError(f"Invalid faces loaded from {task.mesh_path}.")

    if not task.cache_path.is_file():
        atomic_torch_save(mesh, task.cache_path)
    if task.welded_cache_path is not None and not task.welded_cache_path.is_file():
        welded = _weld_neutral_mesh_payload(
            mesh,
            tolerance=task.weld_tolerance,
        )
        atomic_torch_save(welded, task.welded_cache_path)
    return "cached", int(vertices.shape[0]), int(faces.shape[0])


def image_mesh_cache_task_complete(task: ImageMeshCacheTask) -> bool:
    return task.cache_path.is_file() and (
        task.welded_cache_path is None or task.welded_cache_path.is_file()
    )


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def cache_tasks(
    tasks: Sequence[ImageMeshCacheTask],
    *,
    workers: int,
    max_tasks_per_child: int,
    log_every: int,
) -> None:
    missing = [task for task in tasks if not image_mesh_cache_task_complete(task)]
    existing = len(tasks) - len(missing)
    print(
        f"[INFO] Shard has {len(tasks)} eligible image mesh(es): "
        f"{existing} cached, {len(missing)} missing.",
        flush=True,
    )
    if not missing:
        return

    worker_count = min(max(int(workers), 1), len(missing))
    started_at = time.time()
    completed = 0
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        processes=worker_count,
        maxtasksperchild=max(int(max_tasks_per_child), 1),
    ) as pool:
        for status, _vertices, _faces in pool.imap_unordered(
            cache_image_mesh_task,
            missing,
            chunksize=1,
        ):
            if status == "cached":
                completed += 1
            if completed == len(missing) or (
                log_every > 0 and completed % log_every == 0
            ):
                elapsed = max(time.time() - started_at, 1.0e-6)
                rate = completed / elapsed
                print(
                    f"[INFO] {completed}/{len(missing)} newly cached "
                    f"({rate:.2f} meshes/s, {rate * 60.0:.1f} meshes/min).",
                    flush=True,
                )


def validate_landmark_tasks(tasks: Sequence[ImageMeshCacheTask]) -> None:
    checked = 0
    failures: list[str] = []
    for task in tasks:
        if task.landmark_path is None:
            continue
        checked += 1
        if not task.landmark_path.is_file():
            failures.append(
                f"{task.person_id}: missing {task.landmark_path}"
            )
            continue
        if not task.cache_path.is_file():
            failures.append(f"{task.person_id}: missing {task.cache_path}")
            continue
        try:
            with task.landmark_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            mapping = (
                loaded.get("mapping")
                if isinstance(loaded, dict)
                and isinstance(loaded.get("mapping"), dict)
                else loaded
            )
            if not isinstance(mapping, dict):
                raise ValueError("landmark JSON is not a mapping")
            cached = torch.load(
                task.cache_path,
                map_location="cpu",
                weights_only=True,
            )
            vertices = cached.get("vertices") if isinstance(cached, dict) else None
            if not isinstance(vertices, torch.Tensor) or vertices.ndim != 2:
                raise ValueError("tensor cache has invalid vertices")
            vertex_count = int(vertices.shape[0])
            landmark_count = 0
            invalid_ids: list[int] = []
            for raw_vertex_id in mapping.values():
                if raw_vertex_id is None:
                    continue
                vertex_id = int(raw_vertex_id)
                landmark_count += 1
                if vertex_id < 0 or vertex_id >= vertex_count:
                    invalid_ids.append(vertex_id)
            if landmark_count < task.min_landmarks:
                raise ValueError(
                    f"only {landmark_count} landmarks; expected at least "
                    f"{task.min_landmarks}"
                )
            if invalid_ids:
                shown = ", ".join(str(value) for value in invalid_ids[:5])
                suffix = "..." if len(invalid_ids) > 5 else ""
                raise ValueError(
                    f"{len(invalid_ids)} landmark vertex ids are outside "
                    f"[0, {vertex_count}): {shown}{suffix}"
                )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{task.person_id}: {type(exc).__name__}: {exc}")

    if failures:
        shown = "\n".join(f"  - {failure}" for failure in failures[:20])
        suffix = (
            f"\n  - ... and {len(failures) - 20} more"
            if len(failures) > 20
            else ""
        )
        raise RuntimeError(
            f"Image landmark preflight failed for {len(failures)} of "
            f"{checked} identities:\n{shown}{suffix}"
        )
    print(
        f"[INFO] Landmark preflight passed for all {checked} image identities.",
        flush=True,
    )


def main() -> None:
    args = build_parser().parse_args()
    resolve_shard_args(args)
    validate_shard_args(int(args.i), int(args.n))
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.max_tasks_per_child < 1:
        raise ValueError("--max-tasks-per-child must be at least 1.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")

    config = load_config(args.config)
    image_args = image_args_from_config(config, args.split)
    person_ids = eligible_person_ids(image_args)
    shard = shard_items(person_ids, int(args.i), int(args.n))
    if args.limit is not None:
        shard = shard[: args.limit]
    print(
        f"[INFO] Selected shard {args.i}/{args.n}: {len(shard)} of "
        f"{len(person_ids)} eligible image identities for {args.split}.",
        flush=True,
    )
    tasks = build_tasks(image_args, shard)
    cache_tasks(
        tasks,
        workers=int(args.workers),
        max_tasks_per_child=int(args.max_tasks_per_child),
        log_every=int(args.log_every),
    )
    if not args.skip_landmark_validation:
        validate_landmark_tasks(tasks)


if __name__ == "__main__":
    main()
