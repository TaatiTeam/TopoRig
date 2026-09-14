from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.cache_preparation import (
    image_args_from_config,
    load_config,
    resolve_shard_args,
    shard_items,
    validate_shard_args,
)
from scripts.cache_image_meshes import (
    build_tasks as build_image_mesh_tasks,
    eligible_person_ids,
)
from utils.welded_eyelid_landmarks import (
    VALIDATED_WELDED_LANDMARK_VERSION,
    build_validated_welded_landmark_payload,
)


DEFAULT_CONFIG = (
    ROOT
    / "config"
    / "stage2.yaml"
)


@dataclass(frozen=True)
class ValidatedLandmarkCacheTask:
    person_id: str
    source_landmark_path: Path
    welded_mesh_cache_path: Path
    output_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and cache MediaPipe landmarks in welded image-mesh space."
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--i", type=int, default=None, help="1-based shard index.")
    parser.add_argument("--n", type=int, default=None, help="Total shard count.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    return parser


def validation_config(image_args: Mapping[str, Any]) -> dict[str, Any]:
    value = image_args.get("welded_landmark_validation")
    if not isinstance(value, Mapping) or not bool(value.get("enabled", False)):
        raise ValueError(
            "image_args.welded_landmark_validation.enabled must be true."
        )
    return dict(value)


def build_tasks(
    image_args: dict[str, Any],
    person_ids: Sequence[str],
) -> list[ValidatedLandmarkCacheTask]:
    if not bool(image_args.get("weld_coincident_vertices", False)):
        raise ValueError("Validated welded landmarks require welding to be enabled.")
    config = validation_config(image_args)
    output_dir = Path(str(config["cache_dir"])).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    welded_dir = Path(
        str(image_args["welded_neutral_mesh_tensor_dir"])
    ).expanduser()
    mesh_tasks = build_image_mesh_tasks(image_args, person_ids)
    tasks = []
    for mesh_task in mesh_tasks:
        if mesh_task.landmark_path is None:
            raise ValueError(
                f"Identity {mesh_task.person_id} has no MediaPipe landmark path."
            )
        tasks.append(
            ValidatedLandmarkCacheTask(
                person_id=mesh_task.person_id,
                source_landmark_path=mesh_task.landmark_path,
                welded_mesh_cache_path=welded_dir / mesh_task.cache_path.name,
                output_path=output_dir
                / Path(mesh_task.cache_path.name).with_suffix(".json"),
            )
        )
    return tasks


def cache_task(
    task: ValidatedLandmarkCacheTask,
    *,
    action_unit_ids: Sequence[int],
    config: Mapping[str, Any],
    force: bool,
) -> tuple[str, dict[str, Any]]:
    if task.output_path.is_file() and not force:
        with task.output_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if (
            isinstance(existing, dict)
            and int(existing.get("version", -1))
            == VALIDATED_WELDED_LANDMARK_VERSION
        ):
            return "existing", existing
    if not task.source_landmark_path.is_file():
        raise FileNotFoundError(task.source_landmark_path)
    if not task.welded_mesh_cache_path.is_file():
        raise FileNotFoundError(task.welded_mesh_cache_path)

    with task.source_landmark_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    raw = loaded.get("mapping") if isinstance(loaded, dict) else None
    if raw is None:
        raw = loaded
    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected {task.source_landmark_path} to contain a landmark mapping."
        )
    raw_mapping = {
        int(mediapipe_id): int(vertex_id)
        for mediapipe_id, vertex_id in raw.items()
        if vertex_id is not None
    }

    welded = torch.load(
        task.welded_mesh_cache_path,
        map_location="cpu",
        weights_only=True,
    )
    required = (
        "vertices",
        "source_vertex_to_welded",
        "component_ids",
    )
    missing = [name for name in required if name not in welded]
    if missing:
        raise ValueError(
            f"Welded cache {task.welded_mesh_cache_path} is missing {missing}."
        )
    payload = build_validated_welded_landmark_payload(
        person_id=task.person_id,
        raw_mapping=raw_mapping,
        welded_vertices=welded["vertices"],
        source_to_welded=welded["source_vertex_to_welded"],
        component_ids=welded["component_ids"],
        action_unit_ids=action_unit_ids,
        config=config,
    )
    payload["source_landmark_path"] = str(task.source_landmark_path)
    payload["welded_mesh_cache_path"] = str(task.welded_mesh_cache_path)
    _atomic_write_json(task.output_path, payload)
    return str(payload["status"]), payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = build_parser().parse_args()
    resolve_shard_args(args)
    validate_shard_args(int(args.i), int(args.n))
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")

    config = load_config(args.config)
    image_args = image_args_from_config(config, args.split)
    validator = validation_config(image_args)
    action_unit_ids = tuple(
        int(value)
        for value in validator.get(
            "action_units",
            image_args.get("action_units", ()),
        )
        if int(value) in {10, 11}
    )
    if not action_unit_ids:
        raise ValueError("Validated eyelid caching requires AU10 and/or AU11.")
    person_ids = eligible_person_ids(image_args)
    shard = shard_items(person_ids, int(args.i), int(args.n))
    if args.limit is not None:
        shard = shard[: args.limit]
    tasks = build_tasks(image_args, shard)
    print(
        f"[INFO] Validating shard {args.i}/{args.n}: {len(tasks)} of "
        f"{len(person_ids)} eligible identities for AU{list(action_unit_ids)}.",
        flush=True,
    )

    counts: Counter[str] = Counter()
    manifest_rows = []
    failures = []
    started_at = time.time()
    for index, task in enumerate(tasks, 1):
        try:
            status, payload = cache_task(
                task,
                action_unit_ids=action_unit_ids,
                config=validator,
                force=bool(args.force),
            )
            counts[status] += 1
            manifest_rows.append(
                {
                    "person_id": task.person_id,
                    "status": payload.get("status", status),
                    "reasons": payload.get("reasons", []),
                    "output_path": str(task.output_path),
                }
            )
        except Exception as exc:
            failures.append(f"{task.person_id}: {type(exc).__name__}: {exc}")
        if index == len(tasks) or (args.log_every > 0 and index % args.log_every == 0):
            elapsed = max(time.time() - started_at, 1.0e-6)
            print(
                f"[INFO] {index}/{len(tasks)} checked ({index / elapsed:.2f}/s); "
                f"statuses={dict(counts)} failures={len(failures)}",
                flush=True,
            )

    output_dir = Path(str(validator["cache_dir"])).expanduser()
    manifest_path = output_dir / (
        f"manifest_shard_{int(args.i):03d}_of_{int(args.n):03d}.json"
    )
    _atomic_write_json(
        manifest_path,
        {
            "version": VALIDATED_WELDED_LANDMARK_VERSION,
            "shard_index": int(args.i),
            "shard_count": int(args.n),
            "status_counts": dict(counts),
            "failures": failures,
            "identities": manifest_rows,
        },
    )
    print(f"[INFO] Wrote audit manifest: {manifest_path}", flush=True)
    if failures:
        shown = "\n".join(f"  - {value}" for value in failures[:20])
        raise RuntimeError(
            f"Validated landmark caching failed for {len(failures)} identity/"
            f"identities:\n{shown}"
        )


if __name__ == "__main__":
    main()
