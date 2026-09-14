#!/usr/bin/env python3
"""Evaluate a TopoRig checkpoint with physical-unit MAE and head-level Q95."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train
from dataset.mesh_dataset import MeshDataset
from dataset.hf_assets import prepare_assets, versioned_cache_dir
from dataset.layout import MANIFEST_NAME
from utils.evaluation import (
    STANDARD_HEAD_HEIGHT_MM,
    atomic_write_csv,
    atomic_write_json,
    q95,
    read_paired_manifest,
)
from utils.fbx_to_tensor import AU_NAME
from utils.paths import runtime_path


GAZE_ACTION_UNITS = frozenset(range(12, 20))
NON_GAZE_ACTION_UNITS = tuple(
    action_unit_id
    for action_unit_id in sorted(AU_NAME)
    if action_unit_id not in GAZE_ACTION_UNITS
)
# The transferred Pixel3D fits do not contain reliable blink or eye-wide motion.
PIXEL3D_INVALID_ACTION_UNITS = frozenset((10, 11, 22, 23))
PIXEL3D_ACTION_UNITS = tuple(
    action_unit_id
    for action_unit_id in NON_GAZE_ACTION_UNITS
    if action_unit_id not in PIXEL3D_INVALID_ACTION_UNITS
)
PREPROCESSING_VERSION = (
    "toporig_moving_vertices_plus_240mm_moving_vertices_non_gaze_v4"
)
SAMPLE_FIELDS = (
    "model",
    "source",
    "mesh_id",
    "action_unit_id",
    "action_unit_name",
    "absolute_error_sum_mm",
    "element_count",
    "mae_mm",
    "moving_vertex_count",
    "total_vertex_count",
    "moving_vertex_ratio",
    "standard_head_error_sum_mm",
    "standard_head_vertex_count",
    "standard_head_mae_mm",
    "standard_head_mae_q95_mm",
    "neutral_head_height_m",
    "standard_head_height_mm",
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    mesh_dir: Path
    split: str
    split_csv: Path
    cache_dir: Path
    landmark_dir: Path | None
    action_units: tuple[int, ...]
    cache_format: str
    generate_landmarks: bool
    blendshape_name_mapping: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the main TopoRig checkpoint on recorded ICT/custom heads "
            "in physical units."
        )
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--paired-manifest", type=Path, required=True,
        help="Preserved evaluation membership CSV; no heads are resampled.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--head-count", type=int, default=350)
    parser.add_argument("--cache-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--moving-threshold-mm", type=float, default=0.01)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--source-dataset", choices=("ict",), default="ict")
    parser.add_argument("--asset-cache-dir", type=Path, default=runtime_path("cache", "assets"),
                        help="External cache for FBXs materialized from a downloaded shard release.")
    parser.add_argument(
        "--ict-mesh-dir", type=Path,
        default=runtime_path("data", "meshes_ict"),
    )
    parser.add_argument(
        "--pixel3d-mesh-dir", type=Path,
        default=runtime_path("data", "meshes_custom"),
    )
    parser.add_argument(
        "--ict-landmark-dir", type=Path,
        default=runtime_path("metadata", "ict_eval_landmarks"),
        help="Recorded benchmark mappings, kept separate from training ICT mappings.",
    )
    parser.add_argument(
        "--ict-cache-dir", type=Path,
        default=runtime_path("cache", "physical_eval/ict"),
    )
    parser.add_argument(
        "--pixel3d-cache-dir", type=Path,
        default=runtime_path("cache", "physical_eval/custom"),
    )
    return parser.parse_args(argv)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested, but CUDA is unavailable.")
    return device


def load_checkpoint(path: Path, device: torch.device) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a mapping.")
    if not isinstance(checkpoint.get("config"), Mapping):
        raise ValueError("Checkpoint does not contain its training config.")
    if not isinstance(checkpoint.get("model_state_dict"), Mapping):
        raise ValueError("Checkpoint does not contain model_state_dict.")
    return checkpoint


def source_specs(args: argparse.Namespace) -> tuple[SourceSpec, SourceSpec]:
    records = read_paired_manifest(args.paired_manifest)
    roots = {
        "ict": args.ict_mesh_dir.expanduser().resolve(),
        "pixel3d": args.pixel3d_mesh_dir.expanduser().resolve(),
    }
    cache_root = args.ict_cache_dir.expanduser().resolve()
    custom_cache_root = args.pixel3d_cache_dir.expanduser().resolve()
    selected_ids: dict[str, list[str]] = {}
    for source in ("ict", "pixel3d"):
        selected = [record for record in records if record.dataset == source]
        if len(selected) != args.head_count:
            raise ValueError(
                f"Paired manifest contains {len(selected)} {source} heads; "
                f"expected {args.head_count}."
            )
        ids = [record.mesh_id for record in selected]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Paired manifest contains duplicate {source} IDs.")
        release = roots[source].parent
        component = "meshes_ict" if source == "ict" else "meshes_custom"
        if roots[source].name == component and (release / MANIFEST_NAME).is_file():
            materialized = prepare_assets(release, args.asset_cache_dir, (component,), identities=ids)
            roots[source] = materialized / component
            if source == "ict":
                cache_root = versioned_cache_dir(cache_root, release, component)
            else:
                custom_cache_root = versioned_cache_dir(custom_cache_root, release, component)
        # Explicit roots relocate recorded identities without opening obsolete
        # absolute paths from the historical manifest or changing membership.
        for identity in ids:
            mesh = roots[source] / f"{identity}.fbx"
            if not mesh.is_file():
                raise FileNotFoundError(f"Missing paired {source} FBX: {mesh}")
        selected_ids[source] = ids
    cache_root.mkdir(parents=True, exist_ok=True)
    source_split = cache_root / f"ict_{args.head_count}_split.csv"
    custom_split = cache_root / f"pixel3d_{args.head_count}_split.csv"
    for source, path in [("ict", source_split), ("pixel3d", custom_split)]:
        atomic_write_csv(
            path, ("mesh_id", "split"),
            ({"mesh_id": identity, "split": "val"} for identity in selected_ids[source]),
        )
    return (
        SourceSpec(
            name="ict", mesh_dir=roots["ict"], split="val", split_csv=source_split,
            cache_dir=cache_root,
            landmark_dir=args.ict_landmark_dir.expanduser().resolve(),
            action_units=NON_GAZE_ACTION_UNITS, cache_format="hdf5",
            generate_landmarks=False, blendshape_name_mapping="canonical",
        ),
        SourceSpec(
            name="pixel3d_100k", mesh_dir=roots["pixel3d"], split="val",
            split_csv=custom_split, cache_dir=custom_cache_root,
            landmark_dir=roots["pixel3d"], action_units=PIXEL3D_ACTION_UNITS,
            cache_format="hdf5", generate_landmarks=False,
            blendshape_name_mapping="canonical",
        ),
    )


def build_dataset(spec: SourceSpec, cache_workers: int) -> MeshDataset:
    with spec.split_csv.open(newline="") as handle:
        identities = [row["mesh_id"] for row in csv.DictReader(handle)]
    missing = [identity for identity in identities if spec.landmark_dir is None
               or not (spec.landmark_dir / f"{identity}.json").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} {spec.name} evaluation landmark mappings in "
            f"{spec.landmark_dir}: {', '.join(missing[:8])}"
        )
    dataset = MeshDataset(
        mesh_dir=spec.mesh_dir,
        split=spec.split,
        split_csv=spec.split_csv,
        cache_dir=spec.cache_dir,
        action_units=spec.action_units,
        cache_workers=cache_workers,
        trust_existing_cache=True,
        cache_format=spec.cache_format,
        cache_compression="gzip" if spec.cache_format == "hdf5" else None,
        cache_compression_level=1,
        hdf5_blendshape_chunk_vertices=8192,
        rigged_mesh_cache_size=1,
        use_mediapipe_landmarks=True,
        generate_mediapipe_landmarks=spec.generate_landmarks,
        mediapipe_landmark_dir=spec.landmark_dir,
        mesh_up_axis="z",
        mesh_front_axis="y",
        normalize_on_get=False,
        normalized_extent=2.0,
        drop_mouth_probability=0.0,
        cut_eyeballs_probability=0.0,
        topology_augmentation_probability=0.0,
        shape_augmentation_probability=0.0,
        blendshape_name_mapping=spec.blendshape_name_mapping,
    )
    if len(dataset.mesh_ids) * len(spec.action_units) != len(dataset):
        raise RuntimeError(f"Unexpected sample coverage for {spec.name}.")
    return dataset


def normalize_for_model(
    mesh: Mapping[str, torch.Tensor],
    landmarks: Mapping[str, Any],
    normalized_extent: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], torch.Tensor]:
    vertices = mesh["vertices"]
    bounds_min = vertices.amin(dim=0)
    bounds_max = vertices.amax(dim=0)
    center = (bounds_min + bounds_max) * 0.5
    max_span_m = (bounds_max - bounds_min).amax().clamp_min(1.0e-8)
    model_units_per_meter = vertices.new_tensor(normalized_extent) / max_span_m

    normalized_mesh = dict(mesh)
    normalized_mesh["vertices"] = (
        (vertices - center) * model_units_per_meter
    ).contiguous()
    normalized_landmarks = dict(landmarks)
    for key in ("neutral_positions", "target_positions"):
        value = normalized_landmarks.get(key)
        if isinstance(value, torch.Tensor) and value.shape[-1:] == (3,):
            normalized_landmarks[key] = (
                (value - center) * model_units_per_meter
            ).contiguous()
    target_delta = normalized_landmarks.get("target_delta")
    if isinstance(target_delta, torch.Tensor) and target_delta.shape[-1:] == (3,):
        normalized_landmarks["target_delta"] = (
            target_delta * model_units_per_meter
        ).contiguous()
    return normalized_mesh, normalized_landmarks, model_units_per_meter


def predict_normalized_delta(
    model: torch.nn.Module,
    mesh: Mapping[str, torch.Tensor],
    action_unit_id: int,
    landmarks: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    device_mesh = train.to_device(mesh, device)
    vertices = device_mesh["vertices"].unsqueeze(0)
    normals = device_mesh.get("normals")
    if normals is not None:
        normals = normals.unsqueeze(0)
    faces = device_mesh.get("faces")
    record = {"mesh": device_mesh, "landmarks_3d": landmarks}
    facs = train.action_unit_vector(
        action_unit_id,
        int(config["model"]["facs_dim"]),
        device,
        float(config["training"].get("action_unit_scale", 1.0)),
    )
    landmark_features = train.build_model_landmark_features(
        record=record,
        vertices=vertices,
        config=config,
        device=device,
    )
    return model(
        vertices,
        facs,
        normals=normals,
        faces=faces,
        landmark_features=landmark_features,
    ).squeeze(0)


def moving_vertex_error_metrics(
    predicted_delta_m: torch.Tensor,
    target_delta_m: torch.Tensor,
    threshold_mm: float,
) -> dict[str, float | int]:
    if predicted_delta_m.shape != target_delta_m.shape:
        raise ValueError("Predicted and target displacement shapes must match.")
    if predicted_delta_m.ndim != 2 or predicted_delta_m.shape[-1] != 3:
        raise ValueError("Displacements must have shape [V, 3].")
    if threshold_mm < 0.0:
        raise ValueError("Moving-vertex threshold must be non-negative.")

    target_magnitude_mm = torch.linalg.vector_norm(target_delta_m, dim=-1) * 1000.0
    moving_mask = target_magnitude_mm > float(threshold_mm)
    moving_vertex_count = int(moving_mask.sum().item())
    total_vertex_count = int(moving_mask.numel())
    if moving_vertex_count == 0:
        raise ValueError(
            "Ground truth has no vertices above the moving threshold "
            f"of {threshold_mm:g} mm."
        )

    absolute_error_mm = (
        (predicted_delta_m - target_delta_m).abs()[moving_mask] * 1000.0
    )
    absolute_sum_mm = float(absolute_error_mm.sum().item())
    element_count = int(absolute_error_mm.numel())
    return {
        "absolute_error_sum_mm": absolute_sum_mm,
        "element_count": element_count,
        "mae_mm": absolute_sum_mm / element_count,
        "moving_vertex_count": moving_vertex_count,
        "total_vertex_count": total_vertex_count,
        "moving_vertex_ratio": moving_vertex_count / total_vertex_count,
    }


def standard_head_vertex_error_metrics(
    predicted_delta_m: torch.Tensor,
    target_delta_m: torch.Tensor,
    neutral_vertices_m: torch.Tensor,
    *,
    moving_mask: torch.Tensor,
    standard_head_height_mm: float = STANDARD_HEAD_HEIGHT_MM,
    up_axis: int = 2,
) -> dict[str, float | int]:
    """Moving-vertex Euclidean errors after scaling head height to 240 mm."""

    if predicted_delta_m.shape != target_delta_m.shape:
        raise ValueError("Predicted and target displacement shapes must match.")
    if predicted_delta_m.ndim != 2 or predicted_delta_m.shape[-1] != 3:
        raise ValueError("Displacements must have shape [V, 3].")
    neutral = neutral_vertices_m.to(
        device=predicted_delta_m.device,
        dtype=predicted_delta_m.dtype,
    )
    if neutral.shape != predicted_delta_m.shape:
        raise ValueError("Neutral vertices must match displacement shape.")
    moving_mask = moving_mask.to(device=predicted_delta_m.device)
    if moving_mask.dtype != torch.bool or moving_mask.shape != predicted_delta_m.shape[:1]:
        raise ValueError("Moving mask must be boolean with shape [V].")
    moving_vertex_count = int(moving_mask.sum().item())
    if moving_vertex_count == 0:
        raise ValueError("Standard-head metric requires at least one moving vertex.")
    if not 0 <= up_axis < 3:
        raise ValueError(f"Invalid up axis: {up_axis}.")
    neutral_axis = neutral[:, up_axis]
    neutral_head_height_m = float(
        (neutral_axis.max() - neutral_axis.min()).item()
    )
    if not math.isfinite(neutral_head_height_m) or neutral_head_height_m <= 0.0:
        raise ValueError(f"Invalid neutral head height: {neutral_head_height_m} m.")
    if not math.isfinite(standard_head_height_mm) or standard_head_height_mm <= 0:
        raise ValueError(
            f"Invalid standardized head height: {standard_head_height_mm} mm."
        )

    errors_mm = torch.linalg.vector_norm(
        predicted_delta_m - target_delta_m,
        dim=-1,
    )[moving_mask] * (float(standard_head_height_mm) / neutral_head_height_m)
    if not bool(torch.isfinite(errors_mm).all()):
        raise ValueError("Standard-head vertex errors contain non-finite values.")
    return {
        "standard_head_error_sum_mm": float(errors_mm.sum().item()),
        "standard_head_vertex_count": moving_vertex_count,
        "standard_head_mae_mm": float(errors_mm.mean().item()),
        "standard_head_mae_q95_mm": float(torch.quantile(errors_mm, 0.95).item()),
        "neutral_head_height_m": neutral_head_height_m,
        "standard_head_height_mm": float(standard_head_height_mm),
    }


def evaluate_source(
    spec: SourceSpec,
    dataset: MeshDataset,
    model_name: str,
    model: torch.nn.Module,
    config: Mapping[str, Any],
    device: torch.device,
    log_every: int,
    moving_threshold_mm: float,
    landmark_perturbation: Mapping[str, Any],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    normalized_extent = 2.0
    print(
        f"[INFO] Evaluating {spec.name}: heads={len(dataset.mesh_ids)} "
        f"AUs={len(spec.action_units)} samples={len(dataset)}",
        flush=True,
    )
    with torch.inference_mode():
        for index in range(len(dataset)):
            mesh, action_unit_id, target_delta_m, landmarks = dataset[index]
            sample = dataset.samples[index]
            normalized_mesh, normalized_landmarks, scale = normalize_for_model(
                mesh,
                landmarks,
                normalized_extent,
            )
            predicted_normalized = predict_normalized_delta(
                model,
                normalized_mesh,
                int(action_unit_id),
                normalized_landmarks,
                config,
                device,
            )
            target_delta_m = target_delta_m.to(
                device=device,
                dtype=predicted_normalized.dtype,
            )
            predicted_delta_m = predicted_normalized / scale.to(device=device)
            error_metrics = moving_vertex_error_metrics(
                predicted_delta_m,
                target_delta_m,
                moving_threshold_mm,
            )
            moving_mask = (
                torch.linalg.vector_norm(target_delta_m, dim=-1) * 1000.0
                > float(moving_threshold_mm)
            )
            standard_head_metrics = standard_head_vertex_error_metrics(
                predicted_delta_m,
                target_delta_m,
                mesh["vertices"],
                moving_mask=moving_mask,
            )
            rows.append(
                {
                    "model": model_name,
                    "source": spec.name,
                    "mesh_id": sample.mesh_id,
                    "action_unit_id": int(action_unit_id),
                    "action_unit_name": AU_NAME[int(action_unit_id)],
                    **error_metrics,
                    **standard_head_metrics,
                }
            )
            if log_every > 0 and (index + 1) % log_every == 0:
                print(
                    f"[INFO] {spec.name}: {index + 1}/{len(dataset)} samples",
                    flush=True,
                )
    return rows


def aggregate_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    if not rows:
        raise ValueError("No evaluation rows were supplied.")
    model_names = {str(row["model"]) for row in rows}
    if len(model_names) != 1:
        raise ValueError(f"Expected one model, found {sorted(model_names)}.")
    model_name = next(iter(model_names))

    by_head: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    by_au: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        source = str(row["source"])
        by_head[(source, str(row["mesh_id"]))].append(row)
        by_au[(source, int(row["action_unit_id"]))].append(row)

    per_head: list[dict[str, object]] = []
    for (source, mesh_id), selected in sorted(by_head.items()):
        absolute_sum = sum(float(row["absolute_error_sum_mm"]) for row in selected)
        element_count = sum(int(row["element_count"]) for row in selected)
        moving_vertex_count = sum(
            int(row["moving_vertex_count"]) for row in selected
        )
        total_vertex_count = sum(int(row["total_vertex_count"]) for row in selected)
        standard_head_error_sum = sum(
            float(row["standard_head_error_sum_mm"]) for row in selected
        )
        standard_head_vertex_count = sum(
            int(row["standard_head_vertex_count"]) for row in selected
        )
        per_head.append(
            {
                "model": model_name,
                "source": source,
                "mesh_id": mesh_id,
                "action_unit_count": len(selected),
                "absolute_error_sum_mm": absolute_sum,
                "element_count": element_count,
                "mae_mm": absolute_sum / max(element_count, 1),
                "moving_vertex_count": moving_vertex_count,
                "total_vertex_count": total_vertex_count,
                "moving_vertex_ratio": moving_vertex_count
                / max(total_vertex_count, 1),
                "standard_head_error_sum_mm": standard_head_error_sum,
                "standard_head_vertex_count": standard_head_vertex_count,
                "standard_head_mae_mm": standard_head_error_sum
                / max(standard_head_vertex_count, 1),
                "standard_head_mae_q95_mm": sum(
                    float(row["standard_head_mae_q95_mm"]) for row in selected
                )
                / len(selected),
            }
        )

    per_au: list[dict[str, object]] = []
    for (source, action_unit_id), selected in sorted(by_au.items()):
        absolute_sum = sum(float(row["absolute_error_sum_mm"]) for row in selected)
        element_count = sum(int(row["element_count"]) for row in selected)
        moving_vertex_count = sum(
            int(row["moving_vertex_count"]) for row in selected
        )
        total_vertex_count = sum(int(row["total_vertex_count"]) for row in selected)
        sample_maes = [float(row["mae_mm"]) for row in selected]
        standard_head_error_sum = sum(
            float(row["standard_head_error_sum_mm"]) for row in selected
        )
        standard_head_vertex_count = sum(
            int(row["standard_head_vertex_count"]) for row in selected
        )
        per_au.append(
            {
                "model": model_name,
                "source": source,
                "action_unit_id": action_unit_id,
                "action_unit_name": AU_NAME[action_unit_id],
                "head_count": len(selected),
                "mae_mm": absolute_sum / max(element_count, 1),
                "mae_q95_mm": q95(sample_maes),
                "moving_vertex_count": moving_vertex_count,
                "total_vertex_count": total_vertex_count,
                "moving_vertex_ratio": moving_vertex_count
                / max(total_vertex_count, 1),
                "standard_head_mae_mm": standard_head_error_sum
                / max(standard_head_vertex_count, 1),
                "standard_head_mae_q95_mm": sum(
                    float(row["standard_head_mae_q95_mm"]) for row in selected
                )
                / len(selected),
            }
        )

    summaries: list[dict[str, object]] = []
    sources = sorted({str(row["source"]) for row in rows})
    for source in (*sources, "combined"):
        selected_heads = [
            row for row in per_head if source == "combined" or row["source"] == source
        ]
        selected_samples = [
            row for row in rows if source == "combined" or row["source"] == source
        ]
        head_maes = [float(row["mae_mm"]) for row in selected_heads]
        absolute_sum = sum(
            float(row["absolute_error_sum_mm"]) for row in selected_samples
        )
        element_count = sum(int(row["element_count"]) for row in selected_samples)
        moving_vertex_count = sum(
            int(row["moving_vertex_count"]) for row in selected_samples
        )
        total_vertex_count = sum(
            int(row["total_vertex_count"]) for row in selected_samples
        )
        action_units = {int(row["action_unit_id"]) for row in selected_samples}
        standard_head_maes = [
            float(row["standard_head_mae_mm"]) for row in selected_heads
        ]
        standard_head_q95s = [
            float(row["standard_head_mae_q95_mm"]) for row in selected_heads
        ]
        summaries.append(
            {
                "model": model_name,
                "source": source,
                "head_count": len(selected_heads),
                "action_unit_count": len(action_units),
                "sample_count": len(selected_samples),
                "mae_mm": sum(head_maes) / max(len(head_maes), 1),
                "mae_q95_mm": q95(head_maes),
                "element_weighted_mae_mm": absolute_sum / max(element_count, 1),
                "moving_vertex_count": moving_vertex_count,
                "total_vertex_count": total_vertex_count,
                "moving_vertex_ratio": moving_vertex_count
                / max(total_vertex_count, 1),
                "standard_head_height_mm": STANDARD_HEAD_HEIGHT_MM,
                "standard_head_mae_mm": sum(standard_head_maes)
                / max(len(standard_head_maes), 1),
                "standard_head_mae_q95_mm": sum(standard_head_q95s)
                / max(len(standard_head_q95s), 1),
            }
        )
    return per_head, per_au, summaries


def validate_coverage(
    rows: Sequence[Mapping[str, object]],
    specs: Sequence[SourceSpec],
    head_count: int,
) -> None:
    for spec in specs:
        selected = [row for row in rows if row["source"] == spec.name]
        heads = {str(row["mesh_id"]) for row in selected}
        samples = {(str(row["mesh_id"]), int(row["action_unit_id"])) for row in selected}
        expected = {
            (mesh_id, action_unit_id)
            for mesh_id in heads
            for action_unit_id in spec.action_units
        }
        if len(heads) != head_count:
            raise RuntimeError(
                f"{spec.name} produced {len(heads)} heads; expected {head_count}."
            )
        if samples != expected:
            raise RuntimeError(
                f"{spec.name} result coverage does not match its head/AU product."
            )


def save_reports(
    output_dir: Path,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    rows: list[dict[str, object]],
    per_head: list[dict[str, object]],
    per_au: list[dict[str, object]],
    summaries: list[dict[str, object]],
    elapsed_seconds: float,
    moving_threshold_mm: float,
    landmark_perturbation: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_dir / "sample_metrics.csv", SAMPLE_FIELDS, rows)
    atomic_write_csv(
        output_dir / "per_head_mae.csv",
        (
            "model",
            "source",
            "mesh_id",
            "action_unit_count",
            "absolute_error_sum_mm",
            "element_count",
            "mae_mm",
            "moving_vertex_count",
            "total_vertex_count",
            "moving_vertex_ratio",
            "standard_head_error_sum_mm",
            "standard_head_vertex_count",
            "standard_head_mae_mm",
            "standard_head_mae_q95_mm",
        ),
        per_head,
    )
    atomic_write_csv(
        output_dir / "per_au_mae.csv",
        (
            "model",
            "source",
            "action_unit_id",
            "action_unit_name",
            "head_count",
            "mae_mm",
            "mae_q95_mm",
            "moving_vertex_count",
            "total_vertex_count",
            "moving_vertex_ratio",
            "standard_head_mae_mm",
            "standard_head_mae_q95_mm",
        ),
        per_au,
    )
    atomic_write_csv(
        output_dir / "summary.csv",
        (
            "model",
            "source",
            "head_count",
            "action_unit_count",
            "sample_count",
            "mae_mm",
            "mae_q95_mm",
            "element_weighted_mae_mm",
            "moving_vertex_count",
            "total_vertex_count",
            "moving_vertex_ratio",
            "standard_head_height_mm",
            "standard_head_mae_mm",
            "standard_head_mae_q95_mm",
        ),
        summaries,
    )
    atomic_write_json(
        output_dir / "summary.json",
        {
            "status": "complete",
            "preprocessing_version": PREPROCESSING_VERSION,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
            "elapsed_seconds": elapsed_seconds,
            "moving_threshold_mm": moving_threshold_mm,
            "landmark_perturbation": dict(landmark_perturbation),
            "models": summaries,
            "metrics": {
                "mae_mm": (
                    "Mean of per-head coordinate-wise absolute displacement "
                    "errors in millimetres, evaluated only on vertices whose "
                    "ground-truth displacement exceeds the configured threshold."
                ),
                "mae_q95_mm": (
                    "Linear 95th percentile of the per-head MAE distribution "
                    "in millimetres."
                ),
                "physical_conversion": (
                    "Predictions are divided by each input head's exact model "
                    "normalization scale and converted from metres to millimetres."
                ),
                "moving_vertex_selection": (
                    "The moving mask is determined only from ground-truth vertex "
                    "displacement magnitude, never from model predictions."
                ),
                "standard_head_mae_mm": (
                    "Mean Euclidean displacement error on ground-truth moving "
                    "vertices after scaling each neutral head to "
                    f"{STANDARD_HEAD_HEIGHT_MM:g} mm tall."
                ),
                "standard_head_mae_q95_mm": (
                    "Mean across each head's action units of the per-sample linear "
                    "95th-percentile Euclidean error on ground-truth moving "
                    "vertices at standardized human-head scale."
                ),
            },
        },
    )


def log_wandb(
    config: Mapping[str, Any],
    run_name: str,
    checkpoint_path: Path,
    per_au: Sequence[Mapping[str, object]],
    summaries: Sequence[Mapping[str, object]],
    moving_threshold_mm: float,
    landmark_perturbation: Mapping[str, Any],
) -> None:
    import wandb

    wandb_config = config.get("wandb", {})
    source_head_counts = {
        str(summary["source"]): int(summary["head_count"])
        for summary in summaries
        if summary["source"] != "combined"
    }
    run = wandb.init(
        project=wandb_config.get("project", "toporig"),
        entity=wandb_config.get("entity"),
        name=run_name,
        job_type="evaluation",
        config={
            "checkpoint": str(checkpoint_path.resolve()),
            "metric": "physical displacement MAE and per-head Q95 in mm",
            "source_head_counts": source_head_counts,
            "moving_threshold_mm": moving_threshold_mm,
            "landmark_perturbation": dict(landmark_perturbation),
        },
        tags=[
            "evaluation",
            "physical-mm",
            "q95",
            str(landmark_perturbation["label"]),
        ],
    )
    for summary in summaries:
        source = str(summary["source"])
        run.summary[f"physical/{source}/mae_mm"] = float(summary["mae_mm"])
        run.summary[f"physical/{source}/mae_q95_mm"] = float(
            summary["mae_q95_mm"]
        )
        run.summary[f"physical/{source}/moving_vertex_ratio"] = float(
            summary["moving_vertex_ratio"]
        )
        run.summary[f"standard_head/{source}/mae_mm"] = float(
            summary["standard_head_mae_mm"]
        )
        run.summary[f"standard_head/{source}/mae_q95_mm"] = float(
            summary["standard_head_mae_q95_mm"]
        )
    columns = (
        "source",
        "action_unit_id",
        "action_unit_name",
        "head_count",
        "mae_mm",
        "mae_q95_mm",
        "moving_vertex_ratio",
        "standard_head_mae_mm",
        "standard_head_mae_q95_mm",
    )
    run.log(
        {
            "physical/per_au_mae": wandb.Table(
                columns=list(columns),
                data=[[row[column] for column in columns] for row in per_au],
            )
        }
    )
    run.finish()


def main() -> None:
    args = parse_args()
    if args.head_count < 1:
        raise ValueError("--head-count must be positive.")
    if args.moving_threshold_mm < 0.0:
        raise ValueError("--moving-threshold-mm must be non-negative.")
    landmark_perturbation = {
        "label": "clean", "drop_all": False, "drop_regions": [],
        "position_jitter_std_head_fraction": 0.0, "position_jitter_regions": [],
        "permute_regions": [], "head_up_axis": 2, "seed": 20260820,
    }
    specs = source_specs(args)
    datasets = {
        spec.name: build_dataset(spec, args.cache_workers) for spec in specs
    }
    for spec in specs:
        if len(datasets[spec.name].mesh_ids) != args.head_count:
            raise RuntimeError(
                f"{spec.name} selected {len(datasets[spec.name].mesh_ids)} heads; "
                f"expected {args.head_count}."
            )
    if args.prepare_only:
        print(
            f"[DONE] Prepared {args.head_count} {specs[0].name} and "
            f"{args.head_count} Pixel3D heads.",
            flush=True,
        )
        return
    if args.checkpoint is None or args.output_dir is None:
        raise ValueError("--checkpoint and --output-dir are required for evaluation.")

    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    config = checkpoint["config"]
    model = train.build_model(config, None, device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Checkpoint model state is incompatible with its config.")
    model.eval()
    model_name = args.wandb_run_name or args.output_dir.resolve().name
    print(
        f"[INFO] Checkpoint epoch={checkpoint.get('epoch')} "
        f"global_step={checkpoint.get('global_step')} device={device}",
        flush=True,
    )
    start_time = time.monotonic()
    rows: list[dict[str, object]] = []
    for spec in specs:
        rows.extend(
            evaluate_source(
                spec,
                datasets[spec.name],
                model_name,
                model,
                config,
                device,
                args.log_every,
                args.moving_threshold_mm,
                landmark_perturbation,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    validate_coverage(rows, specs, args.head_count)
    per_head, per_au, summaries = aggregate_rows(rows)
    elapsed_seconds = time.monotonic() - start_time
    output_dir = args.output_dir.expanduser().resolve()
    save_reports(
        output_dir,
        checkpoint_path,
        checkpoint,
        rows,
        per_head,
        per_au,
        summaries,
        elapsed_seconds,
        args.moving_threshold_mm,
        landmark_perturbation,
    )
    print(
        "[RESULT] source heads AUs moving_MAE_mm moving_MAE_Q95_mm "
        "standard_head_MAE_mm standard_head_MAE_Q95_mm",
        flush=True,
    )
    for summary in summaries:
        print(
            f"[RESULT] {str(summary['source']):14s} "
            f"{int(summary['head_count']):4d} "
            f"{int(summary['action_unit_count']):2d} "
            f"{float(summary['mae_mm']):.6f} "
            f"{float(summary['mae_q95_mm']):.6f} "
            f"{float(summary['standard_head_mae_mm']):.6f} "
            f"{float(summary['standard_head_mae_q95_mm']):.6f} "
            f"moving={float(summary['moving_vertex_ratio']):.2%}",
            flush=True,
        )
    if args.use_wandb:
        log_wandb(
            config,
            args.wandb_run_name or f"{model_name}-physical-mm-eval",
            checkpoint_path,
            per_au,
            summaries,
            args.moving_threshold_mm,
            landmark_perturbation,
        )
    print(f"[DONE] Reports: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
