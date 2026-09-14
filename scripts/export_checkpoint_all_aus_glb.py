#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import demo
from utils.fbx_to_tensor import AU_NAME
from utils.lip_split import snap_mediapipe_lip_landmarks_to_face_surface


def main() -> None:
    args = parse_args()
    mesh_path = args.mesh.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    prediction_output = (
        args.prediction_output.expanduser().resolve()
        if args.prediction_output is not None
        else output_path.with_suffix(".predictions.npz")
    )

    checkpoint = demo.load_checkpoint(checkpoint_path)
    config = demo.checkpoint_config(checkpoint, args.config)
    device = demo.resolve_device(args.device, config)
    model = demo.load_model(checkpoint, config, device)

    raw_mesh = demo.load_mesh_for_model(mesh_path)
    input_landmark_json = demo.ensure_mediapipe_landmarks(
        mesh_path=mesh_path,
        cache_dir=args.landmark_cache_dir.expanduser(),
        mapper_path=args.mediapipe_mapper.expanduser(),
        blender=args.blender,
        force=args.refresh_landmarks,
    )
    reference_mesh_path = demo.metahuman_sample_mesh(
        args.metahuman_sample_dir.expanduser()
    )
    reference_landmark_json = demo.ensure_mediapipe_landmarks(
        mesh_path=reference_mesh_path,
        cache_dir=args.landmark_cache_dir.expanduser(),
        mapper_path=args.mediapipe_mapper.expanduser(),
        blender=args.blender,
        force=args.refresh_landmarks,
        prefer_sidecar=True,
    )
    input_landmarks = demo.load_landmark_mapping(input_landmark_json)
    lip_landmark_report = None
    lip_landmark_output = output_path.with_suffix(".lip_landmarks.json")
    if args.snap_lip_landmarks_to_face_surface:
        input_landmarks, lip_landmark_report = (
            snap_mediapipe_lip_landmarks_to_face_surface(
                raw_mesh,
                input_landmarks,
                front_direction=(
                    -1
                    if args.lip_landmark_front_direction == "negative"
                    else 1
                ),
            )
        )
        write_lip_landmark_mapping(
            lip_landmark_output,
            mapping=input_landmarks,
            report=lip_landmark_report.to_dict(),
            source_mesh=mesh_path,
            source_landmarks=input_landmark_json,
        )
    original_mesh = demo.preprocess_mesh_for_model(
        raw_mesh,
        config,
        input_suffix=mesh_path.suffix.lower(),
        input_convention=args.input_convention,
        input_landmarks=input_landmarks,
        reference_mesh=demo.load_reference_mesh_for_alignment(reference_mesh_path),
        reference_landmarks=demo.load_landmark_mapping(reference_landmark_json),
        min_alignment_landmarks=args.min_alignment_landmarks,
    )
    weld_tolerance, snap_landmark_ids = demo.inference_weld_settings(
        config,
        tolerance_override=args.weld_tolerance,
    )
    model_mesh, original_to_welded = demo.weld_mesh_for_inference(
        original_mesh,
        tolerance=weld_tolerance,
        snap_mediapipe_ids=snap_landmark_ids,
    )

    trained_ids = trained_action_unit_ids(config)
    action_unit_ids = (
        trained_ids
        if args.action_units is None
        else sorted({int(value) for value in args.action_units})
    )
    unavailable_ids = sorted(set(action_unit_ids) - set(trained_ids))
    if unavailable_ids:
        raise ValueError(
            "Requested AUs are not present in the checkpoint: "
            f"{unavailable_ids}"
        )
    is_ict_face = any(
        "ictfacemodel" in str(name).lower()
        for name in raw_mesh.get("source_mesh_object_names", ())
    )
    blendshape_vertices: dict[str, torch.Tensor] = {}
    rows: list[dict[str, str]] = []
    for index, action_unit_id in enumerate(action_unit_ids, start=1):
        _welded_deformed, welded_delta = demo.run_model(
            model=model,
            mesh=model_mesh,
            action_unit_id=action_unit_id,
            config=config,
            device=device,
        )
        predicted_delta = demo.map_welded_values_to_original(
            welded_delta,
            original_to_welded,
        )
        dental_report: dict[str, Any] | None = None
        if (
            not args.no_rigid_jaw_teeth
            and action_unit_id == demo.JAW_OPEN_AU_ID
            and is_ict_face
        ):
            predicted_delta, dental_report = demo.apply_rigid_jaw_teeth_postprocess(
                neutral_vertices=original_mesh["vertices"],
                predicted_delta=predicted_delta,
                faces=original_mesh["faces"],
                vertex_groups=raw_mesh.get("source_vertex_groups"),
                landmarks_3d=original_mesh.get("landmarks_3d"),
                action_unit_id=action_unit_id,
            )
        deformed_vertices = original_mesh["vertices"] + predicted_delta
        action_unit_name = AU_NAME[action_unit_id]
        blendshape_vertices[action_unit_name] = demo.output_vertices_for_convention(
            deformed_vertices,
            args.output_convention,
        )
        row = {
            "action_unit_id": str(action_unit_id),
            "action_unit_name": action_unit_name,
            "delta_mean_norm": f"{predicted_delta.norm(dim=-1).mean().item():.9f}",
            "delta_max_norm": f"{predicted_delta.norm(dim=-1).max().item():.9f}",
            "dental_postprocess": (
                str(dental_report["reason"]) if dental_report is not None else ""
            ),
        }
        rows.append(row)
        print(
            f"[{index:02d}/{len(action_unit_ids):02d}] AU{action_unit_id:02d} "
            f"{action_unit_name} mean={row['delta_mean_norm']} "
            f"max={row['delta_max_norm']}",
            flush=True,
        )

    neutral_vertices = demo.output_vertices_for_convention(
        original_mesh["vertices"],
        args.output_convention,
    )
    write_predictions(
        prediction_output,
        neutral_vertices=neutral_vertices,
        action_unit_ids=action_unit_ids,
        blendshape_vertices=blendshape_vertices,
    )
    write_summary(output_path.with_suffix(".csv"), rows)
    imported_targets: set[str] | None = None
    if not args.skip_mesh_export:
        demo.export_fbx_appearance_preserving_blendshape_glb(
            source_path=mesh_path,
            neutral_vertices=neutral_vertices,
            blendshape_vertices=blendshape_vertices,
            path=output_path,
            mesh_object_names=raw_mesh.get("source_mesh_object_names"),
        )
        imported_targets = verify_glb_morph_targets(output_path)
        expected_targets = set(blendshape_vertices)
        if imported_targets != expected_targets:
            missing = sorted(expected_targets - imported_targets)
            unexpected = sorted(imported_targets - expected_targets)
            raise RuntimeError(
                "Exported GLB morph-target verification failed: "
                f"missing={missing}, unexpected={unexpected}."
            )

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Loaded mesh:       {mesh_path}")
    if lip_landmark_report is not None:
        print(
            "Lip landmark surface snap: "
            f"changed={lip_landmark_report.snapped_lip_landmark_count}, "
            f"off_component={lip_landmark_report.off_component_lip_landmark_count}, "
            "duplicates="
            f"{lip_landmark_report.duplicate_lip_landmark_count_before}->"
            f"{lip_landmark_report.duplicate_lip_landmark_count_after}"
        )
        print(f"Wrote corrected landmarks: {lip_landmark_output}")
    print(
        "Model input welding: "
        f"{original_mesh['vertices'].shape[0]} -> "
        f"{model_mesh['vertices'].shape[0]} vertices; output retained "
        f"{original_mesh['vertices'].shape[0]} original vertices"
    )
    if imported_targets is not None:
        print(f"Verified morph targets: {len(imported_targets)}")
        print(f"Wrote rigged GLB: {output_path}")
    else:
        print("Skipped rigged GLB export; prediction vertices were retained for FBX export.")
    print(f"Wrote prediction vertices: {prediction_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run every mesh-trained AU in a TopoRig checkpoint and export one "
            "appearance-preserving GLB with named morph targets."
        )
    )
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prediction-output",
        type=Path,
        help="Optional NPZ path for neutral and per-AU retained-topology vertices.",
    )
    parser.add_argument(
        "--skip-mesh-export",
        action="store_true",
        help="Write predictions and the AU summary without exporting a rigged GLB.",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--action-units",
        type=int,
        nargs="+",
        help=(
            "Optional subset of trained action-unit ids to run. The neutral "
            "mesh is still written to the prediction bank."
        ),
    )
    parser.add_argument(
        "--input-convention",
        choices=("auto", "fbx", "custom-glb"),
        default="auto",
    )
    parser.add_argument(
        "--output-convention",
        choices=("metahuman", "model"),
        default="metahuman",
    )
    parser.add_argument(
        "--metahuman-sample-dir",
        type=Path,
        default=demo.DEFAULT_METAHUMAN_SAMPLE_DIR,
    )
    parser.add_argument(
        "--landmark-cache-dir",
        type=Path,
        default=demo.DEFAULT_LANDMARK_CACHE_DIR,
    )
    parser.add_argument(
        "--mediapipe-mapper",
        type=Path,
        default=demo.DEFAULT_MEDIAPIPE_MAPPER,
    )
    parser.add_argument("--blender", default=None)
    parser.add_argument("--refresh-landmarks", action="store_true")
    parser.add_argument("--min-alignment-landmarks", type=int, default=32)
    parser.add_argument("--weld-tolerance", type=float)
    parser.add_argument(
        "--snap-lip-landmarks-to-face-surface",
        action="store_true",
        help=(
            "Reassign MediaPipe lip landmarks to unique front-facing vertices "
            "on the landmark-voted facial component before alignment/inference."
        ),
    )
    parser.add_argument(
        "--lip-landmark-front-direction",
        choices=("negative", "positive"),
        default="negative",
        help="Direction along the FBX Y axis from the head toward the camera.",
    )
    parser.add_argument(
        "--no-rigid-jaw-teeth",
        action="store_true",
        help="Disable rigid lower-teeth correction for ICT jawOpen output.",
    )
    return parser.parse_args()


def trained_action_unit_ids(config: Mapping[str, Any]) -> list[int]:
    train_config = config.get("data", {}).get("train", {})
    mesh_args = train_config.get("mesh_args", {})
    raw_action_units = mesh_args.get("action_units")
    if raw_action_units is None:
        raise ValueError("Checkpoint config does not list mesh-trained action units.")
    action_units = sorted({int(value) for value in raw_action_units})
    unknown = [value for value in action_units if value not in AU_NAME]
    if unknown:
        raise ValueError(f"Checkpoint config contains unknown AU ids: {unknown}")
    return action_units


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "action_unit_id",
                "action_unit_name",
                "delta_mean_norm",
                "delta_max_norm",
                "dental_postprocess",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_lip_landmark_mapping(
    path: Path,
    *,
    mapping: Mapping[int, int],
    report: Mapping[str, Any],
    source_mesh: Path,
    source_landmarks: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "source_mesh": str(source_mesh),
        "source_landmarks": str(source_landmarks),
        "repair": dict(report),
        "mapping": {
            str(landmark_id): int(vertex_id)
            for landmark_id, vertex_id in sorted(mapping.items())
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_predictions(
    path: Path,
    *,
    neutral_vertices: torch.Tensor,
    action_unit_ids: list[int],
    blendshape_vertices: Mapping[str, torch.Tensor],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [AU_NAME[action_unit_id] for action_unit_id in action_unit_ids]
    targets = torch.stack(
        [blendshape_vertices[name] for name in names],
        dim=0,
    )
    np.savez_compressed(
        path,
        neutral_vertices=neutral_vertices.detach().cpu().numpy().astype(np.float32),
        target_vertices=targets.detach().cpu().numpy().astype(np.float32),
        action_unit_ids=np.asarray(action_unit_ids, dtype=np.int64),
        action_unit_names=np.asarray(names),
    )


def verify_glb_morph_targets(path: Path) -> set[str]:
    try:
        import bpy
    except ImportError as exc:
        raise ImportError("GLB morph-target verification requires Blender bpy.") from exc

    before_objects = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported_objects = tuple(
        obj for obj in bpy.data.objects if obj not in before_objects
    )
    try:
        names: set[str] = set()
        for obj in imported_objects:
            if obj.type != "MESH" or obj.data.shape_keys is None:
                continue
            names.update(
                key.name
                for key in obj.data.shape_keys.key_blocks
                if key.name != "Basis"
            )
        return names
    finally:
        for obj in imported_objects:
            if obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)


if __name__ == "__main__":
    main()
