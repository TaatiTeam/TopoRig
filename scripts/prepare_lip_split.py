#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import demo
import train
from utils.fbx_to_tensor import AU_NAME
from utils.lip_split import (
    LipRegions,
    bridge_excess_mouth_corner_gap,
    build_mediapipe_lip_annulus,
    build_mouth_boundary_lining,
    build_mouth_cavity_disk,
    clear_source_faces_under_lip_annulus,
    clip_mesh_to_smooth_mouth_contour,
    detect_jaw_open_aperture_faces,
    detect_stretched_mouth_faces,
    fill_secondary_mouth_boundary_holes,
    lip_regions_from_mediapipe_mapping,
    lip_regions_from_mediapipe_paths,
    load_lip_regions,
    load_vertex_ids_arg,
    mouth_band_face_ids_from_mediapipe,
    orient_front_surface_faces,
    repair_mediapipe_lip_landmarks_to_dominant_component,
    remap_lower_lip_landmarks,
    remap_vertex_mapping,
    smooth_mouth_boundary_loops,
    split_lip_connections,
    split_lips_along_mediapipe_seam,
    weld_coincident_vertices,
)
from utils.metahuman_oral import (
    fit_oral_assembly,
    remove_disconnected_head_mouth_components,
    remove_head_faces_behind_outer_lips,
)


JAW_OPEN_AU_ID = next(
    action_unit_id
    for action_unit_id, action_unit_name in AU_NAME.items()
    if action_unit_name == "jawOpen"
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def rgb_channel(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 255:
        raise argparse.ArgumentTypeError("RGB channels must be in [0, 255]")
    return parsed


def should_build_proxy_before_split(
    source_face_count: int,
    target_face_count: int,
    maximum_source_ratio: float,
) -> bool:
    """Keep seam construction after decimation only for moderate reductions."""

    if source_face_count <= 0 or target_face_count <= 0:
        raise ValueError("Proxy source and target face counts must be positive.")
    if maximum_source_ratio <= 1.0:
        raise ValueError(
            "Proxy-before-split maximum source ratio must be greater than one."
        )
    return (
        source_face_count > target_face_count
        and float(source_face_count) / float(target_face_count)
        <= float(maximum_source_ratio)
    )


def main() -> None:
    pipeline_started = time.monotonic()
    args = parse_args()
    mesh_path = args.mesh.expanduser().resolve()
    checkpoint_path = demo.resolve_checkpoint_path(args.checkpoint)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ROOT / "outputs" / f"{mesh_path.stem}_lip_split"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or mesh_path.stem

    checkpoint = demo.load_checkpoint(checkpoint_path)
    config = demo.checkpoint_config(checkpoint, args.config)
    device = demo.resolve_device(args.device, config)
    model = demo.load_model(checkpoint, config, device)

    raw_mesh = demo.load_mesh_for_model(mesh_path)
    input_landmarks = resolve_input_landmarks(args, mesh_path)
    weld_report = None
    if args.split_mode in {"mediapipe-seam", "weld-only"} and not args.no_weld_coincident_vertices:
        raw_mesh, weld_report, original_to_welded = weld_coincident_vertices(raw_mesh)
        if input_landmarks is not None:
            input_landmarks = remap_vertex_mapping(
                input_landmarks,
                original_to_welded,
            )
    lip_landmark_topology_repair_report = None
    if (
        args.split_mode == "mediapipe-seam"
        and input_landmarks is not None
        and not args.no_repair_mediapipe_lip_landmarks
    ):
        (
            input_landmarks,
            lip_landmark_topology_repair_report,
        ) = repair_mediapipe_lip_landmarks_to_dominant_component(
            raw_mesh,
            input_landmarks,
            maximum_snap_distance_ratio=(
                args.lip_landmark_maximum_snap_distance_ratio
            ),
            horizontal_axis=args.seam_horizontal_axis,
            depth_axis=args.seam_depth_axis,
            vertical_axis=args.seam_vertical_axis,
        )
    proxy_report = None
    proxy_built_before_split = False
    if args.proxy_before_split_maximum_source_ratio <= 1.0:
        raise ValueError(
            "Proxy-before-split maximum source ratio must be greater than one."
        )
    proxy_source_reduction_ratio = (
        float(raw_mesh["faces"].shape[0]) / float(args.proxy_target_faces)
        if args.proxy_target_faces is not None
        else 0.0
    )
    build_proxy_before_split = (
        args.proxy_before_split
        and args.proxy_target_faces is not None
        and should_build_proxy_before_split(
            int(raw_mesh["faces"].shape[0]),
            int(args.proxy_target_faces),
            float(args.proxy_before_split_maximum_source_ratio),
        )
    )
    if build_proxy_before_split:
        if args.split_mode != "mediapipe-seam" or input_landmarks is None:
            raise ValueError(
                "Pre-split proxy construction requires MediaPipe seam splitting."
            )
        if args.proxy_output_resolution != "proxy":
            raise ValueError(
                "Pre-split proxy construction only supports proxy-resolution output."
            )
        (
            raw_mesh,
            input_landmarks,
            _pre_split_dense_to_proxy,
            proxy_report,
        ) = build_inference_proxy(
            raw_mesh,
            input_landmarks,
            target_faces=args.proxy_target_faces,
            aggression=args.proxy_aggression,
        )
        proxy_report["build_order"] = "before-lip-split"
        proxy_built_before_split = True
        print(
            "Pre-split proxy:   "
            f"{proxy_report['original_face_count']} -> "
            f"{proxy_report['proxy_face_count']} faces in "
            f"{proxy_report['build_seconds']:.2f}s",
            flush=True,
        )
    elif (
        args.proxy_before_split
        and args.proxy_target_faces is not None
        and int(raw_mesh["faces"].shape[0]) > args.proxy_target_faces
    ):
        print(
            "Pre-split proxy:   deferred until after lip split "
            f"(source_ratio={proxy_source_reduction_ratio:.2f} > "
            f"{args.proxy_before_split_maximum_source_ratio:.2f})",
            flush=True,
        )
    lip_regions: Optional[LipRegions] = None
    split_lip_regions: Optional[LipRegions] = None
    if args.split_mode == "mediapipe-seam":
        if input_landmarks is None:
            raise ValueError(
                "--split-mode mediapipe-seam requires MediaPipe landmarks."
            )
        split_mesh, split_report = split_lips_along_mediapipe_seam(
            raw_mesh,
            input_landmarks,
            horizontal_axis=args.seam_horizontal_axis,
            depth_axis=args.seam_depth_axis,
            vertical_axis=args.seam_vertical_axis,
            depth_tolerance=args.seam_depth_tolerance,
            vertical_tolerance=args.seam_vertical_tolerance,
            horizontal_padding=args.seam_horizontal_padding,
            corner_inset_ratio=args.seam_corner_inset_ratio,
            preopen_distance=args.seam_preopen_distance,
        )
    elif args.split_mode == "region":
        lip_regions = resolve_lip_regions(args, raw_mesh, input_landmarks)
        split_mesh, split_report, split_lip_regions = split_lip_connections(
            raw_mesh,
            lip_regions,
            remove_mixed_faces=not args.keep_mixed_faces,
            extra_remove_face_ids=resolve_extra_removed_face_ids(
                args,
                raw_mesh,
                input_landmarks,
            ),
        )
    else:
        split_mesh = raw_mesh
        split_report = None
    if input_landmarks is None:
        split_landmarks = None
    elif split_report is None:
        split_landmarks = input_landmarks
    else:
        split_landmarks = remap_lower_lip_landmarks(input_landmarks, split_report)

    outputs: dict[str, str] = {}
    if not args.no_raw_split_mesh:
        raw_split_path = output_dir / f"{prefix}_raw_lip_split.{args.output_format}"
        demo.export_mesh(
            vertices=split_mesh["vertices"],
            faces=split_mesh["faces"],
            normals=split_mesh["normals"],
            path=raw_split_path,
        )
        outputs["raw_split_mesh"] = str(raw_split_path)

    processed_mesh = preprocess_split_mesh_for_inference(
        split_mesh=split_mesh,
        config=config,
        mesh_path=mesh_path,
        input_convention=args.input_convention,
        input_landmarks=split_landmarks,
        reference_mesh_path=demo.metahuman_sample_mesh(
            args.metahuman_sample_dir.expanduser()
        ),
        reference_landmarks=resolve_reference_landmarks(args),
        reference_input_convention=args.reference_input_convention,
        min_alignment_landmarks=args.min_alignment_landmarks,
        alignment_mode=args.alignment_mode,
        reference_mesh_cache_path=args.reference_processed_cache,
    )
    dense_processed_mesh = processed_mesh
    dense_probe_source_faces = processed_mesh["faces"].clone()
    dense_to_proxy = None
    probe_landmarks = split_landmarks
    if (
        not proxy_built_before_split
        and args.proxy_target_faces is not None
        and int(processed_mesh["faces"].shape[0]) > args.proxy_target_faces
    ):
        if split_landmarks is None:
            raise ValueError("Proxy inference requires MediaPipe landmarks.")
        (
            processed_mesh,
            probe_landmarks,
            dense_to_proxy,
            proxy_report,
        ) = build_inference_proxy(
            processed_mesh,
            split_landmarks,
            target_faces=args.proxy_target_faces,
            aggression=args.proxy_aggression,
        )
        proxy_report["build_order"] = "after-lip-split"
        print(
            "Inference proxy:   "
            f"{proxy_report['original_face_count']} -> "
            f"{proxy_report['proxy_face_count']} faces in "
            f"{proxy_report['build_seconds']:.2f}s",
            flush=True,
        )
    probe_source_faces = processed_mesh["faces"].clone()

    aperture_cut_enabled = (
        args.split_mode == "mediapipe-seam"
        and not args.no_probe_aperture_cut
        and probe_landmarks is not None
    )
    aperture_pass_reports: list[dict[str, Any]] = []
    stretch_pass_reports: list[dict[str, Any]] = []
    aperture_total_removed_faces = 0
    stretch_total_detected_faces = 0
    probe_total_removed_faces = 0
    aperture_remaining_report = None
    stretch_remaining_report = None
    probe_predicted_delta = None
    probe_face_set_converged = not aperture_cut_enabled

    if aperture_cut_enabled:
        source_face_count = int(probe_source_faces.shape[0])
        current_keep_faces = torch.ones(source_face_count, dtype=torch.bool)
        safe_streak = torch.zeros(source_face_count, dtype=torch.int16)
        restored_once = torch.zeros(source_face_count, dtype=torch.bool)
        restore_blocked = torch.zeros(source_face_count, dtype=torch.bool)
        for pass_index in range(args.probe_aperture_max_passes):
            current_deformed_vertices, current_predicted_delta = demo.run_model(
                model=model,
                mesh=processed_mesh,
                action_unit_id=JAW_OPEN_AU_ID,
                config=config,
                device=device,
            )
            if probe_predicted_delta is None:
                probe_predicted_delta = current_predicted_delta

            aperture_cut_report = detect_jaw_open_aperture_faces(
                current_deformed_vertices,
                probe_source_faces,
                probe_landmarks,
                horizontal_axis=args.seam_horizontal_axis,
                depth_axis=args.seam_depth_axis,
                vertical_axis=args.seam_vertical_axis,
                aperture_scale=args.probe_aperture_scale,
                overlap_inset_scale=args.probe_aperture_overlap_inset_scale,
                depth_padding=args.probe_aperture_depth_padding,
            )
            pass_report = aperture_cut_report.to_dict()
            pass_report["pass_index"] = pass_index + 1
            stretch_report = detect_stretched_mouth_faces(
                processed_mesh["vertices"],
                current_deformed_vertices,
                probe_source_faces,
                probe_landmarks,
                horizontal_axis=args.seam_horizontal_axis,
                depth_axis=args.seam_depth_axis,
                vertical_axis=args.seam_vertical_axis,
                edge_ratio_threshold=args.probe_stretch_edge_ratio,
                edge_growth_threshold=args.probe_stretch_edge_growth,
                region_scale=args.probe_stretch_region_scale,
                depth_padding=args.probe_stretch_depth_padding,
            )
            removed_face_ids = sorted(
                set(aperture_cut_report.removed_face_ids)
                | set(stretch_report.removed_face_ids)
            )
            unsafe_faces = torch.zeros(source_face_count, dtype=torch.bool)
            if removed_face_ids:
                unsafe_faces[
                    torch.tensor(removed_face_ids, dtype=torch.long)
                ] = True

            removed_but_safe = (
                ~current_keep_faces & ~unsafe_faces & ~restore_blocked
            )
            safe_streak[removed_but_safe] += 1
            safe_streak[~removed_but_safe] = 0

            next_keep_faces = current_keep_faces.clone()
            newly_unsafe = current_keep_faces & unsafe_faces
            restore_blocked[newly_unsafe & restored_once] = True
            next_keep_faces[newly_unsafe] = False
            safe_to_restore = (
                ~current_keep_faces
                & ~unsafe_faces
                & ~restore_blocked
                & (safe_streak >= args.probe_restore_safe_passes)
            )
            next_keep_faces[safe_to_restore] = True
            restored_once[safe_to_restore] = True
            safe_streak[safe_to_restore] = 0

            newly_removed_count = int(
                (current_keep_faces & ~next_keep_faces).sum().item()
            )
            restored_count = int(
                (~current_keep_faces & next_keep_faces).sum().item()
            )
            face_set_matches = torch.equal(next_keep_faces, current_keep_faces)
            pending_restore_count = int(
                (
                    ~next_keep_faces
                    & ~unsafe_faces
                    & ~restore_blocked
                ).sum().item()
            )
            face_set_stable = face_set_matches and pending_restore_count == 0
            pass_report["action"] = (
                "converged" if face_set_stable else "reconcile"
            )
            pass_report["newly_removed_face_count"] = newly_removed_count
            pass_report["restored_face_count"] = restored_count
            pass_report["restore_blocked_face_count"] = int(
                restore_blocked.sum().item()
            )
            pass_report["pending_restore_face_count"] = pending_restore_count
            pass_report["target_face_count"] = int(next_keep_faces.sum().item())
            aperture_pass_reports.append(pass_report)
            stretch_pass_report = stretch_report.to_dict()
            stretch_pass_report["pass_index"] = pass_index + 1
            stretch_pass_report["action"] = pass_report["action"]
            stretch_pass_reports.append(stretch_pass_report)

            if args.proxy_one_shot_probe:
                processed_mesh = dict(processed_mesh)
                processed_mesh["faces"] = probe_source_faces[
                    next_keep_faces
                ].contiguous()
                current_keep_faces = next_keep_faces
                deformed_vertices = current_deformed_vertices
                predicted_delta = current_predicted_delta
                probe_face_set_converged = True
                pass_report["action"] = "one-shot"
                stretch_pass_report["action"] = "one-shot"
                break

            if face_set_stable:
                deformed_vertices = current_deformed_vertices
                predicted_delta = current_predicted_delta
                probe_face_set_converged = True
                break

            processed_mesh = dict(processed_mesh)
            processed_mesh["faces"] = probe_source_faces[next_keep_faces].contiguous()
            current_keep_faces = next_keep_faces
        else:
            # Return a prediction for the final bounded face set even when the
            # reconciliation limit is reached.
            deformed_vertices, predicted_delta = demo.run_model(
                model=model,
                mesh=processed_mesh,
                action_unit_id=JAW_OPEN_AU_ID,
                config=config,
                device=device,
            )

        aperture_total_removed_faces = (
            aperture_pass_reports[-1]["removed_face_count"]
            if aperture_pass_reports
            else 0
        )
        stretch_total_detected_faces = (
            stretch_pass_reports[-1]["removed_face_count"]
            if stretch_pass_reports
            else 0
        )
        probe_total_removed_faces = int((~current_keep_faces).sum().item())
        aperture_remaining_report = detect_jaw_open_aperture_faces(
            deformed_vertices,
            processed_mesh["faces"],
            probe_landmarks,
            horizontal_axis=args.seam_horizontal_axis,
            depth_axis=args.seam_depth_axis,
            vertical_axis=args.seam_vertical_axis,
            aperture_scale=args.probe_aperture_scale,
            overlap_inset_scale=args.probe_aperture_overlap_inset_scale,
            depth_padding=args.probe_aperture_depth_padding,
        )
        stretch_remaining_report = detect_stretched_mouth_faces(
            processed_mesh["vertices"],
            deformed_vertices,
            processed_mesh["faces"],
            probe_landmarks,
            horizontal_axis=args.seam_horizontal_axis,
            depth_axis=args.seam_depth_axis,
            vertical_axis=args.seam_vertical_axis,
            edge_ratio_threshold=args.probe_stretch_edge_ratio,
            edge_growth_threshold=args.probe_stretch_edge_growth,
            region_scale=args.probe_stretch_region_scale,
            depth_padding=args.probe_stretch_depth_padding,
        )
    else:
        deformed_vertices, predicted_delta = demo.run_model(
            model=model,
            mesh=processed_mesh,
            action_unit_id=JAW_OPEN_AU_ID,
            config=config,
            device=device,
        )
        probe_predicted_delta = predicted_delta

    if dense_to_proxy is not None and args.proxy_output_resolution == "dense":
        transfer_start = time.monotonic()
        proxy_predicted_delta = predicted_delta
        predicted_delta = proxy_predicted_delta.index_select(0, dense_to_proxy)
        if args.proxy_transfer_smoothing_iterations > 0:
            predicted_delta = smooth_proxy_displacement_transfer(
                predicted_delta,
                dense_processed_mesh["faces"],
                pinned_vertex_ids=split_landmarks.values(),
                iterations=args.proxy_transfer_smoothing_iterations,
                blend=args.proxy_transfer_smoothing_blend,
                device=device,
            )
        deformed_vertices = dense_processed_mesh["vertices"] + predicted_delta
        probe_predicted_delta = predicted_delta
        processed_mesh = dict(dense_processed_mesh)
        probe_source_faces = dense_probe_source_faces

        dense_aperture_report = detect_jaw_open_aperture_faces(
            deformed_vertices,
            probe_source_faces,
            split_landmarks,
            horizontal_axis=args.seam_horizontal_axis,
            depth_axis=args.seam_depth_axis,
            vertical_axis=args.seam_vertical_axis,
            aperture_scale=args.probe_aperture_scale,
            overlap_inset_scale=args.probe_aperture_overlap_inset_scale,
            depth_padding=args.probe_aperture_depth_padding,
        )
        dense_stretch_report = detect_stretched_mouth_faces(
            processed_mesh["vertices"],
            deformed_vertices,
            probe_source_faces,
            split_landmarks,
            horizontal_axis=args.seam_horizontal_axis,
            depth_axis=args.seam_depth_axis,
            vertical_axis=args.seam_vertical_axis,
            edge_ratio_threshold=args.probe_stretch_edge_ratio,
            edge_growth_threshold=args.probe_stretch_edge_growth,
            region_scale=args.probe_stretch_region_scale,
            depth_padding=args.probe_stretch_depth_padding,
        )
        dense_removed_ids = sorted(
            set(dense_aperture_report.removed_face_ids)
            | set(dense_stretch_report.removed_face_ids)
        )
        dense_keep = torch.ones(probe_source_faces.shape[0], dtype=torch.bool)
        if dense_removed_ids:
            dense_keep[torch.tensor(dense_removed_ids, dtype=torch.long)] = False
        processed_mesh["faces"] = probe_source_faces[dense_keep].contiguous()
        aperture_total_removed_faces = len(dense_aperture_report.removed_face_ids)
        stretch_total_detected_faces = len(dense_stretch_report.removed_face_ids)
        probe_total_removed_faces = len(dense_removed_ids)
        aperture_remaining_report = detect_jaw_open_aperture_faces(
            deformed_vertices,
            processed_mesh["faces"],
            split_landmarks,
            horizontal_axis=args.seam_horizontal_axis,
            depth_axis=args.seam_depth_axis,
            vertical_axis=args.seam_vertical_axis,
            aperture_scale=args.probe_aperture_scale,
            overlap_inset_scale=args.probe_aperture_overlap_inset_scale,
            depth_padding=args.probe_aperture_depth_padding,
        )
        stretch_remaining_report = detect_stretched_mouth_faces(
            processed_mesh["vertices"],
            deformed_vertices,
            processed_mesh["faces"],
            split_landmarks,
            horizontal_axis=args.seam_horizontal_axis,
            depth_axis=args.seam_depth_axis,
            vertical_axis=args.seam_vertical_axis,
            edge_ratio_threshold=args.probe_stretch_edge_ratio,
            edge_growth_threshold=args.probe_stretch_edge_growth,
            region_scale=args.probe_stretch_region_scale,
            depth_padding=args.probe_stretch_depth_padding,
        )
        probe_face_set_converged = (
            aperture_remaining_report.removed_face_count == 0
            and stretch_remaining_report.removed_face_count == 0
        )
        proxy_report["transfer_seconds"] = time.monotonic() - transfer_start
        proxy_report["dense_removed_face_count"] = probe_total_removed_faces
        proxy_report["transfer_mode"] = "collapse-cluster-displacement"
        proxy_report["transfer_smoothing_iterations"] = int(
            args.proxy_transfer_smoothing_iterations
        )
        proxy_report["transfer_smoothing_blend"] = float(
            args.proxy_transfer_smoothing_blend
        )
    elif dense_to_proxy is not None:
        split_landmarks = probe_landmarks
        proxy_report["transfer_mode"] = "proxy-resolution-output"
        proxy_report["transfer_seconds"] = 0.0
        proxy_report["dense_removed_face_count"] = 0
        proxy_report["transfer_smoothing_iterations"] = 0
        proxy_report["transfer_smoothing_blend"] = 0.0

    if probe_predicted_delta is None:
        raise RuntimeError("The jawOpen probe did not produce a prediction.")

    mouth_contour_clip_report = None
    if args.smooth_contour_clip:
        if args.split_mode != "mediapipe-seam" or split_landmarks is None:
            raise ValueError(
                "Smooth contour clipping requires MediaPipe seam splitting."
            )
        print("Smooth contour clip: reconstructing boundary faces...", flush=True)
        (
            clipped_neutral,
            deformed_vertices,
            clipped_faces,
            mouth_contour_clip_report,
        ) = clip_mesh_to_smooth_mouth_contour(
            processed_mesh["vertices"],
            deformed_vertices,
            probe_source_faces,
            processed_mesh["faces"],
            split_landmarks,
            horizontal_axis=args.seam_horizontal_axis,
            depth_axis=args.seam_depth_axis,
            vertical_axis=args.seam_vertical_axis,
            contour_scale=args.smooth_contour_scale,
            depth_padding=args.smooth_contour_depth_padding,
            edge_ratio_threshold=args.probe_stretch_edge_ratio,
            edge_growth_threshold=args.probe_stretch_edge_growth,
        )
        processed_mesh = dict(processed_mesh)
        processed_mesh["vertices"] = clipped_neutral
        processed_mesh["faces"] = clipped_faces
        print(
            "Smooth contour clip: "
            f"restored={mouth_contour_clip_report.restored_boundary_face_count} "
            f"clipped={mouth_contour_clip_report.clipped_face_count} "
            f"added_vertices={mouth_contour_clip_report.added_vertex_count}",
            flush=True,
        )

    mouth_boundary_smooth_report = None
    mouth_hole_fill_reports = []
    mouth_hole_fill_report = None
    if args.split_mode == "mediapipe-seam" and not args.no_fill_secondary_mouth_holes:
        if split_landmarks is None:
            raise ValueError("Mouth hole filling requires MediaPipe landmarks.")
        (
            filled_neutral,
            deformed_vertices,
            filled_faces,
            mouth_hole_fill_report,
        ) = fill_secondary_mouth_boundary_holes(
            processed_mesh["vertices"],
            deformed_vertices,
            processed_mesh["faces"],
            split_landmarks,
            reference_faces=probe_source_faces,
            horizontal_axis=args.seam_horizontal_axis,
            depth_axis=args.seam_depth_axis,
            vertical_axis=args.seam_vertical_axis,
            maximum_area_ratio=args.mouth_hole_max_area_ratio,
        )
        processed_mesh = dict(processed_mesh)
        processed_mesh["vertices"] = filled_neutral
        processed_mesh["faces"] = filled_faces
        mouth_hole_fill_reports.append(mouth_hole_fill_report)

    if args.split_mode == "mediapipe-seam" and not args.no_smooth_mouth_boundary:
        if split_landmarks is None:
            raise ValueError("Mouth boundary smoothing requires MediaPipe landmarks.")
        smoothed_neutral, deformed_vertices, mouth_boundary_smooth_report = (
            smooth_mouth_boundary_loops(
                processed_mesh["vertices"],
                deformed_vertices,
                processed_mesh["faces"],
                split_landmarks,
                reference_faces=probe_source_faces,
                horizontal_axis=args.seam_horizontal_axis,
                depth_axis=args.seam_depth_axis,
                vertical_axis=args.seam_vertical_axis,
                maximum_displacement=args.mouth_boundary_max_displacement,
                blend_rings=args.mouth_boundary_blend_rings,
            )
        )
        processed_mesh = dict(processed_mesh)
        processed_mesh["vertices"] = smoothed_neutral

    if (
        mouth_hole_fill_report is not None
        and mouth_hole_fill_report.rejected_hole_cycle_count > 0
    ):
        (
            filled_neutral,
            deformed_vertices,
            filled_faces,
            mouth_hole_fill_report,
        ) = fill_secondary_mouth_boundary_holes(
            processed_mesh["vertices"],
            deformed_vertices,
            processed_mesh["faces"],
            split_landmarks,
            reference_faces=probe_source_faces,
            horizontal_axis=args.seam_horizontal_axis,
            depth_axis=args.seam_depth_axis,
            vertical_axis=args.seam_vertical_axis,
            maximum_area_ratio=args.mouth_hole_max_area_ratio,
        )
        processed_mesh = dict(processed_mesh)
        processed_mesh["vertices"] = filled_neutral
        processed_mesh["faces"] = filled_faces
        mouth_hole_fill_reports.append(mouth_hole_fill_report)

    mouth_corner_bridge_report = None
    if args.split_mode == "mediapipe-seam" and not args.no_repair_mouth_corners:
        if split_landmarks is None:
            raise ValueError("Mouth corner repair requires MediaPipe landmarks.")
        (
            repaired_neutral,
            deformed_vertices,
            repaired_faces,
            mouth_corner_bridge_report,
        ) = bridge_excess_mouth_corner_gap(
            processed_mesh["vertices"],
            deformed_vertices,
            processed_mesh["faces"],
            split_landmarks,
            reference_faces=probe_source_faces,
            horizontal_axis=args.seam_horizontal_axis,
            depth_axis=args.seam_depth_axis,
            vertical_axis=args.seam_vertical_axis,
            corner_inset_ratio=args.mouth_corner_inset_ratio,
            minimum_lateral_overshoot_ratio=(
                args.mouth_corner_minimum_overshoot_ratio
            ),
            maximum_repaired_sides=args.mouth_corner_maximum_repaired_sides,
            chain_padding_vertices=args.mouth_corner_chain_padding_vertices,
        )
        processed_mesh = dict(processed_mesh)
        processed_mesh["vertices"] = repaired_neutral
        processed_mesh["faces"] = repaired_faces

    post_bridge_smooth_report = None
    if args.post_bridge_smoothing:
        if args.split_mode != "mediapipe-seam" or split_landmarks is None:
            raise ValueError(
                "Post-bridge smoothing requires MediaPipe seam splitting and landmarks."
            )
        weighted_unbridged_smoothing = (
            mouth_corner_bridge_report is None
            or mouth_corner_bridge_report.added_face_count == 0
        )
        smoothed_neutral, deformed_vertices, post_bridge_smooth_report = (
            smooth_mouth_boundary_loops(
                processed_mesh["vertices"],
                deformed_vertices,
                processed_mesh["faces"],
                split_landmarks,
                reference_faces=probe_source_faces,
                horizontal_axis=args.seam_horizontal_axis,
                depth_axis=args.seam_depth_axis,
                vertical_axis=args.seam_vertical_axis,
                maximum_displacement=args.post_bridge_max_displacement,
                blend_rings=args.post_bridge_blend_rings,
                smoothing_cycles=args.post_bridge_smoothing_cycles,
                preserve_boundary_junctions=weighted_unbridged_smoothing,
                arc_length_weighted=weighted_unbridged_smoothing,
            )
        )
        processed_mesh = dict(processed_mesh)
        processed_mesh["vertices"] = smoothed_neutral

    oral_fit = None
    oral_cleanup_maximum_depth = None
    if args.metahuman_oral_asset is not None:
        if split_landmarks is None:
            raise ValueError("Oral assembly fitting requires mouth landmarks.")
        oral_fit = fit_requested_oral_assembly(
            args,
            processed_mesh["vertices"],
            deformed_vertices,
            split_landmarks,
        )
        oral_cleanup_maximum_depth = float(
            oral_fit[1][:, args.seam_depth_axis].amax().item()
        )
    oral_head_aperture_report = None
    oral_head_component_report = None
    if (
        args.metahuman_oral_asset is not None
        and not args.no_oral_head_aperture_cleanup
    ):
        if split_landmarks is None:
            raise ValueError("Oral head-aperture cleanup requires mouth landmarks.")
        cleaned_faces, oral_head_aperture_report = (
            remove_head_faces_behind_outer_lips(
                deformed_vertices,
                processed_mesh["faces"],
                split_landmarks,
                horizontal_axis=args.seam_horizontal_axis,
                depth_axis=args.seam_depth_axis,
                vertical_axis=args.seam_vertical_axis,
                scale=args.oral_head_aperture_scale,
                depth_padding_ratio=(
                    args.oral_head_depth_padding_ratio
                ),
                maximum_depth=oral_cleanup_maximum_depth,
                boundary_distance_ratio=(
                    args.oral_head_boundary_distance_ratio
                ),
                boundary_blend_rings=args.oral_head_boundary_blend_rings,
                maximum_protected_edge_ratio=(
                    args.oral_head_maximum_protected_edge_ratio
                ),
                protect_lower_lip=(
                    not args.no_oral_head_lower_lip_protection
                ),
                lower_lip_depth_tolerance_ratio=(
                    args.oral_head_lower_lip_depth_tolerance_ratio
                ),
            )
        )
        processed_mesh = dict(processed_mesh)
        processed_mesh["faces"] = cleaned_faces

        cleaned_faces, oral_head_component_report = (
            remove_disconnected_head_mouth_components(
                deformed_vertices,
                processed_mesh["faces"],
                torch.tensor(oral_head_aperture_report.aperture_polygon),
                mapping=split_landmarks,
                depth_axis=args.seam_depth_axis,
                maximum_depth=oral_cleanup_maximum_depth,
                depth_padding_ratio=args.oral_head_depth_padding_ratio,
                horizontal_axis=args.seam_horizontal_axis,
                vertical_axis=args.seam_vertical_axis,
                margin_ratio=args.oral_head_component_margin_ratio,
                maximum_component_faces=args.oral_head_component_maximum_faces,
            )
        )
        processed_mesh["faces"] = cleaned_faces

    lip_annulus_repair_report = None
    lip_annulus_head_cleanup_report = None
    lip_annulus_vertex_count = 0
    if args.reconstruct_lip_annulus:
        if args.split_mode != "mediapipe-seam" or split_landmarks is None:
            raise ValueError(
                "Lip annulus reconstruction requires MediaPipe seam splitting."
            )
        (
            neutral_lip_annulus,
            deformed_lip_annulus,
            lip_annulus_faces,
            lip_annulus_repair_report,
        ) = build_mediapipe_lip_annulus(
            processed_mesh["vertices"],
            deformed_vertices,
            split_landmarks,
            horizontal_axis=args.seam_horizontal_axis,
            depth_axis=args.seam_depth_axis,
            vertical_axis=args.seam_vertical_axis,
            segments=args.lip_annulus_segments,
            radial_rings=args.lip_annulus_radial_rings,
            outer_scale=args.lip_annulus_outer_scale,
            thickness_ratio=args.lip_annulus_thickness_ratio,
            side_thickness_factor=args.lip_annulus_side_thickness_factor,
            maximum_radial_inset_fraction=(
                args.lip_annulus_maximum_radial_inset_fraction
            ),
            inner_recess_depth=args.lip_annulus_inner_recess_depth,
            front_offset=args.lip_annulus_front_offset,
            bulge_depth=args.lip_annulus_bulge_depth,
            upper_smoothing_window=args.lip_annulus_upper_smoothing_window,
            lower_smoothing_window=args.lip_annulus_lower_smoothing_window,
        )
        if args.clear_head_under_lip_annulus:
            cleaned_faces, lip_annulus_head_cleanup_report = (
                clear_source_faces_under_lip_annulus(
                    deformed_vertices,
                    processed_mesh["faces"],
                    deformed_lip_annulus,
                    segments=args.lip_annulus_segments,
                    horizontal_axis=args.seam_horizontal_axis,
                    depth_axis=args.seam_depth_axis,
                    vertical_axis=args.seam_vertical_axis,
                    depth_padding_ratio=(
                        args.lip_annulus_cleanup_depth_padding_ratio
                    ),
                )
            )
            processed_mesh = dict(processed_mesh)
            processed_mesh["faces"] = cleaned_faces
        lip_annulus_vertex_count = int(neutral_lip_annulus.shape[0])
        annulus_offset = int(processed_mesh["vertices"].shape[0])
        processed_mesh = dict(processed_mesh)
        processed_mesh["vertices"] = torch.cat(
            (processed_mesh["vertices"], neutral_lip_annulus),
            dim=0,
        ).contiguous()
        deformed_vertices = torch.cat(
            (deformed_vertices, deformed_lip_annulus),
            dim=0,
        ).contiguous()
        processed_mesh["faces"] = torch.cat(
            (
                processed_mesh["faces"],
                lip_annulus_faces + annulus_offset,
            ),
            dim=0,
        ).contiguous()
        print(
            "Lip annulus:       "
            f"vertices={lip_annulus_repair_report.vertex_count} "
            f"faces={lip_annulus_repair_report.face_count} "
            f"thickness_ratio={lip_annulus_repair_report.thickness_ratio:.3f} "
            f"capped_segments="
            f"{lip_annulus_repair_report.capped_deformed_segment_count} "
            f"cleared_head_faces="
            f"{lip_annulus_head_cleanup_report.removed_face_count if lip_annulus_head_cleanup_report is not None else 0}",
            flush=True,
        )

    front_surface_orientation_report = None
    if args.orient_front_surface_faces:
        if split_landmarks is None:
            raise ValueError(
                "Front-surface orientation requires MediaPipe landmarks."
            )
        oriented_faces, front_surface_orientation_report = (
            orient_front_surface_faces(
                processed_mesh["vertices"],
                deformed_vertices,
                processed_mesh["faces"],
                split_landmarks,
                depth_axis=args.seam_depth_axis,
                region_depth_ratio=args.front_surface_depth_ratio,
                normal_tolerance=args.front_surface_normal_tolerance,
            )
        )
        processed_mesh = dict(processed_mesh)
        processed_mesh["faces"] = oriented_faces
        print(
            "Face orientation:  "
            f"eligible={front_surface_orientation_report.eligible_face_count} "
            f"flipped={front_surface_orientation_report.flipped_face_count}",
            flush=True,
        )

    head_neutral_normals = demo.recompute_vertex_normals(
        processed_mesh["vertices"],
        processed_mesh["faces"],
    )
    head_deformed_normals = demo.recompute_vertex_normals(
        deformed_vertices,
        processed_mesh["faces"],
    )
    mouth_cavity_report = None
    neutral_cavity_report = None
    cavity_vertex_count = 0
    oral_assembly_report = None
    oral_vertex_count = 0
    oral_vertex_colors = None
    mouth_boundary_lining_report = None
    lining_vertex_count = 0
    vertex_offset = int(processed_mesh["vertices"].shape[0])
    output_face_parts = [processed_mesh["faces"]]
    neutral_vertex_parts = [processed_mesh["vertices"]]
    deformed_vertex_parts = [deformed_vertices]
    neutral_normal_parts = [head_neutral_normals]
    deformed_normal_parts = [head_deformed_normals]

    if oral_fit is not None:
        (
            neutral_oral_vertices,
            deformed_oral_vertices,
            oral_faces,
            oral_vertex_colors,
            oral_assembly_report,
        ) = oral_fit
        neutral_oral_normals = demo.recompute_vertex_normals(
            neutral_oral_vertices,
            oral_faces,
        )
        deformed_oral_normals = demo.recompute_vertex_normals(
            deformed_oral_vertices,
            oral_faces,
        )
        oral_vertex_count = int(neutral_oral_vertices.shape[0])
        output_face_parts.append(oral_faces + vertex_offset)
        neutral_vertex_parts.append(neutral_oral_vertices)
        deformed_vertex_parts.append(deformed_oral_vertices)
        neutral_normal_parts.append(neutral_oral_normals)
        deformed_normal_parts.append(deformed_oral_normals)
        vertex_offset += oral_vertex_count

    full_metahuman_oral_shell = (
        oral_assembly_report is not None
        and oral_assembly_report.geometry_mode == "full_shell"
    )
    add_mouth_boundary_lining = (
        args.mouth_boundary_lining
        or (
            args.metahuman_oral_asset is not None
            and not full_metahuman_oral_shell
        )
    ) and not args.no_mouth_boundary_lining
    if add_mouth_boundary_lining:
        if split_landmarks is None:
            raise ValueError("Mouth boundary lining requires MediaPipe landmarks.")
        (
            neutral_lining_vertices,
            lining_faces,
            neutral_lining_normals,
            neutral_lining_report,
        ) = build_mouth_boundary_lining(
            processed_mesh["vertices"],
            processed_mesh["faces"],
            split_landmarks,
            reference_vertices=deformed_vertices,
            horizontal_axis=args.seam_horizontal_axis,
            depth_axis=args.seam_depth_axis,
            vertical_axis=args.seam_vertical_axis,
            depth=args.mouth_boundary_lining_depth,
            rim_depth=args.mouth_boundary_lining_rim_depth,
            side_fraction=args.mouth_boundary_lining_side_fraction,
            inner_scale=args.mouth_boundary_lining_inner_scale,
            radial_rings=args.mouth_boundary_lining_radial_rings,
            smoothing_iterations=args.mouth_boundary_lining_smoothing_iterations,
            analytic_segments=args.mouth_boundary_lining_segments,
            opening_scale=args.mouth_boundary_lining_opening_scale,
            shape_exponent=args.mouth_boundary_lining_shape_exponent,
            cap_back=not args.no_mouth_boundary_lining_cap,
            maximum_cycles=args.mouth_boundary_lining_maximum_cycles,
        )
        (
            deformed_lining_vertices,
            deformed_lining_faces,
            deformed_lining_normals,
            mouth_boundary_lining_report,
        ) = build_mouth_boundary_lining(
            deformed_vertices,
            processed_mesh["faces"],
            split_landmarks,
            reference_vertices=deformed_vertices,
            horizontal_axis=args.seam_horizontal_axis,
            depth_axis=args.seam_depth_axis,
            vertical_axis=args.seam_vertical_axis,
            depth=args.mouth_boundary_lining_depth,
            rim_depth=args.mouth_boundary_lining_rim_depth,
            side_fraction=args.mouth_boundary_lining_side_fraction,
            inner_scale=args.mouth_boundary_lining_inner_scale,
            radial_rings=args.mouth_boundary_lining_radial_rings,
            smoothing_iterations=args.mouth_boundary_lining_smoothing_iterations,
            analytic_segments=args.mouth_boundary_lining_segments,
            opening_scale=args.mouth_boundary_lining_opening_scale,
            shape_exponent=args.mouth_boundary_lining_shape_exponent,
            cap_back=not args.no_mouth_boundary_lining_cap,
            maximum_cycles=args.mouth_boundary_lining_maximum_cycles,
        )
        if not torch.equal(lining_faces, deformed_lining_faces):
            raise RuntimeError("Neutral and jawOpen mouth linings have mismatched topology.")
        if neutral_lining_report.boundary_vertex_count != mouth_boundary_lining_report.boundary_vertex_count:
            raise RuntimeError("Neutral and jawOpen mouth linings use different boundaries.")
        lining_vertex_count = int(neutral_lining_vertices.shape[0])
        output_face_parts.append(lining_faces + vertex_offset)
        neutral_vertex_parts.append(neutral_lining_vertices)
        deformed_vertex_parts.append(deformed_lining_vertices)
        neutral_normal_parts.append(neutral_lining_normals)
        deformed_normal_parts.append(deformed_lining_normals)
        vertex_offset += lining_vertex_count

    capped_mouth_compartment = (
        add_mouth_boundary_lining and not args.no_mouth_boundary_lining_cap
    )
    add_mouth_cavity = (
        args.mouth_cavity
        or (
            args.metahuman_oral_asset is not None
            and not capped_mouth_compartment
            and not full_metahuman_oral_shell
        )
    ) and not args.no_mouth_cavity
    if add_mouth_cavity:
        if split_landmarks is None:
            raise ValueError("Mouth cavity generation requires MediaPipe landmarks.")
        (
            neutral_cavity_vertices,
            cavity_faces,
            neutral_cavity_normals,
            neutral_cavity_report,
        ) = build_mouth_cavity_disk(
            processed_mesh["vertices"],
            split_landmarks,
            horizontal_axis=args.seam_horizontal_axis,
            depth_axis=args.seam_depth_axis,
            vertical_axis=args.seam_vertical_axis,
            scale=args.mouth_cavity_scale,
            shape_exponent=args.mouth_cavity_shape_exponent,
            depth_offset=args.mouth_cavity_depth_offset,
            segments=args.mouth_cavity_segments,
            radial_rings=args.mouth_cavity_radial_rings,
            curvature_depth=args.mouth_cavity_curvature_depth,
            inner_radius_ratio=args.mouth_cavity_inner_radius_ratio,
            side_wall_min_horizontal_ratio=(
                args.mouth_cavity_side_wall_min_horizontal_ratio
            ),
        )
        (
            deformed_cavity_vertices,
            deformed_cavity_faces,
            deformed_cavity_normals,
            mouth_cavity_report,
        ) = build_mouth_cavity_disk(
            deformed_vertices,
            split_landmarks,
            horizontal_axis=args.seam_horizontal_axis,
            depth_axis=args.seam_depth_axis,
            vertical_axis=args.seam_vertical_axis,
            scale=args.mouth_cavity_scale,
            shape_exponent=args.mouth_cavity_shape_exponent,
            depth_offset=args.mouth_cavity_depth_offset,
            segments=args.mouth_cavity_segments,
            radial_rings=args.mouth_cavity_radial_rings,
            curvature_depth=args.mouth_cavity_curvature_depth,
            inner_radius_ratio=args.mouth_cavity_inner_radius_ratio,
            side_wall_min_horizontal_ratio=(
                args.mouth_cavity_side_wall_min_horizontal_ratio
            ),
        )
        if not torch.equal(cavity_faces, deformed_cavity_faces):
            raise RuntimeError("Neutral and jawOpen mouth cavities have mismatched topology.")
        cavity_vertex_count = int(neutral_cavity_vertices.shape[0])
        output_face_parts.append(cavity_faces + vertex_offset)
        neutral_vertex_parts.append(neutral_cavity_vertices)
        deformed_vertex_parts.append(deformed_cavity_vertices)
        neutral_normal_parts.append(neutral_cavity_normals)
        deformed_normal_parts.append(deformed_cavity_normals)

    output_faces = torch.cat(output_face_parts, dim=0)
    render_neutral_vertices = torch.cat(neutral_vertex_parts, dim=0)
    render_deformed_vertices = torch.cat(deformed_vertex_parts, dim=0)
    render_neutral_normals = torch.cat(neutral_normal_parts, dim=0)
    render_deformed_normals = torch.cat(deformed_normal_parts, dim=0)

    cavity_rgb = torch.tensor(args.mouth_cavity_color, dtype=torch.float32) / 255.0
    lining_rgb = torch.tensor(
        args.mouth_boundary_lining_color,
        dtype=torch.float32,
    ) / 255.0
    render_colors = validation_colors_with_cavity(
        render_neutral_vertices,
        render_neutral_normals,
        config,
        cavity_vertex_count,
        cavity_rgb,
    )
    render_colors = apply_oral_vertex_colors(
        render_colors,
        oral_vertex_count,
        oral_vertex_colors,
        trailing_vertex_count=lining_vertex_count + cavity_vertex_count,
    )
    render_colors = apply_component_vertex_color(
        render_colors,
        lining_vertex_count,
        lining_rgb,
        trailing_vertex_count=cavity_vertex_count,
    )

    neutral_path = output_dir / f"{prefix}_canonical_lip_split_neutral.{args.output_format}"
    neutral_vertices = demo.output_vertices_for_convention(
        render_neutral_vertices,
        args.output_convention,
    )
    neutral_normals = demo.output_vertices_for_convention(
        render_neutral_normals,
        args.output_convention,
    )
    demo.export_mesh(
        vertices=neutral_vertices,
        faces=output_faces,
        normals=neutral_normals,
        path=neutral_path,
        vertex_colors=apply_component_vertex_color(
            apply_oral_vertex_colors(
                export_colors_with_cavity(
                    neutral_vertices,
                    cavity_vertex_count,
                    cavity_rgb,
                ),
                oral_vertex_count,
                oral_vertex_colors,
                trailing_vertex_count=lining_vertex_count + cavity_vertex_count,
            ),
            lining_vertex_count,
            lining_rgb,
            trailing_vertex_count=cavity_vertex_count,
        ),
    )
    outputs["canonical_lip_split_neutral_mesh"] = str(neutral_path)
    if split_landmarks is not None:
        canonical_landmark_path = output_dir / (
            f"{prefix}_canonical_lip_split_landmarks.json"
        )
        write_json(
            canonical_landmark_path,
            {
                str(int(mediapipe_id)): int(vertex_id)
                for mediapipe_id, vertex_id in sorted(split_landmarks.items())
            },
        )
        outputs["canonical_lip_split_landmarks"] = str(
            canonical_landmark_path
        )

    image = demo.render_output_image(
        neutral_vertices=render_neutral_vertices,
        deformed_vertices=render_deformed_vertices,
        normals=render_neutral_normals,
        faces=output_faces,
        action_unit_id=JAW_OPEN_AU_ID,
        config=config,
        device=device,
        vertex_colors=render_colors,
    )
    image_path = output_dir / f"{prefix}_jawOpen_lip_split.jpg"
    image.save(image_path, quality=95)
    outputs["jaw_open_image"] = str(image_path)

    jaw_open_path = output_dir / f"{prefix}_jawOpen_lip_split.{args.output_format}"
    export_vertices = demo.output_vertices_for_convention(
        render_deformed_vertices,
        args.output_convention,
    )
    export_normals = demo.output_vertices_for_convention(
        render_deformed_normals,
        args.output_convention,
    )
    demo.export_mesh(
        vertices=export_vertices,
        faces=output_faces,
        normals=export_normals,
        path=jaw_open_path,
        vertex_colors=apply_component_vertex_color(
            apply_oral_vertex_colors(
                export_colors_with_cavity(
                    export_vertices,
                    cavity_vertex_count,
                    cavity_rgb,
                ),
                oral_vertex_count,
                oral_vertex_colors,
                trailing_vertex_count=lining_vertex_count + cavity_vertex_count,
            ),
            lining_vertex_count,
            lining_rgb,
            trailing_vertex_count=cavity_vertex_count,
        ),
    )
    outputs["jaw_open_mesh"] = str(jaw_open_path)

    report_path = output_dir / f"{prefix}_lip_split_jaw_open_report.json"
    outputs["report"] = str(report_path)
    report = {
        "mesh": str(mesh_path),
        "checkpoint": str(checkpoint_path),
        "split_mode": args.split_mode,
        "coincident_vertex_weld": (
            weld_report.to_dict() if weld_report is not None else None
        ),
        "lip_landmark_topology_repair": (
            lip_landmark_topology_repair_report.to_dict()
            if lip_landmark_topology_repair_report is not None
            else None
        ),
        "action_unit_id": JAW_OPEN_AU_ID,
        "action_unit_name": AU_NAME[JAW_OPEN_AU_ID],
        "lip_regions": lip_regions.to_dict() if lip_regions is not None else None,
        "split_lip_regions": (
            split_lip_regions.to_dict() if split_lip_regions is not None else None
        ),
        "split": split_report.to_dict() if split_report is not None else None,
        "inference_proxy": proxy_report,
        "probe_aperture_cut": (
            aperture_pass_reports[0] if aperture_pass_reports else None
        ),
        "probe_aperture_passes": aperture_pass_reports,
        "probe_aperture_total_removed_faces": aperture_total_removed_faces,
        "probe_stretch_passes": stretch_pass_reports,
        "probe_stretch_total_detected_faces": stretch_total_detected_faces,
        "probe_total_removed_faces": probe_total_removed_faces,
        "probe_face_set_converged": probe_face_set_converged,
        "probe_stretch_remaining": (
            stretch_remaining_report.to_dict()
            if stretch_remaining_report is not None
            else None
        ),
        "probe_aperture_remaining": (
            aperture_remaining_report.to_dict()
            if aperture_remaining_report is not None
            else None
        ),
        "probe_aperture_converged": (
            probe_face_set_converged
            and aperture_remaining_report is not None
            and aperture_remaining_report.removed_face_count == 0
            and stretch_remaining_report is not None
            and stretch_remaining_report.removed_face_count == 0
        ),
        "probe_converged": (
            probe_face_set_converged
            and aperture_remaining_report is not None
            and aperture_remaining_report.removed_face_count == 0
            and stretch_remaining_report is not None
            and stretch_remaining_report.removed_face_count == 0
        ),
        "mouth_cavity": (
            {
                "neutral": neutral_cavity_report.to_dict(),
                "jaw_open": mouth_cavity_report.to_dict(),
                "color_rgb": [int(value) for value in args.mouth_cavity_color],
            }
            if mouth_cavity_report is not None
            else None
        ),
        "mouth_boundary_lining": (
            {
                "neutral": neutral_lining_report.to_dict(),
                "jaw_open": mouth_boundary_lining_report.to_dict(),
                "color_rgb": [
                    int(value) for value in args.mouth_boundary_lining_color
                ],
            }
            if mouth_boundary_lining_report is not None
            else None
        ),
        "mouth_boundary_smoothing": (
            mouth_boundary_smooth_report.to_dict()
            if mouth_boundary_smooth_report is not None
            else None
        ),
        "mouth_contour_clip": (
            mouth_contour_clip_report.to_dict()
            if mouth_contour_clip_report is not None
            else None
        ),
        "lip_annulus_repair": (
            lip_annulus_repair_report.to_dict()
            if lip_annulus_repair_report is not None
            else None
        ),
        "lip_annulus_head_cleanup": (
            lip_annulus_head_cleanup_report.to_dict()
            if lip_annulus_head_cleanup_report is not None
            else None
        ),
        "front_surface_orientation": (
            front_surface_orientation_report.to_dict()
            if front_surface_orientation_report is not None
            else None
        ),
        "post_bridge_mouth_boundary_smoothing": (
            post_bridge_smooth_report.to_dict()
            if post_bridge_smooth_report is not None
            else None
        ),
        "metahuman_oral_assembly": (
            oral_assembly_report.to_dict()
            if oral_assembly_report is not None
            else None
        ),
        "output_components": {
            "head_vertex_count": int(processed_mesh["vertices"].shape[0]),
            "lip_annulus_vertex_count": lip_annulus_vertex_count,
            "oral_vertex_count": oral_vertex_count,
            "lining_vertex_count": lining_vertex_count,
            "cavity_vertex_count": cavity_vertex_count,
            "total_vertex_count": int(render_neutral_vertices.shape[0]),
            "total_face_count": int(output_faces.shape[0]),
        },
        "oral_head_aperture_cleanup": (
            oral_head_aperture_report.to_dict()
            if oral_head_aperture_report is not None
            else None
        ),
        "oral_head_component_cleanup": (
            oral_head_component_report.to_dict()
            if oral_head_component_report is not None
            else None
        ),
        "mouth_hole_fill": (
            {
                "pass_count": len(mouth_hole_fill_reports),
                "initial_candidate_hole_cycle_count": (
                    mouth_hole_fill_reports[0].candidate_hole_cycle_count
                ),
                "total_filled_hole_cycle_count": sum(
                    item.filled_hole_cycle_count for item in mouth_hole_fill_reports
                ),
                "remaining_hole_cycle_count": (
                    mouth_hole_fill_reports[-1].candidate_hole_cycle_count
                    - mouth_hole_fill_reports[-1].filled_hole_cycle_count
                ),
                "total_added_face_count": sum(
                    item.added_face_count for item in mouth_hole_fill_reports
                ),
                "reoriented_face_count": sum(
                    item.reoriented_face_count for item in mouth_hole_fill_reports
                ),
                "boundary_edge_count_before": (
                    mouth_hole_fill_reports[0].boundary_edge_count_before
                ),
                "boundary_edge_count_after": (
                    mouth_hole_fill_reports[-1].boundary_edge_count_after
                ),
            }
            if mouth_hole_fill_reports
            else None
        ),
        "mouth_hole_fill_passes": [
            item.to_dict() for item in mouth_hole_fill_reports
        ],
        "mouth_corner_bridge": (
            mouth_corner_bridge_report.to_dict()
            if mouth_corner_bridge_report is not None
            else None
        ),
        "probe_predicted_delta_mean_norm": float(
            probe_predicted_delta.norm(dim=-1).mean().item()
        ),
        "probe_predicted_delta_max_norm": float(
            probe_predicted_delta.norm(dim=-1).max().item()
        ),
        "predicted_delta_mean_norm": float(predicted_delta.norm(dim=-1).mean().item()),
        "predicted_delta_max_norm": float(predicted_delta.norm(dim=-1).max().item()),
        "elapsed_seconds": time.monotonic() - pipeline_started,
        "outputs": outputs,
    }
    write_json(report_path, report)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Loaded mesh:       {mesh_path}")
    if weld_report is not None:
        print(
            "Coincident weld:   "
            f"merged_vertices={weld_report.merged_vertex_count} "
            f"removed_degenerate_faces={weld_report.removed_degenerate_face_count}"
        )
    if lip_landmark_topology_repair_report is not None:
        print(
            "Lip landmarks:     "
            f"invalid={lip_landmark_topology_repair_report.invalid_lip_landmark_count} "
            f"detached={lip_landmark_topology_repair_report.off_component_lip_landmark_count} "
            f"outliers={lip_landmark_topology_repair_report.spatial_outlier_lip_landmark_count} "
            f"snapped={lip_landmark_topology_repair_report.snapped_lip_landmark_count} "
            f"skipped={lip_landmark_topology_repair_report.skipped_lip_landmark_count}"
        )
    if args.split_mode == "mediapipe-seam":
        print(
            "Lip seam split:    "
            f"candidate_faces={split_report.candidate_face_count} "
            f"seam_vertices={split_report.seam_vertex_count} "
            f"duplicated_vertices={split_report.duplicated_vertex_count} "
            f"mixed_faces={split_report.mixed_original_duplicate_face_count}"
        )
        if aperture_pass_reports:
            print(
                "Probe aperture cut: "
                f"passes={len(aperture_pass_reports)} "
                f"removed_faces={aperture_total_removed_faces} "
                "remaining_front_cover_faces="
                f"{aperture_remaining_report.removed_face_count} "
                f"final_faces={processed_mesh['faces'].shape[0]}"
            )
            print(
                "Probe stretch cut: "
                f"detected_faces={stretch_total_detected_faces} "
                "remaining_severe_faces="
                f"{stretch_remaining_report.removed_face_count}"
            )
        if mouth_cavity_report is not None:
            print(
                "Mouth cavity:      "
                f"vertices={mouth_cavity_report.vertex_count} "
                f"faces={mouth_cavity_report.face_count} "
                f"depth={mouth_cavity_report.center[args.seam_depth_axis]:.6f}"
            )
        if mouth_boundary_lining_report is not None:
            print(
                "Mouth edge lining: "
                f"cycles={mouth_boundary_lining_report.boundary_cycle_count} "
                f"vertices={mouth_boundary_lining_report.vertex_count} "
                f"faces={mouth_boundary_lining_report.face_count}"
            )
        if mouth_boundary_smooth_report is not None:
            print(
                "Boundary smoothing:"
                f" components={mouth_boundary_smooth_report.smoothed_component_count} "
                f"vertices={mouth_boundary_smooth_report.smoothed_boundary_vertex_count} "
                f"max_move={mouth_boundary_smooth_report.max_boundary_displacement:.6f}"
            )
        if mouth_hole_fill_reports:
            print(
                "Mouth hole fill:   "
                f"cycles={sum(item.filled_hole_cycle_count for item in mouth_hole_fill_reports)}/"
                f"{mouth_hole_fill_reports[0].candidate_hole_cycle_count} "
                f"added_faces={sum(item.added_face_count for item in mouth_hole_fill_reports)} "
                "remaining_holes="
                f"{mouth_hole_fill_reports[-1].candidate_hole_cycle_count - mouth_hole_fill_reports[-1].filled_hole_cycle_count} "
                "remaining_boundary_edges="
                f"{mouth_hole_fill_reports[-1].boundary_edge_count_after}"
            )
        if mouth_corner_bridge_report is not None:
            print(
                "Mouth corner repair: "
                f"sides={mouth_corner_bridge_report.repaired_sides} "
                f"added_faces={mouth_corner_bridge_report.added_face_count} "
                "remaining_boundary_edges="
                f"{mouth_corner_bridge_report.boundary_edge_count_after} "
                f"severe_faces={mouth_corner_bridge_report.severe_added_face_count} "
                f"flipped_faces={mouth_corner_bridge_report.flipped_added_face_count}"
            )
        if post_bridge_smooth_report is not None:
            print(
                "Post-bridge smooth:"
                f" vertices={post_bridge_smooth_report.smoothed_boundary_vertex_count} "
                f"max_move={post_bridge_smooth_report.max_boundary_displacement:.6f} "
                f"rejected_faces={post_bridge_smooth_report.quality_rejected_face_count}"
            )
        if oral_assembly_report is not None:
            print(
                "MetaHuman mouth:   "
                f"vertices={oral_assembly_report.vertex_count} "
                f"faces={oral_assembly_report.face_count} "
                f"teeth_vertices={oral_assembly_report.tooth_vertex_count} "
                f"mode={oral_assembly_report.geometry_mode}"
            )
    elif args.split_mode == "region":
        print(
            "Lip region split:  "
            f"removed_faces={split_report.removed_face_count} "
            f"duplicated_vertices={split_report.duplicated_vertex_count} "
            f"remaining_direct_faces={split_report.direct_lip_connection_faces_after}"
        )
    else:
        print("Lip seam split:    disabled (weld-only baseline)")
    print(
        "JawOpen mean/max:  "
        f"{report['predicted_delta_mean_norm']:.9f} / "
        f"{report['predicted_delta_max_norm']:.9f}"
    )
    print(f"Wrote report:      {report_path}")
    for label, path in outputs.items():
        if label != "report":
            print(f"Wrote {label}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split connected upper/lower lip topology, then run TopoRig jawOpen "
            "inference for visual inspection."
        )
    )
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="TopoRig checkpoint. Defaults to the newest runs/**/checkpoints/*.pt.",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--proxy-target-faces",
        type=positive_int,
        default=None,
        help=(
            "Run TopoRig and iterative probe cleanup on a compiled decimation "
            "proxy, then transfer displacement back to the dense mesh."
        ),
    )
    parser.add_argument(
        "--proxy-aggression",
        type=float,
        default=7.0,
        help="Compiled proxy decimation aggression from 0 (careful) to 10 (fast).",
    )
    parser.add_argument(
        "--proxy-transfer-smoothing-iterations",
        type=int,
        default=6,
        help="Dense topology smoothing iterations after proxy displacement transfer.",
    )
    parser.add_argument(
        "--proxy-transfer-smoothing-blend",
        type=float,
        default=0.5,
        help="Neighbor-average blend per proxy transfer smoothing iteration.",
    )
    parser.add_argument(
        "--proxy-output-resolution",
        choices=("dense", "proxy"),
        default="dense",
        help=(
            "Transfer deformation back to the dense mesh or export the proxy "
            "directly. Proxy output is substantially faster."
        ),
    )
    parser.add_argument(
        "--proxy-one-shot-probe",
        action="store_true",
        help=(
            "Remove faces detected by the first jawOpen prediction without "
            "rerunning the model. This applies with or without decimation."
        ),
    )
    parser.add_argument(
        "--proxy-before-split",
        action="store_true",
        help=(
            "Build the fast inference proxy before constructing the lip seam so "
            "decimation cannot collapse the newly separated upper/lower boundary."
        ),
    )
    parser.add_argument(
        "--proxy-before-split-maximum-source-ratio",
        type=float,
        default=3.5,
        help=(
            "Maximum source/target face ratio for proxy-before-split. More "
            "aggressive reductions preserve the dense seam before proxying. "
            "Default: 3.5."
        ),
    )
    parser.add_argument("--output-format", choices=("glb", "obj"), default="glb")
    parser.add_argument(
        "--split-mode",
        choices=("mediapipe-seam", "weld-only", "region"),
        default="mediapipe-seam",
        help=(
            "mediapipe-seam welds exact duplicates and separates the closed-mouth "
            "seam; weld-only diagnoses duplicate topology; region retains the "
            "older Sapiens/vertex-region behavior."
        ),
    )
    parser.add_argument(
        "--output-convention",
        choices=("metahuman", "model"),
        default="metahuman",
    )
    parser.add_argument("--seam-horizontal-axis", type=int, default=0)
    parser.add_argument("--seam-depth-axis", type=int, default=1)
    parser.add_argument("--seam-vertical-axis", type=int, default=2)
    parser.add_argument(
        "--seam-depth-tolerance",
        type=float,
        default=0.02,
        help="Maximum depth from the MediaPipe inner-lip centerline to split.",
    )
    parser.add_argument(
        "--seam-vertical-tolerance",
        type=float,
        default=0.012,
        help="Vertical band around the inner-lip centerline used to find the seam.",
    )
    parser.add_argument(
        "--seam-horizontal-padding",
        type=float,
        default=0.0,
        help="Optional horizontal extension beyond the two inner mouth corners.",
    )
    parser.add_argument(
        "--seam-corner-inset-ratio",
        type=float,
        default=0.05,
        help=(
            "Fraction of mouth width left connected at each seam endpoint. "
            "Default: 0.05."
        ),
    )
    parser.add_argument(
        "--seam-preopen-distance",
        type=float,
        default=0.002,
        help=(
            "Total neutral gap inserted between duplicated upper/lower seam "
            "vertices before inference. Default: 0.002."
        ),
    )
    parser.add_argument(
        "--no-weld-coincident-vertices",
        action="store_true",
        help=(
            "Do not merge exact duplicate vertices before the MediaPipe seam split."
        ),
    )
    parser.add_argument(
        "--no-repair-mediapipe-lip-landmarks",
        action="store_true",
        help=(
            "Do not repair invalid or detached lip landmarks onto the dominant "
            "connected mouth surface before seam construction."
        ),
    )
    parser.add_argument(
        "--lip-landmark-maximum-snap-distance-ratio",
        type=float,
        default=0.25,
        help=(
            "Maximum repair distance relative to the mapped mouth span. "
            "Default: 0.25."
        ),
    )
    parser.add_argument(
        "--reconstruct-lip-annulus",
        action="store_true",
        help=(
            "Add a continuous skin strip derived from the repaired outer "
            "MediaPipe lip contour after mouth cleanup."
        ),
    )
    parser.add_argument(
        "--lip-annulus-segments",
        type=positive_int,
        default=64,
        help="Even segment count around the reconstructed lip rim. Default: 64.",
    )
    parser.add_argument(
        "--lip-annulus-radial-rings",
        type=positive_int,
        default=3,
        help="Radial subdivisions across the reconstructed lip rim. Default: 3.",
    )
    parser.add_argument(
        "--lip-annulus-outer-scale",
        type=float,
        default=1.03,
        help="Scale applied to the outer lip repair curve. Default: 1.03.",
    )
    parser.add_argument(
        "--lip-annulus-thickness-ratio",
        type=float,
        default=0.055,
        help="Uniform lip-rim width relative to mouth span. Default: 0.055.",
    )
    parser.add_argument(
        "--lip-annulus-maximum-radial-inset-fraction",
        type=float,
        default=0.45,
        help=(
            "Maximum fraction of the local mouth radius used by the lip rim. "
            "Default: 0.45."
        ),
    )
    parser.add_argument(
        "--lip-annulus-side-thickness-factor",
        type=float,
        default=0.45,
        help="Rim thickness at mouth corners relative to upper/lower lips. Default: 0.45.",
    )
    parser.add_argument(
        "--lip-annulus-inner-recess-depth",
        type=float,
        default=0.003,
        help="Depth that recesses the inner rim behind the outer lip. Default: 0.003.",
    )
    parser.add_argument(
        "--lip-annulus-front-offset",
        type=float,
        default=0.002,
        help=(
            "Small frontward offset that prevents overlap with damaged source "
            "lip triangles. Default: 0.002."
        ),
    )
    parser.add_argument(
        "--lip-annulus-bulge-depth",
        type=float,
        default=0.0025,
        help="Frontward curvature across the repaired lip strip. Default: 0.0025.",
    )
    parser.add_argument(
        "--lip-annulus-upper-smoothing-window",
        type=positive_int,
        default=9,
        help="Odd smoothing window for the upper repair curve. Default: 9.",
    )
    parser.add_argument(
        "--lip-annulus-lower-smoothing-window",
        type=positive_int,
        default=17,
        help="Odd smoothing window for the lower repair curve. Default: 17.",
    )
    parser.add_argument(
        "--clear-head-under-lip-annulus",
        action="store_true",
        help="Remove source-head triangles replaced by the generated lip rim.",
    )
    parser.add_argument(
        "--lip-annulus-cleanup-depth-padding-ratio",
        type=float,
        default=0.30,
        help="Depth padding for source faces cleared under the lip rim. Default: 0.30.",
    )
    parser.add_argument(
        "--orient-front-surface-faces",
        action="store_true",
        help=(
            "Correct reversed triangle winding on the external facial shell "
            "before appending the oral assembly."
        ),
    )
    parser.add_argument(
        "--front-surface-depth-ratio",
        type=float,
        default=0.18,
        help="Depth of the external facial orientation band. Default: 0.18.",
    )
    parser.add_argument(
        "--front-surface-normal-tolerance",
        type=float,
        default=0.01,
        help="Minimum reversed normal component required for a flip. Default: 0.01.",
    )
    parser.add_argument(
        "--no-probe-aperture-cut",
        action="store_true",
        help="Disable the probe jawOpen pass that removes faces covering the aperture.",
    )
    parser.add_argument(
        "--probe-aperture-scale",
        type=float,
        default=1.0,
        help="Scale of the opened inner-lip polygon used by the probe cut.",
    )
    parser.add_argument(
        "--probe-aperture-depth-padding",
        type=float,
        default=0.01,
        help="Depth padding around the deformed inner lips for the probe cut.",
    )
    parser.add_argument(
        "--probe-aperture-overlap-inset-scale",
        type=float,
        default=0.9,
        help=(
            "Inset scale used to catch triangles crossing the aperture when their "
            "centroids remain outside it. Default: 0.9."
        ),
    )
    parser.add_argument(
        "--probe-aperture-max-passes",
        type=positive_int,
        default=10,
        help="Maximum number of probe/cut passes before final verification.",
    )
    parser.add_argument(
        "--probe-restore-safe-passes",
        type=positive_int,
        default=2,
        help=(
            "Consecutive safe predictions required before restoring a removed "
            "source face. Default: 2."
        ),
    )
    parser.add_argument("--probe-stretch-edge-ratio", type=float, default=3.0)
    parser.add_argument("--probe-stretch-edge-growth", type=float, default=0.02)
    parser.add_argument("--probe-stretch-region-scale", type=float, default=1.5)
    parser.add_argument("--probe-stretch-depth-padding", type=float, default=0.08)
    parser.add_argument(
        "--smooth-contour-clip",
        action="store_true",
        help=(
            "Reconstruct cut boundary triangles against a smooth MediaPipe "
            "inner-lip contour. Disabled by default."
        ),
    )
    parser.add_argument(
        "--smooth-contour-scale",
        type=float,
        default=1.0,
        help="Scale of the smooth inner-lip contour. Default: 1.0.",
    )
    parser.add_argument(
        "--smooth-contour-depth-padding",
        type=float,
        default=0.08,
        help="Depth band in which removed boundary faces may be reconstructed.",
    )
    parser.add_argument(
        "--no-fill-secondary-mouth-holes",
        action="store_true",
        help="Do not cap secondary mouth-area boundary cycles after probe cleanup.",
    )
    parser.add_argument(
        "--mouth-hole-max-area-ratio",
        type=float,
        default=0.15,
        help="Largest secondary cycle area to cap relative to the main aperture.",
    )
    parser.add_argument(
        "--no-smooth-mouth-boundary",
        action="store_true",
        help="Disable bounded smoothing of the final mouth cut loops.",
    )
    parser.add_argument(
        "--mouth-boundary-max-displacement",
        type=float,
        default=0.006,
        help="Maximum projected movement of a mouth boundary vertex.",
    )
    parser.add_argument(
        "--mouth-boundary-blend-rings",
        type=positive_int,
        default=5,
        help="Adjacent face rings used to blend the boundary correction.",
    )
    parser.add_argument(
        "--post-bridge-smoothing",
        action="store_true",
        help="Apply a second conservative boundary pass after mouth-corner repair.",
    )
    parser.add_argument(
        "--post-bridge-max-displacement",
        type=float,
        default=0.0025,
        help="Maximum movement in the post-bridge smoothing pass.",
    )
    parser.add_argument(
        "--post-bridge-blend-rings",
        type=positive_int,
        default=3,
        help="Adjacent rings blended by the post-bridge smoothing pass.",
    )
    parser.add_argument(
        "--post-bridge-smoothing-cycles",
        type=positive_int,
        default=4,
        help="Taubin cycles used by the conservative post-bridge pass.",
    )
    parser.add_argument(
        "--no-repair-mouth-corners",
        action="store_true",
        help="Disable localized bridging of a laterally overextended mouth corner.",
    )
    parser.add_argument(
        "--mouth-corner-inset-ratio",
        type=float,
        default=0.13,
        help=(
            "Fraction of lip width included in a localized corner bridge. "
            "Default: 0.13."
        ),
    )
    parser.add_argument(
        "--mouth-corner-minimum-overshoot-ratio",
        type=float,
        default=0.12,
        help=(
            "Minimum boundary overshoot relative to lip width before repair. "
            "Default: 0.12."
        ),
    )
    parser.add_argument(
        "--mouth-corner-maximum-repaired-sides",
        type=positive_int,
        choices=(1, 2),
        default=1,
        help="Maximum number of overshooting mouth corners to bridge. Default: 1.",
    )
    parser.add_argument(
        "--mouth-corner-chain-padding-vertices",
        type=positive_int,
        default=1,
        help=(
            "Boundary vertices included beyond each detected corner chain end. "
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--mouth-cavity",
        action="store_true",
        help=(
            "Add the inner-mouth occluder. This is enabled automatically when "
            "a dental-only oral asset is fitted."
        ),
    )
    parser.add_argument(
        "--mouth-boundary-lining",
        action="store_true",
        help=(
            "Extrude the actual mouth cut loop into an oral compartment. This "
            "is enabled automatically when a dental-only oral asset is fitted."
        ),
    )
    parser.add_argument(
        "--no-mouth-boundary-lining",
        action="store_true",
        help="Disable the automatic cut-boundary oral compartment.",
    )
    parser.add_argument(
        "--mouth-boundary-lining-depth",
        type=float,
        default=0.24,
        help="Depth of the oral compartment behind the cut edge. Default: 0.24.",
    )
    parser.add_argument(
        "--mouth-boundary-lining-rim-depth",
        type=float,
        default=0.002,
        help="Small recession applied at the visible lining rim. Default: 0.002.",
    )
    parser.add_argument(
        "--mouth-boundary-lining-side-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of lip width retained at each side of the boundary lining. "
            "One keeps the complete loops."
        ),
    )
    parser.add_argument(
        "--mouth-boundary-lining-inner-scale",
        type=float,
        default=0.95,
        help=(
            "Scale the rear compartment ring toward the mouth center. Default: 0.95."
        ),
    )
    parser.add_argument(
        "--mouth-boundary-lining-radial-rings",
        type=positive_int,
        default=6,
        help="Depth sections used by the oral compartment. Default: 6.",
    )
    parser.add_argument(
        "--mouth-boundary-lining-segments",
        type=positive_int,
        default=96,
        help="Ordered superellipse segments around the full oral opening. Default: 96.",
    )
    parser.add_argument(
        "--mouth-boundary-lining-opening-scale",
        type=float,
        default=1.08,
        help="Scale of the compartment rim relative to detected opening bounds.",
    )
    parser.add_argument(
        "--mouth-boundary-lining-shape-exponent",
        type=float,
        default=2.8,
        help="Superellipse exponent of the oral compartment rim. Default: 2.8.",
    )
    parser.add_argument(
        "--mouth-boundary-lining-smoothing-iterations",
        type=int,
        default=12,
        help="Cyclic smoothing passes for recessed compartment rings. Default: 12.",
    )
    parser.add_argument(
        "--mouth-boundary-lining-maximum-cycles",
        type=positive_int,
        default=1,
        help="Maximum dominant mouth openings enclosed. Default: 1.",
    )
    parser.add_argument(
        "--no-mouth-boundary-lining-cap",
        action="store_true",
        help="Leave the rear of the mouth boundary lining open.",
    )
    parser.add_argument(
        "--mouth-boundary-lining-color",
        type=rgb_channel,
        nargs=3,
        default=(87, 19, 22),
        metavar=("R", "G", "B"),
        help="Mouth boundary lining RGB color. Default: 87 19 22.",
    )
    parser.add_argument(
        "--oral-asset",
        "--metahuman-oral-asset",
        dest="metahuman_oral_asset",
        type=Path,
        help="Fitted teeth, tongue, gums, palate, and oral-cavity NPZ.",
    )
    parser.add_argument(
        "--oral-assembly-scale",
        type=float,
        default=1.0,
        help="Scale the fitted oral assembly around the mouth center.",
    )
    parser.add_argument(
        "--no-oral-head-aperture-cleanup",
        action="store_true",
        help="Keep source head surfaces projected inside the inserted oral aperture.",
    )
    parser.add_argument(
        "--oral-head-aperture-scale",
        type=float,
        default=0.90,
        help="Inset outer-lip scale used to clear source mouth interiors. Default: 0.90.",
    )
    parser.add_argument(
        "--oral-head-depth-padding-ratio",
        type=float,
        default=0.15,
        help="Front-surface cleanup depth padding relative to mouth span. Default: 0.15.",
    )
    parser.add_argument(
        "--oral-head-component-margin-ratio",
        type=float,
        default=0.25,
        help="Extra mouth-bounds margin for disconnected source interiors.",
    )
    parser.add_argument(
        "--oral-head-boundary-distance-ratio",
        type=float,
        default=0.125,
        help="Distance from the outer contour protected as visible lip boundary.",
    )
    parser.add_argument(
        "--oral-head-boundary-blend-rings",
        type=int,
        default=2,
        help="Adjacent locally sized lip rings protected from cleanup. Default: 2.",
    )
    parser.add_argument(
        "--oral-head-maximum-protected-edge-ratio",
        type=float,
        default=0.25,
        help="Largest protected adjacent edge relative to mouth span. Default: 0.25.",
    )
    parser.add_argument(
        "--oral-head-component-maximum-faces",
        type=int,
        default=5000,
        help="Largest disconnected source-mouth component to remove. Default: 5000.",
    )
    parser.add_argument(
        "--no-oral-head-lower-lip-protection",
        action="store_true",
        help="Disable preservation of locally sized faces in the lower-lip landmark ribbon.",
    )
    parser.add_argument(
        "--oral-head-lower-lip-depth-tolerance-ratio",
        type=float,
        default=0.12,
        help="Maximum lower-lip face depth deviation relative to mouth span.",
    )
    parser.add_argument(
        "--oral-depth-offset",
        type=float,
        default=0.012,
        help=(
            "Move the oral assembly behind the lips in model units to prevent "
            "gum intersections. Default: 0.012."
        ),
    )
    parser.add_argument(
        "--no-oral-level-target-frame",
        action="store_true",
        help="Allow mouth-corner slope to rotate the fitted oral assembly.",
    )
    parser.add_argument(
        "--oral-auto-center-teeth",
        action="store_true",
        help=(
            "Center visible tooth bounds horizontally after landmark fitting."
        ),
    )
    parser.add_argument(
        "--oral-center-teeth-vertically",
        action="store_true",
        help=(
            "Also center tooth bounds vertically. Disabled by default because the "
            "source upper/lower jaw offset should normally be preserved."
        ),
    )
    parser.add_argument(
        "--oral-maximum-center-offset-ratio",
        type=float,
        default=0.25,
        help=(
            "Maximum tooth-centering correction as a fraction of target mouth "
            "span. Default: 0.25."
        ),
    )
    parser.add_argument(
        "--oral-tooth-depth-offset",
        type=float,
        default=0.0,
        help=(
            "Additional model-depth offset applied to tooth crowns with a local "
            "falloff. Negative values move teeth forward for the default axes."
        ),
    )
    parser.add_argument(
        "--oral-tooth-depth-blend-rings",
        type=int,
        default=2,
        help="Mesh rings used to taper tooth-only depth movement. Default: 2.",
    )
    parser.add_argument(
        "--oral-auto-center-lower-teeth",
        action="store_true",
        help="Center only the moving lower dental region horizontally.",
    )
    parser.add_argument(
        "--oral-lower-teeth-depth-offset",
        type=float,
        default=0.0,
        help="Additional depth offset for lower crowns and adjacent moving gum.",
    )
    parser.add_argument(
        "--oral-lower-teeth-blend-rings",
        type=int,
        default=2,
        help="Mesh rings used to taper lower dental movement. Default: 2.",
    )
    parser.add_argument(
        "--no-oral-containment",
        action="store_true",
        help="Disable the default deformation that keeps oral geometry inside the lips.",
    )
    parser.add_argument(
        "--oral-containment-scale",
        type=float,
        default=0.96,
        help="Scale of the outer-lip polygon used for oral containment. Default: 0.96.",
    )
    parser.add_argument(
        "--oral-containment-depth-inset",
        type=float,
        default=0.025,
        help="Depth recession applied to oral vertices outside the lip contour.",
    )
    parser.add_argument(
        "--oral-containment-blend-rings",
        type=int,
        default=2,
        help="Adjacent gum rings blended into oral containment. Default: 2.",
    )
    parser.add_argument(
        "--oral-rim-recession-ratio",
        type=float,
        default=0.08,
        help=(
            "Minimum gum recession behind the nearest inner-lip segment as a "
            "fraction of mouth span. Teeth and tongue are unchanged. Default: 0.08."
        ),
    )
    parser.add_argument(
        "--oral-recessed-shell-horizontal-scale",
        type=float,
        default=1.0,
        help=(
            "Horizontal scale for recessed oral-wall tissue only. Teeth, tongue, "
            "and front gum remain fixed. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--oral-recessed-shell-start-depth-ratio",
        type=float,
        default=0.06,
        help="Recessed depth where oral-wall widening begins. Default: 0.06.",
    )
    parser.add_argument(
        "--oral-recessed-shell-blend-depth-ratio",
        type=float,
        default=0.28,
        help="Depth interval used to blend in oral-wall widening. Default: 0.28.",
    )
    parser.add_argument(
        "--no-oral-rim-transition-trim",
        action="store_true",
        help="Keep exposed gum/root transition triangles outside the inner lips.",
    )
    parser.add_argument(
        "--oral-rim-transition-distance-ratio",
        type=float,
        default=0.08,
        help="Maximum transition-face distance from the inner-lip rim. Default: 0.08.",
    )
    parser.add_argument(
        "--oral-rim-transition-depth-tolerance-ratio",
        type=float,
        default=0.01,
        help="Local lip-depth tolerance for oral rim trimming. Default: 0.01.",
    )
    parser.add_argument(
        "--oral-upper-crown-quantile",
        type=float,
        default=0.50,
        help=(
            "Fraction of the static upper tooth column colored and moved as "
            "visible crown. Lower values hide more root. Default: 0.50."
        ),
    )
    parser.add_argument(
        "--oral-upper-teeth-vertical-offset",
        type=float,
        default=0.0,
        help=(
            "Vertical offset for static upper crowns only. Negative values move "
            "them downward for the default axes."
        ),
    )
    parser.add_argument(
        "--oral-upper-teeth-vertical-blend-rings",
        type=int,
        default=2,
        help="Mesh rings used to taper upper-crown vertical movement. Default: 2.",
    )
    oral_geometry_group = parser.add_mutually_exclusive_group()
    oral_geometry_group.add_argument(
        "--oral-full-shell",
        "--no-oral-wall-uv-removal",
        dest="oral_full_shell",
        action="store_true",
        help=(
            "Retain the complete MetaHuman palate, floor, side walls, gums, "
            "teeth, and tongue. This is the default."
        ),
    )
    oral_geometry_group.add_argument(
        "--oral-dental-only",
        action="store_true",
        help=(
            "Keep only the MetaHuman dental arches and tongue, then use the "
            "generated oral compartment fallback."
        ),
    )
    parser.add_argument(
        "--oral-remove-tongue-uv-island",
        action="store_true",
        help=(
            "Remove the broad MetaHuman tongue UV island while retaining both "
            "dental arches and oral-wall geometry."
        ),
    )
    parser.add_argument(
        "--oral-tongue-planar-scale",
        type=float,
        default=1.40,
        help="Scale tongue width and visible height around its center. Default: 1.40.",
    )
    parser.add_argument(
        "--oral-tongue-depth-offset",
        type=float,
        default=0.0,
        help="Additional model-depth offset for the UV-defined tongue island.",
    )
    parser.add_argument(
        "--oral-tongue-vertical-offset",
        type=float,
        default=0.0,
        help="Additional vertical offset for the UV-defined tongue island.",
    )
    parser.add_argument(
        "--oral-tongue-offset-blend-rings",
        type=int,
        default=2,
        help="Mesh rings used to taper tongue repositioning. Default: 2.",
    )
    parser.add_argument(
        "--oral-aperture-crop-scale",
        type=float,
        default=None,
        help=(
            "Optional outer-lip crop scale for debugging. Disabled by default "
            "because cropping can remove teeth and gum geometry."
        ),
    )
    parser.add_argument(
        "--no-oral-aperture-crop",
        action="store_true",
        help="Keep the full oral assembly without outer-lip cropping.",
    )
    parser.add_argument(
        "--no-mouth-cavity",
        action="store_true",
        help="Disable the inner-mouth cavity generated for the oral assembly.",
    )
    parser.add_argument(
        "--mouth-cavity-scale",
        type=float,
        default=1.05,
        help="Scale of the cavity disk relative to the opened inner lips.",
    )
    parser.add_argument(
        "--mouth-cavity-depth-offset",
        type=float,
        default=0.03,
        help="Distance behind the rearmost inner-lip point for the cavity rim.",
    )
    parser.add_argument(
        "--mouth-cavity-shape-exponent",
        type=float,
        default=2.4,
        help="Superellipse exponent for the elliptical cavity rim.",
    )
    parser.add_argument(
        "--mouth-cavity-segments",
        type=positive_int,
        default=64,
        help="Number of radial segments in the cavity disk.",
    )
    parser.add_argument(
        "--mouth-cavity-radial-rings",
        type=positive_int,
        default=8,
        help="Concentric rings used to form the curved cavity lining.",
    )
    parser.add_argument(
        "--mouth-cavity-curvature-depth",
        type=float,
        default=0.20,
        help="Additional recession at the center of the cavity lining.",
    )
    parser.add_argument(
        "--mouth-cavity-inner-radius-ratio",
        type=float,
        default=0.0,
        help=(
            "Leave the center open and generate only an annular side lining. "
            "Zero retains the filled cavity."
        ),
    )
    parser.add_argument(
        "--mouth-cavity-side-wall-min-horizontal-ratio",
        type=float,
        default=0.0,
        help=(
            "Keep only left/right annular segments whose normalized horizontal "
            "position reaches this ratio. Zero keeps every segment."
        ),
    )
    parser.add_argument(
        "--mouth-cavity-color",
        type=rgb_channel,
        nargs=3,
        default=(105, 32, 45),
        metavar=("R", "G", "B"),
        help="Mouth cavity RGB color. Default: 105 32 45.",
    )
    parser.add_argument(
        "--sapiens-lip-regions",
        "--lip-regions",
        dest="lip_regions",
        type=Path,
        help=(
            "Projected Sapiens lip regions. JSON/NPZ may contain "
            "upper_lip_vertex_ids and lower_lip_vertex_ids, boolean masks, "
            "or per-vertex labels."
        ),
    )
    parser.add_argument(
        "--upper-lip-vertices",
        help="Comma/space-separated vertex ids, or a text file containing them.",
    )
    parser.add_argument(
        "--lower-lip-vertices",
        help="Comma/space-separated vertex ids, or a text file containing them.",
    )
    parser.add_argument(
        "--upper-label-ids",
        type=int,
        nargs="*",
        default=(),
        help="Label ids to treat as upper lip when --lip-regions is a label array.",
    )
    parser.add_argument(
        "--lower-label-ids",
        type=int,
        nargs="*",
        default=(),
        help="Label ids to treat as lower lip when --lip-regions is a label array.",
    )
    parser.add_argument(
        "--lip-mask-rings",
        type=int,
        default=1,
        help="Fallback MediaPipe lip seed expansion in mesh edge rings.",
    )
    parser.add_argument(
        "--include-outer-lip-landmarks",
        action="store_true",
        help="Use outer MediaPipe lip landmarks in the fallback lip mask.",
    )
    parser.add_argument(
        "--use-mediapipe-lip-paths",
        action="store_true",
        help=(
            "Build fallback lip masks from mesh paths between MediaPipe lip "
            "landmarks. This is more reliable on closed-mouth meshes than "
            "isolated landmark seeds."
        ),
    )
    parser.add_argument(
        "--keep-mixed-faces",
        action="store_true",
        help="Duplicate shared seam vertices but do not remove upper/lower bridge faces.",
    )
    parser.add_argument(
        "--mouth-band-cut",
        action="store_true",
        help=(
            "Also remove faces whose centroids lie in the MediaPipe inner-mouth "
            "lip polygon. Useful for closed-mouth meshes with a broad welded "
            "patch between lips."
        ),
    )
    parser.add_argument(
        "--mouth-band-padding",
        type=float,
        default=0.0,
        help="Scale the mouth-band polygon outward by this fraction before cutting.",
    )
    parser.add_argument(
        "--mouth-band-axes",
        type=int,
        nargs=2,
        default=(0, 2),
        metavar=("AXIS_A", "AXIS_B"),
        help="Coordinate axes used for the 2D mouth-band polygon. Default: 0 2.",
    )
    parser.add_argument(
        "--mouth-band-depth-axis",
        type=int,
        default=1,
        help="Coordinate axis used to constrain the mouth-band depth. Default: 1.",
    )
    parser.add_argument(
        "--mouth-band-depth-padding",
        type=float,
        default=0.006,
        help=(
            "Padding around inner-lip landmarks on the depth axis. Lower values "
            "remove fewer off-surface face patches. Default: 0.006."
        ),
    )
    parser.add_argument("--no-raw-split-mesh", action="store_true")
    parser.add_argument(
        "--input-convention",
        choices=(
            "auto",
            "fbx",
            "custom-glb",
            "custom-glb-y-front",
            "custom-glb-neg-y-front",
            "custom-glb-y-up-z-front",
            "custom-glb-y-up-neg-z-front",
        ),
        default="auto",
    )
    parser.add_argument(
        "--reference-input-convention",
        choices=(
            "auto",
            "fbx",
            "custom-glb",
            "custom-glb-y-front",
            "custom-glb-neg-y-front",
            "custom-glb-y-up-z-front",
            "custom-glb-y-up-neg-z-front",
        ),
        default="auto",
    )
    parser.add_argument(
        "--alignment-mode",
        choices=("none", "similarity", "scale_translation"),
        default="similarity",
    )
    parser.add_argument(
        "--landmark-json",
        type=Path,
        help="Optional precomputed MediaPipe mesh landmark mapping for the input mesh.",
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
    parser.add_argument(
        "--reference-processed-cache",
        type=Path,
        default=ROOT / "runs" / "cache" / "metahuman_reference_preprocessed.pt",
        help="Cached preprocessed reference mesh used for landmark alignment.",
    )
    parser.add_argument(
        "--no-mediapipe-landmarks",
        action="store_true",
        help=(
            "Do not generate MediaPipe landmarks. Requires explicit lip regions "
            "and --alignment-mode none."
        ),
    )
    return parser.parse_args()


def resolve_input_landmarks(
    args: argparse.Namespace,
    mesh_path: Path,
) -> Optional[dict[int, int]]:
    if args.no_mediapipe_landmarks:
        return None
    if args.landmark_json is not None:
        return demo.load_landmark_mapping(args.landmark_json.expanduser())

    needs_fallback_lips = (
        args.split_mode == "mediapipe-seam"
        or (
            args.lip_regions is None
            and not (args.upper_lip_vertices and args.lower_lip_vertices)
        )
    )
    needs_alignment = args.alignment_mode != "none"
    if not needs_fallback_lips and not needs_alignment:
        return None

    if mesh_path.suffix.lower() == ".obj":
        raise ValueError(
            "MediaPipe fallback landmark generation supports FBX/GLB/GLTF in "
            "map_mediapipe_landmarks.py. For OBJ input, pass --lip-regions or "
            "--upper-lip-vertices/--lower-lip-vertices and use "
            "--alignment-mode none, or pass --landmark-json."
        )
    landmark_path = demo.ensure_mediapipe_landmarks(
        mesh_path=mesh_path,
        cache_dir=args.landmark_cache_dir.expanduser(),
        mapper_path=args.mediapipe_mapper.expanduser(),
        blender=args.blender,
        force=args.refresh_landmarks,
    )
    return demo.load_landmark_mapping(landmark_path)


def resolve_lip_regions(
    args: argparse.Namespace,
    raw_mesh: Mapping[str, torch.Tensor],
    input_landmarks: Optional[Mapping[int, int]],
) -> LipRegions:
    vertex_count = int(raw_mesh["vertices"].shape[0])
    explicit_upper = load_vertex_ids_arg(args.upper_lip_vertices, vertex_count)
    explicit_lower = load_vertex_ids_arg(args.lower_lip_vertices, vertex_count)
    if explicit_upper.numel() > 0 or explicit_lower.numel() > 0:
        if explicit_upper.numel() == 0 or explicit_lower.numel() == 0:
            raise ValueError(
                "Pass both --upper-lip-vertices and --lower-lip-vertices, or neither."
            )
        return LipRegions(explicit_upper, explicit_lower, source="cli")

    if args.use_mediapipe_lip_paths:
        if input_landmarks is None:
            raise ValueError(
                "--use-mediapipe-lip-paths requires MediaPipe landmarks. Pass "
                "--landmark-json or allow landmark generation."
            )
        return lip_regions_from_mediapipe_paths(
            raw_mesh["vertices"],
            raw_mesh["faces"],
            input_landmarks,
            rings=args.lip_mask_rings,
            include_outer_lip=args.include_outer_lip_landmarks,
        )

    if args.lip_regions is not None:
        return load_lip_regions(
            args.lip_regions,
            vertex_count,
            upper_label_ids=args.upper_label_ids,
            lower_label_ids=args.lower_label_ids,
        )

    if input_landmarks is None:
        raise ValueError(
            "No lip regions were supplied and MediaPipe landmarks are disabled."
        )
    return lip_regions_from_mediapipe_mapping(
        raw_mesh["vertices"],
        raw_mesh["faces"],
        input_landmarks,
        rings=args.lip_mask_rings,
        include_outer_lip=args.include_outer_lip_landmarks,
    )


def resolve_extra_removed_face_ids(
    args: argparse.Namespace,
    raw_mesh: Mapping[str, torch.Tensor],
    input_landmarks: Optional[Mapping[int, int]],
) -> torch.Tensor:
    if not args.mouth_band_cut:
        return torch.empty(0, dtype=torch.long)
    if input_landmarks is None:
        raise ValueError(
            "--mouth-band-cut requires MediaPipe landmarks. Pass --landmark-json "
            "or allow landmark generation."
        )
    return mouth_band_face_ids_from_mediapipe(
        raw_mesh["vertices"],
        raw_mesh["faces"],
        input_landmarks,
        axes=(int(args.mouth_band_axes[0]), int(args.mouth_band_axes[1])),
        padding=float(args.mouth_band_padding),
        depth_axis=int(args.mouth_band_depth_axis),
        depth_padding=float(args.mouth_band_depth_padding),
    )


def resolve_reference_landmarks(args: argparse.Namespace) -> Optional[dict[int, int]]:
    if args.alignment_mode == "none":
        return None
    reference_mesh_path = demo.metahuman_sample_mesh(args.metahuman_sample_dir.expanduser())
    reference_landmark_json = demo.ensure_mediapipe_landmarks(
        mesh_path=reference_mesh_path,
        cache_dir=args.landmark_cache_dir.expanduser(),
        mapper_path=args.mediapipe_mapper.expanduser(),
        blender=args.blender,
        force=args.refresh_landmarks,
        prefer_sidecar=True,
    )
    return demo.load_landmark_mapping(reference_landmark_json)


def build_inference_proxy(
    mesh: Mapping[str, torch.Tensor],
    landmarks: Mapping[int, int],
    *,
    target_faces: int,
    aggression: float,
) -> tuple[
    dict[str, torch.Tensor],
    dict[int, int],
    torch.Tensor,
    dict[str, Any],
]:
    """Decimate a canonical mesh and retain an exact dense-to-proxy map."""

    if not 0.0 <= aggression <= 10.0:
        raise ValueError("Proxy aggression must be in [0, 10].")
    try:
        import fast_simplification
        from fast_simplification.replay import replay_simplification
    except ImportError as exc:
        raise ImportError(
            "Proxy inference requires the fast-simplification package."
        ) from exc

    vertices = mesh["vertices"].detach().cpu().float().numpy()
    faces = mesh["faces"].detach().cpu().long().numpy()
    original_vertex_count = int(vertices.shape[0])
    referenced_vertex_ids = np.unique(faces.reshape(-1))
    unreferenced_vertex_count = original_vertex_count - int(
        referenced_vertex_ids.shape[0]
    )
    source_to_compact = np.full(original_vertex_count, -1, dtype=np.int64)
    source_to_compact[referenced_vertex_ids] = np.arange(
        referenced_vertex_ids.shape[0],
        dtype=np.int64,
    )
    if unreferenced_vertex_count:
        from scipy.spatial import cKDTree

        unreferenced_vertex_ids = np.flatnonzero(source_to_compact < 0)
        compact_tree = cKDTree(vertices[referenced_vertex_ids])
        _distances, nearest_compact = compact_tree.query(
            vertices[unreferenced_vertex_ids],
            k=1,
        )
        source_to_compact[unreferenced_vertex_ids] = nearest_compact.astype(
            np.int64,
            copy=False,
        )
    compact_vertices = vertices[referenced_vertex_ids]
    compact_faces = source_to_compact[faces]
    started = time.monotonic()
    _vertices, _faces, collapses = fast_simplification.simplify(
        compact_vertices,
        compact_faces,
        target_count=int(target_faces),
        agg=float(aggression),
        return_collapses=True,
    )
    proxy_vertices, proxy_faces, compact_to_proxy = replay_simplification(
        compact_vertices,
        compact_faces,
        collapses,
    )
    original_to_proxy = compact_to_proxy[source_to_compact]
    proxy_vertices_tensor = torch.from_numpy(proxy_vertices).float().contiguous()
    proxy_faces_tensor = torch.from_numpy(proxy_faces).long().contiguous()
    dense_to_proxy = torch.from_numpy(original_to_proxy).long().contiguous()
    proxy_landmarks = {
        int(landmark_id): int(dense_to_proxy[int(vertex_id)].item())
        for landmark_id, vertex_id in landmarks.items()
        if 0 <= int(vertex_id) < dense_to_proxy.numel()
    }
    proxy_mesh = {
        "vertices": proxy_vertices_tensor,
        "faces": proxy_faces_tensor,
        "normals": demo.recompute_vertex_normals(
            proxy_vertices_tensor,
            proxy_faces_tensor,
        ),
    }
    proxy_mesh["landmarks_3d"] = train.landmark_payload_from_mapping(
        proxy_mesh,
        proxy_landmarks,
    )
    cluster_sizes = torch.bincount(
        dense_to_proxy,
        minlength=proxy_vertices_tensor.shape[0],
    )
    report = {
        "target_face_count": int(target_faces),
        "aggression": float(aggression),
        "original_vertex_count": original_vertex_count,
        "original_face_count": int(mesh["faces"].shape[0]),
        "referenced_vertex_count": int(referenced_vertex_ids.shape[0]),
        "unreferenced_vertex_count": int(unreferenced_vertex_count),
        "proxy_vertex_count": int(proxy_vertices_tensor.shape[0]),
        "proxy_face_count": int(proxy_faces_tensor.shape[0]),
        "landmark_count": len(proxy_landmarks),
        "unique_landmark_vertex_count": len(set(proxy_landmarks.values())),
        "maximum_dense_cluster_size": int(cluster_sizes.max().item()),
        "build_seconds": time.monotonic() - started,
    }
    return proxy_mesh, proxy_landmarks, dense_to_proxy, report


def smooth_proxy_displacement_transfer(
    displacement: torch.Tensor,
    faces: torch.Tensor,
    *,
    pinned_vertex_ids: Any,
    iterations: int,
    blend: float,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Remove collapse-cluster steps without smoothing across the split seam."""

    if iterations < 0:
        raise ValueError("Proxy transfer smoothing iterations cannot be negative.")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("Proxy transfer smoothing blend must be in [0, 1].")
    if iterations == 0 or blend == 0.0:
        return displacement.contiguous()

    work_device = device or displacement.device
    faces = faces.detach().to(device=work_device, dtype=torch.long).contiguous()
    displacement = displacement.detach().to(
        device=work_device,
        dtype=torch.float32,
    ).contiguous()
    source = torch.cat(
        (
            faces[:, 0], faces[:, 1], faces[:, 2],
            faces[:, 1], faces[:, 2], faces[:, 0],
        )
    )
    target = torch.cat(
        (
            faces[:, 1], faces[:, 2], faces[:, 0],
            faces[:, 0], faces[:, 1], faces[:, 2],
        )
    )
    degree = torch.bincount(target, minlength=displacement.shape[0]).clamp_min(1)
    pinned = torch.zeros(
        displacement.shape[0],
        dtype=torch.bool,
        device=work_device,
    )
    pinned_ids = torch.tensor(
        sorted({int(value) for value in pinned_vertex_ids}),
        dtype=torch.long,
        device=work_device,
    )
    pinned_ids = pinned_ids[
        (pinned_ids >= 0) & (pinned_ids < displacement.shape[0])
    ]
    pinned[pinned_ids] = True
    fixed = displacement.clone()

    for _ in range(iterations):
        neighbor_sum = torch.zeros_like(displacement)
        neighbor_sum.index_add_(0, target, displacement.index_select(0, source))
        neighbor_average = neighbor_sum / degree.to(displacement.dtype).unsqueeze(1)
        displacement = torch.lerp(displacement, neighbor_average, float(blend))
        displacement[pinned] = fixed[pinned]
    return displacement.detach().cpu().contiguous()


def preprocess_split_mesh_for_inference(
    *,
    split_mesh: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    mesh_path: Path,
    input_convention: str,
    input_landmarks: Optional[Mapping[int, int]],
    reference_mesh_path: Path,
    reference_landmarks: Optional[Mapping[int, int]],
    reference_input_convention: str,
    min_alignment_landmarks: int,
    alignment_mode: str,
    reference_mesh_cache_path: Optional[Path] = None,
) -> dict[str, torch.Tensor]:
    processed_mesh = train.preprocess_custom_visualization_mesh(
        split_mesh,
        config,
        input_suffix=mesh_path.suffix.lower(),
        input_convention=input_convention,
    )
    if alignment_mode == "none":
        processed_mesh["landmarks_3d"] = train.landmark_payload_from_mapping(
            processed_mesh,
            input_landmarks or {},
        )
        return processed_mesh

    if input_landmarks is None or reference_landmarks is None:
        raise ValueError("Landmark alignment requires input and reference landmarks.")
    reference_path = reference_mesh_path.expanduser().resolve()
    reference_stat = reference_path.stat()
    cache_metadata = {
        "source": str(reference_path),
        "source_size": int(reference_stat.st_size),
        "source_mtime_ns": int(reference_stat.st_mtime_ns),
        "input_convention": str(reference_input_convention),
        "preprocess_settings": demo.mesh_preprocess_settings(config),
    }
    reference_mesh = None
    cache_path = (
        reference_mesh_cache_path.expanduser().resolve()
        if reference_mesh_cache_path is not None
        else None
    )
    if cache_path is not None and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("metadata") == cache_metadata:
            reference_mesh = cached.get("mesh")
    if reference_mesh is None:
        reference_mesh = train.preprocess_custom_visualization_mesh(
            demo.load_mesh_for_model(reference_path),
            config,
            input_suffix=reference_path.suffix.lower(),
            input_convention=reference_input_convention,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            torch.save(
                {
                    "metadata": cache_metadata,
                    "mesh": {
                        key: value.detach().cpu().contiguous()
                        for key, value in reference_mesh.items()
                        if isinstance(value, torch.Tensor)
                    },
                },
                temporary_path,
            )
            temporary_path.replace(cache_path)
    aligned_mesh = train.align_mesh_to_reference_landmarks(
        mesh=processed_mesh,
        input_landmarks=input_landmarks,
        reference_mesh=reference_mesh,
        reference_landmarks=reference_landmarks,
        min_landmarks=min_alignment_landmarks,
        alignment_mode=alignment_mode,
    )
    aligned_mesh["landmarks_3d"] = train.landmark_payload_from_mapping(
        aligned_mesh,
        input_landmarks,
    )
    return aligned_mesh


def validation_colors_with_cavity(
    vertices: torch.Tensor,
    normals: torch.Tensor,
    config: Mapping[str, Any],
    cavity_vertex_count: int,
    cavity_rgb: torch.Tensor,
) -> torch.Tensor:
    batched_vertices = vertices.detach().cpu().float().unsqueeze(0)
    batched_normals = normals.detach().cpu().float().unsqueeze(0)
    render_up_axis = train.resolve_validation_render_up_axis(config, batched_vertices)
    render_vertices = train.vertices_to_render_axes(batched_vertices, render_up_axis)
    render_normals = train.vertices_to_render_axes(batched_normals, render_up_axis)
    colors = train.validation_vertex_colors(
        render_vertices,
        normals=render_normals,
    )[0]
    return apply_cavity_color(colors, cavity_vertex_count, cavity_rgb)


def export_colors_with_cavity(
    vertices: torch.Tensor,
    cavity_vertex_count: int,
    cavity_rgb: torch.Tensor,
) -> torch.Tensor:
    colors = train.validation_vertex_colors(vertices.detach().cpu().float().unsqueeze(0))[0]
    return apply_cavity_color(colors, cavity_vertex_count, cavity_rgb)


def apply_cavity_color(
    colors: torch.Tensor,
    cavity_vertex_count: int,
    cavity_rgb: torch.Tensor,
) -> torch.Tensor:
    colors = colors.clone()
    if cavity_vertex_count > 0:
        colors[-cavity_vertex_count:] = cavity_rgb.to(dtype=colors.dtype)
    return colors.contiguous()


def fit_requested_oral_assembly(
    args: argparse.Namespace,
    neutral_vertices: torch.Tensor,
    deformed_vertices: torch.Tensor,
    split_landmarks: Mapping[int, int],
):
    return fit_oral_assembly(
        args.metahuman_oral_asset,
        neutral_vertices,
        deformed_vertices,
        split_landmarks,
        horizontal_axis=args.seam_horizontal_axis,
        depth_axis=args.seam_depth_axis,
        vertical_axis=args.seam_vertical_axis,
        assembly_scale=args.oral_assembly_scale,
        depth_offset=args.oral_depth_offset,
        level_target_frame=not args.no_oral_level_target_frame,
        aperture_crop_scale=(
            None if args.no_oral_aperture_crop else args.oral_aperture_crop_scale
        ),
        auto_center_teeth=args.oral_auto_center_teeth,
        maximum_center_offset_ratio=args.oral_maximum_center_offset_ratio,
        center_teeth_vertically=args.oral_center_teeth_vertically,
        tooth_depth_offset=args.oral_tooth_depth_offset,
        tooth_depth_blend_rings=args.oral_tooth_depth_blend_rings,
        auto_center_lower_teeth=args.oral_auto_center_lower_teeth,
        lower_teeth_depth_offset=args.oral_lower_teeth_depth_offset,
        lower_teeth_blend_rings=args.oral_lower_teeth_blend_rings,
        contain_inside_lips=not args.no_oral_containment,
        containment_scale=args.oral_containment_scale,
        containment_depth_inset=args.oral_containment_depth_inset,
        containment_blend_rings=args.oral_containment_blend_rings,
        rim_recession_ratio=args.oral_rim_recession_ratio,
        recessed_shell_horizontal_scale=(
            args.oral_recessed_shell_horizontal_scale
        ),
        recessed_shell_start_depth_ratio=(
            args.oral_recessed_shell_start_depth_ratio
        ),
        recessed_shell_blend_depth_ratio=(
            args.oral_recessed_shell_blend_depth_ratio
        ),
        trim_rim_transition_faces=(
            not args.no_oral_rim_transition_trim
        ),
        rim_transition_maximum_distance_ratio=(
            args.oral_rim_transition_distance_ratio
        ),
        rim_transition_depth_tolerance_ratio=(
            args.oral_rim_transition_depth_tolerance_ratio
        ),
        upper_crown_quantile=args.oral_upper_crown_quantile,
        upper_teeth_vertical_offset=args.oral_upper_teeth_vertical_offset,
        upper_teeth_vertical_blend_rings=(
            args.oral_upper_teeth_vertical_blend_rings
        ),
        remove_oral_wall_uv_faces=args.oral_dental_only,
        remove_tongue_uv_island=args.oral_remove_tongue_uv_island,
        tongue_planar_scale=args.oral_tongue_planar_scale,
        tongue_depth_offset=args.oral_tongue_depth_offset,
        tongue_vertical_offset=args.oral_tongue_vertical_offset,
        tongue_offset_blend_rings=args.oral_tongue_offset_blend_rings,
    )

def apply_oral_vertex_colors(
    colors: torch.Tensor,
    oral_vertex_count: int,
    oral_vertex_colors: Optional[torch.Tensor],
    *,
    trailing_vertex_count: int = 0,
) -> torch.Tensor:
    if oral_vertex_count <= 0:
        return colors.contiguous()
    if oral_vertex_colors is None or oral_vertex_colors.shape != (oral_vertex_count, 3):
        raise ValueError("Oral vertex colors must have shape [oral_vertex_count, 3].")
    if trailing_vertex_count < 0 or oral_vertex_count + trailing_vertex_count > colors.shape[0]:
        raise ValueError("Oral/trailing vertex counts exceed the color array.")
    colors = colors.clone()
    oral_end = colors.shape[0] - trailing_vertex_count
    oral_start = oral_end - oral_vertex_count
    colors[oral_start:oral_end] = oral_vertex_colors.to(dtype=colors.dtype)
    return colors.contiguous()


def apply_component_vertex_color(
    colors: torch.Tensor,
    vertex_count: int,
    rgb: torch.Tensor,
    *,
    trailing_vertex_count: int = 0,
) -> torch.Tensor:
    if vertex_count <= 0:
        return colors.contiguous()
    if trailing_vertex_count < 0 or vertex_count + trailing_vertex_count > colors.shape[0]:
        raise ValueError("Component/trailing vertex counts exceed the color array.")
    colors = colors.clone()
    component_end = colors.shape[0] - trailing_vertex_count
    component_start = component_end - vertex_count
    colors[component_start:component_end] = rgb.to(dtype=colors.dtype)
    return colors.contiguous()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
