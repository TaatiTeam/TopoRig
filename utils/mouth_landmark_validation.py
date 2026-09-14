from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch


VALIDATED_MOUTH_LANDMARK_VERSION = 2

OUTER_UPPER_LIP_IDS = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)
OUTER_LOWER_LIP_IDS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291)
INNER_UPPER_LIP_IDS = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
INNER_LOWER_LIP_IDS = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308)
MOUTH_CURVES = (
    OUTER_UPPER_LIP_IDS,
    OUTER_LOWER_LIP_IDS,
    INNER_UPPER_LIP_IDS,
    INNER_LOWER_LIP_IDS,
)
MOUTH_LANDMARK_IDS = tuple(
    dict.fromkeys(value for curve in MOUTH_CURVES for value in curve)
)
CHIN_LANDMARK_ID = 152
CRITICAL_LANDMARK_IDS = (13, 14, 17, CHIN_LANDMARK_ID)
REQUIRED_LANDMARK_IDS = (*MOUTH_LANDMARK_IDS, CHIN_LANDMARK_ID)
CHIN_NEIGHBOR_IDS = (148, 176, 377, 400)


@dataclass(frozen=True)
class MouthMappingValidation:
    valid: bool
    reasons: tuple[str, ...]
    invalid_landmark_ids: tuple[int, ...]
    invalid_action_units: tuple[int, ...]
    metrics: dict[str, float]


def build_validated_mouth_landmark_payload(
    *,
    mesh_id: str,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    mapping: Mapping[int, int],
    target_deltas: Mapping[int, torch.Tensor],
    vertex_groups: Optional[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    vertices = _cpu_vertices(vertices)
    faces = _cpu_faces(faces, vertices.shape[0])
    normalized_mapping = {int(key): int(value) for key, value in mapping.items()}
    normalized_targets = {
        int(action_unit): _cpu_target_delta(delta, vertices)
        for action_unit, delta in target_deltas.items()
    }
    component_ids = mesh_connected_component_ids(vertices.shape[0], faces)
    validation = validate_mouth_landmark_mapping(
        vertices=vertices,
        mapping=normalized_mapping,
        target_deltas=normalized_targets,
        component_ids=component_ids,
        vertex_groups=vertex_groups,
        config=config,
    )

    status = "valid"
    repair_metrics: dict[str, float] = {}
    repaired_mapping = normalized_mapping
    should_repair = bool(validation.reasons or validation.invalid_action_units)
    if should_repair and bool(_repair_config(config).get("enabled", True)):
        repaired_mapping, repair_metrics = repair_mouth_landmark_mapping(
            vertices=vertices,
            mapping=normalized_mapping,
            target_deltas=normalized_targets,
            component_ids=component_ids,
            vertex_groups=vertex_groups,
            config=config,
        )
        repaired_validation = validate_mouth_landmark_mapping(
            vertices=vertices,
            mapping=repaired_mapping,
            target_deltas=normalized_targets,
            component_ids=component_ids,
            vertex_groups=vertex_groups,
            config=config,
        )
        if repaired_mapping != normalized_mapping:
            status = "repaired"
        validation = repaired_validation

    if validation.reasons:
        status = "invalid"

    return {
        "version": VALIDATED_MOUTH_LANDMARK_VERSION,
        "mesh_id": str(mesh_id),
        "status": status,
        "reasons": list(validation.reasons),
        "invalid_landmark_ids": list(validation.invalid_landmark_ids),
        "invalid_action_units": list(validation.invalid_action_units),
        "checked_action_units": sorted(int(value) for value in normalized_targets),
        "metrics": {**validation.metrics, **repair_metrics},
        "mapping": {
            str(mediapipe_id): int(vertex_id)
            for mediapipe_id, vertex_id in sorted(repaired_mapping.items())
        },
    }


def validate_mouth_landmark_mapping(
    *,
    vertices: torch.Tensor,
    mapping: Mapping[int, int],
    target_deltas: Mapping[int, torch.Tensor],
    component_ids: torch.Tensor,
    vertex_groups: Optional[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> MouthMappingValidation:
    vertices = _cpu_vertices(vertices)
    components = component_ids.detach().cpu().long()
    reasons: list[str] = []
    invalid_ids: set[int] = set()
    metrics: dict[str, float] = {}

    missing = [value for value in REQUIRED_LANDMARK_IDS if value not in mapping]
    if missing:
        reasons.append("mouth:missing_landmarks")
        invalid_ids.update(missing)
        metrics["mouth_missing_landmark_count"] = float(len(missing))
        return MouthMappingValidation(
            valid=False,
            reasons=tuple(reasons),
            invalid_landmark_ids=tuple(sorted(invalid_ids)),
            invalid_action_units=tuple(sorted(int(value) for value in target_deltas)),
            metrics=metrics,
        )

    required_vertex_ids = torch.tensor(
        [int(mapping[value]) for value in REQUIRED_LANDMARK_IDS],
        dtype=torch.long,
    )
    out_of_range_mask = (required_vertex_ids < 0) | (
        required_vertex_ids >= vertices.shape[0]
    )
    if bool(out_of_range_mask.any()):
        reasons.append("mouth:out_of_range_vertex")
        invalid_ids.update(
            REQUIRED_LANDMARK_IDS[index]
            for index in torch.where(out_of_range_mask)[0].tolist()
        )
        return MouthMappingValidation(
            valid=False,
            reasons=tuple(reasons),
            invalid_landmark_ids=tuple(sorted(invalid_ids)),
            invalid_action_units=tuple(sorted(int(value) for value in target_deltas)),
            metrics=metrics,
        )

    unique_count = int(torch.unique(required_vertex_ids).numel())
    minimum_unique = int(config.get("min_unique_vertices", len(REQUIRED_LANDMARK_IDS)))
    metrics["mouth_unique_vertex_count"] = float(unique_count)
    if unique_count < minimum_unique:
        reasons.append("mouth:too_few_unique_vertices")
        counts = Counter(required_vertex_ids.tolist())
        invalid_ids.update(
            mediapipe_id
            for mediapipe_id, vertex_id in zip(
                REQUIRED_LANDMARK_IDS,
                required_vertex_ids.tolist(),
            )
            if counts[vertex_id] > 1
        )

    facial_vertex_ids, facial_component_count = facial_surface_vertex_ids(
        component_ids=components,
        mapping=mapping,
        vertex_groups=vertex_groups,
        config=config,
    )
    facial_mask = torch.zeros(vertices.shape[0], dtype=torch.bool)
    facial_mask[facial_vertex_ids] = True
    off_face = ~facial_mask.index_select(0, required_vertex_ids)
    metrics["mouth_facial_component_count"] = float(facial_component_count)
    metrics["mouth_facial_vertex_count"] = float(facial_vertex_ids.numel())
    metrics["mouth_off_facial_component_count"] = float(off_face.sum().item())
    if bool(off_face.any()):
        reasons.append("mouth:non_facial_component")
        invalid_ids.update(
            REQUIRED_LANDMARK_IDS[index]
            for index in torch.where(off_face)[0].tolist()
        )

    points = {
        mediapipe_id: vertices[int(mapping[mediapipe_id])]
        for mediapipe_id in REQUIRED_LANDMARK_IDS
    }
    mesh_extent = max(
        float((vertices.amax(dim=0) - vertices.amin(dim=0)).amax().item()),
        1.0e-12,
    )
    left_corner = points[61]
    right_corner = points[291]
    horizontal = right_corner - left_corner
    mouth_width = float(torch.linalg.vector_norm(horizontal).item())
    width_ratio = mouth_width / mesh_extent
    metrics["mouth_width"] = mouth_width
    metrics["mouth_width_mesh_extent_ratio"] = width_ratio
    if width_ratio < float(config.get("min_mouth_width_mesh_extent_ratio", 0.01)):
        reasons.append("mouth:width_too_small")
        invalid_ids.update((61, 291))

    if mouth_width > 1.0e-12:
        horizontal = horizontal / mouth_width
        _validate_mouth_curve_geometry(
            points=points,
            left_corner=left_corner,
            horizontal=horizontal,
            mouth_width=mouth_width,
            config=config,
            reasons=reasons,
            invalid_ids=invalid_ids,
            metrics=metrics,
        )
        _validate_critical_landmark_geometry(
            points=points,
            left_corner=left_corner,
            horizontal=horizontal,
            mouth_width=mouth_width,
            config=config,
            reasons=reasons,
            invalid_ids=invalid_ids,
            metrics=metrics,
        )

    invalid_action_units, motion_invalid_ids, motion_metrics = (
        validate_mouth_target_motion(
            vertices=vertices,
            mapping=mapping,
            target_deltas=target_deltas,
            config=config,
        )
    )
    invalid_ids.update(motion_invalid_ids)
    metrics.update(motion_metrics)
    return MouthMappingValidation(
        valid=not reasons and not invalid_action_units,
        reasons=tuple(dict.fromkeys(reasons)),
        invalid_landmark_ids=tuple(sorted(invalid_ids)),
        invalid_action_units=tuple(sorted(invalid_action_units)),
        metrics=metrics,
    )


def _validate_mouth_curve_geometry(
    *,
    points: Mapping[int, torch.Tensor],
    left_corner: torch.Tensor,
    horizontal: torch.Tensor,
    mouth_width: float,
    config: Mapping[str, Any],
    reasons: list[str],
    invalid_ids: set[int],
    metrics: dict[str, float],
) -> None:
    max_adjacent_ratio = float(config.get("max_adjacent_mouth_width_ratio", 0.35))
    min_order_step_ratio = float(config.get("min_order_step_ratio", -0.04))
    max_depth_ratio = float(config.get("max_curve_depth_mouth_width_ratio", 0.65))
    depth_axis, depth_sign = _axis(config.get("mesh_front_axis", "y"))

    for curve_index, curve in enumerate(MOUTH_CURVES):
        curve_points = torch.stack([points[value] for value in curve])
        adjacent = torch.linalg.vector_norm(
            curve_points[1:] - curve_points[:-1],
            dim=-1,
        ) / mouth_width
        projected = ((curve_points - left_corner) @ horizontal) / mouth_width
        order_steps = projected[1:] - projected[:-1]
        depth = curve_points[:, depth_axis] * depth_sign
        depth_ratio = float((depth.max() - depth.min()).item()) / mouth_width
        metrics[f"mouth_curve_{curve_index}_max_adjacent_ratio"] = float(
            adjacent.max().item()
        )
        metrics[f"mouth_curve_{curve_index}_min_order_step_ratio"] = float(
            order_steps.min().item()
        )
        metrics[f"mouth_curve_{curve_index}_depth_range_ratio"] = depth_ratio

        bad_adjacent = torch.where(adjacent > max_adjacent_ratio)[0].tolist()
        if bad_adjacent:
            reasons.append("mouth:curve_discontinuity")
            for index in bad_adjacent:
                invalid_ids.update((curve[index], curve[index + 1]))
        bad_order = torch.where(order_steps < min_order_step_ratio)[0].tolist()
        if bad_order:
            reasons.append("mouth:curve_ordering")
            for index in bad_order:
                invalid_ids.update((curve[index], curve[index + 1]))
        if depth_ratio > max_depth_ratio:
            reasons.append("mouth:curve_depth_inconsistent")
            depth_median = depth.median()
            deviations = (depth - depth_median).abs() / mouth_width
            invalid_ids.add(curve[int(torch.argmax(deviations).item())])


def _validate_critical_landmark_geometry(
    *,
    points: Mapping[int, torch.Tensor],
    left_corner: torch.Tensor,
    horizontal: torch.Tensor,
    mouth_width: float,
    config: Mapping[str, Any],
    reasons: list[str],
    invalid_ids: set[int],
    metrics: dict[str, float],
) -> None:
    up_axis, up_sign = _axis(config.get("mesh_up_axis", "z"))
    depth_axis, depth_sign = _axis(config.get("mesh_front_axis", "y"))
    up = {
        value: float((points[value][up_axis] * up_sign).item())
        for value in CRITICAL_LANDMARK_IDS
    }
    horizontal_offsets = {
        value: abs(float(((points[value] - left_corner) @ horizontal).item()) / mouth_width - 0.5)
        for value in CRITICAL_LANDMARK_IDS
    }
    for value, offset in horizontal_offsets.items():
        metrics[f"mouth_landmark_{value}_horizontal_center_offset_ratio"] = offset
        if offset > float(config.get("max_center_horizontal_offset_ratio", 0.30)):
            reasons.append("mouth:critical_horizontal_ordering")
            invalid_ids.add(value)

    inner_gap = (up[13] - up[14]) / mouth_width
    lower_lip_gap = (up[14] - up[17]) / mouth_width
    chin_gap = (up[17] - up[CHIN_LANDMARK_ID]) / mouth_width
    metrics.update(
        {
            "mouth_13_14_vertical_gap_ratio": inner_gap,
            "mouth_14_17_vertical_gap_ratio": lower_lip_gap,
            "mouth_17_152_vertical_gap_ratio": chin_gap,
        }
    )
    if inner_gap < float(config.get("min_13_14_vertical_gap_ratio", -0.01)):
        reasons.append("mouth:landmark_13_14_ordering")
        invalid_ids.update((13, 14))
    if lower_lip_gap < float(config.get("min_14_17_vertical_gap_ratio", 0.01)):
        reasons.append("mouth:landmark_14_17_ordering")
        invalid_ids.update((14, 17))
    minimum_chin_gap = float(config.get("min_17_152_vertical_gap_ratio", 0.10))
    maximum_chin_gap = float(config.get("max_17_152_vertical_gap_ratio", 1.60))
    if chin_gap < minimum_chin_gap or chin_gap > maximum_chin_gap:
        reasons.append("mouth:landmark_17_152_proximity")
        invalid_ids.update((17, CHIN_LANDMARK_ID))

    loop_depth = torch.stack(
        [points[value][depth_axis] * depth_sign for value in MOUTH_LANDMARK_IDS]
    )
    median_depth = loop_depth.median()
    maximum_depth_offset = float(
        config.get("max_critical_depth_mouth_width_ratio", 0.65)
    )
    for value in (13, 14, 17):
        depth_offset = abs(
            float((points[value][depth_axis] * depth_sign - median_depth).item())
        ) / mouth_width
        metrics[f"mouth_landmark_{value}_depth_offset_ratio"] = depth_offset
        if depth_offset > maximum_depth_offset:
            reasons.append("mouth:critical_depth_inconsistent")
            invalid_ids.add(value)
    chin_depth_offset = abs(
        float(
            (
                points[CHIN_LANDMARK_ID][depth_axis] * depth_sign
                - points[17][depth_axis] * depth_sign
            ).item()
        )
    ) / mouth_width
    metrics["mouth_landmark_152_depth_offset_ratio"] = chin_depth_offset
    if chin_depth_offset > float(config.get("max_chin_depth_mouth_width_ratio", 0.85)):
        reasons.append("mouth:landmark_152_depth_inconsistent")
        invalid_ids.add(CHIN_LANDMARK_ID)


def validate_mouth_target_motion(
    *,
    vertices: torch.Tensor,
    mapping: Mapping[int, int],
    target_deltas: Mapping[int, torch.Tensor],
    config: Mapping[str, Any],
) -> tuple[set[int], set[int], dict[str, float]]:
    motion_config = _motion_config(config)
    if not bool(motion_config.get("enabled", True)):
        return set(), set(), {}
    if any(value not in mapping for value in REQUIRED_LANDMARK_IDS):
        return set(target_deltas), set(REQUIRED_LANDMARK_IDS), {}

    mesh_extent = max(
        float((vertices.amax(dim=0) - vertices.amin(dim=0)).amax().item()),
        1.0e-12,
    )
    minimum_motion = float(
        motion_config.get("min_active_motion_mesh_extent_ratio", 1.0e-4)
    ) * mesh_extent
    max_residual_ratio = float(motion_config.get("max_neighbor_residual_ratio", 2.0))
    max_critical_ratio = float(
        motion_config.get("max_critical_neighbor_residual_ratio", 0.75)
    )
    max_chin_ratio = float(motion_config.get("max_chin_neighbor_residual_ratio", 0.75))
    neighbor_ids = mouth_landmark_neighbors()
    invalid_action_units: set[int] = set()
    invalid_landmark_ids: set[int] = set()
    metrics: dict[str, float] = {}

    mouth_vertex_ids = torch.tensor(
        [int(mapping[value]) for value in MOUTH_LANDMARK_IDS],
        dtype=torch.long,
    )
    for action_unit, target_delta in sorted(target_deltas.items()):
        if target_delta.shape != vertices.shape:
            raise ValueError(
                f"AU{action_unit} target delta must match neutral vertices."
            )
        mouth_delta = target_delta.index_select(0, mouth_vertex_ids)
        motion_scale = float(
            torch.quantile(
                torch.linalg.vector_norm(mouth_delta, dim=-1),
                0.90,
            ).item()
        )
        metrics[f"mouth_au{action_unit}_motion_scale"] = motion_scale
        if motion_scale < minimum_motion:
            continue

        worst_ratio = 0.0
        worst_critical_ratio = 0.0
        for mediapipe_id in MOUTH_LANDMARK_IDS:
            neighbors = [
                value for value in neighbor_ids.get(mediapipe_id, ()) if value in mapping
            ]
            if len(neighbors) < 2:
                continue
            actual = target_delta[int(mapping[mediapipe_id])]
            expected = torch.stack(
                [target_delta[int(mapping[value])] for value in neighbors]
            ).mean(dim=0)
            ratio = float(torch.linalg.vector_norm(actual - expected).item()) / max(
                motion_scale,
                1.0e-12,
            )
            worst_ratio = max(worst_ratio, ratio)
            threshold = (
                max_critical_ratio
                if mediapipe_id in {13, 14, 17}
                else max_residual_ratio
            )
            if mediapipe_id in {13, 14, 17}:
                worst_critical_ratio = max(worst_critical_ratio, ratio)
            if ratio > threshold:
                invalid_action_units.add(int(action_unit))
                invalid_landmark_ids.add(int(mediapipe_id))

        if all(value in mapping for value in CHIN_NEIGHBOR_IDS):
            actual = target_delta[int(mapping[CHIN_LANDMARK_ID])]
            expected = torch.stack(
                [target_delta[int(mapping[value])] for value in CHIN_NEIGHBOR_IDS]
            ).mean(dim=0)
            chin_ratio = float(torch.linalg.vector_norm(actual - expected).item()) / max(
                motion_scale,
                1.0e-12,
            )
            metrics[f"mouth_au{action_unit}_landmark_152_motion_residual_ratio"] = (
                chin_ratio
            )
            if chin_ratio > max_chin_ratio:
                invalid_action_units.add(int(action_unit))
                invalid_landmark_ids.add(CHIN_LANDMARK_ID)

        metrics[f"mouth_au{action_unit}_max_motion_residual_ratio"] = worst_ratio
        metrics[f"mouth_au{action_unit}_max_critical_motion_residual_ratio"] = (
            worst_critical_ratio
        )

    return invalid_action_units, invalid_landmark_ids, metrics


def repair_mouth_landmark_mapping(
    *,
    vertices: torch.Tensor,
    mapping: Mapping[int, int],
    target_deltas: Mapping[int, torch.Tensor],
    component_ids: torch.Tensor,
    vertex_groups: Optional[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[dict[int, int], dict[str, float]]:
    try:
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError("Mouth landmark repair requires scipy.") from exc

    repaired = {int(key): int(value) for key, value in mapping.items()}
    if any(value not in repaired for value in REQUIRED_LANDMARK_IDS):
        return repaired, {"mouth_repair_landmark_count": 0.0}
    current_ids = torch.tensor(
        [repaired[value] for value in REQUIRED_LANDMARK_IDS],
        dtype=torch.long,
    )
    if bool(((current_ids < 0) | (current_ids >= vertices.shape[0])).any()):
        return repaired, {"mouth_repair_landmark_count": 0.0}

    repair_config = _repair_config(config)
    facial_vertex_ids, facial_component_count = facial_surface_vertex_ids(
        component_ids=component_ids,
        mapping=repaired,
        vertex_groups=vertex_groups,
        config=config,
    )
    metrics = {
        "mouth_repair_facial_component_count": float(facial_component_count),
        "mouth_repair_facial_vertex_count": float(facial_vertex_ids.numel()),
        "mouth_repair_landmark_count": 0.0,
    }
    if facial_vertex_ids.numel() < len(REQUIRED_LANDMARK_IDS):
        return repaired, metrics

    source_points_tensor = vertices.index_select(0, current_ids).clone()
    position_outliers = mouth_position_outlier_ids(
        vertices=vertices,
        mapping=repaired,
        config=config,
    )
    depth_outliers, mouth_depth, mouth_width = global_mouth_depth_outlier_ids(
        vertices=vertices,
        mapping=repaired,
        config=config,
    )
    position_outliers.update(depth_outliers)
    metrics["mouth_repair_position_outlier_count"] = float(
        len(position_outliers)
    )
    required_index = {
        mediapipe_id: index
        for index, mediapipe_id in enumerate(REQUIRED_LANDMARK_IDS)
    }
    source_points_tensor = interpolate_mouth_curve_outliers(
        vertices=vertices,
        mapping=repaired,
        source_points=source_points_tensor,
        required_index=required_index,
        outlier_ids=position_outliers,
    )
    metrics["mouth_repair_global_depth_outlier_count"] = float(
        len(depth_outliers)
    )
    metrics["mouth_repair_reference_depth"] = float(mouth_depth)

    # Fragmented Pixel3D heads can put a coherent run of MediaPipe points on a
    # rear face or mouth layer.  Once the source curve has been projected back
    # to its stable neighbours, do not let the assignment snap it straight back
    # onto that rear layer.
    depth_axis, depth_sign = _axis(config.get("mesh_front_axis", "y"))
    candidate_depth = (
        vertices.index_select(0, facial_vertex_ids)[:, depth_axis] * depth_sign
    )
    maximum_candidate_depth = float(
        repair_config.get("max_candidate_depth_mouth_width_ratio", 0.80)
    ) * max(mouth_width, 1.0e-12)
    depth_candidate_mask = (
        candidate_depth - float(mouth_depth)
    ).abs() <= maximum_candidate_depth
    depth_filtered_vertex_ids = facial_vertex_ids[depth_candidate_mask]
    if depth_filtered_vertex_ids.numel() >= len(REQUIRED_LANDMARK_IDS):
        facial_vertex_ids = depth_filtered_vertex_ids
    metrics["mouth_repair_depth_filtered_vertex_count"] = float(
        facial_vertex_ids.numel()
    )
    source_points = source_points_tensor.numpy()
    candidate_points = vertices.index_select(0, facial_vertex_ids).numpy()
    tree = cKDTree(candidate_points)
    nearest_count = min(
        max(int(repair_config.get("nearest_candidates", 48)), 1),
        int(facial_vertex_ids.numel()),
    )
    distances, local_indices = tree.query(source_points, k=nearest_count)
    distances = np.asarray(distances)
    local_indices = np.asarray(local_indices)
    if nearest_count == 1:
        distances = distances[:, None]
        local_indices = local_indices[:, None]
    candidate_local_ids = np.unique(local_indices.reshape(-1))
    assignment_points = candidate_points[candidate_local_ids]
    mesh_extent = max(
        float((vertices.amax(dim=0) - vertices.amin(dim=0)).amax().item()),
        1.0e-12,
    )
    maximum_distance = float(
        repair_config.get("max_distance_mesh_extent_ratio", 0.06)
    ) * mesh_extent
    position_cost = np.linalg.norm(
        source_points[:, None, :] - assignment_points[None, :, :],
        axis=-1,
    ) / max(maximum_distance, 1.0e-12)
    costs = position_cost + _repair_motion_cost(
        mapping=repaired,
        required_ids=REQUIRED_LANDMARK_IDS,
        candidate_vertex_ids=facial_vertex_ids.numpy()[candidate_local_ids],
        target_deltas=target_deltas,
        vertices=vertices,
        config=config,
        ignore_mediapipe_ids=position_outliers,
    )
    costs[position_cost > 1.0] = 1.0e6
    row_ids, column_ids = linear_sum_assignment(costs)
    if len(row_ids) != len(REQUIRED_LANDMARK_IDS):
        return repaired, metrics
    assigned_costs = costs[row_ids, column_ids]
    if bool(np.any(assigned_costs >= 1.0e6)):
        return repaired, metrics

    assigned_vertex_ids = facial_vertex_ids.numpy()[candidate_local_ids[column_ids]]
    repaired_count = 0
    max_assigned_distance = 0.0
    for row_id, vertex_id, column_id in zip(
        row_ids.tolist(),
        assigned_vertex_ids.tolist(),
        column_ids.tolist(),
    ):
        mediapipe_id = REQUIRED_LANDMARK_IDS[row_id]
        if repaired[mediapipe_id] != int(vertex_id):
            repaired_count += 1
        repaired[mediapipe_id] = int(vertex_id)
        max_assigned_distance = max(
            max_assigned_distance,
            float(np.linalg.norm(source_points[row_id] - assignment_points[column_id])),
        )
    metrics.update(
        {
            "mouth_repair_landmark_count": float(repaired_count),
            "mouth_repair_max_distance": max_assigned_distance,
            "mouth_repair_max_distance_mesh_extent_ratio": (
                max_assigned_distance / mesh_extent
            ),
        }
    )
    return repaired, metrics


def mouth_position_outlier_ids(
    *,
    vertices: torch.Tensor,
    mapping: Mapping[int, int],
    config: Mapping[str, Any],
) -> set[int]:
    if any(value not in mapping for value in MOUTH_LANDMARK_IDS):
        return set()
    left = vertices[int(mapping[61])]
    right = vertices[int(mapping[291])]
    width = float(torch.linalg.vector_norm(right - left).item())
    if width <= 1.0e-12:
        return set(MOUTH_LANDMARK_IDS)
    max_adjacent_ratio = float(config.get("max_adjacent_mouth_width_ratio", 0.35))
    depth_outliers, _mouth_depth, _mouth_width = global_mouth_depth_outlier_ids(
        vertices=vertices,
        mapping=mapping,
        config=config,
    )
    output: set[int] = set()
    for curve in MOUTH_CURVES:
        points = torch.stack([vertices[int(mapping[value])] for value in curve])
        adjacent = torch.linalg.vector_norm(
            points[1:] - points[:-1],
            dim=-1,
        ) / width
        bad_edges = adjacent > max_adjacent_ratio
        for index in range(1, len(curve) - 1):
            if bool(bad_edges[index - 1] and bad_edges[index]):
                # A valid front-surface island between two rear-layer points is
                # not the outlier.  The global depth test identifies the two
                # neighbours that should be projected forward instead.
                if (
                    curve[index] not in depth_outliers
                    and curve[index - 1] in depth_outliers
                    and curve[index + 1] in depth_outliers
                ):
                    continue
                output.add(curve[index])
    output.update(depth_outliers)
    return output


def global_mouth_depth_outlier_ids(
    *,
    vertices: torch.Tensor,
    mapping: Mapping[int, int],
    config: Mapping[str, Any],
) -> tuple[set[int], float, float]:
    """Find mouth points that landed on a coherent rear geometry layer."""

    if any(value not in mapping for value in MOUTH_LANDMARK_IDS):
        return set(), 0.0, 0.0
    left = vertices[int(mapping[61])]
    right = vertices[int(mapping[291])]
    mouth_width = float(torch.linalg.vector_norm(right - left).item())
    if mouth_width <= 1.0e-12:
        return set(MOUTH_LANDMARK_IDS), 0.0, mouth_width

    depth_axis, depth_sign = _axis(config.get("mesh_front_axis", "y"))
    # The outer lips and inner upper lip provide enough redundant anchors that
    # their median remains on the visible surface even when a few raycasts hit
    # a rear layer.  Do not derive this reference from each curve separately:
    # an entire bad inner-lower curve can otherwise validate its own depth.
    reference_ids = tuple(
        dict.fromkeys(
            (*OUTER_UPPER_LIP_IDS, *OUTER_LOWER_LIP_IDS, *INNER_UPPER_LIP_IDS)
        )
    )
    reference_depths = torch.tensor(
        [
            float(vertices[int(mapping[value]), depth_axis].item()) * depth_sign
            for value in reference_ids
        ],
        dtype=vertices.dtype,
    )
    reference_depth = float(reference_depths.median().item())
    maximum_depth = float(
        config.get(
            "max_global_mouth_depth_width_ratio",
            config.get("max_curve_depth_mouth_width_ratio", 0.65),
        )
    ) * mouth_width
    outliers = {
        int(mediapipe_id)
        for mediapipe_id in MOUTH_LANDMARK_IDS
        if abs(
            float(vertices[int(mapping[mediapipe_id]), depth_axis].item())
            * depth_sign
            - reference_depth
        )
        > maximum_depth
    }
    return outliers, reference_depth, mouth_width


def interpolate_mouth_curve_outliers(
    *,
    vertices: torch.Tensor,
    mapping: Mapping[int, int],
    source_points: torch.Tensor,
    required_index: Mapping[int, int],
    outlier_ids: set[int],
) -> torch.Tensor:
    """Interpolate bad curve runs between their nearest stable landmarks."""

    if not outlier_ids:
        return source_points
    repaired_points = source_points.clone()
    estimates: defaultdict[int, list[torch.Tensor]] = defaultdict(list)
    for curve in MOUTH_CURVES:
        stable_indices = [
            index
            for index, mediapipe_id in enumerate(curve)
            if mediapipe_id not in outlier_ids
        ]
        for index, mediapipe_id in enumerate(curve):
            if mediapipe_id not in outlier_ids:
                continue
            left_indices = [value for value in stable_indices if value < index]
            right_indices = [value for value in stable_indices if value > index]
            if not left_indices or not right_indices:
                continue
            left_index = left_indices[-1]
            right_index = right_indices[0]
            left_point = vertices[int(mapping[curve[left_index]])]
            right_point = vertices[int(mapping[curve[right_index]])]
            alpha = float(index - left_index) / float(right_index - left_index)
            estimates[int(mediapipe_id)].append(
                torch.lerp(left_point, right_point, alpha)
            )

    neighbors = mouth_landmark_neighbors()
    for mediapipe_id in outlier_ids:
        if mediapipe_id not in required_index:
            continue
        candidates = estimates.get(mediapipe_id, ())
        if candidates:
            repaired_points[required_index[mediapipe_id]] = torch.stack(
                list(candidates)
            ).mean(dim=0)
            continue
        neighbor_ids = [
            value
            for value in neighbors.get(mediapipe_id, ())
            if value in mapping and value not in outlier_ids
        ]
        if len(neighbor_ids) >= 2:
            repaired_points[required_index[mediapipe_id]] = torch.stack(
                [vertices[int(mapping[value])] for value in neighbor_ids]
            ).mean(dim=0)
    return repaired_points


def _repair_motion_cost(
    *,
    mapping: Mapping[int, int],
    required_ids: Sequence[int],
    candidate_vertex_ids: np.ndarray,
    target_deltas: Mapping[int, torch.Tensor],
    vertices: torch.Tensor,
    config: Mapping[str, Any],
    ignore_mediapipe_ids: Optional[set[int]] = None,
) -> np.ndarray:
    weight = float(_repair_config(config).get("target_motion_cost_weight", 0.35))
    output = np.zeros((len(required_ids), len(candidate_vertex_ids)), dtype=np.float64)
    if weight <= 0.0 or not target_deltas:
        return output

    mesh_extent = max(
        float((vertices.amax(dim=0) - vertices.amin(dim=0)).amax().item()),
        1.0e-12,
    )
    minimum_motion = float(
        _motion_config(config).get("min_active_motion_mesh_extent_ratio", 1.0e-4)
    ) * mesh_extent
    neighbors = mouth_landmark_neighbors()
    candidate_tensor = torch.from_numpy(candidate_vertex_ids).long()
    active_count = 0
    for target_delta in target_deltas.values():
        current_mouth_ids = torch.tensor(
            [mapping[value] for value in MOUTH_LANDMARK_IDS],
            dtype=torch.long,
        )
        scale = float(
            torch.quantile(
                torch.linalg.vector_norm(
                    target_delta.index_select(0, current_mouth_ids),
                    dim=-1,
                ),
                0.90,
            ).item()
        )
        if scale < minimum_motion:
            continue
        candidate_motion = target_delta.index_select(0, candidate_tensor)
        for row_id, mediapipe_id in enumerate(required_ids):
            # The target motion attached to a landmark on the wrong geometry
            # layer is not semantic evidence.  Using it during repair can pull
            # a correctly projected lip point back toward the bad layer.
            if (
                ignore_mediapipe_ids is not None
                and mediapipe_id in ignore_mediapipe_ids
            ):
                continue
            expected_ids = (
                CHIN_NEIGHBOR_IDS
                if mediapipe_id == CHIN_LANDMARK_ID
                else neighbors.get(mediapipe_id, ())
            )
            expected_ids = tuple(value for value in expected_ids if value in mapping)
            if len(expected_ids) < 2:
                continue
            expected = torch.stack(
                [target_delta[mapping[value]] for value in expected_ids]
            ).mean(dim=0)
            residual = torch.linalg.vector_norm(
                candidate_motion - expected,
                dim=-1,
            ) / max(scale, 1.0e-12)
            output[row_id] += residual.numpy()
        active_count += 1
    if active_count > 0:
        output *= weight / float(active_count)
    return output


def facial_surface_vertex_ids(
    *,
    component_ids: torch.Tensor,
    mapping: Mapping[int, int],
    vertex_groups: Optional[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, int]:
    components = component_ids.detach().cpu().long()
    head_vertex_ids = _head_vertex_ids(vertex_groups, components.numel())
    if head_vertex_ids.numel() > 0:
        component_count = int(
            torch.unique(components.index_select(0, head_vertex_ids)).numel()
        )
        return head_vertex_ids, component_count

    excluded_ids = set(REQUIRED_LANDMARK_IDS)
    component_votes: Counter[int] = Counter()
    for mediapipe_id, vertex_id in mapping.items():
        vertex_id = int(vertex_id)
        if int(mediapipe_id) in excluded_ids:
            continue
        if vertex_id < 0 or vertex_id >= components.numel():
            continue
        component_votes[int(components[vertex_id].item())] += 1
    minimum_votes = max(int(config.get("min_component_landmark_votes", 2)), 1)
    selected = sorted(
        component_id
        for component_id, votes in component_votes.items()
        if votes >= minimum_votes
    )
    if not selected:
        return torch.empty(0, dtype=torch.long), 0
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    mask = torch.isin(components, selected_tensor)
    return torch.where(mask)[0], len(selected)


def mesh_connected_component_ids(
    vertex_count: int,
    faces: torch.Tensor,
) -> torch.Tensor:
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError as exc:
        raise ImportError("Mouth component validation requires scipy.") from exc

    faces_np = faces.detach().cpu().long().numpy()
    if faces_np.size == 0:
        return torch.arange(vertex_count, dtype=torch.long)
    edges = np.concatenate(
        (
            faces_np[:, [0, 1]],
            faces_np[:, [1, 2]],
            faces_np[:, [2, 0]],
        ),
        axis=0,
    )
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    columns = np.concatenate((edges[:, 1], edges[:, 0]))
    adjacency = coo_matrix(
        (np.ones(rows.shape[0], dtype=np.uint8), (rows, columns)),
        shape=(vertex_count, vertex_count),
    ).tocsr()
    _count, labels = connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    return torch.from_numpy(labels.astype(np.int64, copy=False))


def mouth_landmark_neighbors() -> dict[int, tuple[int, ...]]:
    neighbors: defaultdict[int, set[int]] = defaultdict(set)
    for curve in MOUTH_CURVES:
        for first, second in zip(curve[:-1], curve[1:]):
            neighbors[first].add(second)
            neighbors[second].add(first)
    return {
        mediapipe_id: tuple(sorted(values))
        for mediapipe_id, values in neighbors.items()
    }


def _head_vertex_ids(
    vertex_groups: Optional[Mapping[str, Any]],
    vertex_count: int,
) -> torch.Tensor:
    if not isinstance(vertex_groups, Mapping):
        return torch.empty(0, dtype=torch.long)
    sections = vertex_groups.get("sections")
    if not isinstance(sections, Mapping):
        return torch.empty(0, dtype=torch.long)
    value = sections.get("head")
    if not isinstance(value, torch.Tensor):
        return torch.empty(0, dtype=torch.long)
    ids = value.detach().cpu().long().flatten()
    ids = ids[(ids >= 0) & (ids < vertex_count)].unique(sorted=True)
    has_semantic_non_head_section = any(
        name != "head"
        and isinstance(section_ids, torch.Tensor)
        and section_ids.numel() > 0
        for name, section_ids in sections.items()
    )
    if (
        not has_semantic_non_head_section
        and ids.numel() >= int(vertex_count * 0.98)
    ):
        return torch.empty(0, dtype=torch.long)
    return ids


def _motion_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("target_motion", {})
    return value if isinstance(value, Mapping) else {}


def _repair_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("repair", {})
    return value if isinstance(value, Mapping) else {}


def _axis(value: Any) -> tuple[int, float]:
    name = str(value).strip().lower()
    sign = -1.0 if name.startswith("-") else 1.0
    axis_name = name.lstrip("-")
    indices = {"x": 0, "y": 1, "z": 2}
    if axis_name not in indices:
        raise ValueError(f"Unsupported mesh axis: {value!r}.")
    return indices[axis_name], sign


def _cpu_vertices(vertices: torch.Tensor) -> torch.Tensor:
    if not isinstance(vertices, torch.Tensor) or vertices.dim() != 2:
        raise ValueError("vertices must have shape [V, 3].")
    if vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape [V, 3].")
    return vertices.detach().cpu().float().contiguous()


def _cpu_faces(faces: torch.Tensor, vertex_count: int) -> torch.Tensor:
    if not isinstance(faces, torch.Tensor) or faces.dim() != 2:
        raise ValueError("faces must have shape [F, 3].")
    faces = faces.detach().cpu().long().contiguous()
    if faces.shape[-1] != 3:
        raise ValueError("faces must have shape [F, 3].")
    if faces.numel() and (int(faces.min()) < 0 or int(faces.max()) >= vertex_count):
        raise ValueError("faces contain out-of-range vertex IDs.")
    return faces


def _cpu_target_delta(
    target_delta: torch.Tensor,
    vertices: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(target_delta, torch.Tensor) or target_delta.shape != vertices.shape:
        raise ValueError("target deltas must match vertices [V, 3].")
    return target_delta.detach().cpu().float().contiguous()
