from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from utils.eyelid_landmarks import eyelid_landmarks_for_action_unit


VALIDATED_WELDED_LANDMARK_VERSION = 1

# These central-face landmarks are deliberately spread across the nose, cheeks,
# mouth, and eye corners. Their component votes identify facial surface fragments
# without assuming that the largest mesh component is skin.
DEFAULT_FACE_ANCHOR_IDS = (
    1,
    4,
    5,
    6,
    10,
    13,
    14,
    19,
    33,
    61,
    78,
    94,
    133,
    152,
    168,
    197,
    234,
    263,
    291,
    308,
    362,
    454,
)


@dataclass(frozen=True)
class EyelidMappingValidation:
    valid: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float]


def build_validated_welded_landmark_payload(
    *,
    person_id: str,
    raw_mapping: Mapping[int, int],
    welded_vertices: torch.Tensor,
    source_to_welded: torch.Tensor,
    component_ids: torch.Tensor,
    action_unit_ids: Sequence[int],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    vertices = _cpu_vertices(welded_vertices)
    source_map = source_to_welded.detach().cpu().long()
    components = component_ids.detach().cpu().long()
    if source_map.dim() != 1:
        raise ValueError("source_to_welded must have shape [source_vertex_count].")
    if components.shape != vertices.shape[:1]:
        raise ValueError("component_ids must have shape [welded_vertex_count].")

    source_vertex_count = int(source_map.numel())
    out_of_range = [
        (int(mediapipe_id), int(vertex_id))
        for mediapipe_id, vertex_id in raw_mapping.items()
        if int(vertex_id) < 0 or int(vertex_id) >= source_vertex_count
    ]
    if out_of_range:
        shown = ", ".join(f"{mid}:{vid}" for mid, vid in out_of_range[:5])
        raise ValueError(
            f"Raw landmark mapping contains source vertex IDs outside "
            f"[0, {source_vertex_count}): {shown}"
        )

    welded_mapping = {
        int(mediapipe_id): int(source_map[int(source_vertex_id)].item())
        for mediapipe_id, source_vertex_id in raw_mapping.items()
    }
    validation = validate_welded_eyelid_mapping(
        vertices=vertices,
        mapping=welded_mapping,
        action_unit_ids=action_unit_ids,
        config=config,
    )
    status = "valid"
    repair_metrics: dict[str, float] = {}
    if not validation.valid and bool(_repair_config(config).get("enabled", True)):
        repaired_mapping, repair_metrics = repair_welded_eyelid_mapping(
            vertices=vertices,
            component_ids=components,
            mapping=welded_mapping,
            action_unit_ids=action_unit_ids,
            config=config,
        )
        repaired_validation = validate_welded_eyelid_mapping(
            vertices=vertices,
            mapping=repaired_mapping,
            action_unit_ids=action_unit_ids,
            config=config,
        )
        if repaired_validation.valid:
            welded_mapping = repaired_mapping
            validation = repaired_validation
            status = "repaired"

    if not validation.valid:
        status = "invalid"

    return {
        "version": VALIDATED_WELDED_LANDMARK_VERSION,
        "person_id": str(person_id),
        "status": status,
        "reasons": list(validation.reasons),
        "metrics": {**validation.metrics, **repair_metrics},
        "mapping": {
            str(mediapipe_id): int(vertex_id)
            for mediapipe_id, vertex_id in sorted(welded_mapping.items())
        },
    }


def validate_welded_eyelid_mapping(
    *,
    vertices: torch.Tensor,
    mapping: Mapping[int, int],
    action_unit_ids: Sequence[int],
    config: Mapping[str, Any],
) -> EyelidMappingValidation:
    vertices = _cpu_vertices(vertices)
    mesh_extent = float((vertices.amax(dim=0) - vertices.amin(dim=0)).amax().item())
    mesh_extent = max(mesh_extent, 1.0e-12)
    min_unique = int(config.get("min_unique_vertices", 14))
    min_width_ratio = float(config.get("min_eye_width_mesh_extent_ratio", 0.01))
    min_gap_ratio = float(config.get("min_gap_eye_width_ratio", 0.05))
    max_gap_ratio = float(config.get("max_gap_eye_width_ratio", 0.75))
    max_adjacent_ratio = float(
        config.get("max_adjacent_eye_width_ratio", 1.25)
    )

    reasons: list[str] = []
    metrics: dict[str, float] = {}
    for action_unit_id in _unique_action_unit_ids(action_unit_ids):
        curves = eyelid_landmarks_for_action_unit(action_unit_id)
        prefix = f"au{action_unit_id}"
        missing = [
            mediapipe_id
            for mediapipe_id in curves.all_ids
            if int(mediapipe_id) not in mapping
        ]
        if missing:
            reasons.append(f"{prefix}:missing_landmarks")
            metrics[f"{prefix}_missing_landmark_count"] = float(len(missing))
            continue

        vertex_ids = torch.tensor(
            [int(mapping[int(mediapipe_id)]) for mediapipe_id in curves.all_ids],
            dtype=torch.long,
        )
        if bool(((vertex_ids < 0) | (vertex_ids >= vertices.shape[0])).any()):
            reasons.append(f"{prefix}:out_of_range_vertex")
            continue

        points = vertices.index_select(0, vertex_ids)
        index_by_id = {
            mediapipe_id: index
            for index, mediapipe_id in enumerate(curves.all_ids)
        }
        corner_indices = (
            index_by_id[curves.corners[0]],
            index_by_id[curves.corners[1]],
        )
        eye_width = float(
            torch.linalg.vector_norm(
                points[corner_indices[1]] - points[corner_indices[0]]
            ).item()
        )
        width_ratio = eye_width / mesh_extent
        unique_count = int(torch.unique(vertex_ids).numel())
        pair_gaps = torch.tensor(
            [
                torch.linalg.vector_norm(
                    points[index_by_id[upper_id]] - points[index_by_id[lower_id]]
                ).item()
                for upper_id, lower_id in curves.closure_pairs
            ],
            dtype=torch.float32,
        )
        mean_gap_ratio = float(pair_gaps.mean().item()) / max(eye_width, 1.0e-12)
        max_gap_to_width = float(pair_gaps.max().item()) / max(eye_width, 1.0e-12)
        adjacent_distances = []
        for curve_ids in (curves.upper, curves.lower):
            curve_points = points.index_select(
                0,
                torch.tensor(
                    [index_by_id[mediapipe_id] for mediapipe_id in curve_ids],
                    dtype=torch.long,
                ),
            )
            adjacent_distances.append(
                torch.linalg.vector_norm(curve_points[1:] - curve_points[:-1], dim=-1)
            )
        max_adjacent_to_width = float(
            torch.cat(adjacent_distances).max().item()
        ) / max(eye_width, 1.0e-12)

        metrics.update(
            {
                f"{prefix}_unique_vertex_count": float(unique_count),
                f"{prefix}_eye_width": eye_width,
                f"{prefix}_eye_width_mesh_extent_ratio": width_ratio,
                f"{prefix}_mean_gap_eye_width_ratio": mean_gap_ratio,
                f"{prefix}_max_gap_eye_width_ratio": max_gap_to_width,
                f"{prefix}_max_adjacent_eye_width_ratio": max_adjacent_to_width,
            }
        )
        if unique_count < min_unique:
            reasons.append(f"{prefix}:too_few_unique_vertices")
        if width_ratio < min_width_ratio:
            reasons.append(f"{prefix}:eye_width_too_small")
        if mean_gap_ratio < min_gap_ratio:
            reasons.append(f"{prefix}:eyelid_gap_too_small")
        if mean_gap_ratio > max_gap_ratio:
            reasons.append(f"{prefix}:eyelid_gap_too_large")
        if max_adjacent_to_width > max_adjacent_ratio:
            reasons.append(f"{prefix}:curve_discontinuity")

    return EyelidMappingValidation(
        valid=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        metrics=metrics,
    )


def repair_welded_eyelid_mapping(
    *,
    vertices: torch.Tensor,
    component_ids: torch.Tensor,
    mapping: Mapping[int, int],
    action_unit_ids: Sequence[int],
    config: Mapping[str, Any],
) -> tuple[dict[int, int], dict[str, float]]:
    try:
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError("Eyelid landmark repair requires scipy.") from exc

    vertices = _cpu_vertices(vertices)
    components = component_ids.detach().cpu().long()
    repaired = {int(key): int(value) for key, value in mapping.items()}
    repair_config = _repair_config(config)
    face_vertex_ids, face_component_count = facial_surface_vertex_ids(
        component_ids=components,
        mapping=repaired,
        config=config,
    )
    metrics = {
        "repair_face_component_count": float(face_component_count),
        "repair_face_vertex_count": float(face_vertex_ids.numel()),
    }
    if face_vertex_ids.numel() == 0:
        return repaired, metrics

    candidate_points = vertices.index_select(0, face_vertex_ids).numpy()
    tree = cKDTree(candidate_points)
    nearest_count = min(
        max(int(repair_config.get("nearest_candidates", 32)), 1),
        int(face_vertex_ids.numel()),
    )
    mesh_extent = max(
        float((vertices.amax(dim=0) - vertices.amin(dim=0)).amax().item()),
        1.0e-12,
    )
    max_distance = float(
        repair_config.get("max_distance_mesh_extent_ratio", 0.08)
    ) * mesh_extent
    repaired_count = 0
    maximum_distance = 0.0

    for action_unit_id in _unique_action_unit_ids(action_unit_ids):
        curves = eyelid_landmarks_for_action_unit(action_unit_id)
        if any(int(mediapipe_id) not in repaired for mediapipe_id in curves.all_ids):
            continue
        current_ids = torch.tensor(
            [repaired[int(mediapipe_id)] for mediapipe_id in curves.all_ids],
            dtype=torch.long,
        )
        if bool(((current_ids < 0) | (current_ids >= vertices.shape[0])).any()):
            continue
        source_points = vertices.index_select(0, current_ids).numpy()
        distances, local_indices = tree.query(
            source_points,
            k=nearest_count,
        )
        distances = np.asarray(distances)
        local_indices = np.asarray(local_indices)
        if nearest_count == 1:
            distances = distances[:, None]
            local_indices = local_indices[:, None]
        candidate_local_ids = np.unique(local_indices.reshape(-1))
        assignment_points = candidate_points[candidate_local_ids]
        costs = np.linalg.norm(
            source_points[:, None, :] - assignment_points[None, :, :],
            axis=-1,
        )
        row_ids, column_ids = linear_sum_assignment(costs)
        if len(row_ids) != len(curves.all_ids):
            continue
        assigned_distances = costs[row_ids, column_ids]
        if float(assigned_distances.max(initial=0.0)) > max_distance:
            continue
        assigned_vertex_ids = face_vertex_ids.numpy()[
            candidate_local_ids[column_ids]
        ]
        for row_id, vertex_id in zip(row_ids.tolist(), assigned_vertex_ids.tolist()):
            repaired[int(curves.all_ids[row_id])] = int(vertex_id)
        repaired_count += len(row_ids)
        maximum_distance = max(
            maximum_distance,
            float(assigned_distances.max(initial=0.0)),
        )

    metrics.update(
        {
            "repair_landmark_count": float(repaired_count),
            "repair_max_distance": maximum_distance,
            "repair_max_distance_mesh_extent_ratio": maximum_distance / mesh_extent,
        }
    )
    return repaired, metrics


def facial_surface_vertex_ids(
    *,
    component_ids: torch.Tensor,
    mapping: Mapping[int, int],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, int]:
    components = component_ids.detach().cpu().long()
    repair_config = _repair_config(config)
    anchor_ids = tuple(
        int(value)
        for value in config.get("facial_anchor_ids", DEFAULT_FACE_ANCHOR_IDS)
    )
    component_votes: Counter[int] = Counter()
    for mediapipe_id in anchor_ids:
        vertex_id = mapping.get(mediapipe_id)
        if vertex_id is None or int(vertex_id) < 0 or int(vertex_id) >= len(components):
            continue
        component_votes[int(components[int(vertex_id)].item())] += 1
    if not component_votes:
        return torch.empty(0, dtype=torch.long), 0

    minimum_votes = max(
        int(repair_config.get("min_component_anchor_votes", 2)),
        1,
    )
    maximum_components = max(int(repair_config.get("max_face_components", 8)), 1)
    ordered_components = sorted(
        component_votes,
        key=lambda component_id: (-component_votes[component_id], component_id),
    )
    selected = [
        component_id
        for component_id in ordered_components
        if component_votes[component_id] >= minimum_votes
    ][:maximum_components]
    if not selected:
        selected = ordered_components[:maximum_components]
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    surface_mask = (components[:, None] == selected_tensor[None, :]).any(dim=1)
    return torch.where(surface_mask)[0], len(selected)


def _repair_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("repair", {})
    return value if isinstance(value, Mapping) else {}


def _cpu_vertices(vertices: torch.Tensor) -> torch.Tensor:
    if not isinstance(vertices, torch.Tensor) or vertices.dim() != 2:
        raise ValueError("vertices must have shape [V, 3].")
    if vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape [V, 3].")
    return vertices.detach().cpu().float().contiguous()


def _unique_action_unit_ids(action_unit_ids: Sequence[int]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(value) for value in action_unit_ids))
