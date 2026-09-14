from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class EyeGazeLandmarkSpec:
    eye: str
    center_id: int
    iris_ids: tuple[int, ...]
    right_iris_id: int
    bottom_iris_id: int
    corner_ids: tuple[int, int]
    contour_ids: tuple[int, ...]


LEFT_EYE_GAZE_LANDMARKS = EyeGazeLandmarkSpec(
    eye="left",
    center_id=473,
    iris_ids=(473, 474, 475, 476, 477),
    right_iris_id=474,
    bottom_iris_id=477,
    corner_ids=(263, 362),
    contour_ids=(263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466),
)

RIGHT_EYE_GAZE_LANDMARKS = EyeGazeLandmarkSpec(
    eye="right",
    center_id=468,
    iris_ids=(468, 469, 470, 471, 472),
    right_iris_id=469,
    bottom_iris_id=472,
    corner_ids=(33, 133),
    contour_ids=(33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
)


EYE_GAZE_ACTION_UNIT_SPECS = {
    12: LEFT_EYE_GAZE_LANDMARKS,
    13: RIGHT_EYE_GAZE_LANDMARKS,
    14: LEFT_EYE_GAZE_LANDMARKS,
    15: RIGHT_EYE_GAZE_LANDMARKS,
    16: LEFT_EYE_GAZE_LANDMARKS,
    17: RIGHT_EYE_GAZE_LANDMARKS,
    18: LEFT_EYE_GAZE_LANDMARKS,
    19: RIGHT_EYE_GAZE_LANDMARKS,
}


def uv_connected_polygon_component(
    *,
    polygon_loop_indices: Sequence[Sequence[int]],
    loop_vertex_ids: np.ndarray,
    loop_uvs: np.ndarray,
    seed_polygon_index: int,
    candidate_polygons: np.ndarray | None = None,
    tolerance: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the UV-continuous polygon and loop component containing a seed.

    Polygon edges connect only when they share the same mesh edge and the UV at
    both edge endpoints agrees. This keeps UV seams and overlapping atlas
    islands disconnected.
    """

    loop_vertex_ids = np.asarray(loop_vertex_ids, dtype=np.int64)
    loop_uvs = np.asarray(loop_uvs, dtype=np.float64)
    polygon_count = len(polygon_loop_indices)
    if loop_vertex_ids.ndim != 1:
        raise ValueError("loop_vertex_ids must have shape [L].")
    if loop_uvs.shape != (loop_vertex_ids.shape[0], 2):
        raise ValueError("loop_uvs must have shape [L, 2].")
    if not np.isfinite(loop_uvs).all():
        raise ValueError("loop_uvs must be finite.")
    if not 0 <= int(seed_polygon_index) < polygon_count:
        raise ValueError("seed_polygon_index is outside polygon_loop_indices.")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")

    if candidate_polygons is None:
        candidates = np.ones(polygon_count, dtype=bool)
    else:
        candidates = np.asarray(candidate_polygons, dtype=bool)
        if candidates.shape != (polygon_count,):
            raise ValueError("candidate_polygons must have shape [P].")
    if not candidates[int(seed_polygon_index)]:
        raise ValueError("The seed polygon is not an enabled candidate.")

    quantized_uvs = np.rint(loop_uvs / float(tolerance)).astype(np.int64)
    polygon_edges: list[list[tuple[int, ...]] | None] = [None] * polygon_count
    edge_to_polygons: dict[tuple[int, ...], list[int]] = {}
    loop_count = loop_vertex_ids.shape[0]
    for polygon_index, raw_loop_indices in enumerate(polygon_loop_indices):
        if not candidates[polygon_index]:
            continue
        loops = tuple(int(value) for value in raw_loop_indices)
        if len(loops) < 3:
            continue
        if any(value < 0 or value >= loop_count for value in loops):
            raise ValueError("polygon_loop_indices contains an invalid loop index.")
        keys: list[tuple[int, ...]] = []
        for edge_index, first_loop in enumerate(loops):
            second_loop = loops[(edge_index + 1) % len(loops)]
            first_vertex = int(loop_vertex_ids[first_loop])
            second_vertex = int(loop_vertex_ids[second_loop])
            if first_vertex == second_vertex:
                continue
            if first_vertex < second_vertex:
                first_uv = quantized_uvs[first_loop]
                second_uv = quantized_uvs[second_loop]
                vertex_pair = (first_vertex, second_vertex)
            else:
                first_uv = quantized_uvs[second_loop]
                second_uv = quantized_uvs[first_loop]
                vertex_pair = (second_vertex, first_vertex)
            key = (
                vertex_pair[0],
                vertex_pair[1],
                int(first_uv[0]),
                int(first_uv[1]),
                int(second_uv[0]),
                int(second_uv[1]),
            )
            keys.append(key)
            edge_to_polygons.setdefault(key, []).append(polygon_index)
        polygon_edges[polygon_index] = keys

    seed = int(seed_polygon_index)
    if polygon_edges[seed] is None:
        raise ValueError("The seed polygon has no usable UV edges.")
    component_polygons = np.zeros(polygon_count, dtype=bool)
    component_polygons[seed] = True
    pending = [seed]
    while pending:
        polygon_index = pending.pop()
        for key in polygon_edges[polygon_index] or ():
            for neighbor in edge_to_polygons[key]:
                if not component_polygons[neighbor]:
                    component_polygons[neighbor] = True
                    pending.append(neighbor)

    component_loops = np.zeros(loop_count, dtype=bool)
    for polygon_index in np.flatnonzero(component_polygons):
        component_loops[np.asarray(polygon_loop_indices[int(polygon_index)], dtype=np.int64)] = True
    return component_polygons, component_loops


def eye_gaze_landmark_spec(action_unit_id: int) -> EyeGazeLandmarkSpec | None:
    return EYE_GAZE_ACTION_UNIT_SPECS.get(int(action_unit_id))


def semantic_eye_gaze_motion(
    action_unit_id: int,
    strength: float,
) -> torch.Tensor:
    """Return screen-space gaze motion in units of neutral eye width."""

    value = abs(float(strength))
    directions = {
        12: (0.0, value),
        13: (0.0, value),
        14: (-value, 0.0),
        15: (value, 0.0),
        16: (value, 0.0),
        17: (-value, 0.0),
        18: (0.0, -value),
        19: (0.0, -value),
    }
    if int(action_unit_id) not in directions:
        raise ValueError(f"AU{action_unit_id} is not an eye-gaze action unit.")
    return torch.tensor(directions[int(action_unit_id)], dtype=torch.float32)


def point_in_polygon_2d(
    point: Sequence[float],
    polygon: Sequence[Sequence[float]],
    *,
    tolerance: float = 1.0e-9,
) -> bool:
    point_array = np.asarray(point, dtype=np.float64)
    polygon_array = np.asarray(polygon, dtype=np.float64)
    if point_array.shape != (2,):
        raise ValueError("point must have shape [2].")
    if polygon_array.ndim != 2 or polygon_array.shape[1] != 2:
        raise ValueError("polygon must have shape [N, 2].")
    if len(polygon_array) < 3:
        raise ValueError("polygon must contain at least three points.")
    if not np.isfinite(point_array).all() or not np.isfinite(polygon_array).all():
        raise ValueError("point and polygon must be finite.")

    inside = False
    for index, first in enumerate(polygon_array):
        second = polygon_array[(index + 1) % len(polygon_array)]
        edge = second - first
        edge_length_squared = float(np.dot(edge, edge))
        if edge_length_squared > 0.0:
            parameter = float(
                np.clip(
                    np.dot(point_array - first, edge) / edge_length_squared,
                    0.0,
                    1.0,
                )
            )
            closest = first + edge * parameter
            if float(np.linalg.norm(point_array - closest)) <= tolerance:
                return True
        if (first[1] > point_array[1]) == (second[1] > point_array[1]):
            continue
        crossing_x = first[0] + (
            (point_array[1] - first[1])
            * (second[0] - first[0])
            / (second[1] - first[1])
        )
        if point_array[0] < crossing_x:
            inside = not inside
    return inside


def tapered_eye_uv_warp(
    uv_coordinates: torch.Tensor,
    *,
    center: torch.Tensor,
    horizontal_axis: torch.Tensor,
    vertical_axis: torch.Tensor,
    horizontal_radius: float,
    vertical_radius: float,
    visual_shift: torch.Tensor,
    active_mask: torch.Tensor | None = None,
    inner_radius: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp UV sampling so the visible texture moves by ``visual_shift``.

    UVs move opposite the desired visible motion. The shift is constant in the
    center and tapers smoothly to zero at the eye boundary.
    """

    uv_coordinates = torch.as_tensor(uv_coordinates, dtype=torch.float32)
    center = torch.as_tensor(center, dtype=uv_coordinates.dtype)
    horizontal_axis = torch.as_tensor(horizontal_axis, dtype=uv_coordinates.dtype)
    vertical_axis = torch.as_tensor(vertical_axis, dtype=uv_coordinates.dtype)
    visual_shift = torch.as_tensor(visual_shift, dtype=uv_coordinates.dtype)
    if uv_coordinates.dim() != 2 or uv_coordinates.shape[-1] != 2:
        raise ValueError("uv_coordinates must have shape [N, 2].")
    for name, value in (
        ("center", center),
        ("horizontal_axis", horizontal_axis),
        ("vertical_axis", vertical_axis),
        ("visual_shift", visual_shift),
    ):
        if value.shape != (2,):
            raise ValueError(f"{name} must have shape [2].")
    if horizontal_radius <= 0.0 or vertical_radius <= 0.0:
        raise ValueError("Eye UV warp radii must be positive.")
    if not 0.0 <= inner_radius < 1.0:
        raise ValueError("inner_radius must be in [0, 1).")

    horizontal_axis = horizontal_axis / torch.linalg.vector_norm(
        horizontal_axis
    ).clamp_min(1.0e-8)
    vertical_axis = vertical_axis - horizontal_axis * torch.dot(
        vertical_axis, horizontal_axis
    )
    vertical_axis = vertical_axis / torch.linalg.vector_norm(vertical_axis).clamp_min(
        1.0e-8
    )
    relative = uv_coordinates - center
    horizontal = relative.matmul(horizontal_axis) / float(horizontal_radius)
    vertical = relative.matmul(vertical_axis) / float(vertical_radius)
    radius = torch.sqrt(horizontal.square() + vertical.square())
    transition = ((1.0 - radius) / (1.0 - float(inner_radius))).clamp(0.0, 1.0)
    weights = transition.square() * (3.0 - 2.0 * transition)
    if active_mask is not None:
        active_mask = torch.as_tensor(active_mask, dtype=torch.bool)
        if active_mask.shape != (uv_coordinates.shape[0],):
            raise ValueError("active_mask must have shape [N].")
        weights = weights * active_mask.to(dtype=weights.dtype)
    warped = uv_coordinates - weights.unsqueeze(-1) * visual_shift
    return warped, weights
