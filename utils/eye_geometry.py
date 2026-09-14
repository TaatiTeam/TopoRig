from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import torch


EYE_CAPABILITY_TEXTURE_ONLY = 0
EYE_CAPABILITY_RIGID_MESH = 1
EYE_CAPABILITY_LAYERED_MESH = 2


@dataclass(frozen=True)
class EyeComponentMetrics:
    component_id: int
    vertex_ids: np.ndarray
    face_ids: np.ndarray
    vertex_count: int
    face_count: int
    surface_distance: float
    surface_distance_ratio: float
    bbox_center: np.ndarray
    bbox_extent: np.ndarray
    sphere_center: np.ndarray
    sphere_radius: float
    sphere_residual_ratio: float
    ellipsoid_center: np.ndarray
    ellipsoid_axes: np.ndarray
    ellipsoid_residual_ratio: float
    ellipsoid_axis_ratio: float
    closedness: float
    radius_eye_width_ratio: float
    iris_surface_residual_ratio: float
    behind_iris_eye_widths: float
    bilateral_score: float
    score: float
    physically_plausible: bool


@dataclass(frozen=True)
class EyeAssemblySelection:
    capability: int
    pivot_component_id: int
    anchor_component_id: int
    assembly_component_ids: tuple[int, ...]
    assembly_vertex_ids: np.ndarray
    rotation_center: np.ndarray
    rotation_radius: float
    confidence: float
    score: float
    rejection_reason: str
    candidates: tuple[EyeComponentMetrics, ...]

    @property
    def is_physical(self) -> bool:
        return self.capability in {
            EYE_CAPABILITY_RIGID_MESH,
            EYE_CAPABILITY_LAYERED_MESH,
        }


def as_numpy(value: np.ndarray | torch.Tensor | Sequence[float]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def fit_sphere(
    points: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, float, float]:
    values = as_numpy(points).astype(np.float64, copy=False)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 4:
        return np.full(3, np.nan), float("nan"), float("inf")
    matrix = np.concatenate(
        (2.0 * values, np.ones((values.shape[0], 1), dtype=np.float64)),
        axis=1,
    )
    try:
        solution, _residuals, rank, _singular_values = np.linalg.lstsq(
            matrix,
            np.sum(values * values, axis=1),
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return np.full(3, np.nan), float("nan"), float("inf")
    if rank < 4:
        return np.full(3, np.nan), float("nan"), float("inf")
    center = solution[:3]
    radii = np.linalg.norm(values - center, axis=1)
    radius = float(np.median(radii))
    if not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 1.0e-10:
        return np.full(3, np.nan), float("nan"), float("inf")
    residual_ratio = float(
        np.sqrt(np.mean(np.square(radii - radius))) / max(radius, 1.0e-10)
    )
    return center, radius, residual_ratio


def fit_ellipsoid(
    points: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit a general ellipsoid and return center, semi-axes and radial error.

    The algebraic fit is intentionally used only as a secondary shape cue. A
    positive-definite quadratic is required, so planes and open facial patches
    fail closed instead of being mistaken for an eyeball.
    """

    values = as_numpy(points).astype(np.float64, copy=False)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 9:
        return (
            np.full(3, np.nan),
            np.full(3, np.nan),
            float("inf"),
            float("inf"),
        )
    x, y, z = values.T
    design = np.stack(
        (
            x * x,
            y * y,
            z * z,
            2.0 * x * y,
            2.0 * x * z,
            2.0 * y * z,
            x,
            y,
            z,
        ),
        axis=1,
    )
    try:
        coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(
            design,
            np.ones(values.shape[0], dtype=np.float64),
            rcond=None,
        )
    except np.linalg.LinAlgError:
        rank = 0
    if rank < 9:
        return (
            np.full(3, np.nan),
            np.full(3, np.nan),
            float("inf"),
            float("inf"),
        )
    a, b, c, xy, xz, yz, dx, dy, dz = coefficients
    quadratic = np.asarray(
        ((a, xy, xz), (xy, b, yz), (xz, yz, c)),
        dtype=np.float64,
    )
    linear = np.asarray((dx, dy, dz), dtype=np.float64)
    try:
        center = -0.5 * np.linalg.solve(quadratic, linear)
        centered_constant = 1.0 + float(center @ quadratic @ center)
        normalized_quadratic = quadratic / centered_constant
        eigenvalues, eigenvectors = np.linalg.eigh(normalized_quadratic)
    except np.linalg.LinAlgError:
        return (
            np.full(3, np.nan),
            np.full(3, np.nan),
            float("inf"),
            float("inf"),
        )
    if (
        not np.isfinite(center).all()
        or not np.isfinite(eigenvalues).all()
        or centered_constant <= 0.0
        or float(eigenvalues.min()) <= 1.0e-12
    ):
        return (
            np.full(3, np.nan),
            np.full(3, np.nan),
            float("inf"),
            float("inf"),
        )
    axes = 1.0 / np.sqrt(eigenvalues)
    local = (values - center) @ eigenvectors
    normalized_radius = np.sqrt(np.sum(np.square(local / axes), axis=1))
    residual_ratio = float(np.sqrt(np.mean(np.square(normalized_radius - 1.0))))
    axis_ratio = float(axes.max() / max(float(axes.min()), 1.0e-10))
    return center, axes, residual_ratio, axis_ratio


def component_closedness(faces: np.ndarray | torch.Tensor) -> float:
    triangles = as_numpy(faces).astype(np.int64, copy=False)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] == 0:
        return 0.0
    edges = np.concatenate(
        (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]),
        axis=0,
    )
    edges.sort(axis=1)
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    if counts.size == 0:
        return 0.0
    boundary_ratio = float(np.count_nonzero(counts == 1) / counts.size)
    nonmanifold_ratio = float(np.count_nonzero(counts > 2) / counts.size)
    return float(np.clip(1.0 - boundary_ratio - nonmanifold_ratio, 0.0, 1.0))


def _safe_log_ratio(value: float, reference: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return 8.0
    return abs(math.log(value / reference))


def _component_surface_distance(
    component_vertices: np.ndarray,
    iris_points: np.ndarray,
) -> float:
    best = float("inf")
    chunk_size = 20_000
    for start in range(0, component_vertices.shape[0], chunk_size):
        chunk = component_vertices[start : start + chunk_size]
        distances = np.linalg.norm(
            chunk[:, None, :] - iris_points[None, :, :],
            axis=-1,
        )
        best = min(best, float(distances.min()))
    return best


def select_eye_assembly(
    *,
    vertices: np.ndarray | torch.Tensor,
    faces: np.ndarray | torch.Tensor,
    component_ids: np.ndarray | torch.Tensor,
    iris_points: np.ndarray | torch.Tensor,
    eye_width: float,
    largest_component_id: int | None = None,
    surface_distances: Mapping[int, float] | None = None,
    max_surface_distance_eye_widths: float = 0.45,
    min_component_vertices: int = 64,
    max_component_fraction: float = 0.30,
    min_radius_eye_widths: float = 0.12,
    max_radius_eye_widths: float = 1.25,
    max_sphere_residual_ratio: float = 0.32,
    max_ellipsoid_residual_ratio: float = 0.24,
    max_ellipsoid_axis_ratio: float = 2.5,
    layer_center_distance_radii: float = 1.35,
) -> EyeAssemblySelection:
    vertices_np = as_numpy(vertices).astype(np.float64, copy=False)
    faces_np = as_numpy(faces).astype(np.int64, copy=False)
    component_ids_np = as_numpy(component_ids).astype(np.int64, copy=False).reshape(-1)
    iris_np = as_numpy(iris_points).astype(np.float64, copy=False)
    if vertices_np.ndim != 2 or vertices_np.shape[1] != 3:
        raise ValueError("vertices must have shape [V, 3].")
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3].")
    if component_ids_np.shape != (vertices_np.shape[0],):
        raise ValueError("component_ids must contain one ID per vertex.")
    if iris_np.ndim != 2 or iris_np.shape[1] != 3 or iris_np.shape[0] < 1:
        raise ValueError("iris_points must have shape [L, 3].")
    if not math.isfinite(float(eye_width)) or float(eye_width) <= 1.0e-8:
        raise ValueError("eye_width must be positive and finite.")

    eye_width = float(eye_width)
    unique_ids, counts = np.unique(component_ids_np, return_counts=True)
    if largest_component_id is None:
        largest_component_id = int(unique_ids[int(np.argmax(counts))])
    largest_component_id = int(largest_component_id)
    iris_center = iris_np.mean(axis=0)
    head_vertices = vertices_np[component_ids_np == largest_component_id]
    head_center = (
        np.median(head_vertices, axis=0)
        if head_vertices.size
        else np.median(vertices_np, axis=0)
    )
    outward = iris_center - head_center
    outward_norm = float(np.linalg.norm(outward))
    if outward_norm <= 1.0e-8:
        outward = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    else:
        outward /= outward_norm

    valid_faces = (
        (faces_np >= 0).all(axis=1)
        & (faces_np < vertices_np.shape[0]).all(axis=1)
        & (faces_np[:, 0] != faces_np[:, 1])
        & (faces_np[:, 1] != faces_np[:, 2])
        & (faces_np[:, 2] != faces_np[:, 0])
    )
    face_component_ids = np.full(faces_np.shape[0], -1, dtype=np.int64)
    face_component_ids[valid_faces] = component_ids_np[faces_np[valid_faces, 0]]
    total_vertices = max(vertices_np.shape[0], 1)
    raw_records: list[dict[str, object]] = []
    surface_distance_map = dict(surface_distances or {})

    for component_id, vertex_count_value in zip(unique_ids.tolist(), counts.tolist()):
        component_id = int(component_id)
        vertex_ids = np.flatnonzero(component_ids_np == component_id)
        face_ids = np.flatnonzero(valid_faces & (face_component_ids == component_id))
        component_vertices = vertices_np[vertex_ids]
        component_faces = faces_np[face_ids]
        if component_vertices.size == 0:
            continue
        bbox_min = component_vertices.min(axis=0)
        bbox_max = component_vertices.max(axis=0)
        bbox_center = (bbox_min + bbox_max) * 0.5
        bbox_extent = bbox_max - bbox_min
        sphere_center, sphere_radius, sphere_residual = fit_sphere(component_vertices)
        ellipsoid_center, ellipsoid_axes, ellipsoid_residual, ellipsoid_axis_ratio = (
            fit_ellipsoid(component_vertices)
        )
        surface_distance = float(
            surface_distance_map.get(
                component_id,
                _component_surface_distance(component_vertices, iris_np),
            )
        )
        radius_ratio = sphere_radius / eye_width if math.isfinite(sphere_radius) else float("inf")
        if np.isfinite(sphere_center).all() and math.isfinite(sphere_radius):
            iris_surface_residual = abs(
                float(np.linalg.norm(iris_center - sphere_center)) - sphere_radius
            ) / max(sphere_radius, 1.0e-10)
            behind = float(np.dot(iris_center - sphere_center, outward) / eye_width)
        else:
            iris_surface_residual = float("inf")
            behind = float("-inf")
        raw_records.append(
            {
                "component_id": component_id,
                "vertex_ids": vertex_ids,
                "face_ids": face_ids,
                "vertex_count": int(vertex_count_value),
                "face_count": int(face_ids.size),
                "surface_distance": surface_distance,
                "surface_distance_ratio": surface_distance / eye_width,
                "bbox_center": bbox_center,
                "bbox_extent": bbox_extent,
                "sphere_center": sphere_center,
                "sphere_radius": sphere_radius,
                "sphere_residual_ratio": sphere_residual,
                "ellipsoid_center": ellipsoid_center,
                "ellipsoid_axes": ellipsoid_axes,
                "ellipsoid_residual_ratio": ellipsoid_residual,
                "ellipsoid_axis_ratio": ellipsoid_axis_ratio,
                "closedness": component_closedness(component_faces),
                "radius_eye_width_ratio": radius_ratio,
                "iris_surface_residual_ratio": iris_surface_residual,
                "behind_iris_eye_widths": behind,
            }
        )

    head_midline_x = float((vertices_np[:, 0].min() + vertices_np[:, 0].max()) * 0.5)
    scored: list[EyeComponentMetrics] = []
    for record in raw_records:
        component_id = int(record["component_id"])
        center = np.asarray(record["sphere_center"])
        radius = float(record["sphere_radius"])
        bilateral_score = 2.0
        if np.isfinite(center).all() and math.isfinite(radius):
            mirrored = center.copy()
            mirrored[0] = 2.0 * head_midline_x - mirrored[0]
            partner_scores = []
            for other in raw_records:
                if int(other["component_id"]) == component_id:
                    continue
                other_center = np.asarray(other["sphere_center"])
                other_radius = float(other["sphere_radius"])
                if not np.isfinite(other_center).all() or not math.isfinite(other_radius):
                    continue
                if (center[0] - head_midline_x) * (other_center[0] - head_midline_x) >= 0.0:
                    continue
                partner_scores.append(
                    float(np.linalg.norm(other_center - mirrored) / eye_width)
                    + _safe_log_ratio(other_radius, radius)
                    + 0.25
                    * _safe_log_ratio(
                        float(other["vertex_count"]),
                        float(record["vertex_count"]),
                    )
                )
            if partner_scores:
                bilateral_score = min(partner_scores)

        surface_ratio = float(record["surface_distance_ratio"])
        sphere_residual = float(record["sphere_residual_ratio"])
        ellipsoid_residual = float(record["ellipsoid_residual_ratio"])
        ellipsoid_axis_ratio = float(record["ellipsoid_axis_ratio"])
        radius_ratio = float(record["radius_eye_width_ratio"])
        iris_surface_residual = float(record["iris_surface_residual_ratio"])
        behind = float(record["behind_iris_eye_widths"])
        closedness = float(record["closedness"])
        vertex_count = int(record["vertex_count"])
        component_fraction = vertex_count / total_vertices
        shape_score = min(
            sphere_residual / max(max_sphere_residual_ratio, 1.0e-8),
            ellipsoid_residual / max(max_ellipsoid_residual_ratio, 1.0e-8)
            + 0.2 * max(ellipsoid_axis_ratio - 1.0, 0.0),
        )
        radius_score = _safe_log_ratio(radius_ratio, 0.5)
        behind_score = (
            abs(behind - radius_ratio)
            if math.isfinite(behind) and math.isfinite(radius_ratio)
            else 4.0
        )
        score = (
            3.0 * surface_ratio
            + 2.5 * shape_score
            + 1.25 * radius_score
            + 1.5 * min(iris_surface_residual, 4.0)
            + 0.75 * min(behind_score, 4.0)
            + 1.0 * (1.0 - closedness)
            + 0.75 * min(bilateral_score, 3.0)
        )
        physically_plausible = bool(
            component_id != largest_component_id
            and vertex_count >= int(min_component_vertices)
            and int(record["face_count"]) >= 8
            and component_fraction <= float(max_component_fraction)
            and surface_ratio <= float(max_surface_distance_eye_widths)
            and min_radius_eye_widths <= radius_ratio <= max_radius_eye_widths
            and iris_surface_residual <= 0.75
            and behind > -0.10
            and (
                sphere_residual <= max_sphere_residual_ratio
                or (
                    ellipsoid_residual <= max_ellipsoid_residual_ratio
                    and ellipsoid_axis_ratio <= max_ellipsoid_axis_ratio
                )
            )
        )
        if component_id == largest_component_id:
            score = float("inf")
        scored.append(
            EyeComponentMetrics(
                component_id=component_id,
                vertex_ids=np.asarray(record["vertex_ids"], dtype=np.int64),
                face_ids=np.asarray(record["face_ids"], dtype=np.int64),
                vertex_count=vertex_count,
                face_count=int(record["face_count"]),
                surface_distance=float(record["surface_distance"]),
                surface_distance_ratio=surface_ratio,
                bbox_center=np.asarray(record["bbox_center"], dtype=np.float64),
                bbox_extent=np.asarray(record["bbox_extent"], dtype=np.float64),
                sphere_center=center.astype(np.float64, copy=False),
                sphere_radius=radius,
                sphere_residual_ratio=sphere_residual,
                ellipsoid_center=np.asarray(record["ellipsoid_center"], dtype=np.float64),
                ellipsoid_axes=np.asarray(record["ellipsoid_axes"], dtype=np.float64),
                ellipsoid_residual_ratio=ellipsoid_residual,
                ellipsoid_axis_ratio=ellipsoid_axis_ratio,
                closedness=closedness,
                radius_eye_width_ratio=radius_ratio,
                iris_surface_residual_ratio=iris_surface_residual,
                behind_iris_eye_widths=behind,
                bilateral_score=bilateral_score,
                score=score,
                physically_plausible=physically_plausible,
            )
        )

    plausible = [record for record in scored if record.physically_plausible]
    if not plausible:
        nearby = [
            record
            for record in scored
            if record.component_id != largest_component_id
            and record.surface_distance_ratio <= max_surface_distance_eye_widths
        ]
        reason = (
            "nearby_components_failed_physical_shape_checks"
            if nearby
            else "no_independent_eye_component_near_iris"
        )
        return EyeAssemblySelection(
            capability=EYE_CAPABILITY_TEXTURE_ONLY,
            pivot_component_id=-1,
            anchor_component_id=largest_component_id,
            assembly_component_ids=(),
            assembly_vertex_ids=np.empty(0, dtype=np.int64),
            rotation_center=np.full(3, np.nan),
            rotation_radius=float("nan"),
            confidence=0.0,
            score=float("inf"),
            rejection_reason=reason,
            candidates=tuple(sorted(scored, key=lambda value: value.score)),
        )

    pivot = min(plausible, key=lambda value: value.score)
    anchor_candidates = [
        record
        for record in scored
        if record.component_id != largest_component_id
        and record.surface_distance_ratio <= max_surface_distance_eye_widths
        and record.vertex_count >= max(4, int(min_component_vertices) // 4)
        and record.face_count >= 8
    ]
    linked: list[EyeComponentMetrics] = []
    for record in anchor_candidates:
        bbox_center_distance = float(
            np.linalg.norm(record.bbox_center - pivot.sphere_center)
        )
        sphere_center_distance = (
            float(np.linalg.norm(record.sphere_center - pivot.sphere_center))
            if np.isfinite(record.sphere_center).all()
            else float("inf")
        )
        near_shared_center = min(bbox_center_distance, sphere_center_distance) <= (
            pivot.sphere_radius * float(layer_center_distance_radii)
        )
        near_iris = record.surface_distance_ratio <= max_surface_distance_eye_widths
        not_head_sized = record.vertex_count / total_vertices <= max_component_fraction
        if near_shared_center and near_iris and not_head_sized:
            linked.append(record)
    if not any(value.component_id == pivot.component_id for value in linked):
        linked.append(pivot)
    linked.sort(key=lambda value: (value.surface_distance, value.component_id))
    component_ids_out = tuple(value.component_id for value in linked)
    assembly_vertex_ids = np.unique(
        np.concatenate([value.vertex_ids for value in linked])
    ).astype(np.int64, copy=False)
    anchor = min(linked, key=lambda value: value.surface_distance)
    confidence = float(np.clip(math.exp(-pivot.score / 7.5), 0.0, 1.0))
    capability = (
        EYE_CAPABILITY_LAYERED_MESH
        if len(component_ids_out) > 1
        else EYE_CAPABILITY_RIGID_MESH
    )
    return EyeAssemblySelection(
        capability=capability,
        pivot_component_id=pivot.component_id,
        anchor_component_id=anchor.component_id,
        assembly_component_ids=component_ids_out,
        assembly_vertex_ids=assembly_vertex_ids,
        rotation_center=pivot.sphere_center.copy(),
        rotation_radius=float(pivot.sphere_radius),
        confidence=confidence,
        score=float(pivot.score),
        rejection_reason="",
        candidates=tuple(sorted(scored, key=lambda value: value.score)),
    )
