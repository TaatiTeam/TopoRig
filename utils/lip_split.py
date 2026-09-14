from __future__ import annotations

import json
import math
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


INNER_UPPER_LIP_MEDIAPIPE_IDS = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
INNER_LOWER_LIP_MEDIAPIPE_IDS = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308)
OUTER_UPPER_LIP_MEDIAPIPE_IDS = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)
OUTER_LOWER_LIP_MEDIAPIPE_IDS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291)

# Corresponding points along the closed inner-mouth contact line.  Averaging
# each upper/lower pair is more stable than choosing either semantic mask on a
# neutral mesh whose lips touch or share vertices.
INNER_LIP_CENTER_PAIRS = (
    (78, 78),
    (191, 95),
    (80, 88),
    (81, 178),
    (82, 87),
    (13, 14),
    (312, 317),
    (311, 402),
    (310, 318),
    (415, 324),
    (308, 308),
)


@dataclass(frozen=True)
class LipRegions:
    upper_vertex_ids: torch.Tensor
    lower_vertex_ids: torch.Tensor
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "upper_vertex_count": int(self.upper_vertex_ids.numel()),
            "lower_vertex_count": int(self.lower_vertex_ids.numel()),
            "upper_vertex_ids": self.upper_vertex_ids.detach().cpu().long().tolist(),
            "lower_vertex_ids": self.lower_vertex_ids.detach().cpu().long().tolist(),
        }


@dataclass(frozen=True)
class LipSplitReport:
    original_vertex_count: int
    original_face_count: int
    final_vertex_count: int
    final_face_count: int
    upper_vertex_count: int
    lower_vertex_count: int
    shared_vertex_count: int
    upper_face_count: int
    lower_face_count: int
    neutral_face_count: int
    mixed_face_count: int
    removed_face_count: int
    duplicated_vertex_count: int
    direct_lip_connection_faces_before: int
    direct_lip_connection_edges_before: int
    direct_lip_connection_faces_after: int
    direct_lip_connection_edges_after: int
    lower_duplicate_vertex_ids: dict[int, int]
    removed_face_ids: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeometricLipSeamReport:
    original_vertex_count: int
    original_face_count: int
    final_vertex_count: int
    final_face_count: int
    curve_point_count: int
    candidate_face_count: int
    upper_candidate_face_count: int
    lower_candidate_face_count: int
    seam_vertex_count: int
    duplicated_vertex_count: int
    remapped_lower_face_count: int
    mixed_original_duplicate_face_count: int
    horizontal_axis: int
    depth_axis: int
    vertical_axis: int
    depth_tolerance: float
    vertical_tolerance: float
    horizontal_padding: float
    corner_inset_ratio: float
    preopen_distance: float
    curve_points: list[list[float]]
    seam_vertex_ids: list[int]
    lower_duplicate_vertex_ids: dict[int, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoincidentVertexWeldReport:
    original_vertex_count: int
    final_vertex_count: int
    merged_vertex_count: int
    original_face_count: int
    final_face_count: int
    removed_degenerate_face_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LipLandmarkTopologyRepairReport:
    input_landmark_count: int
    lip_landmark_count: int
    valid_lip_landmark_count: int
    invalid_lip_landmark_count: int
    connected_component_count: int
    dominant_component_vertex_count: int
    off_component_lip_landmark_count: int
    spatial_outlier_lip_landmark_count: int
    snapped_lip_landmark_count: int
    interpolated_lip_landmark_count: int
    skipped_lip_landmark_count: int
    maximum_snap_distance: float
    mean_snap_distance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LipLandmarkSurfaceSnapReport:
    input_landmark_count: int
    lip_landmark_count: int
    connected_component_count: int
    facial_component_id: int
    facial_component_vertex_count: int
    facial_component_landmark_votes: int
    off_component_lip_landmark_count: int
    duplicate_lip_landmark_count_before: int
    duplicate_lip_landmark_count_after: int
    snapped_lip_landmark_count: int
    skipped_lip_landmark_count: int
    maximum_projected_distance_ratio: float
    mean_projected_distance_ratio: float
    minimum_upper_lower_gap: float
    front_axis: int
    front_direction: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JawOpenApertureCutReport:
    aperture_scale: float
    overlap_inset_scale: float
    depth_padding: float
    front_direction: int
    polygon_point_count: int
    inside_aperture_face_count: int
    overlap_aperture_face_count: int
    removed_face_count: int
    original_face_count: int
    final_face_count: int
    aperture_polygon: list[list[float]]
    removed_face_ids: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MouthCavityReport:
    vertex_count: int
    face_count: int
    segment_count: int
    horizontal_axis: int
    depth_axis: int
    vertical_axis: int
    front_direction: int
    scale: float
    shape_exponent: float
    depth_offset: float
    radial_ring_count: int
    curvature_depth: float
    inner_radius_ratio: float
    side_wall_min_horizontal_ratio: float
    center: list[float]
    radius_horizontal: float
    radius_vertical: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MouthBoundaryLiningReport:
    vertex_count: int
    face_count: int
    boundary_cycle_count: int
    boundary_vertex_count: int
    depth: float
    front_direction: int
    side_fraction: float
    inner_scale: float
    radial_ring_count: int
    smoothing_iterations: int
    analytic_segments: int | None
    opening_scale: float
    shape_exponent: float
    boundary_source: str
    cap_back: bool
    maximum_cycles: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StretchedMouthFaceReport:
    edge_ratio_threshold: float
    edge_growth_threshold: float
    neighbor_growth_rings: int
    neighbor_edge_ratio_threshold: float
    neighbor_edge_growth_threshold: float
    region_scale: float
    depth_padding: float
    candidate_face_count: int
    strict_removed_face_count: int
    grown_removed_face_count: int
    removed_face_count: int
    removed_face_ids: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MouthRemovalConnectivityReport:
    aperture_seed_face_count: int
    stretch_candidate_face_count: int
    candidate_face_count: int
    connected_component_count: int
    seeded_component_count: int
    significant_seeded_component_count: int
    discarded_aperture_component_count: int
    discarded_aperture_face_count: int
    removed_face_count: int
    restored_disconnected_component_count: int
    restored_disconnected_face_count: int
    removed_face_ids: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MouthContourClipReport:
    source_face_count: int
    input_kept_face_count: int
    final_face_count: int
    original_vertex_count: int
    final_vertex_count: int
    clipped_face_count: int
    restored_boundary_face_count: int
    removed_inside_face_count: int
    retained_outside_face_count: int
    added_vertex_count: int
    smooth_curve_sample_count: int
    contour_scale: float
    depth_padding: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LipAnnulusRepairReport:
    vertex_count: int
    face_count: int
    segment_count: int
    radial_ring_count: int
    outer_scale: float
    thickness_ratio: float
    side_thickness_factor: float
    maximum_radial_inset_fraction: float
    inner_recess_depth: float
    front_offset: float
    bulge_depth: float
    capped_neutral_segment_count: int
    capped_deformed_segment_count: int
    upper_smoothing_window: int
    lower_smoothing_window: int
    front_direction: int
    horizontal_axis: int
    depth_axis: int
    vertical_axis: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrontSurfaceOrientationReport:
    face_count: int
    eligible_face_count: int
    flipped_face_count: int
    front_direction: int
    mouth_depth: float
    front_depth_limit: float
    region_depth_ratio: float
    normal_tolerance: float
    depth_axis: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LipAnnulusHeadCleanupReport:
    original_face_count: int
    removed_face_count: int
    final_face_count: int
    projected_overlap_face_count: int
    depth_rejected_overlap_face_count: int
    polygon_point_count: int
    depth_padding_ratio: float
    depth_min: float
    depth_max: float
    horizontal_axis: int
    depth_axis: int
    vertical_axis: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MouthBoundarySnapReport:
    boundary_edge_count: int
    mouth_boundary_edge_count: int
    mouth_boundary_component_count: int
    snapped_component_count: int
    snapped_vertex_count: int
    blended_vertex_count: int
    blend_rings: int
    contour_point_count: int
    max_projected_displacement: float
    mean_projected_displacement: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MouthBoundarySmoothReport:
    boundary_edge_count: int
    mouth_boundary_edge_count: int
    mouth_boundary_component_count: int
    smoothed_component_count: int
    smoothed_boundary_vertex_count: int
    blended_vertex_count: int
    blend_rings: int
    smoothing_cycles: int
    maximum_displacement: float
    mean_boundary_displacement: float
    max_boundary_displacement: float
    quality_rejected_face_count: int
    quality_restored_vertex_count: int
    preserve_boundary_junctions: bool
    preserved_boundary_junction_count: int
    arc_length_weighted: bool
    fragmented_selection_enabled: bool
    minimum_component_span: float | None
    minimum_component_vertices: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MouthHoleFillReport:
    boundary_edge_count_before: int
    reference_boundary_edge_count: int
    new_boundary_edge_count: int
    mouth_boundary_edge_count_before: int
    mouth_cycle_count: int
    preexisting_cycle_count: int
    preserved_aperture_cycle_count: int
    candidate_hole_cycle_count: int
    filled_hole_cycle_count: int
    rejected_hole_cycle_count: int
    added_face_count: int
    added_vertex_count: int
    reoriented_face_count: int
    boundary_edge_count_after: int
    maximum_area_ratio: float
    minimum_new_boundary_edge_fraction: float
    largest_cycle_area: float
    filled_cycle_areas: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MouthCornerBridgeReport:
    boundary_edge_count_before: int
    boundary_edge_count_after: int
    mouth_boundary_component_count: int
    candidate_side_count: int
    repaired_sides: list[str]
    added_face_count: int
    reoriented_face_count: int
    severe_added_face_count: int
    flipped_added_face_count: int
    corner_inset_ratio: float
    chain_padding_vertices: int
    minimum_lateral_overshoot_ratio: float
    lateral_overshoot_ratios: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_jaw_open_aperture_faces(
    deformed_vertices: torch.Tensor,
    faces: torch.Tensor,
    mapping: Mapping[int, int],
    *,
    horizontal_axis: int = 0,
    depth_axis: int = 1,
    vertical_axis: int = 2,
    aperture_scale: float = 1.0,
    overlap_inset_scale: float = 0.9,
    depth_padding: float = 0.01,
) -> JawOpenApertureCutReport:
    """Find front-surface faces covering the opened inner-lip aperture."""

    deformed_vertices = deformed_vertices.detach().cpu().float().contiguous()
    faces = faces.detach().cpu().long().contiguous()
    axes = (int(horizontal_axis), int(depth_axis), int(vertical_axis))
    if sorted(axes) != [0, 1, 2]:
        raise ValueError(
            "horizontal_axis, depth_axis, and vertical_axis must be distinct "
            "coordinate indices in [0, 2]."
        )
    if aperture_scale <= 0.0:
        raise ValueError("Jaw-open aperture scale must be positive.")
    if not 0.0 < overlap_inset_scale <= 1.0:
        raise ValueError("Jaw-open overlap inset scale must be in (0, 1].")
    if depth_padding < 0.0:
        raise ValueError("Jaw-open aperture depth padding cannot be negative.")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError("detect_jaw_open_aperture_faces expects triangular faces.")

    vertex_count = int(deformed_vertices.shape[0])
    upper_ids = _ordered_mapped_vertex_ids(
        mapping,
        INNER_UPPER_LIP_MEDIAPIPE_IDS,
        vertex_count,
    )
    lower_ids = _ordered_mapped_vertex_ids(
        mapping,
        INNER_LOWER_LIP_MEDIAPIPE_IDS,
        vertex_count,
    )
    if len(upper_ids) < 3 or len(lower_ids) < 3:
        raise ValueError(
            "Jaw-open aperture detection requires mapped upper and lower inner lips."
        )
    ordered_ids = upper_ids + list(reversed(lower_ids))
    polygon = deformed_vertices[
        torch.tensor(ordered_ids, dtype=torch.long)
    ][:, (horizontal_axis, vertical_axis)]
    polygon = _compact_polygon_points(polygon)
    if polygon.shape[0] < 3:
        raise ValueError("Jaw-open inner-lip aperture polygon is degenerate.")
    polygon_center = polygon.mean(dim=0, keepdim=True)
    polygon = polygon_center + (polygon - polygon_center) * float(aperture_scale)

    face_centers = deformed_vertices.index_select(
        0,
        faces.reshape(-1),
    ).reshape(-1, 3, 3).mean(dim=1)
    inside_aperture = points_in_polygon_2d(
        face_centers[:, horizontal_axis],
        face_centers[:, vertical_axis],
        polygon,
    )
    lip_depths = deformed_vertices[
        torch.tensor(upper_ids + lower_ids, dtype=torch.long),
        depth_axis,
    ]
    lip_depth_center = float(lip_depths.mean().item())
    mesh_depth_min = float(deformed_vertices[:, depth_axis].amin().item())
    mesh_depth_max = float(deformed_vertices[:, depth_axis].amax().item())
    front_direction = (
        -1
        if lip_depth_center - mesh_depth_min < mesh_depth_max - lip_depth_center
        else 1
    )
    if front_direction < 0:
        front_depth = face_centers[:, depth_axis] <= (
            lip_depths.amax() + float(depth_padding)
        )
    else:
        front_depth = face_centers[:, depth_axis] >= (
            lip_depths.amin() - float(depth_padding)
        )
    overlap_aperture = torch.zeros_like(inside_aperture)
    front_face_ids = torch.nonzero(front_depth, as_tuple=False).flatten().long()
    if front_face_ids.numel() > 0:
        inset_polygon = polygon_center + (
            polygon - polygon_center
        ) * float(overlap_inset_scale)
        front_triangles = deformed_vertices.index_select(
            0,
            faces.index_select(0, front_face_ids).reshape(-1),
        ).reshape(-1, 3, 3)
        edge_midpoints = 0.5 * (
            front_triangles + front_triangles.roll(shifts=-1, dims=1)
        )
        overlap_samples = torch.cat((front_triangles, edge_midpoints), dim=1)
        sample_inside = points_in_polygon_2d(
            overlap_samples[..., horizontal_axis].reshape(-1),
            overlap_samples[..., vertical_axis].reshape(-1),
            inset_polygon,
        ).reshape(front_face_ids.numel(), -1)
        triangle_2d = front_triangles[..., (horizontal_axis, vertical_axis)]
        triangle_origin = triangle_2d[:, :1, :]
        triangle_u = triangle_2d[:, 1:2, :] - triangle_origin
        triangle_v = triangle_2d[:, 2:3, :] - triangle_origin
        polygon_relative = inset_polygon.unsqueeze(0) - triangle_origin
        dot_uu = (triangle_u * triangle_u).sum(dim=2)
        dot_uv = (triangle_u * triangle_v).sum(dim=2)
        dot_vv = (triangle_v * triangle_v).sum(dim=2)
        dot_up = (polygon_relative * triangle_u).sum(dim=2)
        dot_vp = (polygon_relative * triangle_v).sum(dim=2)
        denominator = dot_uu * dot_vv - dot_uv.square()
        valid_triangle = denominator.abs() > 1.0e-12
        safe_denominator = torch.where(
            valid_triangle,
            denominator,
            torch.ones_like(denominator),
        )
        barycentric_u = (
            dot_vv * dot_up - dot_uv * dot_vp
        ) / safe_denominator
        barycentric_v = (
            dot_uu * dot_vp - dot_uv * dot_up
        ) / safe_denominator
        polygon_inside_triangle = (
            valid_triangle
            & (barycentric_u >= -1.0e-6)
            & (barycentric_v >= -1.0e-6)
            & (barycentric_u + barycentric_v <= 1.0 + 1.0e-6)
        ).any(dim=1)
        overlap_aperture[front_face_ids] = (
            sample_inside.any(dim=1) | polygon_inside_triangle
        )
    removed_mask = front_depth & (inside_aperture | overlap_aperture)
    removed_face_ids = torch.nonzero(
        removed_mask,
        as_tuple=False,
    ).flatten().long()
    return JawOpenApertureCutReport(
        aperture_scale=float(aperture_scale),
        overlap_inset_scale=float(overlap_inset_scale),
        depth_padding=float(depth_padding),
        front_direction=int(front_direction),
        polygon_point_count=int(polygon.shape[0]),
        inside_aperture_face_count=int(inside_aperture.sum().item()),
        overlap_aperture_face_count=int(overlap_aperture.sum().item()),
        removed_face_count=int(removed_face_ids.numel()),
        original_face_count=int(faces.shape[0]),
        final_face_count=int(faces.shape[0] - removed_face_ids.numel()),
        aperture_polygon=polygon.tolist(),
        removed_face_ids=[int(face_id) for face_id in removed_face_ids.tolist()],
    )


def build_mouth_cavity_disk(
    vertices: torch.Tensor,
    mapping: Mapping[int, int],
    *,
    horizontal_axis: int = 0,
    depth_axis: int = 1,
    vertical_axis: int = 2,
    scale: float = 1.12,
    shape_exponent: float = 4.0,
    depth_offset: float = 0.02,
    segments: int = 64,
    radial_rings: int = 1,
    curvature_depth: float = 0.0,
    inner_radius_ratio: float = 0.0,
    side_wall_min_horizontal_ratio: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, MouthCavityReport]:
    """Build a recessed mouth-shaped lining behind the inner lips."""

    vertices = vertices.detach().cpu().float().contiguous()
    axes = (int(horizontal_axis), int(depth_axis), int(vertical_axis))
    if sorted(axes) != [0, 1, 2]:
        raise ValueError(
            "horizontal_axis, depth_axis, and vertical_axis must be distinct "
            "coordinate indices in [0, 2]."
        )
    if scale <= 0.0:
        raise ValueError("Mouth cavity scale must be positive.")
    if shape_exponent < 2.0:
        raise ValueError("Mouth cavity shape exponent must be at least 2.0.")
    if depth_offset <= 0.0:
        raise ValueError("Mouth cavity depth offset must be positive.")
    if segments < 8:
        raise ValueError("Mouth cavity requires at least eight segments.")
    if radial_rings < 1:
        raise ValueError("Mouth cavity requires at least one radial ring.")
    if curvature_depth < 0.0:
        raise ValueError("Mouth cavity curvature depth cannot be negative.")
    if not 0.0 <= inner_radius_ratio < 1.0:
        raise ValueError("Mouth cavity inner radius ratio must be in [0, 1).")
    if inner_radius_ratio > 0.0 and radial_rings < 2:
        raise ValueError("Annular mouth lining requires at least two radial rings.")
    if not 0.0 <= side_wall_min_horizontal_ratio < 1.0:
        raise ValueError("Mouth side-wall horizontal ratio must be in [0, 1).")

    lip_ids = _ordered_mapped_vertex_ids(
        mapping,
        INNER_UPPER_LIP_MEDIAPIPE_IDS + INNER_LOWER_LIP_MEDIAPIPE_IDS,
        int(vertices.shape[0]),
    )
    lip_ids = list(dict.fromkeys(lip_ids))
    if len(lip_ids) < 4:
        raise ValueError("Mouth cavity requires at least four mapped inner-lip vertices.")
    lip_points = vertices[torch.tensor(lip_ids, dtype=torch.long)]

    projected = lip_points[:, (horizontal_axis, vertical_axis)]
    projected_min = projected.amin(dim=0)
    projected_max = projected.amax(dim=0)
    projected_center = 0.5 * (projected_min + projected_max)
    projected_radius = (
        0.5 * (projected_max - projected_min) * float(scale)
    ).clamp_min(1.0e-4)

    lip_depths = lip_points[:, depth_axis]
    mesh_depth_min = vertices[:, depth_axis].amin()
    mesh_depth_max = vertices[:, depth_axis].amax()
    lip_depth_center = lip_depths.mean()
    front_direction = (
        -1
        if lip_depth_center - mesh_depth_min < mesh_depth_max - lip_depth_center
        else 1
    )
    if front_direction < 0:
        cavity_depth = lip_depths.amax() + float(depth_offset)
    else:
        cavity_depth = lip_depths.amin() - float(depth_offset)

    center = torch.zeros(3, dtype=torch.float32)
    center[horizontal_axis] = projected_center[0]
    center[vertical_axis] = projected_center[1]
    center[depth_axis] = (
        cavity_depth - float(front_direction) * float(curvature_depth)
    )
    angles = torch.arange(segments, dtype=torch.float32) * (
        2.0 * torch.pi / float(segments)
    )
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    shape_power = 2.0 / float(shape_exponent)
    unit_horizontal = cosine.sign() * cosine.abs().pow(shape_power)
    unit_vertical = sine.sign() * sine.abs().pow(shape_power)
    rings: list[torch.Tensor] = []
    radius_fractions = (
        torch.linspace(float(inner_radius_ratio), 1.0, radial_rings).tolist()
        if inner_radius_ratio > 0.0
        else [
            float(ring_index) / float(radial_rings)
            for ring_index in range(1, radial_rings + 1)
        ]
    )
    for radius_fraction in radius_fractions:
        ring = center.repeat(segments, 1)
        ring[:, horizontal_axis] += (
            projected_radius[0] * radius_fraction * unit_horizontal
        )
        ring[:, vertical_axis] += (
            projected_radius[1] * radius_fraction * unit_vertical
        )
        ring[:, depth_axis] = (
            cavity_depth
            - float(front_direction)
            * float(curvature_depth)
            * (1.0 - radius_fraction * radius_fraction)
        )
        rings.append(ring)
    if inner_radius_ratio > 0.0:
        cavity_vertices = torch.cat(rings, dim=0).contiguous()
        face_blocks: list[torch.Tensor] = []
        ring_vertex_offset = 0
    else:
        cavity_vertices = torch.cat((center.unsqueeze(0), *rings), dim=0).contiguous()
        first_ring_ids = torch.arange(1, segments + 1, dtype=torch.long)
        next_first_ring_ids = first_ring_ids.roll(shifts=-1)
        center_ids = torch.zeros(segments, dtype=torch.long)
        if front_direction < 0:
            face_blocks = [
                torch.stack((center_ids, first_ring_ids, next_first_ring_ids), dim=1)
            ]
        else:
            face_blocks = [
                torch.stack((center_ids, next_first_ring_ids, first_ring_ids), dim=1)
            ]
        ring_vertex_offset = 1
    for ring_index in range(radial_rings - 1):
        inner = (
            ring_vertex_offset
            + ring_index * segments
            + torch.arange(segments, dtype=torch.long)
        )
        outer = inner + segments
        inner_next = inner.roll(shifts=-1)
        outer_next = outer.roll(shifts=-1)
        if front_direction < 0:
            face_blocks.extend(
                (
                    torch.stack((inner, outer, outer_next), dim=1),
                    torch.stack((inner, outer_next, inner_next), dim=1),
                )
            )
        else:
            face_blocks.extend(
                (
                    torch.stack((inner, outer_next, outer), dim=1),
                    torch.stack((inner, inner_next, outer_next), dim=1),
                )
            )
    if side_wall_min_horizontal_ratio > 0.0:
        segment_mid_angles = angles + torch.pi / float(segments)
        side_segments = (
            torch.cos(segment_mid_angles).abs()
            >= float(side_wall_min_horizontal_ratio)
        )
        face_blocks = [block[side_segments] for block in face_blocks]
    cavity_faces = torch.cat(face_blocks, dim=0).contiguous()
    triangles = cavity_vertices[cavity_faces]
    face_normals = torch.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=1,
    )
    cavity_normals = torch.zeros_like(cavity_vertices)
    for corner in range(3):
        cavity_normals.index_add_(0, cavity_faces[:, corner], face_normals)
    cavity_normals = F.normalize(cavity_normals, dim=1, eps=1.0e-8)
    report = MouthCavityReport(
        vertex_count=int(cavity_vertices.shape[0]),
        face_count=int(cavity_faces.shape[0]),
        segment_count=int(segments),
        horizontal_axis=int(horizontal_axis),
        depth_axis=int(depth_axis),
        vertical_axis=int(vertical_axis),
        front_direction=int(front_direction),
        scale=float(scale),
        shape_exponent=float(shape_exponent),
        depth_offset=float(depth_offset),
        radial_ring_count=int(radial_rings),
        curvature_depth=float(curvature_depth),
        inner_radius_ratio=float(inner_radius_ratio),
        side_wall_min_horizontal_ratio=float(side_wall_min_horizontal_ratio),
        center=[float(value) for value in center.tolist()],
        radius_horizontal=float(projected_radius[0].item()),
        radius_vertical=float(projected_radius[1].item()),
    )
    return cavity_vertices, cavity_faces.contiguous(), cavity_normals, report


def build_mouth_boundary_lining(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    mapping: Mapping[int, int],
    *,
    reference_vertices: torch.Tensor | None = None,
    horizontal_axis: int = 0,
    depth_axis: int = 1,
    vertical_axis: int = 2,
    depth: float = 0.06,
    rim_depth: float = 0.002,
    detection_depth_padding: float = 0.12,
    side_fraction: float = 1.0,
    inner_scale: float = 1.0,
    radial_rings: int = 1,
    smoothing_iterations: int = 0,
    analytic_segments: int | None = None,
    opening_scale: float = 1.0,
    shape_exponent: float = 2.8,
    cap_back: bool = False,
    maximum_cycles: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, MouthBoundaryLiningReport]:
    """Extrude real mouth boundary loops into a continuous oral compartment."""

    vertices = vertices.detach().cpu().float().contiguous()
    reference = (
        vertices
        if reference_vertices is None
        else reference_vertices.detach().cpu().float().contiguous()
    )
    faces = faces.detach().cpu().long().contiguous()
    if reference.shape != vertices.shape:
        raise ValueError("Mouth lining reference vertices must match target vertices.")
    if depth <= 0.0:
        raise ValueError("Mouth boundary lining depth must be positive.")
    if rim_depth < 0.0 or rim_depth >= depth:
        raise ValueError("Mouth lining rim depth must be in [0, depth).")
    if not 0.0 < side_fraction <= 1.0:
        raise ValueError("Mouth lining side fraction must be in (0, 1].")
    if not 0.0 < inner_scale <= 1.0:
        raise ValueError("Mouth lining inner scale must be in (0, 1].")
    if radial_rings < 1:
        raise ValueError("Mouth lining requires at least one radial ring.")
    if smoothing_iterations < 0:
        raise ValueError("Mouth lining smoothing iterations cannot be negative.")
    if analytic_segments is not None and analytic_segments < 8:
        raise ValueError("Analytic mouth compartment requires at least eight segments.")
    if opening_scale <= 0.0:
        raise ValueError("Mouth compartment opening scale must be positive.")
    if shape_exponent < 2.0:
        raise ValueError("Mouth compartment shape exponent must be at least 2.0.")
    if analytic_segments is not None and maximum_cycles != 1:
        raise ValueError("Analytic mouth compartment supports exactly one opening.")
    if maximum_cycles < 1:
        raise ValueError("Mouth lining maximum cycles must be positive.")
    if cap_back and side_fraction < 1.0:
        raise ValueError("A capped mouth compartment requires complete side walls.")

    lip_ids = list(
        dict.fromkeys(
            _ordered_mapped_vertex_ids(
                mapping,
                INNER_UPPER_LIP_MEDIAPIPE_IDS + INNER_LOWER_LIP_MEDIAPIPE_IDS,
                int(reference.shape[0]),
            )
        )
    )
    if len(lip_ids) < 4:
        raise ValueError("Mouth boundary lining requires mapped inner lips.")
    lip_points = reference[torch.tensor(lip_ids, dtype=torch.long)]
    target_lip_points = vertices[torch.tensor(lip_ids, dtype=torch.long)]
    target_projected_center = 0.5 * (
        target_lip_points[:, (horizontal_axis, vertical_axis)].amin(dim=0)
        + target_lip_points[:, (horizontal_axis, vertical_axis)].amax(dim=0)
    )
    projected = lip_points[:, (horizontal_axis, vertical_axis)]
    projected_min = projected.amin(dim=0)
    projected_max = projected.amax(dim=0)
    projected_span = (projected_max - projected_min).clamp_min(1.0e-6)

    boundary_edges = _mesh_boundary_edges(faces)
    boundary_midpoints = reference[boundary_edges].mean(dim=1)
    mouth_mask = (
        (boundary_midpoints[:, horizontal_axis] >= projected_min[0] - 0.45 * projected_span[0])
        & (boundary_midpoints[:, horizontal_axis] <= projected_max[0] + 0.45 * projected_span[0])
        & (boundary_midpoints[:, vertical_axis] >= projected_min[1] - 0.45 * projected_span[1])
        & (boundary_midpoints[:, vertical_axis] <= projected_max[1] + 0.45 * projected_span[1])
        & (
            boundary_midpoints[:, depth_axis]
            >= lip_points[:, depth_axis].amin() - float(detection_depth_padding)
        )
        & (
            boundary_midpoints[:, depth_axis]
            <= lip_points[:, depth_axis].amax() + float(detection_depth_padding)
        )
    )
    adjacency: dict[int, list[int]] = {}
    for first, second in boundary_edges[mouth_mask].tolist():
        adjacency.setdefault(int(first), []).append(int(second))
        adjacency.setdefault(int(second), []).append(int(first))
    cycles = [cycle for cycle in _undirected_cycle_basis(adjacency) if len(cycle) >= 3]

    candidates: list[tuple[float, list[int]]] = []
    for cycle in cycles:
        points = reference[torch.tensor(cycle, dtype=torch.long)]
        horizontal_span = points[:, horizontal_axis].amax() - points[:, horizontal_axis].amin()
        vertical_span = points[:, vertical_axis].amax() - points[:, vertical_axis].amin()
        if (
            horizontal_span < 0.8 * projected_span[0]
            or vertical_span < 0.8 * projected_span[1]
        ):
            continue
        points_2d = points[:, (horizontal_axis, vertical_axis)]
        area = abs(
            0.5
            * float(
                (
                    points_2d[:, 0] * torch.roll(points_2d[:, 1], shifts=-1)
                    - points_2d[:, 1] * torch.roll(points_2d[:, 0], shifts=-1)
                ).sum().item()
            )
        )
        candidates.append((area, cycle))
    selected_cycles = [
        item[1] for item in sorted(candidates, reverse=True)[:maximum_cycles]
    ]
    boundary_source = "closed_cycle"
    if not selected_cycles and analytic_segments is not None:
        remaining = set(adjacency)
        components: list[list[int]] = []
        while remaining:
            root = min(remaining)
            stack = [root]
            component: list[int] = []
            remaining.remove(root)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
            if len(component) >= 3:
                components.append(component)

        if components:
            def projected_component_area(component: list[int]) -> float:
                points = reference[torch.tensor(component, dtype=torch.long)]
                projected_points = points[:, (horizontal_axis, vertical_axis)]
                span = projected_points.amax(dim=0) - projected_points.amin(dim=0)
                return float(span.prod().item())

            selected_cycles = [max(components, key=projected_component_area)]
            boundary_source = "open_component"
        else:
            selected_cycles = [lip_ids]
            boundary_source = "lip_landmarks"
    if not selected_cycles:
        raise ValueError("No dominant mouth boundary loops were found for lining.")

    lip_depth_center = lip_points[:, depth_axis].mean()
    front_direction = (
        -1
        if lip_depth_center - reference[:, depth_axis].amin()
        < reference[:, depth_axis].amax() - lip_depth_center
        else 1
    )
    vertex_parts: list[torch.Tensor] = []
    face_parts: list[torch.Tensor] = []
    local_offset = 0
    boundary_vertex_count = 0
    for cycle in selected_cycles:
        ids = torch.tensor(cycle, dtype=torch.long)
        cycle_points = vertices[ids]
        if analytic_segments is None:
            base = cycle_points.clone()
            ring_center = target_projected_center
        else:
            bounds_points = torch.cat((cycle_points, target_lip_points), dim=0)
            cycle_projected = bounds_points[:, (horizontal_axis, vertical_axis)]
            cycle_min = cycle_projected.amin(dim=0)
            cycle_max = cycle_projected.amax(dim=0)
            ring_center = 0.5 * (cycle_min + cycle_max)
            ring_radius = (
                0.5 * (cycle_max - cycle_min) * float(opening_scale)
            ).clamp_min(1.0e-5)
            angles = torch.arange(analytic_segments, dtype=torch.float32) * (
                2.0 * torch.pi / float(analytic_segments)
            )
            shape_power = 2.0 / float(shape_exponent)
            unit_horizontal = (
                torch.cos(angles).sign()
                * torch.cos(angles).abs().pow(shape_power)
            )
            unit_vertical = (
                torch.sin(angles).sign()
                * torch.sin(angles).abs().pow(shape_power)
            )
            base = vertices.new_zeros((analytic_segments, 3))
            base[:, horizontal_axis] = (
                ring_center[0] + ring_radius[0] * unit_horizontal
            )
            base[:, vertical_axis] = (
                ring_center[1] + ring_radius[1] * unit_vertical
            )
            cycle_depths = bounds_points[:, depth_axis]
            base[:, depth_axis] = (
                cycle_depths.amax()
                if front_direction < 0
                else cycle_depths.amin()
            )
        count = int(base.shape[0])
        base_projected = base[:, (horizontal_axis, vertical_axis)]
        smooth_projected = base_projected.clone()
        for _ in range(smoothing_iterations):
            smooth_projected = (
                0.5 * smooth_projected
                + 0.25 * torch.roll(smooth_projected, shifts=1, dims=0)
                + 0.25 * torch.roll(smooth_projected, shifts=-1, dims=0)
            )
        if smoothing_iterations > 0:
            base_min = base_projected.amin(dim=0)
            base_max = base_projected.amax(dim=0)
            smooth_min = smooth_projected.amin(dim=0)
            smooth_max = smooth_projected.amax(dim=0)
            base_center = 0.5 * (base_min + base_max)
            smooth_center = 0.5 * (smooth_min + smooth_max)
            span_scale = (base_max - base_min) / (
                smooth_max - smooth_min
            ).clamp_min(1.0e-6)
            smooth_projected = (
                base_center + (smooth_projected - smooth_center) * span_scale
            )

        ring_parts: list[torch.Tensor] = []
        front_depths = (
            base[:, depth_axis] - float(front_direction) * float(rim_depth)
        )
        back_depth = (
            base[:, depth_axis].mean()
            - float(front_direction) * float(depth)
        )
        for ring_index in range(radial_rings + 1):
            fraction = ring_index / float(radial_rings)
            ring = base.clone()
            ring[:, depth_axis] = (
                front_depths * (1.0 - fraction) + back_depth * fraction
            )
            ring_scale = 1.0 - (1.0 - float(inner_scale)) * fraction
            ring_projected = (
                base_projected * (1.0 - fraction)
                + smooth_projected * fraction
            )
            ring_projected = ring_center + (
                ring_projected - ring_center
            ) * ring_scale
            ring[:, horizontal_axis] = ring_projected[:, 0]
            ring[:, vertical_axis] = ring_projected[:, 1]
            ring_parts.append(ring)

        reference_points = (
            reference[ids]
            if analytic_segments is None
            else base
        )
        edge_horizontal = 0.5 * (
            reference_points[:, horizontal_axis]
            + torch.roll(reference_points[:, horizontal_axis], shifts=-1)
        )
        side_width = float(side_fraction) * projected_span[0]
        side_edges = (
            (edge_horizontal <= projected_min[0] + side_width)
            | (edge_horizontal >= projected_max[0] - side_width)
        )

        for ring_index in range(radial_rings):
            outer_ids = (
                local_offset
                + ring_index * count
                + torch.arange(count, dtype=torch.long)
            )
            inner_ids = outer_ids + count
            outer_next = outer_ids.roll(shifts=-1)
            inner_next = inner_ids.roll(shifts=-1)
            first_block = torch.stack((outer_ids, inner_ids, inner_next), dim=1)
            second_block = torch.stack((outer_ids, inner_next, outer_next), dim=1)
            if side_fraction < 1.0:
                first_block = first_block[side_edges]
                second_block = second_block[side_edges]
            face_parts.extend((first_block, second_block))

        vertex_parts.extend(ring_parts)
        cycle_vertex_count = (radial_rings + 1) * count
        if cap_back:
            back_ring_ids = (
                local_offset
                + radial_rings * count
                + torch.arange(count, dtype=torch.long)
            )
            back_next = back_ring_ids.roll(shifts=-1)
            cap_center = ring_parts[-1].mean(dim=0)
            cap_center[horizontal_axis] = ring_center[0]
            cap_center[vertical_axis] = ring_center[1]
            center_id = torch.full(
                (count,),
                local_offset + cycle_vertex_count,
                dtype=torch.long,
            )
            face_parts.append(
                torch.stack((back_ring_ids, center_id, back_next), dim=1)
            )
            vertex_parts.append(cap_center.unsqueeze(0))
            cycle_vertex_count += 1

        local_offset += cycle_vertex_count
        boundary_vertex_count += count

    lining_vertices = torch.cat(vertex_parts, dim=0).contiguous()
    lining_faces = torch.cat(face_parts, dim=0).contiguous()
    triangles = lining_vertices[lining_faces]
    face_normals = torch.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=1,
    )
    lining_normals = torch.zeros_like(lining_vertices)
    for corner in range(3):
        lining_normals.index_add_(0, lining_faces[:, corner], face_normals)
    lining_normals = F.normalize(lining_normals, dim=1, eps=1.0e-8)
    report = MouthBoundaryLiningReport(
        vertex_count=int(lining_vertices.shape[0]),
        face_count=int(lining_faces.shape[0]),
        boundary_cycle_count=(
            len(selected_cycles) if boundary_source == "closed_cycle" else 0
        ),
        boundary_vertex_count=boundary_vertex_count,
        depth=float(depth),
        front_direction=int(front_direction),
        side_fraction=float(side_fraction),
        inner_scale=float(inner_scale),
        radial_ring_count=int(radial_rings),
        smoothing_iterations=int(smoothing_iterations),
        analytic_segments=(
            int(analytic_segments) if analytic_segments is not None else None
        ),
        opening_scale=float(opening_scale),
        shape_exponent=float(shape_exponent),
        boundary_source=boundary_source,
        cap_back=bool(cap_back),
        maximum_cycles=int(maximum_cycles),
    )
    return lining_vertices, lining_faces, lining_normals, report


def build_mediapipe_lip_annulus(
    neutral_vertices: torch.Tensor,
    deformed_vertices: torch.Tensor,
    mapping: Mapping[int, int],
    *,
    horizontal_axis: int = 0,
    depth_axis: int = 1,
    vertical_axis: int = 2,
    segments: int = 64,
    radial_rings: int = 3,
    outer_scale: float = 1.03,
    thickness_ratio: float = 0.055,
    side_thickness_factor: float = 0.45,
    maximum_radial_inset_fraction: float = 0.45,
    inner_recess_depth: float = 0.003,
    front_offset: float = 0.002,
    bulge_depth: float = 0.0025,
    upper_smoothing_window: int = 9,
    lower_smoothing_window: int = 17,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    LipAnnulusRepairReport,
]:
    """Build a continuous lip strip from a uniformly inset outer contour.

    MediaPipe's inner and outer lip curves do not have dependable point-wise
    correspondence after projection onto fragmented meshes.  Pairing those
    curves can therefore bridge unrelated points across the mouth.  This
    repair samples only the repaired outer contour and derives a narrow inner
    edge geometrically, leaving the predicted mouth opening unobstructed.
    """

    from scipy.interpolate import PchipInterpolator

    neutral_vertices = neutral_vertices.detach().cpu().float().contiguous()
    deformed_vertices = deformed_vertices.detach().cpu().float().contiguous()
    if neutral_vertices.shape != deformed_vertices.shape:
        raise ValueError("Lip annulus poses must have matching vertex arrays.")
    if segments < 16 or segments % 2 != 0:
        raise ValueError("Lip annulus segments must be an even integer of at least 16.")
    if radial_rings < 1:
        raise ValueError("Lip annulus requires at least one radial ring.")
    if outer_scale <= 0.0:
        raise ValueError("Lip annulus outer scale must be positive.")
    if thickness_ratio <= 0.0:
        raise ValueError("Lip annulus thickness ratio must be positive.")
    if not 0.0 < side_thickness_factor <= 1.0:
        raise ValueError("Lip annulus side thickness factor must be in (0, 1].")
    if not 0.0 < maximum_radial_inset_fraction < 1.0:
        raise ValueError(
            "Lip annulus maximum radial inset fraction must be between 0 and 1."
        )
    if inner_recess_depth < 0.0:
        raise ValueError("Lip annulus inner recess depth cannot be negative.")
    if front_offset < 0.0:
        raise ValueError("Lip annulus front offset cannot be negative.")
    if bulge_depth < 0.0:
        raise ValueError("Lip annulus bulge depth cannot be negative.")
    side_samples = segments // 2 + 1
    for name, window in (
        ("upper", upper_smoothing_window),
        ("lower", lower_smoothing_window),
    ):
        if window < 3 or window % 2 == 0 or window > side_samples:
            raise ValueError(
                f"Lip annulus {name} smoothing window must be an odd integer "
                f"between 3 and {side_samples}."
            )
    axes = {int(horizontal_axis), int(depth_axis), int(vertical_axis)}
    if axes != {0, 1, 2}:
        raise ValueError("Lip annulus axes must be a permutation of [0, 1, 2].")

    vertex_count = int(neutral_vertices.shape[0])
    curve_landmark_ids = (
        OUTER_UPPER_LIP_MEDIAPIPE_IDS,
        OUTER_LOWER_LIP_MEDIAPIPE_IDS,
    )
    curve_vertex_ids = [
        _ordered_mapped_vertex_ids(mapping, curve, vertex_count)
        for curve in curve_landmark_ids
    ]
    if any(len(ids) < 4 for ids in curve_vertex_ids):
        raise ValueError(
            "Lip annulus repair requires mapped outer upper/lower curves."
        )

    def sample_curve(
        vertices: torch.Tensor,
        ids: list[int],
        smoothing_window: int,
    ) -> torch.Tensor:
        from scipy.signal import savgol_filter

        points = vertices[torch.tensor(ids, dtype=torch.long)].double().numpy()
        projected = points[:, (horizontal_axis, vertical_axis)]
        distances = np.linalg.norm(np.diff(projected, axis=0), axis=1)
        parameter = np.concatenate(([0.0], np.cumsum(distances)))
        keep = np.concatenate(([True], np.diff(parameter) > 1.0e-10))
        parameter = parameter[keep]
        points = points[keep]
        if parameter.shape[0] < 2 or parameter[-1] <= 1.0e-10:
            raise ValueError("A lip annulus curve collapsed to one point.")
        parameter /= parameter[-1]
        samples = PchipInterpolator(
            parameter,
            points,
            axis=0,
            extrapolate=False,
        )(np.linspace(0.0, 1.0, side_samples))
        endpoints = samples[[0, -1]].copy()
        for axis in (depth_axis, vertical_axis):
            samples[:, axis] = savgol_filter(
                samples[:, axis],
                window_length=int(smoothing_window),
                polyorder=3,
                mode="interp",
            )
        samples[[0, -1]] = endpoints
        return torch.from_numpy(samples.astype(np.float32, copy=False))

    def closed_ring(upper: torch.Tensor, lower: torch.Tensor) -> torch.Tensor:
        return torch.cat((upper, torch.flip(lower[1:-1], dims=(0,))), dim=0)

    neutral_outer_upper = sample_curve(
        neutral_vertices, curve_vertex_ids[0], upper_smoothing_window
    )
    neutral_outer_lower = sample_curve(
        neutral_vertices, curve_vertex_ids[1], lower_smoothing_window
    )
    deformed_outer_upper = sample_curve(
        deformed_vertices, curve_vertex_ids[0], upper_smoothing_window
    )
    deformed_outer_lower = sample_curve(
        deformed_vertices, curve_vertex_ids[1], lower_smoothing_window
    )

    neutral_outer = closed_ring(neutral_outer_upper, neutral_outer_lower)
    deformed_outer = closed_ring(deformed_outer_upper, deformed_outer_lower)
    if any(int(ring.shape[0]) != segments for ring in (neutral_outer, deformed_outer)):
        raise RuntimeError("Lip annulus curve resampling produced inconsistent rings.")

    def build_ring_pair(
        outer: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        projected = outer[:, (horizontal_axis, vertical_axis)]
        center = 0.5 * (projected.amin(dim=0) + projected.amax(dim=0))
        scaled_outer = outer.clone()
        scaled_outer[:, horizontal_axis] = center[0] + float(outer_scale) * (
            outer[:, horizontal_axis] - center[0]
        )
        scaled_outer[:, vertical_axis] = center[1] + float(outer_scale) * (
            outer[:, vertical_axis] - center[1]
        )
        scaled_projected = scaled_outer[:, (horizontal_axis, vertical_axis)]
        radial_delta = scaled_projected - center
        radial_distance = radial_delta.norm(dim=1).clamp_min(1.0e-8)
        projected_span = scaled_projected.amax(dim=0) - scaled_projected.amin(dim=0)
        base_inset = max(
            float(thickness_ratio) * float(projected_span.max().item()),
            1.0e-6,
        )
        vertical_fraction = (
            radial_delta[:, 1].abs() / radial_distance
        ).clamp(0.0, 1.0)
        thickness_weight = float(side_thickness_factor) + (
            1.0 - float(side_thickness_factor)
        ) * vertical_fraction.pow(1.5)
        requested_inset = base_inset * thickness_weight
        inset = torch.minimum(
            requested_inset,
            radial_distance * float(maximum_radial_inset_fraction),
        )
        inner_projected = scaled_projected - (
            radial_delta / radial_distance.unsqueeze(1)
        ) * inset.unsqueeze(1)
        inner = scaled_outer.clone()
        inner[:, horizontal_axis] = inner_projected[:, 0]
        inner[:, vertical_axis] = inner_projected[:, 1]
        capped_count = int((inset < requested_inset - 1.0e-8).sum().item())
        return scaled_outer, inner, capped_count

    (
        neutral_outer,
        neutral_inner,
        capped_neutral_segment_count,
    ) = build_ring_pair(neutral_outer)
    (
        deformed_outer,
        deformed_inner,
        capped_deformed_segment_count,
    ) = build_ring_pair(deformed_outer)
    lip_depth_center = float(neutral_outer[:, depth_axis].mean().item())
    front_direction = (
        -1
        if lip_depth_center - float(neutral_vertices[:, depth_axis].amin().item())
        < float(neutral_vertices[:, depth_axis].amax().item()) - lip_depth_center
        else 1
    )
    neutral_inner[:, depth_axis] -= float(front_direction) * float(inner_recess_depth)
    deformed_inner[:, depth_axis] -= float(front_direction) * float(inner_recess_depth)

    def surface_rows(outer: torch.Tensor, inner: torch.Tensor) -> torch.Tensor:
        rows = []
        for ring_index in range(radial_rings + 1):
            fraction = ring_index / float(radial_rings)
            smooth_fraction = fraction * fraction * (3.0 - 2.0 * fraction)
            row = outer * (1.0 - smooth_fraction) + inner * smooth_fraction
            # Keep the repair just in front of the damaged source lip. The
            # offset grows toward the cut edge and avoids coplanar flicker.
            offset_weight = 0.25 + 0.75 * smooth_fraction
            row[:, depth_axis] += (
                float(front_direction)
                * (
                    float(front_offset) * offset_weight
                    + float(bulge_depth) * math.sin(math.pi * fraction)
                )
            )
            rows.append(row)
        return torch.cat(rows, dim=0).contiguous()

    neutral_annulus = surface_rows(neutral_outer, neutral_inner)
    deformed_annulus = surface_rows(deformed_outer, deformed_inner)
    face_parts: list[torch.Tensor] = []
    ring_ids = torch.arange(segments, dtype=torch.long)
    next_ids = torch.roll(ring_ids, shifts=-1)
    for ring_index in range(radial_rings):
        outer_ids = ring_index * segments + ring_ids
        outer_next = ring_index * segments + next_ids
        inner_ids = outer_ids + segments
        inner_next = outer_next + segments
        face_parts.extend(
            (
                torch.stack((outer_ids, outer_next, inner_next), dim=1),
                torch.stack((outer_ids, inner_next, inner_ids), dim=1),
            )
        )
    annulus_faces = torch.cat(face_parts, dim=0).contiguous()
    triangles = neutral_annulus[annulus_faces]
    normals = torch.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=1,
    )
    if float(normals[:, depth_axis].mean().item()) * front_direction < 0.0:
        annulus_faces = annulus_faces[:, (0, 2, 1)].contiguous()

    report = LipAnnulusRepairReport(
        vertex_count=int(neutral_annulus.shape[0]),
        face_count=int(annulus_faces.shape[0]),
        segment_count=int(segments),
        radial_ring_count=int(radial_rings),
        outer_scale=float(outer_scale),
        thickness_ratio=float(thickness_ratio),
        side_thickness_factor=float(side_thickness_factor),
        maximum_radial_inset_fraction=float(maximum_radial_inset_fraction),
        inner_recess_depth=float(inner_recess_depth),
        front_offset=float(front_offset),
        bulge_depth=float(bulge_depth),
        capped_neutral_segment_count=int(capped_neutral_segment_count),
        capped_deformed_segment_count=int(capped_deformed_segment_count),
        upper_smoothing_window=int(upper_smoothing_window),
        lower_smoothing_window=int(lower_smoothing_window),
        front_direction=int(front_direction),
        horizontal_axis=int(horizontal_axis),
        depth_axis=int(depth_axis),
        vertical_axis=int(vertical_axis),
    )
    return neutral_annulus, deformed_annulus, annulus_faces, report


def clear_source_faces_under_lip_annulus(
    deformed_vertices: torch.Tensor,
    faces: torch.Tensor,
    deformed_annulus: torch.Tensor,
    *,
    segments: int,
    horizontal_axis: int = 0,
    depth_axis: int = 1,
    vertical_axis: int = 2,
    depth_padding_ratio: float = 0.30,
) -> tuple[torch.Tensor, LipAnnulusHeadCleanupReport]:
    """Clear source triangles replaced by the generated lip annulus."""

    deformed_vertices = deformed_vertices.detach().cpu().float().contiguous()
    faces = faces.detach().cpu().long().contiguous()
    deformed_annulus = deformed_annulus.detach().cpu().float().contiguous()
    if segments < 3 or int(deformed_annulus.shape[0]) < segments:
        raise ValueError("Lip annulus cleanup requires one complete outer ring.")
    if depth_padding_ratio < 0.0:
        raise ValueError("Lip annulus cleanup depth padding cannot be negative.")
    axes = {int(horizontal_axis), int(depth_axis), int(vertical_axis)}
    if axes != {0, 1, 2}:
        raise ValueError("Lip annulus cleanup axes must be a permutation of [0, 1, 2].")

    outer_ring = deformed_annulus[:segments]
    polygon_stride = max(1, int(segments) // 32)
    polygon = outer_ring[::polygon_stride, (horizontal_axis, vertical_axis)]
    triangles = deformed_vertices[faces]
    samples = torch.cat(
        (
            triangles.mean(dim=1, keepdim=True),
            0.5 * (triangles[:, 0:1] + triangles[:, 1:2]),
            0.5 * (triangles[:, 1:2] + triangles[:, 2:3]),
            0.5 * (triangles[:, 2:3] + triangles[:, 0:1]),
        ),
        dim=1,
    )
    projected_samples = samples[:, :, (horizontal_axis, vertical_axis)]
    polygon_min = polygon.amin(dim=0)
    polygon_max = polygon.amax(dim=0)
    triangle_projected = triangles[:, :, (horizontal_axis, vertical_axis)]
    bbox_overlap = (
        (triangle_projected.amax(dim=1) >= polygon_min).all(dim=1)
        & (triangle_projected.amin(dim=1) <= polygon_max).all(dim=1)
    )
    overlap = torch.zeros(int(faces.shape[0]), dtype=torch.bool)
    candidate_ids = torch.nonzero(bbox_overlap, as_tuple=False).flatten()
    if candidate_ids.numel() > 0:
        candidate_samples = projected_samples[candidate_ids]
        candidate_overlap = points_in_polygon_2d(
            candidate_samples[:, :, 0].reshape(-1),
            candidate_samples[:, :, 1].reshape(-1),
            polygon,
        ).reshape(int(candidate_ids.shape[0]), -1).any(dim=1)
        overlap[candidate_ids] = candidate_overlap
    projected_span = float(
        (polygon.amax(dim=0) - polygon.amin(dim=0)).amax().item()
    )
    depth_padding = float(depth_padding_ratio) * projected_span
    depth_min = float(outer_ring[:, depth_axis].amin().item()) - depth_padding
    depth_max = float(outer_ring[:, depth_axis].amax().item()) + depth_padding
    face_depth = triangles[:, :, depth_axis].mean(dim=1)
    within_depth = (face_depth >= depth_min) & (face_depth <= depth_max)
    removed = overlap & within_depth
    kept_faces = faces[~removed].contiguous()
    report = LipAnnulusHeadCleanupReport(
        original_face_count=int(faces.shape[0]),
        removed_face_count=int(removed.sum().item()),
        final_face_count=int(kept_faces.shape[0]),
        projected_overlap_face_count=int(overlap.sum().item()),
        depth_rejected_overlap_face_count=int((overlap & ~within_depth).sum().item()),
        polygon_point_count=int(polygon.shape[0]),
        depth_padding_ratio=float(depth_padding_ratio),
        depth_min=float(depth_min),
        depth_max=float(depth_max),
        horizontal_axis=int(horizontal_axis),
        depth_axis=int(depth_axis),
        vertical_axis=int(vertical_axis),
    )
    return kept_faces, report


def orient_front_surface_faces(
    neutral_vertices: torch.Tensor,
    deformed_vertices: torch.Tensor,
    faces: torch.Tensor,
    mapping: Mapping[int, int],
    *,
    depth_axis: int = 1,
    region_depth_ratio: float = 0.18,
    normal_tolerance: float = 0.01,
) -> tuple[torch.Tensor, FrontSurfaceOrientationReport]:
    """Orient the external facial shell toward the viewer.

    Reconstructed scan meshes can contain many disconnected triangles whose
    winding is inconsistent even when their positions form a usable surface.
    Restricting the repair to a depth band around the lips avoids changing the
    rear head or the separately appended oral assembly.
    """

    neutral_vertices = neutral_vertices.detach().cpu().float().contiguous()
    deformed_vertices = deformed_vertices.detach().cpu().float().contiguous()
    faces = faces.detach().cpu().long().contiguous()
    if neutral_vertices.shape != deformed_vertices.shape:
        raise ValueError("Front-surface orientation poses must have matching vertices.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Front-surface orientation faces must have shape [F, 3].")
    if not 0 <= int(depth_axis) <= 2:
        raise ValueError("Front-surface orientation depth axis must be 0, 1, or 2.")
    if region_depth_ratio <= 0.0:
        raise ValueError("Front-surface orientation depth ratio must be positive.")
    if normal_tolerance < 0.0:
        raise ValueError("Front-surface orientation normal tolerance cannot be negative.")
    vertex_count = int(neutral_vertices.shape[0])
    mouth_ids = _ordered_mapped_vertex_ids(
        mapping,
        OUTER_UPPER_LIP_MEDIAPIPE_IDS + OUTER_LOWER_LIP_MEDIAPIPE_IDS,
        vertex_count,
    )
    if len(mouth_ids) < 4:
        raise ValueError("Front-surface orientation requires mapped outer lip points.")

    mouth_index = torch.tensor(mouth_ids, dtype=torch.long)
    neutral_mouth_depth = float(
        neutral_vertices[mouth_index, depth_axis].mean().item()
    )
    neutral_depth_min = float(neutral_vertices[:, depth_axis].amin().item())
    neutral_depth_max = float(neutral_vertices[:, depth_axis].amax().item())
    front_direction = (
        -1
        if neutral_mouth_depth - neutral_depth_min
        < neutral_depth_max - neutral_mouth_depth
        else 1
    )

    deformed_mouth_depth = float(
        deformed_vertices[mouth_index, depth_axis].mean().item()
    )
    deformed_depth_min = float(deformed_vertices[:, depth_axis].amin().item())
    deformed_depth_max = float(deformed_vertices[:, depth_axis].amax().item())
    depth_span = max(deformed_depth_max - deformed_depth_min, 1.0e-6)
    front_depth_limit = deformed_mouth_depth - (
        float(front_direction) * float(region_depth_ratio) * depth_span
    )

    triangles = deformed_vertices[faces]
    face_centers = triangles.mean(dim=1)
    face_normals = torch.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=1,
    )
    normal_magnitude = face_normals.norm(dim=1)
    eligible = (
        float(front_direction)
        * (face_centers[:, depth_axis] - float(front_depth_limit))
        >= 0.0
    )
    reversed_winding = eligible & (
        float(front_direction) * face_normals[:, depth_axis]
        < -float(normal_tolerance) * normal_magnitude
    )
    oriented_faces = faces.clone()
    oriented_faces[reversed_winding] = oriented_faces[reversed_winding][:, (0, 2, 1)]
    report = FrontSurfaceOrientationReport(
        face_count=int(faces.shape[0]),
        eligible_face_count=int(eligible.sum().item()),
        flipped_face_count=int(reversed_winding.sum().item()),
        front_direction=int(front_direction),
        mouth_depth=float(deformed_mouth_depth),
        front_depth_limit=float(front_depth_limit),
        region_depth_ratio=float(region_depth_ratio),
        normal_tolerance=float(normal_tolerance),
        depth_axis=int(depth_axis),
    )
    return oriented_faces.contiguous(), report


def detect_stretched_mouth_faces(
    neutral_vertices: torch.Tensor,
    deformed_vertices: torch.Tensor,
    faces: torch.Tensor,
    mapping: Mapping[int, int],
    *,
    horizontal_axis: int = 0,
    depth_axis: int = 1,
    vertical_axis: int = 2,
    edge_ratio_threshold: float = 3.0,
    edge_growth_threshold: float = 0.02,
    neighbor_growth_rings: int = 2,
    neighbor_edge_ratio_threshold: float = 1.35,
    neighbor_edge_growth_threshold: float = 0.006,
    region_scale: float = 1.5,
    depth_padding: float = 0.08,
) -> StretchedMouthFaceReport:
    """Find strict edge-stretch outliers near the opened inner lips."""

    neutral_vertices = neutral_vertices.detach().cpu().float()
    deformed_vertices = deformed_vertices.detach().cpu().float()
    faces = faces.detach().cpu().long()
    lip_ids = _ordered_mapped_vertex_ids(
        mapping,
        INNER_UPPER_LIP_MEDIAPIPE_IDS + INNER_LOWER_LIP_MEDIAPIPE_IDS,
        int(deformed_vertices.shape[0]),
    )
    lip_ids = list(dict.fromkeys(lip_ids))
    if len(lip_ids) < 4:
        raise ValueError("Stretch detection requires mapped inner-lip vertices.")
    lip_points = deformed_vertices[torch.tensor(lip_ids, dtype=torch.long)]
    projected = lip_points[:, (horizontal_axis, vertical_axis)]
    center = 0.5 * (projected.amin(dim=0) + projected.amax(dim=0))
    radius = 0.5 * (projected.amax(dim=0) - projected.amin(dim=0)) * float(region_scale)

    neutral_triangles = neutral_vertices[faces]
    deformed_triangles = deformed_vertices[faces]
    neutral_edges = torch.stack(
        tuple(
            (neutral_triangles[:, (corner + 1) % 3] - neutral_triangles[:, corner]).norm(
                dim=1
            )
            for corner in range(3)
        ),
        dim=1,
    )
    deformed_edges = torch.stack(
        tuple(
            (deformed_triangles[:, (corner + 1) % 3] - deformed_triangles[:, corner]).norm(
                dim=1
            )
            for corner in range(3)
        ),
        dim=1,
    )
    max_ratio = (deformed_edges / neutral_edges.clamp_min(1.0e-8)).amax(dim=1)
    max_growth = (deformed_edges - neutral_edges).amax(dim=1)
    face_centers = deformed_triangles.mean(dim=1)
    neutral_lip_depth = neutral_vertices[
        torch.tensor(lip_ids, dtype=torch.long), depth_axis
    ].mean()
    neutral_depth_min = neutral_vertices[:, depth_axis].amin()
    neutral_depth_max = neutral_vertices[:, depth_axis].amax()
    front_direction = (
        -1
        if float((neutral_lip_depth - neutral_depth_min).item())
        < float((neutral_depth_max - neutral_lip_depth).item())
        else 1
    )
    if front_direction < 0:
        front_depth = face_centers[:, depth_axis] <= (
            lip_points[:, depth_axis].amax() + float(depth_padding)
        )
    else:
        front_depth = face_centers[:, depth_axis] >= (
            lip_points[:, depth_axis].amin() - float(depth_padding)
        )
    candidate = (
        ((face_centers[:, horizontal_axis] - center[0]).abs() <= radius[0])
        & ((face_centers[:, vertical_axis] - center[1]).abs() <= radius[1])
        & front_depth
    )
    strict_removed = candidate & (max_ratio > float(edge_ratio_threshold)) & (
        max_growth > float(edge_growth_threshold)
    )
    if neighbor_growth_rings < 0:
        raise ValueError("Stretch-neighbor growth rings cannot be negative.")
    if neighbor_edge_ratio_threshold <= 1.0:
        raise ValueError("Stretch-neighbor edge ratio must be greater than one.")
    if neighbor_edge_growth_threshold < 0.0:
        raise ValueError("Stretch-neighbor edge growth cannot be negative.")
    soft_candidate = (
        candidate
        & (max_ratio > float(neighbor_edge_ratio_threshold))
        & (max_growth > float(neighbor_edge_growth_threshold))
    )
    landmark_vertex_mask = torch.zeros(
        deformed_vertices.shape[0],
        dtype=torch.bool,
    )
    mouth_landmark_ids = set(
        INNER_UPPER_LIP_MEDIAPIPE_IDS
        + INNER_LOWER_LIP_MEDIAPIPE_IDS
        + OUTER_UPPER_LIP_MEDIAPIPE_IDS
        + OUTER_LOWER_LIP_MEDIAPIPE_IDS
    )
    mapped_mouth_vertex_ids = [
        int(vertex_id)
        for landmark_id, vertex_id in mapping.items()
        if int(landmark_id) in mouth_landmark_ids
        and 0 <= int(vertex_id) < deformed_vertices.shape[0]
    ]
    if mapped_mouth_vertex_ids:
        landmark_vertex_mask[
            torch.tensor(mapped_mouth_vertex_ids, dtype=torch.long)
        ] = True
    landmark_faces = landmark_vertex_mask[faces].any(dim=1)
    removed = strict_removed.clone()
    frontier = removed.clone()
    for _ring in range(int(neighbor_growth_rings)):
        if not bool(frontier.any().item()):
            break
        frontier_vertices = torch.zeros(
            deformed_vertices.shape[0],
            dtype=torch.bool,
        )
        frontier_vertices[faces[frontier].reshape(-1)] = True
        touching_frontier = frontier_vertices[faces].any(dim=1)
        grown = (
            touching_frontier
            & soft_candidate
            & ~removed
            & ~landmark_faces
        )
        if not bool(grown.any().item()):
            break
        removed |= grown
        frontier = grown
    removed_ids = torch.nonzero(removed, as_tuple=False).flatten().long()
    return StretchedMouthFaceReport(
        edge_ratio_threshold=float(edge_ratio_threshold),
        edge_growth_threshold=float(edge_growth_threshold),
        neighbor_growth_rings=int(neighbor_growth_rings),
        neighbor_edge_ratio_threshold=float(neighbor_edge_ratio_threshold),
        neighbor_edge_growth_threshold=float(neighbor_edge_growth_threshold),
        region_scale=float(region_scale),
        depth_padding=float(depth_padding),
        candidate_face_count=int(candidate.sum().item()),
        strict_removed_face_count=int(strict_removed.sum().item()),
        grown_removed_face_count=int((removed & ~strict_removed).sum().item()),
        removed_face_count=int(removed_ids.numel()),
        removed_face_ids=[int(value) for value in removed_ids.tolist()],
    )


def retain_aperture_connected_face_removals(
    faces: torch.Tensor,
    aperture_face_ids: Sequence[int],
    stretch_face_ids: Sequence[int],
) -> MouthRemovalConnectivityReport:
    """Keep only the dominant aperture-connected removal component."""

    cpu_faces = faces.detach().cpu().long().contiguous()
    if cpu_faces.ndim != 2 or cpu_faces.shape[1] != 3:
        raise ValueError(
            "Mouth-removal connectivity filtering expects triangular faces."
        )
    face_count = int(cpu_faces.shape[0])
    aperture_ids = sorted(set(int(value) for value in aperture_face_ids))
    stretch_ids = sorted(set(int(value) for value in stretch_face_ids))
    invalid_ids = [
        value
        for value in aperture_ids + stretch_ids
        if value < 0 or value >= face_count
    ]
    if invalid_ids:
        raise ValueError(
            "Mouth-removal face ids are outside the source face range: "
            + ", ".join(str(value) for value in invalid_ids[:8])
        )

    candidate_ids = sorted(set(aperture_ids) | set(stretch_ids))
    if not candidate_ids:
        return MouthRemovalConnectivityReport(
            aperture_seed_face_count=0,
            stretch_candidate_face_count=0,
            candidate_face_count=0,
            connected_component_count=0,
            seeded_component_count=0,
            significant_seeded_component_count=0,
            discarded_aperture_component_count=0,
            discarded_aperture_face_count=0,
            removed_face_count=0,
            restored_disconnected_component_count=0,
            restored_disconnected_face_count=0,
            removed_face_ids=[],
        )

    candidate_faces = cpu_faces[
        torch.tensor(candidate_ids, dtype=torch.long)
    ].numpy()
    parent = np.arange(len(candidate_ids), dtype=np.int64)
    component_size = np.ones(len(candidate_ids), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if component_size[left_root] < component_size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        component_size[left_root] += component_size[right_root]

    edge_owner: dict[tuple[int, int], int] = {}
    for local_face_id, face in enumerate(candidate_faces):
        for corner in range(3):
            start = int(face[corner])
            end = int(face[(corner + 1) % 3])
            edge = (start, end) if start < end else (end, start)
            previous_owner = edge_owner.setdefault(edge, local_face_id)
            if previous_owner != local_face_id:
                union(local_face_id, previous_owner)

    roots = np.asarray([find(index) for index in range(len(candidate_ids))])
    component_roots = set(int(value) for value in roots.tolist())
    candidate_to_local = {
        face_id: local_id for local_id, face_id in enumerate(candidate_ids)
    }
    seeded_roots = {
        int(roots[candidate_to_local[face_id]]) for face_id in aperture_ids
    }
    seed_counts = {
        root: sum(
            int(roots[candidate_to_local[face_id]]) == root
            for face_id in aperture_ids
        )
        for root in seeded_roots
    }
    component_counts = Counter(int(root) for root in roots.tolist())
    if seeded_roots:
        significant_seeded_roots = {
            max(
                seeded_roots,
                key=lambda root: (
                    seed_counts[root],
                    component_counts[root],
                    -root,
                ),
            )
        }
    else:
        significant_seeded_roots = set()
    retained_local = np.asarray(
        [int(root) in significant_seeded_roots for root in roots],
        dtype=bool,
    )
    removed_face_ids = [
        candidate_ids[index]
        for index in np.flatnonzero(retained_local).tolist()
    ]
    discarded_seeded_roots = seeded_roots - significant_seeded_roots
    restored_roots = component_roots - significant_seeded_roots
    return MouthRemovalConnectivityReport(
        aperture_seed_face_count=len(aperture_ids),
        stretch_candidate_face_count=len(stretch_ids),
        candidate_face_count=len(candidate_ids),
        connected_component_count=len(component_roots),
        seeded_component_count=len(seeded_roots),
        significant_seeded_component_count=len(significant_seeded_roots),
        discarded_aperture_component_count=len(discarded_seeded_roots),
        discarded_aperture_face_count=sum(
            seed_counts[root] for root in discarded_seeded_roots
        ),
        removed_face_count=len(removed_face_ids),
        restored_disconnected_component_count=len(restored_roots),
        restored_disconnected_face_count=(
            len(candidate_ids) - len(removed_face_ids)
        ),
        removed_face_ids=removed_face_ids,
    )


def _undirected_cycle_basis(
    adjacency: Mapping[int, Sequence[int]],
) -> list[list[int]]:
    """Return edge-ordered Paton cycle-basis paths."""

    cycles: list[list[int]] = []
    remaining = set(adjacency)
    while remaining:
        root = min(remaining)
        stack = [root]
        predecessor = {root: root}
        used: dict[int, set[int]] = {root: set()}
        while stack:
            current = stack.pop()
            current_used = used[current]
            for neighbor in adjacency[current]:
                if neighbor not in used:
                    predecessor[neighbor] = current
                    stack.append(neighbor)
                    used[neighbor] = {current}
                elif neighbor == current:
                    cycles.append([current])
                elif neighbor not in current_used:
                    neighbor_used = used[neighbor]
                    cycle = [neighbor, current]
                    parent = predecessor[current]
                    while parent not in neighbor_used:
                        cycle.append(parent)
                        parent = predecessor[parent]
                    cycle.append(parent)
                    cycles.append(cycle)
                    used[neighbor].add(current)
        remaining.difference_update(predecessor)
    return cycles


def _ear_clip_projected_cycle(
    cycle: Sequence[int],
    projected_vertices: torch.Tensor,
) -> list[list[int]]:
    """Triangulate a simple projected cycle while retaining its winding."""

    points = projected_vertices[torch.tensor(cycle, dtype=torch.long)].double()
    signed_area_twice = float(
        (
            points[:, 0] * torch.roll(points[:, 1], shifts=-1)
            - points[:, 1] * torch.roll(points[:, 0], shifts=-1)
        ).sum().item()
    )
    if abs(signed_area_twice) <= 1.0e-12:
        return []
    orientation = 1.0 if signed_area_twice > 0.0 else -1.0
    remaining = list(range(len(cycle)))
    triangles: list[list[int]] = []

    def cross(first: int, second: int, third: int) -> float:
        first_edge = points[second] - points[first]
        second_edge = points[third] - points[first]
        return float(
            (first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]).item()
        )

    def inside_triangle(
        point: int,
        first: int,
        second: int,
        third: int,
    ) -> bool:
        return min(
            orientation * cross(first, second, point),
            orientation * cross(second, third, point),
            orientation * cross(third, first, point),
        ) >= -1.0e-12

    while len(remaining) > 3:
        found_ear = False
        for offset, current in enumerate(remaining):
            previous = remaining[offset - 1]
            following = remaining[(offset + 1) % len(remaining)]
            if orientation * cross(previous, current, following) <= 1.0e-12:
                continue
            if any(
                inside_triangle(point, previous, current, following)
                for point in remaining
                if point not in (previous, current, following)
            ):
                continue
            triangles.append(
                [int(cycle[previous]), int(cycle[current]), int(cycle[following])]
            )
            remaining.pop(offset)
            found_ear = True
            break
        if not found_ear:
            return []
    triangles.append([int(cycle[index]) for index in remaining])
    return triangles


def _mesh_boundary_edges(faces: torch.Tensor) -> torch.Tensor:
    faces = faces.detach().cpu().long().contiguous()
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Boundary detection expects triangular faces.")
    if faces.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long)
    face_array = faces.numpy()
    edges = np.concatenate(
        (
            face_array[:, (0, 1)],
            face_array[:, (1, 2)],
            face_array[:, (2, 0)],
        ),
        axis=0,
    )
    edges.sort(axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    ordered_edges = edges[order]
    group_start = np.ones(len(ordered_edges), dtype=bool)
    group_start[1:] = np.any(ordered_edges[1:] != ordered_edges[:-1], axis=1)
    group_ids = np.flatnonzero(group_start)
    group_counts = np.diff(np.append(group_ids, len(ordered_edges)))
    boundary = ordered_edges[group_ids[group_counts == 1]]
    return torch.from_numpy(boundary.copy()).long().contiguous()


def _new_boundary_edge_mask(
    boundary_edges: torch.Tensor,
    reference_faces: torch.Tensor | None,
    *,
    reference_boundary_edges: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Mark boundary edges that were not boundaries in the reference topology."""

    boundary_edges = boundary_edges.detach().cpu().long().contiguous()
    if reference_faces is None and reference_boundary_edges is None:
        return torch.ones(boundary_edges.shape[0], dtype=torch.bool), 0
    if reference_boundary_edges is None:
        assert reference_faces is not None
        reference_boundary_edges = _mesh_boundary_edges(reference_faces)
    else:
        reference_boundary_edges = (
            reference_boundary_edges.detach().cpu().long().contiguous()
        )
    if boundary_edges.numel() == 0 or reference_boundary_edges.numel() == 0:
        return (
            torch.ones(boundary_edges.shape[0], dtype=torch.bool),
            int(reference_boundary_edges.shape[0]),
        )
    maximum_vertex_id = max(
        int(boundary_edges.max().item()),
        int(reference_boundary_edges.max().item()),
    )
    key_base = maximum_vertex_id + 1
    boundary_keys = boundary_edges[:, 0] * key_base + boundary_edges[:, 1]
    reference_keys = (
        reference_boundary_edges[:, 0] * key_base
        + reference_boundary_edges[:, 1]
    )
    return (
        ~torch.isin(boundary_keys, reference_keys),
        int(reference_boundary_edges.shape[0]),
    )


def fill_secondary_mouth_boundary_holes(
    neutral_vertices: torch.Tensor,
    deformed_vertices: torch.Tensor,
    faces: torch.Tensor,
    mapping: Mapping[int, int],
    *,
    reference_faces: torch.Tensor | None = None,
    reference_boundary_edges: torch.Tensor | None = None,
    horizontal_axis: int = 0,
    depth_axis: int = 1,
    vertical_axis: int = 2,
    depth_padding: float = 0.12,
    maximum_area_ratio: float = 0.15,
    minimum_aperture_horizontal_span: float = 0.8,
    minimum_aperture_vertical_span: float = 0.8,
    edge_ratio_threshold: float = 3.0,
    edge_growth_threshold: float = 0.02,
    maximum_ear_clip_shifts: int = 8,
    minimum_new_boundary_edge_fraction: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, MouthHoleFillReport]:
    """Cap small cut-created mouth cycles while preserving existing boundaries."""

    neutral_vertices = neutral_vertices.detach().cpu().float().contiguous()
    deformed_vertices = deformed_vertices.detach().cpu().float().contiguous()
    faces = faces.detach().cpu().long().contiguous()
    if neutral_vertices.shape != deformed_vertices.shape:
        raise ValueError("Neutral and deformed vertices must have matching shapes.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Mouth hole filling expects triangular faces.")
    if depth_padding < 0.0:
        raise ValueError("Mouth hole depth padding cannot be negative.")
    if not 0.0 < maximum_area_ratio < 1.0:
        raise ValueError("Mouth hole maximum area ratio must be in (0, 1).")
    if maximum_ear_clip_shifts <= 0:
        raise ValueError("Maximum ear-clip shifts must be positive.")
    if not 0.0 <= minimum_new_boundary_edge_fraction <= 1.0:
        raise ValueError("Minimum new-boundary edge fraction must be in [0, 1].")

    vertex_count = int(deformed_vertices.shape[0])
    lip_ids = list(
        dict.fromkeys(
            _ordered_mapped_vertex_ids(
                mapping,
                INNER_UPPER_LIP_MEDIAPIPE_IDS + INNER_LOWER_LIP_MEDIAPIPE_IDS,
                vertex_count,
            )
        )
    )
    if len(lip_ids) < 4:
        raise ValueError("Mouth hole filling requires mapped inner-lip vertices.")

    boundary_edges = _mesh_boundary_edges(faces)
    new_boundary_mask, reference_boundary_edge_count = _new_boundary_edge_mask(
        boundary_edges,
        reference_faces,
        reference_boundary_edges=reference_boundary_edges,
    )
    new_boundary_edges = boundary_edges[new_boundary_mask]
    new_boundary_edge_keys = {
        (int(first), int(second))
        for first, second in new_boundary_edges.tolist()
    }
    boundary_midpoints = deformed_vertices[boundary_edges].mean(dim=1)
    lip_points = deformed_vertices[torch.tensor(lip_ids, dtype=torch.long)]
    projected_lips = lip_points[:, (horizontal_axis, vertical_axis)]
    projected_min = projected_lips.amin(dim=0)
    projected_max = projected_lips.amax(dim=0)
    projected_span = (projected_max - projected_min).clamp_min(1.0e-6)
    mouth_edge_mask = (
        (boundary_midpoints[:, horizontal_axis] >= projected_min[0] - 0.45 * projected_span[0])
        & (boundary_midpoints[:, horizontal_axis] <= projected_max[0] + 0.45 * projected_span[0])
        & (boundary_midpoints[:, vertical_axis] >= projected_min[1] - 0.45 * projected_span[1])
        & (boundary_midpoints[:, vertical_axis] <= projected_max[1] + 0.45 * projected_span[1])
        & (boundary_midpoints[:, depth_axis] >= lip_points[:, depth_axis].amin() - float(depth_padding))
        & (boundary_midpoints[:, depth_axis] <= lip_points[:, depth_axis].amax() + float(depth_padding))
    )
    mouth_boundary_edges = boundary_edges[mouth_edge_mask]
    adjacency: dict[int, list[int]] = {}
    for first, second in mouth_boundary_edges.tolist():
        adjacency.setdefault(int(first), []).append(int(second))
        adjacency.setdefault(int(second), []).append(int(first))
    cycles = [cycle for cycle in _undirected_cycle_basis(adjacency) if len(cycle) >= 3]

    cycle_metrics: list[tuple[list[int], float, float, float, float]] = []
    for cycle in cycles:
        points = deformed_vertices[torch.tensor(cycle, dtype=torch.long)]
        projected = points[:, (horizontal_axis, vertical_axis)]
        signed_area = 0.5 * float(
            (
                projected[:, 0] * torch.roll(projected[:, 1], shifts=-1)
                - projected[:, 1] * torch.roll(projected[:, 0], shifts=-1)
            ).sum().item()
        )
        horizontal_span = float(
            (points[:, horizontal_axis].amax() - points[:, horizontal_axis].amin()).item()
        )
        vertical_span = float(
            (points[:, vertical_axis].amax() - points[:, vertical_axis].amin()).item()
        )
        cycle_edge_count = len(cycle)
        new_cycle_edge_count = sum(
            (
                min(int(cycle[index]), int(cycle[(index + 1) % cycle_edge_count])),
                max(int(cycle[index]), int(cycle[(index + 1) % cycle_edge_count])),
            )
            in new_boundary_edge_keys
            for index in range(cycle_edge_count)
        )
        cycle_metrics.append(
            (
                cycle,
                abs(signed_area),
                horizontal_span,
                vertical_span,
                float(new_cycle_edge_count) / float(cycle_edge_count),
            )
        )

    ranked_cycles = sorted(
        (
            index
            for index, metrics in enumerate(cycle_metrics)
            if metrics[4] >= float(minimum_new_boundary_edge_fraction)
        ),
        key=lambda index: cycle_metrics[index][1],
        reverse=True,
    )
    primary_cycle_ids = set(ranked_cycles[:1])
    largest_cycle_area = (
        cycle_metrics[ranked_cycles[0]][1] if ranked_cycles else 0.0
    )
    minimum_area = float(projected_span.prod().item()) * 1.0e-7
    candidate_cycles: list[tuple[list[int], float]] = []
    preserved_count = 0
    preexisting_count = 0
    for index, (
        cycle,
        area,
        horizontal_span,
        vertical_span,
        new_boundary_fraction,
    ) in enumerate(cycle_metrics):
        spans_aperture = (
            horizontal_span
            >= float(minimum_aperture_horizontal_span) * float(projected_span[0].item())
            and vertical_span
            >= float(minimum_aperture_vertical_span) * float(projected_span[1].item())
        )
        is_secondary = (
            largest_cycle_area > 0.0
            and area >= minimum_area
            and area <= float(maximum_area_ratio) * largest_cycle_area
        )
        is_cut_created = (
            new_boundary_fraction >= float(minimum_new_boundary_edge_fraction)
        )
        if not is_cut_created:
            preexisting_count += 1
            preserved_count += 1
        elif index in primary_cycle_ids or spans_aperture or not is_secondary:
            preserved_count += 1
        else:
            candidate_cycles.append((cycle, area))

    lip_depth_center = lip_points[:, depth_axis].mean()
    front_direction = (
        -1.0
        if lip_depth_center - deformed_vertices[:, depth_axis].amin()
        < deformed_vertices[:, depth_axis].amax() - lip_depth_center
        else 1.0
    )
    projected_deformed = deformed_vertices[:, (horizontal_axis, vertical_axis)]
    added_faces: list[list[int]] = []
    added_neutral_vertices: list[torch.Tensor] = []
    added_deformed_vertices: list[torch.Tensor] = []
    filled_areas: list[float] = []
    rejected_count = 0
    reoriented_count = 0

    for cycle, area in candidate_cycles:
        cycle_points = deformed_vertices[torch.tensor(cycle, dtype=torch.long)]
        cycle_center = cycle_points.mean(dim=0)
        _, _, local_axes = torch.linalg.svd(
            cycle_points - cycle_center,
            full_matrices=False,
        )
        local_projection = (deformed_vertices - cycle_center) @ local_axes[:2].T
        best_faces: list[list[int]] = []
        best_score: tuple[int, float, float] | None = None
        face_deformed_vertices = deformed_vertices
        for winding in (cycle, list(reversed(cycle))):
            for projection in (projected_deformed, local_projection):
                shift_count = min(len(winding), int(maximum_ear_clip_shifts))
                shifts = sorted(
                    {
                        (index * len(winding)) // shift_count
                        for index in range(shift_count)
                    }
                )
                for shift in shifts:
                    rotated = winding[shift:] + winding[:shift]
                    candidate_faces = _ear_clip_projected_cycle(rotated, projection)
                    if not candidate_faces:
                        continue
                    candidate_tensor = torch.tensor(candidate_faces, dtype=torch.long)
                    neutral_triangles = neutral_vertices[candidate_tensor]
                    deformed_triangles = deformed_vertices[candidate_tensor]
                    neutral_normals = torch.cross(
                        neutral_triangles[:, 1] - neutral_triangles[:, 0],
                        neutral_triangles[:, 2] - neutral_triangles[:, 0],
                        dim=1,
                    )
                    deformed_normals = torch.cross(
                        deformed_triangles[:, 1] - deformed_triangles[:, 0],
                        deformed_triangles[:, 2] - deformed_triangles[:, 0],
                        dim=1,
                    )
                    neutral_lengths = torch.stack(
                        tuple(
                            (
                                neutral_triangles[:, (corner + 1) % 3]
                                - neutral_triangles[:, corner]
                            ).norm(dim=1)
                            for corner in range(3)
                        ),
                        dim=1,
                    )
                    deformed_lengths = torch.stack(
                        tuple(
                            (
                                deformed_triangles[:, (corner + 1) % 3]
                                - deformed_triangles[:, corner]
                            ).norm(dim=1)
                            for corner in range(3)
                        ),
                        dim=1,
                    )
                    severe_stretch = (
                        (deformed_lengths / neutral_lengths.clamp_min(1.0e-8)).amax(dim=1)
                        > float(edge_ratio_threshold)
                    ) & (
                        (deformed_lengths - neutral_lengths).amax(dim=1)
                        > float(edge_growth_threshold)
                    )
                    orientation = (
                        F.normalize(neutral_normals, dim=1, eps=1.0e-8)
                        * F.normalize(deformed_normals, dim=1, eps=1.0e-8)
                    ).sum(dim=1)
                    degenerate = (
                        (neutral_normals.norm(dim=1) <= 1.0e-10)
                        | (deformed_normals.norm(dim=1) <= 1.0e-10)
                    )
                    bad_count = int(
                        (severe_stretch | (orientation < 0.0) | degenerate).sum().item()
                    )
                    score = (
                        bad_count,
                        -float(orientation.min().item()),
                        float(deformed_lengths.max().item()),
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_faces = candidate_faces
                    if bad_count == 0 and float(orientation.min().item()) >= 0.25:
                        break
                if best_score is not None and best_score[0] == 0 and -best_score[1] >= 0.25:
                    break
            if best_score is not None and best_score[0] == 0 and -best_score[1] >= 0.25:
                break

        if best_score is None or best_score[0] > 0:
            neutral_center = neutral_vertices[
                torch.tensor(cycle, dtype=torch.long)
            ].mean(dim=0)
            deformed_center = deformed_vertices[
                torch.tensor(cycle, dtype=torch.long)
            ].mean(dim=0)
            center_id = vertex_count + len(added_neutral_vertices)
            fan_faces = [
                [cycle[index], cycle[(index + 1) % len(cycle)], center_id]
                for index in range(len(cycle))
            ]
            candidate_neutral = torch.cat(
                (
                    neutral_vertices,
                    torch.stack((*added_neutral_vertices, neutral_center)),
                ),
                dim=0,
            )
            candidate_deformed = torch.cat(
                (
                    deformed_vertices,
                    torch.stack((*added_deformed_vertices, deformed_center)),
                ),
                dim=0,
            )
            candidate_tensor = torch.tensor(fan_faces, dtype=torch.long)
            neutral_triangles = candidate_neutral[candidate_tensor]
            deformed_triangles = candidate_deformed[candidate_tensor]
            neutral_normals = torch.cross(
                neutral_triangles[:, 1] - neutral_triangles[:, 0],
                neutral_triangles[:, 2] - neutral_triangles[:, 0],
                dim=1,
            )
            deformed_normals = torch.cross(
                deformed_triangles[:, 1] - deformed_triangles[:, 0],
                deformed_triangles[:, 2] - deformed_triangles[:, 0],
                dim=1,
            )
            neutral_lengths = torch.stack(
                tuple(
                    (
                        neutral_triangles[:, (corner + 1) % 3]
                        - neutral_triangles[:, corner]
                    ).norm(dim=1)
                    for corner in range(3)
                ),
                dim=1,
            )
            deformed_lengths = torch.stack(
                tuple(
                    (
                        deformed_triangles[:, (corner + 1) % 3]
                        - deformed_triangles[:, corner]
                    ).norm(dim=1)
                    for corner in range(3)
                ),
                dim=1,
            )
            severe_stretch = (
                (deformed_lengths / neutral_lengths.clamp_min(1.0e-8)).amax(dim=1)
                > float(edge_ratio_threshold)
            ) & (
                (deformed_lengths - neutral_lengths).amax(dim=1)
                > float(edge_growth_threshold)
            )
            orientation = (
                F.normalize(neutral_normals, dim=1, eps=1.0e-8)
                * F.normalize(deformed_normals, dim=1, eps=1.0e-8)
            ).sum(dim=1)
            degenerate = (
                (neutral_normals.norm(dim=1) <= 1.0e-10)
                | (deformed_normals.norm(dim=1) <= 1.0e-10)
            )
            if bool(
                (severe_stretch | (orientation < 0.0) | degenerate).any().item()
            ):
                rejected_count += 1
                continue
            added_neutral_vertices.append(neutral_center)
            added_deformed_vertices.append(deformed_center)
            best_faces = fan_faces
            face_deformed_vertices = candidate_deformed
        oriented_faces = []
        for first, second, third in best_faces:
            normal = torch.cross(
                face_deformed_vertices[second] - face_deformed_vertices[first],
                face_deformed_vertices[third] - face_deformed_vertices[first],
                dim=0,
            )
            if float(normal[depth_axis].item()) * front_direction < 0.0:
                second, third = third, second
                reoriented_count += 1
            oriented_faces.append([first, second, third])
        best_faces = oriented_faces
        added_faces.extend(best_faces)
        filled_areas.append(float(area))

    final_neutral_vertices = (
        torch.cat((neutral_vertices, torch.stack(added_neutral_vertices)), dim=0)
        if added_neutral_vertices
        else neutral_vertices
    )
    final_deformed_vertices = (
        torch.cat((deformed_vertices, torch.stack(added_deformed_vertices)), dim=0)
        if added_deformed_vertices
        else deformed_vertices
    )
    final_faces = (
        torch.cat((faces, torch.tensor(added_faces, dtype=torch.long)), dim=0).contiguous()
        if added_faces
        else faces
    )
    final_boundary_edge_count = int(_mesh_boundary_edges(final_faces).shape[0])
    report = MouthHoleFillReport(
        boundary_edge_count_before=int(boundary_edges.shape[0]),
        reference_boundary_edge_count=int(reference_boundary_edge_count),
        new_boundary_edge_count=int(new_boundary_edges.shape[0]),
        mouth_boundary_edge_count_before=int(mouth_boundary_edges.shape[0]),
        mouth_cycle_count=len(cycles),
        preexisting_cycle_count=int(preexisting_count),
        preserved_aperture_cycle_count=preserved_count,
        candidate_hole_cycle_count=len(candidate_cycles),
        filled_hole_cycle_count=len(filled_areas),
        rejected_hole_cycle_count=rejected_count,
        added_face_count=len(added_faces),
        added_vertex_count=len(added_neutral_vertices),
        reoriented_face_count=reoriented_count,
        boundary_edge_count_after=final_boundary_edge_count,
        maximum_area_ratio=float(maximum_area_ratio),
        minimum_new_boundary_edge_fraction=float(
            minimum_new_boundary_edge_fraction
        ),
        largest_cycle_area=float(largest_cycle_area),
        filled_cycle_areas=filled_areas,
    )
    return final_neutral_vertices, final_deformed_vertices, final_faces, report


def bridge_excess_mouth_corner_gap(
    neutral_vertices: torch.Tensor,
    deformed_vertices: torch.Tensor,
    faces: torch.Tensor,
    mapping: Mapping[int, int],
    *,
    reference_faces: torch.Tensor | None = None,
    reference_boundary_edges: torch.Tensor | None = None,
    horizontal_axis: int = 0,
    depth_axis: int = 1,
    vertical_axis: int = 2,
    depth_padding: float = 0.12,
    corner_inset_ratio: float = 0.13,
    minimum_lateral_overshoot_ratio: float = 0.12,
    maximum_repaired_sides: int = 1,
    chain_padding_vertices: int = 1,
    minimum_horizontal_span: float = 0.8,
    minimum_vertical_span: float = 0.8,
    edge_ratio_threshold: float = 3.0,
    edge_growth_threshold: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, MouthCornerBridgeReport]:
    """Bridge only a mouth corner whose cut loops extend past the lip contour."""

    neutral_vertices = neutral_vertices.detach().cpu().float().contiguous()
    deformed_vertices = deformed_vertices.detach().cpu().float().contiguous()
    faces = faces.detach().cpu().long().contiguous()
    if neutral_vertices.shape != deformed_vertices.shape:
        raise ValueError("Neutral and deformed vertices must have matching shapes.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Mouth corner bridging expects triangular faces.")
    if not 0.0 < corner_inset_ratio < 0.5:
        raise ValueError("Mouth corner inset ratio must be in (0, 0.5).")
    if minimum_lateral_overshoot_ratio < 0.0:
        raise ValueError("Minimum lateral overshoot ratio cannot be negative.")
    if maximum_repaired_sides < 1:
        raise ValueError("At least one mouth side must be eligible for repair.")
    if chain_padding_vertices < 1:
        raise ValueError("Mouth corner chain padding must be at least one vertex.")

    vertex_count = int(deformed_vertices.shape[0])
    lip_ids = list(
        dict.fromkeys(
            _ordered_mapped_vertex_ids(
                mapping,
                INNER_UPPER_LIP_MEDIAPIPE_IDS + INNER_LOWER_LIP_MEDIAPIPE_IDS,
                vertex_count,
            )
        )
    )
    if len(lip_ids) < 4:
        raise ValueError("Mouth corner bridging requires mapped inner-lip vertices.")
    lip_points = deformed_vertices[torch.tensor(lip_ids, dtype=torch.long)]
    projected_lips = lip_points[:, (horizontal_axis, vertical_axis)]
    projected_min = projected_lips.amin(dim=0)
    projected_max = projected_lips.amax(dim=0)
    projected_span = (projected_max - projected_min).clamp_min(1.0e-6)

    boundary_edges = _mesh_boundary_edges(faces)
    new_boundary_mask, _ = _new_boundary_edge_mask(
        boundary_edges,
        reference_faces,
        reference_boundary_edges=reference_boundary_edges,
    )
    boundary_midpoints = deformed_vertices[boundary_edges].mean(dim=1)
    mouth_edge_mask = (
        (boundary_midpoints[:, horizontal_axis] >= projected_min[0] - 0.45 * projected_span[0])
        & (boundary_midpoints[:, horizontal_axis] <= projected_max[0] + 0.45 * projected_span[0])
        & (boundary_midpoints[:, vertical_axis] >= projected_min[1] - 0.45 * projected_span[1])
        & (boundary_midpoints[:, vertical_axis] <= projected_max[1] + 0.45 * projected_span[1])
        & (boundary_midpoints[:, depth_axis] >= lip_points[:, depth_axis].amin() - float(depth_padding))
        & (boundary_midpoints[:, depth_axis] <= lip_points[:, depth_axis].amax() + float(depth_padding))
    )
    mouth_boundary_edges = boundary_edges[mouth_edge_mask & new_boundary_mask]
    adjacency: dict[int, list[int]] = {}
    for first, second in mouth_boundary_edges.tolist():
        adjacency.setdefault(int(first), []).append(int(second))
        adjacency.setdefault(int(second), []).append(int(first))

    components: list[list[int]] = []
    visited: set[int] = set()
    for start in adjacency:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(component)

    def ordered_cycle(component: list[int]) -> list[int]:
        if len(component) < 3 or any(len(adjacency[value]) != 2 for value in component):
            return []
        start = min(component)
        order = [start]
        previous = -1
        current = start
        while True:
            candidates = [value for value in adjacency[current] if value != previous]
            if not candidates:
                return []
            following = candidates[0]
            if following == start:
                break
            if following in order:
                return []
            order.append(following)
            previous, current = current, following
        return order if len(order) == len(component) else []

    mouth_cycles: list[list[int]] = []
    for component in components:
        cycle = ordered_cycle(component)
        if not cycle:
            continue
        points = deformed_vertices[torch.tensor(cycle, dtype=torch.long)]
        horizontal_span = points[:, horizontal_axis].amax() - points[:, horizontal_axis].amin()
        vertical_span = points[:, vertical_axis].amax() - points[:, vertical_axis].amin()
        if (
            horizontal_span >= float(minimum_horizontal_span) * projected_span[0]
            and vertical_span >= float(minimum_vertical_span) * projected_span[1]
        ):
            mouth_cycles.append(cycle)

    def projected_cycle_area(cycle: list[int]) -> float:
        points = deformed_vertices[
            torch.tensor(cycle, dtype=torch.long)
        ][:, (horizontal_axis, vertical_axis)]
        return abs(
            0.5
            * float(
                (
                    points[:, 0] * torch.roll(points[:, 1], shifts=-1)
                    - points[:, 1] * torch.roll(points[:, 0], shifts=-1)
                ).sum().item()
            )
        )

    mouth_cycles = sorted(mouth_cycles, key=projected_cycle_area, reverse=True)[:2]
    lateral_overshoot_ratios = {"left": 0.0, "right": 0.0}
    if len(mouth_cycles) == 2:
        cycle_ids = torch.tensor(
            mouth_cycles[0] + mouth_cycles[1],
            dtype=torch.long,
        )
        cycle_horizontal = deformed_vertices[cycle_ids, horizontal_axis]
        lateral_overshoot_ratios = {
            "left": max(
                0.0,
                float((projected_min[0] - cycle_horizontal.amin()).item())
                / float(projected_span[0].item()),
            ),
            "right": max(
                0.0,
                float((cycle_horizontal.amax() - projected_max[0]).item())
                / float(projected_span[0].item()),
            ),
        }
    candidate_sides = [
        side
        for side in sorted(
            lateral_overshoot_ratios,
            key=lateral_overshoot_ratios.get,
            reverse=True,
        )
        if lateral_overshoot_ratios[side] >= float(minimum_lateral_overshoot_ratio)
    ][: int(maximum_repaired_sides)]

    def longest_corner_chain(cycle: list[int], side: str) -> list[int]:
        if side == "left":
            threshold = projected_min[0] + float(corner_inset_ratio) * projected_span[0]
            selected = [
                bool((deformed_vertices[value, horizontal_axis] <= threshold).item())
                for value in cycle
            ]
        else:
            threshold = projected_max[0] - float(corner_inset_ratio) * projected_span[0]
            selected = [
                bool((deformed_vertices[value, horizontal_axis] >= threshold).item())
                for value in cycle
            ]
        starts = [
            index
            for index, value in enumerate(selected)
            if value and not selected[index - 1]
        ]
        runs: list[list[int]] = []
        for start in starts:
            run = []
            index = start
            while selected[index]:
                run.append(cycle[index])
                index = (index + 1) % len(cycle)
                if index == start:
                    break
            runs.append(run)
        if not runs:
            return []
        chain = max(runs, key=len)
        positions = {value: index for index, value in enumerate(cycle)}
        start_index = positions[chain[0]]
        end_index = positions[chain[-1]]
        padded = [
            cycle[(start_index - offset) % len(cycle)]
            for offset in range(chain_padding_vertices, 0, -1)
        ]
        padded.extend(chain)
        padded.extend(
            cycle[(end_index + offset) % len(cycle)]
            for offset in range(1, chain_padding_vertices + 1)
        )
        chain = list(dict.fromkeys(padded))
        if (
            deformed_vertices[chain[0], vertical_axis]
            < deformed_vertices[chain[-1], vertical_axis]
        ):
            chain.reverse()
        return chain

    def triangle_quality(vertex_ids: list[int]) -> tuple[float, bool, bool, bool]:
        index = torch.tensor(vertex_ids, dtype=torch.long)
        neutral_triangle = neutral_vertices[index]
        deformed_triangle = deformed_vertices[index]
        neutral_normal = torch.cross(
            neutral_triangle[1] - neutral_triangle[0],
            neutral_triangle[2] - neutral_triangle[0],
            dim=0,
        )
        deformed_normal = torch.cross(
            deformed_triangle[1] - deformed_triangle[0],
            deformed_triangle[2] - deformed_triangle[0],
            dim=0,
        )
        neutral_lengths = torch.stack(
            tuple(
                (
                    neutral_triangle[(corner + 1) % 3]
                    - neutral_triangle[corner]
                ).norm()
                for corner in range(3)
            )
        )
        deformed_lengths = torch.stack(
            tuple(
                (
                    deformed_triangle[(corner + 1) % 3]
                    - deformed_triangle[corner]
                ).norm()
                for corner in range(3)
            )
        )
        degenerate = bool(
            (neutral_normal.norm() <= 1.0e-10).item()
            or (deformed_normal.norm() <= 1.0e-10).item()
        )
        orientation = float(
            (
                F.normalize(neutral_normal, dim=0, eps=1.0e-8)
                * F.normalize(deformed_normal, dim=0, eps=1.0e-8)
            ).sum().item()
        )
        flipped = orientation < 0.0
        severe = bool(
            (
                (deformed_lengths / neutral_lengths.clamp_min(1.0e-8)).amax()
                > float(edge_ratio_threshold)
            ).item()
            and (
                (deformed_lengths - neutral_lengths).amax()
                > float(edge_growth_threshold)
            ).item()
        )
        bad_count = int(degenerate) + int(flipped) + int(severe)
        scale = float(projected_span.max().item())
        cost = (
            bad_count * 1.0e6
            + max(0.0, 0.25 - orientation) * 1.0e4
            + float(deformed_lengths.max().item()) / scale * 10.0
            + float(neutral_lengths.max().item()) / scale * 5.0
        )
        return cost, severe, flipped, degenerate

    added_faces: list[list[int]] = []
    repaired_sides: list[str] = []
    reoriented_count = 0
    lip_depth_center = lip_points[:, depth_axis].mean()
    front_direction = (
        -1.0
        if lip_depth_center - deformed_vertices[:, depth_axis].amin()
        < deformed_vertices[:, depth_axis].amax() - lip_depth_center
        else 1.0
    )
    for side in candidate_sides:
        chains = [longest_corner_chain(cycle, side) for cycle in mouth_cycles]
        if len(chains) != 2 or any(len(chain) < 3 for chain in chains):
            continue
        first, second = chains
        same_direction_cost = float(
            (
                deformed_vertices[first[0]] - deformed_vertices[second[0]]
            ).norm().item()
            + (
                deformed_vertices[first[-1]] - deformed_vertices[second[-1]]
            ).norm().item()
        )
        reversed_cost = float(
            (
                deformed_vertices[first[0]] - deformed_vertices[second[-1]]
            ).norm().item()
            + (
                deformed_vertices[first[-1]] - deformed_vertices[second[0]]
            ).norm().item()
        )
        if reversed_cost < same_direction_cost:
            second = list(reversed(second))

        first_count = len(first)
        second_count = len(second)
        costs = [[float("inf")] * second_count for _ in range(first_count)]
        previous: list[list[tuple[int, int, list[int]] | None]] = [
            [None] * second_count for _ in range(first_count)
        ]
        costs[0][0] = 0.0
        for first_index in range(first_count):
            for second_index in range(second_count):
                current_cost = costs[first_index][second_index]
                if not np.isfinite(current_cost):
                    continue
                if first_index + 1 < first_count:
                    triangle = [
                        first[first_index],
                        first[first_index + 1],
                        second[second_index],
                    ]
                    candidate_cost = current_cost + triangle_quality(triangle)[0]
                    if candidate_cost < costs[first_index + 1][second_index]:
                        costs[first_index + 1][second_index] = candidate_cost
                        previous[first_index + 1][second_index] = (
                            first_index,
                            second_index,
                            triangle,
                        )
                if second_index + 1 < second_count:
                    triangle = [
                        first[first_index],
                        second[second_index + 1],
                        second[second_index],
                    ]
                    candidate_cost = current_cost + triangle_quality(triangle)[0]
                    if candidate_cost < costs[first_index][second_index + 1]:
                        costs[first_index][second_index + 1] = candidate_cost
                        previous[first_index][second_index + 1] = (
                            first_index,
                            second_index,
                            triangle,
                        )

        side_faces: list[list[int]] = []
        first_index = first_count - 1
        second_index = second_count - 1
        while first_index > 0 or second_index > 0:
            step = previous[first_index][second_index]
            if step is None:
                side_faces = []
                break
            prior_first, prior_second, triangle = step
            side_faces.append(triangle)
            first_index, second_index = prior_first, prior_second
        if not side_faces:
            continue
        side_faces.reverse()
        for first_vertex, second_vertex, third_vertex in side_faces:
            normal = torch.cross(
                deformed_vertices[second_vertex] - deformed_vertices[first_vertex],
                deformed_vertices[third_vertex] - deformed_vertices[first_vertex],
                dim=0,
            )
            if float(normal[depth_axis].item()) * front_direction < 0.0:
                second_vertex, third_vertex = third_vertex, second_vertex
                reoriented_count += 1
            added_faces.append([first_vertex, second_vertex, third_vertex])
        repaired_sides.append(side)

    severe_count = 0
    flipped_count = 0
    if added_faces:
        for triangle in added_faces:
            _, severe, flipped, _ = triangle_quality(triangle)
            severe_count += int(severe)
            flipped_count += int(flipped)
        final_faces = torch.cat(
            (faces, torch.tensor(added_faces, dtype=torch.long)),
            dim=0,
        ).contiguous()
    else:
        final_faces = faces
    final_boundary_edge_count = int(_mesh_boundary_edges(final_faces).shape[0])
    report = MouthCornerBridgeReport(
        boundary_edge_count_before=int(boundary_edges.shape[0]),
        boundary_edge_count_after=final_boundary_edge_count,
        mouth_boundary_component_count=len(mouth_cycles),
        candidate_side_count=len(candidate_sides),
        repaired_sides=repaired_sides,
        added_face_count=len(added_faces),
        reoriented_face_count=reoriented_count,
        severe_added_face_count=severe_count,
        flipped_added_face_count=flipped_count,
        corner_inset_ratio=float(corner_inset_ratio),
        chain_padding_vertices=int(chain_padding_vertices),
        minimum_lateral_overshoot_ratio=float(minimum_lateral_overshoot_ratio),
        lateral_overshoot_ratios=lateral_overshoot_ratios,
    )
    return neutral_vertices, deformed_vertices, final_faces, report


def smooth_mouth_boundary_loops(
    neutral_vertices: torch.Tensor,
    deformed_vertices: torch.Tensor,
    faces: torch.Tensor,
    mapping: Mapping[int, int],
    *,
    reference_faces: torch.Tensor | None = None,
    reference_boundary_edges: torch.Tensor | None = None,
    horizontal_axis: int = 0,
    depth_axis: int = 1,
    vertical_axis: int = 2,
    depth_padding: float = 0.12,
    minimum_horizontal_span: float = 0.8,
    minimum_vertical_span: float = 0.8,
    smoothing_cycles: int = 8,
    taubin_lambda: float = 0.4,
    taubin_mu: float = -0.42,
    maximum_displacement: float = 0.006,
    blend_rings: int = 5,
    blend_iterations: int = 35,
    preserve_boundary_junctions: bool = False,
    arc_length_weighted: bool = False,
    minimum_component_span: float | None = None,
    minimum_component_vertices: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, MouthBoundarySmoothReport]:
    """Smooth full mouth cut loops and diffuse the correction into nearby faces."""

    neutral_vertices = neutral_vertices.detach().cpu().float().contiguous()
    deformed_vertices = deformed_vertices.detach().cpu().float().contiguous()
    faces = faces.detach().cpu().long().contiguous()
    vertex_count = int(deformed_vertices.shape[0])
    lip_ids = _ordered_mapped_vertex_ids(
        mapping,
        INNER_UPPER_LIP_MEDIAPIPE_IDS + INNER_LOWER_LIP_MEDIAPIPE_IDS,
        vertex_count,
    )
    lip_ids = list(dict.fromkeys(lip_ids))
    if len(lip_ids) < 4:
        raise ValueError("Boundary smoothing requires mapped inner-lip vertices.")
    if smoothing_cycles < 1 or blend_rings < 1 or blend_iterations < 1:
        raise ValueError("Boundary smoothing iteration and ring counts must be positive.")
    if maximum_displacement <= 0.0:
        raise ValueError("Boundary smoothing maximum displacement must be positive.")
    if minimum_component_span is not None and minimum_component_span <= 0.0:
        raise ValueError("Boundary component span must be positive.")
    if minimum_component_vertices < 3:
        raise ValueError("Boundary components must contain at least three vertices.")

    boundary_edges = _mesh_boundary_edges(faces)
    new_boundary_mask, _ = _new_boundary_edge_mask(
        boundary_edges,
        reference_faces,
        reference_boundary_edges=reference_boundary_edges,
    )
    boundary_midpoints = deformed_vertices[boundary_edges].mean(dim=1)
    lip_points = deformed_vertices[torch.tensor(lip_ids, dtype=torch.long)]
    projected_lips = lip_points[:, (horizontal_axis, vertical_axis)]
    projected_min = projected_lips.amin(dim=0)
    projected_max = projected_lips.amax(dim=0)
    projected_span = (projected_max - projected_min).clamp_min(1.0e-6)
    mouth_edge_mask = (
        (boundary_midpoints[:, horizontal_axis] >= projected_min[0] - 0.45 * projected_span[0])
        & (boundary_midpoints[:, horizontal_axis] <= projected_max[0] + 0.45 * projected_span[0])
        & (boundary_midpoints[:, vertical_axis] >= projected_min[1] - 0.45 * projected_span[1])
        & (boundary_midpoints[:, vertical_axis] <= projected_max[1] + 0.45 * projected_span[1])
        & (boundary_midpoints[:, depth_axis] >= lip_points[:, depth_axis].amin() - float(depth_padding))
        & (boundary_midpoints[:, depth_axis] <= lip_points[:, depth_axis].amax() + float(depth_padding))
    )
    mouth_boundary_edges = boundary_edges[mouth_edge_mask & new_boundary_mask]
    adjacency: dict[int, list[int]] = {}
    for first, second in mouth_boundary_edges.tolist():
        adjacency.setdefault(int(first), []).append(int(second))
        adjacency.setdefault(int(second), []).append(int(first))
    components: list[list[int]] = []
    visited: set[int] = set()
    for start in adjacency:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    selected_components = []
    for component in components:
        points = deformed_vertices[torch.tensor(component, dtype=torch.long)]
        horizontal_span = points[:, horizontal_axis].amax() - points[:, horizontal_axis].amin()
        vertical_span = points[:, vertical_axis].amax() - points[:, vertical_axis].amin()
        spans_full_aperture = (
            horizontal_span >= float(minimum_horizontal_span) * projected_span[0]
            and vertical_span >= float(minimum_vertical_span) * projected_span[1]
        )
        horizontal_ratio = float((horizontal_span / projected_span[0]).item())
        vertical_ratio = float((vertical_span / projected_span[1]).item())
        spans_fragment = (
            minimum_component_span is not None
            and len(component) >= int(minimum_component_vertices)
            and max(horizontal_ratio, vertical_ratio)
            >= float(minimum_component_span)
        )
        if spans_full_aperture or spans_fragment:
            selected_components.append(component)
    boundary_ids = sorted({value for component in selected_components for value in component})
    smoothed_neutral = neutral_vertices.clone()
    smoothed_deformed = deformed_vertices.clone()
    blended_vertex_count = 0
    displacement_norms = torch.empty(0)
    quality_rejected_faces: set[int] = set()
    quality_restored_vertices: set[int] = set()
    preserved_boundary_junction_count = 0
    if boundary_ids:
        boundary_tensor = torch.tensor(boundary_ids, dtype=torch.long)
        if preserve_boundary_junctions:
            preserved_boundary_junction_count = sum(
                len(adjacency[vertex_id]) != 2 for vertex_id in boundary_ids
            )
        boundary_local_by_global = {
            vertex_id: local_id
            for local_id, vertex_id in enumerate(boundary_ids)
        }
        boundary_edge_sources: list[int] = []
        boundary_edge_targets: list[int] = []
        two_neighbor_vertices: list[int] = []
        two_neighbor_first: list[int] = []
        two_neighbor_second: list[int] = []
        boundary_degrees = torch.empty(len(boundary_ids), dtype=torch.long)
        for local_id, vertex_id in enumerate(boundary_ids):
            neighbor_ids = [
                boundary_local_by_global[value]
                for value in adjacency[vertex_id]
            ]
            boundary_degrees[local_id] = len(neighbor_ids)
            boundary_edge_sources.extend([local_id] * len(neighbor_ids))
            boundary_edge_targets.extend(neighbor_ids)
            if len(neighbor_ids) == 2:
                two_neighbor_vertices.append(local_id)
                two_neighbor_first.append(neighbor_ids[0])
                two_neighbor_second.append(neighbor_ids[1])
        boundary_edge_sources_tensor = torch.tensor(
            boundary_edge_sources,
            dtype=torch.long,
        )
        boundary_edge_targets_tensor = torch.tensor(
            boundary_edge_targets,
            dtype=torch.long,
        )
        boundary_movable = boundary_degrees > 0
        if preserve_boundary_junctions:
            boundary_movable &= boundary_degrees == 2
        two_neighbor_vertices_tensor = torch.tensor(
            two_neighbor_vertices,
            dtype=torch.long,
        )
        two_neighbor_first_tensor = torch.tensor(
            two_neighbor_first,
            dtype=torch.long,
        )
        two_neighbor_second_tensor = torch.tensor(
            two_neighbor_second,
            dtype=torch.long,
        )

        def boundary_displacement(vertices: torch.Tensor) -> torch.Tensor:
            original = vertices[boundary_tensor][
                :, (horizontal_axis, vertical_axis)
            ]
            projected = original.clone()
            for _ in range(int(smoothing_cycles)):
                for factor in (float(taubin_lambda), float(taubin_mu)):
                    previous = projected.clone()
                    updated = previous.clone()
                    neighbor_totals = torch.zeros_like(previous)
                    neighbor_totals.index_add_(
                        0,
                        boundary_edge_sources_tensor,
                        previous[boundary_edge_targets_tensor],
                    )
                    neighbor_target = neighbor_totals / boundary_degrees.clamp_min(
                        1
                    ).unsqueeze(1)
                    if arc_length_weighted and two_neighbor_vertices:
                        centers = previous[two_neighbor_vertices_tensor]
                        first_points = previous[two_neighbor_first_tensor]
                        second_points = previous[two_neighbor_second_tensor]
                        first_distances = (first_points - centers).norm(dim=1)
                        second_distances = (second_points - centers).norm(dim=1)
                        distance_sum = first_distances + second_distances
                        weighted = (
                            first_points * second_distances.unsqueeze(1)
                            + second_points * first_distances.unsqueeze(1)
                        ) / distance_sum.clamp_min(1.0e-8).unsqueeze(1)
                        valid_weight = distance_sum > 1.0e-8
                        neighbor_target[
                            two_neighbor_vertices_tensor[valid_weight]
                        ] = weighted[valid_weight]
                    updated[boundary_movable] = previous[boundary_movable] + factor * (
                        neighbor_target[boundary_movable]
                        - previous[boundary_movable]
                    )
                    projected = updated
            displacement = projected - original
            lengths = displacement.norm(dim=1, keepdim=True)
            displacement *= (
                float(maximum_displacement) / lengths.clamp_min(float(maximum_displacement))
            ).clamp_max(1.0)
            return displacement

        deformed_seed = boundary_displacement(deformed_vertices)
        neutral_seed = boundary_displacement(neutral_vertices)
        displacement_norms = deformed_seed.norm(dim=1)
        ring_distance = torch.full((vertex_count,), -1, dtype=torch.long)
        ring_distance[boundary_tensor] = 0
        frontier = torch.zeros(vertex_count, dtype=torch.bool)
        frontier[boundary_tensor] = True
        for ring in range(1, int(blend_rings) + 1):
            touching_faces = frontier[faces].any(dim=1)
            neighbors = faces[touching_faces].reshape(-1)
            fresh = torch.unique(neighbors[ring_distance[neighbors] < 0])
            if fresh.numel() == 0:
                break
            frontier = torch.zeros_like(frontier)
            frontier[fresh] = True
            ring_distance[fresh] = ring
        blend_region = (ring_distance >= 0) & (ring_distance < int(blend_rings))
        local_faces = faces[blend_region[faces].any(dim=1)]
        local_directed_edges = torch.cat(
            (
                local_faces[:, (0, 1)], local_faces[:, (1, 0)],
                local_faces[:, (1, 2)], local_faces[:, (2, 1)],
                local_faces[:, (2, 0)], local_faces[:, (0, 2)],
            ),
            dim=0,
        )
        local_edges = local_directed_edges[
            blend_region[local_directed_edges[:, 0]]
        ]
        blend_ids = torch.nonzero(blend_region, as_tuple=False).flatten()
        blend_local_by_global = torch.full(
            (vertex_count,),
            -1,
            dtype=torch.long,
        )
        blend_local_by_global[blend_ids] = torch.arange(len(blend_ids))
        local_sources = blend_local_by_global[local_edges[:, 0]]
        local_targets = blend_local_by_global[local_edges[:, 1]]
        local_target_mask = local_targets >= 0
        local_counts = torch.zeros(len(blend_ids), dtype=torch.float32)
        local_counts.index_add_(
            0,
            local_sources,
            torch.ones(local_sources.shape[0], dtype=torch.float32),
        )
        local_boundary_ids = blend_local_by_global[boundary_tensor]
        local_movable = ring_distance[blend_ids] > 0
        from scipy import sparse

        valid_sources = local_sources[local_target_mask].numpy()
        valid_targets = local_targets[local_target_mask].numpy()
        averaging_weights = (
            1.0 / local_counts[local_sources[local_target_mask]].numpy()
        ).astype(np.float32, copy=False)
        averaging_matrix = sparse.coo_matrix(
            (
                averaging_weights,
                (valid_sources, valid_targets),
            ),
            shape=(len(blend_ids), len(blend_ids)),
            dtype=np.float32,
        ).tocsr()
        local_boundary_array = local_boundary_ids.numpy()
        local_movable_array = local_movable.numpy()

        def diffuse(seed: torch.Tensor) -> torch.Tensor:
            displacement = np.zeros((len(blend_ids), 2), dtype=np.float32)
            seed_array = seed.numpy()
            displacement[local_boundary_array] = seed_array
            for _ in range(int(blend_iterations)):
                average = averaging_matrix @ displacement
                displacement[local_movable_array] = average[local_movable_array]
                displacement[local_boundary_array] = seed_array
            return torch.from_numpy(displacement)

        deformed_displacement = diffuse(deformed_seed)
        neutral_displacement = diffuse(neutral_seed)
        smoothed_deformed[blend_ids, horizontal_axis] += deformed_displacement[:, 0]
        smoothed_deformed[blend_ids, vertical_axis] += deformed_displacement[:, 1]
        smoothed_neutral[blend_ids, horizontal_axis] += neutral_displacement[:, 0]
        smoothed_neutral[blend_ids, vertical_axis] += neutral_displacement[:, 1]
        blended_vertex_count = int(blend_ids.numel())

        quality_face_ids = torch.nonzero(
            blend_region[faces].any(dim=1),
            as_tuple=False,
        ).flatten()
        quality_faces = faces[quality_face_ids]
        original_triangles = deformed_vertices[quality_faces]
        original_normals = torch.cross(
            original_triangles[:, 1] - original_triangles[:, 0],
            original_triangles[:, 2] - original_triangles[:, 0],
            dim=1,
        )
        for _ in range(6):
            neutral_triangles = smoothed_neutral[quality_faces]
            deformed_triangles = smoothed_deformed[quality_faces]
            smoothed_normals = torch.cross(
                deformed_triangles[:, 1] - deformed_triangles[:, 0],
                deformed_triangles[:, 2] - deformed_triangles[:, 0],
                dim=1,
            )
            orientation_dot = (
                F.normalize(original_normals, dim=1, eps=1.0e-8)
                * F.normalize(smoothed_normals, dim=1, eps=1.0e-8)
            ).sum(dim=1)
            neutral_edges = torch.stack(
                tuple(
                    (
                        neutral_triangles[:, (corner + 1) % 3]
                        - neutral_triangles[:, corner]
                    ).norm(dim=1)
                    for corner in range(3)
                ),
                dim=1,
            )
            deformed_edges = torch.stack(
                tuple(
                    (
                        deformed_triangles[:, (corner + 1) % 3]
                        - deformed_triangles[:, corner]
                    ).norm(dim=1)
                    for corner in range(3)
                ),
                dim=1,
            )
            severe = (
                (deformed_edges / neutral_edges.clamp_min(1.0e-8)).amax(dim=1) > 3.0
            ) & ((deformed_edges - neutral_edges).amax(dim=1) > 0.02)
            rejected = torch.nonzero(
                (orientation_dot < 0.0) | severe,
                as_tuple=False,
            ).flatten()
            if rejected.numel() == 0:
                break
            quality_rejected_faces.update(
                int(value) for value in quality_face_ids[rejected].tolist()
            )
            restore_ids = torch.unique(quality_faces[rejected].reshape(-1))
            neighbor_mask = torch.isin(local_directed_edges[:, 0], restore_ids)
            restore_ids = torch.unique(
                torch.cat((restore_ids, local_directed_edges[neighbor_mask, 1]))
            )
            quality_restored_vertices.update(int(value) for value in restore_ids.tolist())
            smoothed_neutral[restore_ids] = neutral_vertices[restore_ids]
            smoothed_deformed[restore_ids] = deformed_vertices[restore_ids]
    report = MouthBoundarySmoothReport(
        boundary_edge_count=int(boundary_edges.shape[0]),
        mouth_boundary_edge_count=int(mouth_boundary_edges.shape[0]),
        mouth_boundary_component_count=int(len(components)),
        smoothed_component_count=int(len(selected_components)),
        smoothed_boundary_vertex_count=int(len(boundary_ids)),
        blended_vertex_count=int(blended_vertex_count),
        blend_rings=int(blend_rings),
        smoothing_cycles=int(smoothing_cycles),
        maximum_displacement=float(maximum_displacement),
        mean_boundary_displacement=float(displacement_norms.mean().item()) if displacement_norms.numel() else 0.0,
        max_boundary_displacement=float(displacement_norms.max().item()) if displacement_norms.numel() else 0.0,
        quality_rejected_face_count=int(len(quality_rejected_faces)),
        quality_restored_vertex_count=int(len(quality_restored_vertices)),
        preserve_boundary_junctions=bool(preserve_boundary_junctions),
        preserved_boundary_junction_count=int(preserved_boundary_junction_count),
        arc_length_weighted=bool(arc_length_weighted),
        fragmented_selection_enabled=minimum_component_span is not None,
        minimum_component_span=(
            float(minimum_component_span)
            if minimum_component_span is not None
            else None
        ),
        minimum_component_vertices=int(minimum_component_vertices),
    )
    return smoothed_neutral.contiguous(), smoothed_deformed.contiguous(), report


def clip_mesh_to_smooth_mouth_contour(
    neutral_vertices: torch.Tensor,
    deformed_vertices: torch.Tensor,
    source_faces: torch.Tensor,
    kept_faces: torch.Tensor,
    mapping: Mapping[int, int],
    *,
    horizontal_axis: int = 0,
    depth_axis: int = 1,
    vertical_axis: int = 2,
    contour_scale: float = 1.0,
    depth_padding: float = 0.08,
    edge_ratio_threshold: float = 3.0,
    edge_growth_threshold: float = 0.02,
    curve_samples: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, MouthContourClipReport]:
    """Replace a whole-face mouth cut with a spline-clipped triangle boundary."""

    neutral_vertices = neutral_vertices.detach().cpu().float().contiguous()
    deformed_vertices = deformed_vertices.detach().cpu().float().contiguous()
    source_faces = source_faces.detach().cpu().long().contiguous()
    kept_faces = kept_faces.detach().cpu().long().contiguous()
    if neutral_vertices.shape != deformed_vertices.shape:
        raise ValueError("Neutral and deformed vertices must have matching shapes.")
    if source_faces.ndim != 2 or source_faces.shape[1] != 3:
        raise ValueError("source_faces must contain triangles [F, 3].")
    if kept_faces.ndim != 2 or kept_faces.shape[1] != 3:
        raise ValueError("kept_faces must contain triangles [F, 3].")
    if contour_scale <= 0.0:
        raise ValueError("Mouth contour scale must be positive.")
    if depth_padding < 0.0:
        raise ValueError("Mouth contour depth padding cannot be negative.")
    if curve_samples < 16:
        raise ValueError("Mouth contour requires at least 16 curve samples.")

    vertex_count = int(deformed_vertices.shape[0])
    upper_ids = _ordered_mapped_vertex_ids(
        mapping,
        INNER_UPPER_LIP_MEDIAPIPE_IDS,
        vertex_count,
    )
    lower_ids = _ordered_mapped_vertex_ids(
        mapping,
        INNER_LOWER_LIP_MEDIAPIPE_IDS,
        vertex_count,
    )
    if len(upper_ids) < 3 or len(lower_ids) < 3:
        raise ValueError("Smooth contour clipping requires both inner-lip curves.")
    upper_points = deformed_vertices[torch.tensor(upper_ids, dtype=torch.long)]
    lower_points = deformed_vertices[torch.tensor(lower_ids, dtype=torch.long)]
    contour = _smooth_mouth_contour_field(
        deformed_vertices[:, horizontal_axis],
        deformed_vertices[:, vertical_axis],
        upper_points[:, (horizontal_axis, vertical_axis)],
        lower_points[:, (horizontal_axis, vertical_axis)],
        contour_scale=float(contour_scale),
        curve_samples=int(curve_samples),
    )

    source_neutral_triangles = neutral_vertices[source_faces]
    source_deformed_triangles = deformed_vertices[source_faces]
    face_centers = source_deformed_triangles.mean(dim=1)
    lip_depths = torch.cat((upper_points[:, depth_axis], lower_points[:, depth_axis]))
    front_depth = (
        (face_centers[:, depth_axis] >= lip_depths.amin() - float(depth_padding))
        & (face_centers[:, depth_axis] <= lip_depths.amax() + float(depth_padding))
    )
    neutral_edges = torch.stack(
        tuple(
            (
                source_neutral_triangles[:, (corner + 1) % 3]
                - source_neutral_triangles[:, corner]
            ).norm(dim=1)
            for corner in range(3)
        ),
        dim=1,
    )
    deformed_edges = torch.stack(
        tuple(
            (
                source_deformed_triangles[:, (corner + 1) % 3]
                - source_deformed_triangles[:, corner]
            ).norm(dim=1)
            for corner in range(3)
        ),
        dim=1,
    )
    severe_stretch = (
        (deformed_edges / neutral_edges.clamp_min(1.0e-8)).amax(dim=1)
        > float(edge_ratio_threshold)
    ) & (
        (deformed_edges - neutral_edges).amax(dim=1)
        > float(edge_growth_threshold)
    )

    face_key_base = int(neutral_vertices.shape[0]) + 1
    source_keys = (
        (source_faces[:, 0] * face_key_base + source_faces[:, 1])
        * face_key_base
        + source_faces[:, 2]
    )
    kept_keys = (
        (kept_faces[:, 0] * face_key_base + kept_faces[:, 1])
        * face_key_base
        + kept_faces[:, 2]
    )
    kept_mask = torch.from_numpy(
        np.isin(
            source_keys.numpy(),
            kept_keys.numpy(),
            assume_unique=True,
        )
    )
    face_inside = contour[source_faces] > 1.0e-8
    any_inside = face_inside.any(dim=1)
    all_inside = face_inside.all(dim=1)
    crossing = front_depth & any_inside & ~all_inside
    clip_mask = crossing & (kept_mask | ~severe_stretch)
    unchanged_mask = kept_mask & (~front_depth | ~any_inside)

    output_faces: list[list[int]] = source_faces[unchanged_mask].tolist()
    neutral_output = [point.clone() for point in neutral_vertices]
    deformed_output = [point.clone() for point in deformed_vertices]
    edge_intersections: dict[tuple[int, int], int] = {}
    clipped_face_count = int(clip_mask.sum().item())
    restored_boundary_face_count = int((clip_mask & ~kept_mask).sum().item())
    removed_inside_face_count = int(
        (kept_mask & front_depth & all_inside).sum().item()
    )
    retained_outside_face_count = int(
        (kept_mask & front_depth & ~any_inside).sum().item()
    )

    def intersection_vertex(start: int, end: int) -> int:
        key = (min(start, end), max(start, end))
        cached = edge_intersections.get(key)
        if cached is not None:
            return cached
        start_value = float(contour[start].item())
        end_value = float(contour[end].item())
        denominator = start_value - end_value
        if abs(denominator) <= 1.0e-12:
            fraction = 0.5
        else:
            fraction = max(0.0, min(1.0, start_value / denominator))
        neutral_point = neutral_vertices[start] + fraction * (
            neutral_vertices[end] - neutral_vertices[start]
        )
        deformed_point = deformed_vertices[start] + fraction * (
            deformed_vertices[end] - deformed_vertices[start]
        )
        vertex_id = len(neutral_output)
        neutral_output.append(neutral_point)
        deformed_output.append(deformed_point)
        edge_intersections[key] = vertex_id
        return vertex_id

    for face_index in torch.nonzero(clip_mask, as_tuple=False).flatten().tolist():
        face_tensor = source_faces[face_index]
        face = [int(value) for value in face_tensor.tolist()]
        values = [float(contour[vertex_id].item()) for vertex_id in face]
        inside = [value > 1.0e-8 for value in values]

        clipped_polygon: list[int] = []
        for corner in range(3):
            start = face[corner]
            end = face[(corner + 1) % 3]
            start_inside = inside[corner]
            end_inside = inside[(corner + 1) % 3]
            if not start_inside:
                clipped_polygon.append(start)
            if start_inside != end_inside:
                clipped_polygon.append(intersection_vertex(start, end))
        clipped_polygon = list(dict.fromkeys(clipped_polygon))
        if len(clipped_polygon) < 3:
            continue
        for corner in range(1, len(clipped_polygon) - 1):
            triangle = [
                clipped_polygon[0],
                clipped_polygon[corner],
                clipped_polygon[corner + 1],
            ]
            if len(set(triangle)) == 3:
                output_faces.append(triangle)
    final_neutral = torch.stack(neutral_output, dim=0).contiguous()
    final_deformed = torch.stack(deformed_output, dim=0).contiguous()
    final_faces = torch.tensor(output_faces, dtype=torch.long).contiguous()
    report = MouthContourClipReport(
        source_face_count=int(source_faces.shape[0]),
        input_kept_face_count=int(kept_faces.shape[0]),
        final_face_count=int(final_faces.shape[0]),
        original_vertex_count=vertex_count,
        final_vertex_count=int(final_neutral.shape[0]),
        clipped_face_count=int(clipped_face_count),
        restored_boundary_face_count=int(restored_boundary_face_count),
        removed_inside_face_count=int(removed_inside_face_count),
        retained_outside_face_count=int(retained_outside_face_count),
        added_vertex_count=int(final_neutral.shape[0] - vertex_count),
        smooth_curve_sample_count=int(curve_samples),
        contour_scale=float(contour_scale),
        depth_padding=float(depth_padding),
    )
    return final_neutral, final_deformed, final_faces, report




def _smooth_mouth_contour_field(
    horizontal: torch.Tensor,
    vertical: torch.Tensor,
    upper_points: torch.Tensor,
    lower_points: torch.Tensor,
    *,
    contour_scale: float,
    curve_samples: int,
) -> torch.Tensor:
    from scipy.interpolate import PchipInterpolator

    horizontal_np = horizontal.detach().cpu().double().numpy()
    vertical_np = vertical.detach().cpu().double().numpy()
    upper_np = upper_points.detach().cpu().double().numpy()
    lower_np = lower_points.detach().cpu().double().numpy()
    contour = _smooth_mouth_contour_samples_numpy(
        upper_np,
        lower_np,
        contour_scale=float(contour_scale),
        curve_samples=int(curve_samples),
    )
    half = contour.shape[0] // 2
    sample_x = contour[:half, 0]
    top_curve = contour[:half, 1]
    bottom_curve = contour[half:, 1][::-1]
    horizontal_min = float(sample_x.min())
    horizontal_max = float(sample_x.max())
    top = np.interp(horizontal_np, sample_x, top_curve)
    bottom = np.interp(horizontal_np, sample_x, bottom_curve)
    field = np.minimum.reduce(
        (
            horizontal_np - horizontal_min,
            horizontal_max - horizontal_np,
            vertical_np - bottom,
            top - vertical_np,
        )
    )
    return torch.from_numpy(field.astype(np.float32, copy=False))


def _smooth_mouth_contour_samples_numpy(
    upper_points: np.ndarray,
    lower_points: np.ndarray,
    *,
    contour_scale: float,
    curve_samples: int,
) -> np.ndarray:
    from scipy.interpolate import PchipInterpolator

    all_points = np.concatenate((upper_points, lower_points), axis=0)
    center = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    upper_points = center + (upper_points - center) * float(contour_scale)
    lower_points = center + (lower_points - center) * float(contour_scale)
    upper_x, upper_z = _merge_curve_samples_numpy(upper_points)
    lower_x, lower_z = _merge_curve_samples_numpy(lower_points)
    horizontal_min = max(float(upper_x.min()), float(lower_x.min()))
    horizontal_max = min(float(upper_x.max()), float(lower_x.max()))
    sample_x = np.linspace(horizontal_min, horizontal_max, int(curve_samples))
    upper_curve = PchipInterpolator(upper_x, upper_z, extrapolate=False)(sample_x)
    lower_curve = PchipInterpolator(lower_x, lower_z, extrapolate=False)(sample_x)
    top_curve = np.maximum(upper_curve, lower_curve)
    bottom_curve = np.minimum(upper_curve, lower_curve)
    upper_contour = np.stack((sample_x, top_curve), axis=1)
    lower_contour = np.stack((sample_x[::-1], bottom_curve[::-1]), axis=1)
    return np.concatenate((upper_contour, lower_contour), axis=0)


def _merge_curve_samples_numpy(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(points[:, 0], kind="stable")
    points = points[order]
    horizontal, inverse = np.unique(points[:, 0], return_inverse=True)
    vertical = np.zeros_like(horizontal)
    counts = np.zeros_like(horizontal)
    np.add.at(vertical, inverse, points[:, 1])
    np.add.at(counts, inverse, 1.0)
    vertical /= np.maximum(counts, 1.0)
    if horizontal.shape[0] < 3:
        raise ValueError("A smooth lip curve requires at least three distinct x positions.")
    return horizontal, vertical


def weld_coincident_vertices(
    mesh: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], CoincidentVertexWeldReport, torch.Tensor]:
    """Merge exactly coincident vertices while preserving first-seen ordering."""

    vertices = mesh["vertices"].detach().cpu().float().contiguous()
    faces = mesh["faces"].detach().cpu().long().contiguous()
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError("weld_coincident_vertices expects triangular faces [F, 3].")

    vertex_array = vertices.numpy()
    _unique_sorted, first_indices, inverse_sorted = np.unique(
        vertex_array,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    stable_order = np.argsort(first_indices)
    sorted_to_stable = np.empty_like(stable_order)
    sorted_to_stable[stable_order] = np.arange(stable_order.shape[0])
    stable_first_indices = first_indices[stable_order]
    original_to_welded = torch.from_numpy(
        sorted_to_stable[inverse_sorted].astype(np.int64, copy=False)
    ).long()
    welded_vertices = vertices.index_select(
        0,
        torch.from_numpy(stable_first_indices.astype(np.int64, copy=False)).long(),
    ).contiguous()

    welded_faces = original_to_welded.index_select(0, faces.reshape(-1)).reshape_as(faces)
    nondegenerate = (
        (welded_faces[:, 0] != welded_faces[:, 1])
        & (welded_faces[:, 1] != welded_faces[:, 2])
        & (welded_faces[:, 2] != welded_faces[:, 0])
    )
    welded_faces = welded_faces[nondegenerate].contiguous()

    original_normals = mesh.get("normals")
    if not isinstance(original_normals, torch.Tensor) or original_normals.shape != vertices.shape:
        original_normals = compute_vertex_normals(vertices, faces)
    else:
        original_normals = original_normals.detach().cpu().float().contiguous()
    welded_normals = torch.zeros_like(welded_vertices)
    welded_normals.index_add_(0, original_to_welded, original_normals)
    welded_normals = F.normalize(welded_normals, dim=-1, eps=1.0e-6).contiguous()

    welded_mesh = dict(mesh)
    welded_mesh["vertices"] = welded_vertices
    welded_mesh["faces"] = welded_faces
    welded_mesh["normals"] = welded_normals
    report = CoincidentVertexWeldReport(
        original_vertex_count=int(vertices.shape[0]),
        final_vertex_count=int(welded_vertices.shape[0]),
        merged_vertex_count=int(vertices.shape[0] - welded_vertices.shape[0]),
        original_face_count=int(faces.shape[0]),
        final_face_count=int(welded_faces.shape[0]),
        removed_degenerate_face_count=int((~nondegenerate).sum().item()),
    )
    return welded_mesh, report, original_to_welded.contiguous()


def remap_vertex_mapping(
    mapping: Mapping[int, int],
    original_to_updated: torch.Tensor,
) -> dict[int, int]:
    original_to_updated = original_to_updated.detach().cpu().long().flatten()
    remapped: dict[int, int] = {}
    for raw_landmark_id, raw_vertex_id in mapping.items():
        vertex_id = int(raw_vertex_id)
        if 0 <= vertex_id < original_to_updated.numel():
            remapped[int(raw_landmark_id)] = int(original_to_updated[vertex_id].item())
    return remapped


def snap_mediapipe_lip_landmarks_to_face_surface(
    mesh: Mapping[str, torch.Tensor],
    mapping: Mapping[int, int],
    *,
    horizontal_axis: int = 0,
    front_axis: int = 1,
    vertical_axis: int = 2,
    front_direction: int = -1,
    maximum_projected_distance_ratio: float = 0.20,
    local_candidate_radius_ratio: float = 0.04,
    minimum_upper_lower_gap_ratio: float = 0.012,
    front_depth_weight: float = 0.18,
) -> tuple[dict[int, int], LipLandmarkSurfaceSnapReport]:
    """Snap lip landmarks to one coherent, front-facing facial surface.

    Multi-part character FBXs can produce a plausible 2D MediaPipe detection
    whose selected mesh vertices lie on teeth, tongue, hair, or a rear surface.
    The 2D lip locations are still useful.  This routine chooses the facial
    component by landmark votes, then assigns every lip landmark to a unique
    vertex near its projected location on the front of that component.

    Upper/lower curve pairs receive a small minimum vertical separation before
    assignment.  Consequently, touching lips do not collapse landmarks such as
    MediaPipe 13 and 14 onto one vertex when the mesh already has separate lip
    rows.
    """

    from scipy import sparse
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse.csgraph import connected_components

    vertices = mesh["vertices"].detach().cpu().float().contiguous()
    faces = mesh["faces"].detach().cpu().long().contiguous()
    axes = {int(horizontal_axis), int(front_axis), int(vertical_axis)}
    if axes != {0, 1, 2}:
        raise ValueError("Lip landmark snap axes must be a permutation of [0, 1, 2].")
    if int(front_direction) not in {-1, 1}:
        raise ValueError("front_direction must be either -1 or 1.")
    if maximum_projected_distance_ratio <= 0.0:
        raise ValueError("maximum_projected_distance_ratio must be positive.")
    if local_candidate_radius_ratio < 0.0:
        raise ValueError("local_candidate_radius_ratio must be non-negative.")
    if minimum_upper_lower_gap_ratio < 0.0:
        raise ValueError("minimum_upper_lower_gap_ratio must be non-negative.")
    if front_depth_weight < 0.0:
        raise ValueError("front_depth_weight must be non-negative.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Lip landmark surface snap expects triangular faces.")
    if faces.numel() and (
        int(faces.min().item()) < 0
        or int(faces.max().item()) >= vertices.shape[0]
    ):
        raise ValueError("Lip landmark surface snap faces contain invalid vertex IDs.")

    repaired = {int(key): int(value) for key, value in mapping.items()}
    original_mapping = dict(repaired)
    vertex_count = int(vertices.shape[0])
    lip_curves = (
        INNER_UPPER_LIP_MEDIAPIPE_IDS,
        INNER_LOWER_LIP_MEDIAPIPE_IDS,
        OUTER_UPPER_LIP_MEDIAPIPE_IDS,
        OUTER_LOWER_LIP_MEDIAPIPE_IDS,
    )
    lip_landmark_ids = tuple(
        dict.fromkeys(value for curve in lip_curves for value in curve)
    )
    present_ids = [
        landmark_id
        for landmark_id in lip_landmark_ids
        if 0 <= repaired.get(landmark_id, -1) < vertex_count
    ]

    def empty_report(*, component_count: int = 0) -> LipLandmarkSurfaceSnapReport:
        return LipLandmarkSurfaceSnapReport(
            input_landmark_count=len(repaired),
            lip_landmark_count=len(lip_landmark_ids),
            connected_component_count=int(component_count),
            facial_component_id=-1,
            facial_component_vertex_count=0,
            facial_component_landmark_votes=0,
            off_component_lip_landmark_count=0,
            duplicate_lip_landmark_count_before=0,
            duplicate_lip_landmark_count_after=0,
            snapped_lip_landmark_count=0,
            skipped_lip_landmark_count=len(lip_landmark_ids),
            maximum_projected_distance_ratio=0.0,
            mean_projected_distance_ratio=0.0,
            minimum_upper_lower_gap=0.0,
            front_axis=int(front_axis),
            front_direction=int(front_direction),
        )

    if not present_ids or faces.numel() == 0:
        return repaired, empty_report(component_count=vertex_count)

    face_array = faces.numpy()
    graph_rows = np.concatenate(
        (face_array[:, 0], face_array[:, 1], face_array[:, 2])
    )
    graph_columns = np.concatenate(
        (face_array[:, 1], face_array[:, 2], face_array[:, 0])
    )
    graph = sparse.coo_matrix(
        (
            np.ones(graph_rows.shape[0], dtype=np.uint8),
            (graph_rows, graph_columns),
        ),
        shape=(vertex_count, vertex_count),
    ).tocsr()
    component_count, component_labels = connected_components(
        graph,
        directed=False,
    )
    component_sizes = np.bincount(component_labels, minlength=component_count)

    # Face Mesh IDs below the iris extension (468+) provide a much stronger
    # facial-surface vote than mouth landmarks alone when several lips happen
    # to land on the same tongue or tooth component.
    component_votes = Counter(
        int(component_labels[vertex_id])
        for landmark_id, vertex_id in repaired.items()
        if 0 <= int(landmark_id) < 468 and 0 <= int(vertex_id) < vertex_count
    )
    if not component_votes:
        return repaired, empty_report(component_count=component_count)
    facial_component = max(
        component_votes,
        key=lambda component: (
            int(component_votes[component]),
            int(component_sizes[component]),
            -int(component),
        ),
    )
    facial_vertex_ids = np.flatnonzero(
        component_labels == facial_component
    ).astype(np.int64, copy=False)
    if facial_vertex_ids.size == 0:
        return repaired, empty_report(component_count=component_count)

    source_vertex_ids = torch.tensor(
        [repaired[landmark_id] for landmark_id in present_ids],
        dtype=torch.long,
    )
    targets = vertices.index_select(0, source_vertex_ids).clone()
    target_index = {
        landmark_id: index for index, landmark_id in enumerate(present_ids)
    }
    projection_axes = (int(horizontal_axis), int(vertical_axis))

    if all(value in repaired for value in (61, 291)):
        corner_ids = torch.tensor([repaired[61], repaired[291]], dtype=torch.long)
        if bool(((corner_ids >= 0) & (corner_ids < vertex_count)).all()):
            corner_points = vertices.index_select(0, corner_ids)[:, projection_axes]
            mouth_width = float(
                torch.linalg.vector_norm(corner_points[1] - corner_points[0]).item()
            )
        else:
            mouth_width = 0.0
    else:
        mouth_width = 0.0
    projected_targets = targets[:, projection_axes]
    projected_span = projected_targets.amax(dim=0) - projected_targets.amin(dim=0)
    mouth_width = max(mouth_width, float(projected_span.max().item()), 1.0e-8)
    minimum_gap = float(minimum_upper_lower_gap_ratio) * mouth_width

    # MediaPipe defines corresponding upper and lower curves in the same
    # left-to-right order.  Preserve their observed center while separating
    # coincident or inverted samples just enough to select different lip rows.
    for upper_curve, lower_curve in (
        (INNER_UPPER_LIP_MEDIAPIPE_IDS, INNER_LOWER_LIP_MEDIAPIPE_IDS),
        (OUTER_UPPER_LIP_MEDIAPIPE_IDS, OUTER_LOWER_LIP_MEDIAPIPE_IDS),
    ):
        for upper_id, lower_id in zip(upper_curve, lower_curve):
            if (
                upper_id == lower_id
                or upper_id not in target_index
                or lower_id not in target_index
            ):
                continue
            upper_index = target_index[upper_id]
            lower_index = target_index[lower_id]
            upper_height = targets[upper_index, int(vertical_axis)]
            lower_height = targets[lower_index, int(vertical_axis)]
            if float(upper_height - lower_height) >= minimum_gap:
                continue
            center_height = 0.5 * (upper_height + lower_height)
            targets[upper_index, int(vertical_axis)] = center_height + 0.5 * minimum_gap
            targets[lower_index, int(vertical_axis)] = center_height - 0.5 * minimum_gap

    facial_ids_tensor = torch.from_numpy(facial_vertex_ids).long()
    facial_vertices = vertices.index_select(0, facial_ids_tensor)
    projected_targets = targets[:, projection_axes]
    projected_candidates = facial_vertices[:, projection_axes]
    projected_distances = torch.cdist(projected_targets, projected_candidates)
    nearest_distances = projected_distances.amin(dim=1)
    maximum_distance = float(maximum_projected_distance_ratio) * mouth_width
    candidate_slack = float(local_candidate_radius_ratio) * mouth_width
    surface_depth_tolerance = 0.12 * mouth_width
    directed_depth = (
        facial_vertices[:, int(front_axis)] * float(front_direction)
    )
    facial_face_mask = component_labels[face_array[:, 0]] == int(facial_component)
    facial_faces = faces[torch.from_numpy(facial_face_mask)]
    facial_triangles = vertices.index_select(
        0,
        facial_faces.reshape(-1),
    ).reshape(-1, 3, 3)
    projected_triangles = facial_triangles[:, :, projection_axes]
    triangle_first = projected_triangles[:, 0]
    triangle_first_edge = projected_triangles[:, 1] - triangle_first
    triangle_second_edge = projected_triangles[:, 2] - triangle_first
    triangle_denominator = (
        triangle_first_edge[:, 0] * triangle_second_edge[:, 1]
        - triangle_second_edge[:, 0] * triangle_first_edge[:, 1]
    )
    valid_projected_triangle = triangle_denominator.abs() > 1.0e-12
    safe_triangle_denominator = torch.where(
        valid_projected_triangle,
        triangle_denominator,
        torch.ones_like(triangle_denominator),
    )
    facial_local_by_global = np.full(vertex_count, -1, dtype=np.int64)
    facial_local_by_global[facial_vertex_ids] = np.arange(
        facial_vertex_ids.size,
        dtype=np.int64,
    )
    unavailable_cost = 1.0e6 * max(mouth_width, 1.0)
    costs = torch.full_like(projected_distances, unavailable_cost)
    assignable_rows: list[int] = []
    for row_index in range(len(present_ids)):
        nearest_distance = float(nearest_distances[row_index].item())
        if nearest_distance > maximum_distance:
            continue
        point_offset = projected_targets[row_index] - triangle_first
        first_weight = (
            point_offset[:, 0] * triangle_second_edge[:, 1]
            - triangle_second_edge[:, 0] * point_offset[:, 1]
        ) / safe_triangle_denominator
        second_weight = (
            triangle_first_edge[:, 0] * point_offset[:, 1]
            - point_offset[:, 0] * triangle_first_edge[:, 1]
        ) / safe_triangle_denominator
        zeroth_weight = 1.0 - first_weight - second_weight
        ray_hits = (
            valid_projected_triangle
            & (zeroth_weight >= -1.0e-5)
            & (first_weight >= -1.0e-5)
            & (second_weight >= -1.0e-5)
        )
        hit_face_ids = torch.where(ray_hits)[0]
        hit_local_ids: torch.Tensor | None = None
        hit_directed_depth: torch.Tensor | None = None
        if hit_face_ids.numel() > 0:
            hit_depths = (
                zeroth_weight[hit_face_ids]
                * facial_triangles[hit_face_ids, 0, int(front_axis)]
                + first_weight[hit_face_ids]
                * facial_triangles[hit_face_ids, 1, int(front_axis)]
                + second_weight[hit_face_ids]
                * facial_triangles[hit_face_ids, 2, int(front_axis)]
            ) * float(front_direction)
            front_hit_id = hit_face_ids[int(torch.argmax(hit_depths).item())]
            hit_global_ids = facial_faces[front_hit_id]
            hit_local_ids = torch.from_numpy(
                facial_local_by_global[hit_global_ids.numpy()]
            ).long()
            hit_directed_depth = hit_depths.max()

        if hit_local_ids is not None and hit_directed_depth is not None:
            hit_distance = projected_distances[
                row_index,
                hit_local_ids,
            ].amin()
            allowed_distance = min(
                float(hit_distance.item()) + candidate_slack,
                maximum_distance,
            )
            depth_difference = (directed_depth - hit_directed_depth).abs()
            allowed = (
                (projected_distances[row_index] <= allowed_distance)
                & (depth_difference <= surface_depth_tolerance)
            )
            allowed[hit_local_ids] = True
            depth_penalty = depth_difference
            broad_allowed = (
                (projected_distances[row_index] <= maximum_distance)
                & (depth_difference <= 0.30 * mouth_width)
            )
        else:
            allowed_distance = min(
                nearest_distance + candidate_slack,
                maximum_distance,
            )
            broad_allowed = projected_distances[row_index] <= maximum_distance
            if not bool(broad_allowed.any()):
                continue
            frontmost_depth = directed_depth[broad_allowed].amax()
            depth_penalty = (frontmost_depth - directed_depth).clamp_min(0.0)
            allowed = projected_distances[row_index] <= allowed_distance

        row_cost = projected_distances[row_index] + float(
            front_depth_weight
        ) * depth_penalty
        # Dense or low-resolution lip meshes can leave several landmark rows
        # competing for the same tiny set of ideal vertices.  Keep a wider,
        # depth-constrained fallback set at a deliberately higher cost so the
        # global assignment remains unique without preferring a rear surface.
        costs[row_index, broad_allowed] = (
            row_cost[broad_allowed] + maximum_distance
        )
        costs[row_index, allowed] = row_cost[allowed]
        assignable_rows.append(row_index)

    assigned_rows: set[int] = set()
    if assignable_rows:
        row_ids, column_ids = linear_sum_assignment(
            costs[assignable_rows].numpy()
        )
        for local_row, column_index in zip(row_ids.tolist(), column_ids.tolist()):
            row_index = assignable_rows[int(local_row)]
            if float(costs[row_index, column_index].item()) >= unavailable_cost:
                continue
            landmark_id = present_ids[row_index]
            repaired[landmark_id] = int(facial_vertex_ids[column_index])
            assigned_rows.add(row_index)

    # A lip sample can fall just inside the aperture and ray-hit a rear facial
    # wall.  Repair isolated depth jumps from its two curve neighbors while
    # retaining the unique-vertex assignment.
    for _pass in range(2):
        used_vertex_ids = {int(repaired[value]) for value in present_ids}
        changed_outlier = False
        for curve in lip_curves:
            for curve_index in range(1, len(curve) - 1):
                landmark_id = curve[curve_index]
                previous_id = curve[curve_index - 1]
                next_id = curve[curve_index + 1]
                if not all(
                    value in target_index
                    and target_index[value] in assigned_rows
                    for value in (previous_id, landmark_id, next_id)
                ):
                    continue
                current_vertex_id = int(repaired[landmark_id])
                previous_depth = (
                    vertices[int(repaired[previous_id]), int(front_axis)]
                    * float(front_direction)
                )
                next_depth = (
                    vertices[int(repaired[next_id]), int(front_axis)]
                    * float(front_direction)
                )
                if float((previous_depth - next_depth).abs()) > 0.30 * mouth_width:
                    continue
                expected_depth = 0.5 * (previous_depth + next_depth)
                current_depth = (
                    vertices[current_vertex_id, int(front_axis)]
                    * float(front_direction)
                )
                if float((current_depth - expected_depth).abs()) <= 0.30 * mouth_width:
                    continue

                row_index = target_index[landmark_id]
                depth_difference = (directed_depth - expected_depth).abs()
                allowed = (
                    (projected_distances[row_index] <= maximum_distance)
                    & (depth_difference <= 0.15 * mouth_width)
                )
                for used_vertex_id in used_vertex_ids - {current_vertex_id}:
                    local_id = int(facial_local_by_global[used_vertex_id])
                    if local_id >= 0:
                        allowed[local_id] = False
                candidate_local_ids = torch.where(allowed)[0]
                if candidate_local_ids.numel() == 0:
                    continue
                candidate_cost = (
                    projected_distances[row_index, candidate_local_ids]
                    + float(front_depth_weight)
                    * depth_difference[candidate_local_ids]
                )
                replacement_local_id = int(
                    candidate_local_ids[int(torch.argmin(candidate_cost).item())].item()
                )
                replacement_vertex_id = int(facial_vertex_ids[replacement_local_id])
                if replacement_vertex_id == current_vertex_id:
                    continue
                repaired[landmark_id] = replacement_vertex_id
                used_vertex_ids.discard(current_vertex_id)
                used_vertex_ids.add(replacement_vertex_id)
                changed_outlier = True
        if not changed_outlier:
            break

    original_vertex_ids = [int(original_mapping[value]) for value in present_ids]
    final_vertex_ids = [int(repaired[value]) for value in present_ids]
    projected_ratios = [
        float(
            torch.linalg.vector_norm(
                vertices[repaired[landmark_id], projection_axes]
                - targets[target_index[landmark_id], projection_axes]
            ).item()
        )
        / mouth_width
        for landmark_id in present_ids
        if target_index[landmark_id] in assigned_rows
    ]
    original_duplicates = len(original_vertex_ids) - len(set(original_vertex_ids))
    final_duplicates = len(final_vertex_ids) - len(set(final_vertex_ids))
    off_component_count = sum(
        int(component_labels[int(original_mapping[value])]) != int(facial_component)
        for value in present_ids
    )
    report = LipLandmarkSurfaceSnapReport(
        input_landmark_count=len(repaired),
        lip_landmark_count=len(lip_landmark_ids),
        connected_component_count=int(component_count),
        facial_component_id=int(facial_component),
        facial_component_vertex_count=int(facial_vertex_ids.size),
        facial_component_landmark_votes=int(component_votes[facial_component]),
        off_component_lip_landmark_count=int(off_component_count),
        duplicate_lip_landmark_count_before=int(original_duplicates),
        duplicate_lip_landmark_count_after=int(final_duplicates),
        snapped_lip_landmark_count=sum(
            original != final
            for original, final in zip(original_vertex_ids, final_vertex_ids)
        ),
        skipped_lip_landmark_count=(
            len(lip_landmark_ids) - len(present_ids)
            + len(present_ids) - len(assigned_rows)
        ),
        maximum_projected_distance_ratio=max(projected_ratios, default=0.0),
        mean_projected_distance_ratio=(
            float(np.mean(projected_ratios)) if projected_ratios else 0.0
        ),
        minimum_upper_lower_gap=float(minimum_gap),
        front_axis=int(front_axis),
        front_direction=int(front_direction),
    )
    return repaired, report


def repair_mediapipe_lip_landmarks_to_dominant_component(
    mesh: Mapping[str, torch.Tensor],
    mapping: Mapping[int, int],
    *,
    maximum_snap_distance_ratio: float = 0.25,
    horizontal_axis: int = 0,
    depth_axis: int = 1,
    vertical_axis: int = 2,
) -> tuple[dict[int, int], LipLandmarkTopologyRepairReport]:
    """Move invalid or detached lip landmarks onto one coherent face surface."""

    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    vertices = mesh["vertices"].detach().cpu().float().contiguous()
    faces = mesh["faces"].detach().cpu().long().contiguous()
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Lip landmark topology repair expects triangular faces.")
    if maximum_snap_distance_ratio <= 0.0:
        raise ValueError("Maximum lip-landmark snap distance ratio must be positive.")
    if {
        int(horizontal_axis),
        int(depth_axis),
        int(vertical_axis),
    } != {0, 1, 2}:
        raise ValueError("Lip landmark repair axes must be a permutation of [0, 2].")

    input_landmark_count = len(mapping)
    repaired = {int(key): int(value) for key, value in mapping.items()}
    vertex_count = int(vertices.shape[0])
    lip_curves = (
        INNER_UPPER_LIP_MEDIAPIPE_IDS,
        INNER_LOWER_LIP_MEDIAPIPE_IDS,
        OUTER_UPPER_LIP_MEDIAPIPE_IDS,
        OUTER_LOWER_LIP_MEDIAPIPE_IDS,
    )
    lip_landmark_ids = tuple(
        dict.fromkeys(value for curve in lip_curves for value in curve)
    )
    valid_ids = [
        landmark_id
        for landmark_id in lip_landmark_ids
        if 0 <= repaired.get(landmark_id, -1) < vertex_count
    ]
    invalid_ids = set(lip_landmark_ids) - set(valid_ids)
    if not valid_ids or faces.numel() == 0:
        report = LipLandmarkTopologyRepairReport(
            input_landmark_count=input_landmark_count,
            lip_landmark_count=len(lip_landmark_ids),
            valid_lip_landmark_count=len(valid_ids),
            invalid_lip_landmark_count=len(invalid_ids),
            connected_component_count=vertex_count,
            dominant_component_vertex_count=0,
            off_component_lip_landmark_count=0,
            spatial_outlier_lip_landmark_count=0,
            snapped_lip_landmark_count=0,
            interpolated_lip_landmark_count=0,
            skipped_lip_landmark_count=len(invalid_ids),
            maximum_snap_distance=0.0,
            mean_snap_distance=0.0,
        )
        return repaired, report

    face_array = faces.numpy()
    graph_rows = np.concatenate(
        (
            face_array[:, 0], face_array[:, 1], face_array[:, 2],
            face_array[:, 1], face_array[:, 2], face_array[:, 0],
        )
    )
    graph_columns = np.concatenate(
        (
            face_array[:, 1], face_array[:, 2], face_array[:, 0],
            face_array[:, 0], face_array[:, 1], face_array[:, 2],
        )
    )
    graph = sparse.coo_matrix(
        (
            np.ones(graph_rows.shape[0], dtype=np.uint8),
            (graph_rows, graph_columns),
        ),
        shape=(vertex_count, vertex_count),
    ).tocsr()
    component_count, component_labels = connected_components(
        graph,
        directed=False,
    )
    component_sizes = np.bincount(component_labels, minlength=component_count)
    landmark_component_counts = np.bincount(
        np.asarray(
            [component_labels[repaired[landmark_id]] for landmark_id in valid_ids],
            dtype=np.int64,
        ),
        minlength=component_count,
    )
    candidate_components = np.flatnonzero(landmark_component_counts > 0)
    dominant_component = max(
        candidate_components.tolist(),
        key=lambda component: (
            int(landmark_component_counts[component]),
            int(component_sizes[component]),
        ),
    )
    dominant_vertex_ids = np.flatnonzero(
        component_labels == dominant_component
    ).astype(np.int64, copy=False)
    dominant_vertices = vertices.numpy()[dominant_vertex_ids]
    dominant_tree = cKDTree(dominant_vertices)

    valid_points = vertices[
        torch.tensor([repaired[value] for value in valid_ids], dtype=torch.long)
    ]
    projected_span = (
        valid_points[:, (int(horizontal_axis), int(vertical_axis))].amax(dim=0)
        - valid_points[:, (int(horizontal_axis), int(vertical_axis))].amin(dim=0)
    )
    mouth_scale = max(float(projected_span.max().item()), 1.0e-6)
    maximum_snap_distance = float(maximum_snap_distance_ratio) * mouth_scale
    snap_distances: list[float] = []
    snapped_ids: set[int] = set()
    interpolated_ids: set[int] = set()
    off_component_ids = {
        landmark_id
        for landmark_id in valid_ids
        if component_labels[repaired[landmark_id]] != dominant_component
    }

    def snap_landmark(landmark_id: int, target: np.ndarray) -> bool:
        distance, local_index = dominant_tree.query(target, k=1)
        if not np.isfinite(distance) or float(distance) > maximum_snap_distance:
            return False
        repaired[int(landmark_id)] = int(dominant_vertex_ids[int(local_index)])
        snapped_ids.add(int(landmark_id))
        snap_distances.append(float(distance))
        return True

    for landmark_id in sorted(off_component_ids):
        source_vertex_id = repaired[landmark_id]
        snap_landmark(landmark_id, vertices[source_vertex_id].numpy())

    resolved_ids = [
        landmark_id
        for landmark_id in lip_landmark_ids
        if (
            0 <= repaired.get(landmark_id, -1) < vertex_count
            and component_labels[repaired[landmark_id]] == dominant_component
        )
    ]
    resolved_points = vertices[
        torch.tensor([repaired[value] for value in resolved_ids], dtype=torch.long)
    ]
    resolved_depths = resolved_points[:, int(depth_axis)].double().numpy()
    median_depth = float(np.median(resolved_depths))
    median_depth_deviation = float(
        np.median(np.abs(resolved_depths - median_depth))
    )
    depth_outlier_threshold = max(
        0.12 * mouth_scale,
        8.0 * median_depth_deviation,
        1.0e-5,
    )
    spatial_outlier_ids = {
        landmark_id
        for landmark_id in resolved_ids
        if abs(
            float(vertices[repaired[landmark_id], int(depth_axis)].item())
            - median_depth
        )
        > depth_outlier_threshold
    }
    for curve in lip_curves:
        for index, landmark_id in enumerate(curve):
            if landmark_id not in spatial_outlier_ids:
                continue
            neighbor_ids = []
            for neighbor_index in (index - 1, index + 1):
                if not 0 <= neighbor_index < len(curve):
                    continue
                neighbor_id = curve[neighbor_index]
                if (
                    neighbor_id not in spatial_outlier_ids
                    and 0 <= repaired.get(neighbor_id, -1) < vertex_count
                    and component_labels[repaired[neighbor_id]]
                    == dominant_component
                ):
                    neighbor_ids.append(repaired[neighbor_id])
            if not neighbor_ids:
                continue
            target = vertices[
                torch.tensor(neighbor_ids, dtype=torch.long)
            ].mean(dim=0).numpy()
            snap_landmark(landmark_id, target)

    unresolved_ids = {
        landmark_id
        for landmark_id in lip_landmark_ids
        if not (
            0 <= repaired.get(landmark_id, -1) < vertex_count
            and component_labels[repaired[landmark_id]] == dominant_component
        )
    }
    for curve in lip_curves:
        known_indices = [
            index
            for index, landmark_id in enumerate(curve)
            if landmark_id not in unresolved_ids
        ]
        if not known_indices:
            continue
        known_points = vertices[
            torch.tensor(
                [repaired[curve[index]] for index in known_indices],
                dtype=torch.long,
            )
        ].numpy()
        for index, landmark_id in enumerate(curve):
            if landmark_id not in unresolved_ids:
                continue
            insertion = int(np.searchsorted(known_indices, index))
            if insertion <= 0:
                target = known_points[0]
            elif insertion >= len(known_indices):
                target = known_points[-1]
            else:
                first_index = known_indices[insertion - 1]
                second_index = known_indices[insertion]
                fraction = float(index - first_index) / float(
                    second_index - first_index
                )
                target = known_points[insertion - 1] + fraction * (
                    known_points[insertion] - known_points[insertion - 1]
                )
            if snap_landmark(landmark_id, target):
                interpolated_ids.add(int(landmark_id))
                unresolved_ids.discard(landmark_id)

    # Closed-mouth upper and lower curves are close enough that one can seed
    # the other when an FBX sidecar refers entirely to a different mesh object.
    for first_curve, second_curve in (
        (INNER_UPPER_LIP_MEDIAPIPE_IDS, INNER_LOWER_LIP_MEDIAPIPE_IDS),
        (OUTER_UPPER_LIP_MEDIAPIPE_IDS, OUTER_LOWER_LIP_MEDIAPIPE_IDS),
    ):
        for source_curve, target_curve in (
            (first_curve, second_curve),
            (second_curve, first_curve),
        ):
            for landmark_id, counterpart_id in zip(source_curve, target_curve):
                if landmark_id not in unresolved_ids:
                    continue
                counterpart_vertex_id = repaired.get(counterpart_id, -1)
                if not (
                    0 <= counterpart_vertex_id < vertex_count
                    and component_labels[counterpart_vertex_id]
                    == dominant_component
                ):
                    continue
                if snap_landmark(
                    landmark_id,
                    vertices[counterpart_vertex_id].numpy(),
                ):
                    interpolated_ids.add(int(landmark_id))
                    unresolved_ids.discard(landmark_id)

    report = LipLandmarkTopologyRepairReport(
        input_landmark_count=input_landmark_count,
        lip_landmark_count=len(lip_landmark_ids),
        valid_lip_landmark_count=len(valid_ids),
        invalid_lip_landmark_count=len(invalid_ids),
        connected_component_count=int(component_count),
        dominant_component_vertex_count=int(component_sizes[dominant_component]),
        off_component_lip_landmark_count=len(off_component_ids),
        spatial_outlier_lip_landmark_count=len(spatial_outlier_ids),
        snapped_lip_landmark_count=len(snapped_ids),
        interpolated_lip_landmark_count=len(interpolated_ids),
        skipped_lip_landmark_count=len(unresolved_ids),
        maximum_snap_distance=max(snap_distances, default=0.0),
        mean_snap_distance=(
            float(np.mean(snap_distances)) if snap_distances else 0.0
        ),
    )
    return repaired, report


def split_lips_along_mediapipe_seam(
    mesh: Mapping[str, torch.Tensor],
    mapping: Mapping[int, int],
    *,
    horizontal_axis: int = 0,
    depth_axis: int = 1,
    vertical_axis: int = 2,
    depth_tolerance: float = 0.02,
    vertical_tolerance: float = 0.012,
    horizontal_padding: float = 0.0,
    corner_inset_ratio: float = 0.0,
    preopen_distance: float = 0.0,
) -> tuple[dict[str, torch.Tensor], GeometricLipSeamReport]:
    """Cut a closed-mouth mesh along its MediaPipe inner-lip centerline.

    Faces are retained.  The function labels local faces above and below a
    piecewise-linear inner-lip curve, finds vertices shared by both sides, and
    duplicates those vertices for lower-side faces.  The resulting coincident
    boundaries look unchanged in neutral pose but are disconnected in the
    graph supplied to TopoRig inference.
    """

    vertices = mesh["vertices"].detach().cpu().float().contiguous()
    faces = mesh["faces"].detach().cpu().long().contiguous()
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(
            "split_lips_along_mediapipe_seam expects triangular faces [F, 3]."
        )
    axes = (int(horizontal_axis), int(depth_axis), int(vertical_axis))
    if sorted(axes) != [0, 1, 2]:
        raise ValueError(
            "horizontal_axis, depth_axis, and vertical_axis must be distinct "
            "coordinate indices in [0, 2]."
        )
    if depth_tolerance <= 0.0 or vertical_tolerance <= 0.0:
        raise ValueError("Lip seam depth and vertical tolerances must be positive.")
    if horizontal_padding < 0.0:
        raise ValueError("Lip seam horizontal padding cannot be negative.")
    if not 0.0 <= corner_inset_ratio < 0.5:
        raise ValueError("Lip seam corner inset ratio must be in [0, 0.5).")
    if preopen_distance < 0.0:
        raise ValueError("Lip seam pre-open distance cannot be negative.")

    curve_points = _inner_lip_center_curve(vertices, mapping)
    if curve_points.shape[0] < 2:
        raise ValueError(
            "At least two usable MediaPipe inner-lip center points are required."
        )
    curve_points = curve_points[
        torch.argsort(curve_points[:, horizontal_axis])
    ].contiguous()
    curve_points = _merge_curve_points_with_duplicate_horizontal_values(
        curve_points,
        horizontal_axis=horizontal_axis,
    )
    if curve_points.shape[0] < 2:
        raise ValueError(
            "MediaPipe inner-lip center points do not span the horizontal axis."
        )

    face_centers = vertices.index_select(0, faces.reshape(-1)).reshape(-1, 3, 3).mean(dim=1)
    center_horizontal = face_centers[:, horizontal_axis]
    curve_vertical = _piecewise_linear_interpolate(
        center_horizontal,
        curve_points[:, horizontal_axis],
        curve_points[:, vertical_axis],
    )
    curve_depth = _piecewise_linear_interpolate(
        center_horizontal,
        curve_points[:, horizontal_axis],
        curve_points[:, depth_axis],
    )
    vertical_offset = face_centers[:, vertical_axis] - curve_vertical
    depth_offset = face_centers[:, depth_axis] - curve_depth
    curve_horizontal_min = curve_points[:, horizontal_axis].amin()
    curve_horizontal_max = curve_points[:, horizontal_axis].amax()
    corner_inset = (
        curve_horizontal_max - curve_horizontal_min
    ) * float(corner_inset_ratio)
    horizontal_min = (
        curve_horizontal_min + corner_inset - float(horizontal_padding)
    )
    horizontal_max = (
        curve_horizontal_max - corner_inset + float(horizontal_padding)
    )
    candidate_faces = (
        (center_horizontal >= horizontal_min)
        & (center_horizontal <= horizontal_max)
        & (depth_offset.abs() <= float(depth_tolerance))
        & (vertical_offset.abs() <= float(vertical_tolerance))
    )
    upper_candidate_faces = candidate_faces & (vertical_offset >= 0.0)
    lower_candidate_faces = candidate_faces & (vertical_offset < 0.0)
    if not bool(upper_candidate_faces.any().item()):
        raise ValueError("No upper-side faces were found near the MediaPipe lip seam.")
    if not bool(lower_candidate_faces.any().item()):
        raise ValueError("No lower-side faces were found near the MediaPipe lip seam.")

    vertex_count = int(vertices.shape[0])
    upper_vertices = torch.zeros(vertex_count, dtype=torch.bool)
    lower_vertices = torch.zeros(vertex_count, dtype=torch.bool)
    upper_vertices[faces[upper_candidate_faces].reshape(-1)] = True
    lower_vertices[faces[lower_candidate_faces].reshape(-1)] = True
    seam_vertex_ids = torch.nonzero(
        upper_vertices & lower_vertices,
        as_tuple=False,
    ).flatten().long()
    if seam_vertex_ids.numel() < 2:
        raise ValueError(
            "The MediaPipe lip curve did not produce a usable mesh seam. "
            "Increase the seam depth or vertical tolerance."
        )

    duplicate_ids = torch.arange(
        vertex_count,
        vertex_count + seam_vertex_ids.numel(),
        dtype=torch.long,
    )
    lower_duplicate_vertex_ids = {
        int(original): int(duplicate)
        for original, duplicate in zip(seam_vertex_ids.tolist(), duplicate_ids.tolist())
    }
    updated_vertices = torch.cat(
        (vertices, vertices.index_select(0, seam_vertex_ids)),
        dim=0,
    ).contiguous()
    if preopen_distance > 0.0:
        half_distance = float(preopen_distance) * 0.5
        updated_vertices[seam_vertex_ids, vertical_axis] += half_distance
        updated_vertices[duplicate_ids, vertical_axis] -= half_distance
    lower_remap = torch.arange(updated_vertices.shape[0], dtype=torch.long)
    lower_remap[seam_vertex_ids] = duplicate_ids

    seam_mask = torch.zeros(vertex_count, dtype=torch.bool)
    seam_mask[seam_vertex_ids] = True
    face_has_seam_vertex = seam_mask.index_select(0, faces.reshape(-1)).reshape_as(faces).any(dim=1)
    remapped_lower_faces = face_has_seam_vertex & (vertical_offset < 0.0)
    updated_faces = faces.clone()
    updated_faces[remapped_lower_faces] = lower_remap.index_select(
        0,
        updated_faces[remapped_lower_faces].reshape(-1),
    ).reshape(-1, 3)

    original_in_face = torch.zeros(updated_vertices.shape[0], dtype=torch.bool)
    duplicate_in_face = torch.zeros_like(original_in_face)
    original_in_face[seam_vertex_ids] = True
    duplicate_in_face[duplicate_ids] = True
    flat_faces = updated_faces.reshape(-1)
    face_has_original = original_in_face.index_select(0, flat_faces).reshape_as(updated_faces).any(dim=1)
    face_has_duplicate = duplicate_in_face.index_select(0, flat_faces).reshape_as(updated_faces).any(dim=1)
    mixed_faces = face_has_original & face_has_duplicate

    updated_mesh = dict(mesh)
    updated_mesh["vertices"] = updated_vertices
    updated_mesh["faces"] = updated_faces.contiguous()
    original_normals = mesh.get("normals")
    if not isinstance(original_normals, torch.Tensor) or original_normals.shape != vertices.shape:
        original_normals = compute_vertex_normals(vertices, faces)
    else:
        original_normals = original_normals.detach().cpu().float().contiguous()
    updated_mesh["normals"] = torch.cat(
        (original_normals, original_normals.index_select(0, seam_vertex_ids)),
        dim=0,
    ).contiguous()
    report = GeometricLipSeamReport(
        original_vertex_count=vertex_count,
        original_face_count=int(faces.shape[0]),
        final_vertex_count=int(updated_vertices.shape[0]),
        final_face_count=int(updated_faces.shape[0]),
        curve_point_count=int(curve_points.shape[0]),
        candidate_face_count=int(candidate_faces.sum().item()),
        upper_candidate_face_count=int(upper_candidate_faces.sum().item()),
        lower_candidate_face_count=int(lower_candidate_faces.sum().item()),
        seam_vertex_count=int(seam_vertex_ids.numel()),
        duplicated_vertex_count=int(seam_vertex_ids.numel()),
        remapped_lower_face_count=int(remapped_lower_faces.sum().item()),
        mixed_original_duplicate_face_count=int(mixed_faces.sum().item()),
        horizontal_axis=horizontal_axis,
        depth_axis=depth_axis,
        vertical_axis=vertical_axis,
        depth_tolerance=float(depth_tolerance),
        vertical_tolerance=float(vertical_tolerance),
        horizontal_padding=float(horizontal_padding),
        corner_inset_ratio=float(corner_inset_ratio),
        preopen_distance=float(preopen_distance),
        curve_points=curve_points.tolist(),
        seam_vertex_ids=[int(vertex_id) for vertex_id in seam_vertex_ids.tolist()],
        lower_duplicate_vertex_ids=lower_duplicate_vertex_ids,
    )
    return updated_mesh, report


def split_lip_connections(
    mesh: Mapping[str, torch.Tensor],
    lip_regions: LipRegions,
    *,
    remove_mixed_faces: bool = True,
    extra_remove_face_ids: torch.Tensor | Sequence[int] | None = None,
) -> tuple[dict[str, torch.Tensor], LipSplitReport, LipRegions]:
    """Remove upper/lower bridge faces and duplicate shared lower-lip vertices.

    Sapiens-style dense face parsing is expected to have already been projected
    to mesh vertices. Vertices that are marked as both upper and lower lip are
    treated as closed-mouth seam vertices: upper-side faces keep the original
    vertex id while lower-side faces receive a duplicate.
    """

    vertices = mesh["vertices"].detach().cpu().float().contiguous()
    faces = mesh["faces"].detach().cpu().long().contiguous()
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError("split_lip_connections expects triangular faces [F, 3].")

    vertex_count = int(vertices.shape[0])
    upper_ids = normalize_vertex_ids(lip_regions.upper_vertex_ids, vertex_count)
    lower_ids = normalize_vertex_ids(lip_regions.lower_vertex_ids, vertex_count)
    if upper_ids.numel() == 0:
        raise ValueError("Upper lip region is empty after filtering vertex ids.")
    if lower_ids.numel() == 0:
        raise ValueError("Lower lip region is empty after filtering vertex ids.")

    upper_mask = vertex_id_mask(upper_ids, vertex_count)
    lower_mask = vertex_id_mask(lower_ids, vertex_count)
    shared_mask = upper_mask & lower_mask
    exclusive_upper = upper_mask & ~lower_mask
    exclusive_lower = lower_mask & ~upper_mask

    before_faces, before_edges = count_direct_lip_connections(
        faces,
        upper_ids,
        lower_ids,
        vertex_count,
    )

    face_exclusive_upper = (
        exclusive_upper.index_select(0, faces.reshape(-1)).reshape_as(faces).any(dim=-1)
    )
    face_exclusive_lower = (
        exclusive_lower.index_select(0, faces.reshape(-1)).reshape_as(faces).any(dim=-1)
    )
    face_shared = shared_mask.index_select(0, faces.reshape(-1)).reshape_as(faces).any(dim=-1)

    mixed_faces = face_exclusive_upper & face_exclusive_lower
    lower_faces = face_exclusive_lower & ~face_exclusive_upper
    upper_faces = face_exclusive_upper & ~face_exclusive_lower
    neutral_faces = ~(lower_faces | upper_faces | mixed_faces | face_shared)

    keep_face_mask = torch.ones(faces.shape[0], dtype=torch.bool)
    if remove_mixed_faces:
        keep_face_mask &= ~mixed_faces
    if extra_remove_face_ids is not None:
        extra_face_ids = normalize_index_ids(extra_remove_face_ids, int(faces.shape[0]))
        if extra_face_ids.numel() > 0:
            keep_face_mask[extra_face_ids] = False
    kept_faces = faces[keep_face_mask].clone()
    kept_lower_faces = lower_faces[keep_face_mask]

    lower_shared_vertex_ids = _lower_shared_vertex_ids(
        faces=kept_faces,
        lower_face_mask=kept_lower_faces,
        shared_mask=shared_mask,
    )
    lower_duplicate_vertex_ids: dict[int, int] = {}
    updated_vertices = vertices
    if lower_shared_vertex_ids.numel() > 0:
        duplicate_ids = torch.arange(
            vertex_count,
            vertex_count + lower_shared_vertex_ids.numel(),
            dtype=torch.long,
        )
        lower_duplicate_vertex_ids = {
            int(old_id): int(new_id)
            for old_id, new_id in zip(lower_shared_vertex_ids.tolist(), duplicate_ids.tolist())
        }
        updated_vertices = torch.cat(
            (vertices, vertices.index_select(0, lower_shared_vertex_ids)),
            dim=0,
        ).contiguous()
        lower_remap = torch.arange(updated_vertices.shape[0], dtype=torch.long)
        lower_remap[lower_shared_vertex_ids] = duplicate_ids
        kept_faces[kept_lower_faces] = lower_remap.index_select(
            0,
            kept_faces[kept_lower_faces].reshape(-1),
        ).reshape(-1, 3)

    updated_upper_ids = upper_ids
    lower_exclusive_ids = lower_ids[~upper_mask.index_select(0, lower_ids)]
    if lower_duplicate_vertex_ids:
        duplicate_tensor = torch.tensor(
            list(lower_duplicate_vertex_ids.values()),
            dtype=torch.long,
        )
        updated_lower_ids = torch.cat((lower_exclusive_ids, duplicate_tensor), dim=0)
    else:
        updated_lower_ids = lower_exclusive_ids
    updated_lower_ids = updated_lower_ids.unique(sorted=True).contiguous()

    updated_mesh = dict(mesh)
    updated_mesh["vertices"] = updated_vertices
    updated_mesh["faces"] = kept_faces.contiguous()
    updated_mesh["normals"] = compute_vertex_normals(updated_vertices, kept_faces)

    after_faces, after_edges = count_direct_lip_connections(
        kept_faces,
        updated_upper_ids,
        updated_lower_ids,
        int(updated_vertices.shape[0]),
    )
    removed_face_ids = torch.nonzero(~keep_face_mask, as_tuple=False).flatten().tolist()

    report = LipSplitReport(
        original_vertex_count=vertex_count,
        original_face_count=int(faces.shape[0]),
        final_vertex_count=int(updated_vertices.shape[0]),
        final_face_count=int(kept_faces.shape[0]),
        upper_vertex_count=int(upper_ids.numel()),
        lower_vertex_count=int(lower_ids.numel()),
        shared_vertex_count=int(shared_mask.sum().item()),
        upper_face_count=int(upper_faces.sum().item()),
        lower_face_count=int(lower_faces.sum().item()),
        neutral_face_count=int(neutral_faces.sum().item()),
        mixed_face_count=int(mixed_faces.sum().item()),
        removed_face_count=int((~keep_face_mask).sum().item()),
        duplicated_vertex_count=int(len(lower_duplicate_vertex_ids)),
        direct_lip_connection_faces_before=int(before_faces),
        direct_lip_connection_edges_before=int(before_edges),
        direct_lip_connection_faces_after=int(after_faces),
        direct_lip_connection_edges_after=int(after_edges),
        lower_duplicate_vertex_ids=lower_duplicate_vertex_ids,
        removed_face_ids=[int(face_id) for face_id in removed_face_ids],
    )
    return (
        updated_mesh,
        report,
        LipRegions(
            upper_vertex_ids=updated_upper_ids,
            lower_vertex_ids=updated_lower_ids,
            source=f"{lip_regions.source}:split",
        ),
    )


def count_direct_lip_connections(
    faces: torch.Tensor,
    upper_vertex_ids: torch.Tensor | Sequence[int],
    lower_vertex_ids: torch.Tensor | Sequence[int],
    vertex_count: int,
) -> tuple[int, int]:
    faces = faces.detach().cpu().long().contiguous()
    upper_mask = vertex_id_mask(normalize_vertex_ids(upper_vertex_ids, vertex_count), vertex_count)
    lower_mask = vertex_id_mask(normalize_vertex_ids(lower_vertex_ids, vertex_count), vertex_count)
    if faces.numel() == 0:
        return 0, 0

    upper_face = upper_mask.index_select(0, faces.reshape(-1)).reshape_as(faces).any(dim=-1)
    lower_face = lower_mask.index_select(0, faces.reshape(-1)).reshape_as(faces).any(dim=-1)
    connection_faces = upper_face & lower_face

    edges = torch.cat(
        (
            faces[:, (0, 1)],
            faces[:, (1, 2)],
            faces[:, (2, 0)],
        ),
        dim=0,
    )
    edge_upper = upper_mask.index_select(0, edges.reshape(-1)).reshape_as(edges)
    edge_lower = lower_mask.index_select(0, edges.reshape(-1)).reshape_as(edges)
    connection_edges = (edge_upper[:, 0] & edge_lower[:, 1]) | (
        edge_lower[:, 0] & edge_upper[:, 1]
    )
    if connection_edges.any():
        sorted_edges = edges[connection_edges].sort(dim=-1).values
        edge_count = int(torch.unique(sorted_edges, dim=0).shape[0])
    else:
        edge_count = 0
    return int(connection_faces.sum().item()), edge_count


def lip_regions_from_mediapipe_mapping(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    mapping: Mapping[int, int],
    *,
    rings: int = 1,
    include_outer_lip: bool = False,
) -> LipRegions:
    vertex_count = int(vertices.shape[0])
    upper_landmarks = list(INNER_UPPER_LIP_MEDIAPIPE_IDS)
    lower_landmarks = list(INNER_LOWER_LIP_MEDIAPIPE_IDS)
    upper_seed_ids = _vertex_ids_for_landmarks(mapping, upper_landmarks, vertex_count)
    lower_seed_ids = _vertex_ids_for_landmarks(mapping, lower_landmarks, vertex_count)
    if include_outer_lip or _needs_outer_lip_landmarks(upper_seed_ids, lower_seed_ids):
        upper_landmarks.extend(OUTER_UPPER_LIP_MEDIAPIPE_IDS)
        lower_landmarks.extend(OUTER_LOWER_LIP_MEDIAPIPE_IDS)
        upper_seed_ids = _vertex_ids_for_landmarks(mapping, upper_landmarks, vertex_count)
        lower_seed_ids = _vertex_ids_for_landmarks(mapping, lower_landmarks, vertex_count)
    if rings > 0:
        upper_seed_ids = expand_vertex_ids_by_rings(
            faces,
            upper_seed_ids,
            vertex_count=vertex_count,
            rings=rings,
        )
        lower_seed_ids = expand_vertex_ids_by_rings(
            faces,
            lower_seed_ids,
            vertex_count=vertex_count,
            rings=rings,
        )
    return LipRegions(
        upper_vertex_ids=upper_seed_ids,
        lower_vertex_ids=lower_seed_ids,
        source="mediapipe",
    )


def lip_regions_from_mediapipe_paths(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    mapping: Mapping[int, int],
    *,
    rings: int = 2,
    include_outer_lip: bool = True,
) -> LipRegions:
    """Build lip bands by tracing mesh paths between MediaPipe lip landmarks."""

    vertex_count = int(vertices.shape[0])
    upper_landmarks = list(INNER_UPPER_LIP_MEDIAPIPE_IDS)
    lower_landmarks = list(INNER_LOWER_LIP_MEDIAPIPE_IDS)
    if include_outer_lip:
        upper_landmarks.extend(OUTER_UPPER_LIP_MEDIAPIPE_IDS)
        lower_landmarks.extend(OUTER_LOWER_LIP_MEDIAPIPE_IDS)

    upper_seed_ids = _vertex_ids_for_landmarks(mapping, upper_landmarks, vertex_count)
    lower_seed_ids = _vertex_ids_for_landmarks(mapping, lower_landmarks, vertex_count)
    if upper_seed_ids.numel() == 0:
        raise ValueError("No usable upper MediaPipe lip landmarks were found.")
    if lower_seed_ids.numel() == 0:
        raise ValueError("No usable lower MediaPipe lip landmarks were found.")

    neighbors = build_vertex_neighbors(faces, vertex_count)
    upper_path_ids = _landmark_path_vertex_ids(
        mapping,
        upper_landmarks,
        vertex_count,
        neighbors,
    )
    lower_path_ids = _landmark_path_vertex_ids(
        mapping,
        lower_landmarks,
        vertex_count,
        neighbors,
    )
    upper_ids = normalize_vertex_ids(
        list(upper_seed_ids.tolist()) + sorted(upper_path_ids),
        vertex_count,
    )
    lower_ids = normalize_vertex_ids(
        list(lower_seed_ids.tolist()) + sorted(lower_path_ids),
        vertex_count,
    )
    if rings > 0:
        upper_ids = expand_vertex_ids_by_rings(
            faces,
            upper_ids,
            vertex_count=vertex_count,
            rings=rings,
            neighbors=neighbors,
        )
        lower_ids = expand_vertex_ids_by_rings(
            faces,
            lower_ids,
            vertex_count=vertex_count,
            rings=rings,
            neighbors=neighbors,
        )
    return LipRegions(
        upper_vertex_ids=upper_ids,
        lower_vertex_ids=lower_ids,
        source="mediapipe_paths",
    )


def mouth_band_face_ids_from_mediapipe(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    mapping: Mapping[int, int],
    *,
    axes: tuple[int, int] = (0, 2),
    padding: float = 0.0,
    depth_axis: int | None = 1,
    depth_padding: float = 0.006,
) -> torch.Tensor:
    """Faces whose centroids lie in the inner-mouth MediaPipe lip band."""

    vertices = vertices.detach().cpu().float().contiguous()
    faces = faces.detach().cpu().long().contiguous()
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError("mouth_band_face_ids_from_mediapipe expects triangular faces.")
    axis_x, axis_y = int(axes[0]), int(axes[1])
    if axis_x == axis_y or axis_x not in {0, 1, 2} or axis_y not in {0, 1, 2}:
        raise ValueError("axes must contain two different coordinate indices in [0, 2].")
    if depth_axis is not None:
        depth_axis = int(depth_axis)
        if depth_axis not in {0, 1, 2}:
            raise ValueError("depth_axis must be a coordinate index in [0, 2].")

    lip_landmark_ids = list(INNER_UPPER_LIP_MEDIAPIPE_IDS) + list(INNER_LOWER_LIP_MEDIAPIPE_IDS)
    polygon = _lip_landmark_polygon(
        vertices,
        mapping,
        list(INNER_UPPER_LIP_MEDIAPIPE_IDS),
        list(reversed(INNER_LOWER_LIP_MEDIAPIPE_IDS)),
        axes=(axis_x, axis_y),
    )
    if polygon.shape[0] < 3:
        raise ValueError("Not enough MediaPipe lip landmarks to build a mouth band.")
    if padding > 0:
        center = polygon.mean(dim=0, keepdim=True)
        polygon = center + (polygon - center) * (1.0 + float(padding))

    centroids = vertices.index_select(0, faces.reshape(-1)).reshape(-1, 3, 3).mean(dim=1)
    mask = points_in_polygon_2d(
        centroids[:, axis_x],
        centroids[:, axis_y],
        polygon,
    )
    if depth_axis is not None:
        depth_values = _landmark_axis_values(
            vertices,
            mapping,
            lip_landmark_ids,
            axis=depth_axis,
        )
        if depth_values.numel() > 0:
            low = depth_values.min() - float(depth_padding)
            high = depth_values.max() + float(depth_padding)
            mask &= (centroids[:, depth_axis] >= low) & (centroids[:, depth_axis] <= high)
    return torch.nonzero(mask, as_tuple=False).flatten().long().contiguous()


def remap_lower_lip_landmarks(
    mapping: Mapping[int, int],
    split_report: LipSplitReport | GeometricLipSeamReport,
) -> dict[int, int]:
    remapped = {int(key): int(value) for key, value in mapping.items()}
    duplicate_by_original = split_report.lower_duplicate_vertex_ids
    if not duplicate_by_original:
        return remapped
    for mediapipe_id in INNER_LOWER_LIP_MEDIAPIPE_IDS + OUTER_LOWER_LIP_MEDIAPIPE_IDS:
        vertex_id = remapped.get(int(mediapipe_id))
        if vertex_id in duplicate_by_original:
            remapped[int(mediapipe_id)] = duplicate_by_original[int(vertex_id)]
    return remapped


def _inner_lip_center_curve(
    vertices: torch.Tensor,
    mapping: Mapping[int, int],
) -> torch.Tensor:
    points: list[torch.Tensor] = []
    vertex_count = int(vertices.shape[0])
    for upper_landmark_id, lower_landmark_id in INNER_LIP_CENTER_PAIRS:
        upper_vertex_id = mapping.get(int(upper_landmark_id))
        lower_vertex_id = mapping.get(int(lower_landmark_id))
        if upper_vertex_id is None or lower_vertex_id is None:
            continue
        upper_vertex_id = int(upper_vertex_id)
        lower_vertex_id = int(lower_vertex_id)
        if not (
            0 <= upper_vertex_id < vertex_count
            and 0 <= lower_vertex_id < vertex_count
        ):
            continue
        points.append(
            (vertices[upper_vertex_id] + vertices[lower_vertex_id]) * 0.5
        )
    if not points:
        return torch.empty(0, 3, dtype=vertices.dtype)
    return torch.stack(points, dim=0).contiguous()


def _merge_curve_points_with_duplicate_horizontal_values(
    points: torch.Tensor,
    *,
    horizontal_axis: int,
    epsilon: float = 1.0e-7,
) -> torch.Tensor:
    groups: list[list[torch.Tensor]] = []
    for point in points:
        if not groups or abs(
            float(point[horizontal_axis] - groups[-1][-1][horizontal_axis])
        ) > epsilon:
            groups.append([point])
        else:
            groups[-1].append(point)
    return torch.stack(
        [torch.stack(group, dim=0).mean(dim=0) for group in groups],
        dim=0,
    ).contiguous()


def _piecewise_linear_interpolate(
    query: torch.Tensor,
    knot_x: torch.Tensor,
    knot_y: torch.Tensor,
) -> torch.Tensor:
    knot_x = knot_x.contiguous()
    knot_y = knot_y.contiguous()
    indices = torch.searchsorted(knot_x, query.contiguous(), right=True)
    right = indices.clamp(min=1, max=int(knot_x.numel()) - 1)
    left = right - 1
    x0 = knot_x.index_select(0, left)
    x1 = knot_x.index_select(0, right)
    y0 = knot_y.index_select(0, left)
    y1 = knot_y.index_select(0, right)
    weight = (query - x0) / (x1 - x0).clamp_min(1.0e-12)
    return y0 + (y1 - y0) * weight


def load_lip_regions(
    path: Path,
    vertex_count: int,
    *,
    upper_label_ids: Sequence[int] = (),
    lower_label_ids: Sequence[int] = (),
) -> LipRegions:
    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    elif suffix == ".npz":
        payload = dict(np.load(path, allow_pickle=True))
    elif suffix == ".npy":
        payload = {"vertex_labels": np.load(path, allow_pickle=True)}
    elif suffix in {".pt", ".pth"}:
        loaded = torch.load(path, map_location="cpu")
        payload = loaded if isinstance(loaded, Mapping) else {"vertex_labels": loaded}
    else:
        raise ValueError(
            f"Unsupported lip region file {path.suffix!r}. "
            "Use JSON, NPZ, NPY, PT, or PTH."
        )
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected {path} to contain a mapping-like payload.")

    upper = _extract_region_ids(
        payload,
        vertex_count,
        keys=(
            "upper_lip_vertex_ids",
            "upper_vertex_ids",
            "upper_lip_ids",
            "upper_ids",
            "upper_lip",
            "upper",
            "upper_lip_mask",
            "upper_mask",
        ),
    )
    lower = _extract_region_ids(
        payload,
        vertex_count,
        keys=(
            "lower_lip_vertex_ids",
            "lower_vertex_ids",
            "lower_lip_ids",
            "lower_ids",
            "lower_lip",
            "lower",
            "lower_lip_mask",
            "lower_mask",
        ),
    )
    if upper is None or lower is None:
        upper, lower = _extract_regions_from_labels(
            payload,
            vertex_count,
            upper_label_ids=upper_label_ids,
            lower_label_ids=lower_label_ids,
        )
    if upper is None or lower is None:
        raise ValueError(
            f"{path} must provide upper/lower lip vertex ids, masks, or labels."
        )
    return LipRegions(
        upper_vertex_ids=normalize_vertex_ids(upper, vertex_count),
        lower_vertex_ids=normalize_vertex_ids(lower, vertex_count),
        source=str(path),
    )


def load_vertex_ids_arg(value: str | None, vertex_count: int) -> torch.Tensor:
    if value is None or str(value).strip() == "":
        return torch.empty(0, dtype=torch.long)
    candidate = Path(str(value)).expanduser()
    if candidate.is_file():
        text = candidate.read_text(encoding="utf-8")
    else:
        text = str(value)
    ids: list[int] = []
    for token in text.replace(",", " ").split():
        stripped = token.strip()
        if stripped:
            ids.append(int(stripped))
    return normalize_vertex_ids(ids, vertex_count)


def expand_vertex_ids_by_rings(
    faces: torch.Tensor,
    vertex_ids: torch.Tensor | Sequence[int],
    *,
    vertex_count: int,
    rings: int,
    neighbors: list[set[int]] | None = None,
) -> torch.Tensor:
    selected = set(normalize_vertex_ids(vertex_ids, vertex_count).tolist())
    if not selected or rings <= 0:
        return torch.tensor(sorted(selected), dtype=torch.long)
    if neighbors is None:
        neighbors = build_vertex_neighbors(faces, vertex_count)

    frontier = set(selected)
    for _ in range(int(rings)):
        grown: set[int] = set()
        for vertex_id in frontier:
            grown.update(neighbors[vertex_id])
        grown -= selected
        if not grown:
            break
        selected.update(grown)
        frontier = grown
    return torch.tensor(sorted(selected), dtype=torch.long)


def build_vertex_neighbors(
    faces: torch.Tensor,
    vertex_count: int,
) -> list[set[int]]:
    neighbors: list[set[int]] = [set() for _ in range(vertex_count)]
    for a, b, c in faces.detach().cpu().long().tolist():
        if 0 <= a < vertex_count and 0 <= b < vertex_count:
            neighbors[a].add(b)
            neighbors[b].add(a)
        if 0 <= b < vertex_count and 0 <= c < vertex_count:
            neighbors[b].add(c)
            neighbors[c].add(b)
        if 0 <= c < vertex_count and 0 <= a < vertex_count:
            neighbors[c].add(a)
            neighbors[a].add(c)
    return neighbors


def normalize_vertex_ids(
    vertex_ids: torch.Tensor | Sequence[int] | np.ndarray,
    vertex_count: int,
) -> torch.Tensor:
    if isinstance(vertex_ids, torch.Tensor):
        ids = vertex_ids.detach().cpu().long().flatten()
    else:
        ids = torch.as_tensor(vertex_ids, dtype=torch.long).flatten()
    if ids.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    ids = ids[(ids >= 0) & (ids < int(vertex_count))]
    return ids.unique(sorted=True).contiguous()


def normalize_index_ids(
    index_ids: torch.Tensor | Sequence[int] | np.ndarray,
    count: int,
) -> torch.Tensor:
    if isinstance(index_ids, torch.Tensor):
        ids = index_ids.detach().cpu().long().flatten()
    else:
        ids = torch.as_tensor(index_ids, dtype=torch.long).flatten()
    if ids.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    ids = ids[(ids >= 0) & (ids < int(count))]
    return ids.unique(sorted=True).contiguous()


def vertex_id_mask(vertex_ids: torch.Tensor, vertex_count: int) -> torch.Tensor:
    mask = torch.zeros(int(vertex_count), dtype=torch.bool)
    if vertex_ids.numel() > 0:
        mask[vertex_ids.long()] = True
    return mask


def compute_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    if faces.numel() == 0:
        return torch.zeros_like(vertices)
    triangles = vertices[faces]
    face_normals = torch.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=-1,
    )
    normals = torch.zeros_like(vertices)
    for corner in range(3):
        normals.index_add_(0, faces[:, corner], face_normals)
    return F.normalize(normals, dim=-1, eps=1.0e-6).contiguous()


def _lower_shared_vertex_ids(
    *,
    faces: torch.Tensor,
    lower_face_mask: torch.Tensor,
    shared_mask: torch.Tensor,
) -> torch.Tensor:
    if faces.numel() == 0 or not bool(lower_face_mask.any().item()):
        return torch.empty(0, dtype=torch.long)
    lower_faces = faces[lower_face_mask]
    shared_in_lower_faces = shared_mask.index_select(0, lower_faces.reshape(-1))
    if not bool(shared_in_lower_faces.any().item()):
        return torch.empty(0, dtype=torch.long)
    return lower_faces.reshape(-1)[shared_in_lower_faces].unique(sorted=True).contiguous()


def _vertex_ids_for_landmarks(
    mapping: Mapping[int, int],
    landmark_ids: Sequence[int],
    vertex_count: int,
) -> torch.Tensor:
    ids = []
    for landmark_id in landmark_ids:
        vertex_id = mapping.get(int(landmark_id))
        if vertex_id is not None and 0 <= int(vertex_id) < vertex_count:
            ids.append(int(vertex_id))
    return normalize_vertex_ids(ids, vertex_count)


def _ordered_mapped_vertex_ids(
    mapping: Mapping[int, int],
    landmark_ids: Sequence[int],
    vertex_count: int,
) -> list[int]:
    ids: list[int] = []
    for landmark_id in landmark_ids:
        vertex_id = mapping.get(int(landmark_id))
        if vertex_id is None:
            continue
        vertex_id = int(vertex_id)
        if 0 <= vertex_id < vertex_count:
            ids.append(vertex_id)
    return ids


def _compact_polygon_points(points: torch.Tensor) -> torch.Tensor:
    compact: list[torch.Tensor] = []
    for point in points:
        if not compact or not bool(torch.equal(point, compact[-1])):
            compact.append(point)
    if len(compact) > 1 and bool(torch.equal(compact[0], compact[-1])):
        compact.pop()
    if not compact:
        return torch.empty(0, 2, dtype=points.dtype)
    return torch.stack(compact, dim=0).contiguous()


def _landmark_path_vertex_ids(
    mapping: Mapping[int, int],
    landmark_ids: Sequence[int],
    vertex_count: int,
    neighbors: Sequence[set[int]],
) -> set[int]:
    path_ids: set[int] = set()
    previous: int | None = None
    for landmark_id in landmark_ids:
        vertex_id = mapping.get(int(landmark_id))
        if vertex_id is None:
            continue
        vertex_id = int(vertex_id)
        if vertex_id < 0 or vertex_id >= vertex_count:
            continue
        path_ids.add(vertex_id)
        if previous is not None and previous != vertex_id:
            path_ids.update(_shortest_vertex_path(previous, vertex_id, neighbors))
        previous = vertex_id
    return path_ids


def _shortest_vertex_path(
    start: int,
    target: int,
    neighbors: Sequence[set[int]],
) -> list[int]:
    if start == target:
        return [int(start)]
    parents: dict[int, int] = {int(start): -1}
    queue: deque[int] = deque([int(start)])
    while queue:
        current = queue.popleft()
        for neighbor in neighbors[current]:
            neighbor = int(neighbor)
            if neighbor in parents:
                continue
            parents[neighbor] = current
            if neighbor == target:
                path = [target]
                while path[-1] != start:
                    path.append(parents[path[-1]])
                path.reverse()
                return path
            queue.append(neighbor)
    return []


def _lip_landmark_polygon(
    vertices: torch.Tensor,
    mapping: Mapping[int, int],
    first_curve: Sequence[int],
    second_curve: Sequence[int],
    *,
    axes: tuple[int, int],
) -> torch.Tensor:
    points: list[list[float]] = []
    vertex_count = int(vertices.shape[0])
    for landmark_id in list(first_curve) + list(second_curve):
        vertex_id = mapping.get(int(landmark_id))
        if vertex_id is None:
            continue
        vertex_id = int(vertex_id)
        if 0 <= vertex_id < vertex_count:
            points.append(
                [
                    float(vertices[vertex_id, axes[0]].item()),
                    float(vertices[vertex_id, axes[1]].item()),
                ]
            )
    if not points:
        return torch.empty(0, 2, dtype=torch.float32)
    compact: list[list[float]] = []
    for point in points:
        if not compact or point != compact[-1]:
            compact.append(point)
    if len(compact) > 1 and compact[0] == compact[-1]:
        compact.pop()
    return torch.tensor(compact, dtype=torch.float32)


def _landmark_axis_values(
    vertices: torch.Tensor,
    mapping: Mapping[int, int],
    landmark_ids: Sequence[int],
    *,
    axis: int,
) -> torch.Tensor:
    values: list[float] = []
    vertex_count = int(vertices.shape[0])
    for landmark_id in landmark_ids:
        vertex_id = mapping.get(int(landmark_id))
        if vertex_id is None:
            continue
        vertex_id = int(vertex_id)
        if 0 <= vertex_id < vertex_count:
            values.append(float(vertices[vertex_id, int(axis)].item()))
    if not values:
        return torch.empty(0, dtype=torch.float32)
    return torch.tensor(values, dtype=torch.float32)


def points_in_polygon_2d(
    xs: torch.Tensor,
    ys: torch.Tensor,
    polygon: torch.Tensor,
) -> torch.Tensor:
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        return torch.zeros_like(xs, dtype=torch.bool)
    xs = xs.detach().cpu().float()
    ys = ys.detach().cpu().float()
    poly = polygon.detach().cpu().float()
    inside = torch.zeros_like(xs, dtype=torch.bool)
    previous = int(poly.shape[0]) - 1
    for current in range(int(poly.shape[0])):
        xi = poly[current, 0]
        yi = poly[current, 1]
        xj = poly[previous, 0]
        yj = poly[previous, 1]
        crosses = (yi > ys) != (yj > ys)
        intersect_x = (xj - xi) * (ys - yi) / (yj - yi + 1.0e-12) + xi
        inside ^= crosses & (xs < intersect_x)
        previous = current
    return inside


def _needs_outer_lip_landmarks(
    upper_seed_ids: torch.Tensor,
    lower_seed_ids: torch.Tensor,
) -> bool:
    if upper_seed_ids.numel() == 0 or lower_seed_ids.numel() == 0:
        return True
    upper = set(int(vertex_id) for vertex_id in upper_seed_ids.tolist())
    lower = set(int(vertex_id) for vertex_id in lower_seed_ids.tolist())
    return not (upper - lower) or not (lower - upper)


def _extract_region_ids(
    payload: Mapping[str, Any],
    vertex_count: int,
    *,
    keys: Sequence[str],
) -> torch.Tensor | None:
    for key in keys:
        if key not in payload:
            continue
        value = payload[key]
        array = _to_numpy(value)
        if array is None:
            continue
        flat = array.reshape(-1)
        if flat.size == vertex_count and (
            flat.dtype == np.bool_ or np.all((flat == 0) | (flat == 1))
        ):
            return torch.from_numpy(np.nonzero(flat.astype(bool))[0]).long()
        return normalize_vertex_ids(flat.astype(np.int64), vertex_count)
    return None


def _extract_regions_from_labels(
    payload: Mapping[str, Any],
    vertex_count: int,
    *,
    upper_label_ids: Sequence[int],
    lower_label_ids: Sequence[int],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    labels = None
    for key in ("vertex_labels", "labels", "sapiens_labels", "label_ids"):
        if key in payload:
            labels = _to_numpy(payload[key])
            break
    if labels is None:
        return None, None
    labels = labels.reshape(-1)
    if labels.size != vertex_count:
        raise ValueError(
            "Per-vertex lip label arrays must have one label per mesh vertex."
        )
    upper_values = set(int(value) for value in upper_label_ids)
    lower_values = set(int(value) for value in lower_label_ids)
    if not upper_values:
        upper_values = _label_ids_for_names(
            payload,
            ("upper_lip", "upper lip", "upperlip"),
        )
    if not lower_values:
        lower_values = _label_ids_for_names(
            payload,
            ("lower_lip", "lower lip", "lowerlip"),
        )
    if not upper_values or not lower_values:
        return None, None
    upper_mask = np.isin(labels.astype(np.int64), list(upper_values))
    lower_mask = np.isin(labels.astype(np.int64), list(lower_values))
    return (
        torch.from_numpy(np.nonzero(upper_mask)[0]).long(),
        torch.from_numpy(np.nonzero(lower_mask)[0]).long(),
    )


def _label_ids_for_names(payload: Mapping[str, Any], names: Sequence[str]) -> set[int]:
    label_map = payload.get("label_map", payload.get("labels_map"))
    if not isinstance(label_map, Mapping):
        return set()
    wanted = {name.lower().replace("-", "_") for name in names}
    ids: set[int] = set()
    for raw_key, raw_value in label_map.items():
        try:
            label_id = int(raw_key)
            label_name = str(raw_value)
        except (TypeError, ValueError):
            try:
                label_id = int(raw_value)
                label_name = str(raw_key)
            except (TypeError, ValueError):
                continue
        normalized_name = label_name.lower().replace("-", "_")
        if normalized_name in wanted:
            ids.add(label_id)
    return ids


def _to_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return None
