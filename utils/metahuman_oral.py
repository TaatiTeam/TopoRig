from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import trimesh


MOUTH_LANDMARK_IDS = (
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82,
    84, 87, 88, 91, 95, 146, 178, 181, 185, 191,
    267, 269, 270, 291, 308, 310, 311, 312, 314,
    317, 318, 321, 324, 375, 402, 405, 409, 415,
)
UPPER_MOUTH_LANDMARK_IDS = (
    0, 13, 37, 39, 40, 78, 80, 81, 82, 185, 191,
    267, 269, 270, 308, 310, 311, 312, 409, 415,
)
LOWER_MOUTH_LANDMARK_IDS = (
    14, 17, 61, 84, 87, 88, 91, 95, 146, 178, 181,
    291, 314, 317, 318, 321, 324, 375, 402, 405,
)
LEFT_CORNER_ID = 61
RIGHT_CORNER_ID = 291
OUTER_UPPER_LIP_IDS = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)
OUTER_LOWER_LIP_IDS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291)
INNER_UPPER_LIP_IDS = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
INNER_LOWER_LIP_IDS = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308)


@dataclass(frozen=True)
class MouthFrame:
    center_2d: torch.Tensor
    horizontal_direction: torch.Tensor
    vertical_direction: torch.Tensor
    horizontal_span: float
    vertical_span: float
    depth_center: float


@dataclass(frozen=True)
class OralHeadApertureReport:
    scale: float
    depth_padding_ratio: float
    depth_min: float
    depth_max: float
    boundary_distance_ratio: float
    boundary_blend_rings: int
    maximum_protected_edge_ratio: float
    lower_lip_depth_tolerance_ratio: float
    protected_face_count: int
    contour_protected_face_count: int
    lower_lip_protected_face_count: int
    polygon_point_count: int
    projected_overlap_face_count: int
    projected_center_inside_face_count: int
    depth_rejected_overlap_face_count: int
    removed_face_count: int
    original_face_count: int
    final_face_count: int
    aperture_polygon: list[list[float]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OralHeadComponentReport:
    margin_ratio: float
    maximum_component_faces: int
    depth_guard_enabled: bool
    depth_padding_ratio: float
    depth_min: float | None
    depth_max: float | None
    depth_rejected_component_count: int
    lip_landmark_protected_component_count: int
    lip_landmark_protected_face_count: int
    boundary_protected_component_count: int
    boundary_protected_face_count: int
    removed_component_count: int
    removed_face_count: int
    original_face_count: int
    final_face_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MetahumanOralAssemblyReport:
    asset_path: str
    geometry_mode: str
    source_vertex_count: int
    source_face_count: int
    vertex_count: int
    face_count: int
    cropped_vertex_count: int
    cropped_face_count: int
    aperture_crop_scale: float | None
    landmark_count: int
    neutral_horizontal_scale: float
    neutral_vertical_scale: float
    neutral_depth_scale: float
    jaw_open_horizontal_scale: float
    jaw_open_vertical_scale: float
    jaw_open_depth_scale: float
    assembly_scale: float
    depth_offset: float
    level_target_frame: bool
    auto_center_teeth: bool
    maximum_center_offset_ratio: float
    neutral_center_offset_horizontal: float
    neutral_center_offset_vertical: float
    jaw_open_center_offset_horizontal: float
    jaw_open_center_offset_vertical: float
    tooth_depth_offset: float
    tooth_depth_blend_rings: int
    tooth_depth_affected_vertex_count: int
    auto_center_lower_teeth: bool
    neutral_lower_teeth_center_offset_horizontal: float
    jaw_open_lower_teeth_center_offset_horizontal: float
    lower_teeth_depth_offset: float
    lower_teeth_blend_rings: int
    lower_teeth_affected_vertex_count: int
    containment_enabled: bool
    containment_scale: float
    containment_depth_inset: float
    containment_blend_rings: int
    containment_affected_vertex_count: int
    containment_direct_vertex_count: int
    containment_max_planar_displacement: float
    rim_recession_ratio: float
    rim_recession_affected_vertex_count: int
    rim_recession_max_depth_displacement: float
    recessed_shell_horizontal_scale: float
    recessed_shell_start_depth_ratio: float
    recessed_shell_blend_depth_ratio: float
    recessed_shell_affected_vertex_count: int
    recessed_shell_max_horizontal_displacement: float
    rim_transition_trim_enabled: bool
    rim_transition_maximum_distance_ratio: float
    rim_transition_depth_tolerance_ratio: float
    rim_transition_trimmed_face_count: int
    upper_teeth_vertical_offset: float
    upper_teeth_vertical_blend_rings: int
    upper_teeth_vertical_affected_vertex_count: int
    upper_crown_quantile: float
    dental_uv_mask_used: bool
    dental_arch_vertex_count: int
    removed_oral_wall_uv_face_count: int
    removed_tongue_uv_face_count: int
    tongue_planar_scale: float
    tongue_depth_offset: float
    tongue_vertical_offset: float
    tongue_offset_blend_rings: int
    tongue_offset_affected_vertex_count: int
    tooth_vertex_count: int
    tongue_vertex_count: int
    oral_tissue_vertex_count: int
    source_metadata: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _model_axes(values: torch.Tensor) -> torch.Tensor:
    """Return oral-asset coordinates in the model-alignment frame."""

    return values.clone()


def _points_by_landmark(
    landmark_ids: torch.Tensor,
    positions: torch.Tensor,
) -> dict[int, torch.Tensor]:
    return {
        int(landmark_id): positions[index]
        for index, landmark_id in enumerate(landmark_ids.tolist())
    }


def _target_landmark_points(
    vertices: torch.Tensor,
    mapping: Mapping[int, int],
) -> dict[int, torch.Tensor]:
    points: dict[int, torch.Tensor] = {}
    for raw_landmark_id, raw_vertex_id in mapping.items():
        landmark_id = int(raw_landmark_id)
        vertex_id = int(raw_vertex_id)
        if landmark_id in MOUTH_LANDMARK_IDS and 0 <= vertex_id < vertices.shape[0]:
            points[landmark_id] = vertices[vertex_id]
    return points


def _mouth_frame(
    points: Mapping[int, torch.Tensor],
    *,
    horizontal_axis: int,
    depth_axis: int,
    vertical_axis: int,
    level_horizontal: bool = False,
) -> MouthFrame:
    required = set(MOUTH_LANDMARK_IDS) & set(points)
    if LEFT_CORNER_ID not in points or RIGHT_CORNER_ID not in points or len(required) < 12:
        raise ValueError(
            "Oral assembly fitting requires both mouth corners and at least 12 "
            "mapped MediaPipe mouth landmarks."
        )

    axes = (horizontal_axis, vertical_axis)
    left = points[LEFT_CORNER_ID][list(axes)].float()
    right = points[RIGHT_CORNER_ID][list(axes)].float()
    horizontal = right - left
    horizontal_span = float(
        (horizontal[0].abs() if level_horizontal else horizontal.norm()).item()
    )
    if level_horizontal and horizontal_span <= 1.0e-6:
        horizontal_span = float(horizontal.norm().item())
    if horizontal_span <= 1.0e-6:
        raise ValueError("The mapped mouth corners have no usable horizontal span.")
    if level_horizontal:
        horizontal_sign = 1.0 if float(horizontal[0].item()) >= 0.0 else -1.0
        horizontal_direction = horizontal.new_tensor((horizontal_sign, 0.0))
    else:
        horizontal_direction = horizontal / horizontal_span
    vertical_direction = torch.stack(
        (-horizontal_direction[1], horizontal_direction[0])
    )

    upper = [points[index][list(axes)].float() for index in UPPER_MOUTH_LANDMARK_IDS if index in points]
    lower = [points[index][list(axes)].float() for index in LOWER_MOUTH_LANDMARK_IDS if index in points]
    if upper and lower:
        upper_to_lower = torch.stack(upper).mean(dim=0) - torch.stack(lower).mean(dim=0)
        if float(torch.dot(vertical_direction, upper_to_lower).item()) < 0.0:
            vertical_direction = -vertical_direction

    mouth_positions = torch.stack([points[index].float() for index in sorted(required)])
    mouth_2d = mouth_positions[:, list(axes)]
    corner_center = 0.5 * (left + right)
    vertical_coordinates = (mouth_2d - corner_center) @ vertical_direction
    vertical_min = vertical_coordinates.amin()
    vertical_max = vertical_coordinates.amax()
    vertical_span = float((vertical_max - vertical_min).item())
    if vertical_span <= 1.0e-6:
        raise ValueError("The mapped mouth landmarks have no usable vertical span.")
    center_2d = corner_center + vertical_direction * (0.5 * (vertical_min + vertical_max))
    depth_center = float(mouth_positions[:, depth_axis].median().item())
    return MouthFrame(
        center_2d=center_2d,
        horizontal_direction=horizontal_direction,
        vertical_direction=vertical_direction,
        horizontal_span=horizontal_span,
        vertical_span=vertical_span,
        depth_center=depth_center,
    )


def _fit_to_frame(
    vertices: torch.Tensor,
    source_frame: MouthFrame,
    target_frame: MouthFrame,
    *,
    horizontal_axis: int,
    depth_axis: int,
    vertical_axis: int,
    assembly_scale: float,
    depth_offset: float,
) -> tuple[torch.Tensor, tuple[float, float, float]]:
    horizontal_scale = target_frame.horizontal_span / source_frame.horizontal_span
    vertical_scale = target_frame.vertical_span / source_frame.vertical_span
    depth_scale = (horizontal_scale * vertical_scale) ** 0.5

    source_2d = vertices[:, (horizontal_axis, vertical_axis)]
    relative_2d = source_2d - source_frame.center_2d
    local_horizontal = relative_2d @ source_frame.horizontal_direction
    local_vertical = relative_2d @ source_frame.vertical_direction
    fitted_2d = (
        target_frame.center_2d
        + target_frame.horizontal_direction
        * local_horizontal.unsqueeze(1)
        * horizontal_scale
        * float(assembly_scale)
        + target_frame.vertical_direction
        * local_vertical.unsqueeze(1)
        * vertical_scale
        * float(assembly_scale)
    )

    fitted = vertices.new_empty(vertices.shape)
    fitted[:, horizontal_axis] = fitted_2d[:, 0]
    fitted[:, vertical_axis] = fitted_2d[:, 1]
    fitted[:, depth_axis] = (
        target_frame.depth_center
        + (vertices[:, depth_axis] - source_frame.depth_center)
        * depth_scale
        * float(assembly_scale)
        + float(depth_offset)
    )
    return fitted.contiguous(), (
        float(horizontal_scale),
        float(vertical_scale),
        float(depth_scale),
    )


def _oral_vertex_colors(
    basis_vertices: torch.Tensor,
    jaw_open_vertices: torch.Tensor,
    dental_arch_mask: torch.Tensor | None = None,
    tongue_uv_mask: torch.Tensor | None = None,
    faces: torch.Tensor | None = None,
    upper_crown_quantile: float = 0.72,
) -> tuple[torch.Tensor, int, int, int]:
    if not 0.0 <= upper_crown_quantile <= 1.0:
        raise ValueError("Upper-crown quantile must be in [0, 1].")
    model_basis = _model_axes(basis_vertices)
    model_jaw = _model_axes(jaw_open_vertices)
    movement = (model_jaw - model_basis).norm(dim=1)
    moving = movement > max(1.0e-5, float(movement.max().item()) * 0.02)

    depth = model_jaw[:, 1]
    depth_min = depth.amin()
    depth_span = (depth.amax() - depth_min).clamp_min(1.0e-6)
    depth_normalized = (depth - depth_min) / depth_span
    static_front = depth_normalized < 0.25
    moving_front = depth_normalized < 0.40

    static_front_z = model_jaw[(~moving) & static_front, 2]
    moving_front_z = model_jaw[moving & moving_front, 2]
    upper_cut = (
        torch.quantile(static_front_z, float(upper_crown_quantile))
        if static_front_z.numel()
        else model_jaw[:, 2].median()
    )
    lower_cut = (
        torch.quantile(moving_front_z, 0.15)
        if moving_front_z.numel()
        else model_jaw[:, 2].median()
    )
    teeth = ((~moving) & static_front & (model_jaw[:, 2] <= upper_cut)) | (
        moving & moving_front & (model_jaw[:, 2] >= lower_cut)
    )
    if dental_arch_mask is not None:
        teeth &= dental_arch_mask

    if dental_arch_mask is not None and faces is not None and faces.numel() > 0:
        triangles = model_jaw[faces]
        face_normals = torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=1,
        )
        vertex_normals = torch.zeros_like(model_jaw)
        for corner in range(3):
            vertex_normals.index_add_(0, faces[:, corner], face_normals)
        vertex_normals = torch.nn.functional.normalize(
            vertex_normals,
            dim=1,
            eps=1.0e-8,
        )

        # The source UV dental strips include gum tissue, so selecting the
        # entire strip paints the oral walls as teeth. Recover only strongly
        # occlusal rear surfaces in addition to the reliable front crowns.
        upper_rear_crowns = (
            (~moving)
            & dental_arch_mask
            & (depth_normalized >= 0.18)
            & (vertex_normals[:, 2] <= -0.55)
        )
        lower_rear_crowns = (
            moving
            & dental_arch_mask
            & (depth_normalized >= 0.25)
            & (vertex_normals[:, 2] >= 0.55)
        )
        teeth |= upper_rear_crowns | lower_rear_crowns
    horizontal = model_jaw[:, 0]
    horizontal_center = 0.5 * (horizontal.amin() + horizontal.amax())
    horizontal_half_span = 0.5 * (horizontal.amax() - horizontal.amin()).clamp_min(1.0e-6)
    tongue = (
        moving
        & ~teeth
        & (depth_normalized > 0.28)
        & ((horizontal - horizontal_center).abs() < 0.62 * horizontal_half_span)
    )
    if tongue_uv_mask is not None:
        tongue = tongue_uv_mask & ~teeth

    colors = torch.empty((basis_vertices.shape[0], 3), dtype=torch.float32)
    colors[:] = torch.tensor((0.34, 0.075, 0.085), dtype=torch.float32)
    colors[tongue] = torch.tensor((0.58, 0.20, 0.23), dtype=torch.float32)
    colors[teeth] = torch.tensor((0.91, 0.86, 0.72), dtype=torch.float32)
    tooth_count = int(teeth.sum().item())
    tongue_count = int(tongue.sum().item())
    tissue_count = int(colors.shape[0] - tooth_count - tongue_count)
    return colors.contiguous(), tooth_count, tongue_count, tissue_count


def _dental_arch_vertex_mask(
    faces: torch.Tensor,
    face_uvs: torch.Tensor,
    vertex_count: int,
    *,
    minimum_v: float = 0.61,
) -> torch.Tensor:
    """Map the upper/lower dental UV strips to a robust per-vertex mask."""

    if face_uvs.shape != (*faces.shape, 2):
        raise ValueError("Oral face UVs must have shape [F, 3, 2].")
    corner_count = torch.zeros(vertex_count, dtype=torch.int32)
    dental_corner_count = torch.zeros(vertex_count, dtype=torch.int32)
    for corner in range(3):
        vertex_ids = faces[:, corner]
        corner_count.index_add_(
            0,
            vertex_ids,
            torch.ones_like(vertex_ids, dtype=torch.int32),
        )
        dental_corner_count.index_add_(
            0,
            vertex_ids,
            (face_uvs[:, corner, 1] >= float(minimum_v)).to(torch.int32),
        )
    required_count = torch.clamp((corner_count + 1) // 2, min=1)
    return (dental_corner_count >= required_count).contiguous()


def _tongue_uv_vertex_mask(
    faces: torch.Tensor,
    face_uvs: torch.Tensor,
    vertex_count: int,
) -> torch.Tensor:
    """Map the isolated lower-left tongue UV island to mesh vertices."""

    if face_uvs.shape != (*faces.shape, 2):
        raise ValueError("Oral face UVs must have shape [F, 3, 2].")
    corner_count = torch.zeros(vertex_count, dtype=torch.int32)
    tongue_corner_count = torch.zeros(vertex_count, dtype=torch.int32)
    tongue_corners = (face_uvs[..., 0] < 0.42) & (face_uvs[..., 1] < 0.61)
    for corner in range(3):
        vertex_ids = faces[:, corner]
        corner_count.index_add_(
            0,
            vertex_ids,
            torch.ones_like(vertex_ids, dtype=torch.int32),
        )
        tongue_corner_count.index_add_(
            0,
            vertex_ids,
            tongue_corners[:, corner].to(torch.int32),
        )
    required_count = torch.clamp((corner_count + 1) // 2, min=1)
    return (tongue_corner_count >= required_count).contiguous()


def _offset_tongue(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    tongue_mask: torch.Tensor,
    *,
    horizontal_axis: int,
    depth_axis: int,
    vertical_axis: int,
    planar_scale: float,
    depth_offset: float,
    vertical_offset: float,
    blend_rings: int,
) -> tuple[torch.Tensor, int]:
    """Scale and position the UV-defined tongue while preserving attachment."""

    if planar_scale <= 0.0:
        raise ValueError("Tongue planar scale must be positive.")
    if blend_rings < 0:
        raise ValueError("Tongue offset blend rings cannot be negative.")
    if planar_scale == 1.0 and depth_offset == 0.0 and vertical_offset == 0.0:
        return vertices, 0
    weights = vertices.new_zeros(vertices.shape[0])
    weights[tongue_mask] = 1.0
    frontier = tongue_mask.clone()
    visited = tongue_mask.clone()
    for ring in range(1, blend_rings + 1):
        touching_faces = frontier[faces].any(dim=1)
        neighbors = torch.zeros_like(frontier)
        if bool(touching_faces.any().item()):
            neighbors[faces[touching_faces].reshape(-1)] = True
        frontier = neighbors & ~visited
        if not bool(frontier.any().item()):
            break
        weights[frontier] = 1.0 - ring / float(blend_rings + 1)
        visited |= frontier

    moved = vertices.clone()
    if planar_scale != 1.0 and bool(tongue_mask.any().item()):
        tongue_center = vertices[tongue_mask][:, (horizontal_axis, vertical_axis)].mean(
            dim=0
        )
        scale_delta = float(planar_scale) - 1.0
        moved[:, horizontal_axis] += (
            weights
            * (vertices[:, horizontal_axis] - tongue_center[0])
            * scale_delta
        )
        moved[:, vertical_axis] += (
            weights
            * (vertices[:, vertical_axis] - tongue_center[1])
            * scale_delta
        )
    moved[:, depth_axis] += weights * float(depth_offset)
    moved[:, vertical_axis] += weights * float(vertical_offset)
    return moved.contiguous(), int((weights > 0.0).sum().item())


def _remove_tongue_uv_faces(
    faces: torch.Tensor,
    face_uvs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Remove the isolated lower-left tongue island from the MetaHuman UV map."""

    if face_uvs.shape != (*faces.shape, 2):
        raise ValueError("Oral face UVs must have shape [F, 3, 2].")
    centers = face_uvs.mean(dim=1)
    tongue_faces = (centers[:, 0] < 0.42) & (centers[:, 1] < 0.61)
    return (
        faces[~tongue_faces].contiguous(),
        face_uvs[~tongue_faces].contiguous(),
        int(tongue_faces.sum().item()),
    )


def _remove_oral_wall_uv_faces(
    faces: torch.Tensor,
    face_uvs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Keep the dental strips and tongue while removing source wall panels."""

    if face_uvs.shape != (*faces.shape, 2):
        raise ValueError("Oral face UVs must have shape [F, 3, 2].")
    centers = face_uvs.mean(dim=1)
    dental_faces = centers[:, 1] >= 0.61
    tongue_faces = (centers[:, 0] < 0.42) & (centers[:, 1] < 0.61)
    keep = dental_faces | tongue_faces
    removed_count = int((~keep).sum().item())
    return faces[keep].contiguous(), face_uvs[keep].contiguous(), removed_count


def _offset_teeth_in_depth(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    colors: torch.Tensor,
    *,
    depth_axis: int,
    offset: float,
    blend_rings: int,
) -> tuple[torch.Tensor, int]:
    """Move tooth crowns in depth and taper the displacement into nearby tissue."""

    if blend_rings < 0:
        raise ValueError("Tooth depth blend rings cannot be negative.")
    if offset == 0.0:
        return vertices, 0
    tooth_rgb = colors.new_tensor((0.91, 0.86, 0.72))
    tooth_mask = (colors - tooth_rgb).abs().amax(dim=1) < 1.0e-6
    if not bool(tooth_mask.any().item()):
        return vertices, 0

    weights = vertices.new_zeros(vertices.shape[0])
    weights[tooth_mask] = 1.0
    frontier = tooth_mask.clone()
    visited = tooth_mask.clone()
    for ring in range(1, blend_rings + 1):
        touching_faces = frontier[faces].any(dim=1)
        neighbors = torch.zeros_like(frontier)
        if bool(touching_faces.any().item()):
            neighbors[faces[touching_faces].reshape(-1)] = True
        frontier = neighbors & ~visited
        if not bool(frontier.any().item()):
            break
        weights[frontier] = 1.0 - ring / float(blend_rings + 1)
        visited |= frontier

    moved = vertices.clone()
    moved[:, depth_axis] += weights * float(offset)
    return moved.contiguous(), int((weights > 0.0).sum().item())


def _offset_upper_teeth_vertically(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    colors: torch.Tensor,
    moving_mask: torch.Tensor,
    *,
    vertical_axis: int,
    offset: float,
    blend_rings: int,
) -> tuple[torch.Tensor, int]:
    """Reposition static upper crowns without dragging the moving lower jaw."""

    if blend_rings < 0:
        raise ValueError("Upper-tooth vertical blend rings cannot be negative.")
    if offset == 0.0:
        return vertices, 0
    tooth_rgb = colors.new_tensor((0.91, 0.86, 0.72))
    upper_mask = ((colors - tooth_rgb).abs().amax(dim=1) < 1.0e-6) & ~moving_mask
    if not bool(upper_mask.any().item()):
        return vertices, 0

    weights = vertices.new_zeros(vertices.shape[0])
    weights[upper_mask] = 1.0
    frontier = upper_mask.clone()
    visited = upper_mask.clone()
    for ring in range(1, blend_rings + 1):
        touching_faces = frontier[faces].any(dim=1)
        neighbors = torch.zeros_like(frontier)
        if bool(touching_faces.any().item()):
            neighbors[faces[touching_faces].reshape(-1)] = True
        frontier = neighbors & ~visited & ~moving_mask
        if not bool(frontier.any().item()):
            break
        weights[frontier] = 1.0 - ring / float(blend_rings + 1)
        visited |= frontier

    moved = vertices.clone()
    moved[:, vertical_axis] += weights * float(offset)
    return moved.contiguous(), int((weights > 0.0).sum().item())


def _adjust_lower_dental_region(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    colors: torch.Tensor,
    moving_mask: torch.Tensor,
    target_frame: MouthFrame,
    *,
    horizontal_axis: int,
    vertical_axis: int,
    depth_axis: int,
    auto_center: bool,
    depth_offset: float,
    blend_rings: int,
    maximum_center_offset_ratio: float,
) -> tuple[torch.Tensor, float, int]:
    """Center and advance lower crowns with a taper into moving gum tissue."""

    if blend_rings < 0:
        raise ValueError("Lower-tooth blend rings cannot be negative.")
    tooth_rgb = colors.new_tensor((0.91, 0.86, 0.72))
    lower_teeth = (
        (colors - tooth_rgb).abs().amax(dim=1) < 1.0e-6
    ) & moving_mask
    if not bool(lower_teeth.any().item()):
        return vertices, 0.0, 0

    horizontal_offset = vertices.new_zeros(())
    if auto_center:
        axes = (int(horizontal_axis), int(vertical_axis))
        tooth_points = vertices[lower_teeth][:, axes]
        tooth_center = 0.5 * (tooth_points.amin(dim=0) + tooth_points.amax(dim=0))
        raw_offset = target_frame.center_2d - tooth_center
        horizontal_offset = torch.dot(
            raw_offset,
            target_frame.horizontal_direction,
        ).clamp(
            min=-float(maximum_center_offset_ratio) * target_frame.horizontal_span,
            max=float(maximum_center_offset_ratio) * target_frame.horizontal_span,
        )
    if float(horizontal_offset.item()) == 0.0 and depth_offset == 0.0:
        return vertices, 0.0, 0

    weights = vertices.new_zeros(vertices.shape[0])
    weights[lower_teeth] = 1.0
    frontier = lower_teeth.clone()
    visited = lower_teeth.clone()
    for ring in range(1, blend_rings + 1):
        touching_faces = frontier[faces].any(dim=1)
        neighbors = torch.zeros_like(frontier)
        if bool(touching_faces.any().item()):
            neighbors[faces[touching_faces].reshape(-1)] = True
        frontier = neighbors & ~visited & moving_mask
        if not bool(frontier.any().item()):
            break
        weights[frontier] = 1.0 - ring / float(blend_rings + 1)
        visited |= frontier

    horizontal_displacement = (
        target_frame.horizontal_direction * horizontal_offset
    )
    moved = vertices.clone()
    moved[:, horizontal_axis] += weights * horizontal_displacement[0]
    moved[:, vertical_axis] += weights * horizontal_displacement[1]
    moved[:, depth_axis] += weights * float(depth_offset)
    return (
        moved.contiguous(),
        float(horizontal_offset.item()),
        int((weights > 0.0).sum().item()),
    )


def _center_oral_teeth_in_frame(
    vertices: torch.Tensor,
    colors: torch.Tensor,
    target_frame: MouthFrame,
    *,
    horizontal_axis: int,
    vertical_axis: int,
    maximum_offset_ratio: float,
    center_vertically: bool = False,
) -> tuple[torch.Tensor, tuple[float, float]]:
    """Center visible tooth bounds with a correction bounded by mouth size."""

    tooth_rgb = colors.new_tensor((0.91, 0.86, 0.72))
    tooth_mask = (colors - tooth_rgb).abs().amax(dim=1) < 1.0e-6
    if not bool(tooth_mask.any().item()):
        return vertices, (0.0, 0.0)

    axes = (int(horizontal_axis), int(vertical_axis))
    tooth_points = vertices[tooth_mask][:, axes]
    tooth_center = 0.5 * (tooth_points.amin(dim=0) + tooth_points.amax(dim=0))
    raw_offset = target_frame.center_2d - tooth_center
    horizontal_offset = torch.dot(
        raw_offset,
        target_frame.horizontal_direction,
    ).clamp(
        min=-float(maximum_offset_ratio) * target_frame.horizontal_span,
        max=float(maximum_offset_ratio) * target_frame.horizontal_span,
    )
    vertical_offset = raw_offset.new_zeros(())
    if center_vertically:
        vertical_offset = torch.dot(
            raw_offset,
            target_frame.vertical_direction,
        ).clamp(
            min=-float(maximum_offset_ratio) * target_frame.vertical_span,
            max=float(maximum_offset_ratio) * target_frame.vertical_span,
        )
    bounded_offset = (
        target_frame.horizontal_direction * horizontal_offset
        + target_frame.vertical_direction * vertical_offset
    )
    centered = vertices.clone()
    centered[:, horizontal_axis] += bounded_offset[0]
    centered[:, vertical_axis] += bounded_offset[1]
    return centered.contiguous(), (
        float(horizontal_offset.item()),
        float(vertical_offset.item()),
    )


def _crop_to_outer_lip_aperture(
    neutral_vertices: torch.Tensor,
    jaw_open_vertices: torch.Tensor,
    faces: torch.Tensor,
    colors: torch.Tensor,
    target_jaw_points: Mapping[int, torch.Tensor],
    *,
    horizontal_axis: int,
    vertical_axis: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale < 1.0:
        raise ValueError("Oral aperture crop scale must be at least 1.0.")
    polygon_ids = OUTER_UPPER_LIP_IDS + tuple(
        reversed(OUTER_LOWER_LIP_IDS[1:-1])
    )
    missing = [landmark_id for landmark_id in polygon_ids if landmark_id not in target_jaw_points]
    if missing:
        raise ValueError(
            "Oral aperture cropping requires all outer-lip landmarks; missing "
            + ", ".join(str(value) for value in sorted(set(missing)))
        )
    axes = (int(horizontal_axis), int(vertical_axis))
    polygon = torch.stack(
        [target_jaw_points[landmark_id][list(axes)] for landmark_id in polygon_ids]
    ).float()
    center = polygon.mean(dim=0, keepdim=True)
    polygon = center + (polygon - center) * float(scale)

    points = jaw_open_vertices[:, axes]
    inside = torch.zeros(points.shape[0], dtype=torch.bool)
    previous = polygon[-1]
    for current in polygon:
        crosses_vertical = (current[1] > points[:, 1]) != (
            previous[1] > points[:, 1]
        )
        denominator = previous[1] - current[1]
        if abs(float(denominator.item())) < 1.0e-8:
            denominator = denominator.new_tensor(1.0e-8)
        intersection_x = (
            (previous[0] - current[0])
            * (points[:, 1] - current[1])
            / denominator
            + current[0]
        )
        inside ^= crosses_vertical & (points[:, 0] < intersection_x)
        previous = current

    keep_faces = inside[faces].all(dim=1)
    cropped_faces = faces[keep_faces]
    if cropped_faces.numel() == 0:
        raise ValueError("Oral aperture cropping removed every oral face.")
    used_vertex_ids = torch.unique(cropped_faces.reshape(-1), sorted=True)
    remap = torch.full((neutral_vertices.shape[0],), -1, dtype=torch.long)
    remap[used_vertex_ids] = torch.arange(used_vertex_ids.numel(), dtype=torch.long)
    return (
        neutral_vertices[used_vertex_ids].contiguous(),
        jaw_open_vertices[used_vertex_ids].contiguous(),
        remap[cropped_faces].contiguous(),
        colors[used_vertex_ids].contiguous(),
    )


def _outer_lip_polygon(
    target_points: Mapping[int, torch.Tensor],
    *,
    horizontal_axis: int,
    vertical_axis: int,
    scale: float,
) -> torch.Tensor:
    if not 0.0 < scale:
        raise ValueError("Outer-lip polygon scale must be positive.")
    polygon_ids = OUTER_UPPER_LIP_IDS + tuple(
        reversed(OUTER_LOWER_LIP_IDS[1:-1])
    )
    missing = [
        landmark_id
        for landmark_id in polygon_ids
        if landmark_id not in target_points
    ]
    if missing:
        raise ValueError(
            "Oral containment requires all outer-lip landmarks; missing "
            + ", ".join(str(value) for value in sorted(set(missing)))
        )
    axes = (int(horizontal_axis), int(vertical_axis))
    polygon = torch.stack(
        [target_points[landmark_id][list(axes)] for landmark_id in polygon_ids]
    ).float()
    center = polygon.mean(dim=0, keepdim=True)
    return (center + (polygon - center) * float(scale)).contiguous()


def _lower_lip_ribbon_polygon(
    target_points: Mapping[int, torch.Tensor],
    *,
    horizontal_axis: int,
    vertical_axis: int,
) -> torch.Tensor:
    polygon_ids = INNER_LOWER_LIP_IDS + tuple(reversed(OUTER_LOWER_LIP_IDS))
    missing = [landmark_id for landmark_id in polygon_ids if landmark_id not in target_points]
    if missing:
        raise ValueError(
            "Lower-lip protection requires all inner and outer lower-lip landmarks; "
            "missing " + ", ".join(str(value) for value in sorted(set(missing)))
        )
    axes = (int(horizontal_axis), int(vertical_axis))
    return torch.stack(
        [target_points[landmark_id][list(axes)] for landmark_id in polygon_ids]
    ).float().contiguous()


def _points_inside_polygon(
    points: torch.Tensor,
    polygon: torch.Tensor,
) -> torch.Tensor:
    inside = torch.zeros(points.shape[0], dtype=torch.bool)
    previous = polygon[-1]
    for current in polygon:
        crosses_vertical = (current[1] > points[:, 1]) != (
            previous[1] > points[:, 1]
        )
        denominator = previous[1] - current[1]
        safe_denominator = torch.where(
            denominator.abs() < 1.0e-8,
            denominator.new_tensor(1.0e-8),
            denominator,
        )
        intersection_x = (
            (previous[0] - current[0])
            * (points[:, 1] - current[1])
            / safe_denominator
            + current[0]
        )
        inside ^= crosses_vertical & (points[:, 0] < intersection_x)
        previous = current
    return inside


def _contain_oral_vertices_in_polygon(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    polygon: torch.Tensor,
    *,
    horizontal_axis: int,
    depth_axis: int,
    vertical_axis: int,
    depth_inset: float,
    blend_rings: int,
    movable_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, int, float]:
    """Pull projected oral outliers inside the lips and recess their neighbors."""

    if blend_rings < 0:
        raise ValueError("Oral-containment blend rings cannot be negative.")
    if movable_mask is not None and movable_mask.shape != (vertices.shape[0],):
        raise ValueError("Oral-containment movable mask must match the vertex count.")
    axes = (int(horizontal_axis), int(vertical_axis))
    points = vertices[:, axes]
    direct_mask = ~_points_inside_polygon(points, polygon)
    if movable_mask is not None:
        direct_mask &= movable_mask
    direct_count = int(direct_mask.sum().item())
    if direct_count == 0:
        return vertices, 0, 0, 0.0

    segment_start = polygon
    segment_end = torch.roll(polygon, shifts=-1, dims=0)
    segment = segment_end - segment_start
    relative = points[direct_mask].unsqueeze(1) - segment_start.unsqueeze(0)
    denominator = segment.square().sum(dim=1).clamp_min(1.0e-12)
    along = (
        (relative * segment.unsqueeze(0)).sum(dim=2)
        / denominator.unsqueeze(0)
    ).clamp(0.0, 1.0)
    closest = segment_start.unsqueeze(0) + along.unsqueeze(2) * segment.unsqueeze(0)
    distances = (points[direct_mask].unsqueeze(1) - closest).square().sum(dim=2)
    nearest_segment = distances.argmin(dim=1)
    nearest = closest[
        torch.arange(direct_count, dtype=torch.long),
        nearest_segment,
    ]

    displacement = vertices.new_zeros(vertices.shape)
    displacement[direct_mask, horizontal_axis] = nearest[:, 0] - points[direct_mask, 0]
    displacement[direct_mask, vertical_axis] = nearest[:, 1] - points[direct_mask, 1]
    displacement[direct_mask, depth_axis] = float(depth_inset)
    max_planar_displacement = float(
        (nearest - points[direct_mask]).norm(dim=1).amax().item()
    )

    frontier = direct_mask.clone()
    visited = direct_mask.clone()
    for ring in range(1, blend_rings + 1):
        touching_faces = frontier[faces].any(dim=1)
        candidate_mask = torch.zeros_like(frontier)
        if bool(touching_faces.any().item()):
            candidate_mask[faces[touching_faces].reshape(-1)] = True
        candidate_mask &= ~visited
        if movable_mask is not None:
            candidate_mask &= movable_mask
        if not bool(candidate_mask.any().item()):
            break

        touching = touching_faces.nonzero(as_tuple=False).flatten()
        candidate_ids = candidate_mask.nonzero(as_tuple=False).flatten()
        accumulated = vertices.new_zeros(vertices.shape)
        counts = vertices.new_zeros(vertices.shape[0])
        for corner in range(3):
            ids = faces[touching, corner]
            accumulated.index_add_(0, ids, displacement[faces[touching]].mean(dim=1))
            counts.index_add_(0, ids, torch.ones_like(ids, dtype=vertices.dtype))
        decay = 1.0 - ring / float(blend_rings + 1)
        displacement[candidate_ids] = (
            accumulated[candidate_ids]
            / counts[candidate_ids].clamp_min(1.0).unsqueeze(1)
            * decay
        )
        visited |= candidate_mask
        frontier = candidate_mask

    moved = vertices + displacement
    return (
        moved.contiguous(),
        int(visited.sum().item()),
        direct_count,
        max_planar_displacement,
    )


def _recess_oral_tissue_behind_inner_lips(
    vertices: torch.Tensor,
    colors: torch.Tensor,
    target_points: Mapping[int, torch.Tensor],
    *,
    horizontal_axis: int,
    depth_axis: int,
    vertical_axis: int,
    recession_ratio: float,
) -> tuple[torch.Tensor, int, float]:
    """Keep oral geometry outside the opening behind the local inner-lip rim."""

    if recession_ratio < 0.0:
        raise ValueError("Oral rim-recession ratio cannot be negative.")
    if recession_ratio == 0.0:
        return vertices, 0, 0.0

    contour_ids = INNER_UPPER_LIP_IDS + tuple(
        reversed(INNER_LOWER_LIP_IDS[1:-1])
    )
    missing = [landmark_id for landmark_id in contour_ids if landmark_id not in target_points]
    if missing:
        raise ValueError(
            "Oral rim recession requires all inner-lip landmarks; missing "
            + ", ".join(str(value) for value in sorted(set(missing)))
        )

    axes = (int(horizontal_axis), int(vertical_axis))
    contour_points = torch.stack(
        [target_points[landmark_id].float() for landmark_id in contour_ids]
    )
    polygon = contour_points[:, axes]
    polygon_depth = contour_points[:, depth_axis]
    projected_span = float(
        (polygon.amax(dim=0) - polygon.amin(dim=0)).amax().item()
    )
    if projected_span <= 1.0e-8:
        return vertices, 0, 0.0

    tongue_rgb = colors.new_tensor((0.58, 0.20, 0.23))
    tongue_mask = (colors - tongue_rgb).abs().amax(dim=1) < 1.0e-6
    tooth_rgb = colors.new_tensor((0.91, 0.86, 0.72))
    tooth_mask = (colors - tooth_rgb).abs().amax(dim=1) < 1.0e-6
    outside_opening = ~_points_inside_polygon(vertices[:, axes], polygon)
    tissue_mask = ~(tooth_mask | tongue_mask)
    candidates = tissue_mask & outside_opening
    candidate_ids = torch.nonzero(candidates, as_tuple=False).flatten()
    if candidate_ids.numel() == 0:
        return vertices, 0, 0.0

    segment_start = polygon
    segment_end = torch.roll(polygon, shifts=-1, dims=0)
    segment = segment_end - segment_start
    candidate_points = vertices[candidate_ids][:, axes]
    relative = candidate_points.unsqueeze(1) - segment_start.unsqueeze(0)
    along = (
        (relative * segment.unsqueeze(0)).sum(dim=2)
        / segment.square().sum(dim=1).clamp_min(1.0e-12).unsqueeze(0)
    ).clamp(0.0, 1.0)
    closest = segment_start.unsqueeze(0) + along.unsqueeze(2) * segment.unsqueeze(0)
    nearest_segment = (
        candidate_points.unsqueeze(1) - closest
    ).square().sum(dim=2).argmin(dim=1)
    row_ids = torch.arange(candidate_ids.numel(), dtype=torch.long)
    nearest_along = along[row_ids, nearest_segment]
    next_depth = torch.roll(polygon_depth, shifts=-1, dims=0)
    local_lip_depth = polygon_depth[nearest_segment] + nearest_along * (
        next_depth[nearest_segment] - polygon_depth[nearest_segment]
    )

    lip_depth_center = polygon_depth.median()
    tooth_depth_center = (
        vertices[tooth_mask, depth_axis].median()
        if bool(tooth_mask.any().item())
        else vertices[:, depth_axis].median()
    )
    front_direction = 1.0 if lip_depth_center >= tooth_depth_center else -1.0
    depth_limit = local_lip_depth - front_direction * (
        float(recession_ratio) * projected_span
    )
    current_depth = vertices[candidate_ids, depth_axis]
    too_far_forward = front_direction * (current_depth - depth_limit) > 0.0
    affected_ids = candidate_ids[too_far_forward]
    if affected_ids.numel() == 0:
        return vertices, 0, 0.0

    moved = vertices.clone()
    moved[affected_ids, depth_axis] = depth_limit[too_far_forward]
    max_displacement = float(
        (current_depth[too_far_forward] - depth_limit[too_far_forward])
        .abs()
        .amax()
        .item()
    )
    return moved.contiguous(), int(affected_ids.numel()), max_displacement


def _expand_recessed_oral_tissue_horizontally(
    vertices: torch.Tensor,
    colors: torch.Tensor,
    target_points: Mapping[int, torch.Tensor],
    head_vertices: torch.Tensor,
    *,
    horizontal_axis: int,
    depth_axis: int,
    vertical_axis: int,
    scale: float,
    start_depth_ratio: float,
    blend_depth_ratio: float,
) -> tuple[torch.Tensor, int, float]:
    """Widen the recessed oral wall without moving teeth, tongue, or front gum."""

    if scale < 1.0:
        raise ValueError("Recessed oral-shell scale must be at least one.")
    if start_depth_ratio < 0.0:
        raise ValueError("Recessed oral-shell start depth cannot be negative.")
    if blend_depth_ratio <= 0.0:
        raise ValueError("Recessed oral-shell blend depth must be positive.")
    if scale == 1.0:
        return vertices, 0, 0.0

    contour_ids = tuple(dict.fromkeys(OUTER_UPPER_LIP_IDS + OUTER_LOWER_LIP_IDS))
    missing = [landmark_id for landmark_id in contour_ids if landmark_id not in target_points]
    if missing:
        raise ValueError(
            "Recessed oral-shell expansion requires all outer-lip landmarks; missing "
            + ", ".join(str(value) for value in sorted(set(missing)))
        )
    contour = torch.stack(
        [target_points[landmark_id].float() for landmark_id in contour_ids]
    )
    projected = contour[:, (horizontal_axis, vertical_axis)]
    projected_span = float(
        (projected.amax(dim=0) - projected.amin(dim=0)).amax().item()
    )
    if projected_span <= 1.0e-8:
        return vertices, 0, 0.0

    tooth_rgb = colors.new_tensor((0.91, 0.86, 0.72))
    tongue_rgb = colors.new_tensor((0.58, 0.20, 0.23))
    tooth_mask = (colors - tooth_rgb).abs().amax(dim=1) < 1.0e-6
    tongue_mask = (colors - tongue_rgb).abs().amax(dim=1) < 1.0e-6
    tissue_mask = ~(tooth_mask | tongue_mask)

    lip_depth = contour[:, depth_axis].median()
    head_depth_min = head_vertices[:, depth_axis].amin()
    head_depth_max = head_vertices[:, depth_axis].amax()
    front_direction = (
        -1.0
        if lip_depth - head_depth_min < head_depth_max - lip_depth
        else 1.0
    )
    signed_front_depth = front_direction * (
        vertices[:, depth_axis] - lip_depth
    )
    recessed_depth = (-signed_front_depth).clamp_min(0.0)
    start_depth = float(start_depth_ratio) * projected_span
    blend_depth = float(blend_depth_ratio) * projected_span
    blend = ((recessed_depth - start_depth) / blend_depth).clamp(0.0, 1.0)
    blend *= tissue_mask.to(dtype=blend.dtype)
    affected = blend > 0.0
    if not bool(affected.any().item()):
        return vertices, 0, 0.0

    horizontal_center = 0.5 * (
        contour[:, horizontal_axis].amin() + contour[:, horizontal_axis].amax()
    )
    local_scale = 1.0 + (float(scale) - 1.0) * blend
    moved = vertices.clone()
    horizontal_offset = vertices[:, horizontal_axis] - horizontal_center
    moved[:, horizontal_axis] = horizontal_center + horizontal_offset * local_scale
    displacement = (moved[:, horizontal_axis] - vertices[:, horizontal_axis]).abs()
    return (
        moved.contiguous(),
        int(affected.sum().item()),
        float(displacement.amax().item()),
    )


def _trim_oral_rim_transition_faces(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    colors: torch.Tensor,
    target_points: Mapping[int, torch.Tensor],
    *,
    horizontal_axis: int,
    depth_axis: int,
    vertical_axis: int,
    maximum_distance_ratio: float,
    depth_tolerance_ratio: float,
) -> tuple[torch.Tensor, int]:
    """Remove exposed gum/root transition faces just outside the lip opening."""

    if maximum_distance_ratio < 0.0:
        raise ValueError("Oral rim-transition distance ratio cannot be negative.")
    if depth_tolerance_ratio < 0.0:
        raise ValueError("Oral rim-transition depth tolerance cannot be negative.")
    contour_ids = INNER_UPPER_LIP_IDS + tuple(
        reversed(INNER_LOWER_LIP_IDS[1:-1])
    )
    missing = [landmark_id for landmark_id in contour_ids if landmark_id not in target_points]
    if missing:
        raise ValueError(
            "Oral rim-transition trimming requires all inner-lip landmarks; missing "
            + ", ".join(str(value) for value in sorted(set(missing)))
        )

    axes = (int(horizontal_axis), int(vertical_axis))
    contour_points = torch.stack(
        [target_points[landmark_id].float() for landmark_id in contour_ids]
    )
    polygon = contour_points[:, axes]
    polygon_depth = contour_points[:, depth_axis]
    projected_span = float(
        (polygon.amax(dim=0) - polygon.amin(dim=0)).amax().item()
    )
    if projected_span <= 1.0e-8 or faces.numel() == 0:
        return faces, 0

    triangles = vertices[faces]
    centers = triangles.mean(dim=1)
    projected_centers = centers[:, axes]
    outside_opening = ~_points_inside_polygon(projected_centers, polygon)

    segment_start = polygon
    segment_end = torch.roll(polygon, shifts=-1, dims=0)
    segment = segment_end - segment_start
    relative = projected_centers.unsqueeze(1) - segment_start.unsqueeze(0)
    along = (
        (relative * segment.unsqueeze(0)).sum(dim=2)
        / segment.square().sum(dim=1).clamp_min(1.0e-12).unsqueeze(0)
    ).clamp(0.0, 1.0)
    closest = segment_start.unsqueeze(0) + along.unsqueeze(2) * segment.unsqueeze(0)
    distances = (projected_centers.unsqueeze(1) - closest).norm(dim=2)
    nearest_segment = distances.argmin(dim=1)
    rows = torch.arange(faces.shape[0], dtype=torch.long)
    nearest_distance = distances[rows, nearest_segment]
    nearest_along = along[rows, nearest_segment]
    next_depth = torch.roll(polygon_depth, shifts=-1, dims=0)
    local_lip_depth = polygon_depth[nearest_segment] + nearest_along * (
        next_depth[nearest_segment] - polygon_depth[nearest_segment]
    )

    tooth_rgb = colors.new_tensor((0.91, 0.86, 0.72))
    tissue_rgb = colors.new_tensor((0.34, 0.075, 0.085))
    tooth_mask = (colors - tooth_rgb).abs().amax(dim=1) < 1.0e-6
    tissue_mask = (colors - tissue_rgb).abs().amax(dim=1) < 1.0e-6
    transition_faces = tooth_mask[faces].any(dim=1) & (
        tissue_mask[faces].sum(dim=1) >= 2
    )

    lip_depth_center = polygon_depth.median()
    tooth_depth_center = (
        vertices[tooth_mask, depth_axis].median()
        if bool(tooth_mask.any().item())
        else vertices[:, depth_axis].median()
    )
    front_direction = 1.0 if lip_depth_center >= tooth_depth_center else -1.0
    face_front_depth = (
        triangles[:, :, depth_axis].amax(dim=1)
        if front_direction > 0.0
        else triangles[:, :, depth_axis].amin(dim=1)
    )
    crosses_lip_depth = front_direction * (
        face_front_depth - local_lip_depth
    ) >= -float(depth_tolerance_ratio) * projected_span
    remove = (
        outside_opening
        & transition_faces
        & crosses_lip_depth
        & (
            nearest_distance
            <= float(maximum_distance_ratio) * projected_span
        )
    )
    return faces[~remove].contiguous(), int(remove.sum().item())


def _faces_overlapping_polygon_2d(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    polygon: torch.Tensor,
    *,
    horizontal_axis: int,
    vertical_axis: int,
) -> torch.Tensor:
    triangles = vertices[faces]
    centers = triangles.mean(dim=1)
    center_inside = _points_inside_polygon(
        centers[:, (horizontal_axis, vertical_axis)],
        polygon,
    )
    edge_midpoints = 0.5 * (triangles + triangles.roll(shifts=-1, dims=1))
    samples = torch.cat((triangles, edge_midpoints), dim=1)
    sample_inside = _points_inside_polygon(
        samples[..., (horizontal_axis, vertical_axis)].reshape(-1, 2),
        polygon,
    ).reshape(faces.shape[0], -1)

    triangle_2d = triangles[..., (horizontal_axis, vertical_axis)]
    origin = triangle_2d[:, :1, :]
    triangle_u = triangle_2d[:, 1:2, :] - origin
    triangle_v = triangle_2d[:, 2:3, :] - origin
    polygon_relative = polygon.unsqueeze(0) - origin
    dot_uu = (triangle_u * triangle_u).sum(dim=2)
    dot_uv = (triangle_u * triangle_v).sum(dim=2)
    dot_vv = (triangle_v * triangle_v).sum(dim=2)
    dot_up = (polygon_relative * triangle_u).sum(dim=2)
    dot_vp = (polygon_relative * triangle_v).sum(dim=2)
    denominator = dot_uu * dot_vv - dot_uv.square()
    valid = denominator.abs() > 1.0e-12
    safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
    barycentric_u = (dot_vv * dot_up - dot_uv * dot_vp) / safe_denominator
    barycentric_v = (dot_uu * dot_vp - dot_uv * dot_up) / safe_denominator
    polygon_inside_triangle = (
        valid
        & (barycentric_u >= -1.0e-6)
        & (barycentric_v >= -1.0e-6)
        & (barycentric_u + barycentric_v <= 1.0 + 1.0e-6)
    ).any(dim=1)
    return (
        center_inside | sample_inside.any(dim=1) | polygon_inside_triangle
    ).contiguous()


def remove_head_faces_behind_outer_lips(
    deformed_vertices: torch.Tensor,
    faces: torch.Tensor,
    mapping: Mapping[int, int],
    *,
    horizontal_axis: int = 0,
    vertical_axis: int = 2,
    depth_axis: int = 1,
    scale: float = 0.90,
    depth_padding_ratio: float = 0.15,
    maximum_depth: float | None = None,
    boundary_distance_ratio: float = 0.125,
    boundary_blend_rings: int = 2,
    maximum_protected_edge_ratio: float = 0.25,
    protect_lower_lip: bool = True,
    lower_lip_depth_tolerance_ratio: float = 0.12,
) -> tuple[torch.Tensor, OralHeadApertureReport]:
    """Clear source head surfaces from an inserted oral assembly's aperture."""

    if not 0.0 < scale <= 1.0:
        raise ValueError("Oral head-aperture scale must be in (0, 1].")
    if depth_padding_ratio < 0.0:
        raise ValueError("Oral head-aperture depth padding ratio cannot be negative.")
    if maximum_depth is not None and not np.isfinite(maximum_depth):
        raise ValueError("Oral head-aperture maximum depth must be finite.")
    if boundary_distance_ratio < 0.0:
        raise ValueError("Oral lip-boundary distance ratio cannot be negative.")
    if boundary_blend_rings < 0:
        raise ValueError("Oral lip-boundary blend rings cannot be negative.")
    if maximum_protected_edge_ratio <= 0.0:
        raise ValueError("Maximum protected lip-edge ratio must be positive.")
    if lower_lip_depth_tolerance_ratio < 0.0:
        raise ValueError("Lower-lip depth tolerance ratio cannot be negative.")
    target_points = _target_landmark_points(deformed_vertices, mapping)
    polygon = _outer_lip_polygon(
        target_points,
        horizontal_axis=horizontal_axis,
        vertical_axis=vertical_axis,
        scale=scale,
    )
    cpu_vertices = deformed_vertices.detach().cpu().float()
    cpu_faces = faces.detach().cpu().long()
    overlap = _faces_overlapping_polygon_2d(
        cpu_vertices,
        cpu_faces,
        polygon,
        horizontal_axis=horizontal_axis,
        vertical_axis=vertical_axis,
    )
    all_edges = torch.sort(
        torch.cat(
            (
                cpu_faces[:, (0, 1)],
                cpu_faces[:, (1, 2)],
                cpu_faces[:, (2, 0)],
            ),
            dim=0,
        ),
        dim=1,
    ).values
    unique_edges, edge_counts = torch.unique(
        all_edges,
        dim=0,
        return_counts=True,
    )
    boundary_vertex_ids = torch.unique(unique_edges[edge_counts == 1])
    projected_span = (polygon.amax(dim=0) - polygon.amin(dim=0)).amax()
    triangles = cpu_vertices[cpu_faces]
    face_centers_2d = triangles.mean(dim=1)[:, (horizontal_axis, vertical_axis)]
    center_inside = _points_inside_polygon(face_centers_2d, polygon)
    lip_depth_ids = OUTER_UPPER_LIP_IDS + OUTER_LOWER_LIP_IDS
    lip_depths = torch.stack(
        [target_points[landmark_id][depth_axis] for landmark_id in lip_depth_ids]
    ).detach().cpu().float()
    depth_padding = float(depth_padding_ratio) * projected_span
    depth_min = lip_depths.amin() - depth_padding
    depth_max = lip_depths.amax() + depth_padding
    if maximum_depth is not None:
        depth_max = torch.maximum(
            depth_max, depth_max.new_tensor(float(maximum_depth))
        )
    face_depth = triangles[:, :, depth_axis].mean(dim=1)
    front_depth = (face_depth >= depth_min) & (face_depth <= depth_max)
    maximum_edge = torch.stack(
        (
            (triangles[:, 1] - triangles[:, 0]).norm(dim=1),
            (triangles[:, 2] - triangles[:, 1]).norm(dim=1),
            (triangles[:, 0] - triangles[:, 2]).norm(dim=1),
        ),
        dim=1,
    ).amax(dim=1)
    locally_sized = maximum_edge <= (
        float(maximum_protected_edge_ratio) * projected_span
    )
    protected = torch.zeros(cpu_faces.shape[0], dtype=torch.bool)
    if boundary_vertex_ids.numel() > 0:
        axes = (int(horizontal_axis), int(vertical_axis))
        boundary_points = cpu_vertices[boundary_vertex_ids][:, axes]
        segment_start = polygon
        segment_end = torch.roll(polygon, shifts=-1, dims=0)
        segment = segment_end - segment_start
        relative = boundary_points.unsqueeze(1) - segment_start.unsqueeze(0)
        along = (
            (relative * segment.unsqueeze(0)).sum(dim=2)
            / segment.square().sum(dim=1).clamp_min(1.0e-12).unsqueeze(0)
        ).clamp(0.0, 1.0)
        closest = segment_start.unsqueeze(0) + along.unsqueeze(2) * segment.unsqueeze(0)
        distances = (
            boundary_points.unsqueeze(1) - closest
        ).norm(dim=2).amin(dim=1)
        lip_boundary_ids = boundary_vertex_ids[
            distances <= float(boundary_distance_ratio) * projected_span
        ]
        protected_vertices = torch.zeros(cpu_vertices.shape[0], dtype=torch.bool)
        protected_vertices[lip_boundary_ids] = True
        direct_boundary_faces = protected_vertices[cpu_faces].any(dim=1)
        expanded_faces = direct_boundary_faces.clone()
        frontier = protected_vertices.clone()
        visited = protected_vertices.clone()
        for _ring in range(boundary_blend_rings):
            touching_faces = frontier[cpu_faces].any(dim=1)
            expanded_faces |= touching_faces
            neighbors = torch.zeros_like(frontier)
            if bool(touching_faces.any().item()):
                neighbors[cpu_faces[touching_faces].reshape(-1)] = True
            frontier = neighbors & ~visited
            visited |= neighbors
        protected = expanded_faces & locally_sized
    segment_start = polygon
    segment_end = torch.roll(polygon, shifts=-1, dims=0)
    segment = segment_end - segment_start
    center_relative = face_centers_2d.unsqueeze(1) - segment_start.unsqueeze(0)
    center_along = (
        (center_relative * segment.unsqueeze(0)).sum(dim=2)
        / segment.square().sum(dim=1).clamp_min(1.0e-12).unsqueeze(0)
    ).clamp(0.0, 1.0)
    center_closest = (
        segment_start.unsqueeze(0)
        + center_along.unsqueeze(2) * segment.unsqueeze(0)
    )
    center_boundary_distance = (
        face_centers_2d.unsqueeze(1) - center_closest
    ).norm(dim=2).amin(dim=1)
    contour_protected = (
        center_boundary_distance
        <= float(boundary_distance_ratio) * projected_span
    ) & locally_sized
    protected |= contour_protected
    lower_lip_protected = torch.zeros_like(protected)
    if protect_lower_lip:
        lower_lip_polygon = _lower_lip_ribbon_polygon(
            target_points,
            horizontal_axis=horizontal_axis,
            vertical_axis=vertical_axis,
        )
        lower_lip_overlap = _faces_overlapping_polygon_2d(
            cpu_vertices,
            cpu_faces,
            lower_lip_polygon,
            horizontal_axis=horizontal_axis,
            vertical_axis=vertical_axis,
        )
        lower_lip_ids = INNER_LOWER_LIP_IDS + OUTER_LOWER_LIP_IDS
        lower_lip_depths = torch.stack(
            [target_points[landmark_id][depth_axis] for landmark_id in lower_lip_ids]
        ).detach().cpu().float()
        face_depth = triangles[:, :, depth_axis].mean(dim=1)
        near_lower_lip_depth = (
            face_depth.unsqueeze(1) - lower_lip_depths.unsqueeze(0)
        ).abs().amin(dim=1) <= (
            float(lower_lip_depth_tolerance_ratio) * projected_span
        )
        lower_lip_protected = (
            lower_lip_overlap & locally_sized & near_lower_lip_depth
        )
        protected |= lower_lip_protected
    protected &= front_depth
    # Boundary triangles commonly overlap the polygon while remaining visible
    # lip. Clear center-inside faces and oversized crossing flaps, but retain
    # locally sized faces close to the contour.
    removal_candidate = center_inside | (overlap & ~locally_sized)
    removed = removal_candidate & front_depth & ~protected
    kept_faces = faces[~removed].contiguous()
    report = OralHeadApertureReport(
        scale=float(scale),
        depth_padding_ratio=float(depth_padding_ratio),
        depth_min=float(depth_min.item()),
        depth_max=float(depth_max.item()),
        boundary_distance_ratio=float(boundary_distance_ratio),
        boundary_blend_rings=int(boundary_blend_rings),
        maximum_protected_edge_ratio=float(maximum_protected_edge_ratio),
        lower_lip_depth_tolerance_ratio=float(lower_lip_depth_tolerance_ratio),
        protected_face_count=int(protected.sum().item()),
        contour_protected_face_count=int(contour_protected.sum().item()),
        lower_lip_protected_face_count=int(lower_lip_protected.sum().item()),
        polygon_point_count=int(polygon.shape[0]),
        projected_overlap_face_count=int(overlap.sum().item()),
        projected_center_inside_face_count=int(center_inside.sum().item()),
        depth_rejected_overlap_face_count=int(
            (overlap & ~front_depth).sum().item()
        ),
        removed_face_count=int(removed.sum().item()),
        original_face_count=int(faces.shape[0]),
        final_face_count=int(kept_faces.shape[0]),
        aperture_polygon=polygon.tolist(),
    )
    return kept_faces, report


def remove_disconnected_head_mouth_components(
    deformed_vertices: torch.Tensor,
    faces: torch.Tensor,
    aperture_polygon: torch.Tensor,
    *,
    mapping: Mapping[int, int] | None = None,
    depth_axis: int = 1,
    depth_padding_ratio: float = 0.15,
    maximum_depth: float | None = None,
    horizontal_axis: int = 0,
    vertical_axis: int = 2,
    margin_ratio: float = 0.25,
    maximum_component_faces: int = 5000,
    protect_lip_landmark_components: bool = True,
    maximum_boundary_edge_ratio: float = 0.25,
) -> tuple[torch.Tensor, OralHeadComponentReport]:
    """Remove bounded source-mouth islands after the main aperture is cleared."""

    if margin_ratio < 0.0:
        raise ValueError("Oral head-component margin ratio cannot be negative.")
    if depth_padding_ratio < 0.0:
        raise ValueError("Oral component depth padding ratio cannot be negative.")
    if maximum_depth is not None and not np.isfinite(maximum_depth):
        raise ValueError("Oral component maximum depth must be finite.")
    if maximum_component_faces <= 0:
        raise ValueError("Maximum oral head-component face count must be positive.")
    if maximum_boundary_edge_ratio <= 0.0:
        raise ValueError("Maximum oral boundary-edge ratio must be positive.")
    depth_guard_enabled = mapping is not None
    depth_min: float | None = None
    depth_max: float | None = None
    if mapping is not None:
        target_points = _target_landmark_points(deformed_vertices, mapping)
        lip_depth_ids = OUTER_UPPER_LIP_IDS + OUTER_LOWER_LIP_IDS
        missing_depth_ids = [
            landmark_id
            for landmark_id in lip_depth_ids
            if landmark_id not in target_points
        ]
        if missing_depth_ids:
            raise ValueError(
                "Oral component depth protection requires all outer-lip landmarks; "
                "missing "
                + ", ".join(str(value) for value in sorted(set(missing_depth_ids)))
            )
        lip_depths = torch.stack(
            [target_points[landmark_id][depth_axis] for landmark_id in lip_depth_ids]
        ).detach().cpu().float()
        projected_span = (
            aperture_polygon.detach().cpu().float().amax(dim=0)
            - aperture_polygon.detach().cpu().float().amin(dim=0)
        ).amax()
        depth_padding = float(depth_padding_ratio) * float(projected_span.item())
        depth_min = float(lip_depths.amin().item()) - depth_padding
        depth_max = float(lip_depths.amax().item()) + depth_padding
        if maximum_depth is not None:
            depth_max = max(depth_max, float(maximum_depth))
    cpu_faces = faces.detach().cpu().long().numpy()
    cpu_vertices = deformed_vertices.detach().cpu().float().numpy()
    adjacency = trimesh.graph.face_adjacency(faces=cpu_faces)
    components = trimesh.graph.connected_components(
        adjacency,
        nodes=np.arange(cpu_faces.shape[0]),
    )
    if not components:
        report = OralHeadComponentReport(
            margin_ratio=float(margin_ratio),
            maximum_component_faces=int(maximum_component_faces),
            depth_guard_enabled=depth_guard_enabled,
            depth_padding_ratio=float(depth_padding_ratio),
            depth_min=depth_min,
            depth_max=depth_max,
            depth_rejected_component_count=0,
            lip_landmark_protected_component_count=0,
            lip_landmark_protected_face_count=0,
            boundary_protected_component_count=0,
            boundary_protected_face_count=0,
            removed_component_count=0,
            removed_face_count=0,
            original_face_count=int(faces.shape[0]),
            final_face_count=int(faces.shape[0]),
        )
        return faces, report

    main_component_index = int(np.argmax([len(component) for component in components]))
    polygon = aperture_polygon.detach().cpu().float().numpy()
    projected_min = polygon.min(axis=0)
    projected_max = polygon.max(axis=0)
    projected_span = float(np.max(projected_max - projected_min))
    margin = (projected_max - projected_min) * float(margin_ratio)
    projected_min -= margin
    projected_max += margin
    remove_face_ids: list[int] = []
    removed_component_count = 0
    depth_rejected_component_count = 0
    lip_landmark_protected_component_count = 0
    lip_landmark_protected_face_count = 0
    boundary_protected_component_count = 0
    boundary_protected_face_count = 0
    lip_landmark_vertex_ids: set[int] = set()
    if mapping is not None and protect_lip_landmark_components:
        lip_landmark_vertex_ids = {
            int(vertex_id)
            for landmark_id, vertex_id in mapping.items()
            if int(landmark_id) in MOUTH_LANDMARK_IDS
            and 0 <= int(vertex_id) < deformed_vertices.shape[0]
        }
    for component_index, component in enumerate(components):
        component = np.asarray(component, dtype=np.int64)
        if component_index == main_component_index:
            continue
        if len(component) > int(maximum_component_faces):
            continue
        vertex_ids = np.unique(cpu_faces[component].reshape(-1))
        if lip_landmark_vertex_ids.intersection(int(value) for value in vertex_ids):
            lip_landmark_protected_component_count += 1
            lip_landmark_protected_face_count += int(len(component))
            continue
        points = cpu_vertices[vertex_ids]
        component_min = points[:, (horizontal_axis, vertical_axis)].min(axis=0)
        component_max = points[:, (horizontal_axis, vertical_axis)].max(axis=0)
        inside_projected_bounds = bool(
            np.all(component_min >= projected_min)
            and np.all(component_max <= projected_max)
        )
        if not inside_projected_bounds:
            continue
        projected_points = points[:, (horizontal_axis, vertical_axis)]
        points_inside_aperture = _points_inside_polygon(
            torch.from_numpy(projected_points).float(),
            aperture_polygon.detach().cpu().float(),
        ).numpy()
        component_triangles = cpu_vertices[cpu_faces[component]]
        component_edge_lengths = np.linalg.norm(
            component_triangles - np.roll(component_triangles, shift=-1, axis=1),
            axis=2,
        )
        maximum_component_edge = float(component_edge_lengths.max(initial=0.0))
        if (
            not bool(np.all(points_inside_aperture))
            and maximum_component_edge
            <= float(maximum_boundary_edge_ratio) * projected_span
        ):
            boundary_protected_component_count += 1
            boundary_protected_face_count += int(len(component))
            continue
        if depth_guard_enabled:
            assert depth_min is not None and depth_max is not None
            component_depth = points[:, depth_axis]
            if component_depth.min() < depth_min or component_depth.max() > depth_max:
                depth_rejected_component_count += 1
                continue
        remove_face_ids.extend(int(value) for value in component.tolist())
        removed_component_count += 1

    keep = torch.ones(faces.shape[0], dtype=torch.bool)
    if remove_face_ids:
        keep[torch.tensor(remove_face_ids, dtype=torch.long)] = False
    kept_faces = faces[keep].contiguous()
    report = OralHeadComponentReport(
        margin_ratio=float(margin_ratio),
        depth_guard_enabled=depth_guard_enabled,
        depth_padding_ratio=float(depth_padding_ratio),
        depth_min=depth_min,
        depth_max=depth_max,
        depth_rejected_component_count=depth_rejected_component_count,
        lip_landmark_protected_component_count=(
            lip_landmark_protected_component_count
        ),
        lip_landmark_protected_face_count=lip_landmark_protected_face_count,
        boundary_protected_component_count=boundary_protected_component_count,
        boundary_protected_face_count=boundary_protected_face_count,
        maximum_component_faces=int(maximum_component_faces),
        removed_component_count=removed_component_count,
        removed_face_count=len(remove_face_ids),
        original_face_count=int(faces.shape[0]),
        final_face_count=int(kept_faces.shape[0]),
    )
    return kept_faces, report


def fit_oral_assembly(
    asset_path: Path,
    neutral_vertices: torch.Tensor,
    deformed_vertices: torch.Tensor,
    mapping: Mapping[int, int],
    *,
    horizontal_axis: int = 0,
    depth_axis: int = 1,
    vertical_axis: int = 2,
    assembly_scale: float = 1.0,
    depth_offset: float = 0.0,
    level_target_frame: bool = False,
    aperture_crop_scale: float | None = None,
    auto_center_teeth: bool = False,
    maximum_center_offset_ratio: float = 0.25,
    center_teeth_vertically: bool = False,
    tooth_depth_offset: float = 0.0,
    tooth_depth_blend_rings: int = 2,
    auto_center_lower_teeth: bool = False,
    lower_teeth_depth_offset: float = 0.0,
    lower_teeth_blend_rings: int = 2,
    contain_inside_lips: bool = False,
    containment_scale: float = 0.96,
    containment_depth_inset: float = 0.025,
    containment_blend_rings: int = 2,
    rim_recession_ratio: float = 0.08,
    recessed_shell_horizontal_scale: float = 1.0,
    recessed_shell_start_depth_ratio: float = 0.06,
    recessed_shell_blend_depth_ratio: float = 0.28,
    trim_rim_transition_faces: bool = True,
    rim_transition_maximum_distance_ratio: float = 0.08,
    rim_transition_depth_tolerance_ratio: float = 0.01,
    upper_teeth_vertical_offset: float = 0.0,
    upper_teeth_vertical_blend_rings: int = 2,
    upper_crown_quantile: float = 0.50,
    remove_oral_wall_uv_faces: bool = False,
    remove_tongue_uv_island: bool = False,
    tongue_planar_scale: float = 1.0,
    tongue_depth_offset: float = 0.0,
    tongue_vertical_offset: float = 0.0,
    tongue_offset_blend_rings: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MetahumanOralAssemblyReport]:
    if assembly_scale <= 0.0:
        raise ValueError("Oral assembly scale must be positive.")
    if maximum_center_offset_ratio < 0.0:
        raise ValueError("Maximum oral center offset ratio cannot be negative.")
    if tooth_depth_blend_rings < 0:
        raise ValueError("Tooth depth blend rings cannot be negative.")
    if lower_teeth_blend_rings < 0:
        raise ValueError("Lower-tooth blend rings cannot be negative.")
    if not 0.0 < containment_scale <= 1.0:
        raise ValueError("Oral-containment scale must be in (0, 1].")
    if containment_depth_inset < 0.0:
        raise ValueError("Oral-containment depth inset cannot be negative.")
    if containment_blend_rings < 0:
        raise ValueError("Oral-containment blend rings cannot be negative.")
    if rim_recession_ratio < 0.0:
        raise ValueError("Oral rim-recession ratio cannot be negative.")
    if recessed_shell_horizontal_scale < 1.0:
        raise ValueError("Recessed oral-shell scale must be at least one.")
    if recessed_shell_start_depth_ratio < 0.0:
        raise ValueError("Recessed oral-shell start depth cannot be negative.")
    if recessed_shell_blend_depth_ratio <= 0.0:
        raise ValueError("Recessed oral-shell blend depth must be positive.")
    if rim_transition_maximum_distance_ratio < 0.0:
        raise ValueError("Oral rim-transition distance ratio cannot be negative.")
    if rim_transition_depth_tolerance_ratio < 0.0:
        raise ValueError("Oral rim-transition depth tolerance cannot be negative.")
    if upper_teeth_vertical_blend_rings < 0:
        raise ValueError("Upper-tooth vertical blend rings cannot be negative.")
    if not 0.0 <= upper_crown_quantile <= 1.0:
        raise ValueError("Upper-crown quantile must be in [0, 1].")
    if tongue_planar_scale <= 0.0:
        raise ValueError("Tongue planar scale must be positive.")
    if tongue_offset_blend_rings < 0:
        raise ValueError("Tongue offset blend rings cannot be negative.")
    axes = (int(horizontal_axis), int(depth_axis), int(vertical_axis))
    if sorted(axes) != [0, 1, 2]:
        raise ValueError("Oral assembly axes must be distinct indices in [0, 2].")

    resolved_asset_path = asset_path.expanduser().resolve()
    with np.load(resolved_asset_path, allow_pickle=False) as data:
        required_keys = {
            "teeth_basis_vertices",
            "teeth_jaw_open_vertices",
            "teeth_faces",
            "reference_landmark_ids",
            "reference_landmark_positions",
            "reference_landmark_jaw_open_positions",
        }
        missing = sorted(required_keys - set(data.files))
        if missing:
            raise ValueError(
                f"Oral asset is missing arrays: {', '.join(missing)}"
            )
        basis_vertices = torch.from_numpy(data["teeth_basis_vertices"].copy()).float()
        jaw_open_vertices = torch.from_numpy(data["teeth_jaw_open_vertices"].copy()).float()
        faces = torch.from_numpy(data["teeth_faces"].copy()).long()
        landmark_ids = torch.from_numpy(data["reference_landmark_ids"].copy()).long()
        reference_positions = torch.from_numpy(
            data["reference_landmark_positions"].copy()
        ).float()
        reference_jaw_positions = torch.from_numpy(
            data["reference_landmark_jaw_open_positions"].copy()
        ).float()
        metadata = {}
        face_uvs = None
        explicit_colors = None
        explicit_dental_arch_mask = None
        explicit_tongue_mask = None
        explicit_moving_mask = None
        if "teeth_face_uvs" in data.files:
            face_uvs = torch.from_numpy(data["teeth_face_uvs"].copy()).float()
        if "oral_vertex_colors" in data.files:
            explicit_colors = torch.from_numpy(data["oral_vertex_colors"].copy()).float()
        if "dental_arch_vertex_mask" in data.files:
            explicit_dental_arch_mask = torch.from_numpy(
                data["dental_arch_vertex_mask"].copy()
            ).bool()
        if "tongue_vertex_mask" in data.files:
            explicit_tongue_mask = torch.from_numpy(
                data["tongue_vertex_mask"].copy()
            ).bool()
        if "moving_vertex_mask" in data.files:
            explicit_moving_mask = torch.from_numpy(
                data["moving_vertex_mask"].copy()
            ).bool()
        if "metadata_json" in data.files:
            metadata = json.loads(str(data["metadata_json"].item()))

    vertex_count = int(basis_vertices.shape[0])
    for name, mask in (
        ("dental_arch_vertex_mask", explicit_dental_arch_mask),
        ("tongue_vertex_mask", explicit_tongue_mask),
        ("moving_vertex_mask", explicit_moving_mask),
    ):
        if mask is not None and mask.shape != (vertex_count,):
            raise ValueError(
                f"Oral asset {name} must have shape [{vertex_count}], got {list(mask.shape)}."
            )
    if explicit_colors is not None and explicit_colors.shape != (vertex_count, 3):
        raise ValueError(
            "Oral asset oral_vertex_colors must have shape "
            f"[{vertex_count}, 3], got {list(explicit_colors.shape)}."
        )

    source_neutral_points = _points_by_landmark(
        landmark_ids,
        _model_axes(reference_positions),
    )
    source_jaw_points = _points_by_landmark(
        landmark_ids,
        _model_axes(reference_jaw_positions),
    )
    target_neutral_points = _target_landmark_points(neutral_vertices, mapping)
    target_jaw_points = _target_landmark_points(deformed_vertices, mapping)
    frame_args = {
        "horizontal_axis": int(horizontal_axis),
        "depth_axis": int(depth_axis),
        "vertical_axis": int(vertical_axis),
    }
    source_neutral_frame = _mouth_frame(source_neutral_points, **frame_args)
    source_jaw_frame = _mouth_frame(source_jaw_points, **frame_args)
    target_neutral_frame = _mouth_frame(
        target_neutral_points,
        level_horizontal=level_target_frame,
        **frame_args,
    )
    target_jaw_frame = _mouth_frame(
        target_jaw_points,
        level_horizontal=level_target_frame,
        **frame_args,
    )

    neutral_oral, neutral_scales = _fit_to_frame(
        _model_axes(basis_vertices),
        source_neutral_frame,
        target_neutral_frame,
        assembly_scale=assembly_scale,
        depth_offset=depth_offset,
        **frame_args,
    )
    deformed_oral, jaw_scales = _fit_to_frame(
        _model_axes(jaw_open_vertices),
        source_jaw_frame,
        target_jaw_frame,
        assembly_scale=assembly_scale,
        depth_offset=depth_offset,
        **frame_args,
    )
    source_vertex_count = int(basis_vertices.shape[0])
    source_face_count = int(faces.shape[0])
    removed_tongue_uv_face_count = 0
    if remove_tongue_uv_island:
        if face_uvs is None:
            raise ValueError(
                "Removing the tongue UV island requires teeth_face_uvs in the oral asset."
            )
        faces, face_uvs, removed_tongue_uv_face_count = _remove_tongue_uv_faces(
            faces,
            face_uvs,
        )

    feature_faces = faces
    feature_face_uvs = face_uvs
    source_oral_wall_face_count = 0
    if face_uvs is not None:
        feature_faces, feature_face_uvs, source_oral_wall_face_count = (
            _remove_oral_wall_uv_faces(faces, face_uvs)
        )
    elif remove_oral_wall_uv_faces:
        raise ValueError(
            "Removing oral wall UV faces requires teeth_face_uvs in the oral asset."
        )

    removed_oral_wall_uv_face_count = 0
    geometry_mode = "full_shell"
    if remove_oral_wall_uv_faces:
        faces = feature_faces
        face_uvs = feature_face_uvs
        removed_oral_wall_uv_face_count = source_oral_wall_face_count
        geometry_mode = "dental_only"

    dental_arch_mask = explicit_dental_arch_mask
    if dental_arch_mask is None and feature_face_uvs is not None:
        dental_arch_mask = _dental_arch_vertex_mask(
            feature_faces,
            feature_face_uvs,
            vertex_count,
        )
    tongue_uv_mask = explicit_tongue_mask
    if tongue_uv_mask is None and feature_face_uvs is not None:
        tongue_uv_mask = _tongue_uv_vertex_mask(
            feature_faces,
            feature_face_uvs,
            vertex_count,
        )
    if explicit_colors is not None:
        colors = explicit_colors.contiguous()
    else:
        colors, _tooth_count, _tongue_count, _tissue_count = _oral_vertex_colors(
            basis_vertices,
            jaw_open_vertices,
            dental_arch_mask,
            tongue_uv_mask,
            feature_faces,
            upper_crown_quantile=upper_crown_quantile,
        )
    if explicit_moving_mask is not None:
        moving_mask = explicit_moving_mask
    else:
        source_movement = (jaw_open_vertices - basis_vertices).norm(dim=1)
        moving_mask = source_movement > max(
            1.0e-5,
            float(source_movement.max().item()) * 0.02,
        )
    neutral_center_offsets = (0.0, 0.0)
    jaw_center_offsets = (0.0, 0.0)
    if auto_center_teeth:
        neutral_oral, neutral_center_offsets = _center_oral_teeth_in_frame(
            neutral_oral,
            colors,
            target_neutral_frame,
            horizontal_axis=horizontal_axis,
            vertical_axis=vertical_axis,
            maximum_offset_ratio=maximum_center_offset_ratio,
            center_vertically=center_teeth_vertically,
        )
        deformed_oral, jaw_center_offsets = _center_oral_teeth_in_frame(
            deformed_oral,
            colors,
            target_jaw_frame,
            horizontal_axis=horizontal_axis,
            vertical_axis=vertical_axis,
            maximum_offset_ratio=maximum_center_offset_ratio,
            center_vertically=center_teeth_vertically,
        )
    neutral_oral, neutral_tooth_depth_count = _offset_teeth_in_depth(
        neutral_oral,
        feature_faces,
        colors,
        depth_axis=depth_axis,
        offset=tooth_depth_offset,
        blend_rings=tooth_depth_blend_rings,
    )
    deformed_oral, jaw_tooth_depth_count = _offset_teeth_in_depth(
        deformed_oral,
        feature_faces,
        colors,
        depth_axis=depth_axis,
        offset=tooth_depth_offset,
        blend_rings=tooth_depth_blend_rings,
    )
    neutral_lower_center_offset = 0.0
    jaw_lower_center_offset = 0.0
    neutral_lower_count = 0
    jaw_lower_count = 0
    if auto_center_lower_teeth or lower_teeth_depth_offset != 0.0:
        (
            neutral_oral,
            neutral_lower_center_offset,
            neutral_lower_count,
        ) = _adjust_lower_dental_region(
            neutral_oral,
            feature_faces,
            colors,
            moving_mask,
            target_neutral_frame,
            horizontal_axis=horizontal_axis,
            vertical_axis=vertical_axis,
            depth_axis=depth_axis,
            auto_center=auto_center_lower_teeth,
            depth_offset=lower_teeth_depth_offset,
            blend_rings=lower_teeth_blend_rings,
            maximum_center_offset_ratio=maximum_center_offset_ratio,
        )
        (
            deformed_oral,
            jaw_lower_center_offset,
            jaw_lower_count,
        ) = _adjust_lower_dental_region(
            deformed_oral,
            feature_faces,
            colors,
            moving_mask,
            target_jaw_frame,
            horizontal_axis=horizontal_axis,
            vertical_axis=vertical_axis,
            depth_axis=depth_axis,
            auto_center=auto_center_lower_teeth,
            depth_offset=lower_teeth_depth_offset,
            blend_rings=lower_teeth_blend_rings,
            maximum_center_offset_ratio=maximum_center_offset_ratio,
        )
    neutral_oral, neutral_upper_vertical_count = _offset_upper_teeth_vertically(
        neutral_oral,
        feature_faces,
        colors,
        moving_mask,
        vertical_axis=vertical_axis,
        offset=upper_teeth_vertical_offset,
        blend_rings=upper_teeth_vertical_blend_rings,
    )
    deformed_oral, jaw_upper_vertical_count = _offset_upper_teeth_vertically(
        deformed_oral,
        feature_faces,
        colors,
        moving_mask,
        vertical_axis=vertical_axis,
        offset=upper_teeth_vertical_offset,
        blend_rings=upper_teeth_vertical_blend_rings,
    )
    neutral_tongue_offset_count = 0
    jaw_tongue_offset_count = 0
    if tongue_uv_mask is not None:
        neutral_oral, neutral_tongue_offset_count = _offset_tongue(
            neutral_oral,
            feature_faces,
            tongue_uv_mask,
            horizontal_axis=horizontal_axis,
            depth_axis=depth_axis,
            vertical_axis=vertical_axis,
            planar_scale=tongue_planar_scale,
            depth_offset=tongue_depth_offset,
            vertical_offset=tongue_vertical_offset,
            blend_rings=tongue_offset_blend_rings,
        )
        deformed_oral, jaw_tongue_offset_count = _offset_tongue(
            deformed_oral,
            feature_faces,
            tongue_uv_mask,
            horizontal_axis=horizontal_axis,
            depth_axis=depth_axis,
            vertical_axis=vertical_axis,
            planar_scale=tongue_planar_scale,
            depth_offset=tongue_depth_offset,
            vertical_offset=tongue_vertical_offset,
            blend_rings=tongue_offset_blend_rings,
        )
    neutral_containment_count = 0
    jaw_containment_count = 0
    neutral_containment_direct_count = 0
    jaw_containment_direct_count = 0
    neutral_containment_max_displacement = 0.0
    jaw_containment_max_displacement = 0.0
    if contain_inside_lips:
        tooth_rgb = colors.new_tensor((0.91, 0.86, 0.72))
        containment_movable_mask = (
            (colors - tooth_rgb).abs().amax(dim=1) >= 1.0e-6
        )
        neutral_polygon = _outer_lip_polygon(
            target_neutral_points,
            horizontal_axis=horizontal_axis,
            vertical_axis=vertical_axis,
            scale=containment_scale,
        )
        jaw_polygon = _outer_lip_polygon(
            target_jaw_points,
            horizontal_axis=horizontal_axis,
            vertical_axis=vertical_axis,
            scale=containment_scale,
        )
        (
            neutral_oral,
            neutral_containment_count,
            neutral_containment_direct_count,
            neutral_containment_max_displacement,
        ) = _contain_oral_vertices_in_polygon(
            neutral_oral,
            feature_faces,
            neutral_polygon,
            horizontal_axis=horizontal_axis,
            depth_axis=depth_axis,
            vertical_axis=vertical_axis,
            depth_inset=containment_depth_inset,
            blend_rings=containment_blend_rings,
            movable_mask=containment_movable_mask,
        )
        (
            deformed_oral,
            jaw_containment_count,
            jaw_containment_direct_count,
            jaw_containment_max_displacement,
        ) = _contain_oral_vertices_in_polygon(
            deformed_oral,
            feature_faces,
            jaw_polygon,
            horizontal_axis=horizontal_axis,
            depth_axis=depth_axis,
            vertical_axis=vertical_axis,
            depth_inset=containment_depth_inset,
            blend_rings=containment_blend_rings,
            movable_mask=containment_movable_mask,
        )
    neutral_shell_expansion_count = 0
    jaw_shell_expansion_count = 0
    neutral_shell_expansion_max_displacement = 0.0
    jaw_shell_expansion_max_displacement = 0.0
    if recessed_shell_horizontal_scale > 1.0:
        (
            neutral_oral,
            neutral_shell_expansion_count,
            neutral_shell_expansion_max_displacement,
        ) = _expand_recessed_oral_tissue_horizontally(
            neutral_oral,
            colors,
            target_neutral_points,
            neutral_vertices,
            horizontal_axis=horizontal_axis,
            depth_axis=depth_axis,
            vertical_axis=vertical_axis,
            scale=recessed_shell_horizontal_scale,
            start_depth_ratio=recessed_shell_start_depth_ratio,
            blend_depth_ratio=recessed_shell_blend_depth_ratio,
        )
        (
            deformed_oral,
            jaw_shell_expansion_count,
            jaw_shell_expansion_max_displacement,
        ) = _expand_recessed_oral_tissue_horizontally(
            deformed_oral,
            colors,
            target_jaw_points,
            deformed_vertices,
            horizontal_axis=horizontal_axis,
            depth_axis=depth_axis,
            vertical_axis=vertical_axis,
            scale=recessed_shell_horizontal_scale,
            start_depth_ratio=recessed_shell_start_depth_ratio,
            blend_depth_ratio=recessed_shell_blend_depth_ratio,
        )
    neutral_rim_recession_count = 0
    jaw_rim_recession_count = 0
    neutral_rim_recession_max_displacement = 0.0
    jaw_rim_recession_max_displacement = 0.0
    if rim_recession_ratio > 0.0:
        (
            neutral_oral,
            neutral_rim_recession_count,
            neutral_rim_recession_max_displacement,
        ) = _recess_oral_tissue_behind_inner_lips(
            neutral_oral,
            colors,
            target_neutral_points,
            horizontal_axis=horizontal_axis,
            depth_axis=depth_axis,
            vertical_axis=vertical_axis,
            recession_ratio=rim_recession_ratio,
        )
        (
            deformed_oral,
            jaw_rim_recession_count,
            jaw_rim_recession_max_displacement,
        ) = _recess_oral_tissue_behind_inner_lips(
            deformed_oral,
            colors,
            target_jaw_points,
            horizontal_axis=horizontal_axis,
            depth_axis=depth_axis,
            vertical_axis=vertical_axis,
            recession_ratio=rim_recession_ratio,
        )
    rim_transition_trimmed_face_count = 0
    if trim_rim_transition_faces:
        faces, rim_transition_trimmed_face_count = (
            _trim_oral_rim_transition_faces(
                deformed_oral,
                faces,
                colors,
                target_jaw_points,
                horizontal_axis=horizontal_axis,
                depth_axis=depth_axis,
                vertical_axis=vertical_axis,
                maximum_distance_ratio=(
                    rim_transition_maximum_distance_ratio
                ),
                depth_tolerance_ratio=(
                    rim_transition_depth_tolerance_ratio
                ),
            )
        )
    if aperture_crop_scale is not None:
        neutral_oral, deformed_oral, faces, colors = _crop_to_outer_lip_aperture(
            neutral_oral,
            deformed_oral,
            faces,
            colors,
            target_jaw_points,
            horizontal_axis=horizontal_axis,
            vertical_axis=vertical_axis,
            scale=float(aperture_crop_scale),
        )
    tooth_rgb = colors.new_tensor((0.91, 0.86, 0.72))
    tongue_rgb = colors.new_tensor((0.58, 0.20, 0.23))
    tooth_count = int(((colors - tooth_rgb).abs().amax(dim=1) < 1.0e-6).sum().item())
    tongue_count = int(((colors - tongue_rgb).abs().amax(dim=1) < 1.0e-6).sum().item())
    tissue_count = int(colors.shape[0] - tooth_count - tongue_count)
    shared_landmarks = set(source_neutral_points) & set(target_neutral_points)
    report = MetahumanOralAssemblyReport(
        asset_path=str(resolved_asset_path),
        geometry_mode=geometry_mode,
        source_vertex_count=source_vertex_count,
        source_face_count=source_face_count,
        vertex_count=int(neutral_oral.shape[0]),
        face_count=int(faces.shape[0]),
        cropped_vertex_count=int(source_vertex_count - neutral_oral.shape[0]),
        cropped_face_count=int(source_face_count - faces.shape[0]),
        aperture_crop_scale=(
            float(aperture_crop_scale) if aperture_crop_scale is not None else None
        ),
        landmark_count=len(shared_landmarks & set(MOUTH_LANDMARK_IDS)),
        neutral_horizontal_scale=neutral_scales[0],
        neutral_vertical_scale=neutral_scales[1],
        neutral_depth_scale=neutral_scales[2],
        jaw_open_horizontal_scale=jaw_scales[0],
        jaw_open_vertical_scale=jaw_scales[1],
        jaw_open_depth_scale=jaw_scales[2],
        assembly_scale=float(assembly_scale),
        depth_offset=float(depth_offset),
        level_target_frame=bool(level_target_frame),
        auto_center_teeth=bool(auto_center_teeth),
        maximum_center_offset_ratio=float(maximum_center_offset_ratio),
        neutral_center_offset_horizontal=neutral_center_offsets[0],
        neutral_center_offset_vertical=neutral_center_offsets[1],
        jaw_open_center_offset_horizontal=jaw_center_offsets[0],
        jaw_open_center_offset_vertical=jaw_center_offsets[1],
        tooth_depth_offset=float(tooth_depth_offset),
        tooth_depth_blend_rings=int(tooth_depth_blend_rings),
        tooth_depth_affected_vertex_count=max(
            neutral_tooth_depth_count,
            jaw_tooth_depth_count,
        ),
        auto_center_lower_teeth=bool(auto_center_lower_teeth),
        neutral_lower_teeth_center_offset_horizontal=neutral_lower_center_offset,
        jaw_open_lower_teeth_center_offset_horizontal=jaw_lower_center_offset,
        lower_teeth_depth_offset=float(lower_teeth_depth_offset),
        lower_teeth_blend_rings=int(lower_teeth_blend_rings),
        lower_teeth_affected_vertex_count=max(
            neutral_lower_count,
            jaw_lower_count,
        ),
        containment_enabled=bool(contain_inside_lips),
        containment_scale=float(containment_scale),
        containment_depth_inset=float(containment_depth_inset),
        containment_blend_rings=int(containment_blend_rings),
        containment_affected_vertex_count=max(
            neutral_containment_count,
            jaw_containment_count,
        ),
        containment_direct_vertex_count=max(
            neutral_containment_direct_count,
            jaw_containment_direct_count,
        ),
        containment_max_planar_displacement=max(
            neutral_containment_max_displacement,
            jaw_containment_max_displacement,
        ),
        rim_recession_ratio=float(rim_recession_ratio),
        rim_recession_affected_vertex_count=max(
            neutral_rim_recession_count,
            jaw_rim_recession_count,
        ),
        rim_recession_max_depth_displacement=max(
            neutral_rim_recession_max_displacement,
            jaw_rim_recession_max_displacement,
        ),
        recessed_shell_horizontal_scale=float(
            recessed_shell_horizontal_scale
        ),
        recessed_shell_start_depth_ratio=float(
            recessed_shell_start_depth_ratio
        ),
        recessed_shell_blend_depth_ratio=float(
            recessed_shell_blend_depth_ratio
        ),
        recessed_shell_affected_vertex_count=max(
            neutral_shell_expansion_count,
            jaw_shell_expansion_count,
        ),
        recessed_shell_max_horizontal_displacement=max(
            neutral_shell_expansion_max_displacement,
            jaw_shell_expansion_max_displacement,
        ),
        rim_transition_trim_enabled=bool(trim_rim_transition_faces),
        rim_transition_maximum_distance_ratio=float(
            rim_transition_maximum_distance_ratio
        ),
        rim_transition_depth_tolerance_ratio=float(
            rim_transition_depth_tolerance_ratio
        ),
        rim_transition_trimmed_face_count=int(
            rim_transition_trimmed_face_count
        ),
        upper_teeth_vertical_offset=float(upper_teeth_vertical_offset),
        upper_teeth_vertical_blend_rings=int(upper_teeth_vertical_blend_rings),
        upper_teeth_vertical_affected_vertex_count=max(
            neutral_upper_vertical_count,
            jaw_upper_vertical_count,
        ),
        upper_crown_quantile=float(upper_crown_quantile),
        dental_uv_mask_used=(
            explicit_dental_arch_mask is None and feature_face_uvs is not None
        ),
        dental_arch_vertex_count=(
            int(dental_arch_mask.sum().item())
            if dental_arch_mask is not None
            else int(basis_vertices.shape[0])
        ),
        removed_oral_wall_uv_face_count=removed_oral_wall_uv_face_count,
        removed_tongue_uv_face_count=removed_tongue_uv_face_count,
        tongue_planar_scale=float(tongue_planar_scale),
        tongue_depth_offset=float(tongue_depth_offset),
        tongue_vertical_offset=float(tongue_vertical_offset),
        tongue_offset_blend_rings=int(tongue_offset_blend_rings),
        tongue_offset_affected_vertex_count=max(
            neutral_tongue_offset_count,
            jaw_tongue_offset_count,
        ),
        tooth_vertex_count=tooth_count,
        tongue_vertex_count=tongue_count,
        oral_tissue_vertex_count=tissue_count,
        source_metadata={str(key): str(value) for key, value in metadata.items()},
    )
    return neutral_oral, deformed_oral, faces, colors, report


# Retain the original public name for existing scripts and downstream imports.
fit_metahuman_oral_assembly = fit_oral_assembly
