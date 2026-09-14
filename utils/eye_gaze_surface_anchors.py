from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from utils.eye_geometry import (
    EYE_CAPABILITY_LAYERED_MESH,
    EYE_CAPABILITY_RIGID_MESH,
    EYE_CAPABILITY_TEXTURE_ONLY,
)


EYE_GAZE_SURFACE_ANCHOR_VERSION = 3
SUPPORTED_EYE_GAZE_SURFACE_ANCHOR_VERSIONS = (2, 3)


def eye_capability_name(value: int) -> str:
    names = {
        EYE_CAPABILITY_TEXTURE_ONLY: "texture_only",
        EYE_CAPABILITY_RIGID_MESH: "rigid_mesh",
        EYE_CAPABILITY_LAYERED_MESH: "layered_mesh",
    }
    if int(value) not in names:
        raise ValueError(f"Unknown eye geometry capability code: {value}.")
    return names[int(value)]


def load_eye_gaze_surface_anchor_payload(
    path: Path,
    *,
    vertex_count: int | None = None,
) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected eye-gaze anchor cache {path} to be a mapping.")

    version_value = payload.get("version")
    if isinstance(version_value, torch.Tensor):
        version = int(version_value.item())
    else:
        version = int(version_value or -1)
    if version not in SUPPORTED_EYE_GAZE_SURFACE_ANCHOR_VERSIONS:
        raise ValueError(
            f"Eye-gaze anchor cache {path} has version {version}; expected "
            f"one of {SUPPORTED_EYE_GAZE_SURFACE_ANCHOR_VERSIONS}."
        )

    required = (
        "mediapipe_ids",
        "face_vertex_ids",
        "barycentric_weights",
        "neutral_positions",
        "rigid_vertex_ids",
        "roi_vertex_ids",
        "component_id",
    )
    missing = [name for name in required if not isinstance(payload.get(name), torch.Tensor)]
    if missing:
        raise ValueError(f"Eye-gaze anchor cache {path} is missing tensors: {missing}.")

    result = {
        name: value.detach().cpu().contiguous()
        for name, value in payload.items()
        if isinstance(value, torch.Tensor)
    }
    mediapipe_ids = result["mediapipe_ids"].long().flatten()
    face_vertex_ids = result["face_vertex_ids"].long()
    barycentric_weights = result["barycentric_weights"].float()
    neutral_positions = result["neutral_positions"].float()
    rigid_vertex_ids = result["rigid_vertex_ids"].long().flatten()
    roi_vertex_ids = result["roi_vertex_ids"].long().flatten()
    if face_vertex_ids.shape != (mediapipe_ids.numel(), 3):
        raise ValueError("Eye-gaze face_vertex_ids must have shape [L, 3].")
    if barycentric_weights.shape != face_vertex_ids.shape:
        raise ValueError("Eye-gaze barycentric_weights must match face_vertex_ids.")
    if neutral_positions.shape != (mediapipe_ids.numel(), 3):
        raise ValueError("Eye-gaze neutral_positions must have shape [L, 3].")
    if mediapipe_ids.unique().numel() != mediapipe_ids.numel():
        raise ValueError("Eye-gaze mediapipe_ids must be unique.")
    if not torch.isfinite(barycentric_weights).all():
        raise ValueError("Eye-gaze barycentric weights must be finite.")
    if not torch.allclose(
        barycentric_weights.sum(dim=-1),
        torch.ones(mediapipe_ids.numel()),
        atol=1.0e-4,
        rtol=1.0e-4,
    ):
        raise ValueError("Eye-gaze barycentric weights must sum to one.")
    if (barycentric_weights < -1.0e-4).any():
        raise ValueError("Eye-gaze barycentric weights must be non-negative.")
    if rigid_vertex_ids.numel() < 3:
        raise ValueError("Eye-gaze rigid region must contain at least three vertices.")
    if roi_vertex_ids.numel() < 3:
        raise ValueError("Eye-gaze ROI must contain at least three vertices.")
    if vertex_count is not None:
        for name, values in (
            ("face_vertex_ids", face_vertex_ids),
            ("rigid_vertex_ids", rigid_vertex_ids),
            ("roi_vertex_ids", roi_vertex_ids),
        ):
            if values.numel() and (
                int(values.min()) < 0 or int(values.max()) >= int(vertex_count)
            ):
                raise ValueError(
                    f"Eye-gaze {name} contains an ID outside [0, {vertex_count})."
                )

    result["mediapipe_ids"] = mediapipe_ids
    result["face_vertex_ids"] = face_vertex_ids
    result["barycentric_weights"] = barycentric_weights
    result["neutral_positions"] = neutral_positions
    result["rigid_vertex_ids"] = rigid_vertex_ids.unique(sorted=True)
    result["roi_vertex_ids"] = roi_vertex_ids.unique(sorted=True)
    result["component_id"] = result["component_id"].long().reshape(())
    result["cache_version"] = torch.tensor(version, dtype=torch.long)

    if version < 3:
        # Version 2 frequently selected the face as its rigid region. Keep its
        # legacy anchor/loss fields readable, but never advertise it as a
        # physical eye assembly.
        result["eye_capability"] = torch.tensor(
            EYE_CAPABILITY_TEXTURE_ONLY,
            dtype=torch.long,
        )
        result["eye_index"] = torch.tensor(-1, dtype=torch.long)
        result["pivot_component_id"] = torch.tensor(-1, dtype=torch.long)
        result["assembly_component_ids"] = torch.empty(0, dtype=torch.long)
        result["assembly_vertex_ids"] = torch.empty(0, dtype=torch.long)
        result["rotation_center"] = torch.full((3,), float("nan"))
        result["rotation_radius"] = torch.tensor(float("nan"))
        result["fit_confidence"] = torch.tensor(0.0)
        result["selection_score"] = torch.tensor(float("inf"))
        return result

    physical_required = (
        "eye_capability",
        "eye_index",
        "pivot_component_id",
        "assembly_component_ids",
        "assembly_vertex_ids",
        "rotation_center",
        "rotation_radius",
        "fit_confidence",
        "selection_score",
    )
    missing_physical = [
        name for name in physical_required if not isinstance(result.get(name), torch.Tensor)
    ]
    if missing_physical:
        raise ValueError(
            f"Eye-gaze anchor cache {path} is missing v3 tensors: "
            f"{missing_physical}."
        )
    capability = int(result["eye_capability"].item())
    eye_capability_name(capability)
    eye_index = int(result["eye_index"].item())
    if eye_index not in {-1, 0, 1}:
        raise ValueError("Eye-gaze eye_index must be -1, 0 (left), or 1 (right).")
    assembly_component_ids = (
        result["assembly_component_ids"].long().flatten().unique(sorted=True)
    )
    assembly_vertex_ids = result["assembly_vertex_ids"].long().flatten().unique(sorted=True)
    rotation_center = result["rotation_center"].float().flatten()
    rotation_radius = result["rotation_radius"].float().reshape(())
    fit_confidence = result["fit_confidence"].float().reshape(())
    selection_score = result["selection_score"].float().reshape(())
    if rotation_center.shape != (3,):
        raise ValueError("Eye-gaze rotation_center must have shape [3].")
    if not 0.0 <= float(fit_confidence) <= 1.0:
        raise ValueError("Eye-gaze fit_confidence must lie in [0, 1].")
    is_physical = capability in {
        EYE_CAPABILITY_RIGID_MESH,
        EYE_CAPABILITY_LAYERED_MESH,
    }
    if is_physical:
        if int(result["pivot_component_id"]) < 0:
            raise ValueError("A physical eye cache requires a pivot component.")
        if assembly_component_ids.numel() < 1 or assembly_vertex_ids.numel() < 4:
            raise ValueError("A physical eye cache requires a non-empty assembly.")
        if not torch.isfinite(rotation_center).all():
            raise ValueError("A physical eye cache requires a finite rotation center.")
        if not torch.isfinite(rotation_radius) or float(rotation_radius) <= 0.0:
            raise ValueError("A physical eye cache requires a positive rotation radius.")
    elif assembly_component_ids.numel() or assembly_vertex_ids.numel():
        raise ValueError("A texture-only eye cache must not expose a physical assembly.")
    if vertex_count is not None and assembly_vertex_ids.numel() and (
        int(assembly_vertex_ids.min()) < 0
        or int(assembly_vertex_ids.max()) >= int(vertex_count)
    ):
        raise ValueError(
            "Eye-gaze assembly_vertex_ids contains an ID outside "
            f"[0, {vertex_count})."
        )
    result["eye_capability"] = torch.tensor(capability, dtype=torch.long)
    result["eye_index"] = torch.tensor(eye_index, dtype=torch.long)
    result["pivot_component_id"] = result["pivot_component_id"].long().reshape(())
    result["assembly_component_ids"] = assembly_component_ids
    result["assembly_vertex_ids"] = assembly_vertex_ids
    result["rotation_center"] = rotation_center
    result["rotation_radius"] = rotation_radius
    result["fit_confidence"] = fit_confidence
    result["selection_score"] = selection_score
    return result


def select_surface_anchors(
    payload: Mapping[str, Any],
    required_ids: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    mediapipe_ids = payload.get("gaze_mediapipe_ids")
    face_vertex_ids = payload.get("gaze_face_vertex_ids")
    barycentric_weights = payload.get("gaze_barycentric_weights")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (mediapipe_ids, face_vertex_ids, barycentric_weights)
    ):
        return None

    ids = mediapipe_ids.detach().cpu().long().flatten().tolist()
    index_by_id = {int(value): index for index, value in enumerate(ids)}
    if any(int(value) not in index_by_id for value in required_ids):
        return None
    indices = torch.tensor(
        [index_by_id[int(value)] for value in required_ids],
        dtype=torch.long,
    )
    triangles = face_vertex_ids.detach().cpu().long()
    weights = barycentric_weights.detach().cpu().float()
    if triangles.dim() == 3 and triangles.shape[0] == 1:
        triangles = triangles[0]
    if weights.dim() == 3 and weights.shape[0] == 1:
        weights = weights[0]
    if triangles.dim() != 2 or triangles.shape[-1] != 3:
        raise ValueError("gaze_face_vertex_ids must have shape [L, 3].")
    if weights.shape != triangles.shape:
        raise ValueError("gaze_barycentric_weights must match gaze_face_vertex_ids.")
    return (
        triangles.index_select(0, indices).to(device=device),
        weights.index_select(0, indices).to(device=device, dtype=dtype),
    )


def interpolate_surface_anchors(
    vertices: torch.Tensor,
    face_vertex_ids: torch.Tensor,
    barycentric_weights: torch.Tensor,
) -> torch.Tensor:
    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0)
    if vertices.dim() != 3 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape [V, 3] or [B, V, 3].")
    if face_vertex_ids.dim() == 2:
        face_vertex_ids = face_vertex_ids.unsqueeze(0).expand(vertices.shape[0], -1, -1)
    if barycentric_weights.dim() == 2:
        barycentric_weights = barycentric_weights.unsqueeze(0).expand(
            vertices.shape[0], -1, -1
        )
    if face_vertex_ids.shape != barycentric_weights.shape:
        raise ValueError("Surface-anchor faces and weights must have matching shapes.")
    if face_vertex_ids.shape[0] != vertices.shape[0] or face_vertex_ids.shape[-1] != 3:
        raise ValueError("Surface anchors must have shape [B, L, 3].")
    if face_vertex_ids.numel() and (
        int(face_vertex_ids.min()) < 0 or int(face_vertex_ids.max()) >= vertices.shape[1]
    ):
        raise ValueError("Surface-anchor face vertices are outside the mesh range.")

    batch = torch.arange(vertices.shape[0], device=vertices.device).view(-1, 1, 1)
    triangle_vertices = vertices[batch, face_vertex_ids]
    return (triangle_vertices * barycentric_weights.unsqueeze(-1)).sum(dim=2)
