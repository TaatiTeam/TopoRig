"""Pure tensor loss functions shared by training and evaluation.

Extracted without changing the equations used by retained checkpoints.
"""
from __future__ import annotations
from typing import Any, Mapping, Optional
import torch
import torch.nn.functional as F

def edge_delta_smoothness_loss(
    predicted_delta: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    source = predicted_delta.index_select(1, edges[:, 0])
    target = predicted_delta.index_select(1, edges[:, 1])
    return (source - target).square().mean()


def edge_length_preservation_loss(
    vertices: torch.Tensor,
    deformed_vertices: torch.Tensor,
    edges: torch.Tensor,
    *,
    relative: bool = True,
    min_edge_length: float = 1.0e-4,
) -> torch.Tensor:
    neutral_source = vertices.index_select(1, edges[:, 0])
    neutral_target = vertices.index_select(1, edges[:, 1])
    deformed_source = deformed_vertices.index_select(1, edges[:, 0])
    deformed_target = deformed_vertices.index_select(1, edges[:, 1])
    neutral_length = torch.linalg.vector_norm(
        neutral_source - neutral_target,
        dim=-1,
    )
    deformed_length = torch.linalg.vector_norm(
        deformed_source - deformed_target,
        dim=-1,
    )
    error = deformed_length - neutral_length.detach()
    if relative:
        error = error / neutral_length.detach().clamp_min(float(min_edge_length))
    return error.square().mean()


def face_normal_consistency_loss(
    vertices: torch.Tensor,
    deformed_vertices: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    neutral_normals = face_normals(vertices, faces).detach()
    deformed_normals = face_normals(deformed_vertices, faces)
    cosine = (neutral_normals * deformed_normals).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cosine).mean()


def face_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    triangles = vertices[:, faces]
    normals = torch.cross(
        triangles[:, :, 1] - triangles[:, :, 0],
        triangles[:, :, 2] - triangles[:, :, 0],
        dim=-1,
    )
    return F.normalize(normals, dim=-1, eps=1.0e-6)


def mesh_loss_fn(
    predicted: torch.Tensor,
    target: torch.Tensor,
    config: Mapping[str, Any],
    vertex_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    error = elementwise_loss(predicted, target, config)
    weights = active_motion_vertex_weights(target, config)
    if vertex_mask is not None:
        mask_weights = normalized_vertex_weights(
            vertex_mask,
            predicted,
        ).unsqueeze(-1)
        weights = mask_weights if weights is None else weights * mask_weights
    if weights is None:
        base_loss = error.mean()
    else:
        normalizer = weights.sum().clamp_min(1.0e-8) * error.shape[-1]
        base_loss = (error * weights).sum() / normalizer

    loss = base_loss * float(config.get("base_loss_weight", 1.0))

    active_balanced_weight = float(config.get("active_motion_balanced_weight", 0.0))
    if active_balanced_weight > 0.0:
        active_error = active_motion_error_mean(
            error,
            target,
            config,
            vertex_mask=vertex_mask,
        )
        if active_error is not None:
            loss = loss + active_balanced_weight * active_error

    normalized_region_loss = target_normalized_region_loss(
        error,
        target,
        config,
        vertex_mask=vertex_mask,
    )
    if normalized_region_loss is not None:
        loss = loss + normalized_region_loss

    return loss


def active_motion_error_mean(
    error: torch.Tensor,
    target: torch.Tensor,
    config: Mapping[str, Any],
    vertex_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    active = active_motion_mask(target, config)
    if vertex_mask is not None:
        active_weights = normalized_vertex_weights(vertex_mask, target) * active.to(
            dtype=target.dtype
        )
    else:
        active_weights = active.to(dtype=target.dtype)
    weight_sum = active_weights.sum()
    if float(weight_sum.detach().cpu()) <= 0.0:
        return None

    return (error * active_weights.unsqueeze(-1)).sum() / (
        weight_sum * error.shape[-1]
    )


def target_normalized_region_loss(
    error: torch.Tensor,
    target: torch.Tensor,
    config: Mapping[str, Any],
    *,
    include_weight: bool = True,
    vertex_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    weight = float(config.get("target_normalized_region_weight", 0.0))
    if include_weight and weight <= 0.0:
        return None

    if error.dim() == 2:
        error = error.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)
    if error.shape[:2] != target.shape[:2]:
        raise ValueError("error and target must have matching batch and vertex dims.")

    active = active_motion_mask(target, config)
    mask = (
        normalized_vertex_weights(vertex_mask, target)
        if vertex_mask is not None
        else None
    )
    sample_losses = []
    for batch_index in range(target.shape[0]):
        sample_active = active[batch_index]
        if mask is None:
            sample_mask = torch.ones_like(sample_active, dtype=target.dtype)
        else:
            sample_mask = mask[batch_index]
        sample_active_weights = sample_mask * sample_active.to(dtype=target.dtype)
        sample_inactive_weights = sample_mask * (~sample_active).to(
            dtype=target.dtype
        )
        sample_error = error[batch_index]
        region_terms = []
        if float(sample_active_weights.sum().detach().cpu()) > 0.0:
            region_terms.append(
                (sample_error * sample_active_weights.unsqueeze(-1)).sum()
                / (sample_active_weights.sum() * sample_error.shape[-1])
            )
        if float(sample_inactive_weights.sum().detach().cpu()) > 0.0:
            region_terms.append(
                (sample_error * sample_inactive_weights.unsqueeze(-1)).sum()
                / (sample_inactive_weights.sum() * sample_error.shape[-1])
            )
        if not region_terms:
            continue
        scale_active = sample_active & (sample_mask > 0.0)
        denominator = target_loss_normalizer(
            target[batch_index : batch_index + 1],
            scale_active.unsqueeze(0),
            config,
        )
        sample_losses.append(torch.stack(region_terms).mean() / denominator)

    if not sample_losses:
        return None

    loss = torch.stack(sample_losses).mean()
    if include_weight:
        loss = loss * weight
    return loss


def target_loss_normalizer(
    target: torch.Tensor,
    active: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    scale = target_motion_scale(target, active, config)
    loss_type = str(config.get("type", "mse")).lower()
    if loss_type == "mse":
        return scale.square()
    return scale


def target_motion_scale(
    target: torch.Tensor,
    active: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    min_scale = float(config.get("target_normalization_min", 1.0e-3))
    if bool(active.any()):
        target_norm = torch.linalg.vector_norm(target.detach()[active], dim=-1)
        scale = target_norm.mean()
    else:
        scale = target.detach().new_tensor(min_scale)
    return scale.clamp_min(min_scale)


def normalized_vertex_weights(
    vertex_mask: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if vertex_mask.dim() == 1:
        vertex_mask = vertex_mask.unsqueeze(0)
    if vertex_mask.dim() != 2:
        raise ValueError("vertex_mask must have shape [V] or [B, V].")
    if reference.dim() == 2:
        batch_size = 1
        vertex_count = reference.shape[0]
    else:
        batch_size, vertex_count = reference.shape[:2]
    if vertex_mask.shape[1] != vertex_count:
        raise ValueError("vertex_mask must match the reference vertex count.")
    if vertex_mask.shape[0] == 1 and batch_size > 1:
        vertex_mask = vertex_mask.expand(batch_size, -1)
    if vertex_mask.shape[0] != batch_size:
        raise ValueError(
            "vertex_mask batch size must match the reference batch size."
        )
    return vertex_mask.to(device=reference.device, dtype=reference.dtype).clamp(
        0.0,
        1.0,
    )


def active_motion_mask(
    target: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    target_norm = torch.linalg.vector_norm(target.detach(), dim=-1)
    threshold = float(config.get("active_motion_threshold", 1.0e-3))
    if threshold <= 0.0:
        return target_norm > 0.0
    return target_norm >= threshold


def active_motion_vertex_weights(
    target: torch.Tensor,
    config: Mapping[str, Any],
) -> Optional[torch.Tensor]:
    active_weight = float(config.get("active_motion_weight", 0.0))
    if active_weight <= 0.0:
        return None

    target_norm = torch.linalg.vector_norm(target.detach(), dim=-1, keepdim=True)
    threshold = float(config.get("active_motion_threshold", 1.0e-3))
    power = float(config.get("active_motion_power", 1.0))
    if threshold <= 0.0:
        activation = (target_norm > 0.0).to(dtype=target.dtype)
    else:
        activation = (target_norm / threshold).clamp(0.0, 1.0)
    if power != 1.0:
        activation = activation.pow(power)
    return 1.0 + active_weight * activation


def scalar_loss_gradient_norm(
    loss: torch.Tensor,
    reference: torch.Tensor,
) -> Optional[torch.Tensor]:
    gradient = torch.autograd.grad(
        loss,
        reference,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]
    if gradient is None:
        return None
    return torch.linalg.vector_norm(gradient.detach())


def landmark_active_motion_loss(
    predicted_landmark_delta: torch.Tensor,
    target_landmark_delta: torch.Tensor,
    config: Mapping[str, Any],
) -> Optional[torch.Tensor]:
    active = landmark_active_motion_mask(target_landmark_delta, config)
    if not bool(active.any()):
        return None
    active_error = elementwise_loss(
        predicted_landmark_delta,
        target_landmark_delta,
        config,
    )
    loss = active_error[active].mean()
    if bool(config.get("landmark_active_normalize_by_target", False)):
        loss = loss / target_loss_normalizer(target_landmark_delta, active, config)
    return loss


def landmark_active_motion_mask(
    target_landmark_delta: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    return active_motion_mask(target_landmark_delta, config)


def elementwise_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    name = str(config.get("type", "mse")).lower()
    if name == "mse":
        return (predicted - target).square()
    if name in {"l1", "mae"}:
        return (predicted - target).abs()
    if name in {"smooth_l1", "huber"}:
        return F.smooth_l1_loss(
            predicted,
            target,
            beta=float(config.get("beta", 1.0)),
            reduction="none",
        )
    raise ValueError(f"Unsupported loss type: {name!r}.")

