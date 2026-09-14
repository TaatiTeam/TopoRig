from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

try:
    import nvdiffrast.torch as dr
except ImportError:  # pragma: no cover - depends on optional CUDA package.
    dr = None


PathLike = Union[str, Path]
_NVDIFFRAST_CONTEXTS = {}


def depth_l1_loss(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return mean absolute depth error, optionally averaged over a mask."""
    if not isinstance(pred_depth, torch.Tensor):
        raise TypeError("pred_depth must be a torch.Tensor.")
    if not isinstance(target_depth, torch.Tensor):
        raise TypeError("target_depth must be a torch.Tensor.")

    target_depth = target_depth.to(device=pred_depth.device, dtype=pred_depth.dtype)
    pred_depth, target_depth = torch.broadcast_tensors(pred_depth, target_depth)
    error = (pred_depth - target_depth).abs()

    if mask is None:
        return error.mean()

    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor when provided.")

    mask = mask.to(device=pred_depth.device, dtype=pred_depth.dtype).clamp_min(0.0)
    try:
        mask = torch.broadcast_to(mask, error.shape)
    except RuntimeError as exc:
        raise ValueError("mask must be broadcastable to the depth tensor shape.") from exc

    return (error * mask).sum() / mask.sum().clamp_min(1.0e-8)


def save_depth_debug_image(depth: torch.Tensor, output_path: PathLike) -> None:
    """Save a depth tensor as a PNG visualization using matplotlib."""
    depth_map = _depth_map_2d(depth)
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4, 4), dpi=120)
    image = axis.imshow(depth_map.numpy(), cmap="magma")
    axis.axis("off")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout(pad=0.1)
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def depth_difference_l1_loss(
    neutral_vertices: torch.Tensor,
    faces: torch.Tensor,
    pred_delta: torch.Tensor,
    target_delta: torch.Tensor,
    image_size: tuple[int, int] = (96, 96),
    up_axis: str = "z",
    return_details: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compare predicted and target rasterized depth-change maps."""
    neutral_vertices, was_unbatched = _ensure_batched_vertices(neutral_vertices)
    faces = _ensure_batched_faces(
        faces,
        batch_size=neutral_vertices.shape[0],
        device=neutral_vertices.device,
    )
    pred_delta, _ = _ensure_batched_vertices(pred_delta)
    target_delta, _ = _ensure_batched_vertices(target_delta)
    pred_delta = pred_delta.to(
        device=neutral_vertices.device,
        dtype=neutral_vertices.dtype,
    )
    target_delta = target_delta.to(
        device=neutral_vertices.device,
        dtype=neutral_vertices.dtype,
    )
    if pred_delta.shape != neutral_vertices.shape:
        raise ValueError("pred_delta must have the same shape as neutral_vertices.")
    if target_delta.shape != neutral_vertices.shape:
        raise ValueError("target_delta must have the same shape as neutral_vertices.")

    center, scale = depth_projection_transform(
        neutral_vertices,
        image_size=image_size,
        up_axis=up_axis,
    )
    depth_bounds = depth_projection_depth_bounds(
        torch.cat(
            (
                neutral_vertices.detach(),
                (neutral_vertices + pred_delta).detach(),
                (neutral_vertices + target_delta).detach(),
            ),
            dim=1,
        ),
        center=center,
        up_axis=up_axis,
    )
    neutral_depth, neutral_coverage = render_rasterized_depth_map(
        neutral_vertices,
        faces,
        image_size=image_size,
        center=center,
        scale=scale,
        depth_bounds=depth_bounds,
        up_axis=up_axis,
    )
    pred_depth, pred_coverage = render_rasterized_depth_map(
        neutral_vertices + pred_delta,
        faces,
        image_size=image_size,
        center=center,
        scale=scale,
        depth_bounds=depth_bounds,
        up_axis=up_axis,
    )
    target_depth, target_coverage = render_rasterized_depth_map(
        neutral_vertices + target_delta,
        faces,
        image_size=image_size,
        center=center,
        scale=scale,
        depth_bounds=depth_bounds,
        up_axis=up_axis,
    )
    valid = (
        (neutral_coverage > 0.0)
        & (pred_coverage > 0.0)
        & (target_coverage > 0.0)
    )

    pred_depth_delta = pred_depth - neutral_depth
    target_depth_delta = target_depth - neutral_depth
    global_loss = depth_l1_loss(pred_depth_delta, target_depth_delta, valid)
    loss = global_loss

    if was_unbatched:
        details = {
            "neutral_depth": neutral_depth[0],
            "pred_depth": pred_depth[0],
            "target_depth": target_depth[0],
            "pred_depth_delta": pred_depth_delta[0],
            "target_depth_delta": target_depth_delta[0],
            "valid": valid[0],
            "global_loss": global_loss,
        }
    else:
        details = {
            "neutral_depth": neutral_depth,
            "pred_depth": pred_depth,
            "target_depth": target_depth,
            "pred_depth_delta": pred_depth_delta,
            "target_depth_delta": target_depth_delta,
            "valid": valid,
            "global_loss": global_loss,
        }
    if return_details:
        return loss, details
    return loss


def depth_projection_transform(
    vertices: torch.Tensor,
    image_size: tuple[int, int] = (96, 96),
    up_axis: str = "z",
    occupancy: float = 0.78,
) -> tuple[torch.Tensor, torch.Tensor]:
    vertices, _ = _ensure_batched_vertices(vertices)
    render_vertices = vertices_to_depth_render_axes(vertices.detach(), up_axis=up_axis)
    bounds_min = render_vertices.amin(dim=1, keepdim=True)
    bounds_max = render_vertices.amax(dim=1, keepdim=True)
    center = (bounds_min + bounds_max) * 0.5
    span = (bounds_max[..., :2] - bounds_min[..., :2]).amax(
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-6)
    scale = occupancy * float(min(image_size) - 1) / span
    return center, scale


def depth_projection_depth_bounds(
    vertices: torch.Tensor,
    center: torch.Tensor,
    up_axis: str = "z",
    margin_ratio: float = 0.05,
) -> torch.Tensor:
    vertices, was_unbatched = _ensure_batched_vertices(vertices)
    center = center.to(device=vertices.device, dtype=vertices.dtype)

    render_vertices = vertices_to_depth_render_axes(vertices.detach(), up_axis=up_axis)
    render_vertices = render_vertices - center
    depth = render_vertices[..., 2]
    depth_min = depth.amin(dim=1)
    depth_max = depth.amax(dim=1)
    span = (depth_max - depth_min).clamp_min(1.0e-6)
    margin = span * float(margin_ratio)
    bounds = torch.stack((depth_min - margin, depth_max + margin), dim=-1)
    if was_unbatched:
        return bounds[0]
    return bounds


def render_rasterized_depth_map(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    image_size: tuple[int, int] = (96, 96),
    center: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    depth_bounds: torch.Tensor | None = None,
    up_axis: str = "z",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize triangle-interpolated depth with nvdiffrast."""
    vertices, was_unbatched = _ensure_batched_vertices(vertices)
    faces = _ensure_batched_faces(
        faces,
        batch_size=vertices.shape[0],
        device=vertices.device,
    )
    if center is None or scale is None:
        center, scale = depth_projection_transform(
            vertices,
            image_size=image_size,
            up_axis=up_axis,
        )
    center = center.to(device=vertices.device, dtype=vertices.dtype)
    scale = scale.to(device=vertices.device, dtype=vertices.dtype)
    if depth_bounds is None:
        depth_bounds = depth_projection_depth_bounds(
            vertices,
            center=center,
            up_axis=up_axis,
        )
    depth_bounds = _ensure_batched_depth_bounds(
        depth_bounds,
        batch_size=vertices.shape[0],
        device=vertices.device,
        dtype=vertices.dtype,
    )

    render_vertices = vertices_to_depth_render_axes(vertices, up_axis=up_axis) - center
    depth, coverage = _render_rasterized_depth_nvdiffrast(
        render_vertices,
        faces,
        scale,
        depth_bounds,
        image_size=image_size,
    )
    if was_unbatched:
        return depth[0], coverage[0]
    return depth, coverage


def vertices_to_depth_render_axes(vertices: torch.Tensor, up_axis: str = "z") -> torch.Tensor:
    up_axis = up_axis.lower()
    if up_axis in {"z", "z-up", "z_up", "blender", "fbx"}:
        return torch.stack(
            (
                -vertices[..., 0],
                vertices[..., 2],
                vertices[..., 1],
            ),
            dim=-1,
        )
    if up_axis in {"y", "y-up", "y_up", "render"}:
        return vertices
    raise ValueError("up_axis must be 'z' or 'y'.")


def _depth_map_2d(depth: torch.Tensor) -> torch.Tensor:
    if not isinstance(depth, torch.Tensor):
        raise TypeError("depth must be a torch.Tensor.")

    depth_map = depth.detach().float().cpu()

    if depth_map.dim() == 4:
        depth_map = depth_map[0, 0]
    elif depth_map.dim() == 3:
        if depth_map.shape[0] == 1:
            depth_map = depth_map[0]
        elif depth_map.shape[-1] == 1:
            depth_map = depth_map[..., 0]
        else:
            depth_map = depth_map[0]

    if depth_map.dim() != 2:
        raise ValueError("depth must be a 2D map or a batched/channel depth tensor.")

    depth_map = torch.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_map = (depth_map - depth_min) / (depth_max - depth_min + 1.0e-8)

    return depth_map


def _ensure_batched_vertices(vertices: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if not isinstance(vertices, torch.Tensor):
        raise TypeError("vertices must be a torch.Tensor.")
    if vertices.dim() == 2:
        if vertices.shape[-1] != 3:
            raise ValueError("vertices must have shape [V, 3] or [B, V, 3].")
        return vertices.unsqueeze(0), True
    if vertices.dim() == 3 and vertices.shape[-1] == 3:
        return vertices, False
    raise ValueError("vertices must have shape [V, 3] or [B, V, 3].")


def _ensure_batched_faces(
    faces: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(faces, torch.Tensor):
        raise TypeError("faces must be a torch.Tensor.")
    faces = faces.to(device=device, dtype=torch.long)
    if faces.dim() == 2:
        if faces.shape[-1] != 3:
            raise ValueError("faces must have shape [F, 3] or [B, F, 3].")
        return faces.unsqueeze(0).expand(batch_size, -1, -1)
    if faces.dim() == 3 and faces.shape[-1] == 3:
        if faces.shape[0] != batch_size:
            raise ValueError("batched faces must match the vertices batch size.")
        return faces
    raise ValueError("faces must have shape [F, 3] or [B, F, 3].")


def _ensure_batched_depth_bounds(
    depth_bounds: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(depth_bounds, torch.Tensor):
        raise TypeError("depth_bounds must be a torch.Tensor.")
    depth_bounds = depth_bounds.to(device=device, dtype=dtype)
    if depth_bounds.shape == (2,):
        return depth_bounds.view(1, 2).expand(batch_size, -1)
    if depth_bounds.shape == (batch_size, 2):
        return depth_bounds
    raise ValueError("depth_bounds must have shape [2] or [B, 2].")


def _render_rasterized_depth_nvdiffrast(
    render_vertices: torch.Tensor,
    faces: torch.Tensor,
    scale: torch.Tensor,
    depth_bounds: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_nvdiffrast(render_vertices)
    tri = _shared_nvdiffrast_triangles(faces, render_vertices.device)
    height, width = image_size
    clip_positions = _depth_render_vertices_to_clip(
        render_vertices,
        scale,
        depth_bounds,
        image_size=image_size,
    )
    ctx = _get_nvdiffrast_context(render_vertices.device)
    rast, _ = dr.rasterize(ctx, clip_positions.contiguous(), tri, (height, width))
    depth_values = render_vertices[..., 2:3].contiguous()
    depth, _ = dr.interpolate(depth_values, rast, tri)
    coverage = (rast[..., 3:4] > 0).to(dtype=render_vertices.dtype)

    depth = torch.flip(depth, dims=(1,)).squeeze(-1)
    coverage = torch.flip(coverage, dims=(1,)).squeeze(-1)
    return depth * coverage, coverage


def _depth_render_vertices_to_clip(
    render_vertices: torch.Tensor,
    scale: torch.Tensor,
    depth_bounds: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    height, width = image_size
    scale = scale.view(render_vertices.shape[0], 1)
    screen_x = render_vertices[..., 0] * scale + (width - 1) * 0.5
    screen_y = (height - 1) * 0.5 - render_vertices[..., 1] * scale
    depth = render_vertices[..., 2]

    x_ndc = screen_x / max(width - 1, 1) * 2.0 - 1.0
    y_ndc = 1.0 - screen_y / max(height - 1, 1) * 2.0

    near = depth_bounds[:, 0].view(-1, 1)
    far = depth_bounds[:, 1].view(-1, 1)
    z_ndc = (depth - near) / (far - near).clamp_min(1.0e-6) * 2.0 - 1.0

    ones = torch.ones_like(x_ndc)
    return torch.stack((x_ndc, y_ndc, z_ndc, ones), dim=-1)


def _require_nvdiffrast(tensor: torch.Tensor) -> None:
    if dr is None:
        raise ImportError(
            "The mesh depth renderer requires nvdiffrast. Install nvdiffrast "
            "or set depth_weight: 0 to disable the rasterized depth loss."
        )
    if tensor.device.type != "cuda":
        raise RuntimeError(
            "The nvdiffrast mesh depth renderer requires CUDA tensors. Move the "
            "model/data to CUDA or set depth_weight: 0."
        )


def _shared_nvdiffrast_triangles(
    faces: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if faces.shape[0] > 1 and not bool((faces == faces[:1]).all()):
        raise ValueError("nvdiffrast depth rendering requires shared batched faces.")
    return faces[0].to(device=device, dtype=torch.int32).contiguous()


def _get_nvdiffrast_context(device: torch.device):
    key = device.index if device.index is not None else torch.cuda.current_device()
    if key not in _NVDIFFRAST_CONTEXTS:
        with torch.cuda.device(device):
            _NVDIFFRAST_CONTEXTS[key] = dr.RasterizeCudaContext()
    return _NVDIFFRAST_CONTEXTS[key]
