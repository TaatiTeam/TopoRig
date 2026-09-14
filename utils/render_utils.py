from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

try:
    import nvdiffrast.torch as dr
except ImportError:
    dr = None
    print(
        "[WARNING] nvdiffrast is not installed and cannot be used for "
        "differential rendering"
    )


ImageSize = Union[int, Tuple[int, int]]
TensorMapping = Mapping[str, torch.Tensor]
NVDIFFRAST_AVAILABLE = dr is not None
_NVDIFFRAST_CONTEXTS = {}


DEFAULT_IMAGE_SIZE: Tuple[int, int] = (1024, 1024)
DEFAULT_VIEW_OCCUPANCY = 0.75

# Baseline front-facing screen-space pinhole camera with +X left, +Y up, and
# +Z depth.  The projected vertices are auto-framed by default so samples with
# different mesh scales share the same output framing.
CAMERA_INTRINSICS = {
    "image_size": DEFAULT_IMAGE_SIZE,
    "focal_length": (1030.0, 1006.0),
    "principal_point": (511.5, 514.6),
}

CAMERA_EXTRINSICS = {
    "R": (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    "T": (0.0, 0.0, 1.2),
}


def differentiable_render(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    texture: Optional[TensorMapping] = None,
    image_size: Optional[ImageSize] = None,
    background_color: Sequence[float] = (1.0, 1.0, 1.0),
    backend: str = "torch",
    antialias: bool = True,
    torch_samples_per_face: int = 4,
    splat_radius_px: float = 0.5,
    splat_kernel_radius: int = 1,
    depth_sharpness: float = 100.0,
    chunk_size: int = 200_000,
    fit_to_view: bool = True,
    view_occupancy: float = DEFAULT_VIEW_OCCUPANCY,
    fit_to_view_reference: Optional[torch.Tensor] = None,
    fit_visible_component: bool = False,
    foreground_threshold: float = 0.02,
) -> torch.Tensor:
    """Render a textured mesh to a differentiable torch image tensor.

    By default this uses a lightweight pure-Torch soft splat renderer.  When
    ``backend="nvdiffrast"`` is requested and nvdiffrast is installed, it uses
    GPU triangle rasterization with differentiable UV interpolation instead.
    Gradients flow to floating-point vertices, UVs, texture maps, and vertex
    colors when those inputs require gradients.

    Args:
        vertices: Mesh vertices with shape ``[V, 3]`` or ``[B, V, 3]``.
        faces: Triangle vertex indices with shape ``[F, 3]`` or ``[B, F, 3]``.
        texture: Optional mapping. Supported keys:
            ``texture_image`` or ``maps``: ``[H, W, 3/4]`` or ``[B, H, W, 3/4]``.
            ``verts_uvs``: ``[V, 2]`` or ``[B, V, 2]`` UV coordinates in [0, 1].
            ``faces_uvs``: ``[F, 3]`` or ``[B, F, 3]`` UV indices. Defaults to
            ``faces``.
            ``vertex_colors``: ``[V, 3]`` or ``[B, V, 3]`` fallback colors.
        image_size: Output size as ``int`` or ``(height, width)``. Defaults to
            ``CAMERA_INTRINSICS["image_size"]``.
        background_color: RGB color used where no splat lands.
        backend: ``"torch"``, ``"nvdiffrast"``, or ``"auto"``. ``"auto"``
            uses nvdiffrast only when it is installed and vertices are on CUDA.
        antialias: Apply nvdiffrast edge antialiasing when using that backend.
        torch_samples_per_face: Number of barycentric samples per triangle for
            the pure-Torch renderer. More samples improve texture/edge detail
            at higher memory and runtime cost.
        splat_radius_px: Gaussian sigma in pixels.
        splat_kernel_radius: Integer pixel radius for each local splat.
        depth_sharpness: Larger values make nearer samples dominate more.
        chunk_size: Number of face samples processed per chunk.
        fit_to_view: Center projected vertices in the image and scale them to
            ``view_occupancy`` of the shorter image dimension.
        view_occupancy: Fraction of the shorter image dimension occupied by the
            largest projected mesh span when ``fit_to_view`` is enabled.
        fit_to_view_reference: Optional reference vertices with the same shape
            as ``vertices``. When provided, ``fit_to_view`` computes its center
            and scale from this reference but applies that transform to
            ``vertices``.
        fit_visible_component: Render a first pass, fit the largest connected
            foreground component to ``view_occupancy``, then render again.
        foreground_threshold: RGB distance from the background used to detect
            foreground pixels for ``fit_visible_component``.

    Returns:
        RGB image tensor in ``[H, W, 3]`` for unbatched input or
        ``[B, H, W, 3]`` for batched input.
    """
    if vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape [V, 3] or [B, V, 3].")
    if faces.shape[-1] != 3:
        raise ValueError("faces must have shape [F, 3] or [B, F, 3].")
    backend = backend.lower()
    if backend not in {"torch", "nvdiffrast", "auto"}:
        raise ValueError("backend must be one of: 'torch', 'nvdiffrast', 'auto'.")

    was_unbatched = vertices.dim() == 2
    verts = _ensure_batched(vertices, "vertices")
    batch_size = verts.shape[0]
    faces_batched = _ensure_faces_batched(faces, batch_size, verts.device)
    fit_reference = _prepare_fit_to_view_reference(
        fit_to_view_reference,
        verts,
    )

    height, width = _normalize_image_size(image_size)
    _validate_view_occupancy(view_occupancy)
    if _should_use_nvdiffrast(backend, verts.device):
        rendered = _nvdiffrast_render(
            vertices=verts,
            faces=faces_batched,
            texture=texture,
            image_size=(height, width),
            background_color=background_color,
            antialias=antialias,
            fit_to_view=fit_to_view,
            view_occupancy=view_occupancy,
            fit_to_view_reference=fit_reference,
            fit_visible_component=fit_visible_component,
            foreground_threshold=foreground_threshold,
        )
        if rendered is not None:
            if was_unbatched:
                return rendered[0]
            return rendered

    fx, fy, px, py = _scaled_intrinsics(height, width, verts.device, verts.dtype)
    camera_vertices = _world_to_camera(verts)
    projected = _project_to_screen(camera_vertices, fx, fy, px, py)
    reference_projected = None
    if fit_reference is not None:
        reference_camera_vertices = _world_to_camera(fit_reference)
        reference_projected = _project_to_screen(
            reference_camera_vertices,
            fx,
            fy,
            px,
            py,
        )
    if fit_to_view:
        projected = _fit_projected_to_view(
            projected,
            image_size=(height, width),
            occupancy=view_occupancy,
            reference_vertices=reference_projected,
        )

    barycentric_samples = _barycentric_sample_pattern(
        torch_samples_per_face,
        verts.device,
        verts.dtype,
    )
    sample_positions, sample_depths = _face_samples(
        projected,
        faces_batched,
        barycentric_samples,
    )
    sample_colors = _sample_texture_or_colors(
        faces=faces_batched,
        texture=texture,
        batch_size=batch_size,
        device=verts.device,
        dtype=verts.dtype,
        barycentric_samples=barycentric_samples,
    )

    background = torch.tensor(
        background_color,
        device=verts.device,
        dtype=verts.dtype,
    ).view(1, 1, 3)

    rendered = _soft_splat(
        sample_positions=sample_positions,
        sample_depths=sample_depths,
        sample_colors=sample_colors,
        image_size=(height, width),
        background=background,
        splat_radius_px=splat_radius_px,
        splat_kernel_radius=splat_kernel_radius,
        depth_sharpness=depth_sharpness,
        chunk_size=chunk_size,
    )

    if fit_visible_component:
        sample_positions = _fit_sample_positions_to_visible_component(
            sample_positions=sample_positions,
            preview=rendered,
            background=background,
            image_size=(height, width),
            occupancy=view_occupancy,
            foreground_threshold=foreground_threshold,
        )
        rendered = _soft_splat(
            sample_positions=sample_positions,
            sample_depths=sample_depths,
            sample_colors=sample_colors,
            image_size=(height, width),
            background=background,
            splat_radius_px=splat_radius_px,
            splat_kernel_radius=splat_kernel_radius,
            depth_sharpness=depth_sharpness,
            chunk_size=chunk_size,
        )

    if was_unbatched:
        return rendered[0]
    return rendered




def _should_use_nvdiffrast(backend: str, device: torch.device) -> bool:
    if backend == "torch":
        return False
    if dr is None:
        return False
    if device.type != "cuda":
        if backend == "nvdiffrast":
            print("[WARNING] nvdiffrast requires CUDA tensors; falling back to torch")
        return False
    return True


def _nvdiffrast_render(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    texture: Optional[TensorMapping],
    image_size: Tuple[int, int],
    background_color: Sequence[float],
    antialias: bool,
    fit_to_view: bool,
    view_occupancy: float,
    fit_to_view_reference: Optional[torch.Tensor],
    fit_visible_component: bool,
    foreground_threshold: float,
) -> Optional[torch.Tensor]:
    if dr is None or vertices.device.type != "cuda":
        return None
    if faces.shape[0] > 1 and not bool((faces == faces[:1]).all()):
        print(
            "[WARNING] nvdiffrast renderer only supports shared batched faces; "
            "falling back to torch"
        )
        return None

    height, width = image_size
    tri = faces[0].to(device=vertices.device, dtype=torch.int32).contiguous()
    screen_vertices = _camera_vertices_to_screen(
        vertices,
        height,
        width,
        fit_to_view=fit_to_view,
        view_occupancy=view_occupancy,
        fit_to_view_reference=fit_to_view_reference,
    )
    ctx = _get_nvdiffrast_context(vertices.device)
    background = torch.tensor(
        background_color,
        device=vertices.device,
        dtype=vertices.dtype,
    ).view(1, 1, 1, 3)

    if fit_visible_component:
        with torch.no_grad():
            preview = _nvdiffrast_render_from_screen(
                ctx=ctx,
                screen_vertices=screen_vertices,
                tri=tri,
                faces=faces,
                texture=texture,
                image_size=image_size,
                background=background,
                antialias=antialias,
            )
        screen_vertices = _fit_screen_vertices_to_visible_component(
            screen_vertices=screen_vertices,
            preview=preview,
            background=background,
            image_size=image_size,
            occupancy=view_occupancy,
            foreground_threshold=foreground_threshold,
        )

    return _nvdiffrast_render_from_screen(
        ctx=ctx,
        screen_vertices=screen_vertices,
        tri=tri,
        faces=faces,
        texture=texture,
        image_size=image_size,
        background=background,
        antialias=antialias,
    )


def _nvdiffrast_render_from_screen(
    ctx,
    screen_vertices: torch.Tensor,
    tri: torch.Tensor,
    faces: torch.Tensor,
    texture: Optional[TensorMapping],
    image_size: Tuple[int, int],
    background: torch.Tensor,
    antialias: bool,
) -> torch.Tensor:
    height, width = image_size
    clip_positions = _screen_vertices_to_clip(screen_vertices, height, width)
    rast, _ = dr.rasterize(ctx, clip_positions.contiguous(), tri, (height, width))
    color = _nvdiffrast_shade(
        rast=rast,
        tri=tri,
        faces=faces,
        texture=texture,
        batch_size=screen_vertices.shape[0],
        device=screen_vertices.device,
        dtype=screen_vertices.dtype,
    )

    mask = (rast[..., 3:4] > 0).to(dtype=screen_vertices.dtype)
    image = color * mask + background * (1.0 - mask)

    if antialias:
        image = dr.antialias(image.contiguous(), rast, clip_positions, tri)
    # nvdiffrast returns CUDA raster rows bottom-up; this module exposes images
    # with row 0 at the top, matching the pure-Torch renderer and PIL.
    image = torch.flip(image, dims=(1,))
    return image.clamp(0.0, 1.0)


def _get_nvdiffrast_context(device: torch.device):
    key = device.index if device.index is not None else torch.cuda.current_device()
    if key not in _NVDIFFRAST_CONTEXTS:
        with torch.cuda.device(device):
            _NVDIFFRAST_CONTEXTS[key] = dr.RasterizeCudaContext()
    return _NVDIFFRAST_CONTEXTS[key]


def _camera_vertices_to_clip(
    vertices: torch.Tensor,
    height: int,
    width: int,
    fit_to_view: bool,
    view_occupancy: float,
    fit_to_view_reference: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    screen_vertices = _camera_vertices_to_screen(
        vertices,
        height,
        width,
        fit_to_view=fit_to_view,
        view_occupancy=view_occupancy,
        fit_to_view_reference=fit_to_view_reference,
    )
    return _screen_vertices_to_clip(screen_vertices, height, width)


def _camera_vertices_to_screen(
    vertices: torch.Tensor,
    height: int,
    width: int,
    fit_to_view: bool,
    view_occupancy: float,
    fit_to_view_reference: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    fx, fy, px, py = _scaled_intrinsics(height, width, vertices.device, vertices.dtype)
    camera_vertices = _world_to_camera(vertices)
    screen_vertices = _project_to_screen(camera_vertices, fx, fy, px, py)
    reference_screen_vertices = None
    if fit_to_view_reference is not None:
        reference_camera_vertices = _world_to_camera(fit_to_view_reference)
        reference_screen_vertices = _project_to_screen(
            reference_camera_vertices,
            fx,
            fy,
            px,
            py,
        )
    if fit_to_view:
        screen_vertices = _fit_projected_to_view(
            screen_vertices,
            image_size=(height, width),
            occupancy=view_occupancy,
            reference_vertices=reference_screen_vertices,
        )
    return screen_vertices


def _screen_vertices_to_clip(
    screen_vertices: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    depth = screen_vertices[..., 2].clamp_min(1.0e-6)
    x_ndc = screen_vertices[..., 0] / max(width - 1, 1) * 2.0 - 1.0
    y_ndc = 1.0 - screen_vertices[..., 1] / max(height - 1, 1) * 2.0

    near, far = _clip_depth_bounds(depth)
    z_ndc = (depth - near) / (far - near) * 2.0 - 1.0

    return torch.stack(
        (
            x_ndc * depth,
            y_ndc * depth,
            z_ndc * depth,
            depth,
        ),
        dim=-1,
    )


def _clip_depth_bounds(depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # nvdiffrast clips by clip-space Z, so expand the range for farther cameras
    # while keeping the old defaults for nearby meshes.
    depth_for_bounds = depth.detach().clamp_min(1.0e-6)
    min_depth = depth_for_bounds.amin(dim=1, keepdim=True)
    max_depth = depth_for_bounds.amax(dim=1, keepdim=True)

    default_near = depth.new_tensor(0.01)
    default_far = depth.new_tensor(10.0)
    near = torch.minimum(default_near, min_depth * 0.5).clamp_min(1.0e-6)
    far_margin = (max_depth - near).clamp_min(1.0e-6) * 0.05
    far = torch.maximum(default_far, max_depth + far_margin)
    far = torch.maximum(far, near + 1.0e-6)
    return near, far


def _nvdiffrast_shade(
    rast: torch.Tensor,
    tri: torch.Tensor,
    faces: torch.Tensor,
    texture: Optional[TensorMapping],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if texture is None:
        return torch.ones(
            batch_size,
            rast.shape[1],
            rast.shape[2],
            3,
            device=device,
            dtype=dtype,
        )

    if "texture_image" in texture or "maps" in texture:
        maps = texture.get("texture_image", texture.get("maps"))
        verts_uvs = texture.get("verts_uvs")
        if maps is None or verts_uvs is None:
            raise ValueError("UV texture rendering requires maps and verts_uvs.")

        faces_uvs = texture.get("faces_uvs", faces)
        maps = _prepare_texture_maps(maps, batch_size, device, dtype)
        verts_uvs = _prepare_uvs(verts_uvs, batch_size, device, dtype).contiguous()
        faces_uvs = _ensure_faces_batched(faces_uvs, batch_size, device)
        if faces_uvs.shape[0] > 1 and not bool((faces_uvs == faces_uvs[:1]).all()):
            print(
                "[WARNING] nvdiffrast renderer only supports shared batched UV "
                "faces; falling back to mesh faces for UV interpolation"
            )
            uv_tri = tri
        else:
            uv_tri = faces_uvs[0].to(device=device, dtype=torch.int32).contiguous()

        uv, _ = dr.interpolate(verts_uvs, rast, uv_tri)
        grid = uv.mul(2.0).sub(1.0)
        sampled = F.grid_sample(
            maps,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled.permute(0, 2, 3, 1)
        if sampled.shape[-1] == 4:
            alpha = sampled[..., 3:4].clamp(0.0, 1.0)
            sampled = sampled[..., :3] * alpha + (1.0 - alpha)
        return sampled[..., :3]

    if "vertex_colors" in texture:
        vertex_colors = _ensure_batched(texture["vertex_colors"], "vertex_colors")
        vertex_colors = vertex_colors.to(device=device, dtype=dtype).contiguous()
        if vertex_colors.shape[0] == 1 and batch_size > 1:
            vertex_colors = vertex_colors.expand(batch_size, -1, -1).contiguous()
        color, _ = dr.interpolate(vertex_colors, rast, tri)
        return color[..., :3]

    raise ValueError(
        "texture must contain texture_image/maps + verts_uvs, or vertex_colors."
    )


def _ensure_batched(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    if tensor.dim() == 3:
        return tensor
    raise ValueError(f"{name} must have shape [N, C] or [B, N, C].")


def _prepare_fit_to_view_reference(
    reference_vertices: Optional[torch.Tensor],
    vertices: torch.Tensor,
) -> Optional[torch.Tensor]:
    if reference_vertices is None:
        return None
    reference = _ensure_batched(reference_vertices, "fit_to_view_reference").to(
        device=vertices.device,
        dtype=vertices.dtype,
    )
    if reference.shape[0] == 1 and vertices.shape[0] > 1:
        reference = reference.expand(vertices.shape[0], -1, -1)
    if reference.shape != vertices.shape:
        raise ValueError(
            "fit_to_view_reference must match vertices shape after batching."
        )
    return reference


def _ensure_faces_batched(
    faces: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    faces = faces.to(device=device, dtype=torch.long)
    if faces.dim() == 2:
        return faces.unsqueeze(0).expand(batch_size, -1, -1)
    if faces.dim() == 3:
        if faces.shape[0] != batch_size:
            raise ValueError("batched faces must match the vertices batch size.")
        return faces
    raise ValueError("faces must have shape [F, 3] or [B, F, 3].")


def _normalize_image_size(image_size: Optional[ImageSize]) -> Tuple[int, int]:
    if image_size is None:
        image_size = CAMERA_INTRINSICS["image_size"]
    if isinstance(image_size, int):
        return image_size, image_size
    if len(image_size) != 2:
        raise ValueError("image_size must be an int or a (height, width) tuple.")
    return int(image_size[0]), int(image_size[1])


def _validate_view_occupancy(view_occupancy: float) -> None:
    if not 0.0 < float(view_occupancy) <= 1.0:
        raise ValueError("view_occupancy must be in the range (0, 1].")


def _scaled_intrinsics(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base_height, base_width = CAMERA_INTRINSICS["image_size"]
    base_fx, base_fy = CAMERA_INTRINSICS["focal_length"]
    base_px, base_py = CAMERA_INTRINSICS["principal_point"]

    sx = width / float(base_width)
    sy = height / float(base_height)
    values = (
        base_fx * sx,
        base_fy * sy,
        base_px * sx,
        base_py * sy,
    )
    return tuple(torch.tensor(v, device=device, dtype=dtype) for v in values)


def _world_to_camera(vertices: torch.Tensor) -> torch.Tensor:
    batch_size = vertices.shape[0]
    r = torch.tensor(
        CAMERA_EXTRINSICS["R"],
        device=vertices.device,
        dtype=vertices.dtype,
    ).unsqueeze(0)
    t = torch.tensor(
        CAMERA_EXTRINSICS["T"],
        device=vertices.device,
        dtype=vertices.dtype,
    ).view(1, 1, 3)
    return torch.bmm(vertices, r.expand(batch_size, -1, -1)) + t


def _project_to_screen(
    camera_vertices: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    px: torch.Tensor,
    py: torch.Tensor,
) -> torch.Tensor:
    z = camera_vertices[..., 2].clamp_min(1.0e-6)
    x = px - fx * camera_vertices[..., 0] / z
    y = py - fy * camera_vertices[..., 1] / z
    return torch.stack((x, y, z), dim=-1)


def _fit_projected_to_view(
    projected_vertices: torch.Tensor,
    image_size: Tuple[int, int],
    occupancy: float,
    reference_vertices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    height, width = image_size
    xy = projected_vertices[..., :2]
    fit_vertices = (
        projected_vertices if reference_vertices is None else reference_vertices
    )
    if fit_vertices.shape != projected_vertices.shape:
        raise ValueError("reference_vertices must match projected_vertices shape.")
    fit_xy = fit_vertices[..., :2]
    mins = fit_xy.amin(dim=1, keepdim=True)
    maxs = fit_xy.amax(dim=1, keepdim=True)
    center = (mins + maxs) * 0.5
    spans = maxs - mins
    max_span = spans.amax(dim=-1, keepdim=True).clamp_min(1.0e-6)
    target_span = min(height, width) * float(occupancy)
    scale = projected_vertices.new_tensor(target_span) / max_span
    image_center = projected_vertices.new_tensor(
        ((width - 1) * 0.5, (height - 1) * 0.5)
    ).view(1, 1, 2)
    fitted_xy = (xy - center) * scale + image_center
    return torch.cat((fitted_xy, projected_vertices[..., 2:]), dim=-1)


def _fit_sample_positions_to_visible_component(
    sample_positions: torch.Tensor,
    preview: torch.Tensor,
    background: torch.Tensor,
    image_size: Tuple[int, int],
    occupancy: float,
    foreground_threshold: float,
) -> torch.Tensor:
    if foreground_threshold < 0.0:
        raise ValueError("foreground_threshold must be non-negative.")

    return _fit_positions_to_visible_component(
        positions=sample_positions,
        preview=preview,
        background=background,
        image_size=image_size,
        occupancy=occupancy,
        foreground_threshold=foreground_threshold,
    )


def _fit_screen_vertices_to_visible_component(
    screen_vertices: torch.Tensor,
    preview: torch.Tensor,
    background: torch.Tensor,
    image_size: Tuple[int, int],
    occupancy: float,
    foreground_threshold: float,
) -> torch.Tensor:
    fitted_xy = _fit_positions_to_visible_component(
        positions=screen_vertices[..., :2],
        preview=preview,
        background=background,
        image_size=image_size,
        occupancy=occupancy,
        foreground_threshold=foreground_threshold,
    )
    return torch.cat((fitted_xy, screen_vertices[..., 2:]), dim=-1)


def _fit_positions_to_visible_component(
    positions: torch.Tensor,
    preview: torch.Tensor,
    background: torch.Tensor,
    image_size: Tuple[int, int],
    occupancy: float,
    foreground_threshold: float,
) -> torch.Tensor:
    if foreground_threshold < 0.0:
        raise ValueError("foreground_threshold must be non-negative.")

    height, width = image_size
    background = background.view(1, 1, 1, 3)
    foreground = (preview.detach() - background).abs().amax(dim=-1)
    foreground = foreground > float(foreground_threshold)

    centers = []
    scales = []
    target_span = min(height, width) * float(occupancy)
    image_center = positions.new_tensor(
        ((width - 1) * 0.5, (height - 1) * 0.5)
    )

    for batch_idx in range(positions.shape[0]):
        bbox = _largest_mask_component_bbox(foreground[batch_idx])
        if bbox is None:
            centers.append(image_center)
            scales.append(positions.new_tensor(1.0))
            continue

        x_min, y_min, x_max, y_max = bbox
        bbox_min = positions.new_tensor((x_min, y_min))
        bbox_max = positions.new_tensor((x_max, y_max))
        center = (bbox_min + bbox_max) * 0.5
        span = (bbox_max - bbox_min + 1.0).amax().clamp_min(1.0e-6)

        centers.append(center)
        scales.append(positions.new_tensor(target_span) / span)

    center_tensor = torch.stack(centers, dim=0).view(-1, 1, 2)
    scale_tensor = torch.stack(scales, dim=0).view(-1, 1, 1)
    return (positions - center_tensor) * scale_tensor + image_center.view(
        1,
        1,
        2,
    )


def _largest_mask_component_bbox(
    mask: torch.Tensor,
) -> Optional[Tuple[int, int, int, int]]:
    try:
        import numpy as np
    except ImportError:
        return _largest_mask_component_bbox_torch(mask)

    mask_np = mask.detach().cpu().numpy().astype(bool, copy=False)
    height, width = mask_np.shape
    visited = np.zeros_like(mask_np, dtype=bool)
    best_area = 0
    best_bbox: Optional[Tuple[int, int, int, int]] = None

    ys, xs = np.nonzero(mask_np)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue

        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        area = 0
        x_min = x_max = start_x
        y_min = y_max = start_y

        while stack:
            y, x = stack.pop()
            area += 1
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)

            for next_y, next_x in (
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1),
            ):
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and mask_np[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    stack.append((next_y, next_x))

        if area > best_area:
            best_area = area
            best_bbox = (x_min, y_min, x_max, y_max)

    return best_bbox


def _largest_mask_component_bbox_torch(
    mask: torch.Tensor,
) -> Optional[Tuple[int, int, int, int]]:
    mask_cpu = mask.detach().cpu().to(dtype=torch.bool)
    height, width = mask_cpu.shape
    visited = torch.zeros_like(mask_cpu, dtype=torch.bool)
    best_area = 0
    best_bbox: Optional[Tuple[int, int, int, int]] = None

    for start_y, start_x in torch.nonzero(mask_cpu, as_tuple=False).tolist():
        if bool(visited[start_y, start_x]):
            continue

        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        area = 0
        x_min = x_max = start_x
        y_min = y_max = start_y

        while stack:
            y, x = stack.pop()
            area += 1
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)

            for next_y, next_x in (
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1),
            ):
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and bool(mask_cpu[next_y, next_x])
                    and not bool(visited[next_y, next_x])
                ):
                    visited[next_y, next_x] = True
                    stack.append((next_y, next_x))

        if area > best_area:
            best_area = area
            best_bbox = (x_min, y_min, x_max, y_max)

    return best_bbox


def _gather_faces(values: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    batch_size, _, channels = values.shape
    flat_faces = faces.reshape(batch_size, -1)
    gathered = torch.gather(
        values,
        dim=1,
        index=flat_faces.unsqueeze(-1).expand(-1, -1, channels),
    )
    return gathered.reshape(batch_size, faces.shape[1], faces.shape[2], channels)


def _barycentric_sample_pattern(
    sample_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if sample_count < 1:
        raise ValueError("torch_samples_per_face must be at least 1.")

    pattern = torch.tensor(
        (
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            (0.55, 0.225, 0.225),
            (0.225, 0.55, 0.225),
            (0.225, 0.225, 0.55),
            (0.45, 0.45, 0.10),
            (0.45, 0.10, 0.45),
            (0.10, 0.45, 0.45),
        ),
        device=device,
        dtype=dtype,
    )
    if sample_count <= pattern.shape[0]:
        return pattern[:sample_count]

    repeat_count = (sample_count + pattern.shape[0] - 1) // pattern.shape[0]
    return pattern.repeat(repeat_count, 1)[:sample_count]


def _face_samples(
    projected_vertices: torch.Tensor,
    faces: torch.Tensor,
    barycentric_samples: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    projected_faces = _gather_faces(projected_vertices, faces)
    samples = _interpolate_face_samples(projected_faces, barycentric_samples)
    batch_size, face_count, sample_count, _ = samples.shape
    samples = samples.reshape(batch_size, face_count * sample_count, -1)
    return samples[..., :2], samples[..., 2]


def _interpolate_face_samples(
    face_values: torch.Tensor,
    barycentric_samples: torch.Tensor,
) -> torch.Tensor:
    weights = barycentric_samples.view(1, 1, -1, 3, 1)
    return (face_values.unsqueeze(2) * weights).sum(dim=3)


def _sample_texture_or_colors(
    faces: torch.Tensor,
    texture: Optional[TensorMapping],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    barycentric_samples: torch.Tensor,
) -> torch.Tensor:
    sample_count = barycentric_samples.shape[0]
    if texture is None:
        return torch.ones(
            batch_size,
            faces.shape[1] * sample_count,
            3,
            device=device,
            dtype=dtype,
        )

    if "texture_image" in texture or "maps" in texture:
        maps = texture.get("texture_image", texture.get("maps"))
        verts_uvs = texture.get("verts_uvs")
        if maps is None or verts_uvs is None:
            raise ValueError("UV texture rendering requires maps and verts_uvs.")

        faces_uvs = texture.get("faces_uvs", faces)
        maps = _prepare_texture_maps(maps, batch_size, device, dtype)
        verts_uvs = _prepare_uvs(verts_uvs, batch_size, device, dtype)
        faces_uvs = _ensure_faces_batched(faces_uvs, batch_size, device)
        face_uvs = _gather_faces(verts_uvs, faces_uvs)
        sample_uvs = _interpolate_face_samples(face_uvs, barycentric_samples)
        sample_uvs = sample_uvs.reshape(batch_size, -1, 2)

        grid = sample_uvs.mul(2.0).sub(1.0).view(batch_size, -1, 1, 2)
        sampled = F.grid_sample(
            maps,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled.squeeze(-1).permute(0, 2, 1)
        if sampled.shape[-1] == 4:
            alpha = sampled[..., 3:4].clamp(0.0, 1.0)
            sampled = sampled[..., :3] * alpha + (1.0 - alpha)
        return sampled[..., :3]

    if "vertex_colors" in texture:
        vertex_colors = _ensure_batched(texture["vertex_colors"], "vertex_colors")
        vertex_colors = vertex_colors.to(device=device, dtype=dtype)
        if vertex_colors.shape[0] == 1 and batch_size > 1:
            vertex_colors = vertex_colors.expand(batch_size, -1, -1)
        face_colors = _gather_faces(vertex_colors, faces)
        sample_colors = _interpolate_face_samples(face_colors, barycentric_samples)
        return sample_colors.reshape(batch_size, -1, sample_colors.shape[-1])[..., :3]

    raise ValueError(
        "texture must contain texture_image/maps + verts_uvs, or vertex_colors."
    )


def _prepare_texture_maps(
    maps: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if maps.dim() == 3:
        maps = maps.unsqueeze(0)
    if maps.dim() != 4:
        raise ValueError("texture maps must have shape [H, W, C] or [B, H, W, C].")

    maps = maps.to(device=device)
    if not maps.dtype.is_floating_point:
        maps = maps.to(dtype=dtype) / 255.0
    else:
        maps = maps.to(dtype=dtype)

    if maps.shape[-1] in (3, 4):
        maps = maps.permute(0, 3, 1, 2)
    elif maps.shape[1] not in (3, 4):
        raise ValueError("texture maps must have 3 RGB or 4 RGBA channels.")

    if maps.shape[0] == 1 and batch_size > 1:
        maps = maps.expand(batch_size, -1, -1, -1)
    if maps.shape[0] != batch_size:
        raise ValueError("texture map batch size must be 1 or match vertices.")
    return maps


def _prepare_uvs(
    verts_uvs: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    verts_uvs = _ensure_batched(verts_uvs, "verts_uvs").to(device=device, dtype=dtype)
    if verts_uvs.shape[-1] != 2:
        raise ValueError("verts_uvs must have shape [V, 2] or [B, V, 2].")
    if verts_uvs.shape[0] == 1 and batch_size > 1:
        verts_uvs = verts_uvs.expand(batch_size, -1, -1)
    if verts_uvs.shape[0] != batch_size:
        raise ValueError("verts_uvs batch size must be 1 or match vertices.")
    return verts_uvs


def _soft_splat(
    sample_positions: torch.Tensor,
    sample_depths: torch.Tensor,
    sample_colors: torch.Tensor,
    image_size: Tuple[int, int],
    background: torch.Tensor,
    splat_radius_px: float,
    splat_kernel_radius: int,
    depth_sharpness: float,
    chunk_size: int,
) -> torch.Tensor:
    height, width = image_size
    batch_size, sample_count, channels = sample_colors.shape
    if channels != 3:
        raise ValueError("sample_colors must have exactly 3 channels.")

    accum = sample_colors.new_zeros(batch_size, height * width, 3)
    weights = sample_colors.new_zeros(batch_size, height * width, 1)
    offsets = _kernel_offsets(
        splat_kernel_radius,
        device=sample_colors.device,
        dtype=sample_positions.dtype,
    )

    for batch_idx in range(batch_size):
        positions_b = sample_positions[batch_idx]
        depths_b = sample_depths[batch_idx]
        colors_b = sample_colors[batch_idx]

        base_valid = (
            (depths_b > 1.0e-6)
            & (positions_b[:, 0] > -splat_kernel_radius - 1)
            & (positions_b[:, 0] < width + splat_kernel_radius)
            & (positions_b[:, 1] > -splat_kernel_radius - 1)
            & (positions_b[:, 1] < height + splat_kernel_radius)
        )
        if not bool(base_valid.any()):
            continue

        depth_buffer = sample_depths.new_full((height * width,), float("inf"))
        for start in range(0, sample_count, chunk_size):
            end = min(start + chunk_size, sample_count)
            valid = base_valid[start:end]
            if not bool(valid.any()):
                continue

            pos = positions_b[start:end][valid]
            depth = depths_b[start:end][valid]

            splat = _splat_indices_and_weights(
                pos,
                depth,
                offsets,
                height,
                width,
                splat_radius_px,
            )
            if splat is None:
                continue
            flat_index, _, depth_values, _ = splat
            depth_buffer.scatter_reduce_(
                0,
                flat_index,
                depth_values.detach(),
                reduce="amin",
                include_self=True,
            )

        for start in range(0, sample_count, chunk_size):
            end = min(start + chunk_size, sample_count)
            valid = base_valid[start:end]
            if not bool(valid.any()):
                continue

            pos = positions_b[start:end][valid]
            depth = depths_b[start:end][valid]
            color = colors_b[start:end][valid].clamp(0.0, 1.0)

            splat = _splat_indices_and_weights(
                pos,
                depth,
                offsets,
                height,
                width,
                splat_radius_px,
            )
            if splat is None:
                continue
            flat_index, spatial_weight, depth_values, inside = splat

            nearest_depth = depth_buffer[flat_index]
            depth_delta = (depth_values - nearest_depth).clamp_min(0.0)
            splat_weight = spatial_weight * torch.exp(-depth_delta * depth_sharpness)
            color_values = color[:, None, :].expand(-1, offsets.shape[0], -1)[inside]

            accum[batch_idx].index_add_(
                0,
                flat_index,
                color_values * splat_weight.unsqueeze(-1),
            )
            weights[batch_idx].index_add_(0, flat_index, splat_weight.unsqueeze(-1))

    image = accum / weights.clamp_min(1.0e-8)
    background = background.expand(batch_size, height * width, -1)
    image = torch.where(weights > 1.0e-8, image, background)
    return image.view(batch_size, height, width, 3).clamp(0.0, 1.0)


def _kernel_offsets(
    radius: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if radius < 0:
        raise ValueError("splat_kernel_radius must be non-negative.")
    values = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def _splat_indices_and_weights(
    positions: torch.Tensor,
    depths: torch.Tensor,
    offsets: torch.Tensor,
    height: int,
    width: int,
    splat_radius_px: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    base = positions.floor()
    pixel = base[:, None, :] + offsets[None, :, :]
    px = pixel[..., 0]
    py = pixel[..., 1]
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if not bool(inside.any()):
        return None

    dist2 = (pixel[..., 0] - positions[:, None, 0]).square() + (
        pixel[..., 1] - positions[:, None, 1]
    ).square()
    spatial_weight = torch.exp(
        -dist2 / (2.0 * max(splat_radius_px, 1.0e-6) ** 2)
    )[inside]
    flat_index = (
        py.long().clamp(0, height - 1) * width
        + px.long().clamp(0, width - 1)
    )[inside]
    depth_values = depths[:, None].expand(-1, offsets.shape[0])[inside]
    return flat_index, spatial_weight, depth_values, inside
