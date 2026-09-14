from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F


ChannelOrder = Literal["auto", "chw", "hwc"]


def compute_displacement_2d(
    img_1: torch.Tensor,
    img_2: torch.Tensor,
    *,
    num_levels: int = 4,
    num_iterations: int = 3,
    window_size: int = 31,
    regularization: float = 1.0e-4,
    smooth_flow_sigma: Optional[float] = 1.0,
    max_update_px: Optional[float] = 4.0,
    apply_motion_mask: bool = True,
    apply_valid_mask: bool = True,
    apply_valid_mask_to_flow: bool = False,
    return_mask: bool = False,
    return_all_masks: bool = False,
    mask_quantile: float = 0.98,
    mask_min_threshold: float = 0.025,
    mask_softness: float = 0.003,
    mask_evidence_sigma: Optional[float] = 5.0,
    mask_dilate_px: int = 31,
    mask_blur_sigma: Optional[float] = 4.0,
    valid_consistency_px: float = 1.5,
    valid_relative_threshold: float = 0.05,
    valid_consistency_softness: float = 0.5,
    valid_photometric_threshold: float = 0.08,
    valid_photometric_softness: float = 0.02,
    channel_order: ChannelOrder = "auto",
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
]:
    """Estimate dense 2D displacement from ``img_1`` to ``img_2``.

    The output is an optical-flow-like field in input-image pixel units. Channel
    0 stores horizontal displacement ``dx`` and channel 1 stores vertical
    displacement ``dy``. This follows the image-pair version of the 2D
    displacement supervision described in RigAnyFace, where pixel offsets are
    predicted between neutral and posed renderings. By default, the dense flow
    is multiplied by a soft motion-evidence mask so it stays localized to the
    expression area. A separate soft correspondence validity mask is used for
    the returned loss mask, suppressing occluded or newly visible regions such
    as the inside of an opening mouth without erasing the broader jaw motion
    field from debug visualizations.

    All numeric operations are standard PyTorch tensor ops, so gradients can
    flow through the estimator to floating-point input images.

    Args:
        img_1: Neutral/source image with shape ``[H, W]``, ``[C, H, W]``,
            ``[H, W, C]``, ``[B, C, H, W]``, or ``[B, H, W, C]``.
        img_2: Posed/target image with the same shape convention as ``img_1``.
        num_levels: Number of pyramid levels used for coarse-to-fine flow.
        num_iterations: Lucas-Kanade refinement iterations per pyramid level.
        window_size: Odd local window size for the weighted least-squares fit.
        regularization: Diagonal Tikhonov term for low-texture regions.
        smooth_flow_sigma: Optional Gaussian smoothing sigma applied after each
            refinement step. Set to ``None`` or ``0`` to disable.
        max_update_px: Optional clamp for each per-iteration update in pixels.
        apply_motion_mask: Suppress displacement outside regions with strong
            image-pair evidence of expression change.
        apply_valid_mask: Suppress loss weight in regions that fail
            forward-backward or photometric consistency checks.
        apply_valid_mask_to_flow: Also multiply the returned flow by the
            validity mask. Leave this disabled when the validity mask should
            only weight the downstream loss.
        return_mask: Return ``(flow, loss_mask)``. The mask has one channel and
            should also be used to weight any downstream displacement loss.
        return_all_masks: Return ``(flow, loss_mask, masks)`` where ``masks``
            includes ``motion``, ``valid``, and ``loss`` masks.
        mask_quantile: Per-image robust quantile used as the soft mask threshold
            when it is above ``mask_min_threshold``.
        mask_min_threshold: Minimum absolute RGB difference needed before a
            region can receive a high mask value. Input images are expected to
            be in roughly ``[0, 1]``.
        mask_softness: Sigmoid softness around the mask threshold.
        mask_evidence_sigma: Optional Gaussian smoothing sigma applied to
            image-difference evidence before thresholding. This favors dense
            expression changes over isolated texture or shadow pixels.
        mask_dilate_px: Odd max-pool diameter for soft mask dilation. Set to
            ``0`` or ``1`` to disable.
        mask_blur_sigma: Optional Gaussian blur applied after dilation.
        valid_consistency_px: Base forward-backward consistency tolerance in
            pixels.
        valid_relative_threshold: Extra consistency tolerance proportional to
            the local forward/backward flow magnitude.
        valid_consistency_softness: Sigmoid softness around the consistency
            threshold, in pixels.
        valid_photometric_threshold: RGB reconstruction residual threshold for
            invalidating newly visible or non-corresponding pixels.
        valid_photometric_softness: Sigmoid softness around the photometric
            threshold.
        channel_order: Explicit tensor layout for 3D/4D images, or ``"auto"``.

    Returns:
        ``[2, H, W]`` for unbatched inputs, otherwise ``[B, 2, H, W]``. If
        ``return_mask=True``, returns ``(flow, loss_mask)`` where the mask is
        ``[1, H, W]`` or ``[B, 1, H, W]``.
    """
    if num_levels < 1:
        raise ValueError("num_levels must be >= 1.")
    if num_iterations < 1:
        raise ValueError("num_iterations must be >= 1.")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3.")
    if regularization <= 0.0:
        raise ValueError("regularization must be positive.")
    if not 0.0 < mask_quantile < 1.0:
        raise ValueError("mask_quantile must be between 0 and 1.")
    if mask_min_threshold < 0.0:
        raise ValueError("mask_min_threshold must be non-negative.")
    if mask_softness <= 0.0:
        raise ValueError("mask_softness must be positive.")
    if valid_consistency_px < 0.0:
        raise ValueError("valid_consistency_px must be non-negative.")
    if valid_relative_threshold < 0.0:
        raise ValueError("valid_relative_threshold must be non-negative.")
    if valid_consistency_softness <= 0.0:
        raise ValueError("valid_consistency_softness must be positive.")
    if valid_photometric_threshold < 0.0:
        raise ValueError("valid_photometric_threshold must be non-negative.")
    if valid_photometric_softness <= 0.0:
        raise ValueError("valid_photometric_softness must be positive.")

    img_1_bchw, img_1_was_unbatched = _to_bchw(img_1, "img_1", channel_order)
    img_2_bchw, img_2_was_unbatched = _to_bchw(img_2, "img_2", channel_order)
    if img_1_bchw.shape != img_2_bchw.shape:
        raise ValueError(
            "img_1 and img_2 must have matching normalized shapes; got "
            f"{tuple(img_1_bchw.shape)} and {tuple(img_2_bchw.shape)}."
        )
    if img_1_was_unbatched != img_2_was_unbatched:
        raise ValueError("img_1 and img_2 must either both be batched or unbatched.")

    img_1_gray = _to_grayscale(img_1_bchw)
    img_2_gray = _to_grayscale(img_2_bchw)
    flow = _estimate_dense_flow(
        img_1_gray,
        img_2_gray,
        num_levels=num_levels,
        num_iterations=num_iterations,
        window_size=window_size,
        regularization=regularization,
        smooth_flow_sigma=smooth_flow_sigma,
        max_update_px=max_update_px,
    )

    motion_mask: Optional[torch.Tensor] = None
    valid_mask: Optional[torch.Tensor] = None
    loss_mask = flow.new_ones((flow.shape[0], 1, flow.shape[2], flow.shape[3]))
    if apply_motion_mask or return_mask or return_all_masks:
        motion_mask = _motion_evidence_mask(
            img_1_bchw,
            img_2_bchw,
            quantile=mask_quantile,
            min_threshold=mask_min_threshold,
            softness=mask_softness,
            evidence_sigma=mask_evidence_sigma,
            dilate_px=mask_dilate_px,
            blur_sigma=mask_blur_sigma,
        )
    if apply_motion_mask:
        assert motion_mask is not None
        loss_mask = loss_mask * motion_mask

    if apply_valid_mask or return_mask or return_all_masks:
        backward_flow = _estimate_dense_flow(
            img_2_gray,
            img_1_gray,
            num_levels=num_levels,
            num_iterations=num_iterations,
            window_size=window_size,
            regularization=regularization,
            smooth_flow_sigma=smooth_flow_sigma,
            max_update_px=max_update_px,
        )
        valid_mask = _correspondence_validity_mask(
            img_1_bchw,
            img_2_bchw,
            flow,
            backward_flow,
            consistency_px=valid_consistency_px,
            relative_threshold=valid_relative_threshold,
            consistency_softness=valid_consistency_softness,
            photometric_threshold=valid_photometric_threshold,
            photometric_softness=valid_photometric_softness,
        )
    if apply_valid_mask:
        assert valid_mask is not None
        loss_mask = loss_mask * valid_mask

    flow_mask = loss_mask if apply_valid_mask_to_flow else flow.new_ones(
        (flow.shape[0], 1, flow.shape[2], flow.shape[3])
    )
    if apply_motion_mask and not apply_valid_mask_to_flow:
        assert motion_mask is not None
        flow_mask = motion_mask
    flow = flow * flow_mask

    if img_1_was_unbatched:
        flow = flow[0]
        loss_mask = loss_mask[0]
        if motion_mask is not None:
            motion_mask = motion_mask[0]
        if valid_mask is not None:
            valid_mask = valid_mask[0]
    if return_all_masks:
        masks = {"loss": loss_mask}
        if motion_mask is not None:
            masks["motion"] = motion_mask
        if valid_mask is not None:
            masks["valid"] = valid_mask
        return flow, loss_mask, masks
    if return_mask:
        return flow, loss_mask
    return flow


def displacement_2d(img_1: torch.Tensor, img_2: torch.Tensor, **kwargs) -> torch.Tensor:
    """Alias for call sites that prefer the module name as the function name."""
    return compute_displacement_2d(img_1, img_2, **kwargs)


def _to_bchw(
    image: torch.Tensor,
    name: str,
    channel_order: ChannelOrder,
) -> Tuple[torch.Tensor, bool]:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not torch.is_floating_point(image):
        image = image.float()

    if image.dim() == 2:
        return image.unsqueeze(0).unsqueeze(0), True

    if image.dim() == 3:
        if channel_order == "chw" or (
            channel_order == "auto" and image.shape[0] in {1, 3, 4}
        ):
            return image.unsqueeze(0), True
        if channel_order == "hwc" or (
            channel_order == "auto" and image.shape[-1] in {1, 3, 4}
        ):
            return image.permute(2, 0, 1).unsqueeze(0), True
        raise ValueError(
            f"Cannot infer channel dimension for {name} with shape "
            f"{tuple(image.shape)}. Pass channel_order='chw' or 'hwc'."
        )

    if image.dim() == 4:
        if channel_order == "chw" or (
            channel_order == "auto" and image.shape[1] in {1, 3, 4}
        ):
            return image, False
        if channel_order == "hwc" or (
            channel_order == "auto" and image.shape[-1] in {1, 3, 4}
        ):
            return image.permute(0, 3, 1, 2), False
        raise ValueError(
            f"Cannot infer channel dimension for {name} with shape "
            f"{tuple(image.shape)}. Pass channel_order='chw' or 'hwc'."
        )

    raise ValueError(
        f"{name} must be a 2D, 3D, or 4D image tensor; got {image.dim()}D."
    )


def _to_grayscale(image: torch.Tensor) -> torch.Tensor:
    if image.shape[1] == 1:
        return image
    if image.shape[1] >= 3:
        weights = image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
        return (image[:, :3] * weights).sum(dim=1, keepdim=True)
    return image.mean(dim=1, keepdim=True)


def _estimate_dense_flow(
    img_1_gray: torch.Tensor,
    img_2_gray: torch.Tensor,
    *,
    num_levels: int,
    num_iterations: int,
    window_size: int,
    regularization: float,
    smooth_flow_sigma: Optional[float],
    max_update_px: Optional[float],
) -> torch.Tensor:
    pyramid_1, pyramid_2 = _build_matching_pyramids(
        img_1_gray,
        img_2_gray,
        num_levels,
    )
    kernel = _gaussian_kernel_1d(
        window_size,
        sigma=max(window_size / 6.0, 1.0e-6),
        device=img_1_gray.device,
        dtype=img_1_gray.dtype,
    )

    flow: Optional[torch.Tensor] = None
    for level_img_1, level_img_2 in zip(pyramid_1, pyramid_2):
        batch_size, _, height, width = level_img_1.shape
        if flow is None:
            flow = level_img_1.new_zeros((batch_size, 2, height, width))
        else:
            old_height, old_width = flow.shape[-2:]
            flow = F.interpolate(
                flow,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            scale = flow.new_tensor((width / old_width, height / old_height)).view(
                1,
                2,
                1,
                1,
            )
            flow = flow * scale

        for _ in range(num_iterations):
            warped_img_2 = _warp_image(level_img_2, flow)
            delta = _lucas_kanade_update(
                level_img_1,
                warped_img_2,
                kernel,
                regularization,
            )
            if max_update_px is not None:
                delta = delta.clamp(min=-max_update_px, max=max_update_px)
            flow = flow + delta
            if smooth_flow_sigma is not None and smooth_flow_sigma > 0.0:
                flow = _smooth_flow(flow, smooth_flow_sigma)

    assert flow is not None
    return flow


def _motion_evidence_mask(
    img_1: torch.Tensor,
    img_2: torch.Tensor,
    *,
    quantile: float,
    min_threshold: float,
    softness: float,
    evidence_sigma: Optional[float],
    dilate_px: int,
    blur_sigma: Optional[float],
) -> torch.Tensor:
    if img_1.shape[1] == 1:
        evidence = (img_2 - img_1).abs()
    else:
        evidence = (img_2[:, :3] - img_1[:, :3]).abs().mean(dim=1, keepdim=True)
    if evidence_sigma is not None and evidence_sigma > 0.0:
        kernel_size = max(3, int(round(evidence_sigma * 6.0)) | 1)
        kernel = _gaussian_kernel_1d(
            kernel_size,
            evidence_sigma,
            device=evidence.device,
            dtype=evidence.dtype,
        )
        evidence = _separable_filter2d(evidence, kernel)

    flat_evidence = evidence.flatten(start_dim=1)
    threshold = torch.quantile(flat_evidence, quantile, dim=1, keepdim=True)
    min_threshold_tensor = evidence.new_full(threshold.shape, min_threshold)
    threshold = torch.maximum(threshold, min_threshold_tensor).view(-1, 1, 1, 1)

    mask = torch.sigmoid((evidence - threshold) / softness)
    if dilate_px > 1:
        kernel_size = dilate_px if dilate_px % 2 == 1 else dilate_px + 1
        mask = F.max_pool2d(
            mask,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
    if blur_sigma is not None and blur_sigma > 0.0:
        kernel_size = max(3, int(round(blur_sigma * 6.0)) | 1)
        kernel = _gaussian_kernel_1d(
            kernel_size,
            blur_sigma,
            device=mask.device,
            dtype=mask.dtype,
        )
        mask = _separable_filter2d(mask, kernel)
    return mask.clamp(0.0, 1.0)


def _correspondence_validity_mask(
    img_1: torch.Tensor,
    img_2: torch.Tensor,
    forward_flow: torch.Tensor,
    backward_flow: torch.Tensor,
    *,
    consistency_px: float,
    relative_threshold: float,
    consistency_softness: float,
    photometric_threshold: float,
    photometric_softness: float,
) -> torch.Tensor:
    backward_at_forward = _warp_image(backward_flow, forward_flow)
    fb_residual = (forward_flow + backward_at_forward).norm(dim=1, keepdim=True)
    flow_magnitude = forward_flow.norm(dim=1, keepdim=True) + backward_at_forward.norm(
        dim=1,
        keepdim=True,
    )
    consistency_threshold = consistency_px + relative_threshold * flow_magnitude
    consistency_mask = torch.sigmoid(
        (consistency_threshold - fb_residual) / consistency_softness
    )

    warped_img_2 = _warp_image(img_2, forward_flow)
    channels = min(img_1.shape[1], img_2.shape[1], 3)
    photometric_error = (warped_img_2[:, :channels] - img_1[:, :channels]).abs().mean(
        dim=1,
        keepdim=True,
    )
    photometric_mask = torch.sigmoid(
        (photometric_threshold - photometric_error) / photometric_softness
    )

    return (
        consistency_mask
        * photometric_mask
        * _flow_in_bounds_mask(forward_flow)
    ).clamp(0.0, 1.0)


def _flow_in_bounds_mask(flow: torch.Tensor, softness_px: float = 1.0) -> torch.Tensor:
    batch_size, _, height, width = flow.shape
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    sample_x = x_coords.unsqueeze(0) + flow[:, 0]
    sample_y = y_coords.unsqueeze(0) + flow[:, 1]

    left = torch.sigmoid((sample_x + 0.5) / softness_px)
    right = torch.sigmoid((width - 0.5 - sample_x) / softness_px)
    top = torch.sigmoid((sample_y + 0.5) / softness_px)
    bottom = torch.sigmoid((height - 0.5 - sample_y) / softness_px)
    return (left * right * top * bottom).view(batch_size, 1, height, width)


def _build_matching_pyramids(
    img_1: torch.Tensor,
    img_2: torch.Tensor,
    num_levels: int,
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
    pyramid_1 = [img_1]
    pyramid_2 = [img_2]
    for _ in range(1, num_levels):
        height, width = pyramid_1[-1].shape[-2:]
        if min(height, width) < 16:
            break
        next_size = (max(height // 2, 1), max(width // 2, 1))
        pyramid_1.append(
            F.interpolate(
                pyramid_1[-1],
                size=next_size,
                mode="bilinear",
                align_corners=False,
            )
        )
        pyramid_2.append(
            F.interpolate(
                pyramid_2[-1],
                size=next_size,
                mode="bilinear",
                align_corners=False,
            )
        )
    return tuple(reversed(pyramid_1)), tuple(reversed(pyramid_2))


def _lucas_kanade_update(
    img_1: torch.Tensor,
    warped_img_2: torch.Tensor,
    kernel: torch.Tensor,
    regularization: float,
) -> torch.Tensor:
    gradient_source = 0.5 * (img_1 + warped_img_2)
    grad_x, grad_y = _image_gradients(gradient_source)
    grad_t = warped_img_2 - img_1

    grad_xx = _separable_filter2d(grad_x * grad_x, kernel)
    grad_xy = _separable_filter2d(grad_x * grad_y, kernel)
    grad_yy = _separable_filter2d(grad_y * grad_y, kernel)
    grad_xt = _separable_filter2d(grad_x * grad_t, kernel)
    grad_yt = _separable_filter2d(grad_y * grad_t, kernel)

    a = grad_xx + regularization
    b = grad_xy
    c = grad_yy + regularization
    determinant = (a * c - b * b).clamp_min(regularization * regularization)

    delta_x = (-c * grad_xt + b * grad_yt) / determinant
    delta_y = (b * grad_xt - a * grad_yt) / determinant
    return torch.cat((delta_x, delta_y), dim=1)


def _image_gradients(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    kernel_x = image.new_tensor((-0.5, 0.0, 0.5)).view(1, 1, 1, 3)
    kernel_y = image.new_tensor((-0.5, 0.0, 0.5)).view(1, 1, 3, 1)
    grad_x = F.conv2d(F.pad(image, (1, 1, 0, 0), mode="replicate"), kernel_x)
    grad_y = F.conv2d(F.pad(image, (0, 0, 1, 1), mode="replicate"), kernel_y)
    return grad_x, grad_y


def _warp_image(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = image.shape
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=image.device, dtype=image.dtype),
        torch.arange(width, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    base_grid = torch.stack((x_coords, y_coords), dim=-1).unsqueeze(0)
    sample_grid = base_grid + flow.permute(0, 2, 3, 1)

    if width > 1:
        sample_grid_x = sample_grid[..., 0] / (width - 1) * 2.0 - 1.0
    else:
        sample_grid_x = sample_grid[..., 0].new_zeros((batch_size, height, width))
    if height > 1:
        sample_grid_y = sample_grid[..., 1] / (height - 1) * 2.0 - 1.0
    else:
        sample_grid_y = sample_grid[..., 1].new_zeros((batch_size, height, width))

    normalized_grid = torch.stack((sample_grid_x, sample_grid_y), dim=-1)
    return F.grid_sample(
        image,
        normalized_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def _smooth_flow(flow: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel_size = max(3, int(round(sigma * 6.0)) | 1)
    kernel = _gaussian_kernel_1d(
        kernel_size,
        sigma=sigma,
        device=flow.device,
        dtype=flow.dtype,
    )
    return _separable_filter2d(flow, kernel)


def _separable_filter2d(values: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    channels = values.shape[1]
    radius = kernel.numel() // 2

    kernel_x = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kernel_y = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)

    filtered = F.conv2d(
        F.pad(values, (radius, radius, 0, 0), mode="replicate"),
        kernel_x,
        groups=channels,
    )
    filtered = F.conv2d(
        F.pad(filtered, (0, 0, radius, radius), mode="replicate"),
        kernel_y,
        groups=channels,
    )
    return filtered


def _gaussian_kernel_1d(
    size: int,
    sigma: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if size < 1 or size % 2 == 0:
        raise ValueError("Gaussian kernel size must be a positive odd integer.")
    radius = size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    return kernel / kernel.sum().clamp_min(torch.finfo(dtype).eps)
