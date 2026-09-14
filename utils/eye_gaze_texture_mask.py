from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class IrisTextureSelection:
    bounds: tuple[int, int, int, int]
    mask: np.ndarray
    erase_alpha: np.ndarray
    move_alpha: np.ndarray
    destination_mask: np.ndarray
    background: np.ndarray
    metadata: dict[str, Any]


def _dilate(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(steps))):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(result)
        for row_offset in range(3):
            for column_offset in range(3):
                expanded |= padded[
                    row_offset : row_offset + result.shape[0],
                    column_offset : column_offset + result.shape[1],
                ]
        result = expanded
    return result


def _connected_component(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    selected = np.zeros_like(mask)
    if not mask[seed]:
        candidates = np.argwhere(mask)
        if len(candidates) == 0:
            return selected
        distance = np.sum((candidates - np.asarray(seed)) ** 2, axis=1)
        seed = tuple(int(value) for value in candidates[int(distance.argmin())])
    selected[seed] = True
    pending = [seed]
    while pending:
        row, column = pending.pop()
        for row_offset in (-1, 0, 1):
            for column_offset in (-1, 0, 1):
                if row_offset == 0 and column_offset == 0:
                    continue
                neighbor_row = row + row_offset
                neighbor_column = column + column_offset
                if (
                    0 <= neighbor_row < mask.shape[0]
                    and 0 <= neighbor_column < mask.shape[1]
                    and mask[neighbor_row, neighbor_column]
                    and not selected[neighbor_row, neighbor_column]
                ):
                    selected[neighbor_row, neighbor_column] = True
                    pending.append((neighbor_row, neighbor_column))
    return selected


def _components_touching(mask: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    selected = np.asarray(seeds, dtype=bool) & mask
    pending = [tuple(int(value) for value in row) for row in np.argwhere(selected)]
    while pending:
        row, column = pending.pop()
        for row_offset in (-1, 0, 1):
            for column_offset in (-1, 0, 1):
                if row_offset == 0 and column_offset == 0:
                    continue
                neighbor_row = row + row_offset
                neighbor_column = column + column_offset
                if (
                    0 <= neighbor_row < mask.shape[0]
                    and 0 <= neighbor_column < mask.shape[1]
                    and mask[neighbor_row, neighbor_column]
                    and not selected[neighbor_row, neighbor_column]
                ):
                    selected[neighbor_row, neighbor_column] = True
                    pending.append((neighbor_row, neighbor_column))
    return selected


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    exterior = np.zeros_like(mask)
    pending: list[tuple[int, int]] = []
    for row in (0, mask.shape[0] - 1):
        for column in range(mask.shape[1]):
            if not mask[row, column] and not exterior[row, column]:
                exterior[row, column] = True
                pending.append((row, column))
    for column in (0, mask.shape[1] - 1):
        for row in range(mask.shape[0]):
            if not mask[row, column] and not exterior[row, column]:
                exterior[row, column] = True
                pending.append((row, column))
    while pending:
        row, column = pending.pop()
        for row_offset, column_offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor_row = row + row_offset
            neighbor_column = column + column_offset
            if (
                0 <= neighbor_row < mask.shape[0]
                and 0 <= neighbor_column < mask.shape[1]
                and not mask[neighbor_row, neighbor_column]
                and not exterior[neighbor_row, neighbor_column]
            ):
                exterior[neighbor_row, neighbor_column] = True
                pending.append((neighbor_row, neighbor_column))
    return mask | (~mask & ~exterior)


def _smooth_outer_alpha(mask: np.ndarray, steps: int) -> np.ndarray:
    alpha = np.asarray(mask, dtype=np.float32)
    previous = np.asarray(mask, dtype=bool)
    steps = max(1, int(steps))
    for step in range(steps):
        expanded = _dilate(previous)
        ring = expanded & ~previous
        alpha[ring] = float(steps - step) / float(steps + 1)
        previous = expanded
    return alpha




def segment_iris_texture(
    pixels: np.ndarray,
    *,
    center_xy: tuple[float, float],
    horizontal_axis_uv: tuple[float, float],
    vertical_axis_uv: tuple[float, float],
    horizontal_radius_uv: float,
    vertical_radius_uv: float,
    reference_sclera_color: tuple[float, float, float] | None = None,
    cache_metadata: dict[str, Any] | None = None,
) -> IrisTextureSelection:
    """Select a complete textured iris using MediaPipe's ellipse as a guardrail."""

    pixels = np.asarray(pixels, dtype=np.float32)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError("pixels must have shape [H, W, C] with at least three channels.")
    height, width = pixels.shape[:2]
    center_x, center_y = (float(value) for value in center_xy)
    horizontal_axis = np.asarray(horizontal_axis_uv, dtype=np.float64)
    vertical_axis = np.asarray(vertical_axis_uv, dtype=np.float64)
    if (
        horizontal_axis.shape != (2,)
        or vertical_axis.shape != (2,)
        or horizontal_radius_uv <= 0.0
        or vertical_radius_uv <= 0.0
    ):
        raise ValueError("The iris UV ellipse is invalid.")

    radius_pixels = max(
        horizontal_radius_uv * width,
        vertical_radius_uv * height,
    )
    margin = max(8, int(np.ceil(radius_pixels * 3.2)))
    x0 = max(0, int(np.floor(center_x)) - margin)
    x1 = min(width, int(np.ceil(center_x)) + margin + 1)
    y0 = max(0, int(np.floor(center_y)) - margin)
    y1 = min(height, int(np.ceil(center_y)) + margin + 1)
    if x1 - x0 < 9 or y1 - y0 < 9:
        raise ValueError("The iris texture patch is too small.")

    patch = pixels[y0:y1, x0:x1]
    rgb = patch[..., :3].astype(np.float64)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    delta_u = (xx - center_x) / max(width - 1, 1)
    delta_v = (yy - center_y) / max(height - 1, 1)
    horizontal_coordinate = (
        delta_u * horizontal_axis[0] + delta_v * horizontal_axis[1]
    ) / float(horizontal_radius_uv)
    vertical_coordinate = (
        delta_u * vertical_axis[0] + delta_v * vertical_axis[1]
    ) / float(vertical_radius_uv)
    normalized_radius = np.sqrt(horizontal_coordinate**2 + vertical_coordinate**2)
    luminance = rgb @ np.asarray((0.2126, 0.7152, 0.0722))

    sclera_search = (normalized_radius >= 1.20) & (normalized_radius <= 2.55)
    search_luminance = luminance[sclera_search]
    if len(search_luminance) < 24:
        raise ValueError("The MediaPipe iris ellipse has too little surrounding texture.")
    if reference_sclera_color is None:
        bright_limit = float(np.quantile(search_luminance, 0.70))
        bright_samples = sclera_search & (luminance >= bright_limit)
        if int(np.count_nonzero(bright_samples)) < 16:
            raise ValueError("The iris patch has too few sclera color samples.")
        sclera_color = np.median(rgb[bright_samples], axis=0)
    else:
        sclera_color = np.asarray(reference_sclera_color, dtype=np.float64)
        if sclera_color.shape != (3,) or not np.isfinite(sclera_color).all():
            raise ValueError("The reference sclera color is invalid.")
        reference_distance = np.linalg.norm(rgb - sclera_color, axis=-1)
        reference_luminance = float(
            np.dot(sclera_color, np.asarray((0.2126, 0.7152, 0.0722)))
        )
        bright_samples = (
            sclera_search
            & (reference_distance <= 0.30)
            & (luminance >= reference_luminance * 0.55)
        )
    if int(np.count_nonzero(bright_samples)) < 16:
        raise ValueError("The iris patch has too few sclera color samples.")
    sclera_luminance = float(
        np.dot(sclera_color, np.asarray((0.2126, 0.7152, 0.0722)))
    )
    color_distance = np.linalg.norm(rgb - sclera_color, axis=-1)
    contrast_score = color_distance + np.maximum(0.0, sclera_luminance - luminance) * 0.65
    sclera_scores = contrast_score[bright_samples]
    minimum_threshold = max(
        0.055,
        float(np.quantile(sclera_scores, 0.98)) * 1.35,
    )
    inner_scores = contrast_score[normalized_radius <= 1.15]
    if len(inner_scores) < 12:
        raise ValueError("The MediaPipe iris ellipse contains too few texture pixels.")
    maximum_threshold = max(minimum_threshold, float(np.quantile(inner_scores, 0.78)))

    seed = (
        int(np.clip(round(center_y) - y0, 0, patch.shape[0] - 1)),
        int(np.clip(round(center_x) - x0, 0, patch.shape[1] - 1)),
    )
    selected = None
    selected_threshold = None
    selected_area_ratio = None
    selected_score = None
    ellipse_fallback = False
    expected_area = max(
        1.0,
        np.pi * horizontal_radius_uv * width * vertical_radius_uv * height,
    )
    for threshold in np.linspace(maximum_threshold, minimum_threshold, 20):
        candidates = (normalized_radius <= 2.20) & (contrast_score >= threshold)
        bridged_candidates = _dilate(candidates, 2)
        bridged_component = _connected_component(bridged_candidates, seed)
        component = _fill_holes(bridged_component & candidates)
        area_ratio = float(np.count_nonzero(component) / expected_area)
        if not 0.48 <= area_ratio <= 3.00:
            continue
        horizontal_values = np.abs(horizontal_coordinate[component])
        vertical_values = np.abs(vertical_coordinate[component])
        radial_values = normalized_radius[component]
        horizontal_extent = float(np.quantile(horizontal_values, 0.98))
        vertical_extent = float(np.quantile(vertical_values, 0.98))
        radial_extent = float(np.quantile(radial_values, 0.995))
        if radial_extent >= 2.10 or float(radial_values.max()) >= 2.18:
            continue
        score = (
            float(threshold)
            + max(0.0, abs(area_ratio - 1.0) - 0.35) * 4.0
            + max(0.0, 0.70 - horizontal_extent) * 4.0
            + max(0.0, 0.70 - vertical_extent) * 4.0
        )
        if selected_score is None or score < selected_score:
            selected = component
            selected_threshold = float(threshold)
            selected_area_ratio = area_ratio
            selected_score = score
    if selected is None:
        # Some layered Pixel3D eyes split the visible iris across UV islands,
        # so a connected dark component covers only part of the five-point
        # ellipse.  Accept the guarded ellipse itself only when its interior is
        # broadly and materially different from the surrounding sclera.  This
        # retains the color check while avoiding a pupil-only or partial-ring
        # selection.
        fallback_inner = normalized_radius <= 1.05
        fallback_contrast_fraction = float(
            np.mean(contrast_score[fallback_inner] >= minimum_threshold)
        )
        fallback_median_color_distance = float(
            np.median(color_distance[fallback_inner])
        )
        if (
            fallback_contrast_fraction >= 0.18
            and fallback_median_color_distance >= 0.14
        ):
            selected = fallback_inner
            selected_threshold = float(minimum_threshold)
            selected_area_ratio = float(
                np.count_nonzero(selected) / expected_area
            )
            selected_score = float(
                minimum_threshold + 1.0 - fallback_contrast_fraction
            )
            ellipse_fallback = True
        else:
            raise ValueError(
                "Color segmentation did not find a landmark-sized connected iris."
            )

    low_threshold_candidates = (
        (normalized_radius >= 0.80)
        & (normalized_radius <= 2.10)
        & (contrast_score >= minimum_threshold)
        & ~selected
    )
    angles = (np.arctan2(vertical_coordinate, horizontal_coordinate) + np.pi) / (
        2.0 * np.pi
    )
    angle_bins = np.floor(angles * 36.0).astype(np.int64)
    angle_bins = np.clip(angle_bins, 0, 35)
    angular_counts = np.bincount(
        angle_bins[low_threshold_candidates],
        minlength=36,
    )
    angular_balance = float(
        np.quantile(angular_counts, 0.10)
        / max(float(np.quantile(angular_counts, 0.90)), 1.0)
    )
    ring_augmented = False
    if not ellipse_fallback and angular_balance >= 0.20:
        ring_candidates = (
            (normalized_radius <= 2.10)
            & (contrast_score >= minimum_threshold)
        )
        ring_seeds = ring_candidates & _dilate(selected, 5)
        ring_component = _fill_holes(
            _components_touching(ring_candidates, ring_seeds) | selected
        )
        ring_area_ratio = float(np.count_nonzero(ring_component) / expected_area)
        if 0.55 <= ring_area_ratio <= 3.20:
            selected = ring_component
            selected_area_ratio = ring_area_ratio
            ring_augmented = True

    edge_steps = max(1, int(round(min(
        horizontal_radius_uv * width,
        vertical_radius_uv * height,
    ) * 0.08)))
    # Dark eyelashes can touch the iris in the packed texture and pull the
    # connected color component far outside the five-point iris ellipse. Keep
    # a modest outer-ring margin, but never move or erase that eyelid region.
    maximum_iris_radius = 1.16
    color_mask = _fill_holes(_dilate(selected, edge_steps)) & (
        normalized_radius <= maximum_iris_radius
    )
    # The pupil is much easier to segment than a gray or lightly colored outer
    # iris. Once color validates the location, move the full landmark ellipse
    # so that the outer ring cannot remain behind at the neutral position.
    mask = normalized_radius <= maximum_iris_radius
    if int(np.count_nonzero(color_mask)) < int(np.count_nonzero(mask) * 0.45):
        raise ValueError(
            "The color-selected iris covers too little of the landmark ellipse."
        )
    final_area_ratio = float(np.count_nonzero(mask) / expected_area)
    horizontal_extent = float(np.max(np.abs(horizontal_coordinate[mask])))
    vertical_extent = float(np.max(np.abs(vertical_coordinate[mask])))
    if (
        not 0.55 <= final_area_ratio <= 3.50
        or horizontal_extent < 0.72
        or vertical_extent < 0.72
    ):
        raise ValueError(
            "The color-selected iris does not cover the MediaPipe iris extent."
        )

    sclera_distance_limit = max(
        0.14,
        float(np.quantile(color_distance[bright_samples], 0.98)) * 2.5,
    )
    sclera_candidates = (
        (normalized_radius <= 3.0)
        & (color_distance <= sclera_distance_limit)
        & (luminance >= sclera_luminance * 0.55)
    )
    # A packed atlas can place unrelated skin, hair, or another eye beside this
    # UV island. Only use sclera connected to the selected iris boundary.
    sclera_contact = _dilate(mask, max(3, edge_steps * 4))
    sclera_mask = _components_touching(
        sclera_candidates,
        sclera_candidates & sclera_contact,
    )
    if int(np.count_nonzero(sclera_mask)) < 24:
        raise ValueError(
            "The color-selected iris has too little connected sclera context."
        )
    destination_mask = _fill_holes(
        _dilate(sclera_mask, max(1, edge_steps)) | mask
    )
    selected_radius = float(np.quantile(normalized_radius[mask], 0.995))
    cleanup_radius = min(
        1.58,
        selected_radius
        + max(0.26, edge_steps * 2.0 / max(radius_pixels, 1.0)),
    )
    erase_core = mask | (
        destination_mask & (normalized_radius <= cleanup_radius)
    )
    # A fitted plane can extrapolate skin-colored values beneath the iris even
    # when its samples are valid sclera. A robust constant fill is less exact,
    # but it cannot create a brown wedge when the source footprint is exposed.
    if reference_sclera_color is None:
        sclera_values = rgb[sclera_mask]
        sclera_value_luminance = sclera_values @ np.asarray(
            (0.2126, 0.7152, 0.0722)
        )
        bright_sclera = sclera_values[
            sclera_value_luminance
            >= np.quantile(sclera_value_luminance, 0.85)
        ]
        background_color = np.median(bright_sclera, axis=0)
    else:
        background_color = sclera_color.copy()
    background_luminance = float(
        np.dot(background_color, np.asarray((0.2126, 0.7152, 0.0722)))
    )
    if background_luminance < 0.86:
        background_color = np.clip(
            background_color + (0.86 - background_luminance),
            0.0,
            1.0,
        )
    neutral = np.full(3, float(np.mean(background_color)), dtype=np.float64)
    background_color = neutral + (background_color - neutral) * 0.25
    background = np.broadcast_to(background_color, rgb.shape).copy()
    move_alpha = _smooth_outer_alpha(mask, max(1, edge_steps))
    erase_alpha = _smooth_outer_alpha(
        erase_core,
        max(2, edge_steps + 1),
    )

    metadata = dict(cache_metadata or {})
    metadata.update(
        {
            "version": 14,
            "image_width": width,
            "image_height": height,
            "center_x": center_x,
            "center_y": center_y,
            "horizontal_radius_uv": float(horizontal_radius_uv),
            "vertical_radius_uv": float(vertical_radius_uv),
            "threshold": selected_threshold,
            "selection_score": selected_score,
            "angular_balance": angular_balance,
            "ring_augmented": ring_augmented,
            "ellipse_fallback": ellipse_fallback,
            "initial_area_ratio": selected_area_ratio,
            "area_ratio": final_area_ratio,
            "horizontal_extent": horizontal_extent,
            "vertical_extent": vertical_extent,
            "selected_pixels": int(np.count_nonzero(mask)),
            "cleanup_pixels": int(np.count_nonzero(erase_core)),
            "cleanup_radius": cleanup_radius,
            "sclera_pixels": int(np.count_nonzero(sclera_mask)),
            "sclera_color": sclera_color.tolist(),
            "reference_sclera_color": (
                None
                if reference_sclera_color is None
                else [float(value) for value in reference_sclera_color]
            ),
        }
    )
    return IrisTextureSelection(
        bounds=(x0, y0, x1, y1),
        mask=mask,
        erase_alpha=erase_alpha,
        move_alpha=move_alpha,
        destination_mask=destination_mask,
        background=background.astype(np.float32),
        metadata=metadata,
    )


def apply_iris_texture_translation(
    pixels: np.ndarray,
    selection: IrisTextureSelection,
    *,
    shift_xy: tuple[int, int],
) -> dict[str, int]:
    """Move the selected source pixels rigidly inside the cached eye aperture."""

    x0, y0, x1, y1 = selection.bounds
    patch = pixels[y0:y1, x0:x1]
    if patch.shape[:2] != selection.mask.shape:
        raise ValueError("Cached iris selection does not match the texture patch.")
    source = patch.copy()
    erase_alpha = selection.erase_alpha[..., None]
    patch[..., :3] = (
        patch[..., :3] * (1.0 - erase_alpha)
        + selection.background * erase_alpha
    )

    shift_x, shift_y = (int(value) for value in shift_xy)
    source_rows, source_columns = np.nonzero(selection.move_alpha > 1.0e-4)
    destination_rows = source_rows + shift_y
    destination_columns = source_columns + shift_x
    valid = (
        (destination_rows >= 0)
        & (destination_rows < patch.shape[0])
        & (destination_columns >= 0)
        & (destination_columns < patch.shape[1])
    )
    source_rows = source_rows[valid]
    source_columns = source_columns[valid]
    destination_rows = destination_rows[valid]
    destination_columns = destination_columns[valid]
    allowed = selection.destination_mask[destination_rows, destination_columns]
    source_rows = source_rows[allowed]
    source_columns = source_columns[allowed]
    destination_rows = destination_rows[allowed]
    destination_columns = destination_columns[allowed]
    alpha = selection.move_alpha[source_rows, source_columns, None]
    destination_rgb = patch[destination_rows, destination_columns, :3]
    source_rgb = source[source_rows, source_columns, :3]
    patch[destination_rows, destination_columns, :3] = (
        destination_rgb * (1.0 - alpha) + source_rgb * alpha
    )
    if patch.shape[2] > 3:
        destination_alpha = patch[destination_rows, destination_columns, 3:4]
        source_alpha = source[source_rows, source_columns, 3:4]
        patch[destination_rows, destination_columns, 3:4] = (
            destination_alpha * (1.0 - alpha) + source_alpha * alpha
        )
    return {
        "source_pixels": int(np.count_nonzero(selection.mask)),
        "translated_pixels": int(len(source_rows)),
        "clipped_pixels": int(np.count_nonzero(valid) - len(source_rows)),
    }


def apply_iris_occlusion_cleanup(
    pixels: np.ndarray,
    *,
    center_xy: tuple[float, float],
    horizontal_axis_uv: tuple[float, float],
    vertical_axis_uv: tuple[float, float],
    horizontal_radius_uv: float,
    vertical_radius_uv: float,
    sclera_color: tuple[float, float, float],
) -> dict[str, int]:
    """Cover an occluded rear-layer footprint exposed by iris translation."""

    pixels = np.asarray(pixels, dtype=np.float32)
    height, width = pixels.shape[:2]
    center_x, center_y = (float(value) for value in center_xy)
    horizontal_axis = np.asarray(horizontal_axis_uv, dtype=np.float64)
    vertical_axis = np.asarray(vertical_axis_uv, dtype=np.float64)
    color = np.asarray(sclera_color, dtype=np.float32)
    if (
        pixels.ndim != 3
        or pixels.shape[2] < 3
        or horizontal_axis.shape != (2,)
        or vertical_axis.shape != (2,)
        or horizontal_radius_uv <= 0.0
        or vertical_radius_uv <= 0.0
        or color.shape != (3,)
        or not np.isfinite(color).all()
    ):
        raise ValueError("The iris occlusion cleanup parameters are invalid.")

    radius_pixels = max(
        horizontal_radius_uv * width,
        vertical_radius_uv * height,
    )
    margin = max(4, int(np.ceil(radius_pixels * 1.35)))
    x0 = max(0, int(np.floor(center_x)) - margin)
    x1 = min(width, int(np.ceil(center_x)) + margin + 1)
    y0 = max(0, int(np.floor(center_y)) - margin)
    y1 = min(height, int(np.ceil(center_y)) + margin + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        raise ValueError("The rear iris cleanup patch is too small.")
    yy, xx = np.mgrid[y0:y1, x0:x1]
    delta_u = (xx - center_x) / max(width - 1, 1)
    delta_v = (yy - center_y) / max(height - 1, 1)
    horizontal_coordinate = (
        delta_u * horizontal_axis[0] + delta_v * horizontal_axis[1]
    ) / float(horizontal_radius_uv)
    vertical_coordinate = (
        delta_u * vertical_axis[0] + delta_v * vertical_axis[1]
    ) / float(vertical_radius_uv)
    normalized_radius = np.sqrt(horizontal_coordinate**2 + vertical_coordinate**2)
    alpha = np.clip((1.16 - normalized_radius) / 0.16, 0.0, 1.0).astype(
        np.float32
    )
    cleanup_pixels = int(np.count_nonzero(alpha > 1.0e-4))
    if cleanup_pixels < 16:
        raise ValueError("The rear iris cleanup footprint is empty.")
    patch = pixels[y0:y1, x0:x1, :3]
    patch[:] = patch * (1.0 - alpha[..., None]) + color * alpha[..., None]
    return {
        "source_pixels": cleanup_pixels,
        "translated_pixels": 0,
        "clipped_pixels": 0,
    }


def save_iris_texture_selection(
    path: Path,
    selection: IrisTextureSelection,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary_path,
        bounds=np.asarray(selection.bounds, dtype=np.int64),
        mask=selection.mask.astype(np.uint8),
        erase_alpha=selection.erase_alpha.astype(np.float32),
        move_alpha=selection.move_alpha.astype(np.float32),
        destination_mask=selection.destination_mask.astype(np.uint8),
        background=selection.background.astype(np.float32),
        metadata=np.asarray(json.dumps(selection.metadata, sort_keys=True)),
    )
    temporary_path.replace(path)


def load_iris_texture_selection(path: Path) -> IrisTextureSelection:
    with np.load(path, allow_pickle=False) as payload:
        return IrisTextureSelection(
            bounds=tuple(int(value) for value in payload["bounds"]),
            mask=payload["mask"].astype(bool),
            erase_alpha=payload["erase_alpha"].astype(np.float32),
            move_alpha=payload["move_alpha"].astype(np.float32),
            destination_mask=payload["destination_mask"].astype(bool),
            background=payload["background"].astype(np.float32),
            metadata=json.loads(str(payload["metadata"])),
        )


def selection_matches(
    selection: IrisTextureSelection,
    *,
    width: int,
    height: int,
    center_xy: tuple[float, float],
    horizontal_radius_uv: float,
    vertical_radius_uv: float,
    reference_sclera_color: tuple[float, float, float] | None = None,
) -> bool:
    metadata = selection.metadata
    cached_reference = metadata.get("reference_sclera_color")
    reference_matches = (
        cached_reference is None
        if reference_sclera_color is None
        else cached_reference is not None
        and np.allclose(
            np.asarray(cached_reference, dtype=np.float64),
            np.asarray(reference_sclera_color, dtype=np.float64),
            atol=1.0e-4,
        )
    )
    return (
        int(metadata.get("version", -1)) == 14
        and int(metadata.get("image_width", -1)) == int(width)
        and int(metadata.get("image_height", -1)) == int(height)
        and abs(float(metadata.get("center_x", -1.0)) - float(center_xy[0])) <= 0.75
        and abs(float(metadata.get("center_y", -1.0)) - float(center_xy[1])) <= 0.75
        and np.isclose(
            float(metadata.get("horizontal_radius_uv", -1.0)),
            float(horizontal_radius_uv),
            rtol=0.02,
        )
        and np.isclose(
            float(metadata.get("vertical_radius_uv", -1.0)),
            float(vertical_radius_uv),
            rtol=0.02,
        )
        and reference_matches
    )


def write_iris_selection_debug(
    path: Path,
    pixels: np.ndarray,
    selection: IrisTextureSelection,
) -> None:
    from PIL import Image, ImageDraw

    x0, y0, x1, y1 = selection.bounds
    patch = np.asarray(pixels[y0:y1, x0:x1, :3], dtype=np.float32)
    patch = np.clip(patch, 0.0, 1.0)
    source = Image.fromarray(np.rint(patch * 255.0).astype(np.uint8), mode="RGB")
    overlay = np.asarray(source).copy()
    overlay[selection.destination_mask] = (
        overlay[selection.destination_mask].astype(np.float32) * 0.70
        + np.asarray((0, 180, 255), dtype=np.float32) * 0.30
    ).astype(np.uint8)
    overlay[selection.mask] = (
        overlay[selection.mask].astype(np.float32) * 0.35
        + np.asarray((0, 255, 70), dtype=np.float32) * 0.65
    ).astype(np.uint8)
    overlay_image = Image.fromarray(overlay, mode="RGB")
    canvas = Image.new("RGB", (source.width * 2, source.height + 24), "white")
    canvas.paste(source, (0, 24))
    canvas.paste(overlay_image, (source.width, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 6), "source", fill="black")
    draw.text((source.width + 4, 6), "green=iris cyan=allowed", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
