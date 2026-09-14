"""Landmark input features, dropout and perturbations used by TopoRig."""
from __future__ import annotations
import math
from typing import Any, Mapping, Optional
import torch
from utils.eyelid_landmarks import BLINK_LEFT_EYELIDS, BLINK_RIGHT_EYELIDS
from utils.eye_gaze_uv_warp import LEFT_EYE_GAZE_LANDMARKS, RIGHT_EYE_GAZE_LANDMARKS
from utils.mouth_landmark_validation import MOUTH_LANDMARK_IDS

LANDMARK_FEATURE_REGION_IDS = {
    "left_eye": frozenset(
        (
            *BLINK_LEFT_EYELIDS.all_ids,
            *LEFT_EYE_GAZE_LANDMARKS.contour_ids,
            *LEFT_EYE_GAZE_LANDMARKS.iris_ids,
        )
    ),
    "right_eye": frozenset(
        (
            *BLINK_RIGHT_EYELIDS.all_ids,
            *RIGHT_EYE_GAZE_LANDMARKS.contour_ids,
            *RIGHT_EYE_GAZE_LANDMARKS.iris_ids,
        )
    ),
    "mouth": frozenset(MOUTH_LANDMARK_IDS),
}


def landmark_feature_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    model_config = config.get("model", {})
    if not isinstance(model_config, Mapping):
        return {}
    feature_config = model_config.get("landmark_features", {})
    return feature_config if isinstance(feature_config, Mapping) else {}


def landmark_features_enabled(config: Mapping[str, Any]) -> bool:
    return bool(landmark_feature_config(config).get("enabled", False))


def landmark_feature_dim(config: Mapping[str, Any]) -> int:
    feature_config = landmark_feature_config(config)
    if not bool(feature_config.get("enabled", False)):
        return 0
    k_nearest = int(feature_config.get("k_nearest", 4))
    if k_nearest < 1:
        raise ValueError("model.landmark_features.k_nearest must be at least 1.")
    channels_per_landmark = 0
    if bool(feature_config.get("include_vectors", True)):
        channels_per_landmark += 3
    if bool(feature_config.get("include_distances", True)):
        channels_per_landmark += 1
    if bool(feature_config.get("include_ids", True)):
        channels_per_landmark += 1
    if channels_per_landmark == 0:
        raise ValueError(
            "model.landmark_features must include at least one of vectors, "
            "distances, or ids."
        )
    return k_nearest * channels_per_landmark


def build_model_landmark_features(
    record: Mapping[str, Any],
    vertices: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    *,
    training: bool = False,
    perturbation: Optional[Mapping[str, Any]] = None,
    generator: Optional[torch.Generator] = None,
) -> Optional[torch.Tensor]:
    if not landmark_features_enabled(config):
        return None
    feature_dim = landmark_feature_dim(config)
    if vertices.dim() != 3 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape [B, V, 3].")

    landmarks_3d = record.get("landmarks_3d")
    if not isinstance(landmarks_3d, Mapping):
        return vertices.new_zeros((*vertices.shape[:2], feature_dim))

    positions = landmarks_3d.get("neutral_positions")
    if not isinstance(positions, torch.Tensor) or positions.numel() == 0:
        return vertices.new_zeros((*vertices.shape[:2], feature_dim))

    mediapipe_ids = landmarks_3d.get("mediapipe_ids")
    if not isinstance(mediapipe_ids, torch.Tensor) or mediapipe_ids.numel() == 0:
        mediapipe_ids = torch.arange(
            positions.shape[-2],
            dtype=torch.long,
            device=positions.device,
        )

    positions = positions.to(device=device, dtype=vertices.dtype)
    mediapipe_ids = mediapipe_ids.to(device=device, dtype=torch.long).flatten()
    feature_config = landmark_feature_config(config)
    if training:
        training_augmentation = feature_config.get("training_augmentation", {})
        if isinstance(training_augmentation, Mapping) and bool(
            training_augmentation.get("enabled", False)
        ):
            if landmark_random_event(
                training_augmentation.get("modality_dropout_probability", 0.0),
                device=device,
                generator=generator,
                label="modality_dropout_probability",
            ):
                return vertices.new_zeros((*vertices.shape[:2], feature_dim))
            regions = normalize_landmark_feature_regions(
                training_augmentation.get(
                    "regions", ("left_eye", "right_eye", "mouth")
                )
            )
            if regions and landmark_random_event(
                training_augmentation.get("region_dropout_probability", 0.0),
                device=device,
                generator=generator,
                label="region_dropout_probability",
            ):
                region_index = int(
                    torch.randint(
                        len(regions),
                        (),
                        device=device,
                        generator=generator,
                    ).item()
                )
                positions, mediapipe_ids = drop_model_landmark_regions(
                    positions,
                    mediapipe_ids,
                    (regions[region_index],),
                )
            if landmark_random_event(
                training_augmentation.get("position_jitter_probability", 0.0),
                device=device,
                generator=generator,
                label="position_jitter_probability",
            ):
                positions = jitter_model_landmark_positions(
                    positions,
                    mediapipe_ids,
                    vertices,
                    std_head_fraction=float(
                        training_augmentation.get(
                            "position_jitter_std_head_fraction", 0.0
                        )
                    ),
                    regions=training_augmentation.get("position_jitter_regions"),
                    up_axis=int(training_augmentation.get("head_up_axis", 2)),
                    generator=generator,
                )

    if perturbation:
        if bool(perturbation.get("drop_all", False)):
            return vertices.new_zeros((*vertices.shape[:2], feature_dim))
        positions, mediapipe_ids = drop_model_landmark_regions(
            positions,
            mediapipe_ids,
            normalize_landmark_feature_regions(
                perturbation.get("drop_regions", ()),
            ),
        )
        positions = permute_model_landmark_positions(
            positions,
            mediapipe_ids,
            regions=perturbation.get("permute_regions"),
            generator=generator,
        )
        positions = jitter_model_landmark_positions(
            positions,
            mediapipe_ids,
            vertices,
            std_head_fraction=float(
                perturbation.get("position_jitter_std_head_fraction", 0.0)
            ),
            regions=perturbation.get("position_jitter_regions"),
            up_axis=int(perturbation.get("head_up_axis", 2)),
            generator=generator,
        )

    if positions.shape[-2] == 0 or mediapipe_ids.numel() == 0:
        return vertices.new_zeros((*vertices.shape[:2], feature_dim))

    return landmark_relative_vertex_features(
        vertices=vertices,
        landmark_positions=positions,
        mediapipe_ids=mediapipe_ids,
        config=feature_config,
        device=device,
    )


def normalize_landmark_feature_regions(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    raw_values = (value,) if isinstance(value, str) else tuple(value)
    regions = tuple(dict.fromkeys(str(item).strip().lower() for item in raw_values))
    unknown = sorted(set(regions) - set(LANDMARK_FEATURE_REGION_IDS))
    if unknown:
        raise ValueError(
            "Unknown landmark feature region(s): "
            f"{', '.join(unknown)}. Expected one of: "
            f"{', '.join(sorted(LANDMARK_FEATURE_REGION_IDS))}."
        )
    return regions


def landmark_random_event(
    probability: Any,
    *,
    device: torch.device,
    generator: Optional[torch.Generator],
    label: str,
) -> bool:
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"model.landmark_features.{label} must be in [0, 1].")
    return probability > 0.0 and bool(
        torch.rand((), device=device, generator=generator).item() < probability
    )


def drop_model_landmark_regions(
    positions: torch.Tensor,
    mediapipe_ids: torch.Tensor,
    regions: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    regions = normalize_landmark_feature_regions(regions)
    if not regions:
        return positions, mediapipe_ids
    dropped_ids = set().union(
        *(LANDMARK_FEATURE_REGION_IDS[region] for region in regions)
    )
    keep = torch.tensor(
        [int(value) not in dropped_ids for value in mediapipe_ids.detach().cpu()],
        device=mediapipe_ids.device,
        dtype=torch.bool,
    )
    return positions.index_select(-2, keep.nonzero().flatten()), mediapipe_ids[keep]


def jitter_model_landmark_positions(
    positions: torch.Tensor,
    mediapipe_ids: torch.Tensor,
    vertices: torch.Tensor,
    *,
    std_head_fraction: float,
    regions: Any = None,
    up_axis: int = 2,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    std_head_fraction = float(std_head_fraction)
    if std_head_fraction < 0.0 or not math.isfinite(std_head_fraction):
        raise ValueError("Landmark position jitter must be finite and non-negative.")
    if std_head_fraction == 0.0 or positions.shape[-2] == 0:
        return positions
    if not 0 <= up_axis < 3:
        raise ValueError("Landmark position jitter head_up_axis must be 0, 1, or 2.")
    if positions.dim() == 2:
        batched_positions = positions.unsqueeze(0)
        squeeze_batch = True
    elif positions.dim() == 3:
        batched_positions = positions
        squeeze_batch = False
    else:
        raise ValueError("Landmark positions must have shape [L, 3] or [B, L, 3].")
    if batched_positions.shape[0] not in (1, vertices.shape[0]):
        raise ValueError("Landmark position batch size must match vertices.")

    selected_regions = normalize_landmark_feature_regions(regions)
    if selected_regions:
        selected_ids = set().union(
            *(LANDMARK_FEATURE_REGION_IDS[region] for region in selected_regions)
        )
        selected = torch.tensor(
            [int(value) in selected_ids for value in mediapipe_ids.detach().cpu()],
            device=positions.device,
            dtype=torch.bool,
        )
    else:
        selected = torch.ones(
            mediapipe_ids.shape,
            device=positions.device,
            dtype=torch.bool,
        )
    if not bool(selected.any()):
        return positions

    head_height = (
        vertices[..., up_axis].amax(dim=1)
        - vertices[..., up_axis].amin(dim=1)
    ).clamp_min(torch.finfo(vertices.dtype).eps)
    if batched_positions.shape[0] == 1 and vertices.shape[0] > 1:
        head_height = head_height[:1]
    noise = torch.randn(
        batched_positions.shape,
        device=positions.device,
        dtype=positions.dtype,
        generator=generator,
    )
    noise = noise * head_height.reshape(-1, 1, 1) * std_head_fraction
    noise = noise * selected.reshape(1, -1, 1)
    jittered = batched_positions + noise
    return jittered.squeeze(0) if squeeze_batch else jittered


def permute_model_landmark_positions(
    positions: torch.Tensor,
    mediapipe_ids: torch.Tensor,
    *,
    regions: Any,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    selected_regions = normalize_landmark_feature_regions(regions)
    if not selected_regions or positions.shape[-2] < 2:
        return positions
    selected_ids = set().union(
        *(LANDMARK_FEATURE_REGION_IDS[region] for region in selected_regions)
    )
    selected = torch.tensor(
        [int(value) in selected_ids for value in mediapipe_ids.detach().cpu()],
        device=positions.device,
        dtype=torch.bool,
    ).nonzero().flatten()
    if selected.numel() < 2:
        return positions
    permutation = torch.randperm(
        selected.numel(),
        device=positions.device,
        generator=generator,
    )
    identity = torch.arange(selected.numel(), device=positions.device)
    if bool(torch.equal(permutation, identity)):
        permutation = permutation.roll(1)
    source = selected.index_select(0, permutation)
    output = positions.clone()
    output[..., selected, :] = positions.index_select(-2, source)
    return output


def landmark_relative_vertex_features(
    vertices: torch.Tensor,
    landmark_positions: torch.Tensor,
    mediapipe_ids: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    dtype = vertices.dtype
    vertices = vertices.to(device=device)
    landmark_positions = landmark_positions.to(device=device, dtype=dtype)
    mediapipe_ids = mediapipe_ids.to(device=device, dtype=torch.long).flatten()
    if landmark_positions.dim() == 2:
        landmark_positions = landmark_positions.unsqueeze(0).expand(
            vertices.shape[0],
            -1,
            -1,
        )
    if landmark_positions.dim() != 3 or landmark_positions.shape[-1] != 3:
        raise ValueError("landmark_positions must have shape [L, 3] or [B, L, 3].")
    if landmark_positions.shape[0] != vertices.shape[0]:
        raise ValueError("landmark_positions batch size must match vertices.")
    if mediapipe_ids.numel() != landmark_positions.shape[1]:
        raise ValueError("mediapipe_ids must match the landmark count.")

    k_nearest = int(config.get("k_nearest", 4))
    include_vectors = bool(config.get("include_vectors", True))
    include_distances = bool(config.get("include_distances", True))
    include_ids = bool(config.get("include_ids", True))
    id_normalizer = float(config.get("id_normalizer", 467.0))
    chunk_size = int(config.get("chunk_size", 50000))
    feature_dim = k_nearest * (
        (3 if include_vectors else 0)
        + (1 if include_distances else 0)
        + (1 if include_ids else 0)
    )

    if k_nearest < 1:
        raise ValueError("model.landmark_features.k_nearest must be at least 1.")
    if feature_dim <= 0:
        raise ValueError(
            "model.landmark_features must include at least one of vectors, "
            "distances, or ids."
        )
    if chunk_size < 1:
        raise ValueError("model.landmark_features.chunk_size must be at least 1.")

    output_batches = []
    for batch_index in range(vertices.shape[0]):
        batch_vertices = vertices[batch_index]
        batch_landmarks = landmark_positions[batch_index]
        if batch_landmarks.numel() == 0:
            output_batches.append(
                vertices.new_zeros((batch_vertices.shape[0], feature_dim))
            )
            continue
        actual_k = min(k_nearest, batch_landmarks.shape[0])
        chunks = []
        for start in range(0, batch_vertices.shape[0], chunk_size):
            chunk = batch_vertices[start : start + chunk_size]
            distances = torch.cdist(
                chunk.unsqueeze(0),
                batch_landmarks.unsqueeze(0),
            ).squeeze(0)
            nearest_distances, nearest_indices = distances.topk(
                actual_k,
                largest=False,
                dim=-1,
            )
            nearest_positions = batch_landmarks.index_select(
                0,
                nearest_indices.reshape(-1),
            ).reshape(chunk.shape[0], actual_k, 3)
            nearest_ids = mediapipe_ids.index_select(
                0,
                nearest_indices.reshape(-1),
            ).reshape(chunk.shape[0], actual_k)

            components = []
            if include_vectors:
                components.append(
                    (nearest_positions - chunk.unsqueeze(1)).reshape(
                        chunk.shape[0],
                        actual_k * 3,
                    )
                )
            if include_distances:
                components.append(nearest_distances)
            if include_ids:
                components.append(
                    nearest_ids.to(dtype=dtype)
                    .div(max(id_normalizer, 1.0))
                    .clamp(0.0, 1.0)
                )
            chunk_features = torch.cat(components, dim=-1)
            if actual_k < k_nearest:
                padding = vertices.new_zeros(
                    (chunk.shape[0], feature_dim - chunk_features.shape[-1])
                )
                chunk_features = torch.cat((chunk_features, padding), dim=-1)
            chunks.append(chunk_features)
        output_batches.append(torch.cat(chunks, dim=0))
    return torch.stack(output_batches, dim=0)

