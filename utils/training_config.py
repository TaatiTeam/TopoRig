"""Fixed settings for the final two-phase experiment.

Only machine paths, run resources, and phase schedules are exposed in train.yaml.
Expanded configurations are stored with checkpoints so existing inference and
resume code can read the complete recipe without relying on this module.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
NON_GAZE_AUS = [*range(12), *range(20, 53)]
REFINEMENT_AUS = [24, 25, 26, 27, 28, 33, 34, 40]
CUSTOM_EXCLUDED_AUS = ["eyeBlink_L", "eyeBlink_R", "eyeLookDown_L", "eyeLookDown_R",
                       "eyeLookIn_L", "eyeLookIn_R", "eyeLookOut_L", "eyeLookOut_R",
                       "eyeLookUp_L", "eyeLookUp_R", "eyeWide_L", "eyeWide_R"]

MODEL = {'facs_dim': 53,
 'hidden_dim': 128,
 'num_blocks': 2,
 'use_global_encoder': True,
 'global_dim': 128,
 'condition_dim': 64,
 'input_dim': 46,
 'landmark_features': {'enabled': True,
                       'k_nearest': 8,
                       'include_vectors': True,
                       'include_distances': True,
                       'include_ids': True,
                       'id_normalizer': 467.0,
                       'chunk_size': 50000,
                       'training_augmentation': {'enabled': True,
                                                 'modality_dropout_probability': 0.15,
                                                 'region_dropout_probability': 0.1,
                                                 'regions': ['left_eye', 'right_eye', 'mouth'],
                                                 'position_jitter_probability': 0.0,
                                                 'position_jitter_std_head_fraction': 0.0,
                                                 'head_up_axis': 2}},
 'position_encoding': {'type': 'fourier',
                       'num_frequencies': 2,
                       'normalize_positions': True,
                       'include_raw': False},
 'dropout': 0.3,
 'output_init': 'zero',
 'output_scale': 1.0,
 'output_init_std': 1e-05}

MESH_LOSS = {'type': 'mae',
 'base_loss_weight': 0.0,
 'target_normalized_region_weight': 1.0,
 'target_normalization_min': 0.001,
 'active_motion_weight': 0.0,
 'active_motion_threshold': 0.001,
 'active_motion_power': 1.0,
 'active_motion_balanced_weight': 0.0,
 'landmark_weight': 5.0,
 'landmark_normalize_by_target': True,
 'landmark_active_motion_weight': 20.0,
 'landmark_active_motion_adaptive_balance': True,
 'landmark_active_motion_adaptive_balance_target_ratio': 5.0,
 'landmark_active_motion_adaptive_balance_min_weight': 0.0,
 'landmark_active_motion_gradient_balance': False,
 'landmark_active_normalize_by_target': True,
 'depth_weight': 0,
 'depth_image_size': [96, 96],
 'depth_up_axis': 'z',
 'false_positive_target_threshold': 0.001,
 'false_positive_pred_threshold': 0.01,
 'geometry': {'enabled': True,
              'target_vertex_mask': False,
              'displacement_l2_weight': 0.0,
              'laplacian_weight': 0.05,
              'edge_length_weight': 0.001,
              'edge_length_relative': True,
              'normal_consistency_weight': 0.0005,
              'magnitude_clamp_weight': 5.0,
              'max_displacement': 0.12,
              'min_edge_length': 0.0001,
              'max_edges': 50000,
              'max_faces': 50000}}

IMAGE_LOSS = {'geometry': {'enabled': True,
              'displacement_l2_weight': 0.05,
              'laplacian_weight': 1.5,
              'edge_length_weight': 0.05,
              'edge_length_relative': True,
              'normal_consistency_weight': 0.05,
              'magnitude_clamp_weight': 0.0,
              'max_displacement': 0.025,
              'min_edge_length': 0.0001,
              'max_edges': 200000,
              'max_faces': 200000,
              'anchor_outside_roi': {'enabled': True,
                                     'hard_mask': False,
                                     'weight': 500.0,
                                     'x_min': 0.15,
                                     'x_max': 0.85,
                                     'y_min': 0.1,
                                     'y_max': 0.55,
                                     'landmark_centered_roi': {'enabled': True,
                                                               'mode': 'topology',
                                                               'topology_hops': 8}},
              'synchronize_coincident_vertices': {'enabled': True, 'tolerance': 1e-06},
              'action_unit_overrides': {'cheekSquint_L': {'anchor_outside_roi': {'landmark_centered_roi': {'topology_hops': 12}}},
                                        'cheekSquint_R': {'anchor_outside_roi': {'landmark_centered_roi': {'topology_hops': 12}}}}},
 'eyelid_closure': {'enabled': True,
                    'weight': 3.0,
                    'displacement_weight': 1.0,
                    'gap_reduction_weight': 2.0,
                    'target_mode': 'observed',
                    'synthetic_upper_closure_fraction': 0.8,
                    'synthetic_lower_closure_fraction': 0.2,
                    'type': 'smooth_l1',
                    'smooth_l1_beta': 0.05,
                    'min_eye_width_px': 1.0,
                    'min_unique_vertices': 14,
                    'min_eye_width_mesh_units': 0.02,
                    'max_neutral_gap_eye_width_ratio': 0.75,
                    'remove_corner_translation': True,
                    'dense_band': {'enabled': False},
                    'depth_anchor': {'enabled': True,
                                     'weight': 2.0,
                                     'topology_hops': 8,
                                     'type': 'smooth_l1',
                                     'smooth_l1_beta': 0.01},
                    'landmark_detection_image_size': [512, 512],
                    'refine_landmarks': False,
                    'min_detection_confidence': 0.5,
                    'mediapipe_landmark_disk_cache': True,
                    'action_unit_overrides': {'cheekSquint_L': {'target_mode': 'observed',
                                                                'weight': 3.0},
                                              'cheekSquint_R': {'target_mode': 'observed',
                                                                'weight': 3.0},
                                              'eyeBlink_L': {'target_mode': 'synthetic_closure',
                                                             'weight': 5.0,
                                                             'synthetic_upper_closure_fraction': 0.8,
                                                             'synthetic_lower_closure_fraction': 0.2},
                                              'eyeBlink_R': {'target_mode': 'synthetic_closure',
                                                             'weight': 5.0,
                                                             'synthetic_upper_closure_fraction': 0.8,
                                                             'synthetic_lower_closure_fraction': 0.2},
                                              'eyeSquint_L': {'target_mode': 'observed',
                                                              'weight': 3.0},
                                              'eyeSquint_R': {'target_mode': 'observed',
                                                              'weight': 3.0},
                                              'eyeWide_L': {'target_mode': 'observed',
                                                            'weight': 3.0},
                                              'eyeWide_R': {'target_mode': 'observed',
                                                            'weight': 3.0}}},
 'flow': {'weight': 0.1,
          'backend': 'waft',
          'waft_precompute_target_cache': True,
          'waft_precompute_target_batch_size': 4,
          'waft_target_disk_cache': True,
          'waft_target_cache_dtype': 'float16',
          'waft_target_memory_cache': False,
          'waft_iters': 3,
          'waft_warmup_model': True,
          'waft_flow_image_size': [512, 512],
          'model_flow_image_size': [128, 128],
          'model_flow_samples_per_face': 1,
          'model_flow_splat_radius_px': 0.75,
          'model_flow_splat_kernel_radius': 1,
          'waft_flow_loss_type': 'charbonnier',
          'waft_flow_loss_epsilon': 0.01,
          'waft_target_image_alignment': {'enabled': True,
                                          'mode': 'landmarks',
                                          'target_source': 'projected_mesh',
                                          'alignment_mesh_up_axis': 'z',
                                          'alignment_mesh_front_axis': 'y',
                                          'alignment_mesh_normalize_on_get': True,
                                          'alignment_mesh_normalized_extent': 2.0,
                                          'alignment_mesh_cache_size': 16,
                                          'fallback_mode': 'bbox',
                                          'landmark_detection_image_size': [512, 512],
                                          'min_detection_confidence': 0.5,
                                          'min_landmarks': 8,
                                          'allow_rotation': False,
                                          'landmark_ids': [10,
                                                           33,
                                                           133,
                                                           263,
                                                           362,
                                                           1,
                                                           4,
                                                           5,
                                                           6,
                                                           61,
                                                           78,
                                                           13,
                                                           14,
                                                           291,
                                                           308,
                                                           152,
                                                           172,
                                                           234,
                                                           454,
                                                           199],
                                          'foreground_threshold': 0.97,
                                          'preserve_aspect': True,
                                          'target_bbox': [0.3, 0.05, 0.7, 0.8],
                                          'use_dataset_mesh': True},
          'waft_mask_foreground': True,
          'waft_background_white_threshold': 0.97,
          'waft_include_full_foreground_in_mask': False,
          'waft_photometric_motion_mask': False,
          'waft_motion_mask_threshold': 0.015,
          'waft_flow_mask_threshold_px': 0.05,
          'waft_flow_mask_threshold_percentile': 0.9,
          'waft_mask_dilate_px': 7,
          'waft_mask_blur_sigma': 1.5}}

IMAGE_ARGS = {'split': 'all',
 'identity_profile': 'historical',
 'precompute_displacement_2d': False,
 'split_csv': None,
 'action_units': [8, 9, 10, 11, 20, 21, 22, 23],
 'weld_coincident_vertices': True,
 'weld_tolerance': 1e-06,
 'welded_landmark_validation': {'enabled': True,
                                'require_cache': True,
                                'skip_invalid': True,
                                'action_units': [10, 11],
                                'min_unique_vertices': 14,
                                'min_eye_width_mesh_extent_ratio': 0.01,
                                'min_gap_eye_width_ratio': 0.05,
                                'max_gap_eye_width_ratio': 0.75,
                                'max_adjacent_eye_width_ratio': 1.25,
                                'repair': {'enabled': True,
                                           'min_component_anchor_votes': 2,
                                           'max_face_components': 8,
                                           'nearest_candidates': 32,
                                           'max_distance_mesh_extent_ratio': 0.08}},
 'neutral_mesh_cache_size': 8,
 'displacement_cache_size': 64,
 'displacement_2d_cache_device': 'cuda',
 'displacement_2d_cache_batch_size': 64,
 'displacement_2d_cache_image_size': [128, 128],
 'use_mediapipe_landmarks': True,
 'generate_mediapipe_landmarks': False,
 'require_mediapipe_landmarks': True,
 'min_mediapipe_landmarks': 8,
 'mesh_up_axis': 'z',
 'mesh_front_axis': 'y',
 'normalize_on_get': True,
 'normalized_extent': 2.0}

def _path(value: str) -> Path:
    expanded = os.path.expandvars(str(value))
    if "$" in expanded:
        raise ValueError(f"Set the environment variable in path {value!r}, or edit config/train.yaml.")
    path = Path(expanded).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def expand_training_config(settings: Mapping[str, Any], phase: int | None = None) -> dict[str, Any]:
    """Build the final recipe, with portable paths and no experiment inheritance."""
    recipe = settings.get("recipe")
    if recipe not in {"final", "stage1", "stage2"}:
        raise ValueError("recipe must be stage1, stage2 or final")
    phase = phase if phase is not None else (2 if recipe == "stage2" else 1)
    if phase not in (1, 2):
        raise ValueError("phase must be 1 or 2")
    unknown = set(settings) - {"schema_version", "recipe", "paths", "run", "phase1", "phase2", "overrides"}
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    paths = settings.get("paths", {})
    run = settings.get("run", {})
    for section, allowed in ((paths, {"data_root", "cache_root", "metadata_root", "output_root", "waft_root", "reference_root"}),
                             (run, {"name", "seed", "device", "gpus", "batch_size", "loader_workers", "cache_workers", "wandb", "initial_checkpoint", "resume_checkpoint"})):
        if set(section) - allowed:
            raise ValueError(f"Unknown training settings: {sorted(set(section) - allowed)}")
    data_root = _path(os.environ.get("TOPORIG_DATA_ROOT", paths.get("data_root", "data")))
    cache = _path(os.environ.get("TOPORIG_CACHE_ROOT", paths.get("cache_root", ".cache")))
    metadata = _path(os.environ.get("TOPORIG_METADATA_ROOT", paths.get("metadata_root", str(data_root / "metadata"))))
    output = _path(os.environ.get("TOPORIG_OUTPUT_ROOT", paths.get("output_root", "runs")))
    waft = _path(os.environ.get("TOPORIG_WAFT_ROOT", paths.get("waft_root", "third_party/WAFT")))
    schedule = settings.get(f"phase{phase}", {})
    if set(schedule) - {"epochs", "learning_rate"}:
        raise ValueError(f"Unknown phase{phase} schedule setting")
    epochs = int(schedule.get("epochs", 20 if phase == 1 else 10))
    lr = float(schedule.get("learning_rate", 5e-4 if phase == 1 else 1e-4))
    gpus = int(run.get("gpus", 1))
    batch_size = int(run.get("batch_size", 4))
    workers = int(run.get("loader_workers", 0))
    cache_workers = int(run.get("cache_workers", 4))
    seed = int(run.get("seed", 7))
    if min(epochs, gpus, batch_size, cache_workers) < 1 or workers < 0 or lr <= 0:
        raise ValueError("Epochs, GPUs, batch size, cache workers and learning rate must be positive; loader workers must be nonnegative.")

    def mesh(source: str, split: str, *, refine: bool = False) -> dict[str, Any]:
        ict = source == "ict"
        result = {
            "data_root": str(data_root), "mesh_source": source,
            "asset_cache_dir": str(cache / "assets"),
            "mesh_dir": str(data_root / f"meshes_{source}"),
            "cache_dir": str(cache / f"mesh_{source}"),
            "exclude_mesh_ids_file": str(metadata / ("bad_ict_source_landmark_meshes.txt" if ict else "bad_blendshape_landmark_meshes.txt")),
            "cache_format": "hdf5", "cache_compression": "gzip", "cache_compression_level": 1,
            "hdf5_blendshape_chunk_vertices": 8192, "blendshape_name_mapping": "canonical",
            "cache_workers": cache_workers, "cache_random_seed": seed,
            "mesh_up_axis": "z", "mesh_front_axis": "y", "normalize_on_get": True,
            "normalized_extent": 2.0, "use_mediapipe_landmarks": True,
            "generate_mediapipe_landmarks": False, "action_units": list(REFINEMENT_AUS if refine else NON_GAZE_AUS),
            "drop_mouth_probability": 0.0, "cut_eyeballs_probability": 0.0,
            "topology_augmentation_probability": 0.0, "shape_augmentation_probability": 0.0,
        }
        if ict:
            result.update(mediapipe_landmark_dir=str(metadata / "ict_landmarks"),
                          trust_existing_cache=True, rigged_mesh_cache_size=64 if refine else (384 if split == "train" else 16))
        elif refine:
            result["trust_existing_cache"] = True
        else:
            result["exclude_action_units"] = list(CUSTOM_EXCLUDED_AUS)
        if ict and split == "train" and not refine:
            result.update(drop_mouth_probability=0.5, cut_eyeballs_probability=0.5,
                          topology_augmentation_probability=0.5, topology_augmentation_mode="face_split",
                          topology_augmentation_face_split_probability=0.05,
                          topology_augmentation_max_vertices=50000,
                          shape_augmentation_probability=0.75, shape_augmentation_scale=0.16,
                          shape_augmentation_num_anchors=24, shape_augmentation_smoothness=0.35)
        return result

    data: dict[str, Any] = {"data_root": str(data_root), "asset_cache_dir": str(cache / "assets"),
                            "modalities": ["mesh"] if phase == 1 else ["image", "mesh"]}
    if phase == 2:
        data["pair_image_with_mesh_by_action_unit"] = True
    for split in ("train", "val"):
        transfer = mesh("custom", split)
        transfer.update(enabled=True, loss_weight_key="blendshape_mesh_3d", split=split,
                        split_csv=str(metadata / "custom_split.csv"))
        split_config = {"split": split, "split_csv": str(metadata / "ict_split.csv"),
                        "mesh_args": mesh("ict", split), "extra_mesh_args": {"blendshape_transfer": transfer},
                        "loader": {"batch_size": batch_size,
                                   "shuffle": split == "train", "num_workers": workers, "pin_memory": True}}
        if phase == 2:
            image_args = copy.deepcopy(IMAGE_ARGS)
            image_args.update(cache_dir=str(cache / "image"),
                              exclude_person_ids_file=str(metadata / "bad_waft_blink_left_alignment_person_ids.txt"))
            from dataset.hf_assets import asset_cache_namespace
            image_namespace = asset_cache_namespace(data_root)
            image_args["welded_landmark_validation"]["cache_dir"] = str(cache / "validated_eyelids" / image_namespace)
            split_config["image_args"] = image_args
            for source, name in ((("ict", "high_error_jaw_mouth_ict"), ("custom", "high_error_jaw_mouth_pixel3d_100k")) if split == "train" else ()):
                extra = mesh(source, split, refine=True)
                extra.update(enabled=True, loss_weight_key=f"{name}_mesh_3d", split=split,
                             split_csv=str(metadata / f"{'ict' if source == 'ict' else 'custom'}_split.csv"))
                split_config["extra_mesh_args"][f"{name}_refinement"] = extra
        data[split] = split_config

    loss = {"weights": {"mesh_3d": 1.0, "blendshape_mesh_3d": 1.0, "image_2d": float(phase == 2)},
            "mesh": copy.deepcopy(MESH_LOSS)}
    if phase == 2:
        loss["weights"].update(high_error_jaw_mouth_ict_mesh_3d=1.0, high_error_jaw_mouth_pixel3d_100k_mesh_3d=1.0)
        loss["image"] = copy.deepcopy(IMAGE_LOSS)
        loss["image"]["eyelid_closure"]["mediapipe_landmark_cache_dir"] = str(cache / "image_eyelids" / image_namespace)
        loss["image"]["flow"].update(waft_root_dir=str(waft), waft_config_path=str(waft / "config/a1/tar-c-t.json"),
                                    waft_checkpoint_path=str(waft / "ckpts/waft_a1_adaptation.pth"),
                                    waft_target_cache_dir=str(cache / "waft_targets" / image_namespace))
    config = {
        "schema_version": 1, "recipe": f"stage{phase}", "phase": phase,
        "project": {"run_name": str(run.get("name", f"phase{phase}")), "output_dir": str(output), "seed": seed},
        "data": data, "model": copy.deepcopy(MODEL),
        "training": {"device": str(run.get("device", "auto")), "epochs": epochs,
                     "action_unit_scale": 1.0, "grad_clip_norm": 1.0, "log_every_steps": 10},
        "distributed": {"enabled": gpus > 1, "nproc_per_node": gpus, "find_unused_parameters": False,
                        "static_graph": False, "sync_batchnorm": False, "broadcast_buffers": False, "timeout_seconds": 14400},
        "optimizer": {"type": "adamw", "lr": lr, "weight_decay": 0.0,
                      "scheduler": {"type": "cosine", "t_max": epochs, "eta_min": 1e-6}},
        "loss": loss,
        "rendering": {"backend": "auto", "shared_fit_to_view": True,
                      "train_image_size": [256, 256], "validation_image_size": [256, 256],
                      "kwargs": {"torch_samples_per_face": 2 if phase == 1 else 1,
                                 "splat_radius_px": 0.75, "splat_kernel_radius": 1, "chunk_size": 200000}},
        "validation": {"every_epochs": 1, "max_batches": 128, "primary_metric": "mesh_3d_mae", "mae_only": True,
                       "prediction_glb": {"enabled": False}},
        "visualization": {"mesh_up_axis": "z", "base_sample": "mesh" if phase == 1 else "image"},
        "wandb": {"enabled": bool(run.get("wandb", False)), "project": "toporig", "log_every_steps": 10,
                  "log_render_every_epochs": 1},
        "checkpoint": {"save_every_epochs": 1},
    }
    from utils.visualization_defaults import STAGE_VISUALIZATION
    from dataset.layout import reference_assets_root
    display = copy.deepcopy(STAGE_VISUALIZATION[phase])
    config["validation"]["prediction_glb"] = display["prediction_glb"]
    config["validation"]["prediction_glb"]["output_dir"] = str(output / "prediction_glbs")
    config["visualization"] = display["visualization"]
    bundled_references = data_root / "reference_assets"
    references = (_path(os.environ["TOPORIG_REFERENCE_ROOT"]) if os.environ.get("TOPORIG_REFERENCE_ROOT")
                  else _path(paths["reference_root"]) if paths.get("reference_root")
                  else bundled_references if bundled_references.is_dir()
                  else reference_assets_root())
    config["visualization"].update(custom_mesh_metahuman_sample_dir=str(references / "metahuman"),
                                  custom_mesh_landmark_cache_dir=str(cache / "demo_landmarks"),
                                  custom_mesh_mediapipe_mapper_path=str(ROOT / "map_mediapipe_landmarks.py"))
    for key in ("initial_checkpoint", "resume_checkpoint"):
        if run.get(key):
            config["training"][key] = str(_path(run[key]))
    if run.get("initial_checkpoint") and run.get("resume_checkpoint"):
        raise ValueError("Choose initialization or full resume, not both")
    from utils.config import deep_merge_config, resolve_config_paths
    overrides = settings.get("overrides", {})
    if not isinstance(overrides, Mapping) or set(overrides) - set(config):
        raise ValueError("overrides must use resolved configuration section names")
    config = resolve_config_paths(deep_merge_config(config, overrides), ROOT)
    resolve_model_dimensions(config)
    return config


def resolve_model_dimensions(config: dict[str, Any]) -> None:
    """The input contains XYZ, normals, and the selected landmark channels."""
    from utils.fbx_to_tensor import AU_NAME
    model = config["model"]
    features = model.get("landmark_features", {})
    channels = (3 * bool(features.get("include_vectors", True))
                + bool(features.get("include_distances", True))
                + bool(features.get("include_ids", True)))
    model["input_dim"] = 6 + (int(features.get("k_nearest", 8)) * channels if features.get("enabled", False) else 0)
    model["facs_dim"] = len(AU_NAME)


def check_training_assets(config: Mapping[str, Any]) -> None:
    """Fail before launching workers if the published metadata or WAFT files are missing."""
    data = config["data"]
    required = [Path(data["data_root"])]
    for split in ("train", "val"):
        settings = data[split]
        streams = [(settings.get("mesh_args"), settings.get("split_csv"))]
        streams += [(args, args.get("split_csv")) for args in (settings.get("extra_mesh_args") or {}).values() if args and args.get("enabled", True)]
        for args, split_csv in streams:
            if not args:
                continue
            if split_csv:
                required.append(Path(split_csv))
            for key in ("exclude_mesh_ids_file", "mediapipe_landmark_dir"):
                if args.get(key):
                    required.append(Path(args[key]))
        if settings.get("image_args", {}).get("exclude_person_ids_file"):
            required.append(Path(settings["image_args"]["exclude_person_ids_file"]))
    if config["phase"] == 2 and config["loss"]["image"]["flow"].get("weight", 0) > 0:
        flow = config["loss"]["image"]["flow"]
        required += [Path(flow["waft_config_path"]), Path(flow["waft_checkpoint_path"])]
        waft_config = Path(flow["waft_config_path"])
        if waft_config.is_file():
            backbone = json.loads(waft_config.read_text()).get("dav2_backbone")
            if backbone in {"vits", "vitb", "vitl", "vitg"}:
                required.append(Path(flow["waft_root_dir"]) / "depth-anything-ckpts"
                                / f"depth_anything_v2_{backbone}.pth")
    missing = sorted({str(path) for path in required if not path.exists()})
    if missing:
        raise FileNotFoundError("Missing training assets:\n  " + "\n  ".join(missing) + "\nSee README.md for dataset metadata and phase preparation.")
