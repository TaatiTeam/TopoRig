from __future__ import annotations

import argparse
import copy
import datetime as dt
import faulthandler
import hashlib
import itertools
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
import numpy as np
from PIL import Image, ImageDraw
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from dataset.image_dataset import _load_glb_for_model, _load_neutral_mesh_for_model
from dataset.mesh_dataset import (
    _compute_vertex_normals,
    _load_landmark_payload,
    _transform_model_sample,
)
from dataset.unified_dataset import UnifiedDataset
from model.toporig import TopoRig
from utils.depth_loss_utils import (
    depth_projection_depth_bounds,
    depth_projection_transform,
    depth_difference_l1_loss,
    render_rasterized_depth_map,
)
from utils.fbx_to_tensor import AU_NAME, fbx_to_tensor
from utils.eyelid_landmarks import (
    BLINK_LEFT_EYELIDS,
    BLINK_RIGHT_EYELIDS,
    eyelid_landmarks_for_action_unit,
)
from utils.eye_gaze_uv_warp import (
    LEFT_EYE_GAZE_LANDMARKS,
    RIGHT_EYE_GAZE_LANDMARKS,
)
from utils.eye_gaze_surface_anchors import (
    interpolate_surface_anchors,
    select_surface_anchors,
)
from utils.mouth_landmark_validation import MOUTH_LANDMARK_IDS
from dataset.hf_assets import canonical_identity

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    tqdm = None


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "train.yaml"
from dataset.layout import reference_assets_root

DEFAULT_CUSTOM_VIS_METAHUMAN_SAMPLE_DIR = reference_assets_root() / "metahuman"
DEFAULT_CUSTOM_VIS_LANDMARK_CACHE_DIR = ROOT / ".cache" / "demo_landmarks"
DEFAULT_CUSTOM_VIS_MEDIAPIPE_MAPPER = ROOT / "map_mediapipe_landmarks.py"
CUSTOM_VIS_SUPPORTED_MESH_SUFFIXES = {".glb", ".fbx"}
MODEL_RUNTIME_CONFIG_KEYS = {"landmark_features", "type"}
WAFT_FLOW_MODELS: dict[tuple[str, str, str, int, str], nn.Module] = {}
WAFT_TARGET_FLOW_CACHE: dict[
    tuple[str, str, tuple[int, int], tuple[int, int], str, str, int, str, str],
    torch.Tensor,
] = {}
WAFT_TARGET_MASK_CACHE: dict[
    tuple[str, str, tuple[int, int], str, str, Optional[int]],
    torch.Tensor,
] = {}
WAFT_IMAGE_LANDMARK_CACHE: dict[
    tuple[str, tuple[int, int], str, bool, float], dict[int, tuple[float, float]]
] = {}
IMAGE_MEDIAPIPE_LANDMARK_CACHE_FORMAT_VERSION = 1
WAFT_ALIGNMENT_TRANSFORM_CACHE: dict[
    tuple[str, tuple[int, int], tuple[int, int], str, str, str],
    Optional[torch.Tensor],
] = {}
WAFT_ALIGNMENT_MESH_CACHE: OrderedDict[
    tuple[str, str, str, str, bool, float],
    tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
] = OrderedDict()
WAFT_ALIGNMENT_WARNING_KEYS: set[str] = set()
WAFT_RENDER_ALIGNMENT_TARGET_SOURCES = {"render", "neutral_render", "render_landmarks"}
WAFT_PROJECTED_MESH_ALIGNMENT_TARGET_SOURCES = {
    "mesh",
    "mesh_landmarks",
    "projected_mesh",
}
WAFT_EXTERNAL_MESH_ALIGNMENT_TARGET_SOURCES = {
    "alignment_mesh",
    "center_mesh",
    "centering_mesh",
    "downsampled_mesh",
}
WAFT_TARGET_CACHE_CONTROL_KEYS = {
    "waft_precompute_target_cache",
    "waft_precompute_target_batch_size",
    "waft_target_disk_cache",
    "waft_target_cache_dir",
    "waft_target_cache_dtype",
    "waft_target_memory_cache",
    "waft_warmup_model",
}
WAFT_ALIGNMENT_CACHE_CONTROL_KEYS = {
    "alignment_mesh_cache_size",
    "alignment_transform_cache_dir",
    "alignment_transform_disk_cache",
}
MESH_COMPONENT_CACHE: dict[tuple[int, tuple[int, ...], str], torch.Tensor] = {}
MESH_COMPONENT_PTR_CACHE: dict[tuple[int, tuple[int, ...], int], torch.Tensor] = {}


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = ""

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def timing_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("timing", {})
    return value if isinstance(value, Mapping) else {}


def diagnostics_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("diagnostics", {})
    return value if isinstance(value, Mapping) else {}


def diagnostics_float(
    config: Mapping[str, Any],
    key: str,
    default: Optional[float] = None,
) -> Optional[float]:
    value = diagnostics_config(config).get(key, default)
    if value in (None, ""):
        return default
    return float(value)


def diagnostics_int(
    config: Mapping[str, Any],
    key: str,
    default: int = 0,
) -> int:
    value = diagnostics_config(config).get(key, default)
    if value in (None, ""):
        return default
    return int(value)


def timing_enabled(config: Mapping[str, Any]) -> bool:
    return bool(timing_config(config).get("enabled", False))


def timing_sync_cuda(config: Mapping[str, Any]) -> bool:
    return bool(timing_config(config).get("sync_cuda", False))


def synchronize_for_timing(config: Mapping[str, Any], device: torch.device) -> None:
    if (
        timing_enabled(config)
        and timing_sync_cuda(config)
        and device.type == "cuda"
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize(device)


def synchronize_for_diagnostics(config: Mapping[str, Any], device: torch.device) -> None:
    if (
        bool(diagnostics_config(config).get("sync_cuda", False))
        and device.type == "cuda"
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize(device)


def diagnostic_timer_start(config: Mapping[str, Any], device: torch.device) -> float:
    synchronize_for_diagnostics(config, device)
    return time.perf_counter()


def diagnostic_timer_elapsed(
    started_at: float,
    config: Mapping[str, Any],
    device: torch.device,
) -> float:
    synchronize_for_diagnostics(config, device)
    return time.perf_counter() - started_at


def configure_stack_dumps(
    config: Mapping[str, Any],
    distributed: "DistributedContext",
) -> None:
    interval = diagnostics_float(config, "stack_dump_interval_seconds", 0.0)
    if interval is None or interval <= 0.0:
        return
    faulthandler.enable(all_threads=True)
    print(
        "[INFO] Python per-batch stack dump watchdog enabled "
        f"rank={distributed.rank} timeout={interval:g}s",
        flush=True,
    )


def arm_stack_dump_watchdog(config: Mapping[str, Any]) -> None:
    interval = diagnostics_float(config, "stack_dump_interval_seconds", 0.0)
    if interval is None or interval <= 0.0:
        return
    faulthandler.cancel_dump_traceback_later()
    faulthandler.dump_traceback_later(interval, repeat=False)


def cancel_stack_dumps() -> None:
    try:
        faulthandler.cancel_dump_traceback_later()
    except Exception:
        pass


def timing_start(
    config: Mapping[str, Any],
    device: torch.device,
) -> Optional[float]:
    if not timing_enabled(config):
        return None
    synchronize_for_timing(config, device)
    return time.perf_counter()


def timing_stop(
    metrics: MutableMapping[str, float],
    name: str,
    started_at: Optional[float],
    config: Mapping[str, Any],
    device: torch.device,
) -> None:
    if started_at is None:
        return
    synchronize_for_timing(config, device)
    key = f"timing_{name}_seconds"
    metrics[key] = float(metrics.get(key, 0.0)) + (time.perf_counter() - started_at)


@contextmanager
def timed_metric(
    metrics: MutableMapping[str, float],
    name: str,
    config: Mapping[str, Any],
    device: torch.device,
):
    started_at = timing_start(config, device)
    try:
        yield
    finally:
        timing_stop(metrics, name, started_at, config, device)


from utils.config import load_config, deep_merge_config, resolve_config_paths, save_config
from utils.mesh_losses import (
    active_motion_error_mean,
    active_motion_mask,
    active_motion_vertex_weights,
    edge_delta_smoothness_loss,
    edge_length_preservation_loss,
    elementwise_loss,
    face_normal_consistency_loss,
    face_normals,
    landmark_active_motion_loss,
    landmark_active_motion_mask,
    mesh_loss_fn,
    normalized_vertex_weights,
    scalar_loss_gradient_norm,
    target_loss_normalizer,
    target_motion_scale,
    target_normalized_region_loss,
)
from utils.input_features import (
    landmark_feature_config,
    landmark_features_enabled,
    landmark_feature_dim,
    build_model_landmark_features,
    normalize_landmark_feature_regions,
    landmark_random_event,
    drop_model_landmark_regions,
    jitter_model_landmark_positions,
    permute_model_landmark_positions,
    landmark_relative_vertex_features,
)
from utils.input_features import LANDMARK_FEATURE_REGION_IDS



def main() -> None:
    args = parse_args()
    config = load_config(args.config, phase=args.phase)
    for key, value in (("initial_checkpoint", args.initial_checkpoint), ("resume_checkpoint", args.resume)):
        if value is not None:
            config["training"][key] = str(value.expanduser().resolve())
    if args.action_units is not None:
        config = apply_action_units_override(config, args.action_units)
    elif args.debug_one_au is not None:
        config = apply_action_units_override(config, [args.debug_one_au])
    if args.print_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return
    if config.get("schema_version") == 1:
        from utils.training_config import check_training_assets
        if config["phase"] == 2 and not args.cache_only and not (config["training"].get("initial_checkpoint") or config["training"].get("resume_checkpoint")):
            raise ValueError("Phase 2 requires --initial-checkpoint PHASE1_BEST.pt or --resume PHASE2_LATEST.pt.")
        check_training_assets(config)
    use_wandb = wandb_enabled(args, config)
    maybe_relaunch_for_configured_distributed(args, config)
    distributed = distributed_context_from_env()
    device = resolve_device(str(config["training"].get("device", "auto")), distributed)
    distributed = initialize_distributed(distributed, device, config)
    configure_stack_dumps(config, distributed)
    set_seed(int(config["project"].get("seed", 7)))

    wandb_run = None
    try:
        setup_timings: dict[str, float] = {}
        with timed_metric(setup_timings, "setup_run_dir", config, device):
            run_dir = make_run_dir(config, distributed)
            if distributed.is_main:
                save_config(config, run_dir / "train.yaml")

        with timed_metric(setup_timings, "setup_build_datasets", config, device):
            train_dataset, val_dataset = build_datasets(config, distributed)
        with timed_metric(setup_timings, "setup_build_loaders", config, device):
            train_loader = build_loader(config, train_dataset, "train", distributed)
            val_loader = build_loader(config, val_dataset, "val", distributed)

        with timed_metric(setup_timings, "setup_build_model", config, device):
            model: nn.Module = build_model(config, train_dataset, device)
            if not args.cache_only:
                maybe_load_initial_model_checkpoint(model, config, device)
        if distributed.is_main:
            print(format_model_summary(model))
        with timed_metric(setup_timings, "setup_wrap_model", config, device):
            model = wrap_model_for_distributed(model, distributed, device, config)
        with timed_metric(setup_timings, "setup_build_optimizer", config, device):
            optimizer = build_optimizer(model, config["optimizer"])
        with timed_metric(setup_timings, "setup_warmup_image_flow", config, device):
            warmup_image_flow_backend(config, device, distributed)
        with timed_metric(setup_timings, "setup_precache_targets", config, device):
            precache_image_flow_targets(
                config=config,
                datasets=(train_dataset, val_dataset),
                device=device,
                distributed=distributed,
                cache_shard_rank=args.cache_shard_rank,
                cache_shard_count=args.cache_shard_count,
            )
            if args.cache_only:
                precache_image_mediapipe_landmarks(
                    config=config,
                    datasets=(train_dataset, val_dataset),
                    device=device,
                    distributed=distributed,
                    cache_shard_rank=args.cache_shard_rank,
                    cache_shard_count=args.cache_shard_count,
                )

        if args.cache_only:
            if distributed.is_main:
                print("[INFO] Cache-only setup completed successfully.", flush=True)
            return

        if use_wandb and distributed.is_main:
            with timed_metric(setup_timings, "setup_wandb_init", config, device):
                wandb_run = init_wandb(config, run_dir)

        if distributed.is_main:
            print(f"[INFO] Using device: {device}")
            if distributed.enabled:
                print(
                    "[INFO] DDP enabled: "
                    f"world_size={distributed.world_size}, backend={distributed.backend}"
                )
                print(
                    "[INFO] DDP options: "
                    f"{dict(config.get('distributed', {}))}"
                )
            print(f"[INFO] Run directory: {run_dir}")
            print(f"[INFO] Train samples: {len(train_dataset)}")
            print(f"[INFO] Val samples:   {len(val_dataset)}")
            if timing_enabled(config):
                print(format_metrics("[timing] setup", setup_timings))
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty.")

        num_epochs = int(config["training"]["epochs"])
        scheduler = build_lr_scheduler(optimizer, config["optimizer"], num_epochs)
        start_epoch, global_step, best_val_loss = maybe_resume_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
        )
        validation_metric_name = validation_primary_metric_name(config)
        if distributed.is_main:
            print(
                "[INFO] Validation primary metric: "
                f"{validation_metric_name} (lower is better)",
                flush=True,
            )
        log_every_steps = int(config["training"].get("log_every_steps", 10))
        val_every_epochs = int(config["validation"].get("every_epochs", 1))
        save_every_epochs = int(config["checkpoint"].get("save_every_epochs", 1))

        for epoch in range(start_epoch, num_epochs + 1):
            set_loader_epoch(train_loader, epoch)
            train_metrics, global_step = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                config=config,
                device=device,
                epoch=epoch,
                global_step=global_step,
                log_every_steps=log_every_steps,
                wandb_run=wandb_run,
                distributed=distributed,
                validation_dataset=val_dataset,
                run_dir=run_dir,
            )
            if distributed.is_main:
                print(format_metrics(f"[train] epoch {epoch}", train_metrics))

            val_metrics: Optional[dict[str, float]] = None
            if val_every_epochs > 0 and epoch % val_every_epochs == 0:
                validation_timings: dict[str, float] = {}
                with timed_metric(
                    validation_timings,
                    "validation_total",
                    config,
                    device,
                ):
                    val_metrics = validate(
                        model=model,
                        loader=val_loader,
                        config=config,
                        device=device,
                        distributed=distributed,
                    )
                val_metrics.update(validation_timings)
                val_metrics["lr"] = current_learning_rate(optimizer)
                if distributed.is_main:
                    print(format_metrics(f"[val]   epoch {epoch}", val_metrics))
                    if wandb_run is not None:
                        wandb_run.log(
                            {f"val/{key}": value for key, value in val_metrics.items()},
                            step=global_step,
                        )
                        maybe_log_validation_render(
                            wandb_run=wandb_run,
                            model=unwrap_model(model),
                            dataset=val_dataset,
                            config=config,
                            device=device,
                            epoch=epoch,
                            global_step=global_step,
                        )
                    export_timings: dict[str, float] = {}
                    with timed_metric(
                        export_timings,
                        "validation_export_glb",
                        config,
                        device,
                    ):
                        maybe_export_validation_prediction_glbs(
                            model=unwrap_model(model),
                            dataset=val_dataset,
                            config=config,
                            device=device,
                            run_dir=run_dir,
                            epoch=epoch,
                        )
                    if timing_enabled(config) and export_timings:
                        print(format_metrics(f"[timing] epoch {epoch}", export_timings))

            if scheduler is not None:
                scheduler.step()

            validation_improved = (
                val_metrics is not None
                and "primary_metric" in val_metrics
                and val_metrics["primary_metric"] < best_val_loss
            )
            if validation_improved:
                best_val_loss = val_metrics["primary_metric"]

            if (
                distributed.is_main
                and save_every_epochs > 0
                and epoch % save_every_epochs == 0
            ):
                checkpoint_timings: dict[str, float] = {}
                with timed_metric(
                    checkpoint_timings,
                    "checkpoint_latest",
                    config,
                    device,
                ):
                    save_checkpoint(
                        run_dir / "checkpoints" / "latest.pt",
                        model=model,
                        config=config,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                    )
                if timing_enabled(config):
                    print(
                        format_metrics(
                            f"[timing] epoch {epoch}",
                            checkpoint_timings,
                        )
                    )

            if (
                distributed.is_main
                and validation_improved
            ):
                checkpoint_timings = {}
                with timed_metric(
                    checkpoint_timings,
                    "checkpoint_best",
                    config,
                    device,
                ):
                    save_checkpoint(
                        run_dir / "checkpoints" / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                    )
                if timing_enabled(config):
                    print(
                        format_metrics(
                            f"[timing] epoch {epoch}",
                            checkpoint_timings,
                        )
                    )

        if distributed.is_main:
            final_timings: dict[str, float] = {}
            with timed_metric(final_timings, "final_checkpoint", config, device):
                save_checkpoint(
                    run_dir / "checkpoints" / "final.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    scheduler=scheduler,
                    epoch=num_epochs,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                )
            with timed_metric(final_timings, "final_export_glb", config, device):
                maybe_export_validation_prediction_glbs(
                    model=unwrap_model(model),
                    dataset=val_dataset,
                    config=config,
                    device=device,
                    run_dir=run_dir,
                    epoch=num_epochs,
                    force=True,
                    label="final",
                )
            if timing_enabled(config):
                print(format_metrics("[timing] final", final_timings))
    finally:
        cancel_stack_dumps()
        if wandb_run is not None:
            wandb_run.finish()
        cleanup_distributed(distributed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TopoRig.")
    parser.add_argument("--phase", type=int, choices=(1, 2), default=None,
                        help="Override the stage selected by the recipe.")
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument("--initial-checkpoint", type=Path,
                             help="Initialize model weights, normally from phase 1 best.pt.")
    checkpoints.add_argument("--resume", type=Path, help="Resume model, optimizer and RNG state from latest.pt.")
    parser.add_argument("--print-config", action="store_true", help="Print the expanded recipe and exit without loading assets.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to a YAML training config.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log losses and validation renders to Weights & Biases.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Build dataset caches and precompute image-flow targets, then exit.",
    )
    parser.add_argument(
        "--single-process",
        action="store_true",
        help=(
            "Do not relaunch with torchrun even when distributed training is "
            "enabled in the config. Intended for independent cache shards and "
            "interactive single-GPU runs."
        ),
    )
    parser.add_argument(
        "--cache-shard-rank",
        type=int,
        default=0,
        help="Zero-based cache shard assigned to this independent process.",
    )
    parser.add_argument(
        "--cache-shard-count",
        type=int,
        default=1,
        help="Total number of independent cache shards.",
    )
    parser.add_argument(
        "--debug-oneAU",
        "--debug-one-au",
        dest="debug_one_au",
        type=parse_one_action_unit,
        help=(
            "Debug training on a single action unit, e.g. --debug-oneAU 26. "
            "AU names such as jawOpen are also accepted."
        ),
    )
    parser.add_argument(
        "--AUs",
        "--aus",
        dest="action_units",
        nargs="+",
        type=parse_one_action_unit,
        help=(
            "Train only on the listed action units, e.g. --AUs 26 52 50 1 0. "
            "AU names such as jawOpen are also accepted."
        ),
    )
    args = parser.parse_args()
    if args.debug_one_au is not None and args.action_units is not None:
        parser.error("Use either --debug-oneAU or --AUs, not both.")
    if args.cache_shard_count < 1:
        parser.error("--cache-shard-count must be at least 1.")
    if not 0 <= args.cache_shard_rank < args.cache_shard_count:
        parser.error(
            "--cache-shard-rank must be between 0 and "
            "--cache-shard-count - 1."
        )
    if (
        args.cache_shard_rank != 0 or args.cache_shard_count != 1
    ) and not args.cache_only:
        parser.error("Cache sharding requires --cache-only.")
    return args


def maybe_relaunch_for_configured_distributed(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> None:
    command = configured_distributed_launch_command(args, config)
    if command is None:
        return
    print("[INFO] distributed.enabled=true; relaunching with torchrun:")
    print("[INFO] " + " ".join(command), flush=True)
    os.execvp(command[0], command)


def configured_distributed_launch_command(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    env: Optional[Mapping[str, str]] = None,
) -> Optional[list[str]]:
    env = os.environ if env is None else env
    distributed_config = config.get("distributed", {})
    if not isinstance(distributed_config, Mapping):
        distributed_config = {}

    enabled = bool(distributed_config.get("enabled", False))
    world_size = int(env.get("WORLD_SIZE", "1"))
    if bool(getattr(args, "single_process", False)):
        if world_size > 1:
            raise RuntimeError(
                "--single-process cannot be used inside an active distributed "
                f"environment (WORLD_SIZE={world_size})."
            )
        return None
    if world_size > 1:
        if not enabled:
            raise RuntimeError(
                "PyTorch distributed environment is active "
                f"(WORLD_SIZE={world_size}), but distributed.enabled is false "
                "in the config. Set distributed.enabled: true or launch with "
                "plain python for single-GPU training."
            )
        return None

    if not enabled:
        return None

    nproc_per_node = int(distributed_config.get("nproc_per_node", 4))
    if nproc_per_node < 2:
        raise ValueError("distributed.nproc_per_node must be at least 2 when enabled.")

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc_per_node}",
        str(Path(__file__).resolve()),
        "--config",
        str(args.config),
    ]
    if getattr(args, "phase", None) is not None:
        command.extend(["--phase", str(args.phase)])
    for attribute, flag in (("initial_checkpoint", "--initial-checkpoint"), ("resume", "--resume")):
        value = getattr(args, attribute, None)
        if value is not None:
            command.extend([flag, str(value)])
    if wandb_enabled(args, config):
        command.append("--use-wandb")
    if bool(getattr(args, "cache_only", False)):
        command.append("--cache-only")
    cache_shard_rank = int(getattr(args, "cache_shard_rank", 0))
    cache_shard_count = int(getattr(args, "cache_shard_count", 1))
    if cache_shard_rank != 0 or cache_shard_count != 1:
        command.extend(["--cache-shard-rank", str(cache_shard_rank)])
        command.extend(["--cache-shard-count", str(cache_shard_count)])
    debug_one_au = getattr(args, "debug_one_au", None)
    if debug_one_au is not None:
        command.extend(["--debug-oneAU", str(debug_one_au)])
    action_units = getattr(args, "action_units", None)
    if action_units is not None:
        command.append("--AUs")
        command.extend(str(action_unit) for action_unit in action_units)
    return command


def wandb_enabled(args: argparse.Namespace, config: Mapping[str, Any]) -> bool:
    wandb_config = config.get("wandb", {})
    config_enabled = (
        bool(wandb_config.get("enabled", False))
        if isinstance(wandb_config, Mapping)
        else False
    )
    return bool(getattr(args, "use_wandb", False) or config_enabled)


def wandb_scalar_log_every_steps(config: Mapping[str, Any]) -> int:
    wandb_config = config.get("wandb", {})
    if not isinstance(wandb_config, Mapping):
        return 1
    value = wandb_config.get("log_every_steps", 1)
    if value in (None, ""):
        return 1
    interval = int(value)
    if interval < 0:
        raise ValueError("wandb.log_every_steps must be non-negative.")
    return interval






def config_for_action_unit(
    config: Mapping[str, Any],
    action_unit_id: int,
) -> dict[str, Any]:
    resolved = dict(config)
    overrides = config.get("action_unit_overrides", {})
    if not isinstance(overrides, Mapping):
        return resolved

    action_unit_name = AU_NAME.get(int(action_unit_id))
    keys: list[Any] = [
        int(action_unit_id),
        str(int(action_unit_id)),
        f"AU{int(action_unit_id)}",
        f"au{int(action_unit_id)}",
    ]
    if action_unit_name is not None:
        keys.append(action_unit_name)
    for key in keys:
        override = overrides.get(key)
        if isinstance(override, Mapping):
            return deep_merge_config(resolved, override)
    return resolved


def toporig_model_config(model_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the keys accepted by TopoRig.__init__.

    The ``model`` YAML section can also carry runtime feature-builder settings,
    such as ``landmark_features``. Those settings control train.py input
    assembly and should not be passed into the neural net constructor.
    """
    return {
        key: value
        for key, value in dict(model_config).items()
        if key not in MODEL_RUNTIME_CONFIG_KEYS
    }


def build_model(
    config: Mapping[str, Any],
    train_dataset: Dataset,
    device: torch.device,
) -> nn.Module:
    model_config = config.get("model", {})
    model_type = str(model_config.get("type", "toporig")).lower()
    if model_type in {"toporig", "default"}:
        return TopoRig(**toporig_model_config(model_config)).to(device)
    raise ValueError(f"Unsupported model.type: {model_type!r}.")




def format_model_summary(model: nn.Module) -> str:
    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    summary = (
        f"[INFO] Model: {model.__class__.__name__} "
        f"({trainable_count:,} trainable parameters)"
    )
    max_displacement = getattr(model, "max_displacement", None)
    if max_displacement is not None:
        summary += f", max_displacement={float(max_displacement):.6g}"
    smooth_iterations = int(getattr(model, "smooth_iterations", 0))
    if smooth_iterations > 0:
        summary += (
            f", smooth_iterations={smooth_iterations}, "
            f"smooth_lambda={float(getattr(model, 'smooth_lambda', 0.0)):.3g}"
        )
    return summary






def apply_action_units_override(
    config: Mapping[str, Any],
    action_units: Sequence[int],
) -> dict[str, Any]:
    action_units = list(action_units)
    if not action_units:
        raise ValueError("action_units must not be empty.")

    updated = copy.deepcopy(config)
    data_config = updated["data"]
    modalities = normalize_modalities(data_config.get("modalities", ("image", "mesh")))

    for split_name in ("train", "val"):
        split_config = data_config.get(split_name)
        if not isinstance(split_config, MutableMapping):
            continue

        if "image" in modalities:
            image_args = dict(split_config.get("image_args") or {})
            image_args["action_units"] = action_units
            split_config["image_args"] = image_args

        if "mesh" in modalities:
            mesh_args = dict(split_config.get("mesh_args") or {})
            mesh_args["action_units"] = action_units
            split_config["mesh_args"] = mesh_args

            extra_mesh_args = split_config.get("extra_mesh_args")
            if extra_mesh_args is not None:
                split_config["extra_mesh_args"] = apply_extra_mesh_action_units_override(
                    extra_mesh_args,
                    action_units,
                )

    visualization = updated.setdefault("visualization", {})
    visualization["action_units"] = action_units

    validation = updated.get("validation")
    if isinstance(validation, MutableMapping):
        prediction_glb = validation.get("prediction_glb")
        if isinstance(prediction_glb, MutableMapping):
            prediction_glb["action_units"] = action_units

    return updated


def apply_extra_mesh_action_units_override(
    extra_mesh_args: Any,
    action_units: Sequence[int],
) -> Any:
    if isinstance(extra_mesh_args, Mapping):
        if "mesh_dir" in extra_mesh_args:
            return extra_mesh_dataset_args_with_action_units(
                extra_mesh_args,
                action_units,
            )
        return {
            name: (
                extra_mesh_dataset_args_with_action_units(args, action_units)
                if isinstance(args, Mapping)
                else args
            )
            for name, args in extra_mesh_args.items()
        }
    if isinstance(extra_mesh_args, Sequence) and not isinstance(
        extra_mesh_args,
        (str, bytes),
    ):
        return [
            (
                extra_mesh_dataset_args_with_action_units(args, action_units)
                if isinstance(args, Mapping)
                else args
            )
            for args in extra_mesh_args
        ]
    return extra_mesh_args


def extra_mesh_dataset_args_with_action_units(
    args: Mapping[str, Any],
    action_units: Sequence[int],
) -> dict[str, Any]:
    updated = dict(args)
    updated["action_units"] = list(action_units)
    return updated


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state() -> dict[str, Any]:
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    return state


def save_failure_rng_state(
    run_dir: Path,
    distributed: DistributedContext,
    *,
    epoch: int,
    batch_index: int,
    global_step: int,
    stage: str,
    exception: BaseException,
) -> None:
    try:
        snapshot_dir = run_dir / "rng_states"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = (
            snapshot_dir
            / (
                f"failure_rng_state_{timestamp}"
                f"_rank{distributed.rank:02d}"
                f"_local{distributed.local_rank:02d}"
                f"_pid{os.getpid()}.pt"
            )
        )
        torch.save(
            {
                "rng_state": get_rng_state(),
                "metadata": {
                    "epoch": int(epoch),
                    "batch_index": int(batch_index),
                    "global_step": int(global_step),
                    "stage": str(stage),
                    "rank": int(distributed.rank),
                    "local_rank": int(distributed.local_rank),
                    "world_size": int(distributed.world_size),
                    "pid": int(os.getpid()),
                    "exception_type": type(exception).__name__,
                    "exception": str(exception),
                },
            },
            path,
        )
        print(f"[WARN] Saved failure RNG state to {path}", flush=True)
    except Exception as snapshot_exc:
        print(
            "[WARN] Failed to save failure RNG state: "
            f"{type(snapshot_exc).__name__}: {snapshot_exc}",
            flush=True,
        )


def distributed_context_from_env() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return DistributedContext(
        enabled=world_size > 1,
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=world_size,
    )


def initialize_distributed(
    context: DistributedContext,
    device: torch.device,
    config: Mapping[str, Any],
) -> DistributedContext:
    if not context.enabled:
        return context
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("DDP currently supports CPU or CUDA devices for this script.")

    backend = "nccl" if device.type == "cuda" else "gloo"
    if device.type == "cuda":
        validate_cuda_ddp_device(context, device)
        torch.cuda.set_device(device)
    init_kwargs: dict[str, Any] = {}
    timeout = distributed_process_group_timeout(config)
    if timeout is not None:
        init_kwargs["timeout"] = timeout
    dist.init_process_group(backend=backend, init_method="env://", **init_kwargs)
    return DistributedContext(
        enabled=True,
        rank=context.rank,
        local_rank=context.local_rank,
        world_size=context.world_size,
        backend=backend,
    )


def distributed_process_group_timeout(
    config: Mapping[str, Any],
) -> Optional[dt.timedelta]:
    distributed_config = config.get("distributed", {})
    if not isinstance(distributed_config, Mapping):
        return None
    value = distributed_config.get(
        "timeout_seconds",
        distributed_config.get("process_group_timeout_seconds"),
    )
    if value in (None, ""):
        return None
    seconds = float(value)
    if seconds <= 0.0:
        raise ValueError("distributed.timeout_seconds must be positive.")
    return dt.timedelta(seconds=seconds)


def cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def resolve_device(name: str, distributed: DistributedContext) -> torch.device:
    if distributed.enabled:
        if name == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(name)
        if device.type == "cuda" and device.index is None:
            return torch.device("cuda", distributed.local_rank)
        return device

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def validate_cuda_ddp_device(
    context: DistributedContext,
    device: torch.device,
) -> None:
    visible_devices = torch.cuda.device_count()
    if visible_devices <= 0:
        raise RuntimeError(
            "DDP requested CUDA, but PyTorch does not see any CUDA devices. "
            "Check your GPU allocation and CUDA_VISIBLE_DEVICES."
        )
    if device.index is None:
        raise RuntimeError("Internal error: CUDA DDP device index was not resolved.")
    if device.index >= visible_devices:
        raise RuntimeError(
            "Invalid CUDA device for DDP: "
            f"local_rank={context.local_rank} maps to {device}, but PyTorch only "
            f"sees {visible_devices} CUDA device(s). Launch with "
            f"--nproc_per_node={visible_devices} or request/expose more GPUs. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}."
        )


def make_run_dir(config: Mapping[str, Any], distributed: DistributedContext) -> Path:
    output_dir = Path(config["project"].get("output_dir", ROOT / "runs")).expanduser()
    payload: list[Optional[str]] = [None]
    if distributed.is_main:
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = config["project"].get("run_name") or f"toporig-{timestamp}"
        run_dir = output_dir / str(run_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        payload[0] = str(run_dir)
    if distributed.enabled:
        dist.broadcast_object_list(payload, src=0)
        dist.barrier()
    if payload[0] is None:
        raise RuntimeError("Failed to initialize run directory.")
    run_dir = Path(payload[0])
    if not distributed.is_main:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def build_datasets(
    config: Mapping[str, Any],
    distributed: DistributedContext,
) -> tuple[Dataset, Dataset]:
    if distributed.enabled and not distributed.is_main:
        dist.barrier()

    train_dataset = build_dataset(config, "train")
    val_dataset = build_dataset(config, "val")

    if distributed.enabled and distributed.is_main:
        dist.barrier()
    if distributed.enabled:
        dist.barrier()
    return train_dataset, val_dataset


def build_dataset(config: Mapping[str, Any], split_name: str) -> Dataset:
    data_config = config["data"]
    split_config = data_config[split_name]
    modalities = normalize_modalities(data_config.get("modalities", ("image", "mesh")))

    image_args = None
    mesh_args = None
    extra_mesh_args = None
    if "image" in modalities:
        image_args = dict(split_config.get("image_args") or {})
        image_args = image_args_for_training_split(config, image_args)
    if "mesh" in modalities:
        raw_mesh_args = split_config.get("mesh_args")
        if raw_mesh_args is not None:
            mesh_args = mesh_args_for_training_split(
                dict(raw_mesh_args),
                split_name,
            )
        extra_mesh_args = extra_mesh_args_for_training_split(
            split_config.get("extra_mesh_args"),
            split_name,
        )

    dataset = UnifiedDataset(
        image_args=image_args,
        mesh_args=mesh_args,
        extra_mesh_args=extra_mesh_args,
        data_root=data_config.get("data_root"),
        asset_cache_dir=data_config.get("asset_cache_dir"),
        split=str(split_config.get("split", "all")),
        split_csv=split_config.get("split_csv"),
        pair_image_with_mesh_by_action_unit=bool(
            split_config.get(
                "pair_image_with_mesh_by_action_unit",
                data_config.get("pair_image_with_mesh_by_action_unit", False),
            )
        ),
    )
    return apply_dataset_selection(dataset, split_config, split_name)


def image_args_for_training_split(
    config: Mapping[str, Any],
    image_args: Mapping[str, Any],
) -> dict[str, Any]:
    image_args = dict(image_args)
    if waft_flow_backend_enabled(config):
        image_args.setdefault("precompute_displacement_2d", False)
        image_args.setdefault("neutral_mesh_cache_size", 1)
        image_args.setdefault("displacement_cache_size", 0)
    return image_args


def waft_flow_backend_enabled(config: Mapping[str, Any]) -> bool:
    loss_config = config.get("loss", {})
    if not isinstance(loss_config, Mapping):
        return False
    image_config = loss_config.get("image", {})
    if not isinstance(image_config, Mapping):
        return False
    flow_config = image_config.get("flow", {})
    if not isinstance(flow_config, Mapping):
        return False
    return str(flow_config.get("backend", "local")).lower() == "waft"


def mesh_args_for_training_split(
    mesh_args: Mapping[str, Any],
    split_name: str,
) -> dict[str, Any]:
    mesh_args = dict(mesh_args)
    if split_name == "train":
        return mesh_args

    mesh_args["drop_mouth_probability"] = 0.0
    mesh_args["drop_teeth_probability"] = 0.0
    mesh_args["cut_eyeballs_probability"] = 0.0
    mesh_args["topology_augmentation_probability"] = 0.0
    mesh_args["shape_augmentation_probability"] = 0.0
    return mesh_args


def extra_mesh_args_for_training_split(
    extra_mesh_args: Any,
    split_name: str,
) -> Any:
    if extra_mesh_args is None:
        return None
    if isinstance(extra_mesh_args, Mapping):
        if "mesh_dir" in extra_mesh_args:
            return mesh_args_for_training_split(extra_mesh_args, split_name)
        return {
            name: (
                mesh_args_for_training_split(args, split_name)
                if isinstance(args, Mapping)
                else args
            )
            for name, args in extra_mesh_args.items()
        }
    if isinstance(extra_mesh_args, Sequence) and not isinstance(
        extra_mesh_args,
        (str, bytes),
    ):
        return [
            (
                mesh_args_for_training_split(args, split_name)
                if isinstance(args, Mapping)
                else args
            )
            for args in extra_mesh_args
        ]
    return extra_mesh_args


def apply_dataset_selection(
    dataset: Dataset,
    split_config: Mapping[str, Any],
    split_name: str,
) -> Dataset:
    indices_value = split_config.get("sample_indices")
    if indices_value is not None:
        indices = normalize_sample_indices(indices_value, len(dataset), split_name)
        dataset = Subset(dataset, indices)

    max_samples = split_config.get("max_samples")
    if max_samples is not None:
        limit = int(max_samples)
        if limit < 1:
            raise ValueError(f"data.{split_name}.max_samples must be at least 1.")
        dataset = Subset(dataset, list(range(min(limit, len(dataset)))))

    return dataset


def normalize_sample_indices(
    value: Any,
    dataset_length: int,
    split_name: str,
) -> list[int]:
    if isinstance(value, int):
        raw_indices = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_indices = [int(index) for index in value]
    else:
        raise ValueError(
            f"data.{split_name}.sample_indices must be an int or a list of ints."
        )

    if not raw_indices:
        raise ValueError(f"data.{split_name}.sample_indices must not be empty.")

    normalized = []
    for index in raw_indices:
        if index < 0:
            index += dataset_length
        if index < 0 or index >= dataset_length:
            raise IndexError(
                f"data.{split_name}.sample_indices contains index {index}, "
                f"but the dataset has {dataset_length} samples."
            )
        normalized.append(index)
    return normalized


def normalize_modalities(value: Any) -> set[str]:
    if isinstance(value, str):
        modalities = {value}
    else:
        modalities = {str(item) for item in value}
    invalid = modalities - {"image", "mesh"}
    if invalid:
        raise ValueError(f"Unknown data modalities: {sorted(invalid)}.")
    return modalities


def build_loader(
    config: Mapping[str, Any],
    dataset: Dataset,
    split_name: str,
    distributed: DistributedContext,
) -> DataLoader:
    loader_config = config["data"][split_name].get("loader", {})
    shuffle = bool(loader_config.get("shuffle", split_name == "train"))
    sampler = None
    if distributed.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=shuffle,
            drop_last=bool(loader_config.get("drop_last", False)),
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=int(loader_config.get("batch_size", 1)),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(loader_config.get("num_workers", 0)),
        pin_memory=bool(loader_config.get("pin_memory", False)),
        collate_fn=list_collate,
    )


def list_collate(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return samples


def sample_debug_summary(sample: Mapping[str, Any]) -> str:
    flags = []
    if bool(sample.get("is_mesh", False)):
        flags.append("mesh")
    if bool(sample.get("is_img", False)):
        flags.append("img")
    label = "+".join(flags) if flags else "empty"

    parts = [label]
    if bool(sample.get("is_mesh", False)):
        mesh_record = sample.get("mesh")
        if isinstance(mesh_record, Mapping):
            parts.append(
                "mesh("
                f"au={mesh_record.get('action_unit_id', '?')},"
                f"stream={sample.get('mesh_stream', '?')},"
                f"weight={sample.get('mesh_loss_weight_key', '?')},"
                f"{mesh_shape_debug_summary(mesh_record)}"
                ")"
            )
    if bool(sample.get("is_img", False)):
        image_record = sample.get("image")
        if isinstance(image_record, Mapping):
            parts.append(
                "img("
                f"au={image_record.get('action_unit_id', '?')},"
                f"{mesh_shape_debug_summary(image_record)},"
                f"{image_path_debug_summary(image_record)}"
                ")"
            )
    return " ".join(parts)


def mesh_shape_debug_summary(record: Mapping[str, Any]) -> str:
    mesh = record.get("mesh")
    if not isinstance(mesh, Mapping):
        return "v=?,f=?"
    vertices = mesh.get("vertices")
    faces = mesh.get("faces")
    vertex_count = int(vertices.shape[0]) if isinstance(vertices, torch.Tensor) else "?"
    face_count = int(faces.shape[0]) if isinstance(faces, torch.Tensor) else "?"
    return f"v={vertex_count},f={face_count}"


def image_path_debug_summary(record: Mapping[str, Any]) -> str:
    displacement_2d = record.get("displacement_2d")
    if not isinstance(displacement_2d, Mapping):
        return "neutral=?,expressed=?"
    neutral = displacement_2d.get("neutral_image_path")
    expressed = displacement_2d.get("expressed_image_path")
    neutral_name = Path(str(neutral)).name if neutral not in (None, "") else "?"
    expressed_name = Path(str(expressed)).name if expressed not in (None, "") else "?"
    return f"neutral={neutral_name},expressed={expressed_name}"


def batch_debug_summary(batch: Sequence[Mapping[str, Any]]) -> str:
    return " | ".join(sample_debug_summary(sample) for sample in batch)


def maybe_log_train_batch_start(
    *,
    batch: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    distributed: DistributedContext,
    epoch: int,
    batch_index: int,
    global_step: int,
) -> None:
    limit = diagnostics_int(config, "log_first_train_batches", 0)
    if limit <= 0 or batch_index > limit:
        return
    print(
        "[DDP-BATCH] "
        f"rank={distributed.rank}/{distributed.world_size} "
        f"epoch={epoch} batch={batch_index} global_step={global_step} "
        f"samples={batch_debug_summary(batch)}",
        flush=True,
    )


def maybe_log_slow_train_stage(
    *,
    stage: str,
    elapsed_seconds: float,
    batch: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    distributed: DistributedContext,
    epoch: int,
    batch_index: int,
    global_step: int,
) -> None:
    threshold = diagnostics_float(
        config,
        f"{stage}_slow_seconds",
        diagnostics_float(config, "slow_stage_seconds", 0.0),
    )
    if threshold is None or threshold <= 0.0 or elapsed_seconds < threshold:
        return
    print(
        "[WARN] Slow train stage "
        f"rank={distributed.rank}/{distributed.world_size} "
        f"stage={stage} elapsed={elapsed_seconds:.2f}s "
        f"epoch={epoch} batch={batch_index} global_step={global_step} "
        f"samples={batch_debug_summary(batch)}",
        flush=True,
    )


def maybe_log_slow_train_sample(
    *,
    sample_index: int,
    sample: Mapping[str, Any],
    elapsed_seconds: float,
    config: Mapping[str, Any],
    distributed: DistributedContext,
    epoch: int,
    batch_index: int,
    global_step: int,
) -> None:
    threshold = diagnostics_float(config, "slow_sample_seconds", 0.0)
    if threshold is None or threshold <= 0.0 or elapsed_seconds < threshold:
        return
    print(
        "[WARN] Slow train sample "
        f"rank={distributed.rank}/{distributed.world_size} "
        f"elapsed={elapsed_seconds:.2f}s "
        f"epoch={epoch} batch={batch_index} global_step={global_step} "
        f"sample_index={sample_index} sample={sample_debug_summary(sample)}",
        flush=True,
    )


def set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)


def image_flow_loss_enabled(flow_config: Mapping[str, Any]) -> bool:
    weight = float(flow_config.get("weight", 1.0))
    if weight < 0.0:
        raise ValueError("loss.image.flow.weight must be non-negative.")
    return weight > 0.0


@torch.no_grad()
def warmup_image_flow_backend(
    config: Mapping[str, Any],
    device: torch.device,
    distributed: DistributedContext,
) -> None:
    if "image" not in normalize_modalities(
        config.get("data", {}).get("modalities", ("image", "mesh"))
    ):
        return
    flow_config = config["loss"].get("image", {}).get("flow", {})
    if not image_flow_loss_enabled(flow_config):
        return
    if str(flow_config.get("backend", "local")).lower() != "waft":
        return
    if not bool(flow_config.get("waft_warmup_model", True)):
        return

    get_waft_flow_model(flow_config, device)
    if distributed.enabled:
        dist.barrier()


@torch.no_grad()
def precache_image_flow_targets(
    config: Mapping[str, Any],
    datasets: Sequence[Dataset],
    device: torch.device,
    distributed: DistributedContext,
    cache_shard_rank: int = 0,
    cache_shard_count: int = 1,
) -> None:
    if "image" not in normalize_modalities(
        config.get("data", {}).get("modalities", ("image", "mesh"))
    ):
        return
    flow_config = config["loss"].get("image", {}).get("flow", {})
    if not image_flow_loss_enabled(flow_config):
        return
    if str(flow_config.get("backend", "local")).lower() != "waft":
        return
    if not bool(flow_config.get("waft_precompute_target_cache", True)):
        return

    image_size = normalize_image_size(config["rendering"]["train_image_size"])
    disk_cache_enabled = waft_target_disk_cache_enabled(flow_config)
    batch_size = waft_precompute_target_batch_size(flow_config)
    if cache_shard_count < 1:
        raise ValueError("cache_shard_count must be at least 1.")
    if not 0 <= cache_shard_rank < cache_shard_count:
        raise ValueError("cache_shard_rank must be within cache_shard_count.")
    if distributed.enabled and cache_shard_count != 1:
        raise ValueError(
            "Explicit cache sharding cannot be combined with distributed caching."
        )
    shard_rank = distributed.rank if distributed.enabled else cache_shard_rank
    shard_count = distributed.world_size if distributed.enabled else cache_shard_count
    if distributed.is_main:
        message = "[INFO] Pre-caching WAFT image target flow(s)"
        if disk_cache_enabled:
            message += f" with disk-cache shard {shard_rank}/{shard_count}"
        if batch_size > 1:
            message += f" using batch size {batch_size}"
        print(message + ".")
    iterator = progress_iter(
        unique_image_flow_target_records(
            datasets,
            shard_rank=shard_rank if disk_cache_enabled else 0,
            shard_count=shard_count if disk_cache_enabled else 1,
        ),
        total=None,
        desc="cache WAFT targets",
        disable=not distributed.is_main,
        unit="pair",
    )
    processed_count = 0
    record_batch: list[Mapping[str, Any]] = []
    for record in iterator:
        record_batch.append(record)
        if len(record_batch) < batch_size:
            continue
        precache_image_flow_target_record_batch(
            records=record_batch,
            image_size=image_size,
            flow_config=flow_config,
            render_config=config.get("rendering", {}),
            full_config=config,
            device=device,
        )
        processed_count += len(record_batch)
        record_batch = []
    if record_batch:
        precache_image_flow_target_record_batch(
            records=record_batch,
            image_size=image_size,
            flow_config=flow_config,
            render_config=config.get("rendering", {}),
            full_config=config,
            device=device,
        )
        processed_count += len(record_batch)
    if disk_cache_enabled:
        print(
            "[INFO] WAFT target pre-cache rank "
            f"{shard_rank}/{shard_count} processed "
            f"{processed_count} shard item(s).",
            flush=True,
        )
    release_waft_alignment_mesh_cache()
    if distributed.enabled:
        dist.barrier()


@torch.no_grad()
def precache_image_mediapipe_landmarks(
    config: Mapping[str, Any],
    datasets: Sequence[Dataset],
    device: torch.device,
    distributed: DistributedContext,
    cache_shard_rank: int = 0,
    cache_shard_count: int = 1,
) -> None:
    if "image" not in normalize_modalities(
        config.get("data", {}).get("modalities", ("image", "mesh"))
    ):
        return
    image_config = config.get("loss", {}).get("image", {})
    iris_config = image_config.get("iris_gaze", {})
    eyelid_config = image_config.get("eyelid_closure", {})
    iris_enabled = isinstance(iris_config, Mapping) and bool(
        iris_config.get("enabled", False)
    )
    eyelid_enabled = isinstance(eyelid_config, Mapping) and bool(
        eyelid_config.get("enabled", False)
    )
    if not iris_enabled and not eyelid_enabled:
        return
    if cache_shard_count < 1:
        raise ValueError("cache_shard_count must be at least 1.")
    if not 0 <= cache_shard_rank < cache_shard_count:
        raise ValueError("cache_shard_rank must be within cache_shard_count.")
    if distributed.enabled and cache_shard_count != 1:
        raise ValueError(
            "Explicit cache sharding cannot be combined with distributed caching."
        )

    shard_rank = distributed.rank if distributed.enabled else cache_shard_rank
    shard_count = distributed.world_size if distributed.enabled else cache_shard_count
    flow_config = image_config.get("flow", {})
    if distributed.is_main:
        print(
            "[INFO] Pre-caching MediaPipe image landmarks "
            f"with disk-cache shard {shard_rank}/{shard_count}.",
            flush=True,
        )

    iterator = progress_iter(
        unique_image_flow_target_keys(
            datasets,
            shard_rank=shard_rank,
            shard_count=shard_count,
        ),
        total=None,
        desc="cache image landmarks",
        disable=not distributed.is_main,
        unit="pair",
    )
    processed_count = 0
    incomplete_count = 0
    for neutral_path, expressed_path, action_unit_id in iterator:
        if action_unit_id is None:
            continue
        cache_specs: list[tuple[Mapping[str, Any], set[int], tuple[int, int]]] = []
        if iris_enabled:
            action_iris_config = config_for_action_unit(
                iris_config,
                int(action_unit_id),
            )
            configured_action_units = action_iris_config.get("action_units")
            iris_action_units = (
                parse_action_units(configured_action_units)
                if configured_action_units not in (None, "")
                else None
            )
            if (
                bool(action_iris_config.get("enabled", False))
                and image_mediapipe_landmark_disk_cache_enabled(
                    action_iris_config
                )
                and (
                    iris_action_units is None
                    or int(action_unit_id) in iris_action_units
                )
            ):
                required_ids = set(
                    normalize_int_list(
                        action_iris_config.get("iris_ids"),
                        "loss.image.iris_gaze.iris_ids",
                    )
                )
                required_ids.update(
                    normalize_int_list(
                        action_iris_config.get("corner_ids"),
                        "loss.image.iris_gaze.corner_ids",
                    )
                )
                cache_specs.append(
                    (
                        action_iris_config,
                        required_ids,
                        normalize_image_size(
                            action_iris_config.get(
                                "landmark_detection_image_size",
                                config.get("rendering", {}).get(
                                    "train_image_size",
                                    (256, 256),
                                ),
                            )
                        ),
                    )
                )
        if eyelid_enabled:
            action_eyelid_config = config_for_action_unit(
                eyelid_config,
                int(action_unit_id),
            )
            try:
                eyelid_curves = eyelid_landmarks_for_action_unit(
                    int(action_unit_id)
                )
            except ValueError:
                eyelid_curves = None
            if (
                eyelid_curves is not None
                and bool(action_eyelid_config.get("enabled", False))
                and image_eyelid_target_mode(action_eyelid_config) == "observed"
                and image_mediapipe_landmark_disk_cache_enabled(
                    action_eyelid_config
                )
            ):
                cache_specs.append(
                    (
                        action_eyelid_config,
                        set(eyelid_curves.all_ids),
                        normalize_image_size(
                            action_eyelid_config.get(
                                "landmark_detection_image_size",
                                config.get("rendering", {}).get(
                                    "train_image_size",
                                    (256, 256),
                                ),
                            )
                        ),
                    )
                )
        if not cache_specs:
            continue
        complete = True
        for alignment_config, required_ids, detection_size in cache_specs:
            for path in (neutral_path, expressed_path):
                landmarks = cached_image_mediapipe_landmarks(
                    path=path,
                    image_size=detection_size,
                    flow_config=flow_config,
                    alignment=alignment_config,
                    device=device,
                    dtype=torch.float32,
                )
                if required_ids and not required_ids.issubset(landmarks):
                    complete = False
        processed_count += 1
        incomplete_count += int(not complete)

    print(
        "[INFO] Refined MediaPipe image-landmark pre-cache rank "
        f"{shard_rank}/{shard_count} processed {processed_count} pair(s); "
        f"{incomplete_count} pair(s) lacked one or more required landmarks.",
        flush=True,
    )
    if distributed.enabled:
        dist.barrier()


def waft_precompute_target_batch_size(flow_config: Mapping[str, Any]) -> int:
    value = flow_config.get("waft_precompute_target_batch_size", 1)
    if value in (None, ""):
        return 1
    batch_size = int(value)
    if batch_size < 1:
        raise ValueError(
            "loss.image.flow.waft_precompute_target_batch_size must be at least 1."
        )
    return batch_size


@torch.no_grad()
def precache_image_flow_target_record_batch(
    records: Sequence[Mapping[str, Any]],
    image_size: tuple[int, int],
    flow_config: Mapping[str, Any],
    render_config: Optional[Mapping[str, Any]],
    full_config: Mapping[str, Any],
    device: torch.device,
) -> None:
    if not records:
        return

    neutral_renders: list[Optional[torch.Tensor]] = []
    if waft_alignment_needs_neutral_render(flow_config):
        for record in records:
            neutral_renders.append(
                render_image_loss_rgb(
                    vertices=record["mesh"]["vertices"].unsqueeze(0).to(device),
                    reference_vertices=record["mesh"]["vertices"].unsqueeze(0).to(device),
                    faces=record["mesh"]["faces"].to(device),
                    config=full_config,
                    image_size=image_size,
                )
            )
    else:
        neutral_renders = [None] * len(records)

    if len(records) > 1 and waft_target_disk_cache_enabled(flow_config):
        prepare_waft_image_flow_target_batch(
            records=records,
            image_size=image_size,
            flow_config=flow_config,
            render_config=render_config,
            neutral_renders=neutral_renders,
            device=device,
            dtype=torch.float32,
        )
        return

    for record, neutral_render in zip(records, neutral_renders):
        prepare_image_flow_target(
            record=record,
            image_size=image_size,
            flow_config=flow_config,
            render_config=render_config,
            neutral_render=neutral_render,
            device=device,
            dtype=torch.float32,
        )


def unique_image_flow_target_records(
    datasets: Sequence[Dataset],
    shard_rank: int = 0,
    shard_count: int = 1,
) -> Iterator[Mapping[str, Any]]:
    seen: set[tuple[str, str, Optional[int]]] = set()
    unique_index = 0
    for dataset in datasets:
        for key, record_factory in image_flow_target_record_factories(dataset):
            if key in seen:
                continue
            seen.add(key)
            if shard_count < 1:
                raise ValueError("shard_count must be at least 1.")
            owns_record = unique_index % shard_count == shard_rank
            unique_index += 1
            if owns_record:
                yield record_factory()


def unique_image_flow_target_keys(
    datasets: Sequence[Dataset],
    shard_rank: int = 0,
    shard_count: int = 1,
) -> Iterator[tuple[str, str, Optional[int]]]:
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1.")
    if not 0 <= shard_rank < shard_count:
        raise ValueError("shard_rank must be within shard_count.")
    seen: set[tuple[str, str, Optional[int]]] = set()
    unique_index = 0
    for dataset in datasets:
        for key, _record_factory in image_flow_target_record_factories(dataset):
            if key in seen:
                continue
            seen.add(key)
            owns_record = unique_index % shard_count == shard_rank
            unique_index += 1
            if owns_record:
                yield key


def image_flow_target_record_factories(
    dataset: Dataset,
) -> Iterator[tuple[tuple[str, str, Optional[int]], Any]]:
    if isinstance(dataset, UnifiedDataset) and dataset.image_dataset is not None:
        image_dataset = dataset.image_dataset
        last_person_id = None
        try:
            for index in range(dataset.image_length):
                sample = image_dataset.samples[index]
                key = (
                    str(sample.neutral_image_path),
                    str(sample.expressed_image_path),
                    int(sample.action_unit_id),
                )

                def factory(
                    image_dataset=image_dataset,
                    index=index,
                    person_id=sample.person_id,
                ):
                    nonlocal last_person_id
                    if last_person_id is not None and person_id != last_person_id:
                        clear_image_dataset_record_caches(image_dataset)
                    last_person_id = person_id
                    return image_dataset_flow_record(image_dataset, index)

                yield key, factory
        finally:
            clear_image_dataset_record_caches(image_dataset)
        return

    for index in range(len(dataset)):
        sample = dataset[index]
        if not isinstance(sample, Mapping) or not bool(sample.get("is_img", False)):
            continue
        record = sample.get("image")
        if not isinstance(record, Mapping):
            continue
        displacement_2d = record.get("displacement_2d")
        if not isinstance(displacement_2d, Mapping):
            continue
        neutral_path = displacement_2d.get("neutral_image_path")
        expressed_path = displacement_2d.get("expressed_image_path")
        if neutral_path in (None, "") or expressed_path in (None, ""):
            continue
        key = (str(neutral_path), str(expressed_path), record_action_unit(record))
        yield key, lambda record=record: record


def image_dataset_flow_record(image_dataset: Any, index: int) -> Mapping[str, Any]:
    neutral_mesh, action_unit_id, displacement_2d, landmarks_3d = image_dataset[index]
    return {
        "mesh": neutral_mesh,
        "action_unit_id": action_unit_id,
        "displacement_2d": displacement_2d,
        "landmarks_3d": landmarks_3d,
    }


def clear_image_dataset_record_caches(image_dataset: Any) -> None:
    clear_caches = getattr(image_dataset, "clear_memory_caches", None)
    if callable(clear_caches):
        clear_caches()
        return
    mesh_cache = getattr(image_dataset, "_mesh_cache", None)
    if isinstance(mesh_cache, dict):
        mesh_cache.clear()
    displacement_cache = getattr(image_dataset, "_displacement_cache", None)
    if isinstance(displacement_cache, dict):
        displacement_cache.clear()


def waft_alignment_needs_neutral_render(flow_config: Mapping[str, Any]) -> bool:
    alignment = flow_config.get("waft_target_image_alignment")
    if not isinstance(alignment, Mapping) or not bool(alignment.get("enabled", False)):
        return False
    mode = str(alignment.get("mode", "bbox")).strip().lower()
    if mode not in {"landmark", "landmarks", "mediapipe", "mediapipe_landmarks"}:
        return False
    target_source = str(alignment.get("target_source", "render")).strip().lower()
    return target_source in WAFT_RENDER_ALIGNMENT_TARGET_SOURCES


def wrap_model_for_distributed(
    model: nn.Module,
    distributed: DistributedContext,
    device: torch.device,
    config: Mapping[str, Any],
) -> nn.Module:
    if not distributed.enabled:
        return model
    distributed_config = config.get("distributed", {})
    if bool(distributed_config.get("sync_batchnorm", False)):
        if device.type != "cuda":
            raise ValueError("distributed.sync_batchnorm requires CUDA DDP.")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_kwargs = {
        "broadcast_buffers": bool(distributed_config.get("broadcast_buffers", True)),
        "find_unused_parameters": bool(
            distributed_config.get("find_unused_parameters", False)
        ),
    }
    if "static_graph" in distributed_config:
        ddp_kwargs["static_graph"] = bool(distributed_config["static_graph"])
    if device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            **ddp_kwargs,
        )
    return DistributedDataParallel(model, **ddp_kwargs)


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def build_optimizer(model: nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    optimizer_type = str(config.get("type", "adamw")).lower()
    lr = float(config.get("lr", 1.0e-4))
    weight_decay = float(config.get("weight_decay", 1.0e-4))

    if optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer type: {optimizer_type!r}.")


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    num_epochs: int,
):
    scheduler_config = config.get("scheduler")
    if scheduler_config is None:
        return None
    if isinstance(scheduler_config, str):
        scheduler_config = {"type": scheduler_config}
    if not isinstance(scheduler_config, Mapping):
        raise ValueError("optimizer.scheduler must be a mapping or scheduler name.")

    scheduler_type = str(scheduler_config.get("type", "none")).lower()
    if scheduler_type in {"none", "off", "disabled", "false"}:
        return None
    if scheduler_type in {"cosine", "cosine_annealing", "cosineannealing"}:
        t_max = int(scheduler_config.get("t_max", num_epochs))
        if t_max < 1:
            raise ValueError("optimizer.scheduler.t_max must be at least 1.")
        eta_min = float(
            scheduler_config.get("eta_min", scheduler_config.get("min_lr", 0.0))
        )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )
    raise ValueError(f"Unsupported scheduler type: {scheduler_type!r}.")


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0]["lr"])


def init_wandb(config: Mapping[str, Any], run_dir: Path):
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "--use-wandb was passed, but wandb is not installed. "
            "Install it with `pip install wandb`."
        ) from exc

    wandb_config = config.get("wandb", {})
    return wandb.init(
        project=wandb_config.get("project", "toporig"),
        entity=wandb_config.get("entity"),
        name=wandb_config.get("run_name") or config["project"].get("run_name"),
        tags=wandb_config.get("tags"),
        dir=str(run_dir),
        config=dict(config),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    device: torch.device,
    epoch: int,
    global_step: int,
    log_every_steps: int,
    wandb_run: Any,
    distributed: DistributedContext,
    validation_dataset: Optional[Dataset] = None,
    run_dir: Optional[Path] = None,
) -> tuple[dict[str, float], int]:
    model.train()
    running = MetricAverager()
    epoch_timings: dict[str, float] = {}
    epoch_started_at = timing_start(config, device)
    grad_clip_norm = config["training"].get("grad_clip_norm")
    wandb_log_every_steps = wandb_scalar_log_every_steps(config)
    batches = progress_iter(
        loader,
        total=len(loader),
        desc=f"train epoch {epoch}",
        disable=not distributed.is_main,
        unit="batch",
    )

    batch_iter = iter(batches)
    batch_index = 0
    while True:
        arm_stack_dump_watchdog(config)
        data_started_at = timing_start(config, device)
        data_diagnostic_started_at = diagnostic_timer_start(config, device)
        try:
            batch = next(batch_iter)
        except StopIteration:
            cancel_stack_dumps()
            break
        except Exception as exc:
            if run_dir is not None:
                save_failure_rng_state(
                    run_dir=run_dir,
                    distributed=distributed,
                    epoch=epoch,
                    batch_index=batch_index + 1,
                    global_step=global_step,
                    stage="train_data_load",
                    exception=exc,
                )
            raise
        data_elapsed = diagnostic_timer_elapsed(
            data_diagnostic_started_at,
            config,
            device,
        )
        timing_stop(epoch_timings, "train_data_load", data_started_at, config, device)
        batch_index += 1
        maybe_log_slow_train_stage(
            stage="train_data_load",
            elapsed_seconds=data_elapsed,
            batch=batch,
            config=config,
            distributed=distributed,
            epoch=epoch,
            batch_index=batch_index,
            global_step=global_step,
        )
        maybe_log_train_batch_start(
            batch=batch,
            config=config,
            distributed=distributed,
            epoch=epoch,
            batch_index=batch_index,
            global_step=global_step,
        )

        batch_timings: dict[str, float] = {}
        zero_grad_started_at = diagnostic_timer_start(config, device)
        with timed_metric(batch_timings, "train_zero_grad", config, device):
            optimizer.zero_grad(set_to_none=True)
        maybe_log_slow_train_stage(
            stage="train_zero_grad",
            elapsed_seconds=diagnostic_timer_elapsed(
                zero_grad_started_at,
                config,
                device,
            ),
            batch=batch,
            config=config,
            distributed=distributed,
            epoch=epoch,
            batch_index=batch_index,
            global_step=global_step,
        )
        batch_metrics = MetricAverager()
        batch_sample_count = len(batch)
        if batch_sample_count == 0:
            continue
        batch_loss_total = 0.0
        compute_loss_elapsed = 0.0
        backward_elapsed = 0.0

        for sample_index, sample in enumerate(batch):
            sample_started_at = diagnostic_timer_start(config, device)
            compute_loss_started_at = diagnostic_timer_start(config, device)
            try:
                with timed_metric(batch_timings, "train_compute_loss", config, device):
                    loss, metrics = compute_sample_loss(
                        model=model,
                        sample=sample,
                        config=config,
                        device=device,
                    )
            except Exception as exc:
                if run_dir is not None:
                    save_failure_rng_state(
                        run_dir=run_dir,
                        distributed=distributed,
                        epoch=epoch,
                        batch_index=batch_index,
                        global_step=global_step,
                        stage="train_compute_loss",
                        exception=exc,
                    )
                raise
            compute_loss_elapsed += diagnostic_timer_elapsed(
                compute_loss_started_at,
                config,
                device,
            )
            batch_loss_total += float(loss.detach().cpu())
            batch_metrics.update(metrics)

            backward_started_at = diagnostic_timer_start(config, device)
            try:
                with timed_metric(batch_timings, "train_backward", config, device):
                    (loss / batch_sample_count).backward()
            except Exception as exc:
                if run_dir is not None:
                    save_failure_rng_state(
                        run_dir=run_dir,
                        distributed=distributed,
                        epoch=epoch,
                        batch_index=batch_index,
                        global_step=global_step,
                        stage="train_backward",
                        exception=exc,
                    )
                raise
            backward_elapsed += diagnostic_timer_elapsed(
                backward_started_at,
                config,
                device,
            )
            del loss
            maybe_log_slow_train_sample(
                sample_index=sample_index,
                sample=sample,
                elapsed_seconds=diagnostic_timer_elapsed(
                    sample_started_at,
                    config,
                    device,
                ),
                config=config,
                distributed=distributed,
                epoch=epoch,
                batch_index=batch_index,
                global_step=global_step,
            )

        maybe_log_slow_train_stage(
            stage="train_compute_loss",
            elapsed_seconds=compute_loss_elapsed,
            batch=batch,
            config=config,
            distributed=distributed,
            epoch=epoch,
            batch_index=batch_index,
            global_step=global_step,
        )
        maybe_log_slow_train_stage(
            stage="train_backward",
            elapsed_seconds=backward_elapsed,
            batch=batch,
            config=config,
            distributed=distributed,
            epoch=epoch,
            batch_index=batch_index,
            global_step=global_step,
        )
        if grad_clip_norm is not None:
            grad_clip_started_at = diagnostic_timer_start(config, device)
            with timed_metric(batch_timings, "train_grad_clip", config, device):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(grad_clip_norm),
                )
            maybe_log_slow_train_stage(
                stage="train_grad_clip",
                elapsed_seconds=diagnostic_timer_elapsed(
                    grad_clip_started_at,
                    config,
                    device,
                ),
                batch=batch,
                config=config,
                distributed=distributed,
                epoch=epoch,
                batch_index=batch_index,
                global_step=global_step,
            )
        optimizer_started_at = diagnostic_timer_start(config, device)
        with timed_metric(batch_timings, "train_optimizer_step", config, device):
            optimizer.step()
        maybe_log_slow_train_stage(
            stage="train_optimizer_step",
            elapsed_seconds=diagnostic_timer_elapsed(
                optimizer_started_at,
                config,
                device,
            ),
            batch=batch,
            config=config,
            distributed=distributed,
            epoch=epoch,
            batch_index=batch_index,
            global_step=global_step,
        )

        global_step += 1
        metrics = batch_metrics.compute()
        metrics.update(batch_timings)
        metrics["loss"] = batch_loss_total / batch_sample_count
        metrics["lr"] = current_learning_rate(optimizer)
        running.update(metrics, weight=batch_sample_count)

        if (
            distributed.is_main
            and wandb_run is not None
            and wandb_log_every_steps > 0
            and global_step % wandb_log_every_steps == 0
        ):
            wandb_run.log(
                {f"train/{key}": value for key, value in metrics.items()},
                step=global_step,
            )

        if (
            distributed.is_main
            and log_every_steps > 0
            and global_step % log_every_steps == 0
        ):
            prefix = f"[train] epoch {epoch} step {batch_index}/{len(loader)}"
            progress_write(format_metrics(prefix, metrics))

        maybe_export_step_prediction_snapshot(
            model=unwrap_model(model),
            dataset=validation_dataset,
            config=config,
            device=device,
            run_dir=run_dir,
            epoch=epoch,
            global_step=global_step,
            wandb_run=wandb_run,
            distributed=distributed,
        )
        model.train()

    timing_stop(epoch_timings, "train_epoch_total", epoch_started_at, config, device)
    result = sync_metric_averager(running, distributed)
    result.update(epoch_timings)
    return result, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    config: Mapping[str, Any],
    device: torch.device,
    distributed: DistributedContext,
) -> dict[str, float]:
    model.eval()
    running = MetricAverager()
    primary_running = MetricAverager()
    primary_metric_name = validation_primary_metric_name(config)
    mae_only = bool(config.get("validation", {}).get("mae_only", False))
    if mae_only and primary_metric_name != "mesh_3d_mae":
        raise ValueError(
            "validation.mae_only requires "
            "validation.primary_metric: mesh_3d_mae."
        )
    validation_timings: dict[str, float] = {}
    validation_started_at = timing_start(config, device)
    max_batches = config["validation"].get("max_batches")
    total_batches = len(loader)
    batches_iter = loader
    if max_batches is not None:
        total_batches = min(total_batches, int(max_batches))
        batches_iter = itertools.islice(loader, total_batches)
    batches = progress_iter(
        batches_iter,
        total=total_batches,
        desc="val",
        disable=not distributed.is_main,
        unit="batch",
    )

    batch_iter = iter(batches)
    batch_index = 0
    while True:
        data_started_at = timing_start(config, device)
        try:
            batch = next(batch_iter)
        except StopIteration:
            break
        timing_stop(validation_timings, "val_data_load", data_started_at, config, device)
        batch_index += 1

        sample_loss_total = 0.0
        sample_count = 0
        batch_metrics = MetricAverager()
        batch_timings: dict[str, float] = {}
        with timed_metric(batch_timings, "val_compute_loss", config, device):
            for sample in batch:
                if mae_only:
                    if not bool(sample.get("is_mesh", False)):
                        continue
                    mesh_only_sample = dict(sample)
                    mesh_only_sample["is_img"] = False
                    _, metrics = compute_sample_loss(
                        model=model,
                        sample=mesh_only_sample,
                        config=config,
                        device=device,
                    )
                    if "mesh_3d_mae" not in metrics:
                        raise ValueError(
                            "MAE-only validation mesh sample did not produce "
                            "mesh_3d_mae."
                        )
                    sample_count += 1
                    batch_metrics.update(
                        {"mesh_3d_mae": metrics["mesh_3d_mae"]}
                    )
                    primary_running.update(
                        {"mesh_3d_mae": metrics["mesh_3d_mae"]}
                    )
                    continue

                loss, metrics = compute_sample_loss(
                    model=model,
                    sample=sample,
                    config=config,
                    device=device,
                )
                sample_loss_total += float(loss.detach().cpu())
                sample_count += 1
                batch_metrics.update(metrics)
                if primary_metric_name == "loss":
                    primary_running.update(
                        {primary_metric_name: float(loss.detach().cpu())}
                    )
                elif primary_metric_name in metrics:
                    primary_running.update(
                        {primary_metric_name: metrics[primary_metric_name]}
                    )

        if sample_count > 0:
            metrics = batch_metrics.compute()
            metrics.update(batch_timings)
            if mae_only:
                metrics["loss"] = metrics["mesh_3d_mae"]
            else:
                metrics["loss"] = sample_loss_total / sample_count
            running.update(metrics, weight=sample_count)

    timing_stop(
        validation_timings,
        "val_epoch_total",
        validation_started_at,
        config,
        device,
    )
    result = sync_metric_averager(running, distributed)
    primary_result = sync_metric_averager(primary_running, distributed)
    if primary_metric_name not in primary_result:
        raise ValueError(
            "Validation primary metric was not produced by any validation sample: "
            f"{primary_metric_name!r}."
        )
    primary_value = float(primary_result[primary_metric_name])
    if not math.isfinite(primary_value):
        raise ValueError(
            "Validation primary metric must be finite, got "
            f"{primary_metric_name}={primary_value}."
        )
    if mae_only:
        result = {
            "loss": primary_value,
            "mesh_3d_mae": primary_value,
            "primary_metric": primary_value,
        }
        result.update(validation_timings)
        return result
    result[primary_metric_name] = primary_value
    result["primary_metric"] = primary_value
    result.update(validation_timings)
    return result


def validation_primary_metric_name(config: Mapping[str, Any]) -> str:
    validation_config = config.get("validation", {})
    configured_metric = validation_config.get("primary_metric")
    if configured_metric is None:
        data_config = config.get("data", {})
        modalities = data_config.get("modalities", ())
        configured_metric = (
            "mesh_3d_mae" if "mesh" in set(modalities) else "loss"
        )
    metric_name = str(configured_metric).strip()
    if not metric_name:
        raise ValueError("validation.primary_metric must not be empty.")
    return metric_name


def compute_sample_loss(
    model: nn.Module,
    sample: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    metrics: dict[str, float] = {}
    timing_metrics: dict[str, float] = {}
    weights = config["loss"].get("weights", {})

    if bool(sample["is_mesh"]):
        mesh_grad_context = (
            torch.enable_grad()
            if mesh_active_landmark_gradient_balance_enabled(config)
            else nullcontext()
        )
        with mesh_grad_context, timed_metric(
            timing_metrics,
            "sample_mesh_loss",
            config,
            device,
        ):
            loss_3d, mesh_metrics = compute_mesh_loss(
                model,
                sample["mesh"],
                config,
                device,
            )
        mesh_weight_key = str(sample.get("mesh_loss_weight_key", "mesh_3d"))
        mesh_weight = float(weights.get(mesh_weight_key, weights.get("mesh_3d", 1.0)))
        weighted = loss_3d * mesh_weight
        losses.append(weighted)
        metrics.update(mesh_metrics)
        if mesh_weight_key != "mesh_3d":
            metrics[f"{mesh_weight_key}_loss"] = float(loss_3d.detach().cpu())
            metrics[f"{mesh_weight_key}_weighted_loss"] = float(
                weighted.detach().cpu()
            )

    if bool(sample["is_img"]):
        with timed_metric(timing_metrics, "sample_image_loss", config, device):
            loss_2d, image_metrics = compute_image_loss(
                model,
                sample["image"],
                config,
                device,
            )
        weighted = loss_2d * float(weights.get("image_2d", 1.0))
        losses.append(weighted)
        metrics.update(image_metrics)

    if not losses:
        raise ValueError("Sample did not contain an active image or mesh modality.")

    total = torch.stack(losses).sum()
    metrics.update(timing_metrics)
    metrics["loss"] = float(total.detach().cpu())
    return total, metrics


def mesh_active_landmark_gradient_balance_enabled(config: Mapping[str, Any]) -> bool:
    loss_config = config.get("loss", {})
    if not isinstance(loss_config, Mapping):
        return False
    mesh_config = loss_config.get("mesh", {})
    if not isinstance(mesh_config, Mapping):
        return False
    return bool(mesh_config.get("landmark_active_motion_gradient_balance", False))






















def compute_mesh_loss(
    model: nn.Module,
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    mesh = to_device(record["mesh"], device)
    vertices = mesh["vertices"].unsqueeze(0)
    normals = mesh.get("normals")
    if normals is not None:
        normals = normals.unsqueeze(0)
    faces = mesh.get("faces")
    action_unit_id = as_int(record["action_unit_id"])
    facs = action_unit_vector(
        action_unit_id,
        int(config["model"]["facs_dim"]),
        device,
        float(config["training"].get("action_unit_scale", 1.0)),
    )
    target = record["delta_vertices"].to(
        device=device,
        dtype=vertices.dtype,
    ).unsqueeze(0)
    landmark_features = build_model_landmark_features(
        record=record,
        vertices=vertices,
        config=config,
        device=device,
        training=model.training,
    )
    predicted = model(
        vertices,
        facs,
        normals=normals,
        faces=faces,
        landmark_features=landmark_features,
    )
    mesh_config = config["loss"].get("mesh", {})
    vertex_mask = mesh_target_vertex_mask(
        vertices=vertices,
        target_delta=target,
        landmarks_3d=record.get("landmarks_3d"),
        config=mesh_config,
        faces=record["mesh"].get("faces", faces),
    )
    target = filter_mesh_target_delta(
        vertices=vertices,
        target_delta=target,
        faces=faces,
        vertex_mask=vertex_mask,
        config=mesh_config,
    )
    vertex_loss_mask = mesh_loss_vertex_mask(
        vertex_mask=vertex_mask,
        reference=predicted,
        faces=faces,
        config=mesh_config,
    )
    if mesh_hard_target_vertex_mask_enabled(mesh_config):
        predicted = apply_mesh_delta_vertex_mask(
            predicted,
            vertex_mask,
            faces=faces,
            config=mesh_config.get("hard_target_vertex_mask"),
        )
    predicted = filter_mesh_prediction_delta(
        vertices=vertices,
        predicted_delta=predicted,
        faces=faces,
        vertex_mask=vertex_mask,
        config=mesh_config,
    )
    vertex_loss = mesh_loss_fn(
        predicted,
        target,
        mesh_config,
        vertex_mask=vertex_loss_mask,
    )
    zero_anchor_loss, zero_anchor_metrics = compute_mesh_zero_delta_anchor_loss(
        predicted_delta=predicted,
        vertices=vertices,
        landmarks_3d=record.get("landmarks_3d"),
        target_vertex_mask=vertex_mask,
        config=mesh_config,
    )
    geometry_loss, geometry_metrics = compute_mesh_geometry_regularization(
        vertices=vertices,
        predicted_delta=predicted,
        faces=faces,
        config=mesh_config.get("geometry", {}),
        vertex_mask=vertex_mask,
    )
    rigid_loss, rigid_metrics = compute_mesh_rigid_motion_regularization(
        vertices=vertices,
        predicted_delta=predicted,
        faces=faces,
        vertex_mask=vertex_mask,
        config=mesh_config.get("rigid_motion", {}),
    )
    landmark_loss, landmark_metrics = compute_mesh_landmark_loss(
        predicted_delta=predicted,
        target_delta=target,
        landmarks_3d=record.get("landmarks_3d"),
        config=mesh_config,
        gradient_balance_reference_loss=vertex_loss,
    )
    depth_loss, depth_metrics = compute_mesh_depth_loss(
        vertices=vertices,
        faces=faces,
        predicted_delta=predicted,
        target_delta=target,
        config=mesh_config,
    )
    loss = (
        vertex_loss
        + zero_anchor_loss
        + geometry_loss
        + rigid_loss
        + landmark_loss
        + depth_loss
    )
    metrics = mesh_loss_metrics(
        predicted,
        target,
        loss,
        mesh_config,
        vertex_mask=vertex_loss_mask,
    )
    metrics["mesh_3d_vertex_loss"] = float(vertex_loss.detach().cpu())
    metrics.update(active_motion_weight_metrics(target, mesh_config))
    metrics.update(zero_anchor_metrics)
    metrics.update(geometry_metrics)
    metrics.update(rigid_metrics)
    metrics.update(landmark_metrics)
    metrics.update(depth_metrics)
    return loss, metrics


def compute_image_loss(
    model: nn.Module,
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    from utils.render_utils import differentiable_render

    render_config = config["rendering"]
    mesh = to_device(record["mesh"], device)
    vertices = mesh["vertices"].unsqueeze(0)
    normals = mesh.get("normals")
    if normals is not None:
        normals = normals.unsqueeze(0)
    faces = mesh["faces"]
    action_unit_id = as_int(record["action_unit_id"])
    facs = action_unit_vector(
        action_unit_id,
        int(config["model"]["facs_dim"]),
        device,
        float(config["training"].get("action_unit_scale", 1.0)),
    )

    landmark_features = build_model_landmark_features(
        record=record,
        vertices=vertices,
        config=config,
        device=device,
        training=model.training,
    )
    deformed_vertices, predicted_delta = model(
        vertices,
        facs,
        normals=normals,
        faces=faces,
        landmark_features=landmark_features,
        return_deformed=True,
    )
    raw_predicted_delta = predicted_delta
    image_geometry_config = config_for_action_unit(
        config["loss"].get("image", {}).get("geometry", {}),
        action_unit_id,
    )
    roi_vertex_ids = image_geometry_action_unit_roi_vertex_ids(
        landmarks_3d=record.get("landmarks_3d"),
        action_unit_id=action_unit_id,
        geometry_config=image_geometry_config,
        device=device,
    )
    render_delta = apply_image_geometry_hard_roi_mask(
        vertices=vertices,
        predicted_delta=predicted_delta,
        geometry_config=image_geometry_config,
        render_config=render_config,
        landmark_vertex_ids=roi_vertex_ids,
        faces=faces,
    )
    if render_delta is not predicted_delta:
        predicted_delta = render_delta
        deformed_vertices = vertices + predicted_delta
    render_up_axis = resolve_validation_render_up_axis(config, vertices)
    render_vertices = vertices_to_render_axes(vertices, render_up_axis)
    render_deformed_vertices = vertices_to_render_axes(
        deformed_vertices,
        render_up_axis,
    )
    image_size = normalize_image_size(render_config["train_image_size"])
    texture = {"vertex_colors": vertex_colors(render_vertices)}
    render_kwargs = image_loss_render_kwargs(render_config)
    fit_reference = image_loss_fit_reference(
        render_vertices,
        render_config,
        render_kwargs,
    )
    neutral_render = differentiable_render(
        render_vertices,
        faces,
        texture=texture,
        image_size=image_size,
        backend=str(render_config.get("backend", "auto")),
        fit_to_view_reference=fit_reference,
        **render_kwargs,
    )

    image_loss_config = config["loss"].get("image", {})
    flow_kwargs = image_loss_config.get("flow", {})
    flow_weight = float(flow_kwargs.get("weight", 1.0))
    flow_metrics: dict[str, float]
    if image_flow_loss_enabled(flow_kwargs):
        predicted_flow = model_delta_optical_flow(
            vertices=render_vertices,
            deformed_vertices=render_deformed_vertices,
            faces=faces,
            flow_config=flow_kwargs,
            render_config=render_config,
            image_size=image_size,
        )
        target_flow, target_mask = prepare_image_flow_target(
            record=record,
            image_size=image_size,
            flow_config=flow_kwargs,
            render_config=render_config,
            neutral_render=neutral_render,
            device=device,
            dtype=predicted_flow.dtype,
        )
        predicted_flow = resize_flow_to_size(predicted_flow, target_flow.shape[-2:])
        flow_loss = masked_flow_loss(
            predicted_flow=predicted_flow,
            target_flow=target_flow,
            target_mask=target_mask,
            flow_config=flow_kwargs,
        )
        weighted_flow_loss = flow_loss * flow_weight
        flow_metrics = {
            "image_2d_flow_enabled": 1.0,
            "image_2d_flow_loss": float(flow_loss.detach().cpu()),
            "image_2d_flow_weighted_loss": float(
                weighted_flow_loss.detach().cpu()
            ),
            "image_2d_predicted_flow_mag_mean": float(
                torch.linalg.vector_norm(predicted_flow.detach(), dim=1).mean().cpu()
            ),
            "image_2d_predicted_flow_mag_q99": float(
                torch.quantile(
                    torch.linalg.vector_norm(
                        predicted_flow.detach(), dim=1
                    ).reshape(-1),
                    0.99,
                ).cpu()
            ),
            "image_2d_target_flow_mag_mean": float(
                torch.linalg.vector_norm(target_flow.detach(), dim=1).mean().cpu()
            ),
            "image_2d_target_flow_mag_q99": float(
                torch.quantile(
                    torch.linalg.vector_norm(target_flow.detach(), dim=1).reshape(-1),
                    0.99,
                ).cpu()
            ),
            "image_2d_target_mask_ratio": float(
                (target_mask.detach() > 0.05)
                .to(dtype=target_mask.dtype)
                .mean()
                .cpu()
            ),
        }
    else:
        flow_loss = predicted_delta.sum() * 0.0
        weighted_flow_loss = flow_loss
        flow_metrics = {
            "image_2d_flow_enabled": 0.0,
            "image_2d_flow_loss": 0.0,
            "image_2d_flow_weighted_loss": 0.0,
            "image_2d_predicted_flow_mag_mean": 0.0,
            "image_2d_predicted_flow_mag_q99": 0.0,
            "image_2d_target_flow_mag_mean": 0.0,
            "image_2d_target_flow_mag_q99": 0.0,
            "image_2d_target_mask_ratio": 0.0,
        }
    eyelid_loss, eyelid_metrics = compute_image_eyelid_closure_loss(
        record=record,
        vertices=render_vertices,
        deformed_vertices=render_deformed_vertices,
        faces=faces,
        action_unit_id=action_unit_id,
        image_config=image_loss_config,
        flow_config=flow_kwargs,
        render_config=render_config,
        neutral_render=neutral_render,
        image_size=image_size,
        device=device,
    )
    iris_gaze_loss, iris_gaze_metrics = compute_image_iris_gaze_loss(
        record=record,
        vertices=render_vertices,
        deformed_vertices=render_deformed_vertices,
        action_unit_id=action_unit_id,
        image_config=image_loss_config,
        flow_config=flow_kwargs,
        render_config=render_config,
        image_size=image_size,
        device=device,
    )
    geometry_loss, geometry_metrics = compute_image_geometry_regularization(
        vertices=vertices,
        deformed_vertices=deformed_vertices,
        predicted_delta=raw_predicted_delta,
        faces=faces,
        config=image_geometry_config,
        render_config=render_config,
        landmark_vertex_ids=roi_vertex_ids,
    )
    loss = weighted_flow_loss + eyelid_loss + iris_gaze_loss + geometry_loss
    metrics = {"image_2d_loss": float(loss.detach().cpu())}
    metrics.update(flow_metrics)
    metrics.update(eyelid_metrics)
    metrics.update(iris_gaze_metrics)
    metrics.update(geometry_metrics)
    return loss, metrics


def compute_image_eyelid_closure_loss(
    *,
    record: Mapping[str, Any],
    vertices: torch.Tensor,
    deformed_vertices: torch.Tensor,
    faces: Optional[torch.Tensor] = None,
    action_unit_id: int,
    image_config: Mapping[str, Any],
    flow_config: Mapping[str, Any],
    render_config: Mapping[str, Any],
    neutral_render: Optional[torch.Tensor],
    image_size: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_config = config_for_action_unit(
        image_config.get("eyelid_closure", {}),
        action_unit_id,
    )
    zero = deformed_vertices.sum() * 0.0
    if not isinstance(loss_config, Mapping) or not bool(
        loss_config.get("enabled", False)
    ):
        return zero, {}

    try:
        curves = eyelid_landmarks_for_action_unit(action_unit_id)
    except ValueError:
        return zero, {"image_2d_eyelid_available": 0.0}

    vertex_ids = eyelid_vertex_ids(
        record.get("landmarks_3d"),
        curves.all_ids,
        device=device,
    )
    if vertex_ids is None:
        warn_waft_alignment_once(
            f"missing-eyelid-mesh-landmarks:AU{action_unit_id}",
            f"AU{action_unit_id} image sample is missing one or more required "
            "3D eyelid landmarks; skipping its eyelid-closure loss.",
        )
        return zero, {"image_2d_eyelid_available": 0.0}

    mapping_valid, mapping_metrics = validate_image_eyelid_vertex_mapping(
        vertices=vertices[0],
        vertex_ids=vertex_ids,
        curves=curves,
        config=loss_config,
    )
    if not mapping_valid:
        return zero, {
            "image_2d_eyelid_available": 0.0,
            "image_2d_eyelid_invalid_mapping": 1.0,
            **mapping_metrics,
        }

    displacement_2d = record.get("displacement_2d")
    if not isinstance(displacement_2d, Mapping):
        raise ValueError("Eyelid-closure loss requires displacement_2d metadata.")
    neutral_path = displacement_2d.get("neutral_image_path")
    expressed_path = displacement_2d.get("expressed_image_path")
    if neutral_path in (None, "") or expressed_path in (None, ""):
        raise ValueError(
            "Eyelid-closure loss requires neutral_image_path and "
            "expressed_image_path in the image sample record."
        )

    detection_size = normalize_image_size(
        loss_config.get("landmark_detection_image_size", image_size)
    )
    target_mode = image_eyelid_target_mode(loss_config)
    neutral_image_points = None
    expressed_image_points = None
    if target_mode == "observed":
        neutral_image_by_id = cached_image_mediapipe_landmarks(
            path=str(neutral_path),
            image_size=detection_size,
            flow_config=flow_config,
            alignment=loss_config,
            device=device,
            dtype=vertices.dtype,
        )
        neutral_image_points = eyelid_image_points(
            neutral_image_by_id,
            curves.all_ids,
            from_size=detection_size,
            to_size=image_size,
            device=device,
            dtype=vertices.dtype,
        )
        expressed_image_by_id = cached_image_mediapipe_landmarks(
            path=str(expressed_path),
            image_size=detection_size,
            flow_config=flow_config,
            alignment=loss_config,
            device=device,
            dtype=vertices.dtype,
        )
        expressed_image_points = eyelid_image_points(
            expressed_image_by_id,
            curves.all_ids,
            from_size=detection_size,
            to_size=image_size,
            device=device,
            dtype=vertices.dtype,
        )
    if target_mode == "observed" and (
        neutral_image_points is None or expressed_image_points is None
    ):
        warn_waft_alignment_once(
            f"missing-eyelid-image-landmarks:{expressed_path}",
            f"MediaPipe did not find every required AU{action_unit_id} eyelid "
            "landmark for its eyelid-closure loss; skipping this sample.",
        )
        return zero, {"image_2d_eyelid_available": 0.0}

    # The image-space objective normalizes each domain by its own neutral eye
    # width, so the neutral and expressed landmarks should stay in their shared
    # source-image frame. WAFT alignment is only for the auxiliary flow target.

    render_kwargs = image_loss_render_kwargs(render_config)
    neutral_projected_vertices = validation_project_points_to_screen(
        points=vertices,
        image_size=image_size,
        render_kwargs=render_kwargs,
        fit_reference_vertices=vertices,
    )[0]
    deformed_projected_vertices = validation_project_points_to_screen(
        points=deformed_vertices,
        image_size=image_size,
        render_kwargs=render_kwargs,
        fit_reference_vertices=vertices,
    )[0]
    neutral_mesh_points = neutral_projected_vertices.index_select(0, vertex_ids)
    deformed_mesh_points = deformed_projected_vertices.index_select(0, vertex_ids)

    index_by_id = {
        mediapipe_id: index
        for index, mediapipe_id in enumerate(curves.all_ids)
    }
    pair_indices = torch.tensor(
        [
            (index_by_id[upper_id], index_by_id[lower_id])
            for upper_id, lower_id in curves.closure_pairs
        ],
        device=device,
        dtype=torch.long,
    )
    corner_indices = (
        index_by_id[curves.corners[0]],
        index_by_id[curves.corners[1]],
    )
    if target_mode == "synthetic_closure":
        # Full closure is defined in the projected 100k-mesh frame. Using the
        # source image's neutral gap can request more closure than the mesh has.
        neutral_image_points = neutral_mesh_points.detach()
        expressed_image_points = synthetic_eyelid_closure_target(
            neutral_image_points,
            pair_indices,
            loss_config,
        )
    assert neutral_image_points is not None and expressed_image_points is not None
    raw_loss, component_metrics = normalized_eyelid_closure_objective(
        neutral_mesh_points=neutral_mesh_points,
        deformed_mesh_points=deformed_mesh_points,
        neutral_image_points=neutral_image_points,
        expressed_image_points=expressed_image_points,
        pair_indices=pair_indices,
        corner_indices=corner_indices,
        config=loss_config,
    )
    dense_config = loss_config.get("dense_band", {})
    if isinstance(dense_config, Mapping) and bool(dense_config.get("enabled", False)):
        if faces is None:
            raise ValueError("Dense eyelid-band loss requires mesh faces.")
        dense_roi_mask = image_geometry_landmark_topology_roi_mask(
            vertex_count=vertices.shape[1],
            faces=faces,
            landmark_vertex_ids=vertex_ids,
            config=dense_config,
            batch_size=1,
            device=device,
        )[0]
        dense_loss, dense_metrics = dense_eyelid_band_loss(
            neutral_projected_vertices=neutral_projected_vertices,
            deformed_projected_vertices=deformed_projected_vertices,
            landmark_vertex_ids=vertex_ids,
            neutral_target_landmarks=neutral_image_points,
            expressed_target_landmarks=expressed_image_points,
            upper_indices=torch.tensor(
                [index_by_id[mediapipe_id] for mediapipe_id in curves.upper],
                device=device,
                dtype=torch.long,
            ),
            lower_indices=torch.tensor(
                [index_by_id[mediapipe_id] for mediapipe_id in curves.lower],
                device=device,
                dtype=torch.long,
            ),
            corner_indices=corner_indices,
            roi_mask=dense_roi_mask,
            config=dense_config,
            regression_config=loss_config,
        )
        dense_weight = float(dense_config.get("weight", 1.0))
        raw_loss = raw_loss + dense_loss * dense_weight
        dense_metrics["image_2d_eyelid_dense_band_weighted_loss"] = float(
            (dense_loss.detach() * dense_weight).cpu()
        )
        component_metrics.update(dense_metrics)
    depth_config = loss_config.get("depth_anchor", {})
    if isinstance(depth_config, Mapping) and bool(
        depth_config.get("enabled", False)
    ):
        if faces is None:
            raise ValueError("Eyelid depth-anchor loss requires mesh faces.")
        depth_roi_mask = image_geometry_landmark_topology_roi_mask(
            vertex_count=vertices.shape[1],
            faces=faces,
            landmark_vertex_ids=vertex_ids,
            config=depth_config,
            batch_size=1,
            device=device,
        )[0]
        depth_loss, depth_metrics = eyelid_depth_anchor_loss(
            neutral_vertices=vertices[0],
            deformed_vertices=deformed_vertices[0],
            landmark_vertex_ids=vertex_ids,
            corner_indices=corner_indices,
            roi_mask=depth_roi_mask,
            config=depth_config,
        )
        depth_weight = float(depth_config.get("weight", 1.0))
        raw_loss = raw_loss + depth_loss * depth_weight
        depth_metrics["image_2d_eyelid_depth_anchor_weighted_loss"] = float(
            (depth_loss.detach() * depth_weight).cpu()
        )
        component_metrics.update(depth_metrics)
    weight = float(loss_config.get("weight", 1.0))
    weighted_loss = raw_loss * weight
    metrics = {
        "image_2d_eyelid_available": 1.0,
        "image_2d_eyelid_invalid_mapping": 0.0,
        "image_2d_eyelid_synthetic_target": float(
            target_mode == "synthetic_closure"
        ),
        "image_2d_eyelid_closure_loss": float(raw_loss.detach().cpu()),
        "image_2d_eyelid_closure_weighted_loss": float(
            weighted_loss.detach().cpu()
        ),
    }
    metrics.update(mapping_metrics)
    metrics.update(component_metrics)
    return weighted_loss, metrics


def compute_image_iris_gaze_loss(
    *,
    record: Mapping[str, Any],
    vertices: torch.Tensor,
    deformed_vertices: torch.Tensor,
    action_unit_id: int,
    image_config: Mapping[str, Any],
    flow_config: Mapping[str, Any],
    render_config: Mapping[str, Any],
    image_size: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_config = image_config.get("iris_gaze", {})
    zero = deformed_vertices.sum() * 0.0
    if not isinstance(loss_config, Mapping) or not bool(
        loss_config.get("enabled", False)
    ):
        return zero, {}

    configured_action_units = loss_config.get("action_units")
    if configured_action_units not in (None, ""):
        if action_unit_id not in parse_action_units(configured_action_units):
            return zero, {"image_2d_iris_gaze_available": 0.0}

    iris_ids = normalize_int_list(
        loss_config.get("iris_ids"),
        "loss.image.iris_gaze.iris_ids",
    )
    corner_ids = normalize_int_list(
        loss_config.get("corner_ids"),
        "loss.image.iris_gaze.corner_ids",
    )
    if len(iris_ids) < 1:
        raise ValueError("loss.image.iris_gaze.iris_ids must not be empty.")
    if len(corner_ids) != 2:
        raise ValueError("loss.image.iris_gaze.corner_ids must contain two IDs.")
    center_id = int(loss_config.get("center_id", iris_ids[0]))
    if center_id not in iris_ids:
        raise ValueError("loss.image.iris_gaze.center_id must be in iris_ids.")

    required_ids = list(dict.fromkeys([*iris_ids, *corner_ids]))
    landmarks_3d = record.get("landmarks_3d")
    surface_config = loss_config.get("surface_anchors", {})
    use_surface_anchors = isinstance(surface_config, Mapping) and bool(
        surface_config.get("enabled", False)
    )
    vertex_ids: Optional[torch.Tensor] = None
    surface_anchor_data: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    corner_vertex_ids: Optional[torch.Tensor] = None
    if use_surface_anchors and isinstance(landmarks_3d, Mapping):
        surface_anchor_data = select_surface_anchors(
            landmarks_3d,
            iris_ids,
            device=device,
            dtype=vertices.dtype,
        )
        corner_vertex_ids = eyelid_vertex_ids(
            landmarks_3d,
            corner_ids,
            device=device,
        )
    if surface_anchor_data is None or corner_vertex_ids is None:
        if use_surface_anchors and bool(surface_config.get("required", True)):
            raise ValueError(
                f"AU{action_unit_id} requires cached barycentric iris surface "
                "anchors and two mapped eye corners."
            )
        vertex_ids = eyelid_vertex_ids(
            landmarks_3d,
            required_ids,
            device=device,
        )
        if vertex_ids is None:
            warn_waft_alignment_once(
                f"missing-iris-mesh-landmarks:AU{action_unit_id}",
                f"AU{action_unit_id} image sample is missing one or more required "
                "3D iris landmarks; skipping its iris-gaze loss.",
            )
            return zero, {"image_2d_iris_gaze_available": 0.0}

    if surface_anchor_data is not None:
        unique_iris_vertices = torch.unique(surface_anchor_data[0]).numel()
    else:
        assert vertex_ids is not None
        unique_iris_vertices = torch.unique(vertex_ids[: len(iris_ids)]).numel()
        minimum_unique = int(
            loss_config.get("min_unique_iris_vertices", len(iris_ids))
        )
        if int(unique_iris_vertices) < minimum_unique:
            return zero, {
                "image_2d_iris_gaze_available": 0.0,
                "image_2d_iris_gaze_invalid_mapping": 1.0,
                "image_2d_iris_gaze_unique_vertex_count": float(unique_iris_vertices),
            }

    displacement_2d = record.get("displacement_2d")
    if not isinstance(displacement_2d, Mapping):
        raise ValueError("Iris-gaze loss requires displacement_2d metadata.")
    neutral_path = displacement_2d.get("neutral_image_path")
    expressed_path = displacement_2d.get("expressed_image_path")
    if neutral_path in (None, "") or expressed_path in (None, ""):
        raise ValueError(
            "Iris-gaze loss requires neutral_image_path and expressed_image_path."
        )

    detection_size = normalize_image_size(
        loss_config.get("landmark_detection_image_size", image_size)
    )
    detection_config = dict(loss_config)
    detection_config["refine_landmarks"] = True
    neutral_image_by_id = cached_image_mediapipe_landmarks(
        path=str(neutral_path),
        image_size=detection_size,
        flow_config=flow_config,
        alignment=detection_config,
        device=device,
        dtype=vertices.dtype,
    )
    expressed_image_by_id = cached_image_mediapipe_landmarks(
        path=str(expressed_path),
        image_size=detection_size,
        flow_config=flow_config,
        alignment=detection_config,
        device=device,
        dtype=vertices.dtype,
    )
    neutral_image_points = eyelid_image_points(
        neutral_image_by_id,
        required_ids,
        from_size=detection_size,
        to_size=image_size,
        device=device,
        dtype=vertices.dtype,
    )
    expressed_image_points = eyelid_image_points(
        expressed_image_by_id,
        required_ids,
        from_size=detection_size,
        to_size=image_size,
        device=device,
        dtype=vertices.dtype,
    )
    if neutral_image_points is None or expressed_image_points is None:
        warn_waft_alignment_once(
            f"missing-iris-image-landmarks:{expressed_path}",
            f"MediaPipe did not find every required AU{action_unit_id} iris "
            "landmark; skipping this sample's iris-gaze loss.",
        )
        return zero, {"image_2d_iris_gaze_available": 0.0}

    render_kwargs = image_loss_render_kwargs(render_config)
    if surface_anchor_data is not None:
        assert corner_vertex_ids is not None
        anchor_face_ids, anchor_weights = surface_anchor_data
        neutral_iris_points = interpolate_surface_anchors(
            vertices,
            anchor_face_ids,
            anchor_weights,
        )
        deformed_iris_points = interpolate_surface_anchors(
            deformed_vertices,
            anchor_face_ids,
            anchor_weights,
        )
        neutral_mesh_points_3d = torch.cat(
            (neutral_iris_points, vertices[:, corner_vertex_ids]),
            dim=1,
        )
        deformed_mesh_points_3d = torch.cat(
            (deformed_iris_points, deformed_vertices[:, corner_vertex_ids]),
            dim=1,
        )
        neutral_projected = validation_project_points_to_screen(
            points=neutral_mesh_points_3d,
            image_size=image_size,
            render_kwargs=render_kwargs,
            fit_reference_vertices=vertices,
        )[0]
        deformed_projected = validation_project_points_to_screen(
            points=deformed_mesh_points_3d,
            image_size=image_size,
            render_kwargs=render_kwargs,
            fit_reference_vertices=vertices,
        )[0]
    else:
        assert vertex_ids is not None
        neutral_projected = validation_project_points_to_screen(
            points=vertices,
            image_size=image_size,
            render_kwargs=render_kwargs,
            fit_reference_vertices=vertices,
        )[0].index_select(0, vertex_ids)
        deformed_projected = validation_project_points_to_screen(
            points=deformed_vertices,
            image_size=image_size,
            render_kwargs=render_kwargs,
            fit_reference_vertices=vertices,
        )[0].index_select(0, vertex_ids)

    index_by_id = {
        mediapipe_id: index for index, mediapipe_id in enumerate(required_ids)
    }
    iris_indices = torch.tensor(
        [index_by_id[mediapipe_id] for mediapipe_id in iris_ids],
        device=device,
        dtype=torch.long,
    )
    center_index = index_by_id[center_id]
    corner_indices = (
        index_by_id[corner_ids[0]],
        index_by_id[corner_ids[1]],
    )
    raw_loss, component_metrics = normalized_iris_gaze_objective(
        neutral_mesh_points=neutral_projected,
        deformed_mesh_points=deformed_projected,
        neutral_image_points=neutral_image_points,
        expressed_image_points=expressed_image_points,
        iris_indices=iris_indices,
        center_index=center_index,
        corner_indices=corner_indices,
        config=loss_config,
    )
    weight = float(loss_config.get("weight", 1.0))
    weighted_loss = raw_loss * weight
    rigidity_loss, rigidity_metrics = image_iris_surface_rigidity_loss(
        landmarks_3d=landmarks_3d,
        vertices=vertices,
        deformed_vertices=deformed_vertices,
        config=surface_config if isinstance(surface_config, Mapping) else {},
        device=device,
    )
    total_loss = weighted_loss + rigidity_loss
    metrics = {
        "image_2d_iris_gaze_available": 1.0,
        "image_2d_iris_gaze_invalid_mapping": 0.0,
        "image_2d_iris_gaze_unique_vertex_count": float(unique_iris_vertices),
        "image_2d_iris_gaze_loss": float(raw_loss.detach().cpu()),
        "image_2d_iris_gaze_weighted_loss": float(weighted_loss.detach().cpu()),
        "image_2d_iris_gaze_surface_anchors": float(surface_anchor_data is not None),
    }
    metrics.update(component_metrics)
    metrics.update(rigidity_metrics)
    return total_loss, metrics


def image_iris_surface_rigidity_loss(
    *,
    landmarks_3d: Any,
    vertices: torch.Tensor,
    deformed_vertices: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = deformed_vertices.sum() * 0.0
    weight = float(config.get("rigidity_weight", 0.0))
    if weight <= 0.0:
        return zero, {}
    if not isinstance(landmarks_3d, Mapping):
        if bool(config.get("required", True)):
            raise ValueError("Iris surface rigidity requires gaze anchor metadata.")
        return zero, {}
    roi_vertex_ids = landmarks_3d.get(
        "gaze_rigid_vertex_ids",
        landmarks_3d.get("gaze_roi_vertex_ids"),
    )
    if not isinstance(roi_vertex_ids, torch.Tensor):
        if bool(config.get("required", True)):
            raise ValueError("Iris surface rigidity requires gaze_roi_vertex_ids.")
        return zero, {}
    roi_vertex_ids = roi_vertex_ids.detach().long().flatten().unique(sorted=True).to(device)
    if roi_vertex_ids.numel() < 3:
        raise ValueError("Iris surface rigidity ROI must contain at least three vertices.")
    if int(roi_vertex_ids.min()) < 0 or int(roi_vertex_ids.max()) >= vertices.shape[1]:
        raise ValueError("Iris surface rigidity ROI contains an out-of-range vertex.")

    source = vertices[0].index_select(0, roi_vertex_ids)
    destination = deformed_vertices[0].index_select(0, roi_vertex_ids)
    predicted_delta = destination - source
    rigid_delta = rigid_fit_delta(source, destination)
    scale = torch.linalg.vector_norm(
        source.amax(dim=0) - source.amin(dim=0)
    ).clamp_min(float(config.get("rigidity_min_scale", 1.0e-3)))
    raw = ((predicted_delta - rigid_delta) / scale).square().mean()
    weighted = raw * weight
    return weighted, {
        "image_2d_iris_surface_rigidity_loss": float(raw.detach().cpu()),
        "image_2d_iris_surface_rigidity_weighted_loss": float(
            weighted.detach().cpu()
        ),
        "image_2d_iris_surface_roi_vertex_count": float(roi_vertex_ids.numel()),
    }


def normalized_iris_gaze_objective(
    *,
    neutral_mesh_points: torch.Tensor,
    deformed_mesh_points: torch.Tensor,
    neutral_image_points: torch.Tensor,
    expressed_image_points: torch.Tensor,
    iris_indices: torch.Tensor,
    center_index: int,
    corner_indices: tuple[int, int],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    shape = neutral_mesh_points.shape
    if (
        shape != deformed_mesh_points.shape
        or shape != neutral_image_points.shape
        or shape != expressed_image_points.shape
    ):
        raise ValueError("Mesh and image iris point tensors must match.")
    if neutral_mesh_points.dim() != 2 or neutral_mesh_points.shape[-1] != 2:
        raise ValueError("Iris points must have shape [L, 2].")
    if iris_indices.numel() < 1:
        raise ValueError("iris_indices must not be empty.")

    minimum_width = float(config.get("min_eye_width_px", 1.0))
    mesh_eye_width = torch.linalg.vector_norm(
        neutral_mesh_points[corner_indices[1]]
        - neutral_mesh_points[corner_indices[0]]
    ).clamp_min(minimum_width)
    image_eye_width = torch.linalg.vector_norm(
        neutral_image_points[corner_indices[1]]
        - neutral_image_points[corner_indices[0]]
    ).clamp_min(minimum_width)
    predicted_motion = (deformed_mesh_points - neutral_mesh_points) / mesh_eye_width
    target_motion = (expressed_image_points - neutral_image_points) / image_eye_width

    if bool(config.get("remove_corner_translation", True)):
        corner_tensor = torch.tensor(
            corner_indices,
            device=neutral_mesh_points.device,
            dtype=torch.long,
        )
        predicted_motion = predicted_motion - predicted_motion.index_select(
            0, corner_tensor
        ).mean(dim=0, keepdim=True)
        target_motion = target_motion - target_motion.index_select(
            0, corner_tensor
        ).mean(dim=0, keepdim=True)
    target_motion = target_motion * float(config.get("target_displacement_scale", 1.0))
    target_mode = str(config.get("target_motion_mode", "independent")).strip().lower()
    if target_mode in {"coherent", "coherent_translation", "rigid_translation"}:
        coherent_target = target_motion.index_select(0, iris_indices).mean(
            dim=0,
            keepdim=True,
        )
        target_motion = target_motion.clone()
        target_motion[iris_indices] = coherent_target.expand(iris_indices.numel(), -1)
    elif target_mode not in {"independent", "landmarks", "per_landmark"}:
        raise ValueError(
            "loss.image.iris_gaze.target_motion_mode must be independent or "
            "coherent_translation."
        )

    center_loss = eyelid_regression_loss(
        predicted_motion[center_index],
        target_motion[center_index],
        config,
    )
    perimeter_indices = iris_indices[iris_indices != int(center_index)]
    if perimeter_indices.numel() > 0:
        perimeter_loss = eyelid_regression_loss(
            predicted_motion.index_select(0, perimeter_indices),
            target_motion.index_select(0, perimeter_indices),
            config,
        )
    else:
        perimeter_loss = center_loss * 0.0
    center_weight = float(config.get("center_weight", 2.0))
    perimeter_weight = float(config.get("perimeter_weight", 1.0))
    loss = center_loss * center_weight + perimeter_loss * perimeter_weight

    predicted_center_motion = predicted_motion[center_index]
    target_center_motion = target_motion[center_index]
    return loss, {
        "image_2d_iris_gaze_center_loss": float(center_loss.detach().cpu()),
        "image_2d_iris_gaze_perimeter_loss": float(perimeter_loss.detach().cpu()),
        "image_2d_iris_gaze_mesh_eye_width_px": float(mesh_eye_width.detach().cpu()),
        "image_2d_iris_gaze_image_eye_width_px": float(image_eye_width.detach().cpu()),
        "image_2d_iris_gaze_predicted_center_dx": float(
            predicted_center_motion[0].detach().cpu()
        ),
        "image_2d_iris_gaze_predicted_center_dy": float(
            predicted_center_motion[1].detach().cpu()
        ),
        "image_2d_iris_gaze_target_center_dx": float(
            target_center_motion[0].detach().cpu()
        ),
        "image_2d_iris_gaze_target_center_dy": float(
            target_center_motion[1].detach().cpu()
        ),
    }


def eyelid_vertex_ids(
    landmarks_3d: Any,
    required_ids: Sequence[int],
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not isinstance(landmarks_3d, Mapping):
        return None
    mediapipe_ids = landmarks_3d.get("mediapipe_ids")
    vertex_ids = landmarks_3d.get("vertex_ids")
    if not isinstance(mediapipe_ids, torch.Tensor) or not isinstance(
        vertex_ids, torch.Tensor
    ):
        return None
    mediapipe_values = mediapipe_ids.detach().cpu().long().flatten().tolist()
    vertex_values = vertex_ids.detach().cpu().long().flatten().tolist()
    if len(mediapipe_values) != len(vertex_values):
        raise ValueError("landmarks_3d.mediapipe_ids must match vertex_ids.")
    vertex_by_id = dict(zip(mediapipe_values, vertex_values))
    if any(int(mediapipe_id) not in vertex_by_id for mediapipe_id in required_ids):
        return None
    return torch.tensor(
        [vertex_by_id[int(mediapipe_id)] for mediapipe_id in required_ids],
        device=device,
        dtype=torch.long,
    )


def validate_image_eyelid_vertex_mapping(
    *,
    vertices: torch.Tensor,
    vertex_ids: torch.Tensor,
    curves: Any,
    config: Mapping[str, Any],
) -> tuple[bool, dict[str, float]]:
    if vertices.dim() != 2 or vertices.shape[-1] != 3:
        raise ValueError("Eyelid mapping validation requires vertices shaped [V, 3].")
    vertex_ids = vertex_ids.to(device=vertices.device, dtype=torch.long)
    if vertex_ids.numel() != len(curves.all_ids):
        return False, {
            "image_2d_eyelid_mapping_landmark_count": float(vertex_ids.numel())
        }
    if bool(((vertex_ids < 0) | (vertex_ids >= vertices.shape[0])).any()):
        return False, {"image_2d_eyelid_mapping_out_of_range": 1.0}

    points = vertices.index_select(0, vertex_ids)
    index_by_id = {
        mediapipe_id: index
        for index, mediapipe_id in enumerate(curves.all_ids)
    }
    corner_a = index_by_id[curves.corners[0]]
    corner_b = index_by_id[curves.corners[1]]
    eye_width = torch.linalg.vector_norm(points[corner_b] - points[corner_a])
    gaps = torch.stack(
        [
            torch.linalg.vector_norm(
                points[index_by_id[upper_id]] - points[index_by_id[lower_id]]
            )
            for upper_id, lower_id in curves.closure_pairs
        ]
    )
    unique_count = int(torch.unique(vertex_ids).numel())
    width_value = float(eye_width.detach().cpu())
    mean_gap_ratio = float(
        (gaps.mean() / eye_width.clamp_min(1.0e-12)).detach().cpu()
    )
    minimum_unique = int(config.get("min_unique_vertices", 2))
    minimum_width = float(config.get("min_eye_width_mesh_units", 1.0e-3))
    maximum_gap_ratio = float(
        config.get("max_neutral_gap_eye_width_ratio", float("inf"))
    )
    valid = (
        unique_count >= minimum_unique
        and width_value >= minimum_width
        and mean_gap_ratio <= maximum_gap_ratio
    )
    return valid, {
        "image_2d_eyelid_mapping_unique_vertex_count": float(unique_count),
        "image_2d_eyelid_mapping_eye_width": width_value,
        "image_2d_eyelid_mapping_mean_gap_eye_width_ratio": mean_gap_ratio,
    }


def eyelid_image_points(
    landmarks_by_id: Mapping[int, tuple[float, float]],
    required_ids: Sequence[int],
    *,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if any(int(mediapipe_id) not in landmarks_by_id for mediapipe_id in required_ids):
        return None
    points = torch.tensor(
        [landmarks_by_id[int(mediapipe_id)] for mediapipe_id in required_ids],
        device=device,
        dtype=dtype,
    )
    if tuple(from_size) == tuple(to_size):
        return points
    from_height, from_width = from_size
    to_height, to_width = to_size
    scale = points.new_tensor(
        (
            float(max(to_width - 1, 1)) / float(max(from_width - 1, 1)),
            float(max(to_height - 1, 1)) / float(max(from_height - 1, 1)),
        )
    )
    return points * scale


def image_eyelid_target_mode(config: Mapping[str, Any]) -> str:
    mode = str(config.get("target_mode", "observed")).strip().lower()
    if mode in {"observed", "image", "expressed", "mediapipe"}:
        return "observed"
    if mode in {"synthetic", "synthetic_closure", "full_closure", "closed"}:
        return "synthetic_closure"
    raise ValueError(
        "loss.image.eyelid_closure.target_mode must be observed or "
        "synthetic_closure."
    )


def synthetic_eyelid_closure_target(
    neutral_points: torch.Tensor,
    pair_indices: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    if neutral_points.dim() != 2 or neutral_points.shape[-1] != 2:
        raise ValueError("neutral_points must have shape [L, 2].")
    if pair_indices.dim() != 2 or pair_indices.shape[-1] != 2:
        raise ValueError("pair_indices must have shape [P, 2].")
    upper_fraction = float(config.get("synthetic_upper_closure_fraction", 0.8))
    lower_fraction = float(config.get("synthetic_lower_closure_fraction", 0.2))
    if upper_fraction < 0.0 or lower_fraction < 0.0:
        raise ValueError("Synthetic eyelid closure fractions must be non-negative.")
    if upper_fraction + lower_fraction > 1.0 + 1.0e-6:
        raise ValueError(
            "Synthetic upper and lower eyelid closure fractions must sum to at "
            "most 1.0."
        )

    upper_indices = pair_indices[:, 0]
    lower_indices = pair_indices[:, 1]
    upper_points = neutral_points.index_select(0, upper_indices)
    lower_points = neutral_points.index_select(0, lower_indices)
    target = neutral_points.clone()
    target.index_copy_(
        0,
        upper_indices,
        upper_points + (lower_points - upper_points) * upper_fraction,
    )
    target.index_copy_(
        0,
        lower_indices,
        lower_points + (upper_points - lower_points) * lower_fraction,
    )
    return target




def normalized_eyelid_closure_objective(
    *,
    neutral_mesh_points: torch.Tensor,
    deformed_mesh_points: torch.Tensor,
    neutral_image_points: torch.Tensor,
    expressed_image_points: torch.Tensor,
    pair_indices: torch.Tensor,
    corner_indices: tuple[int, int],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    point_shape = neutral_mesh_points.shape
    if point_shape != deformed_mesh_points.shape:
        raise ValueError("Neutral and deformed mesh eyelid points must match.")
    if (
        point_shape != neutral_image_points.shape
        or point_shape != expressed_image_points.shape
    ):
        raise ValueError("Mesh and image eyelid point tensors must match.")
    if neutral_mesh_points.dim() != 2 or neutral_mesh_points.shape[-1] != 2:
        raise ValueError("Eyelid points must have shape [L, 2].")
    if pair_indices.dim() != 2 or pair_indices.shape[-1] != 2:
        raise ValueError("pair_indices must have shape [P, 2].")

    minimum_width = float(config.get("min_eye_width_px", 1.0))
    mesh_eye_width = torch.linalg.vector_norm(
        neutral_mesh_points[corner_indices[1]]
        - neutral_mesh_points[corner_indices[0]]
    ).clamp_min(minimum_width)
    image_eye_width = torch.linalg.vector_norm(
        neutral_image_points[corner_indices[1]]
        - neutral_image_points[corner_indices[0]]
    ).clamp_min(minimum_width)

    predicted_motion = (deformed_mesh_points - neutral_mesh_points) / mesh_eye_width
    target_motion = (expressed_image_points - neutral_image_points) / image_eye_width
    if bool(config.get("remove_corner_translation", True)):
        corner_index_tensor = torch.tensor(
            corner_indices,
            device=neutral_mesh_points.device,
            dtype=torch.long,
        )
        predicted_motion = predicted_motion - predicted_motion.index_select(
            0,
            corner_index_tensor,
        ).mean(dim=0, keepdim=True)
        target_motion = target_motion - target_motion.index_select(
            0,
            corner_index_tensor,
        ).mean(dim=0, keepdim=True)
    displacement_loss = eyelid_regression_loss(
        predicted_motion,
        target_motion,
        config,
    )

    upper_indices = pair_indices[:, 0]
    lower_indices = pair_indices[:, 1]
    neutral_mesh_gap = torch.linalg.vector_norm(
        neutral_mesh_points.index_select(0, upper_indices)
        - neutral_mesh_points.index_select(0, lower_indices),
        dim=-1,
    )
    deformed_mesh_gap = torch.linalg.vector_norm(
        deformed_mesh_points.index_select(0, upper_indices)
        - deformed_mesh_points.index_select(0, lower_indices),
        dim=-1,
    )
    neutral_image_gap = torch.linalg.vector_norm(
        neutral_image_points.index_select(0, upper_indices)
        - neutral_image_points.index_select(0, lower_indices),
        dim=-1,
    )
    expressed_image_gap = torch.linalg.vector_norm(
        expressed_image_points.index_select(0, upper_indices)
        - expressed_image_points.index_select(0, lower_indices),
        dim=-1,
    )
    predicted_gap_reduction = (
        neutral_mesh_gap - deformed_mesh_gap
    ) / mesh_eye_width
    maximum_gap_reduction = neutral_mesh_gap / mesh_eye_width
    target_gap_reduction = (
        neutral_image_gap - expressed_image_gap
    ) / image_eye_width
    gap_loss = eyelid_regression_loss(
        predicted_gap_reduction,
        target_gap_reduction,
        config,
    )

    displacement_weight = float(config.get("displacement_weight", 1.0))
    gap_weight = float(config.get("gap_reduction_weight", 1.0))
    loss = displacement_loss * displacement_weight + gap_loss * gap_weight
    metrics = {
        "image_2d_eyelid_displacement_loss": float(displacement_loss.detach().cpu()),
        "image_2d_eyelid_gap_reduction_loss": float(gap_loss.detach().cpu()),
        "image_2d_eyelid_predicted_motion_mean": float(
            torch.linalg.vector_norm(predicted_motion.detach(), dim=-1).mean().cpu()
        ),
        "image_2d_eyelid_target_motion_mean": float(
            torch.linalg.vector_norm(target_motion.detach(), dim=-1).mean().cpu()
        ),
        "image_2d_eyelid_predicted_gap_reduction_mean": float(
            predicted_gap_reduction.detach().mean().cpu()
        ),
        "image_2d_eyelid_target_gap_reduction_mean": float(
            target_gap_reduction.detach().mean().cpu()
        ),
        "image_2d_eyelid_max_gap_reduction_mean": float(
            maximum_gap_reduction.detach().mean().cpu()
        ),
        "image_2d_eyelid_predicted_closure_fraction_mean": float(
            (
                predicted_gap_reduction.detach()
                / maximum_gap_reduction.detach().clamp_min(1.0e-6)
            )
            .mean()
            .cpu()
        ),
    }
    return loss, metrics


def dense_eyelid_band_loss(
    *,
    neutral_projected_vertices: torch.Tensor,
    deformed_projected_vertices: torch.Tensor,
    landmark_vertex_ids: torch.Tensor,
    neutral_target_landmarks: torch.Tensor,
    expressed_target_landmarks: torch.Tensor,
    upper_indices: torch.Tensor,
    lower_indices: torch.Tensor,
    corner_indices: tuple[int, int],
    roi_mask: torch.Tensor,
    config: Mapping[str, Any],
    regression_config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    if neutral_projected_vertices.shape != deformed_projected_vertices.shape:
        raise ValueError("Dense eyelid neutral and deformed vertices must match.")
    if neutral_projected_vertices.dim() != 2 or neutral_projected_vertices.shape[-1] != 2:
        raise ValueError("Dense eyelid projected vertices must have shape [V, 2].")
    if roi_mask.shape != neutral_projected_vertices.shape[:1]:
        raise ValueError("Dense eyelid roi_mask must have shape [V].")

    landmark_vertex_ids = landmark_vertex_ids.to(
        device=neutral_projected_vertices.device,
        dtype=torch.long,
    )
    neutral_mesh_landmarks = neutral_projected_vertices.index_select(
        0,
        landmark_vertex_ids,
    )
    deformed_mesh_landmarks = deformed_projected_vertices.index_select(
        0,
        landmark_vertex_ids,
    )
    mesh_eye_width = torch.linalg.vector_norm(
        neutral_mesh_landmarks[corner_indices[1]]
        - neutral_mesh_landmarks[corner_indices[0]]
    ).clamp_min(1.0e-6)
    target_eye_width = torch.linalg.vector_norm(
        neutral_target_landmarks[corner_indices[1]]
        - neutral_target_landmarks[corner_indices[0]]
    ).clamp_min(1.0e-6)

    target_landmark_motion = (
        expressed_target_landmarks - neutral_target_landmarks
    ) / target_eye_width
    predicted_corner_motion = (
        deformed_mesh_landmarks - neutral_mesh_landmarks
    ) / mesh_eye_width
    if bool(regression_config.get("remove_corner_translation", True)):
        corner_ids = torch.tensor(
            corner_indices,
            device=neutral_projected_vertices.device,
            dtype=torch.long,
        )
        target_landmark_motion = target_landmark_motion - target_landmark_motion.index_select(
            0,
            corner_ids,
        ).mean(dim=0, keepdim=True)
        corner_translation = predicted_corner_motion.index_select(
            0,
            corner_ids,
        ).mean(dim=0, keepdim=True)
    else:
        corner_translation = predicted_corner_motion.new_zeros((1, 2))

    roi_vertex_ids = torch.where(roi_mask.to(dtype=torch.bool))[0]
    if roi_vertex_ids.numel() == 0:
        zero = deformed_projected_vertices.sum() * 0.0
        return zero, {
            "image_2d_eyelid_dense_band_loss": 0.0,
            "image_2d_eyelid_dense_band_vertex_count": 0.0,
        }
    neutral_roi = neutral_projected_vertices.index_select(0, roi_vertex_ids)
    normalized_distances = torch.cdist(
        neutral_roi / mesh_eye_width,
        neutral_mesh_landmarks / mesh_eye_width,
    )
    upper_indices = upper_indices.to(
        device=neutral_projected_vertices.device,
        dtype=torch.long,
    )
    lower_indices = lower_indices.to(
        device=neutral_projected_vertices.device,
        dtype=torch.long,
    )
    if upper_indices.numel() == 0 or lower_indices.numel() == 0:
        raise ValueError("Dense eyelid curves must each contain at least one landmark.")

    def interpolate_curve_motion(
        curve_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        curve_distances = normalized_distances.index_select(1, curve_indices)
        curve_k_nearest = min(
            max(1, int(config.get("k_nearest_landmarks", 3))),
            curve_distances.shape[1],
        )
        nearest_distances, nearest_curve_indices = curve_distances.topk(
            curve_k_nearest,
            dim=-1,
            largest=False,
        )
        nearest_landmark_indices = curve_indices.index_select(
            0,
            nearest_curve_indices.reshape(-1),
        ).reshape_as(nearest_curve_indices)
        inverse_distance_power = float(config.get("inverse_distance_power", 2.0))
        weights = (nearest_distances + 1.0e-6).pow(-inverse_distance_power)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        target_neighbors = target_landmark_motion.index_select(
            0,
            nearest_landmark_indices.reshape(-1),
        ).reshape(nearest_landmark_indices.shape[0], nearest_landmark_indices.shape[1], 2)
        return (
            (target_neighbors * weights.unsqueeze(-1)).sum(dim=1),
            nearest_distances[:, 0],
        )

    upper_target, upper_distance = interpolate_curve_motion(upper_indices)
    lower_target, lower_distance = interpolate_curve_motion(lower_indices)
    use_upper_curve = upper_distance <= lower_distance
    interpolated_target = torch.where(
        use_upper_curve.unsqueeze(-1),
        upper_target,
        lower_target,
    )
    nearest_distance = torch.minimum(upper_distance, lower_distance)
    max_distance = float(config.get("max_distance_eye_widths", 0.15))
    if max_distance <= 0.0:
        raise ValueError("Dense eyelid max_distance_eye_widths must be positive.")
    distance_falloff = (1.0 - nearest_distance / max_distance).clamp(0.0, 1.0)
    falloff_power = float(config.get("distance_falloff_power", 1.0))
    if falloff_power <= 0.0:
        raise ValueError("Dense eyelid distance_falloff_power must be positive.")
    distance_falloff = distance_falloff.pow(falloff_power)
    active_mask = distance_falloff > 0.0
    supervise_full_roi = bool(config.get("supervise_full_roi", False))
    supervision_mask = (
        torch.ones_like(active_mask) if supervise_full_roi else active_mask
    )
    if not bool(supervision_mask.any().detach().cpu()):
        zero = deformed_projected_vertices.sum() * 0.0
        return zero, {
            "image_2d_eyelid_dense_band_loss": 0.0,
            "image_2d_eyelid_dense_band_vertex_count": 0.0,
        }
    interpolated_target = interpolated_target * distance_falloff.unsqueeze(-1)

    deformed_roi = deformed_projected_vertices.index_select(0, roi_vertex_ids)
    predicted_motion = (
        deformed_roi - neutral_roi
    ) / mesh_eye_width - corner_translation
    predicted_band = predicted_motion[supervision_mask]
    target_band = interpolated_target[supervision_mask]
    loss = eyelid_regression_loss(predicted_band, target_band, regression_config)
    return loss, {
        "image_2d_eyelid_dense_band_loss": float(loss.detach().cpu()),
        "image_2d_eyelid_dense_band_vertex_count": float(
            supervision_mask.sum().detach().cpu()
        ),
        "image_2d_eyelid_dense_band_active_vertex_count": float(
            active_mask.sum().detach().cpu()
        ),
        "image_2d_eyelid_dense_band_roi_count": float(roi_vertex_ids.numel()),
        "image_2d_eyelid_dense_band_predicted_motion_mean": float(
            torch.linalg.vector_norm(predicted_band.detach(), dim=-1).mean().cpu()
        ),
        "image_2d_eyelid_dense_band_target_motion_mean": float(
            torch.linalg.vector_norm(target_band.detach(), dim=-1).mean().cpu()
        ),
        "image_2d_eyelid_dense_band_falloff_mean": float(
            distance_falloff.detach().mean().cpu()
        ),
    }


def eyelid_depth_anchor_loss(
    *,
    neutral_vertices: torch.Tensor,
    deformed_vertices: torch.Tensor,
    landmark_vertex_ids: torch.Tensor,
    corner_indices: tuple[int, int],
    roi_mask: torch.Tensor,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    if neutral_vertices.shape != deformed_vertices.shape:
        raise ValueError("Eyelid depth-anchor neutral and deformed vertices must match.")
    if neutral_vertices.dim() != 2 or neutral_vertices.shape[-1] != 3:
        raise ValueError("Eyelid depth-anchor vertices must have shape [V, 3].")
    if roi_mask.shape != neutral_vertices.shape[:1]:
        raise ValueError("Eyelid depth-anchor roi_mask must have shape [V].")

    landmark_vertex_ids = landmark_vertex_ids.to(
        device=neutral_vertices.device,
        dtype=torch.long,
    )
    neutral_landmarks = neutral_vertices.index_select(0, landmark_vertex_ids)
    eye_width = torch.linalg.vector_norm(
        neutral_landmarks[corner_indices[1]]
        - neutral_landmarks[corner_indices[0]]
    ).clamp_min(1.0e-6)
    roi_vertex_ids = torch.where(roi_mask.to(dtype=torch.bool))[0]
    if roi_vertex_ids.numel() == 0:
        zero = deformed_vertices.sum() * 0.0
        return zero, {
            "image_2d_eyelid_depth_anchor_loss": 0.0,
            "image_2d_eyelid_depth_anchor_vertex_count": 0.0,
        }

    depth_motion = (
        deformed_vertices.index_select(0, roi_vertex_ids)[:, 2]
        - neutral_vertices.index_select(0, roi_vertex_ids)[:, 2]
    ) / eye_width
    loss = eyelid_regression_loss(
        depth_motion,
        torch.zeros_like(depth_motion),
        config,
    )
    return loss, {
        "image_2d_eyelid_depth_anchor_loss": float(loss.detach().cpu()),
        "image_2d_eyelid_depth_anchor_vertex_count": float(
            roi_vertex_ids.numel()
        ),
        "image_2d_eyelid_depth_motion_abs_mean": float(
            depth_motion.detach().abs().mean().cpu()
        ),
        "image_2d_eyelid_depth_motion_abs_max": float(
            depth_motion.detach().abs().amax().cpu()
        ),
    }


def eyelid_regression_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    loss_type = str(config.get("type", "smooth_l1")).strip().lower()
    if loss_type in {"smooth_l1", "huber"}:
        beta = float(config.get("smooth_l1_beta", 0.05))
        return F.smooth_l1_loss(predicted, target, beta=beta)
    if loss_type in {"l1", "mae", "absolute"}:
        return F.l1_loss(predicted, target)
    if loss_type in {"l2", "mse", "squared"}:
        return F.mse_loss(predicted, target)
    raise ValueError(
        "loss.image.eyelid_closure.type must be smooth_l1, l1, or l2."
    )


def image_loss_render_kwargs(render_config: Mapping[str, Any]) -> dict[str, Any]:
    render_kwargs = dict(render_config.get("kwargs", {}))
    shared_render_framing = bool(render_config.get("shared_fit_to_view", True))
    if shared_render_framing and bool(
        render_kwargs.get("fit_visible_component", False)
    ):
        render_kwargs["fit_visible_component"] = False
    return render_kwargs


def image_loss_fit_reference(
    vertices: torch.Tensor,
    render_config: Mapping[str, Any],
    render_kwargs: Mapping[str, Any],
) -> Optional[torch.Tensor]:
    if not bool(render_config.get("shared_fit_to_view", True)):
        return None
    if not bool(render_kwargs.get("fit_to_view", True)):
        return None
    return vertices


def model_delta_optical_flow(
    vertices: torch.Tensor,
    deformed_vertices: torch.Tensor,
    faces: torch.Tensor,
    flow_config: Mapping[str, Any],
    render_config: Mapping[str, Any],
    image_size: tuple[int, int],
) -> torch.Tensor:
    if vertices.dim() != 3 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape [B, V, 3].")
    if deformed_vertices.shape != vertices.shape:
        raise ValueError("deformed_vertices must match vertices shape.")

    flow_size = model_flow_image_size(flow_config, image_size)
    render_kwargs = image_loss_render_kwargs(render_config)
    shared_render_framing = bool(render_config.get("shared_fit_to_view", True))
    fit_to_view = bool(render_kwargs.get("fit_to_view", True))
    neutral_reference = vertices if fit_to_view else None
    deformed_reference = vertices if shared_render_framing and fit_to_view else None

    neutral_screen = project_vertices_to_flow_screen(
        vertices,
        image_size=flow_size,
        render_kwargs=render_kwargs,
        fit_reference_vertices=neutral_reference,
    )
    deformed_screen = project_vertices_to_flow_screen(
        deformed_vertices,
        image_size=flow_size,
        render_kwargs=render_kwargs,
        fit_reference_vertices=deformed_reference,
    )
    vertex_flow = deformed_screen[..., :2] - neutral_screen[..., :2]
    sample_count = int(
        flow_config.get(
            "model_flow_samples_per_face",
            render_kwargs.get("torch_samples_per_face", 1),
        )
    )
    barycentric_samples = flow_barycentric_sample_pattern(
        sample_count,
        device=vertices.device,
        dtype=vertices.dtype,
    )
    faces_batched = flow_faces_batched(faces, vertices.shape[0], vertices.device)
    sample_screen = sample_face_values(
        neutral_screen,
        faces_batched,
        barycentric_samples,
    )
    sample_flow = sample_face_values(
        vertex_flow,
        faces_batched,
        barycentric_samples,
    )

    flow_scale = (
        float(flow_size[0]) / float(max(image_size[0], 1))
        + float(flow_size[1]) / float(max(image_size[1], 1))
    ) * 0.5
    splat_radius_px = float(
        flow_config.get(
            "model_flow_splat_radius_px",
            float(render_kwargs.get("splat_radius_px", 0.75)) * flow_scale,
        )
    )
    splat_kernel_radius = int(
        flow_config.get(
            "model_flow_splat_kernel_radius",
            max(
                1,
                int(round(float(render_kwargs.get("splat_kernel_radius", 1)) * flow_scale)),
            ),
        )
    )
    dense_flow = soft_splat_sample_values(
        sample_positions=sample_screen[..., :2],
        sample_depths=sample_screen[..., 2],
        sample_values=sample_flow,
        image_size=flow_size,
        background=vertices.new_zeros(2),
        splat_radius_px=splat_radius_px,
        splat_kernel_radius=splat_kernel_radius,
        depth_sharpness=float(render_kwargs.get("depth_sharpness", 100.0)),
        chunk_size=int(render_kwargs.get("chunk_size", 200000)),
    )
    return dense_flow.permute(0, 3, 1, 2).contiguous()


def model_flow_image_size(
    flow_config: Mapping[str, Any],
    default_size: tuple[int, int],
) -> tuple[int, int]:
    value = flow_config.get("model_flow_image_size")
    if value in (None, ""):
        value = flow_config.get("waft_flow_image_size")
    if value in (None, ""):
        return default_size
    return normalize_image_size(value)


def project_vertices_to_flow_screen(
    vertices: torch.Tensor,
    image_size: tuple[int, int],
    render_kwargs: Mapping[str, Any],
    fit_reference_vertices: Optional[torch.Tensor],
) -> torch.Tensor:
    from utils.render_utils import (
        CAMERA_EXTRINSICS,
        CAMERA_INTRINSICS,
        DEFAULT_VIEW_OCCUPANCY,
    )

    height, width = image_size
    base_height, base_width = CAMERA_INTRINSICS["image_size"]
    base_fx, base_fy = CAMERA_INTRINSICS["focal_length"]
    base_px, base_py = CAMERA_INTRINSICS["principal_point"]
    fx = vertices.new_tensor(base_fx * width / float(base_width))
    fy = vertices.new_tensor(base_fy * height / float(base_height))
    px = vertices.new_tensor(base_px * width / float(base_width))
    py = vertices.new_tensor(base_py * height / float(base_height))

    projected = validation_camera_project(
        validation_world_to_camera(vertices, CAMERA_EXTRINSICS),
        fx=fx,
        fy=fy,
        px=px,
        py=py,
    )
    if not bool(render_kwargs.get("fit_to_view", True)):
        return projected

    reference_vertices = vertices if fit_reference_vertices is None else fit_reference_vertices
    reference_vertices = reference_vertices.to(device=vertices.device, dtype=vertices.dtype)
    if reference_vertices.shape[0] == 1 and vertices.shape[0] > 1:
        reference_vertices = reference_vertices.expand(vertices.shape[0], -1, -1)
    if reference_vertices.shape != vertices.shape:
        raise ValueError("fit_reference_vertices must match vertices shape.")

    reference_projected = validation_camera_project(
        validation_world_to_camera(reference_vertices, CAMERA_EXTRINSICS),
        fx=fx,
        fy=fy,
        px=px,
        py=py,
    )
    occupancy = float(render_kwargs.get("view_occupancy", DEFAULT_VIEW_OCCUPANCY))
    reference_xy = reference_projected[..., :2]
    mins = reference_xy.amin(dim=1, keepdim=True)
    maxs = reference_xy.amax(dim=1, keepdim=True)
    center = (mins + maxs) * 0.5
    max_span = (maxs - mins).amax(dim=-1, keepdim=True).clamp_min(1.0e-6)
    target_span = min(height, width) * occupancy
    scale = projected.new_tensor(target_span) / max_span
    image_center = projected.new_tensor(
        ((width - 1) * 0.5, (height - 1) * 0.5)
    ).view(1, 1, 2)
    fitted_xy = (projected[..., :2] - center) * scale + image_center
    return torch.cat((fitted_xy, projected[..., 2:]), dim=-1)


def flow_barycentric_sample_pattern(
    sample_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if sample_count < 1:
        raise ValueError("model_flow_samples_per_face must be at least 1.")
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


def flow_faces_batched(
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


def sample_face_values(
    values: torch.Tensor,
    faces: torch.Tensor,
    barycentric_samples: torch.Tensor,
) -> torch.Tensor:
    batch_size, _value_count, channels = values.shape
    flat_faces = faces.reshape(batch_size, -1)
    gathered = torch.gather(
        values,
        dim=1,
        index=flat_faces.unsqueeze(-1).expand(-1, -1, channels),
    )
    face_values = gathered.reshape(batch_size, faces.shape[1], faces.shape[2], channels)
    weights = barycentric_samples.view(1, 1, -1, 3, 1)
    samples = (face_values.unsqueeze(2) * weights).sum(dim=3)
    return samples.reshape(batch_size, faces.shape[1] * barycentric_samples.shape[0], channels)


def soft_splat_sample_values(
    sample_positions: torch.Tensor,
    sample_depths: torch.Tensor,
    sample_values: torch.Tensor,
    image_size: tuple[int, int],
    background: torch.Tensor,
    splat_radius_px: float,
    splat_kernel_radius: int,
    depth_sharpness: float,
    chunk_size: int,
) -> torch.Tensor:
    height, width = image_size
    batch_size, sample_count, channels = sample_values.shape
    accum = sample_values.new_zeros(batch_size, height * width, channels)
    weights = sample_values.new_zeros(batch_size, height * width, 1)
    offsets = flow_kernel_offsets(
        splat_kernel_radius,
        device=sample_values.device,
        dtype=sample_positions.dtype,
    )

    for batch_index in range(batch_size):
        positions_b = sample_positions[batch_index]
        depths_b = sample_depths[batch_index]
        values_b = sample_values[batch_index]
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
            splat = flow_splat_indices_and_weights(
                positions=positions_b[start:end][valid],
                depths=depths_b[start:end][valid],
                offsets=offsets,
                height=height,
                width=width,
                splat_radius_px=splat_radius_px,
            )
            if splat is None:
                continue
            flat_index, _spatial_weight, depth_values, _inside = splat
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
            values = values_b[start:end][valid]
            splat = flow_splat_indices_and_weights(
                positions=positions_b[start:end][valid],
                depths=depths_b[start:end][valid],
                offsets=offsets,
                height=height,
                width=width,
                splat_radius_px=splat_radius_px,
            )
            if splat is None:
                continue
            flat_index, spatial_weight, depth_values, inside = splat
            nearest_depth = depth_buffer[flat_index]
            depth_delta = (depth_values - nearest_depth).clamp_min(0.0)
            splat_weight = spatial_weight * torch.exp(-depth_delta * depth_sharpness)
            splat_values = values[:, None, :].expand(-1, offsets.shape[0], -1)[inside]
            accum[batch_index].index_add_(
                0,
                flat_index,
                splat_values * splat_weight.unsqueeze(-1),
            )
            weights[batch_index].index_add_(
                0,
                flat_index,
                splat_weight.unsqueeze(-1),
            )

    image = accum / weights.clamp_min(1.0e-8)
    background = background.to(
        device=sample_values.device,
        dtype=sample_values.dtype,
    ).view(1, 1, channels)
    background = background.expand(batch_size, height * width, channels)
    image = torch.where(weights > 1.0e-8, image, background)
    return image.view(batch_size, height, width, channels)


def flow_kernel_offsets(
    radius: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if radius < 0:
        raise ValueError("splat_kernel_radius must be non-negative.")
    values = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def flow_splat_indices_and_weights(
    positions: torch.Tensor,
    depths: torch.Tensor,
    offsets: torch.Tensor,
    height: int,
    width: int,
    splat_radius_px: float,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
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
        -dist2 / (2.0 * max(float(splat_radius_px), 1.0e-6) ** 2)
    )[inside]
    flat_index = (
        py.long().clamp(0, height - 1) * width
        + px.long().clamp(0, width - 1)
    )[inside]
    depth_values = depths[:, None].expand(-1, offsets.shape[0])[inside]
    return flat_index, spatial_weight, depth_values, inside


def resize_flow_to_size(
    flow: torch.Tensor,
    image_size: tuple[int, int] | torch.Size,
) -> torch.Tensor:
    target_height, target_width = int(image_size[0]), int(image_size[1])
    old_height, old_width = flow.shape[-2:]
    if (old_height, old_width) == (target_height, target_width):
        return flow
    resized = F.interpolate(
        flow,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    scale = resized.new_tensor(
        (target_width / float(old_width), target_height / float(old_height))
    ).view(1, 2, 1, 1)
    return resized * scale


def masked_flow_loss(
    predicted_flow: torch.Tensor,
    target_flow: torch.Tensor,
    target_mask: torch.Tensor,
    flow_config: Mapping[str, Any],
) -> torch.Tensor:
    diff = predicted_flow - target_flow
    loss_type = str(
        flow_config.get(
            "flow_loss_type",
            flow_config.get("waft_flow_loss_type", "l2"),
        )
    ).lower()
    if loss_type in {"l2", "mse", "squared"}:
        element_loss = diff.square()
    elif loss_type in {"l1", "mae", "absolute"}:
        element_loss = diff.abs()
    elif loss_type in {"charbonnier", "robust_l1"}:
        epsilon = float(flow_config.get("waft_flow_loss_epsilon", 1.0e-3))
        element_loss = torch.sqrt(diff.square() + epsilon * epsilon) - epsilon
    elif loss_type in {"huber", "smooth_l1"}:
        delta = float(flow_config.get("waft_flow_loss_huber_delta", 1.0))
        abs_diff = diff.abs()
        quadratic = torch.minimum(abs_diff, abs_diff.new_tensor(delta))
        linear = abs_diff - quadratic
        element_loss = 0.5 * quadratic.square() / max(delta, 1.0e-8) + linear
    else:
        raise ValueError(
            "loss.image.flow.waft_flow_loss_type must be one of: "
            "l2, l1, charbonnier, huber."
        )

    normalizer = target_mask.sum().clamp_min(1.0e-6) * element_loss.shape[1]
    return (element_loss * target_mask).sum() / normalizer


def prepare_image_flow_target(
    record: Mapping[str, Any],
    image_size: tuple[int, int],
    flow_config: Mapping[str, Any],
    render_config: Optional[Mapping[str, Any]],
    neutral_render: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    backend = str(flow_config.get("backend", "local")).lower()
    if backend == "waft":
        displacement_2d = record["displacement_2d"]
        neutral_path = displacement_2d.get("neutral_image_path")
        expressed_path = displacement_2d.get("expressed_image_path")
        if neutral_path is None or expressed_path is None:
            raise ValueError(
                "WAFT image loss requires neutral_image_path and "
                "expressed_image_path in the image sample record."
            )
        action_unit_id = as_int(record["action_unit_id"])
        source_keyed_cache = waft_target_disk_cache_uses_source_alignment(flow_config)
        disk_cache_path = (
            waft_target_disk_cache_path(
                neutral_path=str(neutral_path),
                expressed_path=str(expressed_path),
                image_size=image_size,
                flow_config=flow_config,
                alignment_transform=None,
                action_unit_id=action_unit_id,
                render_config=render_config,
            )
            if source_keyed_cache
            else None
        )
        if disk_cache_path is not None:
            cached_target = load_waft_target_disk_cache(
                disk_cache_path,
                device=device,
                dtype=dtype,
            )
            if cached_target is not None:
                return cached_target

        alignment_transform = waft_target_landmark_alignment_transform(
            record=record,
            neutral_path=str(neutral_path),
            image_size=image_size,
            flow_config=flow_config,
            render_config=render_config,
            neutral_render=neutral_render,
            device=device,
            dtype=dtype,
        )
        if not source_keyed_cache:
            disk_cache_path = waft_target_disk_cache_path(
                neutral_path=str(neutral_path),
                expressed_path=str(expressed_path),
                image_size=image_size,
                flow_config=flow_config,
                alignment_transform=alignment_transform,
                action_unit_id=action_unit_id,
                render_config=render_config,
            )
            if disk_cache_path is not None:
                cached_target = load_waft_target_disk_cache(
                    disk_cache_path,
                    device=device,
                    dtype=dtype,
                )
                if cached_target is not None:
                    return cached_target
        if (
            disk_cache_path is not None
            and source_keyed_cache
        ):
            legacy_cache_path = waft_target_legacy_disk_cache_path(
                neutral_path=str(neutral_path),
                expressed_path=str(expressed_path),
                image_size=image_size,
                flow_config=flow_config,
                alignment_transform=alignment_transform,
                action_unit_id=action_unit_id,
            )
            cached_target = load_waft_target_disk_cache(
                legacy_cache_path,
                device=device,
                dtype=dtype,
            )
            if cached_target is not None:
                promote_waft_target_disk_cache(legacy_cache_path, disk_cache_path)
                return cached_target

        target_flow = cached_waft_target_flow(
            neutral_path=str(neutral_path),
            expressed_path=str(expressed_path),
            image_size=image_size,
            flow_config=flow_config,
            alignment_transform=alignment_transform,
            device=device,
            dtype=dtype,
        )
        target_mask = waft_target_loss_mask(
            neutral_path=str(neutral_path),
            expressed_path=str(expressed_path),
            image_size=tuple(int(value) for value in target_flow.shape[-2:]),
            flow=target_flow,
            flow_config=flow_config,
            alignment_transform=alignment_transform,
            action_unit_id=action_unit_id,
            device=device,
            dtype=dtype,
        )
        if disk_cache_path is not None:
            save_waft_target_disk_cache(
                disk_cache_path,
                target_flow=target_flow,
                target_mask=target_mask,
                flow_config=flow_config,
            )
        return target_flow, target_mask

    return prepare_2d_target(
        displacement=record["displacement_2d"]["displacement"],
        mask=record["displacement_2d"].get("mask"),
        image_size=image_size,
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def prepare_waft_image_flow_target_batch(
    records: Sequence[Mapping[str, Any]],
    image_size: tuple[int, int],
    flow_config: Mapping[str, Any],
    render_config: Optional[Mapping[str, Any]],
    neutral_renders: Sequence[Optional[torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    pending: list[dict[str, Any]] = []
    for record, neutral_render in zip(records, neutral_renders):
        displacement_2d = record["displacement_2d"]
        neutral_path = displacement_2d.get("neutral_image_path")
        expressed_path = displacement_2d.get("expressed_image_path")
        if neutral_path is None or expressed_path is None:
            raise ValueError(
                "WAFT image loss requires neutral_image_path and "
                "expressed_image_path in the image sample record."
            )
        action_unit_id = as_int(record["action_unit_id"])
        source_keyed_cache = waft_target_disk_cache_uses_source_alignment(flow_config)
        disk_cache_path = (
            waft_target_disk_cache_path(
                neutral_path=str(neutral_path),
                expressed_path=str(expressed_path),
                image_size=image_size,
                flow_config=flow_config,
                alignment_transform=None,
                action_unit_id=action_unit_id,
                render_config=render_config,
            )
            if source_keyed_cache
            else None
        )
        if disk_cache_path is None:
            if not waft_target_disk_cache_enabled(flow_config):
                prepare_image_flow_target(
                    record=record,
                    image_size=image_size,
                    flow_config=flow_config,
                    render_config=render_config,
                    neutral_render=neutral_render,
                    device=device,
                    dtype=dtype,
                )
                continue
        elif disk_cache_path.is_file():
            continue
        alignment_transform = waft_target_landmark_alignment_transform(
            record=record,
            neutral_path=str(neutral_path),
            image_size=image_size,
            flow_config=flow_config,
            render_config=render_config,
            neutral_render=neutral_render,
            device=device,
            dtype=dtype,
        )
        if not source_keyed_cache:
            disk_cache_path = waft_target_disk_cache_path(
                neutral_path=str(neutral_path),
                expressed_path=str(expressed_path),
                image_size=image_size,
                flow_config=flow_config,
                alignment_transform=alignment_transform,
                action_unit_id=action_unit_id,
                render_config=render_config,
            )
            if disk_cache_path is not None and disk_cache_path.is_file():
                continue
        if source_keyed_cache:
            legacy_cache_path = waft_target_legacy_disk_cache_path(
                neutral_path=str(neutral_path),
                expressed_path=str(expressed_path),
                image_size=image_size,
                flow_config=flow_config,
                alignment_transform=alignment_transform,
                action_unit_id=action_unit_id,
            )
            if legacy_cache_path.is_file():
                promote_waft_target_disk_cache(legacy_cache_path, disk_cache_path)
                continue
        pending.append(
            {
                "neutral_path": str(neutral_path),
                "expressed_path": str(expressed_path),
                "alignment_transform": alignment_transform,
                "action_unit_id": action_unit_id,
                "action_flow_config": waft_flow_config_for_action_unit(
                    flow_config,
                    action_unit_id,
                ),
                "disk_cache_path": disk_cache_path,
            }
        )

    if not pending:
        return

    neutral_images = torch.cat(
        [
            load_waft_target_rgb_image(
                item["neutral_path"],
                image_size,
                flow_config,
                item["alignment_transform"],
                device,
                dtype,
            )
            for item in pending
        ],
        dim=0,
    )
    expressed_images = torch.cat(
        [
            load_waft_target_rgb_image(
                item["expressed_path"],
                image_size,
                flow_config,
                item["alignment_transform"],
                device,
                dtype,
            )
            for item in pending
        ],
        dim=0,
    )
    target_flows = compute_waft_flow(
        neutral_images,
        expressed_images,
        flow_config=flow_config,
        device=device,
    ).to(device=device, dtype=dtype)
    flow_image_size = tuple(int(value) for value in target_flows.shape[-2:])
    mask_neutral_images = torch.cat(
        [
            load_waft_target_rgb_image(
                item["neutral_path"],
                flow_image_size,
                item["action_flow_config"],
                item["alignment_transform"],
                device,
                dtype,
            )
            for item in pending
        ],
        dim=0,
    )
    mask_expressed_images = torch.cat(
        [
            load_waft_target_rgb_image(
                item["expressed_path"],
                flow_image_size,
                item["action_flow_config"],
                item["alignment_transform"],
                device,
                dtype,
            )
            for item in pending
        ],
        dim=0,
    )

    for index, item in enumerate(pending):
        target_flow = target_flows[index : index + 1]
        target_mask = compute_waft_target_loss_mask_from_images(
            neutral=mask_neutral_images[index : index + 1],
            expressed=mask_expressed_images[index : index + 1],
            flow=target_flow,
            flow_config=item["action_flow_config"],
            dtype=dtype,
        )
        save_waft_target_disk_cache(
            item["disk_cache_path"],
            target_flow=target_flow,
            target_mask=target_mask,
            flow_config=flow_config,
        )


def waft_target_disk_cache_enabled(flow_config: Mapping[str, Any]) -> bool:
    value = flow_config.get("waft_target_disk_cache")
    if value is not None:
        return bool(value)
    return flow_config.get("waft_target_cache_dir") not in (None, "")


def waft_target_memory_cache_enabled(flow_config: Mapping[str, Any]) -> bool:
    default = not waft_target_disk_cache_enabled(flow_config)
    return bool(flow_config.get("waft_target_memory_cache", default))


def waft_target_disk_cache_dir(flow_config: Mapping[str, Any]) -> Path:
    value = flow_config.get("waft_target_cache_dir")
    cache_dir = (
        Path(str(value)).expanduser()
        if value not in (None, "")
        else ROOT / ".cache" / "waft_targets"
    )
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    return cache_dir


def waft_target_disk_cache_path(
    neutral_path: str,
    expressed_path: str,
    image_size: tuple[int, int],
    flow_config: Mapping[str, Any],
    alignment_transform: Optional[torch.Tensor],
    action_unit_id: Optional[int],
    render_config: Optional[Mapping[str, Any]] = None,
) -> Optional[Path]:
    if not waft_target_disk_cache_enabled(flow_config):
        return None
    metadata = waft_target_disk_cache_metadata(
        neutral_path=neutral_path,
        expressed_path=expressed_path,
        image_size=image_size,
        flow_config=flow_config,
        alignment_transform=alignment_transform,
        action_unit_id=action_unit_id,
        render_config=render_config,
    )
    return waft_target_disk_cache_path_from_metadata(flow_config, metadata)


def waft_target_legacy_disk_cache_path(
    neutral_path: str,
    expressed_path: str,
    image_size: tuple[int, int],
    flow_config: Mapping[str, Any],
    alignment_transform: Optional[torch.Tensor],
    action_unit_id: Optional[int],
) -> Path:
    metadata = waft_target_legacy_disk_cache_metadata(
        neutral_path=neutral_path,
        expressed_path=expressed_path,
        image_size=image_size,
        flow_config=flow_config,
        alignment_transform=alignment_transform,
        action_unit_id=action_unit_id,
    )
    return waft_target_disk_cache_path_from_metadata(flow_config, metadata)


def waft_target_disk_cache_path_from_metadata(
    flow_config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Path:
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return waft_target_disk_cache_dir(flow_config) / digest[:2] / f"{digest}.pt"


def waft_target_disk_cache_metadata(
    neutral_path: str,
    expressed_path: str,
    image_size: tuple[int, int],
    flow_config: Mapping[str, Any],
    alignment_transform: Optional[torch.Tensor],
    action_unit_id: Optional[int],
    render_config: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    if not waft_target_disk_cache_uses_source_alignment(flow_config):
        return waft_target_legacy_disk_cache_metadata(
            neutral_path=neutral_path,
            expressed_path=expressed_path,
            image_size=image_size,
            flow_config=flow_config,
            alignment_transform=alignment_transform,
            action_unit_id=action_unit_id,
        )

    waft_root, config_path, checkpoint_path, iters, _device = waft_model_key(
        flow_config,
        torch.device("cpu"),
    )
    action_flow_config = waft_flow_config_for_action_unit(flow_config, action_unit_id)
    alignment = flow_config["waft_target_image_alignment"]
    return {
        "format_version": 2,
        "neutral_image": waft_cache_file_signature(neutral_path),
        "expressed_image": waft_cache_file_signature(expressed_path),
        "image_size": tuple(int(value) for value in image_size),
        "waft_flow_image_size": waft_flow_image_size(flow_config, image_size),
        "waft_root": waft_root,
        "waft_config": waft_cache_file_signature(config_path),
        "waft_checkpoint": waft_cache_file_signature(checkpoint_path),
        "waft_iters": int(iters),
        "alignment_config": waft_functional_alignment_config(alignment),
        "alignment_target": waft_alignment_mesh_signature_metadata(
            neutral_path,
            alignment,
        ),
        "alignment_render_kwargs": waft_alignment_render_kwargs(render_config),
        "mask_config": waft_target_mask_functional_config_key(action_flow_config),
        "action_unit_id": action_unit_id,
    }


def waft_target_legacy_disk_cache_metadata(
    neutral_path: str,
    expressed_path: str,
    image_size: tuple[int, int],
    flow_config: Mapping[str, Any],
    alignment_transform: Optional[torch.Tensor],
    action_unit_id: Optional[int],
) -> dict[str, Any]:
    waft_root, config_path, checkpoint_path, iters, _device = waft_model_key(
        flow_config,
        torch.device("cpu"),
    )
    action_flow_config = waft_flow_config_for_action_unit(flow_config, action_unit_id)
    return {
        "format_version": 1,
        "neutral_path": str(Path(neutral_path).expanduser()),
        "expressed_path": str(Path(expressed_path).expanduser()),
        "image_size": tuple(int(value) for value in image_size),
        "waft_flow_image_size": waft_flow_image_size(flow_config, image_size),
        "waft_root": waft_root,
        "waft_config": waft_cache_file_signature(config_path),
        "waft_checkpoint": waft_cache_file_signature(checkpoint_path),
        "waft_iters": int(iters),
        "alignment_config": waft_target_image_alignment_key(flow_config),
        "alignment_transform": waft_alignment_transform_key(alignment_transform),
        "mask_config": waft_target_mask_config_key(action_flow_config),
        "action_unit_id": action_unit_id,
    }


def waft_target_disk_cache_uses_source_alignment(
    flow_config: Mapping[str, Any],
) -> bool:
    alignment = flow_config.get("waft_target_image_alignment")
    if not isinstance(alignment, Mapping) or not bool(alignment.get("enabled", False)):
        return False
    mode = str(alignment.get("mode", "bbox")).strip().lower()
    if mode not in {"landmark", "landmarks", "mediapipe", "mediapipe_landmarks"}:
        return False
    target_source = str(alignment.get("target_source", "render")).strip().lower()
    return target_source in WAFT_EXTERNAL_MESH_ALIGNMENT_TARGET_SOURCES


def waft_functional_alignment_config(
    alignment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in alignment.items()
        if str(key) not in WAFT_ALIGNMENT_CACHE_CONTROL_KEYS
    }


def waft_cache_file_signature(path: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    signature: dict[str, Any] = {"path": str(resolved)}
    try:
        stat = resolved.stat()
    except OSError:
        return signature
    signature["size"] = int(stat.st_size)
    signature["mtime_ns"] = int(stat.st_mtime_ns)
    return signature


def load_waft_target_disk_cache(
    cache_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    if not cache_path.is_file():
        return None
    try:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
    except Exception as exc:
        print(
            "[WARN] Failed to load WAFT target cache "
            f"{cache_path}: {type(exc).__name__}: {exc}. Recomputing.",
            flush=True,
        )
        return None

    if not isinstance(payload, Mapping):
        return None
    flow = payload.get("flow")
    mask = payload.get("mask")
    if not isinstance(flow, torch.Tensor) or not isinstance(mask, torch.Tensor):
        return None
    if flow.dim() != 4 or flow.shape[1] != 2:
        return None
    if mask.dim() != 4 or mask.shape[1] != 1:
        return None
    return flow.to(device=device, dtype=dtype), mask.to(device=device, dtype=dtype)


def save_waft_target_disk_cache(
    cache_path: Path,
    target_flow: torch.Tensor,
    target_mask: torch.Tensor,
    flow_config: Mapping[str, Any],
) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        storage_dtype = waft_target_cache_storage_dtype(flow_config)
        payload = {
            "format_version": 1,
            "flow": target_flow.detach().cpu().to(dtype=storage_dtype),
            "mask": target_mask.detach().cpu().to(dtype=storage_dtype),
        }
        tmp_path = cache_path.with_name(
            f".{cache_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        torch.save(payload, tmp_path)
        os.replace(tmp_path, cache_path)
    except Exception as exc:
        print(
            "[WARN] Failed to save WAFT target cache "
            f"{cache_path}: {type(exc).__name__}: {exc}",
            flush=True,
        )


def promote_waft_target_disk_cache(source_path: Path, target_path: Path) -> bool:
    if source_path == target_path or target_path.is_file():
        return True
    tmp_path: Optional[Path] = None
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_name(
            f".{target_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        os.link(source_path, tmp_path)
        os.replace(tmp_path, target_path)
        return True
    except FileExistsError:
        return target_path.is_file()
    except OSError as exc:
        warn_waft_alignment_once(
            "waft-target-cache-promotion",
            "Could not hard-link legacy WAFT target cache entries into the "
            f"source-keyed cache ({type(exc).__name__}: {exc}). Legacy entries "
            "will still be reused through the alignment-transform cache.",
        )
        return False
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def waft_target_cache_storage_dtype(flow_config: Mapping[str, Any]) -> torch.dtype:
    value = str(flow_config.get("waft_target_cache_dtype", "float16")).lower()
    if value in {"float16", "half", "fp16"}:
        return torch.float16
    if value in {"float32", "float", "fp32"}:
        return torch.float32
    raise ValueError("loss.image.flow.waft_target_cache_dtype must be float16 or float32.")


def cached_waft_target_flow(
    neutral_path: str,
    expressed_path: str,
    image_size: tuple[int, int],
    flow_config: Mapping[str, Any],
    alignment_transform: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    model_key = waft_model_key(flow_config, device)
    cache_key = (
        neutral_path,
        expressed_path,
        image_size,
        waft_flow_image_size(flow_config, image_size),
        model_key[1],
        model_key[2],
        model_key[3],
        waft_target_image_alignment_key(flow_config),
        waft_alignment_transform_key(alignment_transform),
    )
    use_memory_cache = waft_target_memory_cache_enabled(flow_config)
    if use_memory_cache and cache_key in WAFT_TARGET_FLOW_CACHE:
        return WAFT_TARGET_FLOW_CACHE[cache_key].to(device=device, dtype=dtype)

    neutral = load_waft_target_rgb_image(
        neutral_path,
        image_size,
        flow_config,
        alignment_transform,
        device,
        dtype,
    )
    expressed = load_waft_target_rgb_image(
        expressed_path,
        image_size,
        flow_config,
        alignment_transform,
        device,
        dtype,
    )
    with torch.no_grad():
        flow = compute_waft_flow(
            neutral,
            expressed,
            flow_config=flow_config,
            device=device,
        )
    if use_memory_cache:
        WAFT_TARGET_FLOW_CACHE[cache_key] = flow.detach().cpu()
        return WAFT_TARGET_FLOW_CACHE[cache_key].to(device=device, dtype=dtype)
    return flow.to(device=device, dtype=dtype)


def compute_waft_flow(
    image_1: torch.Tensor,
    image_2: torch.Tensor,
    flow_config: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    estimator = get_waft_flow_model(flow_config, device)
    image_1_bchw = waft_image_to_bchw_255(image_1, device)
    image_2_bchw = waft_image_to_bchw_255(image_2, device)
    flow_size = waft_flow_image_size(
        flow_config,
        tuple(int(value) for value in image_1_bchw.shape[-2:]),
    )
    if tuple(image_1_bchw.shape[-2:]) != flow_size:
        image_1_bchw = F.interpolate(
            image_1_bchw,
            size=flow_size,
            mode="bilinear",
            align_corners=False,
        )
        image_2_bchw = F.interpolate(
            image_2_bchw,
            size=flow_size,
            mode="bilinear",
            align_corners=False,
        )
    output = estimator(image_1_bchw, image_2_bchw)
    flow = output["flow"][-1]
    if flow.dim() != 4 or flow.shape[1] != 2:
        raise ValueError("WAFT flow output must have shape [B, 2, H, W].")
    return flow


def waft_target_loss_mask(
    neutral_path: str,
    expressed_path: str,
    image_size: tuple[int, int],
    flow: torch.Tensor,
    flow_config: Mapping[str, Any],
    alignment_transform: Optional[torch.Tensor],
    action_unit_id: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    flow_config = waft_flow_config_for_action_unit(flow_config, action_unit_id)
    cache_key = (
        neutral_path,
        expressed_path,
        tuple(int(value) for value in image_size),
        waft_target_mask_config_key(flow_config),
        waft_alignment_transform_key(alignment_transform),
        action_unit_id,
    )
    use_memory_cache = waft_target_memory_cache_enabled(flow_config)
    if use_memory_cache and cache_key in WAFT_TARGET_MASK_CACHE:
        return WAFT_TARGET_MASK_CACHE[cache_key].to(device=device, dtype=dtype)

    neutral = load_waft_target_rgb_image(
        neutral_path,
        image_size,
        flow_config,
        alignment_transform,
        device,
        dtype,
    )
    expressed = load_waft_target_rgb_image(
        expressed_path,
        image_size,
        flow_config,
        alignment_transform,
        device,
        dtype,
    )
    mask = compute_waft_target_loss_mask_from_images(
        neutral=neutral,
        expressed=expressed,
        flow=flow,
        flow_config=flow_config,
        dtype=dtype,
    )
    if use_memory_cache:
        WAFT_TARGET_MASK_CACHE[cache_key] = mask.detach().cpu()
    return mask


def compute_waft_target_loss_mask_from_images(
    neutral: torch.Tensor,
    expressed: torch.Tensor,
    flow: torch.Tensor,
    flow_config: Mapping[str, Any],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not bool(flow_config.get("waft_mask_foreground", True)):
        return flow.new_ones((flow.shape[0], 1, flow.shape[-2], flow.shape[-1]))

    white_threshold = float(flow_config.get("waft_background_white_threshold", 0.97))
    motion_threshold = float(flow_config.get("waft_motion_mask_threshold", 0.015))
    neutral_foreground = (neutral < white_threshold).any(dim=-1, keepdim=True)
    expressed_foreground = (expressed < white_threshold).any(dim=-1, keepdim=True)
    foreground = neutral_foreground | expressed_foreground
    foreground = foreground & waft_flow_mask_roi(
        foreground,
        flow_config.get("waft_flow_mask_roi"),
    )
    photometric_motion = (neutral - expressed).abs().amax(dim=-1, keepdim=True)
    flow_motion = torch.linalg.vector_norm(flow, dim=1, keepdim=True).permute(
        0,
        2,
        3,
        1,
    )
    motion_masks: list[torch.Tensor] = []
    if bool(flow_config.get("waft_photometric_motion_mask", True)):
        motion_masks.append(photometric_motion > motion_threshold)
    if bool(flow_config.get("waft_flow_motion_mask", True)):
        flow_threshold = waft_flow_motion_threshold_px(
            flow_motion=flow_motion,
            foreground=foreground,
            flow_config=flow_config,
        )
        motion_masks.append(flow_motion > flow_threshold)
    if bool(flow_config.get("waft_include_full_foreground_in_mask", False)):
        motion_masks.append(foreground)

    if motion_masks:
        motion_mask = motion_masks[0]
        for extra_mask in motion_masks[1:]:
            motion_mask = motion_mask | extra_mask
        mask = (foreground & motion_mask).to(dtype=dtype)
    else:
        mask = foreground.to(dtype=dtype)

    dilate_px = int(flow_config.get("waft_mask_dilate_px", 15))
    blur_sigma = float(flow_config.get("waft_mask_blur_sigma", 3.0))
    if dilate_px > 1:
        if dilate_px % 2 == 0:
            dilate_px += 1
        mask = F.max_pool2d(
            mask.permute(0, 3, 1, 2),
            kernel_size=dilate_px,
            stride=1,
            padding=dilate_px // 2,
        ).permute(0, 2, 3, 1)
    if blur_sigma > 0.0:
        mask = gaussian_blur_mask(mask, sigma=blur_sigma)
    mask = mask.permute(0, 3, 1, 2).clamp(0.0, 1.0)
    return mask


def waft_target_mask_config_key(flow_config: Mapping[str, Any]) -> str:
    target_config = {
        key: value
        for key, value in flow_config.items()
        if key not in WAFT_TARGET_CACHE_CONTROL_KEYS
    }
    return json.dumps(target_config, sort_keys=True, default=str)


def waft_target_mask_functional_config_key(
    flow_config: Mapping[str, Any],
) -> str:
    target_config = {
        key: value
        for key, value in flow_config.items()
        if key not in WAFT_TARGET_CACHE_CONTROL_KEYS
    }
    alignment = target_config.get("waft_target_image_alignment")
    if isinstance(alignment, Mapping):
        target_config["waft_target_image_alignment"] = (
            waft_functional_alignment_config(alignment)
        )
    return json.dumps(target_config, sort_keys=True, default=str)


def waft_flow_config_for_action_unit(
    flow_config: Mapping[str, Any],
    action_unit_id: Optional[int],
) -> dict[str, Any]:
    resolved = dict(flow_config)
    if action_unit_id is None:
        return resolved
    overrides = flow_config.get("waft_flow_mask_action_unit_overrides", {})
    if not isinstance(overrides, Mapping):
        return resolved
    override = None
    for key in (
        action_unit_id,
        str(action_unit_id),
        f"AU{action_unit_id}",
        f"au{action_unit_id}",
    ):
        if key in overrides:
            override = overrides[key]
            break
    if isinstance(override, Mapping):
        resolved.update(override)
    return resolved


def waft_flow_mask_roi(
    foreground: torch.Tensor,
    roi_config: Any,
) -> torch.Tensor:
    if not isinstance(roi_config, Mapping):
        return torch.ones_like(foreground, dtype=torch.bool)
    if foreground.dim() != 4 or foreground.shape[-1] != 1:
        raise ValueError("foreground must have shape [B, H, W, 1].")

    relative_to = str(roi_config.get("relative_to", "foreground_bbox")).lower()
    y_min = normalize_roi_fraction(roi_config.get("y_min", 0.0))
    y_max = normalize_roi_fraction(roi_config.get("y_max", 1.0))
    x_min = normalize_roi_fraction(roi_config.get("x_min", 0.0))
    x_max = normalize_roi_fraction(roi_config.get("x_max", 1.0))
    if y_max < y_min:
        y_min, y_max = y_max, y_min
    if x_max < x_min:
        x_min, x_max = x_max, x_min

    batch_size, height, width = foreground.shape[:3]
    yy = torch.arange(height, device=foreground.device).view(1, height, 1, 1)
    xx = torch.arange(width, device=foreground.device).view(1, 1, width, 1)
    roi = torch.zeros_like(foreground, dtype=torch.bool)

    for batch_index in range(batch_size):
        if relative_to in {"image", "full_image", "frame"}:
            y0, y1 = 0.0, float(height)
            x0, x1 = 0.0, float(width)
        elif relative_to in {"foreground", "foreground_bbox", "bbox"}:
            positions = torch.where(foreground[batch_index, :, :, 0])
            if positions[0].numel() == 0:
                continue
            y0 = float(positions[0].min())
            y1 = float(positions[0].max() + 1)
            x0 = float(positions[1].min())
            x1 = float(positions[1].max() + 1)
        else:
            raise ValueError(
                "waft_flow_mask_roi.relative_to must be one of: "
                "foreground_bbox or image."
            )

        y_low = y0 + y_min * max(y1 - y0, 1.0)
        y_high = y0 + y_max * max(y1 - y0, 1.0)
        x_low = x0 + x_min * max(x1 - x0, 1.0)
        x_high = x0 + x_max * max(x1 - x0, 1.0)
        roi[batch_index : batch_index + 1] = (
            (yy >= y_low)
            & (yy < y_high)
            & (xx >= x_low)
            & (xx < x_high)
        )

    return roi


def waft_flow_motion_threshold_px(
    flow_motion: torch.Tensor,
    foreground: torch.Tensor,
    flow_config: Mapping[str, Any],
) -> torch.Tensor:
    if flow_motion.shape != foreground.shape:
        raise ValueError("flow_motion and foreground must have matching BHWC shapes.")

    foreground_values = flow_motion[foreground]
    fallback = flow_motion.new_tensor(
        float(flow_config.get("waft_flow_mask_threshold_px", 0.0))
    )
    if foreground_values.numel() == 0:
        return fallback

    thresholds = [fallback]
    percentile = optional_float_config(
        flow_config,
        "waft_flow_mask_threshold_percentile",
    )
    if percentile is not None:
        thresholds.append(
            torch.quantile(foreground_values, normalize_quantile(percentile))
        )

    max_fraction = optional_float_config(
        flow_config,
        "waft_flow_mask_threshold_max_fraction",
    )
    if max_fraction is not None:
        reference_percentile = optional_float_config(
            flow_config,
            "waft_flow_mask_reference_percentile",
        )
        if reference_percentile is None:
            reference = foreground_values.max()
        else:
            reference = torch.quantile(
                foreground_values,
                normalize_quantile(reference_percentile),
            )
        thresholds.append(reference * float(max_fraction))

    return torch.stack(thresholds).amax()


def optional_float_config(config: Mapping[str, Any], key: str) -> Optional[float]:
    value = config.get(key)
    if value in (None, ""):
        return None
    return float(value)


def normalize_quantile(value: float) -> float:
    quantile = float(value)
    if quantile > 1.0:
        quantile = quantile / 100.0
    return min(max(quantile, 0.0), 1.0)


def normalize_roi_fraction(value: Any) -> float:
    fraction = float(value)
    if fraction > 1.0:
        fraction = fraction / 100.0
    return min(max(fraction, 0.0), 1.0)


def gaussian_blur_mask(mask_hwc: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return mask_hwc
    radius = max(int(round(float(sigma) * 3.0)), 1)
    coords = torch.arange(
        -radius,
        radius + 1,
        device=mask_hwc.device,
        dtype=mask_hwc.dtype,
    )
    kernel = torch.exp(-(coords.square()) / (2.0 * float(sigma) * float(sigma)))
    kernel = kernel / kernel.sum().clamp_min(1.0e-8)
    mask = mask_hwc.permute(0, 3, 1, 2)
    mask = F.pad(mask, (radius, radius, 0, 0), mode="replicate")
    mask = F.conv2d(mask, kernel.view(1, 1, 1, -1))
    mask = F.pad(mask, (0, 0, radius, radius), mode="replicate")
    mask = F.conv2d(mask, kernel.view(1, 1, -1, 1))
    return mask.permute(0, 2, 3, 1)


def get_waft_flow_model(
    flow_config: Mapping[str, Any],
    device: torch.device,
) -> nn.Module:
    key = waft_model_key(flow_config, device)
    if key not in WAFT_FLOW_MODELS:
        WAFT_FLOW_MODELS[key] = build_waft_flow_model(
            waft_root=Path(key[0]),
            config_path=Path(key[1]),
            checkpoint_path=Path(key[2]),
            iters=key[3],
            device=device,
        )
    return WAFT_FLOW_MODELS[key]


def waft_model_key(
    flow_config: Mapping[str, Any],
    device: torch.device,
) -> tuple[str, str, str, int, str]:
    waft_root = Path(str(flow_config.get("waft_root_dir", "third_party/WAFT")))
    config_path = Path(
        str(flow_config.get("waft_config_path", waft_root / "config/a1/tar-c-t.json"))
    )
    checkpoint_path = Path(
        str(
            flow_config.get(
                "waft_checkpoint_path",
                waft_root / "ckpts/waft_a1_adaptation.pth",
            )
        )
    )
    iters = int(flow_config.get("waft_iters", flow_config.get("iters", 3)))
    return (
        str(waft_root.expanduser().resolve()),
        str(config_path.expanduser().resolve()),
        str(checkpoint_path.expanduser().resolve()),
        iters,
        str(device),
    )


def build_waft_flow_model(
    waft_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    iters: int,
    device: torch.device,
) -> nn.Module:
    if not waft_root.is_dir():
        raise FileNotFoundError(f"WAFT root does not exist: {waft_root}")
    if not config_path.is_file():
        raise FileNotFoundError(f"WAFT config does not exist: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"WAFT checkpoint does not exist: {checkpoint_path}")

    module_prefixes = ("model", "utils", "config")
    saved_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name in module_prefixes
        or any(name.startswith(f"{prefix}.") for prefix in module_prefixes)
    }
    old_sys_path = list(sys.path)
    old_cwd = os.getcwd()
    for name in saved_modules:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(waft_root))
    os.chdir(waft_root)
    try:
        from config.parser import json_to_args
        from model import fetch_model
        from utils.utils import load_ckpt

        args = json_to_args(str(config_path))
        args.iters = int(iters)
        waft_model = fetch_model(args).to(device).eval()
        load_ckpt(waft_model, str(checkpoint_path))
    finally:
        os.chdir(old_cwd)
        for name in list(sys.modules):
            if name in module_prefixes or any(
                name.startswith(f"{prefix}.") for prefix in module_prefixes
            ):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path[:] = old_sys_path

    for parameter in waft_model.parameters():
        parameter.requires_grad_(False)
    print(
        "[INFO] Loaded WAFT flow model "
        f"config={config_path} checkpoint={checkpoint_path} iters={iters}"
    )
    return waft_model


def waft_image_to_bchw_255(
    image: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if image.dim() == 3:
        if image.shape[-1] == 3:
            image = image.permute(2, 0, 1).unsqueeze(0)
        elif image.shape[0] == 3:
            image = image.unsqueeze(0)
        else:
            raise ValueError("WAFT image must have 3 RGB channels.")
    elif image.dim() == 4:
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        elif image.shape[1] != 3:
            raise ValueError("WAFT image must have 3 RGB channels.")
    else:
        raise ValueError("WAFT image must be [H, W, 3], [3, H, W], or batched.")
    image = image.to(device=device, dtype=torch.float32)
    if float(image.detach().amax().cpu()) <= 2.0:
        image = image * 255.0
    return image.contiguous()


def waft_flow_image_size(
    flow_config: Mapping[str, Any],
    default_size: tuple[int, int],
) -> tuple[int, int]:
    value = flow_config.get("waft_flow_image_size")
    if value in (None, ""):
        return default_size
    return normalize_image_size(value)


def load_loss_rgb_image(
    path: str,
    image_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        tensor = torch.tensor(
            list(image.getdata()),
            dtype=torch.float32,
        ).view(image.height, image.width, 3)
    tensor = tensor / 255.0
    if tuple(tensor.shape[:2]) != tuple(image_size):
        tensor = (
            F.interpolate(
                tensor.permute(2, 0, 1).unsqueeze(0),
                size=image_size,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )
    return tensor.to(device=device, dtype=dtype).unsqueeze(0)


WAFT_DEFAULT_ALIGNMENT_LANDMARK_IDS = (
    10,
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
    199,
)


def waft_target_landmark_alignment_transform(
    record: Mapping[str, Any],
    neutral_path: str,
    image_size: tuple[int, int],
    flow_config: Mapping[str, Any],
    render_config: Optional[Mapping[str, Any]],
    neutral_render: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    alignment = flow_config.get("waft_target_image_alignment")
    if not isinstance(alignment, Mapping) or not bool(alignment.get("enabled", False)):
        return None
    mode = str(alignment.get("mode", "bbox")).strip().lower()
    if mode not in {"landmark", "landmarks", "mediapipe", "mediapipe_landmarks"}:
        return None

    detection_size = waft_alignment_detection_image_size(alignment, image_size)
    target_source = str(alignment.get("target_source", "render")).strip().lower()
    cache_key = (
        str(neutral_path),
        tuple(int(value) for value in image_size),
        tuple(int(value) for value in detection_size),
        waft_functional_alignment_key(alignment),
        target_source,
        waft_alignment_target_cache_key(record, neutral_path, alignment, target_source),
    )
    if cache_key in WAFT_ALIGNMENT_TRANSFORM_CACHE:
        cached = WAFT_ALIGNMENT_TRANSFORM_CACHE[cache_key]
        return None if cached is None else cached.to(device=device, dtype=dtype)

    disk_cache_path = waft_alignment_transform_disk_cache_path(
        neutral_path=neutral_path,
        image_size=image_size,
        detection_size=detection_size,
        flow_config=flow_config,
        alignment=alignment,
        render_config=render_config,
        target_source=target_source,
    )
    if disk_cache_path is not None:
        found, cached = load_waft_alignment_transform_disk_cache(disk_cache_path)
        if found:
            WAFT_ALIGNMENT_TRANSFORM_CACHE[cache_key] = cached
            return None if cached is None else cached.to(device=device, dtype=dtype)

    transform = compute_waft_target_landmark_alignment_transform(
        record=record,
        neutral_path=neutral_path,
        image_size=image_size,
        detection_size=detection_size,
        flow_config=flow_config,
        alignment=alignment,
        render_config=render_config,
        neutral_render=neutral_render,
        target_source=target_source,
        device=device,
        dtype=dtype,
    )
    WAFT_ALIGNMENT_TRANSFORM_CACHE[cache_key] = (
        None if transform is None else transform.detach().cpu()
    )
    if disk_cache_path is not None:
        save_waft_alignment_transform_disk_cache(disk_cache_path, transform)
    return transform


def compute_waft_target_landmark_alignment_transform(
    record: Mapping[str, Any],
    neutral_path: str,
    image_size: tuple[int, int],
    detection_size: tuple[int, int],
    flow_config: Mapping[str, Any],
    alignment: Mapping[str, Any],
    render_config: Optional[Mapping[str, Any]],
    neutral_render: Optional[torch.Tensor],
    target_source: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    source_landmarks = cached_image_mediapipe_landmarks(
        path=neutral_path,
        image_size=detection_size,
        flow_config=flow_config,
        alignment=alignment,
        device=device,
        dtype=dtype,
    )
    if not source_landmarks:
        warn_waft_alignment_once(
            f"missing-image-landmarks:{neutral_path}",
            f"MediaPipe did not find PNG landmarks in {neutral_path}. "
            "Falling back to bbox alignment.",
        )
        return None

    if target_source in WAFT_RENDER_ALIGNMENT_TARGET_SOURCES:
        target_by_id = waft_render_landmarks_by_id(
            neutral_render=neutral_render,
            detection_size=detection_size,
            alignment=alignment,
        )
    elif target_source in WAFT_PROJECTED_MESH_ALIGNMENT_TARGET_SOURCES:
        target_by_id = waft_projected_mesh_landmarks_by_id(
            record=record,
            detection_size=detection_size,
            render_config=render_config,
            device=device,
            dtype=dtype,
        )
    elif target_source in WAFT_EXTERNAL_MESH_ALIGNMENT_TARGET_SOURCES:
        target_by_id = waft_alignment_mesh_landmarks_by_id(
            neutral_path=neutral_path,
            alignment=alignment,
            detection_size=detection_size,
            render_config=render_config,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(
            "waft_target_image_alignment.target_source must be 'render', 'mesh', "
            "or 'alignment_mesh'."
        )
    if not target_by_id:
        return None

    source_points: list[tuple[float, float]] = []
    target_points: list[torch.Tensor] = []
    requested_ids = waft_alignment_landmark_ids(alignment)
    for mediapipe_id in requested_ids:
        source_point = source_landmarks.get(mediapipe_id)
        target_point = target_by_id.get(mediapipe_id)
        if source_point is None or target_point is None:
            continue
        source_points.append(source_point)
        target_points.append(target_point)

    min_landmarks = int(alignment.get("min_landmarks", 8))
    if len(source_points) < min_landmarks:
        warn_waft_alignment_once(
            f"few-landmarks:{neutral_path}:{target_source}",
            "WAFT landmark alignment found "
            f"{len(source_points)} shared landmarks, but min_landmarks is "
            f"{min_landmarks}. Falling back to bbox alignment.",
        )
        return None

    source_tensor = torch.tensor(source_points, device=device, dtype=dtype)
    target_tensor = torch.stack(target_points, dim=0).to(device=device, dtype=dtype)
    detection_transform = estimate_waft_similarity_transform(
        source_points=source_tensor,
        target_points=target_tensor,
        allow_rotation=bool(alignment.get("allow_rotation", False)),
    )
    return rescale_waft_alignment_transform(
        detection_transform,
        from_size=detection_size,
        to_size=image_size,
    )


def waft_render_landmarks_by_id(
    neutral_render: Optional[torch.Tensor],
    detection_size: tuple[int, int],
    alignment: Mapping[str, Any],
) -> dict[int, torch.Tensor]:
    if neutral_render is None:
        warn_waft_alignment_once(
            "missing-neutral-render",
            "WAFT render landmark alignment requested, but no neutral render "
            "was provided. Falling back to bbox alignment.",
        )
        return {}
    render = neutral_render.detach()
    if render.dim() == 4 and render.shape[-1] == 3:
        render_hwc = render[0]
    elif render.dim() == 3 and render.shape[-1] == 3:
        render_hwc = render
    else:
        raise ValueError("neutral_render must have shape [B, H, W, 3] or [H, W, 3].")
    if tuple(render_hwc.shape[:2]) != detection_size:
        render_hwc = (
            F.interpolate(
                render_hwc.permute(2, 0, 1).unsqueeze(0),
                size=detection_size,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )
    target_landmarks = detect_image_mediapipe_landmarks(
        render_hwc,
        refine_landmarks=bool(alignment.get("refine_landmarks", False)),
        min_detection_confidence=float(
            alignment.get("target_min_detection_confidence", alignment.get("min_detection_confidence", 0.5))
        ),
    )
    if not target_landmarks:
        warn_waft_alignment_once(
            "missing-render-landmarks",
            "MediaPipe did not find landmarks on the neutral render. Falling "
            "back to bbox alignment.",
        )
        return {}
    device = render_hwc.device
    dtype = render_hwc.dtype
    return {
        mediapipe_id: torch.tensor(point, device=device, dtype=dtype)
        for mediapipe_id, point in target_landmarks.items()
    }


def waft_projected_mesh_landmarks_by_id(
    record: Mapping[str, Any],
    detection_size: tuple[int, int],
    render_config: Optional[Mapping[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    mesh = record.get("mesh")
    landmarks_3d = record.get("landmarks_3d")
    if not isinstance(mesh, Mapping) or not isinstance(landmarks_3d, Mapping):
        warn_waft_alignment_once(
            "missing-record-landmarks",
            "WAFT mesh landmark alignment requested, but this image sample has "
            "no mesh landmarks. Falling back to bbox alignment.",
        )
        return {}

    vertices = mesh.get("vertices")
    vertex_ids = landmarks_3d.get("vertex_ids")
    mediapipe_ids = landmarks_3d.get("mediapipe_ids")
    if (
        not isinstance(vertices, torch.Tensor)
        or not isinstance(vertex_ids, torch.Tensor)
        or not isinstance(mediapipe_ids, torch.Tensor)
        or vertex_ids.numel() == 0
        or mediapipe_ids.numel() == 0
    ):
        warn_waft_alignment_once(
            "empty-mesh-landmarks",
            "WAFT mesh landmark alignment requested, but the mesh landmark "
            "mapping is empty. Falling back to bbox alignment.",
        )
        return {}

    vertices = vertices.to(device=device, dtype=dtype)
    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0)
    elif vertices.dim() != 3:
        raise ValueError("Image mesh vertices must have shape [V, 3] or [B, V, 3].")

    vertices = vertices_to_render_axes(vertices, infer_mesh_up_axis(vertices))
    render_kwargs = waft_alignment_render_kwargs(render_config)
    projected_points = validation_project_landmarks_to_screen(
        vertices=vertices,
        vertex_ids=vertex_ids.to(device=device, dtype=torch.long),
        image_size=detection_size,
        render_kwargs=render_kwargs,
        fit_reference_vertices=vertices,
    )
    projected_points = projected_points.to(device=device, dtype=dtype)
    mediapipe_ids = mediapipe_ids.detach().cpu().long().flatten()
    return {
        int(mediapipe_id): projected_points[index]
        for index, mediapipe_id in enumerate(mediapipe_ids.tolist())
        if index < projected_points.shape[0]
    }


def waft_alignment_target_cache_key(
    record: Mapping[str, Any],
    neutral_path: str,
    alignment: Mapping[str, Any],
    target_source: str,
) -> str:
    if target_source in WAFT_PROJECTED_MESH_ALIGNMENT_TARGET_SOURCES:
        return waft_mesh_landmark_cache_key(record)
    if target_source in WAFT_EXTERNAL_MESH_ALIGNMENT_TARGET_SOURCES:
        return waft_alignment_mesh_cache_key(neutral_path, alignment)
    return "render"


def waft_alignment_transform_disk_cache_path(
    neutral_path: str,
    image_size: tuple[int, int],
    detection_size: tuple[int, int],
    flow_config: Mapping[str, Any],
    alignment: Mapping[str, Any],
    render_config: Optional[Mapping[str, Any]],
    target_source: str,
) -> Optional[Path]:
    if target_source not in WAFT_EXTERNAL_MESH_ALIGNMENT_TARGET_SOURCES:
        return None
    enabled = alignment.get("alignment_transform_disk_cache")
    if enabled is None:
        enabled = waft_target_disk_cache_enabled(flow_config)
    if not bool(enabled):
        return None

    configured_dir = alignment.get("alignment_transform_cache_dir")
    if configured_dir not in (None, ""):
        cache_dir = Path(str(configured_dir)).expanduser()
        if not cache_dir.is_absolute():
            cache_dir = ROOT / cache_dir
    else:
        cache_dir = waft_target_disk_cache_dir(flow_config).parent / "alignment_transforms"
    metadata = {
        "format_version": 1,
        "neutral_image": waft_cache_file_signature(neutral_path),
        "image_size": tuple(int(value) for value in image_size),
        "detection_size": tuple(int(value) for value in detection_size),
        "target_source": target_source,
        "alignment_config": waft_functional_alignment_config(alignment),
        "alignment_target": waft_alignment_mesh_signature_metadata(
            neutral_path,
            alignment,
        ),
        "alignment_render_kwargs": waft_alignment_render_kwargs(render_config),
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return cache_dir / digest[:2] / f"{digest}.pt"


def load_waft_alignment_transform_disk_cache(
    cache_path: Path,
) -> tuple[bool, Optional[torch.Tensor]]:
    if not cache_path.is_file():
        return False, None
    try:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
    except Exception as exc:
        print(
            "[WARN] Failed to load WAFT alignment-transform cache "
            f"{cache_path}: {type(exc).__name__}: {exc}. Recomputing.",
            flush=True,
        )
        return False, None
    if not isinstance(payload, Mapping) or "transform" not in payload:
        return False, None
    transform = payload["transform"]
    if transform is None:
        return True, None
    if not isinstance(transform, torch.Tensor) or transform.shape != (2, 3):
        return False, None
    return True, transform.detach().cpu().to(dtype=torch.float32)


def save_waft_alignment_transform_disk_cache(
    cache_path: Path,
    transform: Optional[torch.Tensor],
) -> None:
    tmp_path: Optional[Path] = None
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "transform": (
                None
                if transform is None
                else transform.detach().cpu().to(dtype=torch.float32)
            ),
        }
        tmp_path = cache_path.with_name(
            f".{cache_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        torch.save(payload, tmp_path)
        os.replace(tmp_path, cache_path)
    except Exception as exc:
        print(
            "[WARN] Failed to save WAFT alignment-transform cache "
            f"{cache_path}: {type(exc).__name__}: {exc}",
            flush=True,
        )
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def waft_alignment_mesh_landmarks_by_id(
    neutral_path: str,
    alignment: Mapping[str, Any],
    detection_size: tuple[int, int],
    render_config: Optional[Mapping[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    loaded = load_waft_alignment_mesh_and_landmarks(neutral_path, alignment)
    if loaded is None:
        return {}
    mesh, landmarks_3d = loaded

    vertices = mesh.get("vertices")
    vertex_ids = landmarks_3d.get("vertex_ids")
    mediapipe_ids = landmarks_3d.get("mediapipe_ids")
    if (
        not isinstance(vertices, torch.Tensor)
        or not isinstance(vertex_ids, torch.Tensor)
        or not isinstance(mediapipe_ids, torch.Tensor)
        or vertex_ids.numel() == 0
        or mediapipe_ids.numel() == 0
    ):
        warn_waft_alignment_once(
            f"empty-alignment-mesh-landmarks:{neutral_path}",
            "WAFT alignment mesh target was requested, but its landmark mapping "
            "is empty. Falling back to bbox alignment.",
        )
        return {}

    vertices = vertices.to(device=device, dtype=dtype)
    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0)
    elif vertices.dim() != 3:
        raise ValueError("Alignment mesh vertices must have shape [V, 3] or [B, V, 3].")

    render_kwargs = waft_alignment_render_kwargs(render_config)
    projected_points = validation_project_landmarks_to_screen(
        vertices=vertices,
        vertex_ids=vertex_ids.to(device=device, dtype=torch.long),
        image_size=detection_size,
        render_kwargs=render_kwargs,
        fit_reference_vertices=vertices,
    )
    projected_points = projected_points.to(device=device, dtype=dtype)
    mediapipe_ids = mediapipe_ids.detach().cpu().long().flatten()
    return {
        int(mediapipe_id): projected_points[index]
        for index, mediapipe_id in enumerate(mediapipe_ids.tolist())
        if index < projected_points.shape[0]
    }


def load_waft_alignment_mesh_and_landmarks(
    neutral_path: str,
    alignment: Mapping[str, Any],
) -> Optional[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
    mesh_path, landmark_path = waft_alignment_mesh_paths(neutral_path, alignment)
    up_axis, front_axis, normalize_on_get, normalized_extent = (
        waft_alignment_mesh_transform_settings(alignment)
    )
    cache_key = (
        str(mesh_path.resolve()),
        str(landmark_path.resolve()),
        up_axis,
        front_axis,
        normalize_on_get,
        normalized_extent,
    )
    cache_size = waft_alignment_mesh_cache_size(alignment)
    if cache_size > 0:
        cached = WAFT_ALIGNMENT_MESH_CACHE.get(cache_key)
        if cached is not None:
            WAFT_ALIGNMENT_MESH_CACHE.move_to_end(cache_key)
            return cached

    if not mesh_path.is_file():
        warn_waft_alignment_once(
            f"missing-alignment-mesh:{mesh_path}",
            f"WAFT alignment mesh target was requested, but {mesh_path} does "
            "not exist. Falling back to bbox alignment.",
        )
        return None
    if not landmark_path.is_file():
        warn_waft_alignment_once(
            f"missing-alignment-landmarks:{landmark_path}",
            f"WAFT alignment mesh target was requested, but {landmark_path} "
            "does not exist. Falling back to bbox alignment.",
        )
        return None

    mesh = _load_neutral_mesh_for_model(
        mesh_path,
        mesh_up_axis=up_axis,
        mesh_front_axis=front_axis,
        normalize_on_get=normalize_on_get,
        normalized_extent=normalized_extent,
    )
    landmarks_3d = _load_landmark_payload(landmark_path, mesh["vertices"])
    if cache_size > 0:
        WAFT_ALIGNMENT_MESH_CACHE[cache_key] = (mesh, landmarks_3d)
        WAFT_ALIGNMENT_MESH_CACHE.move_to_end(cache_key)
        while len(WAFT_ALIGNMENT_MESH_CACHE) > cache_size:
            WAFT_ALIGNMENT_MESH_CACHE.popitem(last=False)
    return mesh, landmarks_3d


def waft_alignment_mesh_cache_size(alignment: Mapping[str, Any]) -> int:
    raw_value = alignment.get("alignment_mesh_cache_size", 16)
    if raw_value in (None, False):
        return 0
    return max(0, int(raw_value))


def release_waft_alignment_mesh_cache() -> None:
    if not WAFT_ALIGNMENT_MESH_CACHE:
        return
    WAFT_ALIGNMENT_MESH_CACHE.clear()
    import gc

    gc.collect()
    try:
        import ctypes

        malloc_trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except (AttributeError, OSError):
        pass


def waft_alignment_mesh_cache_key(
    neutral_path: str,
    alignment: Mapping[str, Any],
) -> str:
    payload = waft_alignment_mesh_signature_metadata(neutral_path, alignment)
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def waft_alignment_mesh_signature_metadata(
    neutral_path: str,
    alignment: Mapping[str, Any],
) -> dict[str, Any]:
    mesh_path, landmark_path = waft_alignment_mesh_paths(neutral_path, alignment)
    up_axis, front_axis, normalize_on_get, normalized_extent = (
        waft_alignment_mesh_transform_settings(alignment)
    )
    return {
        "mesh": waft_cache_file_signature(str(mesh_path)),
        "landmarks": waft_cache_file_signature(str(landmark_path)),
        "mesh_up_axis": up_axis,
        "mesh_front_axis": front_axis,
        "normalize_on_get": normalize_on_get,
        "normalized_extent": normalized_extent,
    }


def waft_alignment_mesh_paths(
    neutral_path: str,
    alignment: Mapping[str, Any],
) -> tuple[Path, Path]:
    if alignment.get("use_dataset_mesh", False):
        neutral = Path(neutral_path)
        if neutral.parent.name != "neutral_imgs":
            raise ValueError("use_dataset_mesh requires the canonical neutral_imgs layout")
        mesh = neutral.parent.parent / "meshes_custom" / f"{canonical_identity(neutral.name)}.fbx"
        return mesh, mesh.with_suffix(".json")
    mesh_path = waft_alignment_template_path(
        neutral_path=neutral_path,
        root=alignment.get("alignment_mesh_dir"),
        template=alignment.get("alignment_mesh_filename_template"),
        exact_path=alignment.get("alignment_mesh_path"),
        required_label="alignment_mesh",
    )
    using_fallback = False
    if (
        not mesh_path.is_file()
        and alignment.get("alignment_mesh_path") in (None, "")
        and alignment.get("alignment_mesh_filename_fallback_template")
        not in (None, "")
    ):
        fallback_mesh_path = waft_alignment_template_path(
            neutral_path=neutral_path,
            root=alignment.get("alignment_mesh_dir"),
            template=alignment.get("alignment_mesh_filename_fallback_template"),
            exact_path=None,
            required_label="alignment_mesh",
        )
        if fallback_mesh_path.is_file():
            mesh_path = fallback_mesh_path
            using_fallback = True

    landmark_template = alignment.get("alignment_landmark_filename_template")
    if using_fallback and alignment.get(
        "alignment_landmark_filename_fallback_template"
    ) not in (None, ""):
        landmark_template = alignment.get(
            "alignment_landmark_filename_fallback_template"
        )
    landmark_path = waft_alignment_template_path(
        neutral_path=neutral_path,
        root=alignment.get("alignment_landmark_dir", alignment.get("alignment_mesh_dir")),
        template=landmark_template,
        exact_path=alignment.get("alignment_landmark_path"),
        required_label="alignment_landmark",
        default_path=mesh_path.with_suffix(".json"),
    )
    return mesh_path, landmark_path


def waft_alignment_template_path(
    neutral_path: str,
    root: Any,
    template: Any,
    exact_path: Any,
    required_label: str,
    default_path: Optional[Path] = None,
) -> Path:
    if exact_path not in (None, ""):
        path = Path(str(exact_path)).expanduser()
        if not path.is_absolute() and root not in (None, ""):
            path = Path(str(root)).expanduser() / path
        return path
    if template in (None, ""):
        if default_path is not None:
            return default_path
        raise ValueError(
            "waft_target_image_alignment must set "
            f"{required_label}_dir and {required_label}_filename_template, "
            f"or {required_label}_path."
        )
    if root in (None, ""):
        raise ValueError(
            "waft_target_image_alignment must set "
            f"{required_label}_dir when {required_label}_filename_template is used."
        )
    formatted = waft_alignment_format_template(str(template), neutral_path)
    return Path(str(root)).expanduser() / formatted


def waft_alignment_format_template(template: str, neutral_path: str) -> str:
    neutral = Path(str(neutral_path))
    values = {
        "neutral_name": neutral.name,
        "neutral_stem": neutral.stem,
        "identity": canonical_identity(neutral.name) if "identity" in template else "",
        "person_id": waft_alignment_person_id(neutral)
        if "person_id" in template
        else "",
    }
    try:
        return template.format_map(values)
    except KeyError as exc:
        raise ValueError(
            "Unknown waft_target_image_alignment path template field "
            f"{exc.args[0]!r}."
        ) from exc


def waft_alignment_person_id(neutral_path: Path) -> str:
    parts = neutral_path.stem.split("_")
    for index, part in enumerate(parts[:-1]):
        if part == "img" and parts[index + 1].isdigit():
            return parts[index + 1]
    for part in parts:
        if part.isdigit():
            return part
    raise ValueError(
        "Could not infer {person_id} for waft_target_image_alignment from "
        f"{neutral_path.name!r}."
    )


def waft_alignment_mesh_transform_settings(
    alignment: Mapping[str, Any],
) -> tuple[str, str, bool, float]:
    up_axis = str(alignment.get("alignment_mesh_up_axis", "z")).strip().lower()
    front_axis = str(alignment.get("alignment_mesh_front_axis", "y")).strip().lower()
    normalize_on_get = bool(alignment.get("alignment_mesh_normalize_on_get", True))
    normalized_extent = float(alignment.get("alignment_mesh_normalized_extent", 2.0))
    return up_axis, front_axis, normalize_on_get, normalized_extent


def waft_mesh_landmark_cache_key(record: Mapping[str, Any]) -> str:
    landmarks_3d = record.get("landmarks_3d")
    if not isinstance(landmarks_3d, Mapping):
        return "none"
    mediapipe_ids = landmarks_3d.get("mediapipe_ids")
    vertex_ids = landmarks_3d.get("vertex_ids")
    if not isinstance(mediapipe_ids, torch.Tensor) or not isinstance(
        vertex_ids,
        torch.Tensor,
    ):
        return "none"
    if mediapipe_ids.numel() == 0 or vertex_ids.numel() == 0:
        return "empty"
    pairs = torch.stack(
        (
            mediapipe_ids.detach().cpu().long().flatten(),
            vertex_ids.detach().cpu().long().flatten(),
        ),
        dim=1,
    )
    digest = hashlib.sha1(pairs.numpy().tobytes()).hexdigest()
    return digest[:12]


def waft_alignment_detection_image_size(
    alignment: Mapping[str, Any],
    image_size: tuple[int, int],
) -> tuple[int, int]:
    value = alignment.get("landmark_detection_image_size")
    if value is None:
        value = alignment.get("detection_image_size")
    if value is None:
        return tuple(int(item) for item in image_size)
    return normalize_image_size(value)


def rescale_waft_alignment_transform(
    transform: torch.Tensor,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> torch.Tensor:
    if tuple(from_size) == tuple(to_size):
        return transform
    from_height, from_width = from_size
    to_height, to_width = to_size
    to_from_detect = transform.new_tensor(
        [
            [
                float(max(to_width - 1, 1)) / float(max(from_width - 1, 1)),
                0.0,
            ],
            [
                0.0,
                float(max(to_height - 1, 1)) / float(max(from_height - 1, 1)),
            ],
        ]
    )
    detect_from_to = transform.new_tensor(
        [
            [
                float(max(from_width - 1, 1)) / float(max(to_width - 1, 1)),
                0.0,
            ],
            [
                0.0,
                float(max(from_height - 1, 1)) / float(max(to_height - 1, 1)),
            ],
        ]
    )
    scaled = transform.new_zeros((2, 3))
    scaled[:, :2] = to_from_detect @ transform[:, :2] @ detect_from_to
    scaled[:, 2] = to_from_detect @ transform[:, 2]
    return scaled


def waft_alignment_render_kwargs(
    render_config: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(render_config, Mapping):
        return {}
    render_kwargs = dict(render_config.get("kwargs", {}))
    shared_render_framing = bool(render_config.get("shared_fit_to_view", True))
    if shared_render_framing and bool(
        render_kwargs.get("fit_visible_component", False)
    ):
        render_kwargs["fit_visible_component"] = False
    return render_kwargs


def waft_alignment_landmark_ids(alignment: Mapping[str, Any]) -> tuple[int, ...]:
    value = alignment.get("landmark_ids")
    if value is None:
        return WAFT_DEFAULT_ALIGNMENT_LANDMARK_IDS
    if isinstance(value, str):
        if value.strip().lower() == "default":
            return WAFT_DEFAULT_ALIGNMENT_LANDMARK_IDS
        raw_values: Sequence[Any] = [part.strip() for part in value.split(",")]
    elif isinstance(value, Sequence):
        raw_values = value
    else:
        raise ValueError("waft_target_image_alignment.landmark_ids must be a list.")
    landmark_ids = tuple(int(item) for item in raw_values if str(item).strip())
    if not landmark_ids:
        raise ValueError("waft_target_image_alignment.landmark_ids must not be empty.")
    return landmark_ids


def cached_image_mediapipe_landmarks(
    path: str,
    image_size: tuple[int, int],
    flow_config: Mapping[str, Any],
    alignment: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, tuple[float, float]]:
    refine_landmarks = bool(alignment.get("refine_landmarks", False))
    min_detection_confidence = float(
        alignment.get("min_detection_confidence", 0.5)
    )
    cache_key = (
        str(path),
        tuple(int(value) for value in image_size),
        waft_target_image_alignment_key(flow_config),
        refine_landmarks,
        min_detection_confidence,
    )
    if cache_key not in WAFT_IMAGE_LANDMARK_CACHE:
        disk_cache_path = image_mediapipe_landmark_disk_cache_path(
            path=path,
            image_size=image_size,
            alignment=alignment,
        )
        landmarks = (
            load_image_mediapipe_landmark_disk_cache(disk_cache_path)
            if disk_cache_path is not None
            else None
        )
        if landmarks is None:
            image = load_loss_rgb_image(path, image_size, device, dtype)
            landmarks = detect_image_mediapipe_landmarks(
                image[0],
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
            )
            if disk_cache_path is not None:
                save_image_mediapipe_landmark_disk_cache(
                    disk_cache_path,
                    landmarks,
                )
        WAFT_IMAGE_LANDMARK_CACHE[cache_key] = landmarks
    return WAFT_IMAGE_LANDMARK_CACHE[cache_key]


def image_mediapipe_landmark_disk_cache_enabled(
    alignment: Mapping[str, Any],
) -> bool:
    value = alignment.get("mediapipe_landmark_disk_cache")
    if value is not None:
        return bool(value)
    return alignment.get("mediapipe_landmark_cache_dir") not in (None, "")


def image_mediapipe_landmark_disk_cache_dir(
    alignment: Mapping[str, Any],
) -> Path:
    value = alignment.get("mediapipe_landmark_cache_dir")
    cache_dir = (
        Path(str(value)).expanduser()
        if value not in (None, "")
        else ROOT / ".cache" / "image_mediapipe_landmarks"
    )
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    return cache_dir


def image_mediapipe_landmark_disk_cache_path(
    *,
    path: str,
    image_size: tuple[int, int],
    alignment: Mapping[str, Any],
) -> Optional[Path]:
    if not image_mediapipe_landmark_disk_cache_enabled(alignment):
        return None
    metadata = {
        "format_version": IMAGE_MEDIAPIPE_LANDMARK_CACHE_FORMAT_VERSION,
        "image": waft_cache_file_signature(path),
        "image_size": tuple(int(value) for value in image_size),
        "refine_landmarks": bool(alignment.get("refine_landmarks", False)),
        "min_detection_confidence": float(
            alignment.get("min_detection_confidence", 0.5)
        ),
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return image_mediapipe_landmark_disk_cache_dir(alignment) / digest[:2] / f"{digest}.pt"


def load_image_mediapipe_landmark_disk_cache(
    path: Path,
) -> Optional[dict[int, tuple[float, float]]]:
    if not path.is_file():
        return None
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            return None
        if int(payload.get("format_version", -1)) != (
            IMAGE_MEDIAPIPE_LANDMARK_CACHE_FORMAT_VERSION
        ):
            return None
        ids = torch.as_tensor(payload.get("ids"), dtype=torch.long).flatten()
        points = torch.as_tensor(payload.get("points"), dtype=torch.float64)
        if points.numel() == 0:
            points = points.reshape(0, 2)
        if points.dim() != 2 or points.shape != (ids.numel(), 2):
            return None
        if not bool(torch.isfinite(points).all()):
            return None
        if ids.numel() != torch.unique(ids).numel():
            return None
        return {
            int(landmark_id): (float(point[0]), float(point[1]))
            for landmark_id, point in zip(ids.tolist(), points.tolist())
        }
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        EOFError,
        pickle.UnpicklingError,
    ):
        return None


def save_image_mediapipe_landmark_disk_cache(
    path: Path,
    landmarks: Mapping[int, tuple[float, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        (int(landmark_id), (float(point[0]), float(point[1])))
        for landmark_id, point in landmarks.items()
    )
    payload = {
        "format_version": IMAGE_MEDIAPIPE_LANDMARK_CACHE_FORMAT_VERSION,
        "ids": torch.tensor([item[0] for item in ordered], dtype=torch.long),
        "points": torch.tensor([item[1] for item in ordered], dtype=torch.float64).reshape(
            -1, 2
        ),
    }
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{random.randrange(1 << 30):08x}.tmp"
    )
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def detect_image_mediapipe_landmarks(
    image: torch.Tensor,
    refine_landmarks: bool,
    min_detection_confidence: float,
) -> dict[int, tuple[float, float]]:
    if image.dim() != 3 or image.shape[-1] != 3:
        raise ValueError("MediaPipe image landmark detection expects HWC RGB input.")
    try:
        import mediapipe as mp
    except ImportError:
        warn_waft_alignment_once(
            "mediapipe-import",
            "mediapipe is not installed; WAFT landmark alignment is disabled.",
        )
        return {}

    height, width = image.shape[:2]
    image_uint8 = (
        image.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(device="cpu", dtype=torch.uint8)
        .numpy()
    )
    image_uint8 = np.ascontiguousarray(image_uint8)
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=bool(refine_landmarks),
        min_detection_confidence=float(min_detection_confidence),
    ) as face_mesh:
        results = face_mesh.process(image_uint8)

    if not results.multi_face_landmarks:
        return {}
    landmarks = results.multi_face_landmarks[0].landmark
    return {
        index: (
            float(landmark.x) * float(max(width - 1, 1)),
            float(landmark.y) * float(max(height - 1, 1)),
        )
        for index, landmark in enumerate(landmarks)
    }


def estimate_waft_similarity_transform(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    allow_rotation: bool,
) -> torch.Tensor:
    if source_points.shape != target_points.shape or source_points.shape[-1] != 2:
        raise ValueError("source_points and target_points must both have shape [N, 2].")
    if source_points.shape[0] < 2:
        raise ValueError("At least two landmarks are required for 2D alignment.")

    source_mean = source_points.mean(dim=0)
    target_mean = target_points.mean(dim=0)
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean
    source_energy = source_centered.square().sum().clamp_min(1.0e-8)

    if allow_rotation:
        covariance = source_centered.T @ target_centered / float(source_points.shape[0])
        u_matrix, singular_values, vh_matrix = torch.linalg.svd(covariance)
        rotation = vh_matrix.T @ u_matrix.T
        if torch.det(rotation) < 0:
            vh_matrix = vh_matrix.clone()
            vh_matrix[-1] *= -1.0
            rotation = vh_matrix.T @ u_matrix.T
        scale = singular_values.sum() * float(source_points.shape[0]) / source_energy
    else:
        rotation = torch.eye(2, device=source_points.device, dtype=source_points.dtype)
        scale = (source_centered * target_centered).sum() / source_energy

    transform = source_points.new_zeros((2, 3))
    transform[:, :2] = rotation * scale
    transform[:, 2] = target_mean - transform[:, :2] @ source_mean
    return transform


def waft_alignment_transform_key(transform: Optional[torch.Tensor]) -> str:
    if transform is None:
        return "none"
    values = transform.detach().cpu().reshape(-1).tolist()
    return ",".join(f"{float(value):.6f}" for value in values)


def warn_waft_alignment_once(key: str, message: str) -> None:
    if key in WAFT_ALIGNMENT_WARNING_KEYS:
        return
    WAFT_ALIGNMENT_WARNING_KEYS.add(key)
    print(f"[WARNING] {message}")


def load_waft_target_rgb_image(
    path: str,
    image_size: tuple[int, int],
    flow_config: Mapping[str, Any],
    alignment_transform: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    image = load_loss_rgb_image(path, image_size, device, dtype)
    return align_waft_target_image(image, flow_config, alignment_transform)


def align_waft_target_image(
    image: torch.Tensor,
    flow_config: Mapping[str, Any],
    alignment_transform: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    alignment = flow_config.get("waft_target_image_alignment")
    if not isinstance(alignment, Mapping) or not bool(alignment.get("enabled", False)):
        return image
    if image.dim() != 4 or image.shape[-1] != 3:
        raise ValueError("WAFT target image alignment expects BHWC RGB images.")
    if alignment_transform is not None:
        return apply_waft_target_alignment_transform(image, alignment_transform)

    mode = str(alignment.get("mode", "bbox")).strip().lower()
    if mode in {"landmark", "landmarks", "mediapipe", "mediapipe_landmarks"}:
        fallback_mode = str(alignment.get("fallback_mode", "bbox")).strip().lower()
        if fallback_mode in {"none", "off", "disabled"}:
            return image

    threshold = float(
        alignment.get(
            "foreground_threshold",
            flow_config.get("waft_background_white_threshold", 0.97),
        )
    )
    target_bbox = parse_alignment_bbox(alignment)
    preserve_aspect = bool(alignment.get("preserve_aspect", True))
    align_x = normalize_roi_fraction(alignment.get("align_x", 0.5))
    align_y = normalize_roi_fraction(alignment.get("align_y", 0.5))

    batch_size, height, width = image.shape[:3]
    output = torch.ones_like(image)
    target_x0 = int(round(target_bbox[0] * width))
    target_y0 = int(round(target_bbox[1] * height))
    target_x1 = int(round(target_bbox[2] * width))
    target_y1 = int(round(target_bbox[3] * height))
    target_x0 = min(max(target_x0, 0), width - 1)
    target_y0 = min(max(target_y0, 0), height - 1)
    target_x1 = min(max(target_x1, target_x0 + 1), width)
    target_y1 = min(max(target_y1, target_y0 + 1), height)
    target_width = target_x1 - target_x0
    target_height = target_y1 - target_y0

    for batch_index in range(batch_size):
        foreground = (image[batch_index] < threshold).any(dim=-1)
        positions = torch.where(foreground)
        if positions[0].numel() == 0:
            output[batch_index] = image[batch_index]
            continue

        y0 = int(positions[0].min())
        y1 = int(positions[0].max()) + 1
        x0 = int(positions[1].min())
        x1 = int(positions[1].max()) + 1
        crop = image[batch_index, y0:y1, x0:x1]
        crop_height, crop_width = crop.shape[:2]
        if preserve_aspect:
            scale = min(
                target_width / max(crop_width, 1),
                target_height / max(crop_height, 1),
            )
            fitted_width = max(int(round(crop_width * scale)), 1)
            fitted_height = max(int(round(crop_height * scale)), 1)
        else:
            fitted_width = target_width
            fitted_height = target_height

        resized = (
            F.interpolate(
                crop.permute(2, 0, 1).unsqueeze(0),
                size=(fitted_height, fitted_width),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )
        free_x = target_width - fitted_width
        free_y = target_height - fitted_height
        paste_x0 = target_x0 + int(round(free_x * align_x))
        paste_y0 = target_y0 + int(round(free_y * align_y))
        paste_x1 = paste_x0 + fitted_width
        paste_y1 = paste_y0 + fitted_height
        output[batch_index, paste_y0:paste_y1, paste_x0:paste_x1] = resized

    return output


def apply_waft_target_alignment_transform(
    image: torch.Tensor,
    source_to_target: torch.Tensor,
) -> torch.Tensor:
    if image.dim() != 4 or image.shape[-1] != 3:
        raise ValueError("WAFT target image alignment expects BHWC RGB images.")
    batch_size, height, width = image.shape[:3]
    transform = source_to_target.to(device=image.device, dtype=image.dtype)
    if transform.shape != (2, 3):
        raise ValueError("WAFT landmark alignment transform must have shape [2, 3].")

    linear = transform[:, :2]
    translation = transform[:, 2]
    inverse_linear = torch.linalg.inv(linear)
    inverse_translation = -(inverse_linear @ translation)

    ys = torch.arange(height, device=image.device, dtype=image.dtype)
    xs = torch.arange(width, device=image.device, dtype=image.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    target_xy = torch.stack((grid_x, grid_y), dim=-1).view(-1, 2)
    source_xy = target_xy @ inverse_linear.T + inverse_translation
    source_x = source_xy[:, 0].view(height, width)
    source_y = source_xy[:, 1].view(height, width)

    if width > 1:
        grid_x = source_x / float(width - 1) * 2.0 - 1.0
        valid_x = (source_x >= 0.0) & (source_x <= float(width - 1))
    else:
        grid_x = torch.zeros_like(source_x)
        valid_x = torch.ones_like(source_x, dtype=torch.bool)
    if height > 1:
        grid_y = source_y / float(height - 1) * 2.0 - 1.0
        valid_y = (source_y >= 0.0) & (source_y <= float(height - 1))
    else:
        grid_y = torch.zeros_like(source_y)
        valid_y = torch.ones_like(source_y, dtype=torch.bool)

    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(
        batch_size,
        -1,
        -1,
        -1,
    )
    valid = (valid_x & valid_y).to(dtype=image.dtype).view(1, 1, height, width)
    valid = valid.expand(batch_size, -1, -1, -1)
    sampled = F.grid_sample(
        image.permute(0, 3, 1, 2),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    warped = sampled * valid + (1.0 - valid)
    return warped.permute(0, 2, 3, 1)


def parse_alignment_bbox(
    alignment: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    value = alignment.get("target_bbox")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 4:
            raise ValueError(
                "waft_target_image_alignment.target_bbox must have 4 values."
            )
        x_min, y_min, x_max, y_max = (
            normalize_roi_fraction(item) for item in value
        )
    else:
        x_min = normalize_roi_fraction(alignment.get("x_min", 0.0))
        y_min = normalize_roi_fraction(alignment.get("y_min", 0.0))
        x_max = normalize_roi_fraction(alignment.get("x_max", 1.0))
        y_max = normalize_roi_fraction(alignment.get("y_max", 1.0))
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min
    if x_max == x_min or y_max == y_min:
        raise ValueError("waft_target_image_alignment target bbox must have area.")
    return x_min, y_min, x_max, y_max


def waft_target_image_alignment_key(flow_config: Mapping[str, Any]) -> str:
    alignment = flow_config.get("waft_target_image_alignment", {})
    if not isinstance(alignment, Mapping):
        return "{}"
    return json.dumps(alignment, sort_keys=True, default=str)


def waft_functional_alignment_key(alignment: Mapping[str, Any]) -> str:
    return json.dumps(
        waft_functional_alignment_config(alignment),
        sort_keys=True,
        default=str,
    )


def compute_image_geometry_regularization(
    vertices: torch.Tensor,
    deformed_vertices: torch.Tensor,
    predicted_delta: torch.Tensor,
    faces: torch.Tensor,
    config: Mapping[str, Any],
    render_config: Optional[Mapping[str, Any]] = None,
    landmark_vertex_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not bool(config.get("enabled", False)):
        return predicted_delta.new_zeros(()), {}

    if predicted_delta.dim() == 4:
        if predicted_delta.shape[1] != 1:
            raise ValueError(
                "Image geometry regularization expects one pose per image sample."
            )
        predicted_delta = predicted_delta[:, 0]
    if deformed_vertices.dim() == 4:
        if deformed_vertices.shape[1] != 1:
            raise ValueError(
                "Image geometry regularization expects one pose per image sample."
            )
        deformed_vertices = deformed_vertices[:, 0]
    if predicted_delta.shape != vertices.shape:
        raise ValueError("predicted_delta must match vertices shape.")
    if deformed_vertices.shape != vertices.shape:
        raise ValueError("deformed_vertices must match vertices shape.")

    loss = predicted_delta.new_zeros(())
    metrics: dict[str, float] = {}

    l2_weight = float(config.get("displacement_l2_weight", 0.0))
    if l2_weight > 0.0:
        raw = predicted_delta.square().mean()
        weighted = raw * l2_weight
        loss = loss + weighted
        metrics["image_2d_geom_displacement_l2"] = float(raw.detach().cpu())
        metrics["image_2d_geom_displacement_l2_weighted"] = float(
            weighted.detach().cpu()
        )

    anchor_config = config.get("anchor_outside_roi", {})
    if isinstance(anchor_config, Mapping) and bool(anchor_config.get("enabled", False)):
        anchor_weight = float(anchor_config.get("weight", 0.0))
        if anchor_weight > 0.0:
            allowed_mask = image_geometry_anchor_roi_mask(
                vertices=vertices,
                anchor_config=anchor_config,
                render_config=render_config,
                landmark_vertex_ids=landmark_vertex_ids,
                faces=faces,
            )
            outside_mask = ~allowed_mask
            delta_vertex_l2 = predicted_delta.square().sum(dim=-1)
            if bool(outside_mask.any().detach().cpu()):
                raw = delta_vertex_l2[outside_mask].mean()
            else:
                raw = delta_vertex_l2.new_zeros(())
            weighted = raw * anchor_weight
            loss = loss + weighted
            metrics["image_2d_geom_anchor_outside_roi"] = float(raw.detach().cpu())
            metrics["image_2d_geom_anchor_outside_roi_weighted"] = float(
                weighted.detach().cpu()
            )
            metrics["image_2d_geom_anchor_allowed_ratio"] = float(
                allowed_mask.to(dtype=predicted_delta.dtype).mean().detach().cpu()
            )
            if landmark_vertex_ids is not None:
                landmark_allowed = image_geometry_landmark_mask_values(
                    allowed_mask,
                    landmark_vertex_ids,
                )
                metrics["image_2d_geom_anchor_landmark_allowed_ratio"] = float(
                    landmark_allowed.to(dtype=predicted_delta.dtype)
                    .mean()
                    .detach()
                    .cpu()
                )

    max_displacement = config.get("max_displacement")
    magnitude_weight = float(config.get("magnitude_clamp_weight", 0.0))
    if max_displacement is not None:
        delta_norm = torch.linalg.vector_norm(predicted_delta, dim=-1)
        metrics["image_2d_geom_delta_norm_mean"] = float(
            delta_norm.mean().detach().cpu()
        )
        metrics["image_2d_geom_delta_norm_max"] = float(
            delta_norm.amax().detach().cpu()
        )
        if magnitude_weight > 0.0:
            excess = (delta_norm - float(max_displacement)).clamp_min(0.0)
            raw = excess.square().mean()
            weighted = raw * magnitude_weight
            loss = loss + weighted
            metrics["image_2d_geom_magnitude_clamp"] = float(raw.detach().cpu())
            metrics["image_2d_geom_magnitude_clamp_weighted"] = float(
                weighted.detach().cpu()
            )

    edges = sampled_face_edges(
        faces,
        max_edges=optional_positive_int(config.get("max_edges")),
    )
    if edges.numel() > 0:
        edges = edges.to(device=vertices.device, dtype=torch.long)
        laplacian_weight = float(config.get("laplacian_weight", 0.0))
        if laplacian_weight > 0.0:
            raw = edge_delta_smoothness_loss(predicted_delta, edges)
            weighted = raw * laplacian_weight
            loss = loss + weighted
            metrics["image_2d_geom_laplacian"] = float(raw.detach().cpu())
            metrics["image_2d_geom_laplacian_weighted"] = float(
                weighted.detach().cpu()
            )

        edge_weight = float(config.get("edge_length_weight", 0.0))
        if edge_weight > 0.0:
            raw = edge_length_preservation_loss(
                vertices=vertices,
                deformed_vertices=deformed_vertices,
                edges=edges,
                relative=bool(config.get("edge_length_relative", True)),
                min_edge_length=float(config.get("min_edge_length", 1.0e-4)),
            )
            weighted = raw * edge_weight
            loss = loss + weighted
            metrics["image_2d_geom_edge_length"] = float(raw.detach().cpu())
            metrics["image_2d_geom_edge_length_weighted"] = float(
                weighted.detach().cpu()
            )

    normal_weight = float(config.get("normal_consistency_weight", 0.0))
    if normal_weight > 0.0:
        face_sample = sampled_faces(
            faces,
            max_faces=optional_positive_int(config.get("max_faces")),
        )
        if face_sample.numel() > 0:
            face_sample = face_sample.to(device=vertices.device, dtype=torch.long)
            raw = face_normal_consistency_loss(vertices, deformed_vertices, face_sample)
            weighted = raw * normal_weight
            loss = loss + weighted
            metrics["image_2d_geom_normal_consistency"] = float(raw.detach().cpu())
            metrics["image_2d_geom_normal_consistency_weighted"] = float(
                weighted.detach().cpu()
            )

    metrics["image_2d_geom_loss"] = float(loss.detach().cpu())
    return loss, metrics


def image_geometry_anchor_roi_mask(
    vertices: torch.Tensor,
    anchor_config: Mapping[str, Any],
    render_config: Optional[Mapping[str, Any]] = None,
    landmark_vertex_ids: Optional[torch.Tensor] = None,
    faces: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if vertices.dim() != 3 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape [B, V, 3].")

    render_config = render_config or {}
    render_kwargs = (
        render_config.get("kwargs", {}) if isinstance(render_config, Mapping) else {}
    )
    if not isinstance(render_kwargs, Mapping):
        render_kwargs = {}
    image_size = normalize_image_size(
        render_config.get("train_image_size", (256, 256))
        if isinstance(render_config, Mapping)
        else (256, 256)
    )
    landmark_roi_config = anchor_config.get("landmark_centered_roi", {})
    if isinstance(landmark_roi_config, Mapping) and bool(
        landmark_roi_config.get("enabled", False)
    ):
        if landmark_vertex_ids is None:
            raise ValueError(
                "Image geometry landmark-centered ROI is enabled, but the active "
                "action unit has no mapped ROI landmark vertex IDs."
            )
        mode = str(landmark_roi_config.get("mode", "projected")).strip().lower()
        if mode in {
            "cached",
            "cached_vertex_ids",
            "surface_anchor",
            "surface_anchor_cache",
        }:
            allowed_mask = image_geometry_direct_vertex_roi_mask(
                vertex_count=vertices.shape[1],
                vertex_ids=landmark_vertex_ids,
                batch_size=vertices.shape[0],
                device=vertices.device,
            )
        elif mode in {"topology", "topological", "face_hops", "mesh_hops"}:
            if faces is None:
                raise ValueError(
                    "Topology-based landmark ROI requires mesh faces."
                )
            allowed_mask = image_geometry_landmark_topology_roi_mask(
                vertex_count=vertices.shape[1],
                faces=faces,
                landmark_vertex_ids=landmark_vertex_ids,
                config=landmark_roi_config,
                batch_size=vertices.shape[0],
                device=vertices.device,
            )
        elif mode in {"component", "components", "connected_component"}:
            if faces is None:
                raise ValueError(
                    "Component-based landmark ROI requires mesh faces."
                )
            allowed_mask = image_geometry_landmark_component_roi_mask(
                vertex_count=vertices.shape[1],
                faces=faces,
                landmark_vertex_ids=landmark_vertex_ids,
                batch_size=vertices.shape[0],
                device=vertices.device,
            )
        elif mode in {"projected", "screen", "screen_bbox", "bbox"}:
            projected = validation_project_points_to_screen(
                points=vertices.detach(),
                image_size=image_size,
                render_kwargs=render_kwargs,
                fit_reference_vertices=vertices.detach(),
            )
            allowed_mask = image_geometry_landmark_centered_roi_mask(
                projected=projected,
                landmark_vertex_ids=landmark_vertex_ids,
                config=landmark_roi_config,
            )
        else:
            raise ValueError(
                "landmark_centered_roi.mode must be projected, topology, or "
                "component."
            )
        landmark_allowed = image_geometry_landmark_mask_values(
            allowed_mask,
            landmark_vertex_ids,
        )
        if not bool(landmark_allowed.all().detach().cpu()):
            raise RuntimeError(
                "Landmark-centered ROI masked one or more required ROI vertices."
            )
        return allowed_mask

    projected = validation_project_points_to_screen(
        points=vertices.detach(),
        image_size=image_size,
        render_kwargs=render_kwargs,
        fit_reference_vertices=vertices.detach(),
    )
    mins = projected.amin(dim=1, keepdim=True)
    maxs = projected.amax(dim=1, keepdim=True)
    normalized = (projected - mins) / (maxs - mins).clamp_min(1.0e-6)
    x_min = normalize_roi_fraction(anchor_config.get("x_min", 0.0))
    x_max = normalize_roi_fraction(anchor_config.get("x_max", 1.0))
    y_min = normalize_roi_fraction(anchor_config.get("y_min", 0.0))
    y_max = normalize_roi_fraction(anchor_config.get("y_max", 1.0))
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min
    return (
        (normalized[..., 0] >= x_min)
        & (normalized[..., 0] <= x_max)
        & (normalized[..., 1] >= y_min)
        & (normalized[..., 1] <= y_max)
    )


def image_geometry_action_unit_roi_vertex_ids(
    *,
    landmarks_3d: Any,
    action_unit_id: int,
    geometry_config: Mapping[str, Any],
    device: torch.device,
) -> Optional[torch.Tensor]:
    anchor_config = geometry_config.get("anchor_outside_roi", {})
    if not isinstance(anchor_config, Mapping):
        return None
    landmark_roi_config = anchor_config.get("landmark_centered_roi", {})
    if not isinstance(landmark_roi_config, Mapping) or not bool(
        landmark_roi_config.get("enabled", False)
    ):
        return None

    mode = str(landmark_roi_config.get("mode", "projected")).strip().lower()
    if mode in {
        "cached",
        "cached_vertex_ids",
        "surface_anchor",
        "surface_anchor_cache",
    }:
        if not isinstance(landmarks_3d, Mapping):
            raise ValueError(
                "Cached image geometry ROI requires gaze surface-anchor metadata."
            )
        cached_ids = landmarks_3d.get("gaze_roi_vertex_ids")
        if not isinstance(cached_ids, torch.Tensor) or cached_ids.numel() == 0:
            raise ValueError(
                "Cached image geometry ROI requires non-empty gaze_roi_vertex_ids."
            )
        return cached_ids.detach().long().flatten().unique(sorted=True).to(device)

    configured_ids = landmark_roi_config.get(
        "mediapipe_ids",
        landmark_roi_config.get("landmark_ids"),
    )
    if configured_ids not in (None, ""):
        required_ids = normalize_int_list(
            configured_ids,
            "landmark_centered_roi.mediapipe_ids",
        )
        if not required_ids:
            raise ValueError(
                "landmark_centered_roi.mediapipe_ids must not be empty."
            )
    else:
        try:
            curves = eyelid_landmarks_for_action_unit(action_unit_id)
        except ValueError as exc:
            raise ValueError(
                "Image geometry landmark-centered ROI requires explicit "
                f"mediapipe_ids for AU{action_unit_id}."
            ) from exc
        required_ids = list(curves.all_ids)
    vertex_ids = eyelid_vertex_ids(
        landmarks_3d,
        required_ids,
        device=device,
    )
    if vertex_ids is None:
        raise ValueError(
            f"Image geometry landmark-centered ROI for AU{action_unit_id} is "
            "missing one or more required 3D eyelid landmarks."
        )
    return vertex_ids


def image_geometry_direct_vertex_roi_mask(
    *,
    vertex_count: int,
    vertex_ids: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    vertex_ids = image_geometry_batched_landmark_vertex_ids(
        vertex_ids,
        batch_size=batch_size,
        vertex_count=vertex_count,
        device=device,
    )
    mask = torch.zeros(batch_size, vertex_count, dtype=torch.bool, device=device)
    mask.scatter_(1, vertex_ids, True)
    return mask


def image_geometry_landmark_centered_roi_mask(
    *,
    projected: torch.Tensor,
    landmark_vertex_ids: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    if projected.dim() != 3 or projected.shape[-1] != 2:
        raise ValueError("projected must have shape [B, V, 2].")
    landmark_vertex_ids = image_geometry_batched_landmark_vertex_ids(
        landmark_vertex_ids,
        batch_size=projected.shape[0],
        vertex_count=projected.shape[1],
        device=projected.device,
    )
    landmark_points = torch.gather(
        projected,
        dim=1,
        index=landmark_vertex_ids.unsqueeze(-1).expand(-1, -1, 2),
    )
    landmark_mins = landmark_points.amin(dim=1, keepdim=True)
    landmark_maxs = landmark_points.amax(dim=1, keepdim=True)
    minimum_eye_width = float(config.get("min_eye_width_px", 1.0))
    eye_width = (landmark_maxs[..., 0] - landmark_mins[..., 0]).clamp_min(
        minimum_eye_width
    )
    x_margin = eye_width * float(config.get("x_margin_eye_widths", 0.75))
    y_margin = eye_width * float(config.get("y_margin_eye_widths", 1.25))
    x_min = landmark_mins[..., 0] - x_margin
    x_max = landmark_maxs[..., 0] + x_margin
    y_min = landmark_mins[..., 1] - y_margin
    y_max = landmark_maxs[..., 1] + y_margin
    return (
        (projected[..., 0] >= x_min)
        & (projected[..., 0] <= x_max)
        & (projected[..., 1] >= y_min)
        & (projected[..., 1] <= y_max)
    )


def image_geometry_landmark_topology_roi_mask(
    *,
    vertex_count: int,
    faces: torch.Tensor,
    landmark_vertex_ids: torch.Tensor,
    config: Mapping[str, Any],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    topology_hops = int(config.get("topology_hops", 8))
    if topology_hops < 0:
        raise ValueError("landmark_centered_roi.topology_hops must be non-negative.")
    vertex_ids = image_geometry_batched_landmark_vertex_ids(
        landmark_vertex_ids,
        batch_size=batch_size,
        vertex_count=vertex_count,
        device=device,
    )
    face_batches = faces.to(device=device, dtype=torch.long)
    if face_batches.dim() == 2:
        face_batches = face_batches.unsqueeze(0).expand(batch_size, -1, -1)
    elif face_batches.dim() == 3 and face_batches.shape[0] == 1 and batch_size > 1:
        face_batches = face_batches.expand(batch_size, -1, -1)
    if (
        face_batches.dim() != 3
        or face_batches.shape[0] != batch_size
        or face_batches.shape[-1] != 3
    ):
        raise ValueError("faces must have shape [F, 3] or [B, F, 3].")
    if bool(
        ((face_batches < 0) | (face_batches >= vertex_count)).any().detach().cpu()
    ):
        raise ValueError("faces contains an out-of-range vertex ID.")

    mask = torch.zeros(batch_size, vertex_count, dtype=torch.bool, device=device)
    mask.scatter_(1, vertex_ids, True)
    for _hop in range(topology_hops):
        face_vertices = torch.gather(
            mask,
            dim=1,
            index=face_batches.reshape(batch_size, -1),
        ).reshape_as(face_batches)
        active_faces = face_vertices.any(dim=-1)
        for batch_index in range(batch_size):
            active_vertex_ids = face_batches[batch_index][
                active_faces[batch_index]
            ].reshape(-1)
            mask[batch_index, active_vertex_ids] = True
    return mask


def image_geometry_landmark_component_roi_mask(
    *,
    vertex_count: int,
    faces: torch.Tensor,
    landmark_vertex_ids: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    vertex_ids = image_geometry_batched_landmark_vertex_ids(
        landmark_vertex_ids,
        batch_size=batch_size,
        vertex_count=vertex_count,
        device=device,
    )
    face_batches = faces.to(device=device, dtype=torch.long)
    if face_batches.dim() == 2:
        face_batches = face_batches.unsqueeze(0).expand(batch_size, -1, -1)
    elif face_batches.dim() == 3 and face_batches.shape[0] == 1 and batch_size > 1:
        face_batches = face_batches.expand(batch_size, -1, -1)
    if (
        face_batches.dim() != 3
        or face_batches.shape[0] != batch_size
        or face_batches.shape[-1] != 3
    ):
        raise ValueError("faces must have shape [F, 3] or [B, F, 3].")

    mask = torch.zeros(batch_size, vertex_count, dtype=torch.bool, device=device)
    for batch_index in range(batch_size):
        component_ids = mesh_connected_component_ids(
            face_batches[batch_index],
            vertex_count,
        ).to(device=device)
        seed_components = component_ids.index_select(
            0,
            vertex_ids[batch_index],
        ).unique()
        mask[batch_index] = torch.isin(component_ids, seed_components)
    return mask


def image_geometry_batched_landmark_vertex_ids(
    landmark_vertex_ids: torch.Tensor,
    *,
    batch_size: int,
    vertex_count: int,
    device: torch.device,
) -> torch.Tensor:
    vertex_ids = landmark_vertex_ids.to(device=device, dtype=torch.long)
    if vertex_ids.dim() == 1:
        vertex_ids = vertex_ids.unsqueeze(0).expand(batch_size, -1)
    elif vertex_ids.dim() == 2 and vertex_ids.shape[0] == 1 and batch_size > 1:
        vertex_ids = vertex_ids.expand(batch_size, -1)
    if vertex_ids.dim() != 2 or vertex_ids.shape[0] != batch_size:
        raise ValueError(
            "landmark_vertex_ids must have shape [L] or [B, L] matching vertices."
        )
    if vertex_ids.numel() == 0:
        raise ValueError("landmark_vertex_ids must not be empty.")
    if bool(((vertex_ids < 0) | (vertex_ids >= vertex_count)).any().detach().cpu()):
        raise ValueError("landmark_vertex_ids contains an out-of-range vertex ID.")
    return vertex_ids


def image_geometry_landmark_mask_values(
    mask: torch.Tensor,
    landmark_vertex_ids: torch.Tensor,
) -> torch.Tensor:
    if mask.dim() != 2:
        raise ValueError("mask must have shape [B, V].")
    vertex_ids = image_geometry_batched_landmark_vertex_ids(
        landmark_vertex_ids,
        batch_size=mask.shape[0],
        vertex_count=mask.shape[1],
        device=mask.device,
    )
    return torch.gather(mask, dim=1, index=vertex_ids)


def apply_image_geometry_hard_roi_mask(
    vertices: torch.Tensor,
    predicted_delta: torch.Tensor,
    geometry_config: Mapping[str, Any],
    render_config: Optional[Mapping[str, Any]] = None,
    landmark_vertex_ids: Optional[torch.Tensor] = None,
    faces: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    anchor_config = geometry_config.get("anchor_outside_roi", {})
    masked_delta = predicted_delta
    if isinstance(anchor_config, Mapping) and bool(
        anchor_config.get("hard_mask", False)
    ):
        allowed_mask = image_geometry_anchor_roi_mask(
            vertices=vertices,
            anchor_config=anchor_config,
            render_config=render_config,
            landmark_vertex_ids=landmark_vertex_ids,
            faces=faces,
        )
        masked_delta = predicted_delta * allowed_mask.unsqueeze(-1).to(
            dtype=predicted_delta.dtype
        )

    seam_config = geometry_config.get("synchronize_coincident_vertices", {})
    if isinstance(seam_config, Mapping) and bool(
        seam_config.get("enabled", False)
    ):
        masked_delta = synchronize_coincident_vertex_displacements(
            vertices=vertices,
            displacements=masked_delta,
            tolerance=float(seam_config.get("tolerance", 1.0e-6)),
        )
    return masked_delta


def synchronize_coincident_vertex_displacements(
    *,
    vertices: torch.Tensor,
    displacements: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    if vertices.shape != displacements.shape:
        raise ValueError("vertices and displacements must have matching shapes.")
    if vertices.dim() != 3 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape [B, V, 3].")
    if tolerance <= 0.0:
        raise ValueError("Coincident-vertex tolerance must be positive.")

    synchronized = []
    for batch_index in range(vertices.shape[0]):
        quantized = torch.round(vertices[batch_index].detach() / tolerance).to(
            dtype=torch.int64
        )
        _unique_positions, group_ids = torch.unique(
            quantized,
            dim=0,
            return_inverse=True,
        )
        group_count = int(group_ids.amax().detach().cpu()) + 1
        group_sums = displacements.new_zeros((group_count, 3))
        group_sums.index_add_(0, group_ids, displacements[batch_index])
        counts = torch.bincount(group_ids, minlength=group_count).to(
            device=displacements.device,
            dtype=displacements.dtype,
        )
        group_means = group_sums / counts.unsqueeze(-1).clamp_min(1.0)
        synchronized.append(group_means.index_select(0, group_ids))
    return torch.stack(synchronized, dim=0)


def apply_image_geometry_hard_roi_mask_for_action_units(
    *,
    vertices: torch.Tensor,
    predicted_delta: torch.Tensor,
    action_unit_ids: Sequence[int],
    landmarks_3d: Any,
    geometry_config: Mapping[str, Any],
    render_config: Optional[Mapping[str, Any]] = None,
    faces: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if vertices.shape != predicted_delta.shape:
        raise ValueError("vertices and predicted_delta must have matching shapes.")
    if vertices.dim() != 3 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape [B, V, 3].")
    if len(action_unit_ids) != vertices.shape[0]:
        raise ValueError("action_unit_ids must match the prediction batch size.")

    masked = []
    for index, action_unit_id in enumerate(action_unit_ids):
        action_geometry_config = config_for_action_unit(
            geometry_config,
            int(action_unit_id),
        )
        landmark_vertex_ids = image_geometry_action_unit_roi_vertex_ids(
            landmarks_3d=landmarks_3d,
            action_unit_id=int(action_unit_id),
            geometry_config=action_geometry_config,
            device=vertices.device,
        )
        masked.append(
            apply_image_geometry_hard_roi_mask(
                vertices=vertices[index : index + 1],
                predicted_delta=predicted_delta[index : index + 1],
                geometry_config=action_geometry_config,
                render_config=render_config,
                landmark_vertex_ids=landmark_vertex_ids,
                faces=faces,
            )
        )
    return torch.cat(masked, dim=0)


def optional_positive_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    parsed = int(value)
    if parsed < 1:
        return None
    return parsed


def sampled_faces(faces: torch.Tensor, max_faces: Optional[int] = None) -> torch.Tensor:
    if faces.dim() == 3:
        if faces.shape[0] != 1:
            raise ValueError("Only shared or single-batch faces are supported.")
        faces = faces[0]
    if faces.dim() != 2 or faces.shape[-1] != 3:
        raise ValueError("faces must have shape [F, 3].")
    if max_faces is None or faces.shape[0] <= max_faces:
        return faces
    indices = torch.linspace(
        0,
        faces.shape[0] - 1,
        steps=max_faces,
        device=faces.device,
    ).round().long()
    return faces.index_select(0, indices)


def sampled_face_edges(
    faces: torch.Tensor,
    max_edges: Optional[int] = None,
) -> torch.Tensor:
    faces = sampled_faces(faces)
    face_count = int(faces.shape[0])
    total_edges = face_count * 3
    if max_edges is not None and total_edges > int(max_edges):
        edge_indices = torch.linspace(
            0,
            total_edges - 1,
            steps=int(max_edges),
            device=faces.device,
        ).round().long()
        face_indices = edge_indices.remainder(face_count)
        edge_slot = torch.div(edge_indices, face_count, rounding_mode="floor")
        face_sample = faces.index_select(0, face_indices)
        source = torch.where(
            edge_slot == 0,
            face_sample[:, 0],
            torch.where(edge_slot == 1, face_sample[:, 1], face_sample[:, 2]),
        )
        target = torch.where(
            edge_slot == 0,
            face_sample[:, 1],
            torch.where(edge_slot == 1, face_sample[:, 2], face_sample[:, 0]),
        )
        return torch.stack((source, target), dim=1)
    edges = torch.cat(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ),
        dim=0,
    )
    if max_edges is None or edges.shape[0] <= max_edges:
        return edges
    indices = torch.linspace(
        0,
        edges.shape[0] - 1,
        steps=max_edges,
        device=edges.device,
    ).round().long()
    return edges.index_select(0, indices)


def selected_face_edges(
    faces: torch.Tensor,
    vertex_mask: Optional[torch.Tensor],
    mode: str,
    max_edges: Optional[int] = None,
) -> torch.Tensor:
    if vertex_mask is None:
        return sampled_face_edges(
            faces.detach().to(dtype=torch.long),
            max_edges=max_edges,
        )
    faces_cpu = faces.detach().cpu().to(dtype=torch.long)
    mask_cpu = vertex_mask.detach().cpu().to(dtype=torch.bool)
    edges = torch.cat(
        (
            faces_cpu[:, [0, 1]],
            faces_cpu[:, [1, 2]],
            faces_cpu[:, [2, 0]],
        ),
        dim=0,
    )
    counts = mask_cpu.index_select(0, edges.reshape(-1)).reshape(edges.shape).sum(dim=1)
    keep = geometry_mask_keep(counts, required_count=2, mode=mode)
    edges = edges[keep]
    return sampled_rows(edges, max_edges)


def selected_faces(
    faces: torch.Tensor,
    vertex_mask: Optional[torch.Tensor],
    mode: str,
    max_faces: Optional[int] = None,
) -> torch.Tensor:
    if vertex_mask is None:
        return sampled_faces(faces.detach().to(dtype=torch.long), max_faces=max_faces)
    faces_cpu = faces.detach().cpu().to(dtype=torch.long)
    mask_cpu = vertex_mask.detach().cpu().to(dtype=torch.bool)
    counts = mask_cpu.index_select(0, faces_cpu.reshape(-1)).reshape(
        faces_cpu.shape,
    ).sum(dim=1)
    keep = geometry_mask_keep(counts, required_count=3, mode=mode)
    faces_cpu = faces_cpu[keep]
    return sampled_rows(faces_cpu, max_faces)


def sampled_rows(values: torch.Tensor, max_rows: Optional[int]) -> torch.Tensor:
    if max_rows is None or values.shape[0] <= max_rows:
        return values
    indices = torch.linspace(
        0,
        values.shape[0] - 1,
        steps=max_rows,
        device=values.device,
    ).round().long()
    return values.index_select(0, indices)


def geometry_mask_keep(
    counts: torch.Tensor,
    required_count: int,
    mode: str,
) -> torch.Tensor:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode in {"touch", "any", "intersect"}:
        return counts > 0
    if normalized_mode in {"inside", "all", "contained"}:
        return counts == int(required_count)
    if normalized_mode in {"boundary", "mixed"}:
        return (counts > 0) & (counts < int(required_count))
    raise ValueError(
        "loss.mesh.geometry.target_vertex_mask_mode must be one of: "
        "touch, inside, or boundary."
    )




def smooth_vertex_displacement(
    displacement: torch.Tensor,
    faces: torch.Tensor,
    *,
    iterations: int,
    blend: float,
    max_edges: Optional[int] = None,
) -> torch.Tensor:
    if iterations <= 0 or blend <= 0.0:
        return displacement
    if displacement.dim() == 2:
        displacement = displacement.unsqueeze(0)
        was_unbatched = True
    elif displacement.dim() == 3:
        was_unbatched = False
    else:
        raise ValueError("displacement must have shape [V, 3] or [B, V, 3].")

    edges = sampled_face_edges(faces, max_edges=max_edges)
    if edges.numel() == 0:
        return displacement[0] if was_unbatched else displacement
    edges = edges.to(device=displacement.device, dtype=torch.long)
    source = edges[:, 0]
    target = edges[:, 1]
    batch_size, vertex_count = displacement.shape[:2]
    lambda_value = float(blend)

    smoothed = displacement
    for _ in range(int(iterations)):
        accumulated = torch.zeros_like(smoothed)
        degree = smoothed.new_zeros((batch_size, vertex_count, 1))
        source_values = smoothed.index_select(1, source)
        target_values = smoothed.index_select(1, target)
        accumulated.index_add_(1, source, target_values)
        accumulated.index_add_(1, target, source_values)
        ones = smoothed.new_ones((batch_size, edges.shape[0], 1))
        degree.index_add_(1, source, ones)
        degree.index_add_(1, target, ones)
        neighbor_mean = accumulated / degree.clamp_min(1.0)
        updated = smoothed.lerp(neighbor_mean, lambda_value)
        smoothed = torch.where(degree > 0.0, updated, smoothed)

    return smoothed[0] if was_unbatched else smoothed


















def mesh_target_vertex_mask(
    vertices: torch.Tensor,
    target_delta: Optional[torch.Tensor],
    landmarks_3d: Any,
    config: Mapping[str, Any],
    faces: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    mask_config = config.get("target_vertex_mask")
    if not isinstance(mask_config, Mapping):
        return None
    if not bool(mask_config.get("enabled", True)):
        return None
    return build_mesh_vertex_mask(
        vertices=vertices,
        target_delta=target_delta,
        landmarks_3d=landmarks_3d,
        faces=faces,
        mask_config=mask_config,
        name="loss.mesh.target_vertex_mask",
    )


def mesh_hard_target_vertex_mask_enabled(config: Mapping[str, Any]) -> bool:
    value = config.get("hard_target_vertex_mask", False)
    if isinstance(value, Mapping):
        return bool(value.get("enabled", True))
    return bool(value)


def mesh_hard_target_vertex_mask(
    vertices: torch.Tensor,
    target_delta: Optional[torch.Tensor],
    landmarks_3d: Any,
    config: Mapping[str, Any],
    faces: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if not mesh_hard_target_vertex_mask_enabled(config):
        return None
    mask_config = config.get("target_vertex_mask")
    if not isinstance(mask_config, Mapping):
        return None
    if target_delta is None and bool(mask_config.get("intersect_target_motion", False)):
        return None
    return mesh_target_vertex_mask(
        vertices=vertices,
        target_delta=target_delta,
        landmarks_3d=landmarks_3d,
        config=config,
        faces=faces,
    )


def mesh_reapply_target_vertex_mask_after_smoothing(config: Mapping[str, Any]) -> bool:
    hard_config = config.get("hard_target_vertex_mask")
    if isinstance(hard_config, Mapping):
        return bool(hard_config.get("reapply_after_smoothing", True))
    return bool(hard_config)


def mesh_loss_vertex_mask(
    vertex_mask: Optional[torch.Tensor],
    reference: torch.Tensor,
    faces: Optional[torch.Tensor],
    config: Mapping[str, Any],
) -> Optional[torch.Tensor]:
    if vertex_mask is None:
        return None
    hard_config = config.get("hard_target_vertex_mask")
    if not isinstance(hard_config, Mapping) or not bool(
        hard_config.get("use_for_loss", False)
    ):
        return vertex_mask
    return mesh_delta_vertex_mask_weights(
        vertex_mask,
        reference,
        faces=faces,
        config=hard_config,
    )


def apply_mesh_delta_vertex_mask(
    predicted_delta: torch.Tensor,
    vertex_mask: Optional[torch.Tensor],
    faces: Optional[torch.Tensor] = None,
    config: Any = None,
) -> torch.Tensor:
    if vertex_mask is None:
        return predicted_delta
    weights = mesh_delta_vertex_mask_weights(
        vertex_mask,
        predicted_delta,
        faces=faces,
        config=config,
    )
    return predicted_delta * weights.unsqueeze(-1)


def mesh_delta_vertex_mask_weights(
    vertex_mask: torch.Tensor,
    reference: torch.Tensor,
    faces: Optional[torch.Tensor] = None,
    config: Any = None,
) -> torch.Tensor:
    mask = normalized_vertex_mask(vertex_mask, reference)
    weights = mask.to(dtype=reference.dtype)
    if not isinstance(config, Mapping):
        return weights
    iterations = int(config.get("feather_iterations", 0) or 0)
    blend = float(config.get("feather_blend", config.get("feather_lambda", 0.5)))
    if iterations <= 0 or blend <= 0.0 or faces is None:
        return weights
    max_edges = optional_positive_int(config.get("feather_max_edges"))
    smoothed = smooth_vertex_displacement(
        weights.unsqueeze(-1),
        faces,
        iterations=iterations,
        blend=blend,
        max_edges=max_edges,
    ).squeeze(-1)
    smoothed = smoothed.clamp(0.0, 1.0)
    if bool(config.get("feather_keep_inside_one", False)):
        smoothed = torch.where(mask, torch.ones_like(smoothed), smoothed)
    if bool(config.get("feather_keep_outside_zero", True)):
        smoothed = torch.where(mask, smoothed, torch.zeros_like(smoothed))
    return smoothed


def filter_mesh_target_delta(
    vertices: torch.Tensor,
    target_delta: torch.Tensor,
    faces: Optional[torch.Tensor],
    vertex_mask: Optional[torch.Tensor],
    config: Mapping[str, Any],
) -> torch.Tensor:
    filter_config = config.get("target_delta_filter")
    if not isinstance(filter_config, Mapping) or not bool(
        filter_config.get("enabled", False)
    ):
        return target_delta
    return filter_mesh_delta_components(
        vertices=vertices,
        delta=target_delta,
        faces=faces,
        vertex_mask=vertex_mask,
        filter_config=filter_config,
        name="loss.mesh.target_delta_filter",
    )


def filter_mesh_prediction_delta(
    vertices: torch.Tensor,
    predicted_delta: torch.Tensor,
    faces: Optional[torch.Tensor],
    vertex_mask: Optional[torch.Tensor],
    config: Mapping[str, Any],
) -> torch.Tensor:
    filter_config = config.get("prediction_delta_filter")
    if not isinstance(filter_config, Mapping) or not bool(
        filter_config.get("enabled", False)
    ):
        return predicted_delta
    return filter_mesh_delta_components(
        vertices=vertices,
        delta=predicted_delta,
        faces=faces,
        vertex_mask=vertex_mask,
        filter_config=filter_config,
        name="loss.mesh.prediction_delta_filter",
    )


def filter_mesh_delta_components(
    vertices: torch.Tensor,
    delta: torch.Tensor,
    faces: Optional[torch.Tensor],
    vertex_mask: Optional[torch.Tensor],
    filter_config: Mapping[str, Any],
    name: str,
) -> torch.Tensor:
    mode = str(filter_config.get("mode", "rigid_components")).lower()
    valid_modes = {"rigid", "rigid_components", "rigid_region", "rigid_mask"}
    if mode not in valid_modes:
        raise ValueError(
            f"{name}.mode must be one of: rigid_components, rigid_region."
        )
    if faces is None:
        raise ValueError(f"{name} requires mesh faces.")
    if vertex_mask is None:
        if bool(filter_config.get("require_vertex_mask", True)):
            raise ValueError(f"{name} requires loss.mesh.target_vertex_mask.")
        mask = torch.ones(
            delta.shape[:2],
            device=delta.device,
            dtype=torch.bool,
        )
    else:
        mask = normalized_vertex_mask(vertex_mask, delta)

    component_ids = mesh_connected_component_ids(faces, delta.shape[1])
    component_ids_device = component_ids.to(device=delta.device)
    component_count = (
        int(component_ids.max().item()) + 1 if component_ids.numel() else 0
    )
    if component_count == 0:
        return delta

    min_vertices = int(filter_config.get("min_vertices", 3) or 3)
    blend = max(0.0, min(1.0, float(filter_config.get("blend", 1.0))))
    zero_outside = bool(filter_config.get("zero_outside_mask", True))
    filtered = torch.zeros_like(delta) if zero_outside else delta.clone()

    if mode in {"rigid_region", "rigid_mask"}:
        for batch_index in range(delta.shape[0]):
            selected = mask[batch_index]
            count = int(selected.sum().detach().cpu())
            if count < min_vertices:
                filtered[batch_index, selected] = delta[batch_index, selected]
                continue
            source = vertices[batch_index, selected]
            destination = source + delta[batch_index, selected]
            rigid_delta = rigid_fit_delta(source, destination)
            if blend < 1.0:
                rigid_delta = delta[batch_index, selected].lerp(
                    rigid_delta,
                    blend,
                )
            filtered[batch_index, selected] = rigid_delta
        return filtered

    for batch_index in range(delta.shape[0]):
        selected_components = torch.unique(component_ids_device[mask[batch_index]])
        for component_id in selected_components.tolist():
            selected = mask[batch_index] & (component_ids_device == int(component_id))
            if int(selected.sum().detach().cpu()) < min_vertices:
                filtered[batch_index, selected] = delta[batch_index, selected]
                continue
            source = vertices[batch_index, selected]
            destination = source + delta[batch_index, selected]
            rigid_delta = rigid_fit_delta(source, destination)
            if blend < 1.0:
                rigid_delta = delta[batch_index, selected].lerp(
                    rigid_delta,
                    blend,
                )
            filtered[batch_index, selected] = rigid_delta
    return filtered


def rigid_fit_delta(
    source: torch.Tensor,
    destination: torch.Tensor,
) -> torch.Tensor:
    if source.dim() != 2 or source.shape[-1] != 3:
        raise ValueError("source must have shape [N, 3].")
    if destination.shape != source.shape:
        raise ValueError("destination must match source shape.")
    if source.shape[0] < 3:
        translation = destination.mean(dim=0, keepdim=True) - source.mean(
            dim=0,
            keepdim=True,
        )
        return translation.expand_as(source)

    source_center = source.mean(dim=0, keepdim=True)
    destination_center = destination.mean(dim=0, keepdim=True)
    source_centered = source - source_center
    destination_centered = destination - destination_center
    covariance = source_centered.transpose(0, 1) @ destination_centered
    try:
        u, _s, vh = torch.linalg.svd(covariance)
    except RuntimeError:
        translation = destination_center - source_center
        return translation.expand_as(source)

    rotation = vh.transpose(0, 1) @ u.transpose(0, 1)
    if torch.linalg.det(rotation) < 0.0:
        vh = vh.clone()
        vh[-1] *= -1.0
        rotation = vh.transpose(0, 1) @ u.transpose(0, 1)
    fitted = source_centered @ rotation.transpose(0, 1) + destination_center
    return fitted - source


def normalized_vertex_mask(
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
        raise ValueError("vertex_mask batch size must match the reference batch size.")
    return vertex_mask.to(device=reference.device, dtype=torch.bool)




def build_mesh_vertex_mask(
    vertices: torch.Tensor,
    landmarks_3d: Any,
    mask_config: Mapping[str, Any],
    name: str,
    target_delta: Optional[torch.Tensor] = None,
    faces: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0)
    if vertices.dim() != 3 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape [B, V, 3].")

    include_parts = mesh_vertex_mask_parts(
        vertices=vertices,
        landmarks_3d=landmarks_3d,
        mask_config=mask_config,
        prefix="include",
        faces=faces,
        target_delta=target_delta,
    )
    if include_parts:
        mask = include_parts[0]
        for part in include_parts[1:]:
            mask = mask | part
    else:
        mask = torch.ones(
            vertices.shape[:2],
            device=vertices.device,
            dtype=torch.bool,
        )

    for excluded in mesh_vertex_mask_parts(
        vertices=vertices,
        landmarks_3d=landmarks_3d,
        mask_config=mask_config,
        prefix="exclude",
        faces=faces,
        target_delta=target_delta,
    ):
        mask = mask & ~excluded

    if bool(mask_config.get("invert", False)):
        mask = ~mask
    if bool(mask_config.get("intersect_target_motion", False)):
        if target_delta is None:
            raise ValueError(f"{name}.intersect_target_motion requires target_delta.")
        mask = mask & target_motion_vertex_mask(target_delta, mask_config)
    mask = postprocess_mesh_vertex_mask(
        mask=mask,
        faces=faces,
        target_delta=target_delta,
        mask_config=mask_config,
        name=name,
    )
    if not bool(mask.any().detach().cpu()):
        raise ValueError(f"{name} selected zero vertices.")
    return mask


def postprocess_mesh_vertex_mask(
    mask: torch.Tensor,
    faces: Optional[torch.Tensor],
    target_delta: Optional[torch.Tensor],
    mask_config: Mapping[str, Any],
    name: str,
) -> torch.Tensor:
    fill_holes_config = mask_config.get("fill_holes")
    if isinstance(fill_holes_config, Mapping) and bool(
        fill_holes_config.get("enabled", False)
    ):
        if faces is None:
            raise ValueError(f"{name}.fill_holes requires mesh faces.")
        mask = fill_mesh_vertex_mask_holes(
            mask=mask,
            faces=faces,
            config=fill_holes_config,
        )

    component_config = mask_config.get("connected_components")
    if not isinstance(component_config, Mapping) or not bool(
        component_config.get("enabled", False)
    ):
        return mask
    if faces is None:
        raise ValueError(f"{name}.connected_components requires mesh faces.")
    if target_delta is None:
        raise ValueError(f"{name}.connected_components requires target_delta.")
    return mesh_connected_component_vertex_mask(
        mask=mask,
        faces=faces,
        target_delta=target_delta,
        mask_config=mask_config,
        component_config=component_config,
    )


def fill_mesh_vertex_mask_holes(
    mask: torch.Tensor,
    faces: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.dim() != 2:
        raise ValueError("fill_holes mask must have shape [V] or [B, V].")
    if mask.shape[1] < 1:
        return mask

    iterations = int(config.get("iterations", 1) or 1)
    min_selected_neighbors = int(config.get("min_selected_neighbors", 3) or 3)
    min_selected_fraction = float(config.get("min_selected_fraction", 0.55))
    if iterations <= 0:
        return mask
    if min_selected_neighbors < 1:
        raise ValueError("fill_holes.min_selected_neighbors must be positive.")
    if not 0.0 <= min_selected_fraction <= 1.0:
        raise ValueError("fill_holes.min_selected_fraction must be in [0, 1].")

    edges = sampled_face_edges(faces.detach().cpu().long(), max_edges=None)
    if edges.numel() == 0:
        return mask
    edges = edges.to(device=mask.device, dtype=torch.long)
    valid = (
        (edges[:, 0] >= 0)
        & (edges[:, 0] < mask.shape[1])
        & (edges[:, 1] >= 0)
        & (edges[:, 1] < mask.shape[1])
    )
    edges = edges[valid]
    if edges.numel() == 0:
        return mask

    source = edges[:, 0]
    target = edges[:, 1]
    batch_size, vertex_count = mask.shape
    filled = mask.to(dtype=torch.bool)
    for _ in range(iterations):
        values = filled.to(dtype=torch.float32)
        selected_neighbors = values.new_zeros((batch_size, vertex_count))
        degree = values.new_zeros((batch_size, vertex_count))
        source_values = values.index_select(1, source)
        target_values = values.index_select(1, target)
        selected_neighbors.index_add_(1, source, target_values)
        selected_neighbors.index_add_(1, target, source_values)
        ones = values.new_ones((batch_size, edges.shape[0]))
        degree.index_add_(1, source, ones)
        degree.index_add_(1, target, ones)
        neighbor_fraction = selected_neighbors / degree.clamp_min(1.0)
        fill = (
            (~filled)
            & (degree > 0.0)
            & (selected_neighbors >= float(min_selected_neighbors))
            & (neighbor_fraction >= min_selected_fraction)
        )
        if not bool(fill.any().detach().cpu()):
            break
        filled = filled | fill
    return filled


def mesh_connected_component_vertex_mask(
    mask: torch.Tensor,
    faces: torch.Tensor,
    target_delta: torch.Tensor,
    mask_config: Mapping[str, Any],
    component_config: Mapping[str, Any],
) -> torch.Tensor:
    mask = normalized_vertex_mask(mask, target_delta)
    active = target_motion_vertex_mask(target_delta, mask_config)
    vertex_count = mask.shape[1]
    component_ids = mesh_connected_component_ids(faces, vertex_count)
    component_count = int(component_ids.max().item()) + 1 if component_ids.numel() else 0
    if component_count == 0:
        return mask

    sizes = torch.bincount(component_ids, minlength=component_count).to(torch.float32)
    min_selected_vertices = int(component_config.get("min_selected_vertices", 1) or 1)
    min_active_fraction = float(component_config.get("min_active_fraction", 0.0))
    min_selected_fraction = float(component_config.get("min_selected_fraction", 0.0))
    max_vertices = optional_positive_int(component_config.get("max_vertices"))
    drop_unfilled = bool(component_config.get("drop_unfilled", False))

    outputs: list[torch.Tensor] = []
    for batch_index in range(mask.shape[0]):
        selected_cpu = mask[batch_index].detach().cpu()
        active_cpu = active[batch_index].detach().cpu()
        selected_counts = torch.bincount(
            component_ids,
            weights=selected_cpu.to(torch.float32),
            minlength=component_count,
        )
        active_counts = torch.bincount(
            component_ids,
            weights=active_cpu.to(torch.float32),
            minlength=component_count,
        )
        selected_fraction = selected_counts / sizes.clamp_min(1.0)
        active_fraction = active_counts / sizes.clamp_min(1.0)
        fill_components = (
            (selected_counts >= float(min_selected_vertices))
            & (active_fraction >= min_active_fraction)
            & (selected_fraction >= min_selected_fraction)
        )
        if max_vertices is not None:
            fill_components = fill_components & (sizes <= float(max_vertices))
        filled_cpu = fill_components.index_select(0, component_ids)
        output_cpu = filled_cpu if drop_unfilled else (selected_cpu | filled_cpu)
        outputs.append(output_cpu.to(device=mask.device, dtype=torch.bool))
    return torch.stack(outputs, dim=0)


def mesh_connected_component_ids(
    faces: torch.Tensor,
    vertex_count: int,
) -> torch.Tensor:
    faces_cpu = faces.detach().cpu().to(dtype=torch.long).contiguous()
    shape_key = tuple(int(value) for value in faces_cpu.shape)
    ptr_key = (int(vertex_count), shape_key, int(faces_cpu.data_ptr()))
    cached = MESH_COMPONENT_PTR_CACHE.get(ptr_key)
    if cached is not None:
        return cached
    array = faces_cpu.numpy()
    digest = hashlib.sha1(array.view(np.uint8)).hexdigest()
    cache_key = (int(vertex_count), shape_key, digest)
    cached = MESH_COMPONENT_CACHE.get(cache_key)
    if cached is None:
        cached = compute_mesh_connected_component_ids_cpu(faces_cpu, vertex_count)
        MESH_COMPONENT_CACHE[cache_key] = cached
    MESH_COMPONENT_PTR_CACHE[ptr_key] = cached
    return cached


def compute_mesh_connected_component_ids_cpu(
    faces: torch.Tensor,
    vertex_count: int,
) -> torch.Tensor:
    parent = list(range(int(vertex_count)))
    rank = [0] * int(vertex_count)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if rank[first_root] < rank[second_root]:
            parent[first_root] = second_root
        elif rank[first_root] > rank[second_root]:
            parent[second_root] = first_root
        else:
            parent[second_root] = first_root
            rank[first_root] += 1

    for first, second, third in faces.tolist():
        union(int(first), int(second))
        union(int(second), int(third))

    root_to_component: dict[int, int] = {}
    component_ids = torch.empty(int(vertex_count), dtype=torch.long)
    for vertex_id in range(int(vertex_count)):
        root = find(vertex_id)
        component = root_to_component.get(root)
        if component is None:
            component = len(root_to_component)
            root_to_component[root] = component
        component_ids[vertex_id] = component
    return component_ids


def target_motion_vertex_mask(
    target_delta: torch.Tensor,
    mask_config: Mapping[str, Any],
) -> torch.Tensor:
    if target_delta.dim() == 2:
        target_delta = target_delta.unsqueeze(0)
    if target_delta.dim() != 3 or target_delta.shape[-1] != 3:
        raise ValueError("target_delta must have shape [B, V, 3].")
    threshold = float(
        mask_config.get(
            "target_motion_threshold",
            mask_config.get("active_motion_threshold", 1.0e-4),
        )
    )
    target_norm = torch.linalg.vector_norm(target_delta.detach(), dim=-1)
    if threshold <= 0.0:
        return target_norm > 0.0
    return target_norm >= threshold


def mesh_vertex_mask_parts(
    vertices: torch.Tensor,
    landmarks_3d: Any,
    mask_config: Mapping[str, Any],
    prefix: str,
    faces: Optional[torch.Tensor] = None,
    target_delta: Optional[torch.Tensor] = None,
) -> list[torch.Tensor]:
    parts: list[torch.Tensor] = []
    group_mask = mesh_vertex_group_mask(
        vertices=vertices,
        landmarks_3d=landmarks_3d,
        section_names=mask_config.get(f"{prefix}_sections"),
        object_names=mask_config.get(f"{prefix}_objects"),
        object_name_tokens=mask_config.get(f"{prefix}_object_name_tokens"),
    )
    if group_mask is not None:
        parts.append(group_mask)

    landmark_regions = mask_config.get(
        f"{prefix}_mediapipe_landmark_regions",
        mask_config.get(f"{prefix}_landmark_regions"),
    )
    landmark_mask = mesh_mediapipe_landmark_region_mask(
        vertices=vertices,
        landmarks_3d=landmarks_3d,
        regions=landmark_regions,
    )
    if landmark_mask is not None:
        parts.append(landmark_mask)

    landmark_ovals = mask_config.get(
        f"{prefix}_mediapipe_landmark_ovals",
        mask_config.get(f"{prefix}_landmark_ovals"),
    )
    landmark_oval_mask = mesh_mediapipe_landmark_oval_mask(
        vertices=vertices,
        landmarks_3d=landmarks_3d,
        ovals=landmark_ovals,
    )
    if landmark_oval_mask is not None:
        parts.append(landmark_oval_mask)

    landmark_components = mask_config.get(
        f"{prefix}_mediapipe_landmark_components",
        mask_config.get(f"{prefix}_landmark_components"),
    )
    landmark_component_mask = mesh_mediapipe_landmark_component_mask(
        vertices=vertices,
        landmarks_3d=landmarks_3d,
        components=landmark_components,
        faces=faces,
        target_delta=target_delta,
    )
    if landmark_component_mask is not None:
        parts.append(landmark_component_mask)

    boxes = normalize_mask_box_list(mask_config.get(f"{prefix}_normalized_boxes"))
    for box in boxes:
        parts.append(mesh_normalized_box_mask(vertices, box))
    return parts


def mesh_mediapipe_landmark_region_mask(
    vertices: torch.Tensor,
    landmarks_3d: Any,
    regions: Any,
) -> Optional[torch.Tensor]:
    landmark_regions = normalize_landmark_region_list(regions)
    if not landmark_regions:
        return None
    if not isinstance(landmarks_3d, Mapping):
        return vertices.new_zeros(vertices.shape[:2], dtype=torch.bool)

    mediapipe_ids = landmarks_3d.get("mediapipe_ids")
    vertex_ids = landmarks_3d.get("vertex_ids")
    if not isinstance(mediapipe_ids, torch.Tensor) or not isinstance(
        vertex_ids,
        torch.Tensor,
    ):
        return vertices.new_zeros(vertices.shape[:2], dtype=torch.bool)

    mediapipe_ids = normalized_landmark_id_tensor(
        mediapipe_ids,
        vertices.shape[0],
        vertices.device,
        "landmarks_3d.mediapipe_ids",
    )
    vertex_ids = normalized_landmark_id_tensor(
        vertex_ids,
        vertices.shape[0],
        vertices.device,
        "landmarks_3d.vertex_ids",
    )
    if mediapipe_ids.shape != vertex_ids.shape:
        raise ValueError("landmarks_3d.mediapipe_ids must match vertex_ids.")

    output = vertices.new_zeros(vertices.shape[:2], dtype=torch.bool)
    for region in landmark_regions:
        ids = normalize_int_list(
            region.get(
                "ids",
                region.get("mediapipe_ids", region.get("landmark_ids")),
            ),
            "mediapipe landmark region ids",
        )
        if not ids:
            raise ValueError("mediapipe landmark regions require non-empty ids.")
        radius = float(region.get("radius", region.get("distance", 0.0)) or 0.0)
        padding = float(
            region.get(
                "bbox_padding",
                region.get("box_padding", region.get("padding", 0.0)),
            )
            or 0.0
        )
        if radius <= 0.0 and padding <= 0.0:
            raise ValueError(
                "mediapipe landmark regions require radius or bbox_padding."
            )
        chunk_size = int(region.get("chunk_size", 50000) or 50000)
        if chunk_size < 1:
            raise ValueError("mediapipe landmark region chunk_size must be positive.")

        region_ids = torch.tensor(ids, device=vertices.device, dtype=torch.long)
        for batch_index in range(vertices.shape[0]):
            selected_landmarks = torch.isin(mediapipe_ids[batch_index], region_ids)
            selected_vertex_ids = vertex_ids[batch_index, selected_landmarks]
            valid = (selected_vertex_ids >= 0) & (
                selected_vertex_ids < vertices.shape[1]
            )
            selected_vertex_ids = selected_vertex_ids[valid].unique(sorted=True)
            if selected_vertex_ids.numel() == 0:
                continue
            seeds = vertices[batch_index].index_select(0, selected_vertex_ids)
            region_mask = vertices.new_zeros((vertices.shape[1],), dtype=torch.bool)
            if padding > 0.0:
                lower = seeds.amin(dim=0) - padding
                upper = seeds.amax(dim=0) + padding
                region_mask = region_mask | (
                    (vertices[batch_index] >= lower)
                    & (vertices[batch_index] <= upper)
                ).all(dim=-1)
            if radius > 0.0:
                radius_squared = radius * radius
                chunks: list[torch.Tensor] = []
                for start in range(0, vertices.shape[1], chunk_size):
                    chunk = vertices[batch_index, start : start + chunk_size]
                    distance_squared = (
                        chunk.unsqueeze(1) - seeds.unsqueeze(0)
                    ).square().sum(dim=-1)
                    chunks.append(distance_squared.amin(dim=1) <= radius_squared)
                region_mask = region_mask | torch.cat(chunks, dim=0)
            output[batch_index] = output[batch_index] | region_mask
    return output


def mesh_mediapipe_landmark_oval_mask(
    vertices: torch.Tensor,
    landmarks_3d: Any,
    ovals: Any,
) -> Optional[torch.Tensor]:
    landmark_ovals = normalize_landmark_region_list(ovals)
    if not landmark_ovals:
        return None
    if not isinstance(landmarks_3d, Mapping):
        return vertices.new_zeros(vertices.shape[:2], dtype=torch.bool)

    mediapipe_ids = landmarks_3d.get("mediapipe_ids")
    vertex_ids = landmarks_3d.get("vertex_ids")
    if not isinstance(mediapipe_ids, torch.Tensor) or not isinstance(
        vertex_ids,
        torch.Tensor,
    ):
        return vertices.new_zeros(vertices.shape[:2], dtype=torch.bool)

    mediapipe_ids = normalized_landmark_id_tensor(
        mediapipe_ids,
        vertices.shape[0],
        vertices.device,
        "landmarks_3d.mediapipe_ids",
    )
    vertex_ids = normalized_landmark_id_tensor(
        vertex_ids,
        vertices.shape[0],
        vertices.device,
        "landmarks_3d.vertex_ids",
    )
    if mediapipe_ids.shape != vertex_ids.shape:
        raise ValueError("landmarks_3d.mediapipe_ids must match vertex_ids.")

    output = vertices.new_zeros(vertices.shape[:2], dtype=torch.bool)
    for oval in landmark_ovals:
        ids = normalize_int_list(
            oval.get(
                "ids",
                oval.get("mediapipe_ids", oval.get("landmark_ids")),
            ),
            "mediapipe landmark oval ids",
        )
        if len(ids) < 3:
            raise ValueError("mediapipe landmark ovals require at least three ids.")
        axes = normalize_projection_axes(oval.get("axes", ("x", "z")))
        padding = float(oval.get("padding", oval.get("radius_padding", 0.0)) or 0.0)
        scale = float(oval.get("scale", 1.0) or 1.0)
        min_major_radius = optional_positive_float(oval.get("min_major_radius"))
        min_minor_radius = optional_positive_float(oval.get("min_minor_radius"))
        depth_axis = optional_axis_index(oval.get("depth_axis"))
        depth_padding = float(oval.get("depth_padding", 0.0) or 0.0)
        if depth_padding < 0.0:
            raise ValueError("landmark oval depth_padding must be non-negative.")
        region_ids = torch.tensor(ids, device=vertices.device, dtype=torch.long)

        for batch_index in range(vertices.shape[0]):
            selected_landmarks = torch.isin(mediapipe_ids[batch_index], region_ids)
            selected_vertex_ids = vertex_ids[batch_index, selected_landmarks]
            valid = (selected_vertex_ids >= 0) & (
                selected_vertex_ids < vertices.shape[1]
            )
            selected_vertex_ids = selected_vertex_ids[valid].unique(sorted=True)
            if selected_vertex_ids.numel() < 3:
                continue
            seed_points = vertices[batch_index].index_select(0, selected_vertex_ids)
            seed_2d = seed_points.index_select(
                -1,
                torch.tensor(axes, device=vertices.device, dtype=torch.long),
            )
            center, basis, radii = fit_landmark_oval(seed_2d)
            radii = (radii + padding).clamp_min(1.0e-8) * scale
            if min_major_radius is not None:
                radii[0] = radii[0].clamp_min(min_major_radius)
            if min_minor_radius is not None:
                radii[1] = radii[1].clamp_min(min_minor_radius)

            vertex_2d = vertices[batch_index].index_select(
                -1,
                torch.tensor(axes, device=vertices.device, dtype=torch.long),
            )
            local = (vertex_2d - center) @ basis
            normalized = local / radii.clamp_min(1.0e-8)
            oval_mask = normalized.square().sum(dim=-1) <= 1.0
            if depth_axis is not None:
                seed_depth = seed_points[..., depth_axis]
                lower_depth = seed_depth.amin() - depth_padding
                upper_depth = seed_depth.amax() + depth_padding
                vertex_depth = vertices[batch_index, :, depth_axis]
                oval_mask = oval_mask & (vertex_depth >= lower_depth) & (
                    vertex_depth <= upper_depth
                )
            output[batch_index] = output[batch_index] | oval_mask
    return output


def mesh_mediapipe_landmark_component_mask(
    vertices: torch.Tensor,
    landmarks_3d: Any,
    components: Any,
    faces: Optional[torch.Tensor],
    target_delta: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    landmark_components = normalize_landmark_region_list(components)
    if not landmark_components:
        return None
    if faces is None:
        raise ValueError("mediapipe landmark component masks require mesh faces.")
    if not isinstance(landmarks_3d, Mapping):
        return vertices.new_zeros(vertices.shape[:2], dtype=torch.bool)

    mediapipe_ids = landmarks_3d.get("mediapipe_ids")
    vertex_ids = landmarks_3d.get("vertex_ids")
    if not isinstance(mediapipe_ids, torch.Tensor) or not isinstance(
        vertex_ids,
        torch.Tensor,
    ):
        return vertices.new_zeros(vertices.shape[:2], dtype=torch.bool)

    mediapipe_ids = normalized_landmark_id_tensor(
        mediapipe_ids,
        vertices.shape[0],
        vertices.device,
        "landmarks_3d.mediapipe_ids",
    )
    vertex_ids = normalized_landmark_id_tensor(
        vertex_ids,
        vertices.shape[0],
        vertices.device,
        "landmarks_3d.vertex_ids",
    )
    if mediapipe_ids.shape != vertex_ids.shape:
        raise ValueError("landmarks_3d.mediapipe_ids must match vertex_ids.")

    component_ids = mesh_connected_component_ids(faces, vertices.shape[1]).to(
        device=vertices.device,
    )
    if component_ids.numel() == 0:
        return vertices.new_zeros(vertices.shape[:2], dtype=torch.bool)
    component_count = int(component_ids.max().detach().cpu().item()) + 1
    component_sizes = torch.bincount(
        component_ids.detach().cpu(),
        minlength=component_count,
    ).to(device=vertices.device, dtype=torch.float32)

    if target_delta is not None:
        if target_delta.dim() == 2:
            normalized_target_delta = target_delta.unsqueeze(0)
        else:
            normalized_target_delta = target_delta
        if normalized_target_delta.shape[:2] != vertices.shape[:2]:
            raise ValueError("target_delta must match vertices for component masks.")
        target_norm = torch.linalg.vector_norm(
            normalized_target_delta.detach(),
            dim=-1,
        )
    else:
        target_norm = None

    output = vertices.new_zeros(vertices.shape[:2], dtype=torch.bool)
    for component_config in landmark_components:
        seed_ids = landmark_component_seed_ids(component_config)
        if not seed_ids:
            raise ValueError(
                "mediapipe landmark component masks require ids, seed_ids, "
                "center_ids, or outer_ids."
            )

        min_vertices = int(component_config.get("min_vertices", 1) or 1)
        max_vertices = optional_positive_int(component_config.get("max_vertices"))
        min_seed_vertices = int(component_config.get("min_seed_vertices", 1) or 1)
        min_active_fraction = float(component_config.get("min_active_fraction", 0.0))
        min_active_vertices = int(component_config.get("min_active_vertices", 0) or 0)
        threshold = float(
            component_config.get(
                "target_motion_threshold",
                component_config.get("active_motion_threshold", 1.0e-4),
            )
        )
        if min_active_fraction > 0.0 or min_active_vertices > 0:
            if target_norm is None:
                raise ValueError(
                    "mediapipe landmark component active gates require target_delta."
                )

        region_ids = torch.tensor(seed_ids, device=vertices.device, dtype=torch.long)
        for batch_index in range(vertices.shape[0]):
            selected_landmarks = torch.isin(mediapipe_ids[batch_index], region_ids)
            selected_vertex_ids = vertex_ids[batch_index, selected_landmarks]
            valid = (selected_vertex_ids >= 0) & (
                selected_vertex_ids < vertices.shape[1]
            )
            selected_vertex_ids = selected_vertex_ids[valid].unique(sorted=True)
            if selected_vertex_ids.numel() == 0:
                continue

            seed_component_ids = component_ids.index_select(0, selected_vertex_ids)
            seed_counts = torch.bincount(
                seed_component_ids,
                minlength=component_count,
            ).to(dtype=torch.float32)
            selected_components = seed_counts >= float(min_seed_vertices)
            selected_components = selected_components & (
                component_sizes >= float(min_vertices)
            )
            if max_vertices is not None:
                selected_components = selected_components & (
                    component_sizes <= float(max_vertices)
                )
            if target_norm is not None and (
                min_active_fraction > 0.0 or min_active_vertices > 0
            ):
                active = target_norm[batch_index]
                if threshold <= 0.0:
                    active = active > 0.0
                else:
                    active = active >= threshold
                active_counts = torch.bincount(
                    component_ids.detach().cpu(),
                    weights=active.detach().cpu().to(torch.float32),
                    minlength=component_count,
                ).to(device=vertices.device)
                active_fraction = active_counts / component_sizes.clamp_min(1.0)
                selected_components = selected_components & (
                    active_fraction >= min_active_fraction
                )
                selected_components = selected_components & (
                    active_counts >= float(min_active_vertices)
                )

            output[batch_index] = output[batch_index] | selected_components.index_select(
                0,
                component_ids,
            )
    return output


def landmark_component_seed_ids(component_config: Mapping[str, Any]) -> list[int]:
    values: list[int] = []
    for key in (
        "ids",
        "seed_ids",
        "mediapipe_ids",
        "landmark_ids",
        "center_ids",
        "outer_ids",
        "outward_ids",
    ):
        values.extend(normalize_int_list(component_config.get(key), key))
    return sorted(set(values))


def fit_landmark_oval(points_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if points_2d.dim() != 2 or points_2d.shape[-1] != 2:
        raise ValueError("points_2d must have shape [N, 2].")
    if points_2d.shape[0] < 3:
        raise ValueError("At least three points are required to fit an oval.")
    center = points_2d.mean(dim=0)
    centered = points_2d - center
    covariance = centered.transpose(0, 1) @ centered / max(points_2d.shape[0] - 1, 1)
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    except RuntimeError:
        eigenvectors = torch.eye(2, device=points_2d.device, dtype=points_2d.dtype)
        local = centered
        radii = local.abs().amax(dim=0)
        return center, eigenvectors, radii
    order = torch.argsort(eigenvalues, descending=True)
    basis = eigenvectors.index_select(1, order)
    local = centered @ basis
    radii = local.abs().amax(dim=0)
    return center, basis, radii


def normalize_projection_axes(value: Any) -> tuple[int, int]:
    if value in (None, ""):
        names = ("x", "z")
    elif isinstance(value, str):
        names = tuple(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, Sequence):
        names = tuple(str(part).strip() for part in value)
    else:
        raise ValueError("landmark oval axes must be a two-item sequence.")
    if len(names) != 2:
        raise ValueError("landmark oval axes must contain exactly two axes.")
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    try:
        return tuple(axis_to_index[name.lower()] for name in names)  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError("landmark oval axes must be chosen from x, y, z.") from exc


def optional_axis_index(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    name = str(value).strip().lower()
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    if name not in axis_to_index:
        raise ValueError("landmark oval depth_axis must be one of x, y, z.")
    return axis_to_index[name]


def optional_positive_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    parsed = float(value)
    if parsed <= 0.0:
        return None
    return parsed


def normalized_landmark_id_tensor(
    value: torch.Tensor,
    batch_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    ids = value.to(device=device, dtype=torch.long)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    elif ids.dim() != 2:
        raise ValueError(f"{name} must have shape [L] or [B, L].")
    if ids.shape[0] == 1 and batch_size > 1:
        ids = ids.expand(batch_size, -1)
    if ids.shape[0] != batch_size:
        raise ValueError(f"{name} batch size must match vertices.")
    return ids


def normalize_landmark_region_list(value: Any) -> list[Mapping[str, Any]]:
    if value in (None, ""):
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("mediapipe landmark regions must be a list of mappings.")
    regions: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("mediapipe landmark regions must be mappings.")
        regions.append(item)
    return regions


def normalize_int_list(value: Any, name: str) -> list[int]:
    if value in (None, ""):
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",")]
    elif isinstance(value, Sequence):
        raw_values = value
    else:
        raw_values = [value]
    return [int(item) for item in raw_values if str(item).strip()]


def mesh_vertex_group_mask(
    vertices: torch.Tensor,
    landmarks_3d: Any,
    section_names: Any = None,
    object_names: Any = None,
    object_name_tokens: Any = None,
) -> Optional[torch.Tensor]:
    sections = normalize_string_set(section_names)
    objects = normalize_string_set(object_names)
    tokens = tuple(sorted(normalize_string_set(object_name_tokens)))
    if not sections and not objects and not tokens:
        return None

    vertex_groups = mesh_record_vertex_groups(landmarks_3d)
    selected_ids: list[torch.Tensor] = []

    section_groups = vertex_groups.get("sections")
    if isinstance(section_groups, Mapping):
        for name, vertex_ids in section_groups.items():
            if str(name).lower() in sections and isinstance(vertex_ids, torch.Tensor):
                selected_ids.append(vertex_ids.detach().long().flatten())

    object_groups = vertex_groups.get("objects")
    if isinstance(object_groups, Mapping):
        for name, vertex_ids in object_groups.items():
            normalized_name = str(name).lower()
            exact_match = normalized_name in objects
            token_match = any(token in normalized_name for token in tokens)
            if (exact_match or token_match) and isinstance(vertex_ids, torch.Tensor):
                selected_ids.append(vertex_ids.detach().long().flatten())

    if not selected_ids:
        return vertices.new_zeros(vertices.shape[:2], dtype=torch.bool)

    ids = torch.cat(selected_ids, dim=0).to(device=vertices.device, dtype=torch.long)
    valid = (ids >= 0) & (ids < vertices.shape[1])
    ids = ids[valid].unique(sorted=True)
    mask = vertices.new_zeros(vertices.shape[:2], dtype=torch.bool)
    if ids.numel() > 0:
        mask[:, ids] = True
    return mask


def mesh_record_vertex_groups(landmarks_3d: Any) -> Mapping[str, Any]:
    if not isinstance(landmarks_3d, Mapping):
        return {}
    vertex_groups = landmarks_3d.get("vertex_groups")
    if not isinstance(vertex_groups, Mapping):
        return {}
    return vertex_groups


def normalize_string_set(value: Any) -> set[str]:
    if value in (None, ""):
        return set()
    if isinstance(value, (str, bytes)):
        raw_values: Sequence[Any] = [
            value.decode() if isinstance(value, bytes) else value
        ]
    elif isinstance(value, Sequence):
        raw_values = value
    else:
        raw_values = [value]
    return {str(item).strip().lower() for item in raw_values if str(item).strip()}


def normalize_mask_box_list(value: Any) -> list[Mapping[str, Any]]:
    if value in (None, ""):
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("normalized box masks must be a list of mappings.")
    boxes: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("normalized box masks must be mappings.")
        boxes.append(item)
    return boxes


def mesh_normalized_box_mask(
    vertices: torch.Tensor,
    box: Mapping[str, Any],
) -> torch.Tensor:
    bounds_min = vertices.amin(dim=1, keepdim=True)
    bounds_max = vertices.amax(dim=1, keepdim=True)
    normalized = (vertices - bounds_min) / (bounds_max - bounds_min).clamp_min(1.0e-8)
    mask = torch.ones(vertices.shape[:2], device=vertices.device, dtype=torch.bool)
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        min_key = f"{axis_name}_min"
        max_key = f"{axis_name}_max"
        lower = normalize_roi_fraction(box.get(min_key, 0.0))
        upper = normalize_roi_fraction(box.get(max_key, 1.0))
        if upper < lower:
            lower, upper = upper, lower
        values = normalized[..., axis_index]
        mask = mask & (values >= lower) & (values <= upper)
    return mask






@torch.no_grad()
def active_motion_weight_metrics(
    target: torch.Tensor,
    config: Mapping[str, Any],
) -> dict[str, float]:
    weights = active_motion_vertex_weights(target, config)
    if weights is None:
        return {}
    return {
        "mesh_3d_active_motion_weight_mean": float(weights.mean().detach().cpu()),
        "mesh_3d_active_motion_vertex_ratio": float(
            (weights > 1.0).to(dtype=target.dtype).mean().detach().cpu()
        ),
    }


def compute_mesh_depth_loss(
    vertices: torch.Tensor,
    faces: torch.Tensor | None,
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    depth_weight = float(config.get("depth_weight", 0.0))
    if depth_weight <= 0.0:
        return predicted_delta.new_zeros(()), {}
    if faces is None:
        raise ValueError("faces are required when mesh depth loss is enabled.")

    image_size = normalize_image_size(config.get("depth_image_size", (96, 96)))
    depth_loss, details = depth_difference_l1_loss(
        neutral_vertices=vertices,
        faces=faces,
        pred_delta=predicted_delta,
        target_delta=target_delta,
        image_size=image_size,
        up_axis=str(config.get("depth_up_axis", "z")),
        return_details=True,
    )
    weighted_depth_loss = depth_loss * depth_weight

    raw_mask = details["mask"] if "mask" in details else details["valid"]
    mask = raw_mask.to(dtype=predicted_delta.dtype)
    mask_sum = mask.sum().clamp_min(1.0)
    valid = details.get("valid", raw_mask).to(dtype=predicted_delta.dtype)
    pred_depth_delta = details["pred_depth_delta"]
    target_depth_delta = details["target_depth_delta"]
    metrics = {
        "mesh_3d_depth_loss": float(depth_loss.detach().cpu()),
        "mesh_3d_depth_global_loss": float(details["global_loss"].detach().cpu()),
        "mesh_3d_depth_weighted_loss": float(weighted_depth_loss.detach().cpu()),
        "mesh_3d_depth_mask_ratio": float((mask > 0).float().mean().detach().cpu()),
        "mesh_3d_depth_valid_ratio": float((valid > 0).float().mean().detach().cpu()),
        "mesh_3d_depth_pred_delta_abs_mean": float(
            (pred_depth_delta.abs() * mask).sum().div(mask_sum).detach().cpu()
        ),
        "mesh_3d_depth_target_delta_abs_mean": float(
            (target_depth_delta.abs() * mask).sum().div(mask_sum).detach().cpu()
        ),
    }
    return weighted_depth_loss, metrics


def compute_mesh_zero_delta_anchor_loss(
    predicted_delta: torch.Tensor,
    vertices: torch.Tensor,
    landmarks_3d: Any,
    target_vertex_mask: Optional[torch.Tensor],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    anchor_config = config.get("zero_delta_vertex_mask")
    if not isinstance(anchor_config, Mapping):
        return predicted_delta.new_zeros(()), {}
    if not bool(anchor_config.get("enabled", True)):
        return predicted_delta.new_zeros(()), {}

    weight = float(anchor_config.get("weight", 0.0))
    if weight <= 0.0:
        return predicted_delta.new_zeros(()), {}

    if bool(anchor_config.get("invert_target_vertex_mask", False)):
        if target_vertex_mask is None:
            raise ValueError(
                "loss.mesh.zero_delta_vertex_mask.invert_target_vertex_mask "
                "requires loss.mesh.target_vertex_mask."
            )
        mask = ~normalized_vertex_mask(target_vertex_mask, predicted_delta)
    else:
        mask = build_mesh_vertex_mask(
            vertices=vertices,
            landmarks_3d=landmarks_3d,
            mask_config=anchor_config,
            name="loss.mesh.zero_delta_vertex_mask",
        )
        mask = normalized_vertex_mask(mask, predicted_delta)

    if not bool(mask.any().detach().cpu()):
        raise ValueError("loss.mesh.zero_delta_vertex_mask selected zero vertices.")

    zero_target = torch.zeros_like(predicted_delta)
    error = elementwise_loss(predicted_delta, zero_target, config)
    normalizer = mask.to(dtype=predicted_delta.dtype).sum().clamp_min(1.0e-8)
    raw = (error * mask.unsqueeze(-1)).sum() / (normalizer * error.shape[-1])
    weighted = raw * weight
    return weighted, {
        "mesh_3d_zero_delta_anchor_loss": float(raw.detach().cpu()),
        "mesh_3d_zero_delta_anchor_weighted_loss": float(weighted.detach().cpu()),
        "mesh_3d_zero_delta_anchor_vertex_ratio": float(
            mask.to(dtype=predicted_delta.dtype).mean().detach().cpu()
        ),
    }


def compute_mesh_geometry_regularization(
    vertices: torch.Tensor,
    predicted_delta: torch.Tensor,
    faces: torch.Tensor | None,
    config: Mapping[str, Any],
    vertex_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not isinstance(config, Mapping) or not bool(config.get("enabled", False)):
        return predicted_delta.new_zeros(()), {}
    if faces is None:
        raise ValueError("faces are required when loss.mesh.geometry is enabled.")

    if predicted_delta.dim() == 4:
        if predicted_delta.shape[1] != 1:
            raise ValueError(
                "Mesh geometry regularization expects one pose per mesh sample."
            )
        predicted_delta = predicted_delta[:, 0]
    if predicted_delta.shape != vertices.shape:
        raise ValueError("predicted_delta must match vertices shape.")

    deformed_vertices = vertices + predicted_delta
    loss = predicted_delta.new_zeros(())
    metrics: dict[str, float] = {}
    geometry_vertex_mask = mesh_geometry_regularization_vertex_mask(
        vertex_mask=vertex_mask,
        reference=predicted_delta,
        config=config,
    )
    mask_mode = str(config.get("target_vertex_mask_mode", "touch")).lower()
    if geometry_vertex_mask is not None:
        metrics["mesh_3d_geom_mask_vertex_ratio"] = float(
            geometry_vertex_mask.to(dtype=predicted_delta.dtype).mean().detach().cpu()
        )

    l2_weight = float(config.get("displacement_l2_weight", 0.0))
    if l2_weight > 0.0:
        raw = predicted_delta.square().mean()
        weighted = raw * l2_weight
        loss = loss + weighted
        metrics["mesh_3d_geom_displacement_l2"] = float(raw.detach().cpu())
        metrics["mesh_3d_geom_displacement_l2_weighted"] = float(
            weighted.detach().cpu()
        )

    max_displacement = config.get("max_displacement")
    magnitude_weight = float(config.get("magnitude_clamp_weight", 0.0))
    if max_displacement is not None:
        delta_norm = torch.linalg.vector_norm(predicted_delta, dim=-1)
        metrics["mesh_3d_geom_delta_norm_mean"] = float(
            delta_norm.mean().detach().cpu()
        )
        metrics["mesh_3d_geom_delta_norm_max"] = float(
            delta_norm.amax().detach().cpu()
        )
        if magnitude_weight > 0.0:
            excess = (delta_norm - float(max_displacement)).clamp_min(0.0)
            raw = excess.square().mean()
            weighted = raw * magnitude_weight
            loss = loss + weighted
            metrics["mesh_3d_geom_magnitude_clamp"] = float(raw.detach().cpu())
            metrics["mesh_3d_geom_magnitude_clamp_weighted"] = float(
                weighted.detach().cpu()
            )

    edges = selected_face_edges(
        faces,
        geometry_vertex_mask,
        mask_mode,
        max_edges=optional_positive_int(config.get("max_edges")),
    )
    if edges.numel() > 0:
        edges = edges.to(device=vertices.device, dtype=torch.long)
        metrics["mesh_3d_geom_edge_count"] = float(edges.shape[0])
        laplacian_weight = float(config.get("laplacian_weight", 0.0))
        if laplacian_weight > 0.0:
            raw = edge_delta_smoothness_loss(predicted_delta, edges)
            weighted = raw * laplacian_weight
            loss = loss + weighted
            metrics["mesh_3d_geom_laplacian"] = float(raw.detach().cpu())
            metrics["mesh_3d_geom_laplacian_weighted"] = float(
                weighted.detach().cpu()
            )

        edge_weight = float(config.get("edge_length_weight", 0.0))
        if edge_weight > 0.0:
            raw = edge_length_preservation_loss(
                vertices=vertices,
                deformed_vertices=deformed_vertices,
                edges=edges,
                relative=bool(config.get("edge_length_relative", True)),
                min_edge_length=float(config.get("min_edge_length", 1.0e-4)),
            )
            weighted = raw * edge_weight
            loss = loss + weighted
            metrics["mesh_3d_geom_edge_length"] = float(raw.detach().cpu())
            metrics["mesh_3d_geom_edge_length_weighted"] = float(
                weighted.detach().cpu()
            )

    normal_weight = float(config.get("normal_consistency_weight", 0.0))
    if normal_weight > 0.0:
        face_sample = selected_faces(
            faces,
            geometry_vertex_mask,
            mask_mode,
            max_faces=optional_positive_int(config.get("max_faces")),
        )
        if face_sample.numel() > 0:
            face_sample = face_sample.to(device=vertices.device, dtype=torch.long)
            metrics["mesh_3d_geom_face_count"] = float(face_sample.shape[0])
            raw = face_normal_consistency_loss(vertices, deformed_vertices, face_sample)
            weighted = raw * normal_weight
            loss = loss + weighted
            metrics["mesh_3d_geom_normal_consistency"] = float(raw.detach().cpu())
            metrics["mesh_3d_geom_normal_consistency_weighted"] = float(
                weighted.detach().cpu()
            )

    metrics["mesh_3d_geom_loss"] = float(loss.detach().cpu())
    return loss, metrics


def compute_mesh_rigid_motion_regularization(
    vertices: torch.Tensor,
    predicted_delta: torch.Tensor,
    faces: torch.Tensor | None,
    vertex_mask: Optional[torch.Tensor],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    if not isinstance(config, Mapping) or not bool(config.get("enabled", False)):
        return predicted_delta.new_zeros(()), {}
    weight = float(config.get("weight", 0.0))
    if weight <= 0.0:
        return predicted_delta.new_zeros(()), {}
    if faces is None:
        raise ValueError("loss.mesh.rigid_motion requires mesh faces.")
    if predicted_delta.dim() == 4:
        if predicted_delta.shape[1] != 1:
            raise ValueError(
                "Mesh rigid-motion regularization expects one pose per mesh sample."
            )
        predicted_delta = predicted_delta[:, 0]
    if predicted_delta.shape != vertices.shape:
        raise ValueError("predicted_delta must match vertices shape.")

    if vertex_mask is None:
        if bool(config.get("require_vertex_mask", True)):
            raise ValueError(
                "loss.mesh.rigid_motion requires loss.mesh.target_vertex_mask."
            )
        mask = torch.ones(
            predicted_delta.shape[:2],
            device=predicted_delta.device,
            dtype=torch.bool,
        )
    else:
        mask = normalized_vertex_mask(vertex_mask, predicted_delta)

    component_ids = mesh_connected_component_ids(faces, predicted_delta.shape[1]).to(
        device=predicted_delta.device,
    )
    if component_ids.numel() == 0:
        return predicted_delta.new_zeros(()), {}
    min_vertices = int(config.get("min_vertices", 8) or 8)
    detach_fit = bool(config.get("detach_fit", True))
    component_losses: list[torch.Tensor] = []
    selected_vertices = 0
    selected_components = 0

    for batch_index in range(predicted_delta.shape[0]):
        component_values = torch.unique(component_ids[mask[batch_index]])
        for component_id in component_values.tolist():
            selected = mask[batch_index] & (component_ids == int(component_id))
            count = int(selected.sum().detach().cpu())
            if count < min_vertices:
                continue
            source = vertices[batch_index, selected]
            delta = predicted_delta[batch_index, selected]
            rigid_delta = rigid_fit_delta(source, source + delta)
            if detach_fit:
                rigid_delta = rigid_delta.detach()
            component_losses.append((delta - rigid_delta).square().mean())
            selected_vertices += count
            selected_components += 1

    if not component_losses:
        return predicted_delta.new_zeros(()), {
            "mesh_3d_rigid_motion_component_count": 0.0,
            "mesh_3d_rigid_motion_vertex_count": 0.0,
        }

    raw = torch.stack(component_losses).mean()
    weighted = raw * weight
    return weighted, {
        "mesh_3d_rigid_motion_loss": float(raw.detach().cpu()),
        "mesh_3d_rigid_motion_weighted_loss": float(weighted.detach().cpu()),
        "mesh_3d_rigid_motion_component_count": float(selected_components),
        "mesh_3d_rigid_motion_vertex_count": float(selected_vertices),
    }


def mesh_geometry_regularization_vertex_mask(
    vertex_mask: Optional[torch.Tensor],
    reference: torch.Tensor,
    config: Mapping[str, Any],
) -> Optional[torch.Tensor]:
    if not bool(config.get("target_vertex_mask", False)):
        return None
    if vertex_mask is None:
        if bool(config.get("require_target_vertex_mask", True)):
            raise ValueError(
                "loss.mesh.geometry.target_vertex_mask requires "
                "loss.mesh.target_vertex_mask."
            )
        return None
    mask = normalized_vertex_mask(vertex_mask, reference)
    if mask.shape[0] == 1:
        return mask[0]
    first = mask[0]
    if bool(torch.equal(mask, first.unsqueeze(0).expand_as(mask))):
        return first
    return mask.any(dim=0)


def compute_mesh_landmark_loss(
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
    landmarks_3d: Any,
    config: Mapping[str, Any],
    *,
    gradient_balance_reference_loss: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    landmark_weight = float(config.get("landmark_weight", 0.0))
    if landmark_weight <= 0.0:
        return predicted_delta.new_zeros(()), {}
    require_landmarks = bool(config.get("landmark_require_nonempty", True))
    if not isinstance(landmarks_3d, Mapping):
        if not require_landmarks:
            return predicted_delta.new_zeros(()), {
                "mesh_3d_landmark_count": 0.0,
                "mesh_3d_landmark_skipped_empty": 1.0,
            }
        raise ValueError(
            "loss.mesh.landmark_weight is enabled, but this mesh sample does "
            "not include landmarks_3d. Generate or provide a MediaPipe "
            "landmark mapping JSON before training with landmark loss."
        )

    vertex_ids = landmarks_3d.get("vertex_ids")
    if not isinstance(vertex_ids, torch.Tensor) or vertex_ids.numel() == 0:
        if not require_landmarks:
            return predicted_delta.new_zeros(()), {
                "mesh_3d_landmark_count": 0.0,
                "mesh_3d_landmark_skipped_empty": 1.0,
            }
        raise ValueError(
            "loss.mesh.landmark_weight is enabled, but landmarks_3d.vertex_ids "
            "is empty. The MediaPipe landmark cache has zero supervised "
            "vertices; create a mapping JSON or set "
            "loss.mesh.landmark_weight: 0.0."
        )

    vertex_ids = vertex_ids.to(device=predicted_delta.device, dtype=torch.long)
    predicted_landmark_delta = predicted_delta.index_select(1, vertex_ids)

    target_landmark_delta = landmarks_3d.get("target_delta")
    if isinstance(target_landmark_delta, torch.Tensor):
        target_landmark_delta = target_landmark_delta.to(
            device=predicted_delta.device,
            dtype=predicted_delta.dtype,
        )
        if target_landmark_delta.dim() == 2:
            target_landmark_delta = target_landmark_delta.unsqueeze(0)
    else:
        target_landmark_delta = target_delta.index_select(1, vertex_ids)

    if target_landmark_delta.shape != predicted_landmark_delta.shape:
        raise ValueError(
            "landmarks_3d.target_delta must match selected landmark delta shape."
        )

    raw_loss = elementwise_loss(
        predicted_landmark_delta,
        target_landmark_delta,
        config,
    ).mean()
    normalized_raw_loss = raw_loss
    if bool(config.get("landmark_normalize_by_target", False)):
        active = landmark_active_motion_mask(target_landmark_delta, config)
        normalized_raw_loss = raw_loss / target_loss_normalizer(
            target_landmark_delta,
            active,
            config,
        )
    active_landmark_loss = landmark_active_motion_loss(
        predicted_landmark_delta,
        target_landmark_delta,
        config,
    )
    active_landmark_weight = float(
        config.get("landmark_active_motion_weight", 0.0)
    )
    weighted_loss = normalized_raw_loss * landmark_weight
    effective_active_landmark_weight = predicted_delta.new_tensor(
        active_landmark_weight
    )
    active_weight_metrics: dict[str, float] = {}
    if active_landmark_loss is not None and active_landmark_weight > 0.0:
        balance_reference_loss = gradient_balance_reference_loss
        if balance_reference_loss is None or not bool(
            getattr(balance_reference_loss, "requires_grad", False)
        ):
            balance_reference_loss = weighted_loss
        (
            effective_active_landmark_weight,
            active_weight_metrics,
        ) = landmark_active_motion_effective_weight(
            reference_loss=balance_reference_loss,
            active_landmark_loss=active_landmark_loss,
            predicted_delta=predicted_delta,
            max_weight=active_landmark_weight,
            config=config,
        )
        weighted_loss = (
            weighted_loss + active_landmark_loss * effective_active_landmark_weight
        )
    target_norm = torch.linalg.vector_norm(target_landmark_delta.detach(), dim=-1)
    predicted_norm = torch.linalg.vector_norm(
        predicted_landmark_delta.detach(),
        dim=-1,
    )
    metrics = {
        "mesh_3d_landmark_count": float(vertex_ids.numel()),
        "mesh_3d_landmark_loss": float(raw_loss.detach().cpu()),
        "mesh_3d_landmark_normalized_loss": float(
            normalized_raw_loss.detach().cpu()
        ),
        "mesh_3d_landmark_weighted_loss": float(weighted_loss.detach().cpu()),
        "mesh_3d_landmark_target_norm_mean": float(target_norm.mean().cpu()),
        "mesh_3d_landmark_pred_norm_mean": float(predicted_norm.mean().cpu()),
    }
    if active_landmark_loss is not None:
        active = landmark_active_motion_mask(target_landmark_delta, config)
        metrics["mesh_3d_landmark_active_count"] = float(
            active.to(dtype=predicted_delta.dtype).sum().detach().cpu()
        )
        metrics["mesh_3d_landmark_active_loss"] = float(
            active_landmark_loss.detach().cpu()
        )
        metrics["mesh_3d_landmark_active_weighted_loss"] = float(
            (active_landmark_loss * effective_active_landmark_weight).detach().cpu()
        )
        metrics["mesh_3d_landmark_active_configured_weight"] = float(
            active_landmark_weight
        )
        metrics["mesh_3d_landmark_active_effective_weight"] = float(
            effective_active_landmark_weight.detach().cpu()
        )
        metrics.update(active_weight_metrics)
    return weighted_loss, metrics


def landmark_active_motion_effective_weight(
    *,
    reference_loss: torch.Tensor,
    active_landmark_loss: torch.Tensor,
    predicted_delta: torch.Tensor,
    max_weight: float,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    weight = predicted_delta.new_tensor(float(max_weight))
    metrics: dict[str, float] = {}
    if bool(config.get("landmark_active_motion_adaptive_balance", False)):
        (
            adaptive_weight,
            adaptive_metrics,
        ) = landmark_active_motion_adaptive_balance_weight(
            reference_loss=reference_loss,
            active_landmark_loss=active_landmark_loss,
            predicted_delta=predicted_delta,
            max_weight=max_weight,
            config=config,
        )
        weight = torch.minimum(weight, adaptive_weight)
        metrics.update(adaptive_metrics)

    if not bool(config.get("landmark_active_motion_gradient_balance", False)):
        return weight, metrics

    metrics["mesh_3d_landmark_active_gradient_balance_enabled"] = 1.0
    if (
        not torch.is_grad_enabled()
        or not predicted_delta.requires_grad
        or not reference_loss.requires_grad
        or not active_landmark_loss.requires_grad
    ):
        metrics["mesh_3d_landmark_active_gradient_balance_skipped_no_grad"] = 1.0
        return weight, metrics

    reference_norm = scalar_loss_gradient_norm(reference_loss, predicted_delta)
    active_norm = scalar_loss_gradient_norm(active_landmark_loss, predicted_delta)
    if reference_norm is None or active_norm is None:
        metrics["mesh_3d_landmark_active_gradient_balance_skipped_unused"] = 1.0
        return weight, metrics

    eps = float(
        config.get("landmark_active_motion_gradient_balance_epsilon", 1.0e-8)
    )
    target_ratio = max(
        0.0,
        float(
            config.get(
                "landmark_active_motion_gradient_balance_target_ratio",
                1.0,
            )
        ),
    )
    min_weight = max(
        0.0,
        float(
            config.get(
                "landmark_active_motion_gradient_balance_min_weight",
                0.0,
            )
        ),
    )
    min_weight = min(min_weight, float(weight.detach().cpu()))

    metrics["mesh_3d_landmark_active_gradient_balance_reference_norm"] = float(
        reference_norm.detach().cpu()
    )
    metrics["mesh_3d_landmark_active_gradient_balance_active_norm"] = float(
        active_norm.detach().cpu()
    )

    if not bool(torch.isfinite(reference_norm).detach().cpu()) or not bool(
        torch.isfinite(active_norm).detach().cpu()
    ):
        metrics["mesh_3d_landmark_active_gradient_balance_skipped_nonfinite"] = 1.0
        return weight, metrics
    if float(active_norm.detach().cpu()) <= eps:
        metrics["mesh_3d_landmark_active_gradient_balance_skipped_zero_active"] = 1.0
        return weight, metrics

    balanced_weight = target_ratio * reference_norm / active_norm.clamp_min(eps)
    balanced_weight = balanced_weight.clamp(
        min=min_weight,
        max=float(weight.detach().cpu()),
    )
    balanced_weight = balanced_weight.detach()
    metrics["mesh_3d_landmark_active_gradient_balance_scale"] = float(
        (balanced_weight / predicted_delta.new_tensor(float(max_weight)).clamp_min(eps))
        .detach()
        .cpu()
    )
    return balanced_weight, metrics


def landmark_active_motion_adaptive_balance_weight(
    *,
    reference_loss: torch.Tensor,
    active_landmark_loss: torch.Tensor,
    predicted_delta: torch.Tensor,
    max_weight: float,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    weight = predicted_delta.new_tensor(float(max_weight))
    metrics: dict[str, float] = {
        "mesh_3d_landmark_active_adaptive_balance_enabled": 1.0,
    }
    eps = float(
        config.get(
            "landmark_active_motion_adaptive_balance_epsilon",
            config.get("landmark_active_motion_gradient_balance_epsilon", 1.0e-8),
        )
    )
    target_ratio = max(
        0.0,
        float(
            config.get(
                "landmark_active_motion_adaptive_balance_target_ratio",
                5.0,
            )
        ),
    )
    min_weight = max(
        0.0,
        float(
            config.get(
                "landmark_active_motion_adaptive_balance_min_weight",
                0.0,
            )
        ),
    )
    min_weight = min(min_weight, float(max_weight))

    reference_value = reference_loss.detach()
    active_value = active_landmark_loss.detach()
    metrics["mesh_3d_landmark_active_adaptive_balance_reference_loss"] = float(
        reference_value.cpu()
    )
    metrics["mesh_3d_landmark_active_adaptive_balance_active_loss"] = float(
        active_value.cpu()
    )
    metrics["mesh_3d_landmark_active_adaptive_balance_target_ratio"] = float(
        target_ratio
    )

    if not bool(torch.isfinite(reference_value).detach().cpu()) or not bool(
        torch.isfinite(active_value).detach().cpu()
    ):
        metrics["mesh_3d_landmark_active_adaptive_balance_skipped_nonfinite"] = 1.0
        return weight, metrics
    if float(active_value.detach().cpu()) <= eps:
        metrics["mesh_3d_landmark_active_adaptive_balance_skipped_zero_active"] = 1.0
        return weight, metrics

    cap = target_ratio * reference_value.clamp_min(0.0) / active_value.clamp_min(eps)
    cap = cap.clamp(min=min_weight, max=float(max_weight)).to(
        device=predicted_delta.device,
        dtype=predicted_delta.dtype,
    )
    cap = cap.detach()
    metrics["mesh_3d_landmark_active_adaptive_balance_scale"] = float(
        (cap / weight.clamp_min(eps)).detach().cpu()
    )
    return cap, metrics










@torch.no_grad()
def mesh_loss_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    loss: torch.Tensor,
    config: Mapping[str, Any],
    vertex_mask: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    error = predicted - target
    target_norm = torch.linalg.vector_norm(target, dim=-1)
    predicted_norm = torch.linalg.vector_norm(predicted, dim=-1)
    target_zero_mse = target.square().mean()
    raw_mse = error.square().mean()
    raw_mae = error.abs().mean()
    false_positive_target_threshold = float(
        config.get("false_positive_target_threshold", 1.0e-3)
    )
    false_positive_pred_threshold = float(
        config.get("false_positive_pred_threshold", 1.0e-2)
    )
    false_positive = (
        (target_norm <= false_positive_target_threshold)
        & (predicted_norm > false_positive_pred_threshold)
    )

    metrics = {
        "mesh_3d_loss": float(loss.detach().cpu()),
        "mesh_3d_mae": float(raw_mae.detach().cpu()),
        "mesh_3d_mse": float(raw_mse.detach().cpu()),
        "mesh_3d_zero_mse": float(target_zero_mse.detach().cpu()),
        "mesh_3d_pred_norm_mean": float(predicted_norm.mean().detach().cpu()),
        "mesh_3d_target_norm_mean": float(target_norm.mean().detach().cpu()),
    }
    if float(target_zero_mse.detach().cpu()) > 0.0:
        ratio = raw_mse / target_zero_mse.clamp_min(1.0e-12)
        metrics["mesh_3d_vs_neutral_ratio"] = float(ratio.detach().cpu())
    if vertex_mask is not None:
        mask_float = normalized_vertex_weights(vertex_mask, predicted)
        mask_sum = mask_float.sum().clamp_min(1.0)
        masked_mse = (error.square().sum(dim=-1) * mask_float).sum() / (
            mask_sum * error.shape[-1]
        )
        metrics["mesh_3d_target_vertex_mask_ratio"] = float(
            mask_float.mean().detach().cpu()
        )
        metrics["mesh_3d_target_vertex_mask_count"] = float(
            mask_float.sum().detach().cpu()
        )
        metrics["mesh_3d_target_vertex_mask_mse"] = float(
            masked_mse.detach().cpu()
        )
    false_positive_float = false_positive.to(dtype=predicted.dtype)
    false_positive_count = false_positive_float.sum()
    metrics["mesh_3d_false_positive_ratio"] = float(
        false_positive_float.mean().detach().cpu()
    )
    metrics["mesh_3d_false_positive_count"] = float(
        false_positive_count.detach().cpu()
    )
    if bool(false_positive.any()):
        metrics["mesh_3d_false_positive_pred_norm_mean"] = float(
            (predicted_norm * false_positive_float)
            .sum()
            .div(false_positive_count.clamp_min(1.0))
            .detach()
            .cpu()
        )
    normalized_region_loss = target_normalized_region_loss(
        elementwise_loss(predicted, target, config),
        target,
        config,
        include_weight=False,
        vertex_mask=vertex_mask,
    )
    if normalized_region_loss is not None:
        active = active_motion_mask(target, config)
        if vertex_mask is not None:
            active = active & (normalized_vertex_weights(vertex_mask, target) > 0.0)
        metrics["mesh_3d_target_normalized_region_loss"] = float(
            normalized_region_loss.detach().cpu()
        )
        metrics["mesh_3d_target_motion_scale"] = float(
            target_motion_scale(target, active, config).detach().cpu()
        )
        metrics["mesh_3d_target_active_vertex_count"] = float(
            active.to(dtype=predicted.dtype).sum().detach().cpu()
        )
    return metrics


def prepare_2d_target(
    displacement: torch.Tensor,
    mask: Optional[torch.Tensor],
    image_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    flow = displacement.to(device=device, dtype=dtype)
    if flow.dim() == 3:
        flow = flow.unsqueeze(0)
    if flow.dim() != 4 or flow.shape[1] != 2:
        raise ValueError("2D displacement target must have shape [2, H, W].")

    old_height, old_width = flow.shape[-2:]
    new_height, new_width = image_size
    if (old_height, old_width) != (new_height, new_width):
        flow = F.interpolate(
            flow,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )
        scale = flow.new_tensor(
            (new_width / old_width, new_height / old_height)
        ).view(1, 2, 1, 1)
        flow = flow * scale

    if mask is None:
        weight = flow.new_ones((flow.shape[0], 1, new_height, new_width))
    else:
        weight = mask.to(device=device, dtype=dtype)
        if weight.dim() == 2:
            weight = weight.unsqueeze(0).unsqueeze(0)
        elif weight.dim() == 3:
            weight = weight.unsqueeze(0)
        if weight.dim() != 4:
            raise ValueError("2D loss mask must have shape [1, H, W] or [H, W].")
        if weight.shape[-2:] != (new_height, new_width):
            weight = F.interpolate(
                weight,
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
            )

    return flow, weight.clamp(0.0, 1.0)


def action_unit_vector(
    action_unit: int,
    facs_dim: int,
    device: torch.device,
    scale: float = 1.0,
) -> torch.Tensor:
    if action_unit < 0 or action_unit >= facs_dim:
        raise ValueError(
            f"Action unit id {action_unit} is outside the configured facs_dim={facs_dim}."
        )
    facs = torch.zeros(1, facs_dim, device=device)
    facs[0, action_unit] = scale
    return facs


def action_unit_vectors(
    action_units: Sequence[int],
    facs_dim: int,
    device: torch.device,
    scale: float = 1.0,
) -> torch.Tensor:
    vectors = [
        action_unit_vector(action_unit, facs_dim, device, scale)
        for action_unit in action_units
    ]
    return torch.cat(vectors, dim=0)


def parse_action_units(values: Sequence[Any]) -> list[int]:
    name_to_id = {name: action_unit for action_unit, name in AU_NAME.items()}
    parsed = []
    for value in values:
        if isinstance(value, str) and not value.isdigit():
            if value not in name_to_id:
                raise ValueError(f"Unknown action unit name: {value!r}.")
            parsed.append(name_to_id[value])
        else:
            parsed.append(int(value))
    return parsed


def parse_one_action_unit(value: Any) -> int:
    try:
        action_unit = parse_action_units([value])[0]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if action_unit not in AU_NAME:
        raise argparse.ArgumentTypeError(f"Unknown action unit id: {action_unit}.")
    return action_unit


@torch.no_grad()
def maybe_log_validation_render(
    wandb_run: Any,
    model: TopoRig,
    dataset: Dataset,
    config: Mapping[str, Any],
    device: torch.device,
    epoch: int,
    global_step: int,
) -> None:
    maybe_log_action_unit_render(
        wandb_run=wandb_run,
        model=model,
        dataset=dataset,
        config=config,
        device=device,
        epoch=epoch,
        global_step=global_step,
        log_key="val/action_unit_renders",
        log_every_config_key="log_render_every_epochs",
    )


def maybe_log_action_unit_render(
    wandb_run: Any,
    model: TopoRig,
    dataset: Dataset,
    config: Mapping[str, Any],
    device: torch.device,
    epoch: int,
    global_step: int,
    log_key: str,
    log_every_config_key: str,
) -> None:
    wandb_config = config.get("wandb", {})
    log_every_epochs = int(
        wandb_config.get(
            log_every_config_key,
            wandb_config.get("log_render_every_epochs", 1),
        )
    )
    if log_every_epochs <= 0 or epoch % log_every_epochs != 0 or len(dataset) == 0:
        return

    action_units = parse_action_units(config["visualization"].get("action_units", []))
    if not action_units:
        action_units = [0]

    sample = validation_visualization_base_sample(dataset, config)
    image = render_validation_action_units(
        model=model,
        sample=sample,
        config=config,
        device=device,
        target_records_by_action_unit=validation_target_records_for_action_units(
            dataset=dataset,
            base_sample=sample,
            action_units=action_units,
        ),
    )
    import wandb

    custom_sample = safe_custom_visualization_sample(config)
    if custom_sample is not None:
        custom_image = render_validation_action_units(
            model=model,
            sample=custom_sample,
            config=config,
            device=device,
            target_records_by_action_unit={},
        )
        image = stack_visualization_sections(
            [
                ("dataset sample", image),
                (custom_visualization_section_label(config), custom_image),
            ]
        )

    wandb_run.log({log_key: wandb.Image(image)}, step=global_step)


def safe_custom_visualization_sample(config: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    try:
        return custom_visualization_sample(config)
    except Exception as exc:
        progress_write(
            "[WARNING] Skipping custom visualization sample because it could not "
            f"be prepared: {exc}"
        )
        return None


@torch.no_grad()
def maybe_export_validation_prediction_glbs(
    model: nn.Module,
    dataset: Dataset,
    config: Mapping[str, Any],
    device: torch.device,
    run_dir: Path,
    epoch: int,
    force: bool = False,
    label: Optional[str] = None,
) -> None:
    export_config = prediction_glb_config(config)
    if not bool(export_config.get("enabled", False)) or len(dataset) == 0:
        return

    every_epochs = int(export_config.get("every_epochs", 0) or 0)
    export_final = bool(export_config.get("export_final", True))
    if force:
        if not export_final:
            return
    elif every_epochs <= 0 or epoch % every_epochs != 0:
        return

    label = label or f"epoch_{epoch:04d}"
    output_root = prediction_glb_output_root(run_dir, export_config)
    sample = prediction_glb_base_sample(dataset, export_config)
    action_units = prediction_glb_action_units(config)
    export_validation_prediction_glbs(
        model=model,
        sample=sample,
        config=config,
        device=device,
        output_dir=output_root / label,
        action_units=action_units,
        output_convention=str(export_config.get("output_convention", "metahuman")),
        target_records_by_action_unit=validation_target_records_for_action_units(
            dataset=dataset,
            base_sample=sample,
            action_units=action_units,
        ),
    )


@torch.no_grad()
def maybe_export_step_prediction_snapshot(
    model: nn.Module,
    dataset: Optional[Dataset],
    config: Mapping[str, Any],
    device: torch.device,
    run_dir: Optional[Path],
    epoch: int,
    global_step: int,
    wandb_run: Any,
    distributed: DistributedContext,
) -> None:
    snapshot_config = prediction_glb_step_snapshot_config(config)
    every_steps = int(snapshot_config.get("every_steps", 0) or 0)
    should_snapshot = (
        bool(snapshot_config.get("enabled", False))
        and every_steps > 0
        and global_step > 0
        and global_step % every_steps == 0
        and dataset is not None
        and run_dir is not None
        and len(dataset) > 0
    )
    if should_snapshot and distributed.is_main:
        was_training = model.training
        try:
            output_paths: list[Path] = []
            try:
                output_paths = export_step_prediction_glbs(
                    model=model,
                    dataset=dataset,
                    config=config,
                    device=device,
                    run_dir=run_dir,
                    global_step=global_step,
                )
            except Exception as exc:
                progress_write(
                    "[WARNING] Step prediction GLB export failed at "
                    f"step {global_step}: {exc}"
                )
            if bool(snapshot_config.get("log_glb_to_wandb", False)):
                log_step_prediction_glbs_to_wandb(
                    wandb_run=wandb_run,
                    output_paths=output_paths,
                    global_step=global_step,
                    max_glbs=int(snapshot_config.get("max_wandb_glbs", 5) or 0),
                )
            if bool(snapshot_config.get("log_render_to_wandb", True)):
                try:
                    log_step_prediction_render_to_wandb(
                        wandb_run=wandb_run,
                        model=model,
                        dataset=dataset,
                        config=config,
                        device=device,
                        global_step=global_step,
                    )
                except Exception as exc:
                    progress_write(
                        "[WARNING] Step prediction WandB render failed at "
                        f"step {global_step}: {exc}"
                    )
            if output_paths:
                progress_write(
                    "[snapshot] "
                    f"epoch {epoch} step {global_step}: "
                    f"saved {len(output_paths)} GLB(s) to "
                    f"{output_paths[0].parent}"
                )
        finally:
            if was_training:
                model.train()
    if should_snapshot and distributed.enabled:
        dist.barrier()


def prediction_glb_step_snapshot_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    export_config = prediction_glb_config(config)
    value = export_config.get("step_snapshot", {})
    return value if isinstance(value, Mapping) else {}


@torch.no_grad()
def export_step_prediction_glbs(
    model: nn.Module,
    dataset: Dataset,
    config: Mapping[str, Any],
    device: torch.device,
    run_dir: Path,
    global_step: int,
) -> list[Path]:
    export_config = prediction_glb_config(config)
    if not bool(export_config.get("enabled", False)):
        return []

    sample = prediction_glb_base_sample(dataset, export_config)
    action_units = prediction_glb_action_units(config)
    output_root = prediction_glb_output_root(run_dir, export_config)
    output_dir = output_root / f"step_{global_step:06d}"
    return export_validation_prediction_glbs(
        model=model,
        sample=sample,
        config=config,
        device=device,
        output_dir=output_dir,
        action_units=action_units,
        output_convention=str(export_config.get("output_convention", "metahuman")),
        target_records_by_action_unit=validation_target_records_for_action_units(
            dataset=dataset,
            base_sample=sample,
            action_units=action_units,
        ),
    )


@torch.no_grad()
def log_step_prediction_render_to_wandb(
    wandb_run: Any,
    model: nn.Module,
    dataset: Dataset,
    config: Mapping[str, Any],
    device: torch.device,
    global_step: int,
) -> None:
    if wandb_run is None:
        return
    action_units = parse_action_units(config["visualization"].get("action_units", []))
    if not action_units:
        action_units = [0]
    sample = validation_visualization_base_sample(dataset, config)
    image = render_validation_action_units(
        model=model,
        sample=sample,
        config=config,
        device=device,
        target_records_by_action_unit=validation_target_records_for_action_units(
            dataset=dataset,
            base_sample=sample,
            action_units=action_units,
        ),
    )
    custom_sample = safe_custom_visualization_sample(config)
    if custom_sample is not None:
        custom_image = render_validation_action_units(
            model=model,
            sample=custom_sample,
            config=config,
            device=device,
            target_records_by_action_unit={},
        )
        image = stack_visualization_sections(
            [
                ("dataset sample", image),
                (custom_visualization_section_label(config), custom_image),
            ]
        )
    import wandb

    wandb_run.log(
        {"train/step_prediction_render": wandb.Image(image)},
        step=global_step,
    )


def log_step_prediction_glbs_to_wandb(
    wandb_run: Any,
    output_paths: Sequence[Path],
    global_step: int,
    max_glbs: int,
) -> None:
    if wandb_run is None or max_glbs <= 0:
        return
    try:
        import wandb
    except ImportError:
        return

    for path in list(output_paths)[:max_glbs]:
        try:
            wandb_run.log(
                {
                    f"train/step_prediction_glb/{path.stem}": wandb.Object3D(
                        str(path)
                    )
                },
                step=global_step,
            )
        except Exception as exc:
            progress_write(
                "[WARNING] Could not log GLB to WandB as Object3D "
                f"({path}): {exc}"
            )
            return


def prediction_glb_base_sample(
    dataset: Dataset,
    export_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    preferred = str(export_config.get("base_sample", "first")).strip().lower()
    if preferred in {"mesh", "3d"}:
        for index in range(len(dataset)):
            sample = dataset[index]
            if bool(sample.get("is_mesh", False)):
                return visualization_sample_with_dataset_index(dataset, sample, index)
    if preferred in {"image", "img", "2d"}:
        if (
            isinstance(dataset, UnifiedDataset)
            and dataset.pair_image_with_mesh_by_action_unit
            and dataset._unpaired_image_indices
        ):
            index = dataset.mesh_length
            return visualization_sample_with_dataset_index(
                dataset, dataset[index], index
            )
        for index in range(len(dataset)):
            sample = dataset[index]
            if bool(sample.get("is_img", False)) and not bool(
                sample.get("is_mesh", False)
            ):
                return visualization_sample_with_dataset_index(
                    dataset, sample, index
                )
        for index in range(len(dataset)):
            sample = dataset[index]
            if bool(sample.get("is_img", False)):
                return visualization_sample_with_dataset_index(
                    dataset, sample, index
                )
    return visualization_sample_with_dataset_index(dataset, dataset[0], 0)


def visualization_sample_with_dataset_index(
    dataset: Dataset,
    sample: Mapping[str, Any],
    index: int,
) -> Mapping[str, Any]:
    candidate_loader = getattr(
        dataset, "visualization_target_candidate_records", None
    )
    if not callable(candidate_loader):
        return sample
    indexed_sample = dict(sample)
    indexed_sample["_visualization_dataset_index"] = int(index)
    return indexed_sample


def validation_visualization_base_sample(
    dataset: Dataset,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    visualization_config = config.get("visualization", {})
    if not isinstance(visualization_config, Mapping):
        visualization_config = {}
    return prediction_glb_base_sample(
        dataset,
        {"base_sample": visualization_config.get("base_sample", "first")},
    )


def prediction_glb_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    validation_config = config.get("validation", {})
    if not isinstance(validation_config, Mapping):
        return {}
    export_config = validation_config.get("prediction_glb", {})
    return export_config if isinstance(export_config, Mapping) else {}


def prediction_glb_output_root(
    run_dir: Path,
    export_config: Mapping[str, Any],
) -> Path:
    value = str(export_config.get("output_dir", "prediction_glbs"))
    path = Path(value).expanduser()
    if path.is_absolute():
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            return path
        return run_dir / relative
    return run_dir / path


def prediction_glb_action_units(config: Mapping[str, Any]) -> list[int]:
    export_config = prediction_glb_config(config)
    raw_values = export_config.get("action_units")
    if raw_values in (None, ""):
        raw_values = config.get("visualization", {}).get("action_units", [])
    if isinstance(raw_values, (int, str, bytes)):
        raw_values = [raw_values]
    action_units = parse_action_units(raw_values)
    if not action_units:
        action_units = [0]
    max_action_units = export_config.get("max_action_units")
    if max_action_units not in (None, ""):
        action_units = action_units[: int(max_action_units)]
    return action_units


@torch.no_grad()
def export_validation_prediction_glbs(
    model: nn.Module,
    sample: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
    output_dir: Path,
    action_units: Sequence[int],
    output_convention: str,
    target_records_by_action_unit: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> list[Path]:
    model.eval()
    record = validation_sample_record(sample)
    render_image_rows = validation_sample_uses_image_record(sample)
    mesh = to_device(record["mesh"], device)
    vertices = mesh["vertices"].unsqueeze(0)
    normals = mesh.get("normals")
    if normals is not None:
        normals = normals.unsqueeze(0)
    faces = mesh["faces"]
    facs = action_unit_vectors(
        action_units,
        int(config["model"]["facs_dim"]),
        device,
        float(config["training"].get("action_unit_scale", 1.0)),
    ).unsqueeze(0)
    landmark_features = build_model_landmark_features(
        record=record,
        vertices=vertices,
        config=config,
        device=device,
    )
    _deformed_vertices, predicted_displacements = model(
        vertices,
        facs,
        normals=normals,
        faces=faces,
        landmark_features=landmark_features,
        return_deformed=True,
    )
    flat_predicted_displacements = predicted_displacements.reshape(
        -1,
        vertices.shape[1],
        3,
    )
    flat_vertices = vertices.expand(flat_predicted_displacements.shape[0], -1, -1)
    if render_image_rows:
        flat_predicted_displacements = apply_image_geometry_hard_roi_mask_for_action_units(
            vertices=flat_vertices,
            predicted_delta=flat_predicted_displacements,
            action_unit_ids=action_units,
            landmarks_3d=record.get("landmarks_3d"),
            geometry_config=config["loss"].get("image", {}).get("geometry", {}),
            render_config=config["rendering"],
            faces=faces,
        )

    target_records_by_action_unit = target_records_by_action_unit or {}
    mesh_config = config["loss"].get("mesh", {})
    export_config = prediction_glb_config(config)
    output_paths = []
    for index, action_unit in enumerate(action_units):
        predicted_delta = flat_predicted_displacements[index : index + 1]
        target_record = target_records_by_action_unit.get(action_unit, record)
        target_delta = validation_target_delta(
            record=target_record,
            action_unit=action_unit,
            device=device,
            dtype=vertices.dtype,
            expected_vertex_count=vertices.shape[1],
        )
        hard_mask = None
        if not render_image_rows:
            hard_mask = mesh_hard_target_vertex_mask(
                vertices=vertices,
                target_delta=target_delta,
                landmarks_3d=target_record.get(
                    "landmarks_3d",
                    record.get("landmarks_3d"),
                ),
                config=mesh_config,
                faces=record["mesh"].get("faces", faces),
            )
            predicted_delta = apply_mesh_delta_vertex_mask(
                predicted_delta,
                hard_mask,
                faces=faces,
                config=mesh_config.get("hard_target_vertex_mask"),
            )
            predicted_delta = filter_mesh_prediction_delta(
                vertices=vertices,
                predicted_delta=predicted_delta,
                faces=faces,
                vertex_mask=hard_mask,
                config=mesh_config,
            )
        predicted_delta = smooth_prediction_glb_displacements(
            predicted_delta,
            faces,
            export_config,
        )
        should_reapply_mask = (
            not render_image_rows
            and mesh_reapply_target_vertex_mask_after_smoothing(mesh_config)
        )
        if should_reapply_mask:
            predicted_delta = apply_mesh_delta_vertex_mask(
                predicted_delta,
                hard_mask,
                faces=faces,
                config=mesh_config.get("hard_target_vertex_mask"),
            )
            predicted_delta = filter_mesh_prediction_delta(
                vertices=vertices,
                predicted_delta=predicted_delta,
                faces=faces,
                vertex_mask=hard_mask,
                config=mesh_config,
            )
        predicted_vertices = vertices + predicted_delta
        export_vertices = prediction_glb_output_vertices(
            predicted_vertices[0],
            output_convention,
        )
        export_faces = faces.detach().cpu().long()
        export_vertex_mask = (
            hard_mask[0].detach().cpu().to(dtype=torch.bool)
            if hard_mask is not None
            else None
        )
        export_vertices = smooth_prediction_glb_vertices(
            export_vertices,
            export_faces,
            export_config,
            vertex_mask=export_vertex_mask,
        )
        export_normals = _compute_vertex_normals(
            export_vertices.detach().cpu(),
            export_faces,
        )
        path = output_dir / f"{action_unit_slug(action_unit)}_predicted.glb"
        export_glb_mesh(
            vertices=export_vertices,
            faces=export_faces,
            normals=export_normals,
            path=path,
        )
        output_paths.append(path)
    maybe_save_prediction_glb_render(
        model=model,
        sample=sample,
        config=config,
        device=device,
        output_dir=output_dir,
        action_units=action_units,
        target_records_by_action_unit=target_records_by_action_unit,
    )
    return output_paths


def prediction_glb_render_png_enabled(export_config: Mapping[str, Any]) -> bool:
    value = export_config.get("render_png", export_config.get("save_render_png", False))
    return bool(value)


@torch.no_grad()
def maybe_save_prediction_glb_render(
    model: nn.Module,
    sample: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
    output_dir: Path,
    action_units: Sequence[int],
    target_records_by_action_unit: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> Optional[Path]:
    export_config = prediction_glb_config(config)
    if not prediction_glb_render_png_enabled(export_config):
        return None

    render_config = copy.deepcopy(config)
    visualization_config = dict(render_config.get("visualization", {}) or {})
    visualization_config["action_units"] = list(action_units)
    render_config["visualization"] = visualization_config

    filename = str(export_config.get("render_png_name", "prediction_render.png"))
    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = render_validation_action_units(
        model=model,
        sample=sample,
        config=render_config,
        device=device,
        target_records_by_action_unit=target_records_by_action_unit,
    )
    image.save(output_path)
    return output_path


def smooth_prediction_glb_vertices(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    export_config: Mapping[str, Any],
    vertex_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    smoothing_config = export_config.get("surface_smoothing", {})
    if not isinstance(smoothing_config, Mapping) or not bool(
        smoothing_config.get("enabled", False)
    ):
        return vertices

    iterations = int(smoothing_config.get("iterations", 0) or 0)
    blend = float(
        smoothing_config.get(
            "blend",
            smoothing_config.get("lambda", 0.2),
        )
    )
    if iterations <= 0 or blend <= 0.0:
        return vertices

    max_edges = optional_positive_int(smoothing_config.get("max_edges"))
    preserve_boundary = bool(smoothing_config.get("preserve_boundary", True))
    mask = None
    if bool(smoothing_config.get("target_vertex_mask", False)):
        if vertex_mask is None:
            return vertices
        mask = vertex_mask.detach().to(device=vertices.device, dtype=torch.bool)
        if mask.dim() == 2:
            if mask.shape[0] != 1:
                raise ValueError("surface_smoothing vertex_mask must have one batch.")
            mask = mask[0]
        if mask.dim() != 1 or mask.shape[0] != vertices.shape[0]:
            raise ValueError("surface_smoothing vertex_mask must match vertices.")

    boundary_mask = (
        mesh_boundary_vertex_mask(faces, vertices.shape[0], vertices.device)
        if preserve_boundary
        else None
    )
    smoothed = smooth_vertex_displacement(
        vertices.unsqueeze(0),
        faces,
        iterations=iterations,
        blend=blend,
        max_edges=max_edges,
    )[0]
    update_mask = torch.ones(vertices.shape[0], dtype=torch.bool, device=vertices.device)
    if mask is not None:
        update_mask = update_mask & mask
        if bool(smoothing_config.get("preserve_mask_boundary", False)):
            mask_boundary = mesh_mask_boundary_vertex_mask(mask, faces)
            update_mask = update_mask & ~mask_boundary
    if boundary_mask is not None and bool(boundary_mask.any().detach().cpu()):
        update_mask = update_mask & ~boundary_mask
    smoothed = torch.where(update_mask.unsqueeze(-1), smoothed, vertices)
    return smoothed.contiguous()


def mesh_mask_boundary_vertex_mask(mask: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    mask = mask.detach().to(dtype=torch.bool)
    if mask.dim() != 1:
        raise ValueError("mask boundary expects a one-dimensional mask.")
    if faces.numel() == 0:
        return torch.zeros_like(mask, dtype=torch.bool)
    edges = sampled_face_edges(faces.detach().cpu().long(), max_edges=None)
    if edges.numel() == 0:
        return torch.zeros_like(mask, dtype=torch.bool)
    edges = edges.to(device=mask.device, dtype=torch.long)
    valid = (
        (edges[:, 0] >= 0)
        & (edges[:, 0] < mask.shape[0])
        & (edges[:, 1] >= 0)
        & (edges[:, 1] < mask.shape[0])
    )
    edges = edges[valid]
    crossing = mask.index_select(0, edges[:, 0]) != mask.index_select(0, edges[:, 1])
    boundary = torch.zeros_like(mask, dtype=torch.bool)
    if bool(crossing.any().detach().cpu()):
        boundary[edges[crossing].reshape(-1).unique()] = True
    return boundary


def mesh_boundary_vertex_mask(
    faces: torch.Tensor,
    vertex_count: int,
    device: torch.device,
) -> torch.Tensor:
    if faces.numel() == 0:
        return torch.zeros(vertex_count, dtype=torch.bool, device=device)
    faces = faces.detach().to(device=device, dtype=torch.long)
    edges = torch.cat(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ),
        dim=0,
    )
    edges = torch.sort(edges, dim=1).values
    unique_edges, counts = torch.unique(edges, dim=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    mask = torch.zeros(vertex_count, dtype=torch.bool, device=device)
    if boundary_edges.numel() > 0:
        mask[boundary_edges.reshape(-1).unique()] = True
    return mask


def smooth_prediction_glb_displacements(
    predicted_displacements: torch.Tensor,
    faces: torch.Tensor,
    export_config: Mapping[str, Any],
) -> torch.Tensor:
    smoothing_config = export_config.get("smoothing", {})
    if not isinstance(smoothing_config, Mapping):
        smoothing_config = {}
    iterations = int(
        smoothing_config.get(
            "iterations",
            export_config.get("smooth_iterations", 0),
        )
        or 0
    )
    blend = float(
        smoothing_config.get(
            "blend",
            smoothing_config.get(
                "lambda",
                export_config.get("smooth_lambda", 0.35),
            ),
        )
    )
    if iterations <= 0 or blend <= 0.0:
        return predicted_displacements
    max_edges = optional_positive_int(
        smoothing_config.get(
            "max_edges",
            export_config.get("smooth_max_edges"),
        )
    )
    return smooth_vertex_displacement(
        predicted_displacements,
        faces,
        iterations=iterations,
        blend=blend,
        max_edges=max_edges,
    )


def action_unit_slug(action_unit: int) -> str:
    name = AU_NAME.get(int(action_unit), "unknown")
    safe_name = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in str(name)
    )
    return f"au{int(action_unit):02d}_{safe_name}"


def prediction_glb_output_vertices(
    vertices: torch.Tensor,
    output_convention: str,
) -> torch.Tensor:
    convention = str(output_convention).strip().lower()
    if convention == "model":
        return vertices.detach().cpu().float()
    if convention in {"metahuman", "fbx"}:
        return torch.stack(
            (
                -vertices[..., 0],
                -vertices[..., 1],
                vertices[..., 2],
            ),
            dim=-1,
        ).detach().cpu().float()
    raise ValueError("prediction_glb.output_convention must be metahuman or model.")


def export_glb_mesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    normals: torch.Tensor,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "GLB export requires trimesh. Install it or disable "
            "validation.prediction_glb.enabled."
        ) from exc

    mesh = trimesh.Trimesh(
        vertices=vertices.detach().cpu().float().numpy(),
        faces=faces.detach().cpu().long().numpy(),
        vertex_normals=normals.detach().cpu().float().numpy(),
        process=False,
    )
    colors = validation_vertex_colors(vertices.detach().cpu().unsqueeze(0))[0]
    rgb = (colors.clamp(0.0, 1.0).numpy() * 255.0).round()
    alpha = np.full((rgb.shape[0], 1), 255.0)
    mesh.visual.vertex_colors = np.concatenate([rgb, alpha], axis=1).astype(np.uint8)
    mesh.export(path)


def validation_target_records_for_action_units(
    dataset: Dataset,
    base_sample: Mapping[str, Any],
    action_units: Sequence[int],
) -> dict[int, Mapping[str, Any]]:
    base_record = validation_sample_record(base_sample)
    remaining = set(action_units)
    records: dict[int, Mapping[str, Any]] = {}

    base_action_unit = record_action_unit(base_record)
    if base_action_unit in remaining and validation_record_has_target(base_record):
        records[base_action_unit] = base_record
        remaining.remove(base_action_unit)

    if not remaining:
        return records

    indexed_records = indexed_validation_target_records(
        dataset=dataset,
        base_sample=base_sample,
        base_record=base_record,
        action_units=action_units,
        existing_records=records,
    )
    if indexed_records is not None:
        return indexed_records

    for index in range(len(dataset)):
        candidate_record = validation_sample_record(dataset[index])
        action_unit = record_action_unit(candidate_record)
        if action_unit not in remaining:
            continue
        if not validation_record_has_target(candidate_record):
            continue
        if not validation_records_share_neutral_mesh(base_record, candidate_record):
            continue
        records[action_unit] = candidate_record
        remaining.remove(action_unit)
        if not remaining:
            break

    return records


def indexed_validation_target_records(
    dataset: Dataset,
    base_sample: Mapping[str, Any],
    base_record: Mapping[str, Any],
    action_units: Sequence[int],
    existing_records: Mapping[int, Mapping[str, Any]],
) -> Optional[dict[int, Mapping[str, Any]]]:
    base_index = base_sample.get("_visualization_dataset_index")
    candidate_loader = getattr(
        dataset, "visualization_target_candidate_records", None
    )
    if not isinstance(base_index, int) or not callable(candidate_loader):
        return None

    cache_key = (base_index, tuple(int(value) for value in action_units))
    cache = getattr(dataset, "_visualization_target_record_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(dataset, "_visualization_target_record_cache", cache)
    if cache_key in cache:
        return cache[cache_key]

    candidates_by_action_unit = candidate_loader(base_index, action_units)
    if candidates_by_action_unit is None:
        return None

    records = dict(existing_records)
    for action_unit in action_units:
        action_unit = int(action_unit)
        if action_unit in records:
            continue
        for candidate_record in candidates_by_action_unit.get(action_unit, ()):
            if not validation_record_has_target(candidate_record):
                continue
            if not validation_records_share_neutral_mesh(
                base_record, candidate_record
            ):
                continue
            records[action_unit] = candidate_record
            break

    missing = sorted(set(int(value) for value in action_units) - set(records))
    if missing:
        progress_write(
            "[WARNING] Skipping unavailable visualization target AU(s) for "
            f"dataset sample {base_index}: {missing}"
        )
    cache[cache_key] = records
    return records


def validation_sample_record(sample: Mapping[str, Any]) -> Mapping[str, Any]:
    if bool(sample.get("is_mesh", False)):
        return sample["mesh"]
    return sample["image"] if bool(sample.get("is_img", False)) else sample["mesh"]


def validation_sample_uses_image_record(sample: Mapping[str, Any]) -> bool:
    return bool(sample.get("is_img", False)) and not bool(sample.get("is_mesh", False))


def record_action_unit(record: Mapping[str, Any]) -> Optional[int]:
    if "action_unit_id" not in record:
        return None
    return as_int(record["action_unit_id"])


def validation_record_has_target(record: Mapping[str, Any]) -> bool:
    return "delta_vertices" in record and "action_unit_id" in record


def validation_records_share_neutral_mesh(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> bool:
    first_vertices = validation_record_vertices(first)
    second_vertices = validation_record_vertices(second)
    if first_vertices is None or second_vertices is None:
        return False
    if first_vertices.shape != second_vertices.shape:
        return False
    return bool(
        torch.equal(
            first_vertices.detach().cpu(),
            second_vertices.detach().cpu(),
        )
    )


def validation_record_vertices(record: Mapping[str, Any]) -> Optional[torch.Tensor]:
    mesh = record.get("mesh")
    if not isinstance(mesh, Mapping):
        return None
    vertices = mesh.get("vertices")
    if not isinstance(vertices, torch.Tensor):
        return None
    return vertices


def custom_visualization_sample(
    config: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    mesh_path = custom_visualization_mesh_path(config)
    if mesh_path is None:
        return None

    visualization_config = config.get("visualization", {}) or {}
    input_convention = str(
        visualization_config.get("custom_mesh_input_convention", "auto")
    )
    mesh = load_custom_visualization_mesh(mesh_path)
    mesh = preprocess_custom_visualization_mesh(
        mesh,
        config,
        input_suffix=mesh_path.suffix.lower(),
        input_convention=input_convention,
    )
    landmarks_3d: dict[str, torch.Tensor] = {}
    if custom_visualization_alignment_enabled(config):
        mesh, landmarks_3d = align_custom_visualization_mesh_to_metahuman(
            mesh=mesh,
            mesh_path=mesh_path,
            config=config,
        )
    return {
        "image": {},
        "mesh": {
            "mesh": mesh,
            "landmarks_3d": landmarks_3d,
            "source_path": str(mesh_path),
        },
        "is_img": False,
        "is_mesh": True,
    }


def custom_visualization_mesh_path(config: Mapping[str, Any]) -> Optional[Path]:
    visualization_config = config.get("visualization", {}) or {}
    if not isinstance(visualization_config, Mapping):
        return None
    value = visualization_config.get("custom_mesh_path")
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"visualization.custom_mesh_path does not exist: {path}"
        )
    return path


def custom_visualization_section_label(config: Mapping[str, Any]) -> str:
    visualization_config = config.get("visualization", {}) or {}
    label = visualization_config.get("custom_mesh_label")
    if label:
        return str(label)
    path = visualization_config.get("custom_mesh_path")
    if path:
        return f"custom mesh: {Path(str(path)).name}"
    return "custom mesh"


def custom_visualization_alignment_enabled(config: Mapping[str, Any]) -> bool:
    visualization_config = config.get("visualization", {}) or {}
    if not isinstance(visualization_config, Mapping):
        return False
    return bool(visualization_config.get("custom_mesh_align_to_metahuman", False))


def align_custom_visualization_mesh_to_metahuman(
    mesh: Mapping[str, torch.Tensor],
    mesh_path: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    visualization_config = config.get("visualization", {}) or {}
    cache_dir = Path(
        str(
            visualization_config.get(
                "custom_mesh_landmark_cache_dir",
                DEFAULT_CUSTOM_VIS_LANDMARK_CACHE_DIR,
            )
        )
    ).expanduser()
    mapper_path = Path(
        str(
            visualization_config.get(
                "custom_mesh_mediapipe_mapper_path",
                DEFAULT_CUSTOM_VIS_MEDIAPIPE_MAPPER,
            )
        )
    ).expanduser()
    blender_value = visualization_config.get("custom_mesh_blender")
    blender = None if blender_value in (None, "") else str(blender_value)
    force = bool(visualization_config.get("custom_mesh_refresh_landmarks", False))
    min_landmarks = int(
        visualization_config.get("custom_mesh_min_alignment_landmarks", 32)
    )
    alignment_mode = str(
        visualization_config.get(
            "custom_mesh_alignment_mode",
            "scale_translation",
        )
    )

    input_landmark_path = ensure_custom_visualization_mediapipe_landmarks(
        mesh_path=mesh_path,
        cache_dir=cache_dir,
        mapper_path=mapper_path,
        blender=blender,
        force=force,
    )
    reference_mesh_path = custom_visualization_reference_mesh_path(config)
    reference_landmark_path = ensure_custom_visualization_mediapipe_landmarks(
        mesh_path=reference_mesh_path,
        cache_dir=cache_dir,
        mapper_path=mapper_path,
        blender=blender,
        force=force,
        prefer_sidecar=True,
    )

    input_landmarks = load_custom_visualization_landmark_mapping(input_landmark_path)
    reference_landmarks = load_custom_visualization_landmark_mapping(
        reference_landmark_path
    )
    reference_mesh = load_custom_visualization_reference_mesh(
        reference_mesh_path,
        config,
    )
    aligned_mesh = align_mesh_to_reference_landmarks(
        mesh=mesh,
        input_landmarks=input_landmarks,
        reference_mesh=reference_mesh,
        reference_landmarks=reference_landmarks,
        min_landmarks=min_landmarks,
        alignment_mode=alignment_mode,
    )
    return aligned_mesh, landmark_payload_from_mapping(aligned_mesh, input_landmarks)


def custom_visualization_reference_mesh_path(config: Mapping[str, Any]) -> Path:
    visualization_config = config.get("visualization", {}) or {}
    sample_dir = Path(
        str(
            visualization_config.get(
                "custom_mesh_metahuman_sample_dir",
                DEFAULT_CUSTOM_VIS_METAHUMAN_SAMPLE_DIR,
            )
        )
    ).expanduser()
    if not sample_dir.is_dir():
        raise FileNotFoundError(sample_dir)

    candidates = [
        path
        for path in sorted(sample_dir.iterdir())
        if path.is_file() and path.suffix.lower() in CUSTOM_VIS_SUPPORTED_MESH_SUFFIXES
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No supported reference mesh was found in {sample_dir}."
        )
    return candidates[0]


def load_custom_visualization_reference_mesh(
    path: Path,
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    visualization_config = config.get("visualization", {}) or {}
    input_convention = str(
        visualization_config.get("custom_mesh_reference_input_convention", "auto")
    )
    return preprocess_custom_visualization_mesh(
        load_custom_visualization_mesh(path),
        config,
        input_suffix=path.suffix.lower(),
        input_convention=input_convention,
    )


def ensure_custom_visualization_mediapipe_landmarks(
    mesh_path: Path,
    cache_dir: Path,
    mapper_path: Path,
    blender: Optional[str],
    force: bool = False,
    prefer_sidecar: bool = False,
) -> Path:
    sidecar_path = mesh_path.with_suffix(".json")
    if prefer_sidecar and sidecar_path.is_file() and not force:
        return sidecar_path

    cache_path = custom_visualization_landmark_cache_path(mesh_path, cache_dir)
    if cache_path.is_file() and not force:
        return cache_path
    if sidecar_path.is_file() and not force:
        return sidecar_path
    if not mapper_path.is_file():
        raise FileNotFoundError(mapper_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(mapper_path),
        "--mesh",
        str(mesh_path),
        "--output",
        str(cache_path),
    ]
    if blender:
        command.extend(["--blender", str(blender)])

    process = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        stdout = process.stdout.strip()
        stderr = process.stderr.strip()
        details = "\n".join(part for part in (stdout, stderr) if part)
        raise RuntimeError(
            "Failed to generate MediaPipe landmark mapping for "
            f"{mesh_path} with {mapper_path}."
            f"\n{details}"
        )
    if not cache_path.is_file():
        raise RuntimeError(
            "MediaPipe landmark mapper finished but did not write "
            f"{cache_path}."
        )
    return cache_path


def custom_visualization_landmark_cache_path(
    mesh_path: Path,
    cache_dir: Path,
) -> Path:
    resolved = str(mesh_path.expanduser().resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{mesh_path.stem}_{digest}.json"


def load_custom_visualization_landmark_mapping(path: Path) -> dict[int, int]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    if isinstance(loaded, dict) and isinstance(loaded.get("mapping"), dict):
        mapping = loaded["mapping"]
    elif isinstance(loaded, dict):
        mapping = loaded
    else:
        raise ValueError(f"Expected {path} to contain a landmark mapping.")

    parsed: dict[int, int] = {}
    for raw_mediapipe_id, raw_vertex_id in mapping.items():
        if raw_vertex_id is None:
            continue
        parsed[int(raw_mediapipe_id)] = int(raw_vertex_id)
    if not parsed:
        raise ValueError(f"No usable MediaPipe landmarks were found in {path}.")
    return parsed


def landmark_payload_from_mapping(
    mesh: Mapping[str, torch.Tensor],
    mapping: Mapping[int, int],
) -> dict[str, torch.Tensor]:
    vertices = mesh["vertices"]
    pairs = [
        (int(mediapipe_id), int(vertex_id))
        for mediapipe_id, vertex_id in mapping.items()
        if 0 <= int(vertex_id) < vertices.shape[0]
    ]
    pairs.sort(key=lambda item: item[0])
    if not pairs:
        return {
            "mediapipe_ids": torch.empty(0, dtype=torch.long),
            "vertex_ids": torch.empty(0, dtype=torch.long),
            "neutral_positions": torch.empty(0, 3, dtype=torch.float32),
        }

    mediapipe_ids = torch.tensor(
        [mediapipe_id for mediapipe_id, _vertex_id in pairs],
        dtype=torch.long,
    )
    vertex_ids = torch.tensor(
        [vertex_id for _mediapipe_id, vertex_id in pairs],
        dtype=torch.long,
    )
    return {
        "mediapipe_ids": mediapipe_ids.contiguous(),
        "vertex_ids": vertex_ids.contiguous(),
        "neutral_positions": vertices.detach()
        .cpu()
        .float()
        .index_select(0, vertex_ids)
        .contiguous(),
    }


def align_mesh_to_reference_landmarks(
    mesh: Mapping[str, torch.Tensor],
    input_landmarks: Mapping[int, int],
    reference_mesh: Mapping[str, torch.Tensor],
    reference_landmarks: Mapping[int, int],
    min_landmarks: int,
    alignment_mode: str = "scale_translation",
) -> dict[str, torch.Tensor]:
    source_points, target_points = shared_landmark_positions(
        mesh=mesh,
        input_landmarks=input_landmarks,
        reference_mesh=reference_mesh,
        reference_landmarks=reference_landmarks,
        min_landmarks=min_landmarks,
    )
    mode = normalize_custom_visualization_alignment_mode(alignment_mode)
    if mode == "similarity":
        scale, rotation, translation = similarity_procrustes(
            source_points,
            target_points,
        )
        return apply_similarity_transform(mesh, scale, rotation, translation)

    scale, translation = scale_translation_landmark_fit(
        source_points,
        target_points,
    )
    return apply_scale_translation_transform(mesh, scale, translation)


def normalize_custom_visualization_alignment_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    if mode in {"scale_translation", "scale_translate", "scale", "no_rotation"}:
        return "scale_translation"
    if mode in {"similarity", "procrustes", "full_similarity", "rigid_scale"}:
        return "similarity"
    raise ValueError(
        "custom_mesh_alignment_mode must be one of: "
        "scale_translation or similarity."
    )


def shared_landmark_positions(
    mesh: Mapping[str, torch.Tensor],
    input_landmarks: Mapping[int, int],
    reference_mesh: Mapping[str, torch.Tensor],
    reference_landmarks: Mapping[int, int],
    min_landmarks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_vertices = mesh["vertices"]
    reference_vertices = reference_mesh["vertices"]
    source_points: list[torch.Tensor] = []
    target_points: list[torch.Tensor] = []

    for mediapipe_id in sorted(set(input_landmarks) & set(reference_landmarks)):
        input_vertex_id = int(input_landmarks[mediapipe_id])
        reference_vertex_id = int(reference_landmarks[mediapipe_id])
        if not 0 <= input_vertex_id < input_vertices.shape[0]:
            continue
        if not 0 <= reference_vertex_id < reference_vertices.shape[0]:
            continue
        source_points.append(input_vertices[input_vertex_id])
        target_points.append(reference_vertices[reference_vertex_id])

    if len(source_points) < int(min_landmarks):
        raise ValueError(
            "Not enough shared valid MediaPipe landmarks for alignment: "
            f"found {len(source_points)}, need at least {int(min_landmarks)}."
        )
    return torch.stack(source_points, dim=0), torch.stack(target_points, dim=0)


def similarity_procrustes(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if source_points.shape != target_points.shape:
        raise ValueError("source_points and target_points must have the same shape.")
    if source_points.ndim != 2 or source_points.shape[-1] != 3:
        raise ValueError("source_points and target_points must have shape [N, 3].")
    if source_points.shape[0] < 3:
        raise ValueError("At least three points are required for Procrustes alignment.")

    dtype = source_points.dtype
    source = source_points.detach().cpu().double()
    target = target_points.detach().cpu().double()
    source_mean = source.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = source_centered.transpose(0, 1).matmul(target_centered)
    u, _singular_values, vh = torch.linalg.svd(covariance, full_matrices=False)
    rotation = u.matmul(vh)
    if torch.linalg.det(rotation) < 0:
        u = u.clone()
        u[:, -1] *= -1.0
        rotation = u.matmul(vh)

    denominator = source_centered.square().sum().clamp_min(1.0e-12)
    scale = (source_centered.matmul(rotation) * target_centered).sum() / denominator
    translation = (
        target_mean.squeeze(0)
        - source_mean.squeeze(0).matmul(rotation) * scale
    )
    return (
        scale.to(dtype=dtype),
        rotation.to(dtype=dtype),
        translation.to(dtype=dtype),
    )


def scale_translation_landmark_fit(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source_points.shape != target_points.shape:
        raise ValueError("source_points and target_points must have the same shape.")
    if source_points.ndim != 2 or source_points.shape[-1] != 3:
        raise ValueError("source_points and target_points must have shape [N, 3].")
    if source_points.shape[0] < 3:
        raise ValueError("At least three points are required for landmark alignment.")

    dtype = source_points.dtype
    source = source_points.detach().cpu().double()
    target = target_points.detach().cpu().double()
    source_mean = source.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    source_centered = source - source_mean
    target_centered = target - target_mean

    source_energy = source_centered.square().sum().clamp_min(1.0e-12)
    scale = torch.sqrt(
        target_centered.square().sum().clamp_min(1.0e-12) / source_energy
    )
    translation = target_mean.squeeze(0) - source_mean.squeeze(0) * scale
    return scale.to(dtype=dtype), translation.to(dtype=dtype)


def apply_scale_translation_transform(
    mesh: Mapping[str, torch.Tensor],
    scale: torch.Tensor,
    translation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    transformed = dict(mesh)
    transformed["vertices"] = (mesh["vertices"] * scale + translation).contiguous()
    transformed["normals"] = F.normalize(mesh["normals"], dim=-1, eps=1.0e-6)
    return transformed


def apply_similarity_transform(
    mesh: Mapping[str, torch.Tensor],
    scale: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    transformed = dict(mesh)
    transformed["vertices"] = (
        mesh["vertices"].matmul(rotation) * scale + translation
    ).contiguous()
    transformed["normals"] = F.normalize(
        mesh["normals"].matmul(rotation),
        dim=-1,
        eps=1.0e-6,
    ).contiguous()
    return transformed


def load_custom_visualization_mesh(path: Path) -> dict[str, torch.Tensor]:
    suffix = path.suffix.lower()
    if suffix == ".glb":
        return {
            name: tensor.detach().cpu().float().contiguous()
            if name != "faces"
            else tensor.detach().cpu().long().contiguous()
            for name, tensor in _load_glb_for_model(path).items()
        }
    if suffix == ".fbx":
        faces, vertices, _blendshapes = fbx_to_tensor(path)
        vertices = vertices.detach().cpu().float().contiguous()
        faces = faces.detach().cpu().long().contiguous()
        return {
            "vertices": vertices,
            "faces": faces,
            "normals": _compute_vertex_normals(vertices, faces),
        }
    raise ValueError(
        "visualization.custom_mesh_path must point to a .glb or .fbx mesh."
    )


def preprocess_custom_visualization_mesh(
    mesh: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    input_suffix: Optional[str] = None,
    input_convention: str = "auto",
) -> dict[str, torch.Tensor]:
    vertices = mesh["vertices"].detach().cpu().float().contiguous()
    faces = mesh["faces"].detach().cpu().long().contiguous()
    normals = mesh.get("normals")
    if not isinstance(normals, torch.Tensor) or normals.shape != vertices.shape:
        normals = _compute_vertex_normals(vertices, faces)
    else:
        normals = normals.detach().cpu().float().contiguous()

    oriented = orient_custom_visualization_mesh(
        {
            "vertices": vertices,
            "faces": faces,
            "normals": normals,
        },
        input_suffix=input_suffix,
        input_convention=input_convention,
    )
    settings = custom_visualization_mesh_preprocess_settings(config)
    zero_delta = torch.zeros_like(oriented["vertices"])
    processed_mesh, _zero_delta, _landmarks = _transform_model_sample(
        mesh=oriented,
        delta_vertices=zero_delta,
        landmarks_3d={},
        mesh_up_axis=settings["mesh_up_axis"],
        mesh_front_axis=settings["mesh_front_axis"],
        normalize_on_get=settings["normalize_on_get"],
        normalized_extent=settings["normalized_extent"],
    )
    return processed_mesh


def orient_custom_visualization_mesh(
    mesh: Mapping[str, torch.Tensor],
    input_suffix: Optional[str],
    input_convention: str,
) -> dict[str, torch.Tensor]:
    convention = resolved_custom_visualization_input_convention(
        input_suffix,
        input_convention,
    )
    if convention == "fbx":
        return dict(mesh)
    if convention not in {
        "custom-glb-y-front",
        "custom-glb-neg-y-front",
        "custom-glb-y-up-z-front",
        "custom-glb-y-up-neg-z-front",
    }:
        raise ValueError(
            "visualization.custom_mesh_input_convention must be one of: "
            "auto, fbx, custom-glb, custom-glb-neg-y-front, "
            "custom-glb-y-up-neg-z-front."
        )

    oriented = dict(mesh)
    if convention in {"custom-glb-y-up-z-front", "custom-glb-y-up-neg-z-front"}:
        glb_front_axis = (
            "-z" if convention == "custom-glb-y-up-neg-z-front" else "z"
        )
        oriented["vertices"] = custom_glb_y_up_visualization_to_dataset_axes(
            mesh["vertices"],
            front_axis=glb_front_axis,
        )
        oriented["normals"] = F.normalize(
            custom_glb_y_up_visualization_to_dataset_axes(
                mesh["normals"],
                front_axis=glb_front_axis,
            ),
            dim=-1,
            eps=1.0e-6,
        )
        return oriented

    glb_front_axis = "-y" if convention == "custom-glb-neg-y-front" else "y"
    oriented["vertices"] = custom_glb_z_down_visualization_to_dataset_axes(
        mesh["vertices"],
        front_axis=glb_front_axis,
    )
    oriented["normals"] = F.normalize(
        custom_glb_z_down_visualization_to_dataset_axes(
            mesh["normals"],
            front_axis=glb_front_axis,
        ),
        dim=-1,
        eps=1.0e-6,
    )
    return oriented


def resolved_custom_visualization_input_convention(
    input_suffix: Optional[str],
    input_convention: str,
) -> str:
    convention = str(input_convention).strip().lower().replace("_", "-")
    if convention != "auto":
        if convention in {"glb", "custom-glb", "custom-glb-y-front"}:
            return "custom-glb-y-front"
        if convention in {
            "custom-glb-neg-y-front",
            "custom-glb-negative-y-front",
            "glb-neg-y-front",
            "glb-negative-y-front",
        }:
            return "custom-glb-neg-y-front"
        if convention in {"custom-glb-y-up-z-front", "glb-y-up-z-front"}:
            return "custom-glb-y-up-z-front"
        if convention in {
            "custom-glb-y-up-neg-z-front",
            "custom-glb-y-up-negative-z-front",
            "glb-y-up-neg-z-front",
            "glb-y-up-negative-z-front",
        }:
            return "custom-glb-y-up-neg-z-front"
        if convention in {"fbx", "metahuman", "dataset"}:
            return "fbx"
        return convention
    return "custom-glb-y-front" if str(input_suffix or "").lower() == ".glb" else "fbx"


def custom_glb_z_down_visualization_to_dataset_axes(
    values: torch.Tensor,
    front_axis: str = "y",
) -> torch.Tensor:
    if front_axis == "y":
        y_values = -values[..., 1]
    elif front_axis in {"-y", "neg-y"}:
        y_values = values[..., 1]
    else:
        raise ValueError("front_axis must be 'y' or '-y'.")
    return torch.stack(
        (
            values[..., 0],
            y_values,
            -values[..., 2],
        ),
        dim=-1,
    ).contiguous()


def custom_glb_y_up_visualization_to_dataset_axes(
    values: torch.Tensor,
    front_axis: str = "-z",
) -> torch.Tensor:
    if front_axis == "z":
        dataset_y = -values[..., 2]
    elif front_axis in {"-z", "neg-z"}:
        dataset_y = values[..., 2]
    else:
        raise ValueError("front_axis must be 'z' or '-z'.")
    return torch.stack(
        (
            values[..., 0],
            dataset_y,
            values[..., 1],
        ),
        dim=-1,
    ).contiguous()


def custom_visualization_mesh_preprocess_settings(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    mesh_args = configured_visualization_mesh_args(config)
    return {
        "mesh_up_axis": str(mesh_args.get("mesh_up_axis", "z")),
        "mesh_front_axis": str(mesh_args.get("mesh_front_axis", "-y")),
        "normalize_on_get": bool(mesh_args.get("normalize_on_get", True)),
        "normalized_extent": float(mesh_args.get("normalized_extent", 2.0)),
    }


def configured_visualization_mesh_args(
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    data_config = config.get("data", {})
    if not isinstance(data_config, Mapping):
        return {}

    for split_name in ("val", "train"):
        split_config = data_config.get(split_name, {})
        if not isinstance(split_config, Mapping):
            continue
        mesh_args = split_config.get("mesh_args", {})
        if isinstance(mesh_args, Mapping):
            return mesh_args
    return {}


@torch.no_grad()
def render_validation_action_units(
    model: TopoRig,
    sample: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
    target_records_by_action_unit: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> Image.Image:
    model.eval()
    record = validation_sample_record(sample)
    render_image_rows = validation_sample_uses_image_record(sample)
    target_records_by_action_unit = target_records_by_action_unit or {}
    mesh = to_device(record["mesh"], device)
    vertices = mesh["vertices"].unsqueeze(0)
    normals = mesh.get("normals")
    if normals is not None:
        normals = normals.unsqueeze(0)
    faces = mesh["faces"]
    action_units = parse_action_units(config["visualization"].get("action_units", []))
    if not action_units:
        action_units = [0]

    facs = action_unit_vectors(
        action_units,
        int(config["model"]["facs_dim"]),
        device,
        float(config["training"].get("action_unit_scale", 1.0)),
    ).unsqueeze(0)
    landmark_features = build_model_landmark_features(
        record=record,
        vertices=vertices,
        config=config,
        device=device,
    )
    _deformed_vertices, predicted_displacements = model(
        vertices,
        facs,
        normals=normals,
        faces=faces,
        landmark_features=landmark_features,
        return_deformed=True,
    )
    flat_predicted_displacements = predicted_displacements.reshape(
        -1,
        vertices.shape[1],
        3,
    )
    flat_vertices = vertices.expand(flat_predicted_displacements.shape[0], -1, -1)
    if render_image_rows:
        flat_predicted_displacements = apply_image_geometry_hard_roi_mask_for_action_units(
            vertices=flat_vertices,
            predicted_delta=flat_predicted_displacements,
            action_unit_ids=action_units,
            landmarks_3d=record.get("landmarks_3d"),
            geometry_config=config["loss"].get("image", {}).get("geometry", {}),
            render_config=config["rendering"],
            faces=faces,
        )
    render_size = normalize_image_size(config["rendering"]["validation_image_size"])
    depth_render_size = normalize_image_size(
        config["loss"].get("mesh", {}).get("depth_image_size", render_size)
    )
    render_up_axis = resolve_validation_render_up_axis(config, vertices)
    render_vertices = vertices_to_render_axes(vertices, render_up_axis)
    render_normals = (
        vertices_to_render_axes(normals, render_up_axis)
        if normals is not None
        else None
    )
    render_center = render_vertex_bounds_center(render_vertices)
    render_vertices = render_vertices - render_center
    predicted_displacements = flat_predicted_displacements
    mesh_config = config["loss"].get("mesh", {})
    neutral_texture = {
        "vertex_colors": validation_vertex_colors(
            render_vertices,
            normals=render_normals,
        )
    }
    render_backend = str(config["rendering"].get("backend", "auto"))
    render_kwargs = config["rendering"].get("kwargs", {})
    neutral_panel = render_validation_mesh_panel(
        render_vertices,
        faces,
        texture=neutral_texture,
        image_size=render_size,
        backend=render_backend,
        render_kwargs=render_kwargs,
    )

    show_target_columns = (
        ("delta_vertices" in record and "action_unit_id" in record)
        or bool(target_records_by_action_unit)
    )
    blank_panel = Image.new("RGB", neutral_panel.size, (255, 255, 255))
    depth_blank_panel = Image.new(
        "RGB",
        (depth_render_size[1], depth_render_size[0]),
        (255, 255, 255),
    )

    panel_rows = []
    label_rows = []
    for index, action_unit in enumerate(action_units):
        label = action_unit_label(action_unit)
        predicted_delta = predicted_displacements[index : index + 1]
        panels = [neutral_panel]
        labels = [f"{label} neutral"]
        render_target_vertices = None
        target_panel = blank_panel

        target_record = target_records_by_action_unit.get(action_unit, record)
        target_delta = validation_target_delta(
            record=target_record,
            action_unit=action_unit,
            device=device,
            dtype=vertices.dtype,
            expected_vertex_count=vertices.shape[1],
        )
        if show_target_columns:
            if target_delta is None:
                panels.append(blank_panel)
                labels.append(f"{label} target n/a")
            else:
                target_vertices = vertices + target_delta
                render_target_vertices = vertices_to_render_axes(
                    target_vertices,
                    render_up_axis,
                ) - render_center
                target_panel = render_validation_mesh_panel(
                    render_target_vertices,
                    faces,
                    texture=neutral_texture,
                    image_size=render_size,
                    backend=render_backend,
                    render_kwargs=render_kwargs,
                )
                panels.append(target_panel)
                labels.append(f"{label} target")

        if not render_image_rows:
            predicted_delta = apply_mesh_delta_vertex_mask(
                predicted_delta,
                mesh_hard_target_vertex_mask(
                    vertices=vertices,
                    target_delta=target_delta,
                    landmarks_3d=target_record.get(
                        "landmarks_3d",
                        record.get("landmarks_3d"),
                    ),
                    config=mesh_config,
                    faces=record["mesh"].get("faces", faces),
                ),
                faces=faces,
                config=mesh_config.get("hard_target_vertex_mask"),
            )
        loss_predicted_vertices = vertices + predicted_delta
        predicted_vertices = vertices_to_render_axes(
            loss_predicted_vertices,
            render_up_axis,
        ) - render_center

        prediction_panel = render_validation_mesh_panel(
            predicted_vertices,
            faces,
            texture=neutral_texture,
            image_size=render_size,
            backend=render_backend,
            render_kwargs=render_kwargs,
        )
        panels.append(prediction_panel)
        labels.append(f"{label} prediction")

        if show_target_columns:
            if target_delta is None:
                panels.append(blank_panel)
                labels.append(f"{label} error n/a")
            else:
                panels.append(
                    render_validation_mesh_panel(
                        predicted_vertices,
                        faces,
                        texture={
                            "vertex_colors": validation_error_vertex_colors(
                                predicted_delta,
                                target_delta,
                            )
                        },
                        image_size=render_size,
                        backend=render_backend,
                        render_kwargs=render_kwargs,
                    )
                )
                labels.append(f"{label} error")

        panel_rows.append(panels)
        label_rows.append(labels)
        if render_image_rows:
            flow_panels, flow_labels = render_validation_flow_row(
                label=label,
                record=record,
                neutral_vertices=vertices,
                predicted_vertices=loss_predicted_vertices,
                faces=faces,
                config=config,
                device=device,
                image_size=render_size,
                blank_panel=blank_panel,
            )
            panel_rows.append(flow_panels)
            label_rows.append(flow_labels)
        else:
            depth_panels, depth_labels = render_validation_depth_row(
                label=label,
                neutral_vertices=render_vertices,
                target_vertices=render_target_vertices,
                predicted_vertices=predicted_vertices,
                faces=faces,
                image_size=depth_render_size,
                show_target_columns=show_target_columns,
                blank_panel=depth_blank_panel,
            )
            panel_rows.append(depth_panels)
            label_rows.append(depth_labels)
        landmark_panels, landmark_labels = render_validation_landmark_row(
            label=label,
            neutral_vertices=render_vertices,
            target_vertices=render_target_vertices,
            predicted_vertices=predicted_vertices,
            landmarks_3d=record.get("landmarks_3d"),
            image_size=render_size,
            render_kwargs=render_kwargs,
            show_target_columns=show_target_columns,
            neutral_panel=neutral_panel,
            target_panel=target_panel,
            prediction_panel=prediction_panel,
            blank_panel=blank_panel,
        )
        panel_rows.append(landmark_panels)
        label_rows.append(landmark_labels)
    return concatenate_labeled_image_grid(panel_rows, label_rows)


def render_validation_mesh_panel(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    texture: Mapping[str, torch.Tensor],
    image_size: tuple[int, int],
    backend: str,
    render_kwargs: Mapping[str, Any],
) -> Image.Image:
    from utils.render_utils import differentiable_render

    rendered = differentiable_render(
        vertices,
        faces,
        texture=texture,
        image_size=image_size,
        backend=backend,
        **render_kwargs,
    )
    rendered = ensure_batched_hwc(rendered, "rendered")
    return tensor_hwc_to_pil(rendered[0])


def render_validation_flow_row(
    label: str,
    record: Mapping[str, Any],
    neutral_vertices: torch.Tensor,
    predicted_vertices: torch.Tensor,
    faces: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    image_size: tuple[int, int],
    blank_panel: Image.Image,
) -> tuple[list[Image.Image], list[str]]:
    try:
        flow_config = config["loss"].get("image", {}).get("flow", {})
        neutral_render = render_image_loss_rgb(
            vertices=neutral_vertices,
            reference_vertices=neutral_vertices,
            faces=faces,
            config=config,
            image_size=image_size,
        )
        predicted_flow = model_delta_optical_flow(
            vertices=neutral_vertices,
            deformed_vertices=predicted_vertices,
            faces=faces,
            flow_config=flow_config,
            render_config=config["rendering"],
            image_size=image_size,
        )
        target_flow, target_mask = prepare_image_flow_target(
            record=record,
            image_size=image_size,
            flow_config=flow_config,
            render_config=config.get("rendering", {}),
            neutral_render=neutral_render,
            device=device,
            dtype=predicted_flow.dtype,
        )
        predicted_flow = resize_flow_to_size(predicted_flow, target_flow.shape[-2:])
        target_mask_hwc = target_mask.permute(0, 2, 3, 1)
        panels = [
            resize_validation_panel(
                tensor_hwc_to_pil(
                    flow_rgb_overlay_on_image(
                        target_flow[0],
                        target_mask_hwc[0],
                        neutral_render[0],
                    )
                ),
                blank_panel.size,
            ),
            resize_validation_panel(
                tensor_hwc_to_pil(
                    flow_rgb_overlay_on_image(
                        predicted_flow[0],
                        target_mask_hwc[0],
                        neutral_render[0],
                    )
                ),
                blank_panel.size,
            ),
        ]
        labels = [f"{label} target flow", f"{label} prediction flow"]
        return panels, labels
    except Exception as exc:
        print(f"[WARNING] Could not render validation optical flow row: {exc}")
        return [blank_panel, blank_panel], [
            f"{label} target flow n/a",
            f"{label} prediction flow n/a",
        ]


def resize_validation_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if image.size == size:
        return image
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return image.resize(size, resampling)


def render_image_loss_rgb(
    vertices: torch.Tensor,
    reference_vertices: torch.Tensor,
    faces: torch.Tensor,
    config: Mapping[str, Any],
    image_size: tuple[int, int],
) -> torch.Tensor:
    from utils.render_utils import differentiable_render

    render_config = config["rendering"]
    render_kwargs = image_loss_render_kwargs(render_config)
    fit_reference = image_loss_fit_reference(
        reference_vertices,
        render_config,
        render_kwargs,
    )
    rendered = differentiable_render(
        vertices,
        faces,
        texture={"vertex_colors": vertex_colors(reference_vertices)},
        image_size=image_size,
        backend=str(render_config.get("backend", "auto")),
        fit_to_view_reference=fit_reference,
        **render_kwargs,
    )
    return ensure_batched_hwc(rendered, "image_loss_render")


def render_validation_depth_row(
    label: str,
    neutral_vertices: torch.Tensor,
    target_vertices: Optional[torch.Tensor],
    predicted_vertices: torch.Tensor,
    faces: torch.Tensor,
    image_size: tuple[int, int],
    show_target_columns: bool,
    blank_panel: Image.Image,
) -> tuple[list[Image.Image], list[str]]:
    try:
        depth_maps = render_validation_depth_maps(
            neutral_vertices=neutral_vertices,
            target_vertices=target_vertices,
            predicted_vertices=predicted_vertices,
            faces=faces,
            image_size=image_size,
        )
    except (ImportError, RuntimeError):
        panels = [blank_panel]
        labels = [f"{label} neutral depth n/a"]
        if show_target_columns:
            panels.append(blank_panel)
            labels.append(f"{label} target depth n/a")
        panels.append(blank_panel)
        labels.append(f"{label} prediction depth n/a")
        if show_target_columns:
            panels.append(blank_panel)
            labels.append(f"{label} depth error n/a")
        return panels, labels

    panels = [depth_maps["neutral"]]
    labels = [f"{label} neutral depth"]
    if show_target_columns:
        panels.append(depth_maps.get("target", blank_panel))
        labels.append(
            f"{label} target depth"
            if "target" in depth_maps
            else f"{label} target depth n/a"
        )
    panels.append(depth_maps["prediction"])
    labels.append(f"{label} prediction depth")
    if show_target_columns:
        panels.append(depth_maps.get("error", blank_panel))
        labels.append(
            f"{label} depth error"
            if "error" in depth_maps
            else f"{label} depth error n/a"
        )
    return panels, labels


def render_validation_depth_maps(
    neutral_vertices: torch.Tensor,
    target_vertices: Optional[torch.Tensor],
    predicted_vertices: torch.Tensor,
    faces: torch.Tensor,
    image_size: tuple[int, int],
) -> dict[str, Image.Image]:
    depth_vertices = [neutral_vertices, predicted_vertices]
    if target_vertices is not None:
        depth_vertices.append(target_vertices)

    center, scale = depth_projection_transform(
        neutral_vertices,
        image_size=image_size,
        up_axis="y",
    )
    depth_bounds = depth_projection_depth_bounds(
        torch.cat([item.detach() for item in depth_vertices], dim=1),
        center=center,
        up_axis="y",
    )

    neutral_depth, neutral_coverage = render_rasterized_depth_map(
        neutral_vertices,
        faces,
        image_size=image_size,
        center=center,
        scale=scale,
        depth_bounds=depth_bounds,
        up_axis="y",
    )
    predicted_depth, predicted_coverage = render_rasterized_depth_map(
        predicted_vertices,
        faces,
        image_size=image_size,
        center=center,
        scale=scale,
        depth_bounds=depth_bounds,
        up_axis="y",
    )

    depths = [neutral_depth, predicted_depth]
    coverages = [neutral_coverage > 0.0, predicted_coverage > 0.0]
    target_depth = None
    target_coverage = None
    if target_vertices is not None:
        target_depth, target_coverage = render_rasterized_depth_map(
            target_vertices,
            faces,
            image_size=image_size,
            center=center,
            scale=scale,
            depth_bounds=depth_bounds,
            up_axis="y",
        )
        depths.append(target_depth)
        coverages.append(target_coverage > 0.0)

    depth_min, depth_max = depth_color_range(depths, coverages)
    images = {
        "neutral": depth_map_to_pil(
            neutral_depth[0],
            coverages[0][0],
            value_min=depth_min,
            value_max=depth_max,
        ),
        "prediction": depth_map_to_pil(
            predicted_depth[0],
            coverages[1][0],
            value_min=depth_min,
            value_max=depth_max,
        ),
    }
    if target_depth is not None and target_coverage is not None:
        target_mask = target_coverage > 0.0
        images["target"] = depth_map_to_pil(
            target_depth[0],
            target_mask[0],
            value_min=depth_min,
            value_max=depth_max,
        )
        valid = coverages[0] & coverages[1] & target_mask
        depth_error = (predicted_depth - neutral_depth) - (target_depth - neutral_depth)
        error = depth_error.abs()
        error_values = error[valid]
        error_max = error_values.max() if error_values.numel() > 0 else error.max()
        images["error"] = depth_map_to_pil(
            error[0],
            valid[0],
            value_min=error.new_zeros(()),
            value_max=error_max.clamp_min(1.0e-8),
        )
    return images


def render_validation_landmark_row(
    label: str,
    neutral_vertices: torch.Tensor,
    target_vertices: Optional[torch.Tensor],
    predicted_vertices: torch.Tensor,
    landmarks_3d: Any,
    image_size: tuple[int, int],
    render_kwargs: Mapping[str, Any],
    show_target_columns: bool,
    neutral_panel: Image.Image,
    target_panel: Image.Image,
    prediction_panel: Image.Image,
    blank_panel: Image.Image,
) -> tuple[list[Image.Image], list[str]]:
    vertex_ids = validation_landmark_vertex_ids(
        landmarks_3d,
        device=neutral_vertices.device,
    )
    if vertex_ids is None:
        panels = [blank_panel]
        labels = [f"{label} neutral landmarks n/a"]
        if show_target_columns:
            panels.append(blank_panel)
            labels.append(f"{label} target landmarks n/a")
        panels.append(blank_panel)
        labels.append(f"{label} prediction landmarks n/a")
        if show_target_columns:
            panels.append(blank_panel)
            labels.append(f"{label} gt/pred landmarks n/a")
        return panels, labels

    neutral_points = validation_project_landmarks_to_screen(
        vertices=neutral_vertices,
        vertex_ids=vertex_ids,
        image_size=image_size,
        render_kwargs=render_kwargs,
        fit_reference_vertices=neutral_vertices,
    )
    predicted_points = validation_project_landmarks_to_screen(
        vertices=predicted_vertices,
        vertex_ids=vertex_ids,
        image_size=image_size,
        render_kwargs=render_kwargs,
        fit_reference_vertices=predicted_vertices,
    )

    panels = [
        draw_landmark_overlay(
            neutral_panel,
            neutral_points=neutral_points,
        )
    ]
    labels = [f"{label} neutral landmarks"]

    if show_target_columns:
        if target_vertices is None:
            target_points = None
            panels.append(blank_panel)
            labels.append(f"{label} target landmarks n/a")
        else:
            target_points = validation_project_landmarks_to_screen(
                vertices=target_vertices,
                vertex_ids=vertex_ids,
                image_size=image_size,
                render_kwargs=render_kwargs,
                fit_reference_vertices=target_vertices,
            )
            panels.append(
                draw_landmark_overlay(
                    target_panel,
                    ground_truth_points=target_points,
                )
            )
            labels.append(f"{label} target landmarks")
    else:
        target_points = None

    panels.append(
        draw_landmark_overlay(
            prediction_panel,
            predicted_points=predicted_points,
        )
    )
    labels.append(f"{label} prediction landmarks")

    if show_target_columns:
        if target_vertices is None:
            panels.append(blank_panel)
            labels.append(f"{label} gt/pred landmarks n/a")
        else:
            comparison_target_points = validation_project_landmarks_to_screen(
                vertices=target_vertices,
                vertex_ids=vertex_ids,
                image_size=image_size,
                render_kwargs=render_kwargs,
                fit_reference_vertices=neutral_vertices,
            )
            comparison_predicted_points = validation_project_landmarks_to_screen(
                vertices=predicted_vertices,
                vertex_ids=vertex_ids,
                image_size=image_size,
                render_kwargs=render_kwargs,
                fit_reference_vertices=neutral_vertices,
            )
            panels.append(
                draw_landmark_overlay(
                    neutral_panel,
                    ground_truth_points=comparison_target_points,
                    predicted_points=comparison_predicted_points,
                    connect_ground_truth_to_prediction=True,
                )
            )
            labels.append(f"{label} gt/pred landmarks")

    return panels, labels


def validation_landmark_vertex_ids(
    landmarks_3d: Any,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not isinstance(landmarks_3d, Mapping):
        return None
    vertex_ids = landmarks_3d.get("vertex_ids")
    if not isinstance(vertex_ids, torch.Tensor) or vertex_ids.numel() == 0:
        return None
    return vertex_ids.to(device=device, dtype=torch.long)


def validation_project_landmarks_to_screen(
    vertices: torch.Tensor,
    vertex_ids: torch.Tensor,
    image_size: tuple[int, int],
    render_kwargs: Mapping[str, Any],
    fit_reference_vertices: torch.Tensor,
) -> torch.Tensor:
    vertex_ids = vertex_ids.to(device=vertices.device, dtype=torch.long)
    landmark_vertices = vertices.index_select(1, vertex_ids)
    projected = validation_project_points_to_screen(
        points=landmark_vertices,
        image_size=image_size,
        render_kwargs=render_kwargs,
        fit_reference_vertices=fit_reference_vertices,
    )
    return projected[0]


def validation_project_points_to_screen(
    points: torch.Tensor,
    image_size: tuple[int, int],
    render_kwargs: Mapping[str, Any],
    fit_reference_vertices: torch.Tensor,
) -> torch.Tensor:
    from utils.render_utils import (
        CAMERA_EXTRINSICS,
        CAMERA_INTRINSICS,
        DEFAULT_VIEW_OCCUPANCY,
    )

    height, width = image_size
    if points.dim() != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B, N, 3].")
    if fit_reference_vertices.dim() != 3 or fit_reference_vertices.shape[-1] != 3:
        raise ValueError("fit_reference_vertices must have shape [B, V, 3].")

    base_height, base_width = CAMERA_INTRINSICS["image_size"]
    base_fx, base_fy = CAMERA_INTRINSICS["focal_length"]
    base_px, base_py = CAMERA_INTRINSICS["principal_point"]
    fx = points.new_tensor(base_fx * width / float(base_width))
    fy = points.new_tensor(base_fy * height / float(base_height))
    px = points.new_tensor(base_px * width / float(base_width))
    py = points.new_tensor(base_py * height / float(base_height))

    projected = validation_camera_project(
        validation_world_to_camera(points, CAMERA_EXTRINSICS),
        fx=fx,
        fy=fy,
        px=px,
        py=py,
    )
    if not bool(render_kwargs.get("fit_to_view", True)):
        return projected[..., :2]

    reference_projected = validation_camera_project(
        validation_world_to_camera(fit_reference_vertices, CAMERA_EXTRINSICS),
        fx=fx,
        fy=fy,
        px=px,
        py=py,
    )
    occupancy = float(render_kwargs.get("view_occupancy", DEFAULT_VIEW_OCCUPANCY))
    reference_xy = reference_projected[..., :2]
    mins = reference_xy.amin(dim=1, keepdim=True)
    maxs = reference_xy.amax(dim=1, keepdim=True)
    center = (mins + maxs) * 0.5
    max_span = (maxs - mins).amax(dim=-1, keepdim=True).clamp_min(1.0e-6)
    target_span = min(height, width) * occupancy
    scale = projected.new_tensor(target_span) / max_span
    image_center = projected.new_tensor(
        ((width - 1) * 0.5, (height - 1) * 0.5)
    ).view(1, 1, 2)
    return (projected[..., :2] - center) * scale + image_center


def validation_world_to_camera(
    vertices: torch.Tensor,
    camera_extrinsics: Mapping[str, Any],
) -> torch.Tensor:
    batch_size = vertices.shape[0]
    rotation = vertices.new_tensor(camera_extrinsics["R"]).unsqueeze(0)
    translation = vertices.new_tensor(camera_extrinsics["T"]).view(1, 1, 3)
    return torch.bmm(vertices, rotation.expand(batch_size, -1, -1)) + translation


def validation_camera_project(
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


def draw_landmark_overlay(
    base: Image.Image,
    neutral_points: Optional[torch.Tensor] = None,
    ground_truth_points: Optional[torch.Tensor] = None,
    predicted_points: Optional[torch.Tensor] = None,
    connect_ground_truth_to_prediction: bool = False,
) -> Image.Image:
    image = base.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    radius = max(1, int(round(min(image.size) * 0.0045)))

    if (
        connect_ground_truth_to_prediction
        and ground_truth_points is not None
        and predicted_points is not None
    ):
        draw_landmark_connections(
            draw,
            ground_truth_points,
            predicted_points,
            image.size,
            color=(255, 193, 7, 120),
            width=max(1, radius),
        )

    if neutral_points is not None:
        draw_landmark_points(
            draw,
            neutral_points,
            image.size,
            radius=radius,
            outline=(26, 70, 150, 215),
        )
    if ground_truth_points is not None:
        draw_landmark_points(
            draw,
            ground_truth_points,
            image.size,
            radius=radius,
            outline=(0, 135, 96, 235),
        )
    if predicted_points is not None:
        draw_landmark_points(
            draw,
            predicted_points,
            image.size,
            radius=radius,
            outline=(190, 30, 96, 235),
        )

    return Image.alpha_composite(image, overlay).convert("RGB")


def draw_landmark_connections(
    draw: ImageDraw.ImageDraw,
    start_points: torch.Tensor,
    end_points: torch.Tensor,
    image_size: tuple[int, int],
    color: tuple[int, int, int, int],
    width: int,
) -> None:
    width_px, height_px = image_size
    for start, end in zip(
        validation_screen_points(start_points, width_px, height_px),
        validation_screen_points(end_points, width_px, height_px),
    ):
        if start is None or end is None:
            continue
        draw.line((start[0], start[1], end[0], end[1]), fill=color, width=width)


def draw_landmark_points(
    draw: ImageDraw.ImageDraw,
    points: torch.Tensor,
    image_size: tuple[int, int],
    radius: int,
    outline: tuple[int, int, int, int],
) -> None:
    width_px, height_px = image_size
    outline_radius = radius + 1
    screen_points = validation_screen_points(points, width_px, height_px, radius + 2)
    visible_y = [point[1] for point in screen_points if point is not None]
    y_min = min(visible_y) if visible_y else 0.0
    y_max = max(visible_y) if visible_y else float(max(height_px - 1, 1))
    for point in screen_points:
        if point is None:
            continue
        x, y = point
        fill = vertical_landmark_gradient_color(y, y_min, y_max)
        draw.ellipse(
            (
                x - outline_radius,
                y - outline_radius,
                x + outline_radius,
                y + outline_radius,
            ),
            fill=outline,
        )
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def vertical_landmark_gradient_color(
    y: float,
    y_min: float,
    y_max: float,
) -> tuple[int, int, int, int]:
    span = max(y_max - y_min, 1.0e-6)
    value = max(0.0, min(1.0, (y - y_min) / span))
    if value < 0.5:
        mix = value / 0.5
        return interpolate_rgba((43, 123, 255, 235), (0, 205, 150, 235), mix)
    mix = (value - 0.5) / 0.5
    return interpolate_rgba((0, 205, 150, 235), (255, 101, 67, 235), mix)


def interpolate_rgba(
    start: tuple[int, int, int, int],
    end: tuple[int, int, int, int],
    mix: float,
) -> tuple[int, int, int, int]:
    mix = max(0.0, min(1.0, mix))
    return tuple(
        int(round(start[channel] * (1.0 - mix) + end[channel] * mix))
        for channel in range(4)
    )


def validation_screen_points(
    points: torch.Tensor,
    width: int,
    height: int,
    margin: int = 8,
) -> list[Optional[tuple[float, float]]]:
    result: list[Optional[tuple[float, float]]] = []
    for point in points.detach().cpu():
        if point.numel() < 2 or not bool(torch.isfinite(point[:2]).all()):
            result.append(None)
            continue
        x = float(point[0])
        y = float(point[1])
        if x < -margin or x > width + margin or y < -margin or y > height + margin:
            result.append(None)
            continue
        result.append((x, y))
    return result


def depth_color_range(
    depths: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    values = [
        depth[mask]
        for depth, mask in zip(depths, masks)
        if bool(mask.any().detach().cpu())
    ]
    if not values:
        fallback = depths[0].new_zeros(())
        return fallback, fallback + 1.0

    stacked = torch.cat([value.reshape(-1) for value in values])
    value_min = stacked.min()
    value_max = stacked.max()
    if not bool((value_max > value_min).item()):
        value_max = value_min + 1.0
    return value_min, value_max


def depth_map_to_pil(
    depth: torch.Tensor,
    valid: torch.Tensor,
    value_min: torch.Tensor,
    value_max: torch.Tensor,
) -> Image.Image:
    value = ((depth - value_min) / (value_max - value_min).clamp_min(1.0e-8)).clamp(
        0.0,
        1.0,
    )
    rgb = magma_like_colormap(value)
    background = rgb.new_ones(rgb.shape)
    rgb = torch.where(valid.unsqueeze(-1), rgb, background)
    return tensor_hwc_to_pil(rgb)


def magma_like_colormap(value: torch.Tensor) -> torch.Tensor:
    palette = value.new_tensor(
        (
            (0.001, 0.000, 0.014),
            (0.170, 0.045, 0.340),
            (0.525, 0.120, 0.455),
            (0.878, 0.311, 0.220),
            (0.988, 0.729, 0.421),
            (0.987, 0.991, 0.749),
        )
    )
    scaled = value * float(palette.shape[0] - 1)
    lower = scaled.floor().long().clamp(0, palette.shape[0] - 1)
    upper = (lower + 1).clamp(0, palette.shape[0] - 1)
    mix = (scaled - lower.to(dtype=value.dtype)).unsqueeze(-1)
    return palette[lower] * (1.0 - mix) + palette[upper] * mix


def validation_target_delta(
    record: Mapping[str, Any],
    action_unit: int,
    device: torch.device,
    dtype: torch.dtype,
    expected_vertex_count: Optional[int] = None,
) -> Optional[torch.Tensor]:
    if "delta_vertices" not in record or "action_unit_id" not in record:
        return None
    if as_int(record["action_unit_id"]) != action_unit:
        return None
    delta_vertices = record["delta_vertices"]
    if not isinstance(delta_vertices, torch.Tensor):
        return None
    if delta_vertices.ndim != 2 or delta_vertices.shape[-1] != 3:
        return None
    if (
        expected_vertex_count is not None
        and delta_vertices.shape[0] != expected_vertex_count
    ):
        return None
    return delta_vertices.to(device=device, dtype=dtype).unsqueeze(0)


def validation_error_vertex_colors(
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
) -> torch.Tensor:
    error_norm = torch.linalg.vector_norm(predicted_delta - target_delta, dim=-1)
    robust_max = torch.quantile(error_norm.flatten(start_dim=1), 0.98, dim=1)
    value = (error_norm / robust_max.clamp_min(1.0e-8).view(-1, 1)).clamp(0.0, 1.0)
    low = predicted_delta.new_tensor((0.05, 0.06, 0.08))
    high = predicted_delta.new_tensor((1.0, 0.16, 0.02))
    return low.view(1, 1, 3) * (1.0 - value.unsqueeze(-1)) + high.view(
        1,
        1,
        3,
    ) * value.unsqueeze(-1)


def action_unit_label(action_unit: int) -> str:
    return f"{action_unit}: {AU_NAME.get(action_unit, 'unknown')}"


def resolve_validation_render_up_axis(
    config: Mapping[str, Any],
    vertices: torch.Tensor,
) -> str:
    visualization_config = config.get("visualization", {}) or {}
    up_axis = str(visualization_config.get("mesh_up_axis", "auto")).strip().lower()
    if up_axis == "auto":
        return infer_mesh_up_axis(vertices)
    if up_axis in {"y", "y-up", "y_up", "render"}:
        return "y"
    if up_axis in {"z", "z-up", "z_up", "blender", "fbx"}:
        return "z"
    raise ValueError("visualization.mesh_up_axis must be 'auto', 'y', or 'z'.")


def infer_mesh_up_axis(vertices: torch.Tensor) -> str:
    bounds_min = vertices.detach().amin(dim=-2)
    bounds_max = vertices.detach().amax(dim=-2)
    spans = (bounds_max - bounds_min).reshape(-1, 3).mean(dim=0)
    return "z" if bool((spans[2] > spans[1]).item()) else "y"


def vertices_to_render_axes(vertices: torch.Tensor, up_axis: str) -> torch.Tensor:
    if up_axis == "y":
        return vertices
    if up_axis != "z":
        raise ValueError("up_axis must be 'y' or 'z'.")
    return torch.stack(
        (
            -vertices[..., 0],
            vertices[..., 2],
            vertices[..., 1],
        ),
        dim=-1,
    )


def render_vertex_bounds_center(vertices: torch.Tensor) -> torch.Tensor:
    bounds_min = vertices.detach().amin(dim=-2, keepdim=True)
    bounds_max = vertices.detach().amax(dim=-2, keepdim=True)
    return (bounds_min + bounds_max) * 0.5


def validation_vertex_colors(
    vertices: torch.Tensor,
    normals: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if normals is None:
        normals = F.normalize(vertices.detach(), dim=-1, eps=1.0e-6)
    else:
        normals = normals.to(device=vertices.device, dtype=vertices.dtype)
        normal_lengths = normals.norm(dim=-1, keepdim=True)
        fallback_normals = F.normalize(vertices.detach(), dim=-1, eps=1.0e-6)
        normals = torch.where(
            normal_lengths > 1.0e-6,
            normals / normal_lengths.clamp_min(1.0e-6),
            fallback_normals,
        )

    key_light = F.normalize(vertices.new_tensor((-0.35, 0.45, -0.82)), dim=0)
    fill_light = F.normalize(vertices.new_tensor((0.55, 0.10, -0.55)), dim=0)
    rim_light = F.normalize(vertices.new_tensor((0.10, 0.75, 0.62)), dim=0)

    key = normals.matmul(key_light).abs().clamp(0.0, 1.0)
    fill = normals.matmul(fill_light).abs().clamp(0.0, 1.0)
    rim = normals.matmul(rim_light).abs().clamp(0.0, 1.0).pow(2.0)
    shade = (0.30 + 0.52 * key + 0.16 * fill + 0.14 * rim).clamp(0.0, 1.0)

    bounds_min = vertices.amin(dim=1, keepdim=True)
    bounds_max = vertices.amax(dim=1, keepdim=True)
    normalized = (vertices - bounds_min) / (bounds_max - bounds_min).clamp_min(
        1.0e-6
    )
    height = normalized[..., 1]
    depth = normalized[..., 2]
    side = normalized[..., 0]
    base_color = torch.stack(
        (
            0.72 + 0.10 * height + 0.04 * (1.0 - depth),
            0.54 + 0.07 * height + 0.03 * side,
            0.46 + 0.04 * side + 0.04 * (1.0 - depth),
        ),
        dim=-1,
    )
    normal_tint = normals.mul(0.5).add(0.5) * vertices.new_tensor(
        (0.08, 0.06, 0.10)
    )
    return (base_color * shade.unsqueeze(-1) + normal_tint).clamp(0.0, 1.0)


def stack_visualization_sections(
    sections: Sequence[tuple[str, Image.Image]],
) -> Image.Image:
    if not sections:
        raise ValueError("At least one visualization section is required.")
    if len(sections) == 1:
        return sections[0][1].convert("RGB")

    label_height = 30
    gutter = 12
    width = max(image.width for _label, image in sections)
    height = (
        sum(image.height + label_height for _label, image in sections)
        + gutter * (len(sections) - 1)
    )
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y = 0
    for label, image in sections:
        draw.text((6, y + 8), label, fill=(0, 0, 0))
        canvas.paste(image.convert("RGB"), (0, y + label_height))
        y += label_height + image.height + gutter
    return canvas


def concatenate_labeled_image_grid(
    image_rows: Sequence[Sequence[Image.Image]],
    label_rows: Sequence[Sequence[str]],
) -> Image.Image:
    if len(image_rows) != len(label_rows):
        raise ValueError("image_rows and label_rows must have the same length.")
    if not image_rows:
        raise ValueError("At least one image row is required.")

    column_count = len(image_rows[0])
    if column_count == 0:
        raise ValueError("At least one image is required.")
    if any(len(row) != column_count for row in image_rows):
        raise ValueError("All image rows must have the same number of columns.")
    if any(len(row) != column_count for row in label_rows):
        raise ValueError("All label rows must match the image column count.")

    label_height = 28
    gutter = 8
    column_widths = [
        max(row[column].width for row in image_rows) for column in range(column_count)
    ]
    row_heights = [max(image.height for image in row) for row in image_rows]
    width = sum(column_widths) + gutter * (column_count - 1)
    height = (
        sum(row_height + label_height for row_height in row_heights)
        + gutter * (len(image_rows) - 1)
    )
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y = 0
    for images, labels, row_height in zip(image_rows, label_rows, row_heights):
        x = 0
        for column, (image, label) in enumerate(zip(images, labels)):
            canvas.paste(image.convert("RGB"), (x, y + label_height))
            draw.text((x + 6, y + 7), label, fill=(0, 0, 0))
            x += column_widths[column] + gutter
        y += row_height + label_height + gutter
    return canvas


def ensure_batched_hwc(image: torch.Tensor, name: str) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4 or image.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [H, W, 3] or [B, H, W, 3].")
    return image


def flow_to_rgb(
    flow: torch.Tensor,
    robust_max: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if flow.dim() == 3:
        flow_batch = flow.unsqueeze(0)
        was_unbatched = True
    elif flow.dim() == 4:
        flow_batch = flow
        was_unbatched = False
    else:
        raise ValueError("Expected flow shape [2, H, W] or [B, 2, H, W].")
    if flow_batch.shape[1] != 2:
        raise ValueError("Expected flow shape [2, H, W] or [B, 2, H, W].")

    dx = flow_batch[:, 0].detach()
    dy = flow_batch[:, 1].detach()
    magnitude = torch.sqrt(dx * dx + dy * dy)
    if robust_max is None:
        flat_magnitude = magnitude.flatten(start_dim=1)
        robust_max = torch.quantile(flat_magnitude, 0.99, dim=1).clamp_min(1.0e-6)
    else:
        robust_max = robust_max.to(device=flow_batch.device, dtype=flow_batch.dtype)
        if robust_max.dim() == 0:
            robust_max = robust_max.view(1).expand(flow_batch.shape[0])
        robust_max = robust_max.reshape(-1).clamp_min(1.0e-6)
        if robust_max.numel() == 1 and flow_batch.shape[0] > 1:
            robust_max = robust_max.expand(flow_batch.shape[0])
        if robust_max.numel() != flow_batch.shape[0]:
            raise ValueError("robust_max must be scalar or match the flow batch size.")
    value = (magnitude / robust_max.view(-1, 1, 1)).clamp(0.0, 1.0)
    hue = (torch.atan2(dy, dx) + math.pi) / (2.0 * math.pi)

    red = 0.5 + 0.5 * torch.cos(2.0 * math.pi * hue)
    green = 0.5 + 0.5 * torch.cos(2.0 * math.pi * (hue - 1.0 / 3.0))
    blue = 0.5 + 0.5 * torch.cos(2.0 * math.pi * (hue + 1.0 / 3.0))
    direction_rgb = torch.stack((red, green, blue), dim=-1)
    rgb = direction_rgb * value.unsqueeze(-1)
    if was_unbatched:
        return rgb[0]
    return rgb


def flow_rgb_overlay_on_image(
    flow: torch.Tensor,
    mask: torch.Tensor,
    image: torch.Tensor,
    alpha: float = 0.85,
) -> torch.Tensor:
    rgb = flow_to_rgb(flow)
    mask_hwc = normalize_flow_mask_hwc(mask, rgb)
    base = ensure_hwc_rgb(image, "flow_overlay_image")
    base = base.to(device=rgb.device, dtype=rgb.dtype).clamp(0.0, 1.0)
    if tuple(base.shape[:2]) != tuple(rgb.shape[:2]):
        base = (
            F.interpolate(
                base.permute(2, 0, 1).unsqueeze(0),
                size=tuple(rgb.shape[:2]),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )
    blend = (mask_hwc * float(alpha)).clamp(0.0, 1.0)
    return base * (1.0 - blend) + rgb * blend


def normalize_flow_mask_hwc(mask: torch.Tensor, reference_rgb: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 3 and mask.shape[-1] == 1:
        mask_hwc = mask
    elif mask.dim() == 2:
        mask_hwc = mask.unsqueeze(-1)
    elif mask.dim() == 3 and mask.shape[0] == 1:
        mask_hwc = mask.permute(1, 2, 0)
    else:
        raise ValueError("mask must have shape [H, W], [H, W, 1], or [1, H, W].")
    mask_hwc = mask_hwc.to(
        device=reference_rgb.device,
        dtype=reference_rgb.dtype,
    ).clamp(0.0, 1.0)
    if tuple(mask_hwc.shape[:2]) != tuple(reference_rgb.shape[:2]):
        mask_hwc = (
            F.interpolate(
                mask_hwc.permute(2, 0, 1).unsqueeze(0),
                size=tuple(reference_rgb.shape[:2]),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )
    return mask_hwc


def ensure_hwc_rgb(image: torch.Tensor, name: str) -> torch.Tensor:
    if image.dim() == 4:
        if image.shape[0] != 1:
            raise ValueError(f"{name} must be unbatched or have batch size 1.")
        image = image[0]
    if image.dim() != 3:
        raise ValueError(f"{name} must have shape [H, W, 3] or [3, H, W].")
    if image.shape[-1] == 3:
        return image
    if image.shape[0] == 3:
        return image.permute(1, 2, 0)
    raise ValueError(f"{name} must have 3 RGB channels.")


def tensor_hwc_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0.0, 1.0)
    if image.dim() == 4:
        image = image[0]
    image_u8 = (image * 255.0).round().to(torch.uint8).contiguous()
    height, width, channels = image_u8.shape
    if channels != 3:
        raise ValueError("Expected an RGB image tensor.")
    return Image.frombytes("RGB", (width, height), image_u8.numpy().tobytes())


def vertex_colors(vertices: torch.Tensor) -> torch.Tensor:
    mins = vertices.amin(dim=1, keepdim=True)
    maxs = vertices.amax(dim=1, keepdim=True)
    return ((vertices - mins) / (maxs - mins).clamp_min(1.0e-6)).detach()


def normalize_image_size(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError("Image size must be an int or [height, width].")


def to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {name: to_device(item, device) for name, item in value.items()}
    return value


def as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    scheduler: Any,
    epoch: int,
    global_step: int,
    best_val_loss: Optional[float] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "config": dict(config),
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if best_val_loss is not None and math.isfinite(float(best_val_loss)):
        checkpoint["best_val_loss"] = float(best_val_loss)
        checkpoint["best_val_metric"] = float(best_val_loss)
        checkpoint["best_val_metric_name"] = validation_primary_metric_name(config)
    torch.save(checkpoint, path)


def maybe_load_initial_model_checkpoint(
    model: nn.Module,
    config: Mapping[str, Any],
    device: torch.device,
) -> Optional[Path]:
    training_config = config.get("training", {})
    configured_path = training_config.get("initial_checkpoint")
    resume_path = training_config.get("resume_checkpoint")
    if configured_path not in (None, "") and resume_path not in (None, ""):
        raise ValueError(
            "training.initial_checkpoint and training.resume_checkpoint cannot "
            "both be set."
        )
    if resume_path not in (None, ""):
        return None
    if configured_path in (None, ""):
        return None
    checkpoint_path = Path(str(configured_path)).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Initial model checkpoint does not exist: {checkpoint_path}"
        )
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Initial checkpoint must contain a mapping.")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise ValueError("Initial checkpoint has no model_state_dict mapping.")
    strict = bool(training_config.get("initial_checkpoint_strict", True))
    incompatible = model.load_state_dict(state_dict, strict=strict)
    print(
        "[INFO] Loaded initial model checkpoint "
        f"{checkpoint_path} strict={strict} "
        f"missing={len(incompatible.missing_keys)} "
        f"unexpected={len(incompatible.unexpected_keys)}",
        flush=True,
    )
    return checkpoint_path


def maybe_resume_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[int, int, float]:
    training_config = config.get("training", {})
    configured_path = training_config.get("resume_checkpoint")
    if configured_path in (None, ""):
        return 1, 0, float("inf")

    checkpoint_path = Path(str(configured_path)).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: {checkpoint_path}"
        )
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Resume checkpoint must contain a mapping.")

    required_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "global_step",
    }
    missing_keys = sorted(required_keys - set(checkpoint))
    if missing_keys:
        raise ValueError(
            "Resume checkpoint is missing required state: "
            + ", ".join(missing_keys)
        )

    strict = bool(training_config.get("resume_checkpoint_strict", True))
    incompatible = unwrap_model(model).load_state_dict(
        checkpoint["model_state_dict"],
        strict=strict,
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is None and scheduler_state is not None:
        raise ValueError(
            "Resume checkpoint contains scheduler state, but the config has no "
            "scheduler."
        )
    if scheduler is not None:
        if scheduler_state is None:
            raise ValueError(
                "Resume checkpoint has no scheduler state, but the config enables "
                "a scheduler."
            )
        scheduler.load_state_dict(scheduler_state)

    completed_epoch = int(checkpoint["epoch"])
    global_step = int(checkpoint["global_step"])
    num_epochs = int(training_config["epochs"])
    if completed_epoch < 0 or completed_epoch > num_epochs:
        raise ValueError(
            f"Resume checkpoint epoch {completed_epoch} is outside the configured "
            f"0..{num_epochs} epoch range."
        )
    configured_best = training_config.get(
        "resume_best_val_metric",
        training_config.get("resume_best_val_loss"),
    )
    primary_metric_name = validation_primary_metric_name(config)
    checkpoint_metric_name = checkpoint.get("best_val_metric_name")
    if checkpoint_metric_name == primary_metric_name:
        stored_best = checkpoint.get(
            "best_val_metric",
            checkpoint.get("best_val_loss"),
        )
    elif checkpoint_metric_name is None and primary_metric_name == "loss":
        stored_best = checkpoint.get("best_val_loss")
    else:
        stored_best = None
        if checkpoint_metric_name is not None or "best_val_loss" in checkpoint:
            print(
                "[WARNING] Resume checkpoint best metric is incompatible with "
                f"validation.primary_metric={primary_metric_name!r}; resetting "
                "the best value for checkpoint selection.",
                flush=True,
            )
    best_val_loss = float(
        stored_best
        if stored_best is not None
        else configured_best if configured_best is not None else float("inf")
    )
    print(
        "[INFO] Resumed training checkpoint "
        f"{checkpoint_path} strict={strict} "
        f"completed_epoch={completed_epoch} global_step={global_step} "
        f"lr={current_learning_rate(optimizer):.8g} "
        f"missing={len(incompatible.missing_keys)} "
        f"unexpected={len(incompatible.unexpected_keys)}",
        flush=True,
    )
    return completed_epoch + 1, global_step, best_val_loss


def sync_metric_averager(
    averager: "MetricAverager",
    distributed: DistributedContext,
) -> dict[str, float]:
    if not distributed.enabled:
        return averager.compute()

    gathered: list[Optional[dict[str, dict[str, float]]]] = [
        None
    ] * distributed.world_size
    dist.all_gather_object(
        gathered,
        {
            "totals": {name: float(value) for name, value in averager.totals.items()},
            "counts": {name: float(value) for name, value in averager.counts.items()},
        },
    )

    totals: dict[str, float] = {}
    counts: dict[str, float] = {}
    for rank_payload in gathered:
        if rank_payload is None:
            continue
        for name, value in rank_payload["totals"].items():
            totals[name] = totals.get(name, 0.0) + float(value)
        for name, value in rank_payload["counts"].items():
            counts[name] = counts.get(name, 0.0) + float(value)

    return {
        name: totals[name] / max(counts.get(name, 0.0), 1.0)
        for name in sorted(totals)
    }


def format_metrics(prefix: str, metrics: Mapping[str, float]) -> str:
    parts = [f"{name}={value:.6f}" for name, value in sorted(metrics.items())]
    return f"{prefix} " + " ".join(parts)


def progress_iter(iterable, *, total: Optional[int], desc: str, disable: bool, unit: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, disable=disable, unit=unit)


def progress_write(message: str) -> None:
    if tqdm is None:
        print(message)
        return
    tqdm.write(message)


class MetricAverager:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, metrics: Mapping[str, float], weight: float = 1.0) -> None:
        weight = float(weight)
        for name, value in metrics.items():
            self.totals[name] = self.totals.get(name, 0.0) + float(value) * weight
            self.counts[name] = self.counts.get(name, 0) + weight

    def compute(self) -> dict[str, float]:
        return {
            name: total / max(self.counts[name], 1)
            for name, total in self.totals.items()
        }


if __name__ == "__main__":
    main()
