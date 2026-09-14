from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

import train
from scripts import animate_glb_eye_aus as complete_eye_au
from dataset.image_dataset import (
    _load_glb_for_model,
    _remap_landmarks_to_welded_mesh,
    _weld_neutral_mesh_payload,
)
from dataset.mesh_dataset import _empty_landmark_payload, _transform_model_sample
from model.toporig import TopoRig
from utils.eye_gaze_uv_warp import (
    eye_gaze_landmark_spec,
    point_in_polygon_2d,
    semantic_eye_gaze_motion,
    tapered_eye_uv_warp,
    uv_connected_polygon_component,
)
from utils.eye_gaze_texture_mask import (
    apply_iris_occlusion_cleanup,
    apply_iris_texture_translation,
    load_iris_texture_selection,
    save_iris_texture_selection,
    segment_iris_texture,
    selection_matches,
    write_iris_selection_debug,
)
from utils.eye_geometry import component_closedness, fit_ellipsoid, fit_sphere
from utils.fbx_to_tensor import AU_NAME, fbx_to_tensor
from utils.paths import runtime_path


ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = runtime_path("output")
DEFAULT_METAHUMAN_SAMPLE_DIR = runtime_path("reference", "metahuman")
DEFAULT_LANDMARK_CACHE_DIR = runtime_path("cache", "demo_landmarks")
DEFAULT_COMPLETE_EYE_CACHE_DIR = runtime_path("cache", "glb_eye_au_landmarks")
DEFAULT_MEDIAPIPE_MAPPER = ROOT / "map_mediapipe_landmarks.py"
COMPLETE_EYE_SOURCE_CACHE_VERSION = 1
SUPPORTED_INPUT_SUFFIXES = {".obj", ".glb", ".fbx"}
MAX_EYE_GAZE_UV_WIDTH = 0.25
MAX_EYE_GAZE_WARPED_LOOPS = 10_000
MAX_EYE_GAZE_SPATIAL_UV_WARPED_LOOPS = 20_000
MAX_PHYSICAL_EYE_COMPONENT_POLYGONS = 1_000
MAX_COPY_TEXTURE_EYE_COMPONENT_POLYGONS = 2_500
EYE_GAZE_WARP_RADIUS_EYE_WIDTHS = 0.68
EYE_GAZE_WARP_INNER_RADIUS = 0.35
MAX_PHYSICAL_EYE_LAYER_CENTER_LUMINANCE = 0.50
MAX_PHYSICAL_EYE_LAYER_DEPTH_EYE_WIDTHS = 2.0
MAX_PHYSICAL_EYE_ROTATION_DEGREES = 35.0
MAX_VISIBLE_IRIS_SHELL_DEPTH_EYE_WIDTHS = 0.12
MAX_PHYSICAL_EYE_LAYERS = 64
MAX_TEXTURE_GAZE_SHIFT_IRIS_RADII = 1.35
DEFAULT_EYE_RADIUS_EYE_WIDTHS = 0.55
JAW_OPEN_AU_ID = 26
RIGID_LOWER_TEETH_MEDIAPIPE_IDS = (148, 152, 377, 14, 17)
ICT_ORAL_MATERIAL_PATTERNS = ("teeth", "gumstongue")


class EyeGazeUVWarpValidationError(ValueError):
    """Raised before export when an eye-gaze UV mapping is unsafe to apply."""


def normalize_eye_gaze_appearance_mode(mode: str) -> str:
    """Resolve public gaze modes without permitting generated eye geometry."""

    normalized = str(mode)
    if normalized in {"screen_render_bake", "decal_eye"}:
        raise EyeGazeUVWarpValidationError(
            "Generated eye overlays and decals are disabled. Use mesh gaze or "
            "the original-surface texture fallback."
        )
    return {
        "mesh": "rotate_eye",
        "texture": "color_segmented_texture",
    }.get(normalized, normalized)


def complete_eye_gaze_enabled(
    args: argparse.Namespace, action_unit_id: int
) -> bool:
    """Use the complete-ball implementation for every supported gaze AU.

    The older gaze switches remain aliases so existing batch commands keep
    working, but AU12-AU19 now select the replacement algorithm automatically.
    """

    explicitly_requested = bool(
        getattr(args, "eye_gaze", False)
        or getattr(args, "eye_gaze_uv_warp", False)
        or getattr(args, "eye_gaze_mode", None) is not None
        or getattr(args, "eye_gaze_intensity", None) is not None
        or getattr(args, "eye_gaze_uv_strength", None) is not None
    )
    if getattr(args, "no_complete_eye_gaze", False):
        if explicitly_requested:
            raise ValueError(
                "--no-complete-eye-gaze cannot be combined with an eye-gaze "
                "enable/intensity option."
            )
        return False
    return action_unit_id in complete_eye_au.AU_SPECS or explicitly_requested


def validate_complete_eye_gaze_request(
    args: argparse.Namespace,
    *,
    mesh_path: Path,
    action_unit_id: int,
    enabled: bool,
) -> None:
    if not enabled:
        return
    if action_unit_id not in complete_eye_au.AU_SPECS:
        raise ValueError(
            f"AU{action_unit_id} is not a supported eye-gaze action unit. "
            "The complete-eye algorithm supports AU12-AU19."
        )
    if mesh_path.suffix.lower() not in {".glb", ".fbx"}:
        raise ValueError(
            "Complete-eye gaze requires a textured GLB or FBX source; OBJ "
            "does not retain the source eye texture."
        )
    if args.output_format != "glb":
        raise ValueError("Complete-eye gaze requires --output-format glb.")
    if args.no_preserve_appearance:
        raise ValueError(
            "Complete-eye gaze needs the source materials; remove "
            "--no-preserve-appearance."
        )
    if args.preserve_eye_gaze_geometry:
        raise ValueError(
            "--preserve-eye-gaze-geometry belongs to the replaced UV/mesh "
            "implementation and cannot be combined with complete-eye gaze."
        )
    if (
        args.eye_gaze_intensity is not None
        and args.eye_gaze_uv_strength is not None
    ):
        raise ValueError(
            "Pass either --eye-gaze-intensity or --eye-gaze-uv-strength, not both."
        )
    normalized_eye_gaze_intensity(args.eye_gaze_intensity, allow_none=True)
    if not math.isfinite(float(args.eye_gaze_duration)) or (
        args.eye_gaze_duration <= 0
    ):
        raise ValueError("--eye-gaze-duration must be positive.")
    if not math.isfinite(float(args.eye_gaze_fps)) or args.eye_gaze_fps <= 0:
        raise ValueError("--eye-gaze-fps must be positive.")
    if not math.isfinite(float(args.eye_gaze_uv_max_motion)) or (
        args.eye_gaze_uv_max_motion <= 0
    ):
        raise ValueError("--eye-gaze-uv-max-motion must be positive.")
    if not math.isfinite(float(args.eye_gaze_max_angle)) or not (
        0.0 < args.eye_gaze_max_angle <= 35.0
    ):
        raise ValueError(
            "--eye-gaze-max-angle must be greater than 0 and no more than 35."
        )
    if (
        args.eye_gaze_blender_threads is not None
        and args.eye_gaze_blender_threads <= 0
    ):
        raise ValueError("--eye-gaze-blender-threads must be positive.")


def normalized_eye_gaze_intensity(
    value: Optional[float], *, allow_none: bool = False
) -> Optional[float]:
    if value is None:
        if allow_none:
            return None
        raise ValueError("Eye-gaze intensity is required.")
    result = float(value)
    if not math.isfinite(result) or not -1.0 <= result <= 1.0:
        raise ValueError("Eye-gaze intensity must be between -1 and 1.")
    return result


def eye_gaze_motion_to_rotation_intensity(
    *,
    action_unit_id: int,
    motion: torch.Tensor,
    max_angle_degrees: float,
    eye_radius_eye_widths: float = DEFAULT_EYE_RADIUS_EYE_WIDTHS,
) -> float:
    """Convert the legacy screen displacement into rigid rotation strength."""

    if action_unit_id not in complete_eye_au.AU_SPECS:
        raise ValueError(f"AU{action_unit_id} is not a supported gaze AU.")
    max_angle_radians = math.radians(float(max_angle_degrees))
    if not 0.0 < max_angle_radians < math.pi * 0.5:
        raise ValueError("max_angle_degrees must be between 0 and 90.")
    radius_ratio = float(eye_radius_eye_widths)
    if not math.isfinite(radius_ratio) or radius_ratio <= 0.0:
        raise ValueError("eye_radius_eye_widths must be positive.")
    expected = semantic_eye_gaze_motion(action_unit_id, 1.0).float()
    motion = torch.as_tensor(motion, dtype=torch.float32).flatten()
    if motion.shape != (2,) or not torch.isfinite(motion).all():
        raise ValueError("Eye-gaze motion must contain two finite values.")
    projected_shift = float(torch.dot(motion, expected))
    maximum_shift = radius_ratio * math.sin(max_angle_radians)
    projected_shift = min(max(projected_shift, -maximum_shift), maximum_shift)
    angle = math.asin(projected_shift / radius_ratio)
    return min(max(angle / max_angle_radians, -1.0), 1.0)


def complete_eye_source_cache_path(source_path: Path, cache_dir: Path) -> Path:
    source_path = source_path.expanduser().resolve()
    stat = source_path.stat()
    fingerprint = "|".join(
        (
            str(COMPLETE_EYE_SOURCE_CACHE_VERSION),
            str(source_path),
            str(stat.st_size),
            str(stat.st_mtime_ns),
        )
    )
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:12]
    return cache_dir.expanduser().resolve() / "source_glb" / (
        f"{source_path.stem}_{digest}.glb"
    )


def prepare_complete_eye_source_glb(
    *,
    source_path: Path,
    cache_dir: Path,
    refresh: bool,
    mesh_object_names: Optional[Sequence[str]],
) -> Path:
    """Return the textured GLB consumed by the complete-eye worker."""

    source_path = source_path.expanduser().resolve()
    if source_path.suffix.lower() == ".glb":
        return source_path
    if source_path.suffix.lower() != ".fbx":
        raise ValueError("Complete-eye gaze supports textured GLB and FBX inputs.")
    cached = complete_eye_source_cache_path(source_path, cache_dir)
    if cached.is_file() and not refresh:
        return cached
    cached.parent.mkdir(parents=True, exist_ok=True)
    export_fbx_appearance_preserving_glb(
        source_path=source_path,
        vertices=None,
        path=cached,
        mesh_object_names=mesh_object_names,
        eye_gaze_uv_warp=None,
    )
    if not cached.is_file():
        raise RuntimeError(f"FBX conversion did not write {cached}.")
    return cached


def complete_eye_animation_command(
    *,
    source_glb: Path,
    output_path: Path,
    report_path: Path,
    action_unit_id: int,
    intensity: float,
    duration: float,
    fps: float,
    max_angle_degrees: float,
    landmarks: Optional[Path],
    mapper_path: Path,
    landmark_cache_dir: Path,
    blender: Optional[str],
    blender_threads: Optional[int],
    sclera_backing: str,
    refresh_landmarks: bool,
    debug: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "animate_glb_eye_aus.py"),
        "--input",
        str(source_glb),
        "--output",
        str(output_path),
        "--report",
        str(report_path),
        "--au",
        str(action_unit_id),
        "--intensity",
        str(intensity),
        "--duration",
        str(duration),
        "--fps",
        str(fps),
        "--max-angle",
        str(max_angle_degrees),
        "--mapper",
        str(mapper_path),
        "--cache-dir",
        str(landmark_cache_dir),
        "--sclera-backing",
        sclera_backing,
    ]
    if landmarks is not None:
        command.extend(("--landmarks", str(landmarks)))
    if blender is not None:
        command.extend(("--blender", str(blender)))
    if blender_threads is not None:
        command.extend(("--blender-threads", str(blender_threads)))
    if refresh_landmarks:
        command.append("--refresh-landmarks")
    if debug:
        command.append("--debug")
    return command


def export_complete_eye_au_animation(
    *,
    source_glb: Path,
    output_path: Path,
    action_unit_id: int,
    intensity: float,
    duration: float,
    fps: float,
    max_angle_degrees: float,
    landmarks: Optional[Path],
    mapper_path: Path,
    landmark_cache_dir: Path,
    blender: Optional[str],
    blender_threads: Optional[int],
    sclera_backing: str,
    refresh_landmarks: bool,
    debug: bool,
) -> dict[str, Any]:
    source_glb = source_glb.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if source_glb.suffix.lower() != ".glb" or not source_glb.is_file():
        raise ValueError(f"Complete-eye source is not a GLB file: {source_glb}")
    if output_path.suffix.lower() != ".glb":
        raise ValueError("Complete-eye output must be a GLB file.")
    report_path = output_path.with_suffix(".eye_animation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = complete_eye_animation_command(
        source_glb=source_glb,
        output_path=output_path,
        report_path=report_path,
        action_unit_id=action_unit_id,
        intensity=intensity,
        duration=duration,
        fps=fps,
        max_angle_degrees=max_angle_degrees,
        landmarks=landmarks,
        mapper_path=mapper_path,
        landmark_cache_dir=landmark_cache_dir,
        blender=blender,
        blender_threads=blender_threads,
        sclera_backing=sclera_backing,
        refresh_landmarks=refresh_landmarks,
        debug=debug,
    )
    process = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = process.stdout.strip()
    if output:
        print(output, flush=True)
    if process.returncode != 0:
        raise RuntimeError(
            "Complete-eye AU animation failed."
            + (f"\n{output}" if output else "")
        )
    if not output_path.is_file() or not report_path.is_file():
        raise RuntimeError(
            "Complete-eye exporter finished without writing its GLB and report."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("status") != "exported":
        raise RuntimeError(
            f"Complete-eye exporter returned an invalid report: {report}"
        )
    return report


def render_complete_eye_animation_preview(
    *,
    mesh_path: Path,
    output_path: Path,
    report: Mapping[str, Any],
    blender: Optional[str],
    blender_threads: Optional[int],
    image_size: int,
) -> None:
    """Render the final animation frame using the landmark-selected face view."""

    import map_mediapipe_landmarks as landmark_mapper

    landmarks_path = Path(str(report.get("landmarks", ""))).expanduser()
    landmark_payload: Mapping[str, Any] = {}
    if landmarks_path.is_file():
        loaded = json.loads(landmarks_path.read_text(encoding="utf-8"))
        if isinstance(loaded, Mapping):
            landmark_payload = loaded
    selected_view = str(landmark_payload.get("selected_view", "neg_z"))
    if selected_view not in complete_eye_au.GLB_LANDMARK_VIEWS:
        selected_view = "neg_z"
    try:
        roll_offset = float(landmark_payload.get("roll_offset", 0.0))
    except (TypeError, ValueError):
        roll_offset = 0.0
    frame_end = max(1, int(report.get("frame_end", 1)))
    blender_executable = complete_eye_au._resolve_blender(blender)

    with tempfile.TemporaryDirectory(prefix="demo_complete_eye_preview_") as name:
        temp_dir = Path(name)
        helper = landmark_mapper.write_blender_helper(temp_dir)
        command = [blender_executable, "--background"]
        if blender_threads is not None:
            command.extend(("--threads", str(blender_threads)))
        command.extend(
            (
                "--python",
                str(helper),
                "--",
                "--mode",
                "render",
                "--mesh",
                str(mesh_path),
                "--output-dir",
                str(temp_dir),
                "--width",
                str(image_size),
                "--height",
                str(image_size),
                "--views",
                selected_view,
                "--yaw-offsets=0",
                "--roll-offset",
                str(roll_offset),
                "--frame",
                str(frame_end),
                "--preserve-materials",
            )
        )
        process = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(
                "Complete-eye preview render failed.\n" + process.stdout.strip()
            )
        views_path = temp_dir / "views.json"
        if not views_path.is_file():
            raise RuntimeError("Complete-eye preview did not write views.json.")
        rendered = json.loads(views_path.read_text(encoding="utf-8"))
        views = rendered.get("views", []) if isinstance(rendered, Mapping) else []
        if len(views) != 1:
            raise RuntimeError(
                f"Complete-eye preview expected one rendered view; found {len(views)}."
            )
        rendered_path = Path(str(views[0]["image"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.open(rendered_path).convert("RGB").save(output_path, quality=95)


def main() -> None:
    args = parse_args()
    mesh_path = args.mesh.expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

    action_unit_id = parse_au_id(args.au_id)
    complete_eye_gaze = complete_eye_gaze_enabled(args, action_unit_id)
    validate_complete_eye_gaze_request(
        args,
        mesh_path=mesh_path,
        action_unit_id=action_unit_id,
        enabled=complete_eye_gaze,
    )
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint_config(checkpoint, args.config)
    device = resolve_device(args.device, config)

    model = load_model(checkpoint, config, device)
    raw_mesh = load_mesh_for_model(mesh_path)
    input_landmark_json = ensure_mediapipe_landmarks(
        mesh_path=mesh_path,
        cache_dir=args.landmark_cache_dir.expanduser(),
        mapper_path=args.mediapipe_mapper.expanduser(),
        blender=args.blender,
        force=args.refresh_landmarks,
        require_surface_anchors=complete_eye_gaze,
        blender_threads=args.eye_gaze_blender_threads,
    )
    reference_mesh_path = metahuman_sample_mesh(args.metahuman_sample_dir.expanduser())
    reference_landmark_json = ensure_mediapipe_landmarks(
        mesh_path=reference_mesh_path,
        cache_dir=args.landmark_cache_dir.expanduser(),
        mapper_path=args.mediapipe_mapper.expanduser(),
        blender=args.blender,
        force=args.refresh_landmarks,
        prefer_sidecar=True,
    )
    input_landmark_payload = load_landmark_payload(input_landmark_json)
    input_landmarks = landmark_mapping_from_payload(input_landmark_payload)
    original_mesh = preprocess_mesh_for_model(
        raw_mesh,
        config,
        input_suffix=mesh_path.suffix.lower(),
        input_convention=args.input_convention,
        input_landmarks=input_landmarks,
        reference_mesh=load_reference_mesh_for_alignment(reference_mesh_path),
        reference_landmarks=load_landmark_mapping(reference_landmark_json),
        min_alignment_landmarks=args.min_alignment_landmarks,
    )
    if args.no_weld_input:
        model_mesh = original_mesh
        original_to_welded = torch.arange(
            original_mesh["vertices"].shape[0],
            dtype=torch.long,
        )
    else:
        weld_tolerance, snap_landmark_ids = inference_weld_settings(
            config,
            tolerance_override=args.weld_tolerance,
        )
        model_mesh, original_to_welded = weld_mesh_for_inference(
            original_mesh,
            tolerance=weld_tolerance,
            snap_mediapipe_ids=snap_landmark_ids,
        )

    welded_deformed_vertices, welded_predicted_delta = run_model(
        model=model,
        mesh=model_mesh,
        action_unit_id=action_unit_id,
        config=config,
        device=device,
    )
    model_delta_mean_norm = float(
        welded_predicted_delta.norm(dim=-1).mean().item()
    )
    eye_gaze_source_motion = None
    eye_gaze_intensity = None
    if complete_eye_gaze:
        if args.eye_gaze_intensity is not None:
            eye_gaze_intensity = normalized_eye_gaze_intensity(
                args.eye_gaze_intensity
            )
        else:
            if args.eye_gaze_uv_strength is None:
                eye_gaze_source_motion = predicted_eye_gaze_screen_motion(
                    mesh=model_mesh,
                    deformed_vertices=welded_deformed_vertices,
                    action_unit_id=action_unit_id,
                    config=config,
                )
            else:
                eye_gaze_source_motion = semantic_eye_gaze_motion(
                    action_unit_id,
                    args.eye_gaze_uv_strength,
                )
            eye_gaze_source_motion = clamp_eye_gaze_motion(
                eye_gaze_source_motion,
                maximum_norm=args.eye_gaze_uv_max_motion,
            )
            eye_gaze_intensity = eye_gaze_motion_to_rotation_intensity(
                action_unit_id=action_unit_id,
                motion=eye_gaze_source_motion,
                max_angle_degrees=args.eye_gaze_max_angle,
            )
        # Eye AUs are now performed by a rigid complete-ball animation after
        # the textured source has been exported. Do not also deform the same
        # region with the checkpoint or the two movements would be applied
        # twice.
        welded_predicted_delta = torch.zeros_like(welded_predicted_delta)

    predicted_delta = map_welded_values_to_original(
        welded_predicted_delta,
        original_to_welded,
    )
    dental_postprocess_report = None
    is_ict_face = any(
        "ictfacemodel" in str(name).lower()
        for name in raw_mesh.get("source_mesh_object_names", ())
    )
    remove_ict_oral_geometry = (
        action_unit_id == JAW_OPEN_AU_ID
        and is_ict_face
        and not args.keep_ict_oral_geometry
    )
    if remove_ict_oral_geometry:
        if (
            args.no_preserve_appearance
            or mesh_path.suffix.lower() != ".fbx"
            or args.output_format != "glb"
        ):
            raise ValueError(
                "ICT oral-geometry removal requires appearance-preserving "
                "FBX-to-GLB export."
            )
        dental_postprocess_report = {
            "applied": True,
            "action_unit_id": action_unit_id,
            "method": "remove_ict_oral_material_faces",
            "reason": "oral_geometry_removed",
            "material_patterns": list(ICT_ORAL_MATERIAL_PATTERNS),
        }
    elif (
        not args.no_rigid_jaw_teeth
        and action_unit_id == JAW_OPEN_AU_ID
        and is_ict_face
    ):
        predicted_delta, dental_postprocess_report = (
            apply_rigid_jaw_teeth_postprocess(
                neutral_vertices=original_mesh["vertices"],
                predicted_delta=predicted_delta,
                faces=original_mesh["faces"],
                vertex_groups=raw_mesh.get("source_vertex_groups"),
                landmarks_3d=original_mesh.get("landmarks_3d"),
                action_unit_id=action_unit_id,
            )
        )
    deformed_vertices = original_mesh["vertices"] + predicted_delta
    neutral_normals = recompute_vertex_normals(
        original_mesh["vertices"],
        original_mesh["faces"],
    )

    image_path, output_mesh_path = default_output_paths(
        mesh_path=mesh_path,
        output_dir=args.output_dir,
        action_unit_id=action_unit_id,
        output_format=args.output_format,
    )
    dental_report_path = output_mesh_path.with_suffix(".dental_postprocess.json")
    image = render_output_image(
        neutral_vertices=original_mesh["vertices"],
        deformed_vertices=deformed_vertices,
        normals=neutral_normals,
        faces=original_mesh["faces"],
        action_unit_id=action_unit_id,
        config=config,
        device=device,
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path, quality=95)

    export_vertices = output_vertices_for_convention(
        deformed_vertices,
        args.output_convention,
    )
    export_normals = output_vertices_for_convention(
        recompute_vertex_normals(
            deformed_vertices,
            original_mesh["faces"],
        ),
        args.output_convention,
    )
    appearance_preserved = False
    eye_animation_report = None
    if complete_eye_gaze:
        assert eye_gaze_intensity is not None
        source_glb = prepare_complete_eye_source_glb(
            source_path=mesh_path,
            cache_dir=args.eye_gaze_landmark_cache_dir.expanduser(),
            refresh=args.refresh_landmarks,
            mesh_object_names=raw_mesh.get("source_mesh_object_names"),
        )
        landmarks = (
            input_landmark_json
            if source_glb.resolve() == mesh_path.resolve()
            else None
        )
        eye_animation_report = export_complete_eye_au_animation(
            source_glb=source_glb,
            output_path=output_mesh_path,
            action_unit_id=action_unit_id,
            intensity=eye_gaze_intensity,
            duration=args.eye_gaze_duration,
            fps=args.eye_gaze_fps,
            max_angle_degrees=args.eye_gaze_max_angle,
            landmarks=landmarks,
            mapper_path=args.mediapipe_mapper.expanduser(),
            landmark_cache_dir=args.eye_gaze_landmark_cache_dir.expanduser(),
            blender=args.blender,
            blender_threads=args.eye_gaze_blender_threads,
            sclera_backing=args.eye_gaze_sclera_backing,
            refresh_landmarks=args.refresh_landmarks,
            debug=args.eye_gaze_debug,
        )
        appearance_preserved = True
    elif (
        not args.no_preserve_appearance
        and mesh_path.suffix.lower() == ".fbx"
        and output_mesh_path.suffix.lower() == ".glb"
    ):
        try:
            export_report = export_fbx_appearance_preserving_glb(
                source_path=mesh_path,
                vertices=export_vertices,
                path=output_mesh_path,
                mesh_object_names=raw_mesh.get("source_mesh_object_names"),
                eye_gaze_uv_warp=None,
                exclude_material_patterns=(
                    ICT_ORAL_MATERIAL_PATTERNS
                    if remove_ict_oral_geometry
                    else None
                ),
            )
            if (
                remove_ict_oral_geometry
                and dental_postprocess_report is not None
                and export_report is not None
            ):
                dental_postprocess_report.update(
                    export_report.get("material_face_removal", {})
                )
            appearance_preserved = True
        except Exception as exc:
            if remove_ict_oral_geometry:
                raise
            print(
                "[WARN] Could not preserve source FBX appearance; falling back "
                f"to geometry-only export: {exc}"
            )
    if not appearance_preserved:
        export_mesh(
            vertices=export_vertices,
            faces=original_mesh["faces"],
            normals=F.normalize(export_normals, dim=-1, eps=1.0e-6),
            path=output_mesh_path,
        )
    if dental_postprocess_report is not None:
        dental_report_path.parent.mkdir(parents=True, exist_ok=True)
        dental_report_path.write_text(
            json.dumps(dental_postprocess_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if eye_animation_report is not None:
        preview_dimensions = train.normalize_image_size(
            config.get("rendering", {}).get("validation_image_size", (512, 512))
        )
        try:
            render_complete_eye_animation_preview(
                mesh_path=output_mesh_path,
                output_path=image_path,
                report=eye_animation_report,
                blender=args.blender,
                blender_threads=args.eye_gaze_blender_threads,
                image_size=max(int(value) for value in preview_dimensions),
            )
        except Exception as exc:
            print(
                "[WARN] Could not render the final complete-eye frame; retained "
                f"the TopoRig diagnostic image instead: {exc}"
            )

    print(f"Loaded checkpoint: {checkpoint_path}")
    if args.no_weld_input:
        print("Model input welding: disabled")
    else:
        print(
            "Model input welding: "
            f"{original_mesh['vertices'].shape[0]} -> "
            f"{model_mesh['vertices'].shape[0]} vertices; output retained "
            f"{original_mesh['vertices'].shape[0]} original vertices"
        )
    print(f"Model delta mean norm: {model_delta_mean_norm:.6f}")
    print(f"Exported delta mean norm: {predicted_delta.norm(dim=-1).mean().item():.6f}")
    if eye_gaze_source_motion is not None:
        print(
            "Eye-gaze source motion (eye widths): "
            f"dx={float(eye_gaze_source_motion[0]):.6f} "
            f"dy={float(eye_gaze_source_motion[1]):.6f}"
        )
    if eye_animation_report is not None:
        print(
            "Complete-eye gaze animation: "
            f"AU{action_unit_id}, intensity={eye_gaze_intensity:.6f}, "
            f"frames={eye_animation_report.get('frame_start')}-"
            f"{eye_animation_report.get('frame_end')}"
        )
        print(
            "Complete-eye report: "
            f"{output_mesh_path.with_suffix('.eye_animation.json')}"
        )
        print(
            "Suppressed checkpoint geometry for this gaze AU; exported rigid "
            "full-ball rotation with transferred source-eye texture"
        )
    elif appearance_preserved:
        print("Preserved source FBX UVs, materials, and packed textures")
    if dental_postprocess_report is not None:
        if remove_ict_oral_geometry:
            print(
                "Removed ICT oral material faces: "
                + ", ".join(
                    dental_postprocess_report.get("material_names", ())
                )
                + f" ({dental_postprocess_report.get('removed_face_count', 0)} faces)"
            )
        else:
            print(
                "Rigid jaw-teeth postprocess: "
                f"{dental_postprocess_report['reason']} "
                f"(applied={dental_postprocess_report['applied']})"
            )
        print(f"Dental postprocess report: {dental_report_path}")
    print(f"Wrote image: {image_path}")
    print(f"Wrote mesh:  {output_mesh_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained TopoRig checkpoint.")
    parser.add_argument(
        "--mesh",
        type=Path,
        required=True,
        help="Input neutral mesh path. Supported formats: OBJ, GLB, FBX.",
    )
    parser.add_argument(
        "--au-id",
        required=True,
        help="Action-unit id, e.g. 26. AU names such as jawOpen are also accepted.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Path to a TopoRig checkpoint. Defaults to checkpoints/stage2.pt "
            "(or TOPORIG_CHECKPOINT_ROOT/stage2.pt), then the newest training checkpoint."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Fallback config for checkpoints that do not contain a saved config. "
            "train.py checkpoints already include this."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for outputs. Defaults to the input mesh directory.",
    )
    parser.add_argument(
        "--output-format",
        choices=("glb", "obj"),
        default="glb",
        help="Mesh output format. Defaults to GLB to match the demo example.",
    )
    parser.add_argument(
        "--output-convention",
        choices=("metahuman", "model"),
        default="metahuman",
        help=(
            "Coordinate convention for the exported mesh. metahuman writes the "
            "MetaHuman/FBX frame; model writes internal model-frame coordinates."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device. Defaults to training.device from the checkpoint config.",
    )
    parser.add_argument(
        "--input-convention",
        choices=("auto", "fbx", "custom-glb"),
        default="auto",
        help=(
            "Input mesh coordinate convention. auto uses the FBX dataset convention "
            "for OBJ/FBX and z-down,+Y-front for GLB."
        ),
    )
    parser.add_argument(
        "--metahuman-sample-dir",
        type=Path,
        default=DEFAULT_METAHUMAN_SAMPLE_DIR,
        help="Reference MetaHuman sample directory used for landmark alignment.",
    )
    parser.add_argument(
        "--landmark-cache-dir",
        type=Path,
        default=DEFAULT_LANDMARK_CACHE_DIR,
        help="Cache directory for generated MediaPipe mesh landmark JSON files.",
    )
    parser.add_argument(
        "--mediapipe-mapper",
        type=Path,
        default=DEFAULT_MEDIAPIPE_MAPPER,
        help="Path to map_mediapipe_landmarks.py.",
    )
    parser.add_argument(
        "--blender",
        default=None,
        help="Optional Blender executable passed to the MediaPipe landmark mapper.",
    )
    parser.add_argument(
        "--refresh-landmarks",
        action="store_true",
        help="Regenerate cached MediaPipe landmark mappings before alignment.",
    )
    parser.add_argument(
        "--min-alignment-landmarks",
        type=int,
        default=32,
        help="Minimum shared MediaPipe landmark vertices required for Procrustes alignment.",
    )
    parser.add_argument(
        "--weld-tolerance",
        type=float,
        default=None,
        help=(
            "Coincident-vertex tolerance for the temporary model-input mesh. "
            "Defaults to the checkpoint image-data setting or 1e-6."
        ),
    )
    parser.add_argument(
        "--no-weld-input",
        action="store_true",
        help="Run TopoRig directly on the original topology without temporary welding.",
    )
    parser.add_argument(
        "--no-preserve-appearance",
        action="store_true",
        help=(
            "Use the geometry-only GLB exporter instead of retaining source FBX "
            "UVs, materials, and textures."
        ),
    )
    parser.add_argument(
        "--no-rigid-jaw-teeth",
        action="store_true",
        help=(
            "Disable the ICT jawOpen postprocess that restores upper teeth and "
            "moves lower teeth as one rigid lower-jaw region."
        ),
    )
    parser.add_argument(
        "--keep-ict-oral-geometry",
        action="store_true",
        help=(
            "Keep ICT teeth and gums/tongue material faces in jawOpen output. "
            "By default those unstable internal-mouth faces are omitted."
        ),
    )
    parser.add_argument(
        "--eye-gaze",
        action="store_true",
        help=(
            "Enable complete-ball eye replacement and rigid AU rotation. This "
            "is already the default for AU12-AU19."
        ),
    )
    parser.add_argument(
        "--eye-gaze-uv-warp",
        action="store_true",
        help=(
            "Deprecated compatibility alias for --eye-gaze. The previous UV "
            "warp is no longer used by the demo."
        ),
    )
    parser.add_argument(
        "--eye-gaze-mode",
        choices=("auto", "mesh", "texture"),
        default=None,
        help=(
            "Deprecated compatibility option. All three legacy values now use "
            "the complete-ball replacement algorithm."
        ),
    )
    parser.add_argument(
        "--eye-gaze-intensity",
        type=float,
        default=None,
        help=(
            "Rigid eye-rotation intensity from -1 to 1. When omitted, the demo "
            "converts the model prediction (or legacy UV strength) into an angle."
        ),
    )
    parser.add_argument(
        "--eye-gaze-uv-strength",
        type=float,
        default=None,
        help=(
            "Legacy gaze magnitude in eye-width units. It is converted into a "
            "rigid sphere-rotation intensity; no UVs or pixels are shifted."
        ),
    )
    parser.add_argument(
        "--eye-gaze-uv-max-motion",
        type=float,
        default=0.25,
        help=(
            "Maximum legacy/model source-motion norm in eye-width units before "
            "conversion to rigid rotation."
        ),
    )
    parser.add_argument(
        "--eye-gaze-duration",
        type=float,
        default=1.0,
        help="Seconds from neutral to the requested eye AU (default: 1).",
    )
    parser.add_argument(
        "--eye-gaze-fps",
        type=float,
        default=30.0,
        help="Eye animation frame rate (default: 30).",
    )
    parser.add_argument(
        "--eye-gaze-max-angle",
        type=float,
        default=25.0,
        help="Rotation in degrees produced by intensity 1 (default: 25).",
    )
    parser.add_argument(
        "--eye-gaze-sclera-backing",
        choices=("always", "auto", "off"),
        default="always",
        help=(
            "Complete-ball replacement policy. 'always' is the validated demo "
            "default and supports partial or fused source eyes."
        ),
    )
    parser.add_argument(
        "--eye-gaze-landmark-cache-dir",
        type=Path,
        default=DEFAULT_COMPLETE_EYE_CACHE_DIR,
        help="Cache for robust multi-view eye landmarks and converted FBX sources.",
    )
    parser.add_argument(
        "--eye-gaze-blender-threads",
        type=int,
        default=None,
        help="Optional Blender thread limit for landmark and eye export workers.",
    )
    parser.add_argument(
        "--eye-gaze-debug",
        action="store_true",
        help="Retain eye-landmark debug outputs and detailed candidate metrics.",
    )
    parser.add_argument(
        "--no-complete-eye-gaze",
        action="store_true",
        help=(
            "Disable automatic complete-ball processing for AU12-AU19 and use "
            "the checkpoint's ordinary geometry export."
        ),
    )
    parser.add_argument(
        "--preserve-eye-gaze-geometry",
        action="store_true",
        help=(
            "Deprecated legacy option. It is incompatible with complete-ball "
            "eye replacement."
        ),
    )
    return parser.parse_args()


def parse_au_id(value: str) -> int:
    value = str(value).strip()
    if value.isdigit():
        return int(value)

    name_to_id = {name: action_unit for action_unit, name in AU_NAME.items()}
    if value not in name_to_id:
        raise ValueError(
            f"Unknown AU {value!r}. Use an integer id or one of: "
            f"{', '.join(sorted(name_to_id))}."
        )
    return name_to_id[value]


def resolve_checkpoint_path(path: Optional[Path]) -> Path:
    if path is not None:
        candidate = path.expanduser().resolve()
        if candidate.is_dir():
            return newest_checkpoint(candidate)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate
    downloaded = runtime_path("checkpoint", "stage2.pt")
    if downloaded.is_file():
        return downloaded.resolve()
    return newest_checkpoint(DEFAULT_RUNS_DIR)


def newest_checkpoint(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(
            f"No checkpoint was provided and {root} does not exist. "
            "Pass --checkpoint /path/to/checkpoints/best.pt."
        )

    candidates = sorted(root.rglob("*.pt"))
    candidates = [path for path in candidates if "checkpoints" in path.parts]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints were found under {root}. "
            "Pass --checkpoint /path/to/checkpoints/best.pt."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Expected {path} to contain a checkpoint mapping.")
    return checkpoint


def checkpoint_config(
    checkpoint: Mapping[str, Any],
    fallback_config_path: Optional[Path],
) -> Mapping[str, Any]:
    config = checkpoint.get("config")
    if isinstance(config, Mapping):
        return config
    if fallback_config_path is None:
        raise ValueError(
            "Checkpoint does not contain a config. Pass --config to build the model."
        )
    return train.load_config(fallback_config_path)


def resolve_device(cli_device: Optional[str], config: Mapping[str, Any]) -> torch.device:
    device_name = cli_device
    if device_name is None:
        device_name = str(config.get("training", {}).get("device", "auto"))
    return train.resolve_device(device_name, train.DistributedContext())


def load_model(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> TopoRig:
    if "model" not in config or not isinstance(config["model"], Mapping):
        raise ValueError("Checkpoint config is missing the model section.")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint is missing model_state_dict.")

    state_dict = strip_module_prefix(state_dict)
    model_config = checkpoint_model_config(config, state_dict)
    model = TopoRig(**model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def checkpoint_model_config(
    config: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    model_config = train.toporig_model_config(config["model"])
    if "use_conditioned_output_head" in model_config:
        return model_config

    first_output_weight = state_dict.get("output_mlp.net.0.weight")
    if not isinstance(first_output_weight, torch.Tensor):
        return model_config

    checkpoint_input_dim = int(first_output_weight.shape[1])
    hidden_dim = int(model_config.get("hidden_dim", 256))
    condition_dim = int(model_config.get("condition_dim", 128))
    if checkpoint_input_dim == hidden_dim:
        model_config["use_conditioned_output_head"] = False
    elif checkpoint_input_dim == hidden_dim + condition_dim:
        model_config["use_conditioned_output_head"] = True
    return model_config


def strip_module_prefix(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not state_dict:
        return dict(state_dict)
    if not all(key.startswith("module.") for key in state_dict):
        return dict(state_dict)
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_mesh_for_model(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        raise ValueError(
            f"Unsupported mesh format {path.suffix!r}. "
            "Supported formats are OBJ, GLB, and FBX."
        )

    if suffix == ".fbx":
        faces, vertices, _blendshapes, metadata = fbx_to_tensor(
            path,
            include_metadata=True,
        )
        normals = recompute_vertex_normals(vertices, faces)
        return {
            "vertices": vertices.float(),
            "faces": faces.long(),
            "normals": normals,
            "source_mesh_object_names": tuple(
                str(name) for name in metadata.get("mesh_object_names", ())
            ),
            "source_vertex_groups": metadata.get("vertex_groups", {}),
        }
    if suffix == ".glb":
        return {
            name: tensor.float() if name != "faces" else tensor.long()
            for name, tensor in _load_glb_for_model(path).items()
        }
    return load_obj_for_model(path)


def preprocess_mesh_for_model(
    mesh: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    input_suffix: Optional[str] = None,
    input_convention: str = "auto",
    input_landmarks: Optional[Mapping[int, int]] = None,
    reference_mesh: Optional[Mapping[str, torch.Tensor]] = None,
    reference_landmarks: Optional[Mapping[int, int]] = None,
    min_alignment_landmarks: int = 32,
) -> dict[str, torch.Tensor]:
    mesh = {
        "vertices": mesh["vertices"].detach().cpu().float().contiguous(),
        "faces": mesh["faces"].detach().cpu().long().contiguous(),
        "normals": mesh["normals"].detach().cpu().float().contiguous(),
    }
    mesh = orient_input_mesh_to_dataset_convention(
        mesh,
        input_suffix=input_suffix,
        input_convention=input_convention,
    )
    landmarks_3d = (
        landmark_payload_from_mapping(input_landmarks, mesh["vertices"])
        if input_landmarks is not None
        else _empty_landmark_payload(mesh["vertices"])
    )
    settings = mesh_preprocess_settings(config)
    processed_mesh, processed_landmarks = transform_mesh_with_dataset_settings(
        mesh,
        settings,
        landmarks_3d=landmarks_3d,
    )
    processed_mesh["landmarks_3d"] = processed_landmarks

    if (
        input_landmarks is not None
        and reference_mesh is not None
        and reference_landmarks is not None
    ):
        reference_processed_mesh, _reference_landmarks = transform_mesh_with_dataset_settings(
            reference_mesh,
            settings,
        )
        processed_mesh = align_mesh_to_reference_landmarks(
            mesh=processed_mesh,
            input_landmarks=input_landmarks,
            reference_mesh=reference_processed_mesh,
            reference_landmarks=reference_landmarks,
            min_landmarks=min_alignment_landmarks,
        )

    return processed_mesh


def weld_mesh_for_inference(
    mesh: Mapping[str, Any],
    *,
    tolerance: float,
    snap_mediapipe_ids: Optional[set[int]] = None,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Build a temporary welded model mesh and retain its source-vertex map."""

    welded_payload = _weld_neutral_mesh_payload(
        {
            "vertices": mesh["vertices"],
            "faces": mesh["faces"],
            "normals": mesh.get("normals"),
        },
        tolerance=tolerance,
    )
    original_to_welded = welded_payload.pop("source_vertex_to_welded")
    component_ids = welded_payload.pop("component_ids")
    largest_component_id = int(welded_payload.pop("largest_component_id").item())

    landmarks = mesh.get("landmarks_3d")
    if isinstance(landmarks, Mapping):
        welded_landmarks = _remap_landmarks_to_welded_mesh(
            landmarks=dict(landmarks),
            welded_vertices=welded_payload["vertices"],
            source_to_welded=original_to_welded,
            component_ids=component_ids,
            largest_component_id=largest_component_id,
            snap_mediapipe_ids=set(snap_mediapipe_ids or ()),
        )
    else:
        welded_landmarks = _empty_landmark_payload(welded_payload["vertices"])

    welded_mesh = dict(welded_payload)
    welded_mesh["landmarks_3d"] = welded_landmarks
    return welded_mesh, original_to_welded.contiguous()


def map_welded_values_to_original(
    welded_values: torch.Tensor,
    original_to_welded: torch.Tensor,
) -> torch.Tensor:
    if welded_values.ndim != 2:
        raise ValueError("welded_values must have shape [V, C].")
    original_to_welded = original_to_welded.detach().cpu().long().flatten()
    if original_to_welded.numel() == 0:
        return welded_values.new_empty((0, welded_values.shape[1]))
    if int(original_to_welded.min().item()) < 0:
        raise ValueError("original_to_welded contains a negative vertex ID.")
    if int(original_to_welded.max().item()) >= welded_values.shape[0]:
        raise ValueError("original_to_welded contains an out-of-range vertex ID.")
    return welded_values.index_select(
        0,
        original_to_welded.to(device=welded_values.device),
    ).contiguous()


def apply_rigid_jaw_teeth_postprocess(
    *,
    neutral_vertices: torch.Tensor,
    predicted_delta: torch.Tensor,
    faces: torch.Tensor,
    vertex_groups: object,
    landmarks_3d: object,
    action_unit_id: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Replace per-vertex jawOpen tooth motion with one rigid lower-jaw motion."""

    report: dict[str, Any] = {
        "applied": False,
        "action_unit_id": int(action_unit_id),
        "method": "rigid_lower_teeth_from_mediapipe_jaw",
        "anchor_mediapipe_ids": list(RIGID_LOWER_TEETH_MEDIAPIPE_IDS),
    }
    if int(action_unit_id) != JAW_OPEN_AU_ID:
        report["reason"] = "not_jaw_open"
        return predicted_delta, report
    if neutral_vertices.shape != predicted_delta.shape:
        raise ValueError("neutral_vertices and predicted_delta must have equal shape.")
    if neutral_vertices.ndim != 2 or neutral_vertices.shape[-1] != 3:
        raise ValueError("neutral_vertices and predicted_delta must have shape [V, 3].")

    tooth_ids, material_names = _tooth_material_vertex_ids(
        vertex_groups,
        vertex_count=neutral_vertices.shape[0],
    )
    report["tooth_materials"] = material_names
    report["tooth_vertex_count"] = int(tooth_ids.numel())
    if tooth_ids.numel() == 0:
        report["reason"] = "no_tooth_material_vertices"
        return predicted_delta, report

    adjusted_delta = predicted_delta.clone()
    device_tooth_ids = tooth_ids.to(device=adjusted_delta.device)
    adjusted_delta.index_fill_(0, device_tooth_ids, 0.0)

    components = _vertex_components_within_selection(
        faces=faces,
        selected_vertex_ids=tooth_ids,
        vertex_count=neutral_vertices.shape[0],
    )
    report["tooth_component_count"] = len(components)
    split = _split_tooth_components_into_rows(neutral_vertices, components)
    if split is None:
        report["reason"] = "could_not_separate_upper_and_lower_teeth"
        return adjusted_delta, report
    lower_ids, upper_ids, row_gap = split
    report.update(
        {
            "lower_tooth_vertex_count": int(lower_ids.numel()),
            "upper_tooth_vertex_count": int(upper_ids.numel()),
            "row_centroid_gap": float(row_gap),
        }
    )

    anchor_ids, used_mediapipe_ids = _jaw_anchor_vertex_ids(
        landmarks_3d,
        vertex_count=neutral_vertices.shape[0],
    )
    report["used_anchor_mediapipe_ids"] = used_mediapipe_ids
    if anchor_ids.numel() < 4:
        report["reason"] = "not_enough_valid_jaw_landmarks"
        return adjusted_delta, report

    source = neutral_vertices.index_select(0, anchor_ids.to(neutral_vertices.device))
    source_delta = predicted_delta.index_select(
        0,
        anchor_ids.to(predicted_delta.device),
    ).to(device=source.device, dtype=source.dtype)
    target = source + source_delta
    transform = _validated_rigid_jaw_transform(source, target, neutral_vertices)
    if transform is None:
        report["reason"] = "unsafe_jaw_transform"
        return adjusted_delta, report
    rotation, translation, transform_report = transform

    lower_device_ids = lower_ids.to(device=neutral_vertices.device)
    lower_source = neutral_vertices.index_select(0, lower_device_ids)
    lower_target = lower_source.matmul(rotation) + translation
    lower_delta = lower_target - lower_source
    adjusted_delta.index_copy_(
        0,
        lower_ids.to(device=adjusted_delta.device),
        lower_delta.to(device=adjusted_delta.device, dtype=adjusted_delta.dtype),
    )
    report.update(transform_report)
    report["applied"] = True
    report["reason"] = "ok"
    return adjusted_delta, report


def _tooth_material_vertex_ids(
    vertex_groups: object,
    *,
    vertex_count: int,
) -> tuple[torch.Tensor, list[str]]:
    if not isinstance(vertex_groups, Mapping):
        return torch.empty(0, dtype=torch.long), []
    materials = vertex_groups.get("materials")
    if not isinstance(materials, Mapping):
        return torch.empty(0, dtype=torch.long), []

    selected: list[torch.Tensor] = []
    material_names: list[str] = []
    for raw_name, raw_ids in materials.items():
        name = str(raw_name)
        normalized = name.lower()
        if "teeth" not in normalized and "tooth" not in normalized:
            continue
        ids = torch.as_tensor(raw_ids, dtype=torch.long).detach().cpu().flatten()
        ids = ids[(ids >= 0) & (ids < int(vertex_count))]
        if ids.numel() == 0:
            continue
        selected.append(ids)
        material_names.append(name)
    if not selected:
        return torch.empty(0, dtype=torch.long), []
    return torch.cat(selected).unique(sorted=True), sorted(material_names)


def _vertex_components_within_selection(
    *,
    faces: torch.Tensor,
    selected_vertex_ids: torch.Tensor,
    vertex_count: int,
) -> list[torch.Tensor]:
    selected_vertex_ids = selected_vertex_ids.detach().cpu().long().unique(sorted=True)
    local_ids = torch.full((int(vertex_count),), -1, dtype=torch.long)
    local_ids[selected_vertex_ids] = torch.arange(selected_vertex_ids.numel())
    local_faces = local_ids.index_select(0, faces.detach().cpu().long().flatten()).view(-1, 3)
    local_faces = local_faces[(local_faces >= 0).all(dim=1)]

    parents = list(range(selected_vertex_ids.numel()))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first, second, third in local_faces.tolist():
        union(first, second)
        union(second, third)

    grouped: dict[int, list[int]] = {}
    for local_id, global_id in enumerate(selected_vertex_ids.tolist()):
        grouped.setdefault(find(local_id), []).append(global_id)
    return [
        torch.tensor(vertex_ids, dtype=torch.long)
        for vertex_ids in grouped.values()
    ]


def _split_tooth_components_into_rows(
    neutral_vertices: torch.Tensor,
    components: Sequence[torch.Tensor],
) -> Optional[tuple[torch.Tensor, torch.Tensor, float]]:
    if len(components) < 2:
        return None
    vertices = neutral_vertices.detach().cpu()
    ordered = sorted(
        [
            (
                float(vertices.index_select(0, component)[:, 2].mean().item()),
                component,
            )
            for component in components
        ],
        key=lambda item: item[0],
    )
    gaps = [ordered[index + 1][0] - ordered[index][0] for index in range(len(ordered) - 1)]
    split_index = max(range(len(gaps)), key=gaps.__getitem__)
    row_gap = gaps[split_index]
    tooth_span = float(
        vertices.index_select(0, torch.cat(components))[:, 2].max().item()
        - vertices.index_select(0, torch.cat(components))[:, 2].min().item()
    )
    if row_gap <= max(1.0e-6, tooth_span * 0.05):
        return None
    lower_ids = torch.cat([component for _height, component in ordered[: split_index + 1]])
    upper_ids = torch.cat([component for _height, component in ordered[split_index + 1 :]])
    if lower_ids.numel() == 0 or upper_ids.numel() == 0:
        return None
    return lower_ids.unique(sorted=True), upper_ids.unique(sorted=True), row_gap


def _jaw_anchor_vertex_ids(
    landmarks_3d: object,
    *,
    vertex_count: int,
) -> tuple[torch.Tensor, list[int]]:
    if not isinstance(landmarks_3d, Mapping):
        return torch.empty(0, dtype=torch.long), []
    mediapipe_ids = torch.as_tensor(
        landmarks_3d.get("mediapipe_ids", []),
        dtype=torch.long,
    ).detach().cpu().flatten()
    vertex_ids = torch.as_tensor(
        landmarks_3d.get("vertex_ids", []),
        dtype=torch.long,
    ).detach().cpu().flatten()
    if mediapipe_ids.numel() != vertex_ids.numel():
        return torch.empty(0, dtype=torch.long), []

    by_mediapipe_id = {
        int(mediapipe_id): int(vertex_id)
        for mediapipe_id, vertex_id in zip(
            mediapipe_ids.tolist(),
            vertex_ids.tolist(),
        )
        if 0 <= int(vertex_id) < int(vertex_count)
    }
    used_mediapipe_ids = [
        mediapipe_id
        for mediapipe_id in RIGID_LOWER_TEETH_MEDIAPIPE_IDS
        if mediapipe_id in by_mediapipe_id
    ]
    return (
        torch.tensor(
            [by_mediapipe_id[mediapipe_id] for mediapipe_id in used_mediapipe_ids],
            dtype=torch.long,
        ),
        used_mediapipe_ids,
    )


def _validated_rigid_jaw_transform(
    source: torch.Tensor,
    target: torch.Tensor,
    neutral_vertices: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor, dict[str, float]]]:
    source = source.detach().cpu().double()
    target = target.detach().cpu().double()
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        return None

    extent = float(
        (neutral_vertices.detach().cpu().double().amax(dim=0)
        - neutral_vertices.detach().cpu().double().amin(dim=0)).amax().item()
    )
    if not math.isfinite(extent) or extent <= 1.0e-8:
        return None
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    covariance = source_centered.transpose(0, 1).matmul(target_centered)
    try:
        u, singular_values, vh = torch.linalg.svd(covariance, full_matrices=False)
    except RuntimeError:
        return None
    if singular_values.numel() < 2 or float(singular_values[1]) <= extent * extent * 1.0e-8:
        return None
    rotation = u.matmul(vh)
    if torch.linalg.det(rotation) < 0:
        u = u.clone()
        u[:, -1] *= -1.0
        rotation = u.matmul(vh)
    translation = target.mean(dim=0) - source.mean(dim=0).matmul(rotation)
    fitted = source.matmul(rotation) + translation
    residuals = (fitted - target).norm(dim=-1)
    cosine = ((torch.trace(rotation) - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle_degrees = float(torch.rad2deg(torch.acos(cosine)).item())
    translation_norm = float(translation.norm().item())
    residual_q95 = float(torch.quantile(residuals, 0.95).item())
    downward_motion = float((target[:, 2] - source[:, 2]).mean().item())
    maximum_anchor_motion = float((target - source).norm(dim=-1).max().item())
    if (
        not math.isfinite(angle_degrees)
        or angle_degrees > 50.0
        or translation_norm > extent * 0.25
        or residual_q95 > extent * 0.05
        or maximum_anchor_motion > extent * 0.25
        or downward_motion >= 0.0
    ):
        return None
    return (
        rotation.to(dtype=neutral_vertices.dtype, device=neutral_vertices.device),
        translation.to(dtype=neutral_vertices.dtype, device=neutral_vertices.device),
        {
            "rotation_degrees": angle_degrees,
            "translation_norm": translation_norm,
            "anchor_residual_q95": residual_q95,
            "mean_vertical_anchor_motion": downward_motion,
        },
    )


def transform_mesh_with_dataset_settings(
    mesh: Mapping[str, torch.Tensor],
    settings: Mapping[str, Any],
    landmarks_3d: Optional[Mapping[str, torch.Tensor]] = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    zero_delta = torch.zeros_like(mesh["vertices"])
    processed_mesh, _zero_delta, processed_landmarks = _transform_model_sample(
        mesh=dict(mesh),
        delta_vertices=zero_delta,
        landmarks_3d=(
            dict(landmarks_3d)
            if landmarks_3d is not None
            else _empty_landmark_payload(mesh["vertices"])
        ),
        mesh_up_axis=settings["mesh_up_axis"],
        mesh_front_axis=settings["mesh_front_axis"],
        normalize_on_get=settings["normalize_on_get"],
        normalized_extent=settings["normalized_extent"],
    )
    return processed_mesh, processed_landmarks


def landmark_payload_from_mapping(
    mapping: Mapping[int, int],
    vertices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    landmark_pairs = []
    for raw_mediapipe_id, raw_vertex_id in mapping.items():
        vertex_id = int(raw_vertex_id)
        if 0 <= vertex_id < vertices.shape[0]:
            landmark_pairs.append((int(raw_mediapipe_id), vertex_id))
    landmark_pairs.sort(key=lambda item: item[0])
    if not landmark_pairs:
        return _empty_landmark_payload(vertices)

    mediapipe_ids = torch.tensor(
        [mediapipe_id for mediapipe_id, _vertex_id in landmark_pairs],
        dtype=torch.long,
    )
    vertex_ids = torch.tensor(
        [vertex_id for _mediapipe_id, vertex_id in landmark_pairs],
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
        "target_delta": torch.empty(0, 3, dtype=torch.float32),
        "target_positions": torch.empty(0, 3, dtype=torch.float32),
    }


def orient_input_mesh_to_dataset_convention(
    mesh: Mapping[str, torch.Tensor],
    input_suffix: Optional[str],
    input_convention: str,
) -> dict[str, torch.Tensor]:
    convention = resolved_input_convention(input_suffix, input_convention)
    if convention == "fbx":
        return dict(mesh)
    if convention != "custom-glb":
        raise ValueError("input_convention must be one of: auto, fbx, custom-glb.")

    oriented = dict(mesh)
    oriented["vertices"] = custom_glb_to_dataset_axes(mesh["vertices"])
    oriented["normals"] = F.normalize(
        custom_glb_to_dataset_axes(mesh["normals"]),
        dim=-1,
        eps=1.0e-6,
    )
    return oriented


def resolved_input_convention(
    input_suffix: Optional[str],
    input_convention: str,
) -> str:
    convention = str(input_convention).strip().lower()
    if convention != "auto":
        return convention
    return "custom-glb" if str(input_suffix or "").lower() == ".glb" else "fbx"


def custom_glb_to_dataset_axes(values: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            values[..., 0],
            -values[..., 1],
            -values[..., 2],
        ),
        dim=-1,
    ).contiguous()


def metahuman_sample_mesh(sample_dir: Path) -> Path:
    if not sample_dir.is_dir():
        raise FileNotFoundError(sample_dir)

    candidates = [
        path
        for path in sorted(sample_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No supported reference mesh was found in {sample_dir}."
        )
    return candidates[0]


def ensure_mediapipe_landmarks(
    mesh_path: Path,
    cache_dir: Path,
    mapper_path: Path,
    blender: Optional[str],
    force: bool = False,
    prefer_sidecar: bool = False,
    require_surface_anchors: bool = False,
    blender_threads: Optional[int] = None,
) -> Path:
    if require_surface_anchors and mesh_path.suffix.lower() == ".glb":
        # Use the same multi-view, texture-aware detector as the final eye
        # exporter. This covers rotated/upside-down identities and prevents the
        # model-alignment stage from failing before complete-eye processing.
        detector_args = argparse.Namespace(
            landmarks=None,
            mapper=mapper_path,
            cache_dir=cache_dir,
            refresh_landmarks=force,
            blender_threads=blender_threads,
            debug=False,
        )
        blender_executable = complete_eye_au._resolve_blender(blender)
        return complete_eye_au.ensure_landmarks(
            detector_args,
            blender_executable,
            mesh_path,
        )

    def usable(path: Path) -> bool:
        if not path.is_file():
            return False
        if not require_surface_anchors:
            return True
        try:
            payload = load_landmark_payload(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        anchors = payload.get("surface_anchors")
        offsets = payload.get("mesh_object_offsets")
        return isinstance(anchors, Mapping) and bool(anchors) and isinstance(
            offsets, Mapping
        )

    sidecar_path = mesh_path.with_suffix(".json")
    if prefer_sidecar and usable(sidecar_path) and not force:
        return sidecar_path

    cache_path = landmark_cache_path(mesh_path, cache_dir)
    if usable(cache_path) and not force:
        return cache_path
    if usable(sidecar_path) and not force:
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
    if require_surface_anchors:
        command.append("--output-surface-anchors")

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


def landmark_cache_path(mesh_path: Path, cache_dir: Path) -> Path:
    resolved = str(mesh_path.expanduser().resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{mesh_path.stem}_{digest}.json"


def load_landmark_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected {path} to contain a landmark mapping.")
    return loaded


def landmark_mapping_from_payload(loaded: Mapping[str, Any]) -> dict[int, int]:

    if isinstance(loaded.get("mapping"), dict):
        mapping = loaded["mapping"]
    else:
        mapping = loaded

    parsed: dict[int, int] = {}
    for raw_mediapipe_id, raw_vertex_id in mapping.items():
        if raw_vertex_id is None:
            continue
        parsed[int(raw_mediapipe_id)] = int(raw_vertex_id)
    if not parsed:
        raise ValueError("No usable MediaPipe landmarks were found in the payload.")
    return parsed


def load_landmark_mapping(path: Path) -> dict[int, int]:
    return landmark_mapping_from_payload(load_landmark_payload(path))


def load_reference_mesh_for_alignment(path: Path) -> dict[str, torch.Tensor]:
    mesh = load_mesh_for_model(path)
    mesh = {
        "vertices": mesh["vertices"].detach().cpu().float().contiguous(),
        "faces": mesh["faces"].detach().cpu().long().contiguous(),
        "normals": mesh["normals"].detach().cpu().float().contiguous(),
    }
    return orient_input_mesh_to_dataset_convention(
        mesh,
        input_suffix=path.suffix.lower(),
        input_convention="auto",
    )


def align_mesh_to_reference_landmarks(
    mesh: Mapping[str, torch.Tensor],
    input_landmarks: Mapping[int, int],
    reference_mesh: Mapping[str, torch.Tensor],
    reference_landmarks: Mapping[int, int],
    min_landmarks: int = 32,
) -> dict[str, torch.Tensor]:
    source_points, target_points = shared_landmark_positions(
        mesh=mesh,
        input_landmarks=input_landmarks,
        reference_mesh=reference_mesh,
        reference_landmarks=reference_landmarks,
        min_landmarks=min_landmarks,
    )
    scale, rotation, translation = similarity_procrustes(
        source_points,
        target_points,
    )
    return apply_similarity_transform(mesh, scale, rotation, translation)


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
    landmarks_3d = mesh.get("landmarks_3d")
    if isinstance(landmarks_3d, Mapping):
        transformed["landmarks_3d"] = transform_landmark_payload(
            landmarks_3d,
            scale,
            rotation,
            translation,
        )
    return transformed


def transform_landmark_payload(
    landmarks: Mapping[str, torch.Tensor],
    scale: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    transformed = dict(landmarks)
    for key in ("neutral_positions", "target_positions"):
        value = transformed.get(key)
        if isinstance(value, torch.Tensor) and value.shape[-1:] == (3,):
            transformed[key] = (value.matmul(rotation) * scale + translation).contiguous()
    target_delta = transformed.get("target_delta")
    if isinstance(target_delta, torch.Tensor) and target_delta.shape[-1:] == (3,):
        transformed["target_delta"] = (target_delta.matmul(rotation) * scale).contiguous()
    return transformed


def output_vertices_for_convention(
    vertices: torch.Tensor,
    output_convention: str,
) -> torch.Tensor:
    convention = str(output_convention).strip().lower()
    if convention == "model":
        return vertices
    if convention == "metahuman":
        return model_to_metahuman_axes(vertices)
    raise ValueError("output_convention must be one of: metahuman, model.")


def model_to_metahuman_axes(values: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            -values[..., 0],
            -values[..., 1],
            values[..., 2],
        ),
        dim=-1,
    ).contiguous()


def mesh_preprocess_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    mesh_args = configured_mesh_args(config)
    return {
        "mesh_up_axis": _config_value(mesh_args, "mesh_up_axis", "z"),
        "mesh_front_axis": _config_value(mesh_args, "mesh_front_axis", "-y"),
        "normalize_on_get": bool(
            _config_value(mesh_args, "normalize_on_get", True)
        ),
        "normalized_extent": float(
            _config_value(mesh_args, "normalized_extent", 2.0)
        ),
    }


def inference_weld_settings(
    config: Mapping[str, Any],
    *,
    tolerance_override: Optional[float] = None,
) -> tuple[float, set[int]]:
    image_args = configured_image_args(config)
    tolerance = (
        float(tolerance_override)
        if tolerance_override is not None
        else float(_config_value(image_args, "weld_tolerance", 1.0e-6))
    )
    if tolerance <= 0.0:
        raise ValueError("Inference weld tolerance must be positive.")
    raw_landmark_ids = image_args.get("weld_landmark_ids_to_largest_component", ())
    if raw_landmark_ids is None:
        raw_landmark_ids = ()
    if isinstance(raw_landmark_ids, (str, bytes)) or not isinstance(
        raw_landmark_ids,
        (list, tuple, set),
    ):
        raise ValueError(
            "weld_landmark_ids_to_largest_component must be a sequence of IDs."
        )
    return tolerance, {int(value) for value in raw_landmark_ids}


def configured_image_args(config: Mapping[str, Any]) -> Mapping[str, Any]:
    data_config = config.get("data", {})
    if not isinstance(data_config, Mapping):
        return {}

    for split_name in ("val", "train"):
        split_config = data_config.get(split_name, {})
        if not isinstance(split_config, Mapping):
            continue
        image_args = split_config.get("image_args", {})
        if isinstance(image_args, Mapping):
            return image_args
    return {}


def configured_mesh_args(config: Mapping[str, Any]) -> Mapping[str, Any]:
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


def _config_value(
    config: Mapping[str, Any],
    key: str,
    default: Any,
) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def load_obj_for_model(path: Path) -> dict[str, torch.Tensor]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if parts[0] == "v":
                if len(parts) < 4:
                    raise ValueError(f"Malformed vertex line in {path}: {stripped}")
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                face = [
                    parse_obj_vertex_index(token, len(vertices))
                    for token in parts[1:]
                ]
                if len(face) < 3:
                    raise ValueError(f"Malformed face line in {path}: {stripped}")
                for index in range(1, len(face) - 1):
                    faces.append([face[0], face[index], face[index + 1]])

    if not vertices:
        raise ValueError(f"No OBJ vertices were found in {path}.")
    if not faces:
        raise ValueError(f"No OBJ triangle faces were found in {path}.")

    vertex_tensor = torch.tensor(vertices, dtype=torch.float32)
    face_tensor = torch.tensor(faces, dtype=torch.long)
    return {
        "vertices": vertex_tensor,
        "faces": face_tensor,
        "normals": recompute_vertex_normals(vertex_tensor, face_tensor),
    }


def parse_obj_vertex_index(token: str, vertex_count: int) -> int:
    raw_index = token.split("/", 1)[0]
    if not raw_index:
        raise ValueError(f"OBJ face token {token!r} does not contain a vertex index.")
    index = int(raw_index)
    if index < 0:
        index = vertex_count + index
    else:
        index -= 1
    if index < 0 or index >= vertex_count:
        raise ValueError(f"OBJ vertex index {raw_index!r} is out of range.")
    return index


@torch.no_grad()
def run_model(
    model: TopoRig,
    mesh: Mapping[str, torch.Tensor],
    action_unit_id: int,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    vertices = mesh["vertices"].to(device=device, dtype=torch.float32).unsqueeze(0)
    normals = mesh.get("normals")
    if normals is not None:
        normals = normals.to(device=device, dtype=torch.float32).unsqueeze(0)
    faces = mesh["faces"].to(device=device, dtype=torch.long)
    facs = train.action_unit_vector(
        action_unit=action_unit_id,
        facs_dim=int(config["model"]["facs_dim"]),
        device=device,
        scale=float(config.get("training", {}).get("action_unit_scale", 1.0)),
    )
    landmark_features = train.build_model_landmark_features(
        record={"landmarks_3d": mesh.get("landmarks_3d")},
        vertices=vertices,
        config=config,
        device=device,
    )
    deformed_vertices, predicted_delta = model(
        vertices,
        facs,
        normals=normals,
        faces=faces,
        landmark_features=landmark_features,
        return_deformed=True,
    )
    return (
        deformed_vertices[0].detach().cpu(),
        predicted_delta[0].detach().cpu(),
    )


def predicted_eye_gaze_screen_motion(
    *,
    mesh: Mapping[str, Any],
    deformed_vertices: torch.Tensor,
    action_unit_id: int,
    config: Mapping[str, Any],
) -> torch.Tensor:
    """Measure model gaze relative to stationary eye corners in eye-width units."""

    spec = eye_gaze_landmark_spec(action_unit_id)
    if spec is None:
        raise ValueError(f"AU{action_unit_id} is not an eye-gaze action unit.")
    vertices = mesh["vertices"].detach().cpu().float().unsqueeze(0)
    deformed_vertices = deformed_vertices.detach().cpu().float().unsqueeze(0)
    if deformed_vertices.shape != vertices.shape:
        raise ValueError("deformed_vertices must match the model mesh vertices.")
    required_ids = (*spec.iris_ids, *spec.corner_ids)
    vertex_ids = train.eyelid_vertex_ids(
        mesh.get("landmarks_3d"),
        required_ids,
        device=torch.device("cpu"),
    )
    if vertex_ids is None:
        raise ValueError(
            f"AU{action_unit_id} UV gaze requires iris and eye-corner landmarks."
        )

    render_config = config.get("rendering", {})
    if not isinstance(render_config, Mapping):
        raise ValueError("Checkpoint config is missing rendering settings.")
    render_up_axis = train.resolve_validation_render_up_axis(config, vertices)
    neutral_render_vertices = train.vertices_to_render_axes(vertices, render_up_axis)
    deformed_render_vertices = train.vertices_to_render_axes(
        deformed_vertices, render_up_axis
    )
    image_size = train.normalize_image_size(
        render_config.get("train_image_size", (256, 256))
    )
    render_kwargs = train.image_loss_render_kwargs(render_config)
    neutral_points = train.validation_project_points_to_screen(
        points=neutral_render_vertices[:, vertex_ids],
        image_size=image_size,
        render_kwargs=render_kwargs,
        fit_reference_vertices=neutral_render_vertices,
    )[0]
    deformed_points = train.validation_project_points_to_screen(
        points=deformed_render_vertices[:, vertex_ids],
        image_size=image_size,
        render_kwargs=render_kwargs,
        fit_reference_vertices=neutral_render_vertices,
    )[0]
    corner_start = len(spec.iris_ids)
    corner_indices = torch.tensor(
        (corner_start, corner_start + 1), dtype=torch.long
    )
    eye_width = torch.linalg.vector_norm(
        neutral_points[corner_indices[1]] - neutral_points[corner_indices[0]]
    ).clamp_min(1.0)
    motion = (deformed_points - neutral_points) / eye_width
    corner_motion = motion.index_select(0, corner_indices).mean(dim=0)
    center_index = spec.iris_ids.index(spec.center_id)
    return (motion[center_index] - corner_motion).detach().cpu().float()


def clamp_eye_gaze_motion(
    motion: torch.Tensor,
    *,
    maximum_norm: float,
) -> torch.Tensor:
    motion = torch.as_tensor(motion, dtype=torch.float32).flatten()
    if motion.shape != (2,) or not torch.isfinite(motion).all():
        raise ValueError("Eye-gaze motion must contain two finite values.")
    if maximum_norm <= 0.0:
        raise ValueError("eye-gaze UV maximum motion must be positive.")
    norm = torch.linalg.vector_norm(motion)
    if float(norm) > float(maximum_norm):
        motion = motion * (float(maximum_norm) / norm)
    return motion.contiguous()


def default_output_paths(
    mesh_path: Path,
    output_dir: Optional[Path],
    action_unit_id: int,
    output_format: str,
) -> tuple[Path, Path]:
    directory = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else mesh_path.parent
    )
    stem = f"{mesh_path.stem}_au{action_unit_id}"
    return directory / f"{stem}.jpg", directory / f"{stem}.{output_format}"


@torch.no_grad()
def render_output_image(
    neutral_vertices: torch.Tensor,
    deformed_vertices: torch.Tensor,
    normals: Optional[torch.Tensor],
    faces: torch.Tensor,
    action_unit_id: int,
    config: Mapping[str, Any],
    device: torch.device,
    vertex_colors: Optional[torch.Tensor] = None,
) -> Image.Image:
    neutral_vertices = neutral_vertices.to(
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    deformed_vertices = deformed_vertices.to(
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    faces = faces.to(device=device, dtype=torch.long)
    render_normals = None
    if normals is not None:
        render_normals = normals.to(device=device, dtype=torch.float32).unsqueeze(0)

    render_size = train.normalize_image_size(
        config.get("rendering", {}).get("validation_image_size", (256, 256))
    )
    render_up_axis = train.resolve_validation_render_up_axis(config, neutral_vertices)
    render_vertices = train.vertices_to_render_axes(neutral_vertices, render_up_axis)
    render_deformed_vertices = train.vertices_to_render_axes(
        deformed_vertices,
        render_up_axis,
    )
    if render_normals is not None:
        render_normals = train.vertices_to_render_axes(render_normals, render_up_axis)

    render_center = train.render_vertex_bounds_center(render_vertices)
    render_vertices = render_vertices - render_center
    render_deformed_vertices = render_deformed_vertices - render_center
    if vertex_colors is None:
        render_vertex_colors = train.validation_vertex_colors(
            render_vertices,
            normals=render_normals,
        )
    else:
        render_vertex_colors = vertex_colors.to(
            device=device,
            dtype=torch.float32,
        )
        if render_vertex_colors.ndim == 2:
            render_vertex_colors = render_vertex_colors.unsqueeze(0)
        if render_vertex_colors.shape != render_vertices.shape:
            raise ValueError(
                "vertex_colors must have shape [V, 3] or [1, V, 3] matching vertices."
            )
    texture = {"vertex_colors": render_vertex_colors}

    rendering_config = config.get("rendering", {})
    backend = str(rendering_config.get("backend", "auto"))
    render_kwargs = rendering_config.get("kwargs", {})
    neutral_panel = train.render_validation_mesh_panel(
        render_vertices,
        faces,
        texture=texture,
        image_size=render_size,
        backend=backend,
        render_kwargs=render_kwargs,
    )
    prediction_panel = train.render_validation_mesh_panel(
        render_deformed_vertices,
        faces,
        texture=texture,
        image_size=render_size,
        backend=backend,
        render_kwargs=render_kwargs,
    )
    label = train.action_unit_label(action_unit_id)
    return train.concatenate_labeled_image_grid(
        [[neutral_panel, prediction_panel]],
        [[f"{label} neutral", f"{label} prediction"]],
    )


def matching_material_slot_indices(
    material_slots: Sequence[Any],
    patterns: Sequence[str],
) -> dict[int, str]:
    normalized_patterns = tuple(
        pattern.strip().lower() for pattern in patterns if pattern.strip()
    )
    matches: dict[int, str] = {}
    for index, slot in enumerate(material_slots):
        material = getattr(slot, "material", None)
        name = str(
            getattr(material, "name", None) or getattr(slot, "name", "")
        )
        if any(pattern in name.lower() for pattern in normalized_patterns):
            matches[index] = name
    return matches


def remove_faces_with_material_patterns(
    mesh_objects: Sequence[Any],
    patterns: Sequence[str],
) -> dict[str, Any]:
    """Remove rendered faces for selected materials while retaining vertex order."""

    import bmesh

    normalized_patterns = tuple(
        pattern.strip().lower() for pattern in patterns if pattern.strip()
    )
    report: dict[str, Any] = {
        "material_patterns": list(normalized_patterns),
        "material_names": [],
        "removed_face_count": 0,
        "objects": {},
    }
    material_names: set[str] = set()
    for obj in mesh_objects:
        matching_slots = matching_material_slot_indices(
            obj.material_slots,
            normalized_patterns,
        )
        if not matching_slots:
            continue
        editable = bmesh.new()
        try:
            editable.from_mesh(obj.data)
            removable_faces = [
                face
                for face in editable.faces
                if int(face.material_index) in matching_slots
            ]
            removed_count = len(removable_faces)
            if removable_faces:
                # FACES_ONLY keeps vertex IDs stable for retained-topology output.
                bmesh.ops.delete(
                    editable,
                    geom=removable_faces,
                    context="FACES_ONLY",
                )
                editable.to_mesh(obj.data)
                obj.data.update(calc_edges=True)
        finally:
            editable.free()
        if removed_count == 0:
            continue
        names = sorted(set(matching_slots.values()))
        material_names.update(names)
        report["objects"][str(obj.name)] = {
            "material_names": names,
            "removed_face_count": removed_count,
        }
        report["removed_face_count"] += removed_count
    report["material_names"] = sorted(material_names)
    report["applied"] = report["removed_face_count"] > 0
    return report


def export_fbx_appearance_preserving_glb(
    *,
    source_path: Path,
    vertices: Optional[torch.Tensor],
    path: Path,
    mesh_object_names: Optional[Sequence[str]] = None,
    eye_gaze_uv_warp: Optional[Mapping[str, Any]] = None,
    exclude_material_patterns: Optional[Sequence[str]] = None,
) -> Optional[dict[str, Any]]:
    try:
        import bpy
        import numpy as np
        from mathutils import Matrix
    except ImportError as exc:
        raise ImportError(
            "Appearance-preserving FBX export requires Blender's bpy module."
        ) from exc

    source_path = source_path.expanduser().resolve()
    path = path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if path.suffix.lower() != ".glb":
        raise ValueError("Appearance-preserving export requires a GLB output path.")
    if vertices is not None and (vertices.ndim != 2 or vertices.shape[1] != 3):
        raise ValueError("vertices must have shape [V, 3].")
    temporary_path = path.with_name(f".{path.stem}.exporting{path.suffix}")
    temporary_path.unlink(missing_ok=True)
    if eye_gaze_uv_warp is not None:
        path.unlink(missing_ok=True)

    before_objects = set(bpy.data.objects)
    if source_path.suffix.lower() == ".glb":
        bpy.ops.import_scene.gltf(filepath=str(source_path))
    else:
        bpy.ops.import_scene.fbx(filepath=str(source_path))
    imported_objects = tuple(
        obj for obj in bpy.data.objects if obj not in before_objects
    )
    try:
        mesh_objects = source_mesh_objects_for_export(
            imported_objects,
            mesh_object_names,
        )
        vertex_count = sum(len(obj.data.vertices) for obj in mesh_objects)
        if vertices is None:
            neutral_world_vertices = []
            for obj in mesh_objects:
                shape_keys = obj.data.shape_keys
                basis = (
                    shape_keys.key_blocks.get("Basis")
                    if shape_keys is not None
                    else None
                )
                coordinate_source = basis.data if basis is not None else obj.data.vertices
                coordinates = np.empty((len(coordinate_source), 3), dtype=np.float64)
                coordinate_source.foreach_get("co", coordinates.reshape(-1))
                matrix_world = np.asarray(
                    [
                        [obj.matrix_world[row][column] for column in range(4)]
                        for row in range(4)
                    ],
                    dtype=np.float64,
                )
                neutral_world_vertices.append(
                    coordinates @ matrix_world[:3, :3].T + matrix_world[:3, 3]
                )
            vertices = torch.from_numpy(
                np.concatenate(neutral_world_vertices, axis=0)
            ).float()
        if vertex_count != int(vertices.shape[0]):
            details = ", ".join(
                f"{obj.name}={len(obj.data.vertices)}" for obj in mesh_objects
            )
            raise ValueError(
                "Source appearance mesh vertex count does not match the retained model "
                f"topology: source={vertex_count}, prediction={vertices.shape[0]} "
                f"({details})."
            )
        if not any(len(obj.data.uv_layers) > 0 for obj in mesh_objects):
            raise ValueError("Source FBX does not contain a UV layer.")
        if not any(len(obj.material_slots) > 0 for obj in mesh_objects):
            raise ValueError("Source FBX does not contain a material.")

        blender_vertices = glb_vertices_to_blender_axes(vertices)
        blender_vertices = blender_vertices.detach().cpu().float().numpy()
        bpy.ops.object.select_all(action="DESELECT")
        neutral_object_coordinates: dict[Any, Any] = {}
        vertex_offset = 0
        for obj in mesh_objects:
            count = len(obj.data.vertices)
            neutral_coordinates = np.empty((count, 3), dtype=np.float64)
            obj.data.vertices.foreach_get("co", neutral_coordinates.reshape(-1))
            neutral_object_coordinates[obj] = neutral_coordinates
            object_vertices = np.ascontiguousarray(
                blender_vertices[vertex_offset : vertex_offset + count]
            )
            vertex_offset += count

            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            if obj.data.shape_keys is not None:
                bpy.ops.object.shape_key_remove(all=True, apply_mix=False)
            obj.modifiers.clear()
            obj.parent = None
            obj.matrix_world = Matrix.Identity(4)
            obj.data.vertices.foreach_set("co", object_vertices.reshape(-1))
            obj.data.update(calc_edges=True)

        eye_gaze_report = None
        if eye_gaze_uv_warp is not None:
            action_unit_id = int(eye_gaze_uv_warp["action_unit_id"])
            requested_appearance_mode = str(
                eye_gaze_uv_warp.get("appearance_mode", "auto")
            )
            appearance_mode = normalize_eye_gaze_appearance_mode(
                requested_appearance_mode
            )
            if appearance_mode not in {
                "auto",
                "hybrid",
                "hybrid_copy",
                "uv_hybrid",
                "uv_layers",
                "rigid_uv_layers",
                "rigid_texture_layers",
                "rigid_texture_cleanup_layers",
                "stacked_texture_layers",
                "stacked_uv_layers",
                "surface_uv_primary",
                "surface_uv_selected",
                "surface_texture_bake",
                "stacked_surface_texture_bake",
                "screen_texture_bake",
                "color_segmented_texture",
                "rotate_eye",
                "translate_eye",
                "texture_only",
                "geometry_only",
                "surface_texture",
                "segmented_texture",
            }:
                raise ValueError(
                    f"Unsupported eye-gaze appearance mode: {appearance_mode!r}."
                )
            layer_payloads = eye_gaze_layer_payloads(
                eye_gaze_uv_warp["landmark_payload"],
                action_unit_id,
            )
            use_packed_texture_warp = eye_gaze_has_visible_texture_layers(
                eye_gaze_uv_warp["landmark_payload"],
                action_unit_id,
            )
            if appearance_mode == "color_segmented_texture":
                use_packed_texture_warp = True
            resolved_appearance_mode = appearance_mode
            topology_component_polygon_count = None
            physical_selection = None
            fallback_reason = None
            if appearance_mode == "auto":
                physical_selection = eye_gaze_physical_assembly_selection(
                    mesh_objects,
                    eye_gaze_uv_warp["landmark_payload"],
                    action_unit_id,
                )
                if physical_selection is None:
                    # Release the full-resolution connectivity index before
                    # allocating packed-texture pixel buffers for fallback.
                    import gc

                    gc.collect()
                    try:
                        import ctypes

                        trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
                        if trim is not None:
                            trim(0)
                    except (AttributeError, OSError):
                        pass
                recommended_mesh_method = (
                    physical_selection.get("auto_recommended_method")
                    if physical_selection is not None
                    else None
                )
                resolved_appearance_mode = {
                    "rigid_rotation": "rotate_eye",
                    "local_translation": "translate_eye",
                    "texture_fallback": "color_segmented_texture",
                }.get(recommended_mesh_method, "color_segmented_texture")
                if resolved_appearance_mode == "color_segmented_texture":
                    fallback_reason = (
                        "No independent sphere-like eye assembly passed physical "
                        "selection; using the original-surface texture fallback."
                        if physical_selection is None
                        else "Independent rigid eye motion could not be proven for "
                        "the fragmented eye layers; using the original-surface "
                        "texture fallback."
                    )
                    use_packed_texture_warp = True
            elif appearance_mode in {"rotate_eye", "translate_eye"}:
                physical_selection = eye_gaze_physical_assembly_selection(
                    mesh_objects,
                    eye_gaze_uv_warp["landmark_payload"],
                    action_unit_id,
                )
                if physical_selection is None:
                    raise EyeGazeUVWarpValidationError(
                        "Mesh gaze requires an independent, sphere-like eye "
                        "component whose visible iris layer belongs to the same "
                        "rigid assembly."
                    )
                if requested_appearance_mode == "mesh" and (
                    physical_selection.get("assembly_strategy")
                    == "partial_consensus_assembly"
                ):
                    raise EyeGazeUVWarpValidationError(
                        "Mesh gaze rejected a fragmented partial-consensus eye "
                        "assembly because moving its disconnected front patches "
                        "would expose stationary rear layers. Use auto or texture "
                        "mode for this asset."
                    )
                if requested_appearance_mode == "mesh" and bool(
                    physical_selection.get("ambiguous_occluded_duplicate", False)
                ):
                    raise EyeGazeUVWarpValidationError(
                        "Mesh gaze rejected an ambiguous concentric dark eye "
                        "duplicate because rotating either copy exposes the other. "
                        "Use auto or texture mode for this asset."
                    )
                if requested_appearance_mode == "mesh" and bool(
                    physical_selection.get(
                        "foreground_non_eye_iris_components", []
                    )
                ):
                    raise EyeGazeUVWarpValidationError(
                        "Mesh gaze rejected a dark foreground iris image on "
                        "non-eye geometry because it would remain stationary "
                        "over the rotated eye assembly. Use auto or texture mode "
                        "for this asset."
                    )
                if requested_appearance_mode == "mesh" and (
                    physical_selection.get("recommended_method")
                    == "local_translation"
                ):
                    # Layered Pixel3D eyes are often fragmented into several
                    # independent front patches.  Rotating a fitted rear shell
                    # then exposes the neutral/skin-coloured backs of those
                    # patches.  Mesh mode remains geometry-only, but uses one
                    # shared tangent translation for the existing bounded eye
                    # layers when a complete rigid front shell is not proven.
                    resolved_appearance_mode = "translate_eye"
            elif appearance_mode == "color_segmented_texture":
                # The general landmark cache can point at a rear duplicate on
                # heavily layered Pixel3D eyes.  Reuse the component evaluator
                # to seed texture segmentation from its foremost coherent dark
                # iris layer, while leaving geometry untouched.
                physical_selection = eye_gaze_physical_assembly_selection(
                    mesh_objects,
                    eye_gaze_uv_warp["landmark_payload"],
                    action_unit_id,
                )
            if resolved_appearance_mode in {
                "rotate_eye",
                "translate_eye",
                "stacked_texture_layers",
                "stacked_uv_layers",
                "stacked_surface_texture_bake",
            }:
                layer_payloads = eye_gaze_geometry_layer_payloads(
                    eye_gaze_uv_warp["landmark_payload"],
                    action_unit_id,
                )
            if (
                resolved_appearance_mode in {"rotate_eye", "translate_eye"}
                and physical_selection is not None
            ):
                layer_payloads = physical_selection[
                    "local_translation_layer_payloads"
                    if resolved_appearance_mode == "translate_eye"
                    else "layer_payloads"
                ]
            if resolved_appearance_mode == "color_segmented_texture":
                # Packed Pixel3D eyes can contain a visible duplicate shell
                # directly behind the painted face surface. Moving only the
                # primary texture exposes that neutral copy. Deeper ray hits
                # are not safe texture targets because their UV islands may
                # belong to unrelated rear-facing head surfaces.
                texture_landmark_payload = eye_gaze_uv_warp["landmark_payload"]
                if physical_selection is not None:
                    layer_payloads = physical_selection[
                        "texture_layer_payloads"
                    ]
                else:
                    layer_payloads = eye_gaze_texture_fallback_layer_payloads(
                        texture_landmark_payload, action_unit_id
                    )
            if resolved_appearance_mode in {
                "surface_uv_primary",
                "surface_texture_bake",
                "screen_texture_bake",
            }:
                layer_payloads = eye_gaze_geometry_layer_payloads(
                    eye_gaze_uv_warp["landmark_payload"],
                    action_unit_id,
                )[:1]
            if resolved_appearance_mode == "surface_uv_selected":
                available_layers = eye_gaze_geometry_layer_payloads(
                    eye_gaze_uv_warp["landmark_payload"],
                    action_unit_id,
                )
                selected_index = int(
                    eye_gaze_uv_warp.get("surface_uv_layer_index", 0)
                )
                if not 0 <= selected_index < len(available_layers):
                    raise EyeGazeUVWarpValidationError(
                        "Selected eye surface layer is unavailable: "
                        f"index={selected_index}, layers={len(available_layers)}."
                    )
                layer_payloads = (
                    available_layers[:1]
                    if selected_index == 0
                    else [available_layers[0], available_layers[selected_index]]
                )
            layer_reports = []
            processed_geometry_components: set[tuple[str, int]] = set()
            geometry_topology_indices: dict[int, dict[str, Any]] = {}
            geometry_coordinate_cache: dict[int, tuple[Any, Any]] = {}
            for layer_index, layer_payload in enumerate(layer_payloads):
                layer_reports.append(
                    apply_fbx_eye_gaze_uv_warp(
                        mesh_objects=mesh_objects,
                        landmark_payload=layer_payload,
                        action_unit_id=action_unit_id,
                        motion=torch.as_tensor(eye_gaze_uv_warp["motion"]).float(),
                        rigid_uv_translation=(
                            resolved_appearance_mode == "rigid_uv_layers"
                        ),
                        apply_uv_coordinates=(
                            resolved_appearance_mode
                            not in {
                                "stacked_uv_layers",
                                "surface_uv_primary",
                                "surface_uv_selected",
                                "surface_texture_bake",
                                "stacked_surface_texture_bake",
                                "screen_texture_bake",
                                "color_segmented_texture",
                                "rotate_eye",
                                "translate_eye",
                            }
                            and (
                                not use_packed_texture_warp
                                or (
                                    resolved_appearance_mode
                                    in {
                                        "uv_hybrid",
                                        "uv_layers",
                                        "rigid_uv_layers",
                                    }
                                    and (
                                        resolved_appearance_mode
                                        in {"uv_layers", "rigid_uv_layers"}
                                        or layer_index == 0
                                    )
                                )
                            )
                        ),
                        apply_geometry_translation=(
                            use_packed_texture_warp
                            and resolved_appearance_mode
                            in {
                                "hybrid",
                                "hybrid_copy",
                                "uv_hybrid",
                                "geometry_only",
                            }
                            and layer_index > 0
                        ),
                        rotate_geometry_component=(
                            resolved_appearance_mode in {"rotate_eye", "translate_eye"}
                        ),
                        force_local_eye_translation=(
                            resolved_appearance_mode == "translate_eye"
                        ),
                        geometry_rotation_center=(
                            layer_reports[0].get("geometry_rotation_center")
                            if layer_index > 0 and layer_reports
                            else None
                        ),
                        geometry_rotation_center_hint=(
                            physical_selection["rotation_center"]
                            if (
                                layer_index == 0
                                and physical_selection is not None
                                and resolved_appearance_mode
                                in {"rotate_eye", "translate_eye"}
                            )
                            else None
                        ),
                        geometry_rotation_iris_position_hint=(
                            physical_selection["rotation_iris_position"]
                            if (
                                layer_index == 0
                                and physical_selection is not None
                                and resolved_appearance_mode
                                in {"rotate_eye", "translate_eye"}
                            )
                            else None
                        ),
                        geometry_rotation_radius_hint=(
                            physical_selection["rotation_radius"]
                            if (
                                layer_index == 0
                                and physical_selection is not None
                                and resolved_appearance_mode
                                in {"rotate_eye", "translate_eye"}
                            )
                            else None
                        ),
                        geometry_rotation_matrix=(
                            layer_reports[0].get("geometry_rotation_matrix")
                            if layer_index > 0 and layer_reports
                            else None
                        ),
                        geometry_shared_translation=(
                            layer_reports[0].get("geometry_shared_translation")
                            if layer_index > 0 and layer_reports
                            else None
                        ),
                        geometry_translation_center=(
                            layer_reports[0].get("geometry_translation_center")
                            if layer_index > 0 and layer_reports
                            else None
                        ),
                        geometry_translation_normal=(
                            layer_reports[0].get("geometry_translation_normal")
                            if layer_index > 0 and layer_reports
                            else None
                        ),
                        geometry_translation_inner_radius=(
                            layer_reports[0].get(
                                "geometry_translation_inner_radius"
                            )
                            if layer_index > 0 and layer_reports
                            else None
                        ),
                        geometry_translation_outer_radius=(
                            layer_reports[0].get(
                                "geometry_translation_outer_radius"
                            )
                            if layer_index > 0 and layer_reports
                            else None
                        ),
                        geometry_seed_only=bool(
                            layer_payload.get("eye_gaze_geometry_seed_only", False)
                            and resolved_appearance_mode
                            in {"rotate_eye", "translate_eye"}
                        ),
                        geometry_calibration_only=bool(
                            layer_payload.get(
                                "eye_gaze_geometry_calibration_only", False
                            )
                            and resolved_appearance_mode
                            in {"rotate_eye", "translate_eye"}
                        ),
                        texture_seed_only=bool(
                            layer_payload.get("eye_gaze_geometry_seed_only", False)
                            and resolved_appearance_mode
                            in {
                                "stacked_texture_layers",
                                "color_segmented_texture",
                            }
                        ),
                        spatial_uv_seed_only=bool(
                            layer_payload.get("eye_gaze_geometry_seed_only", False)
                            and resolved_appearance_mode
                            in {"stacked_uv_layers", "surface_uv_selected"}
                        ),
                        surface_texture_bake_seed_only=bool(
                            layer_payload.get("eye_gaze_geometry_seed_only", False)
                            and resolved_appearance_mode
                            == "stacked_surface_texture_bake"
                        ),
                        apply_spatial_uv_sampling=(
                            resolved_appearance_mode
                            in {"stacked_uv_layers", "surface_uv_primary"}
                            or (
                                resolved_appearance_mode == "surface_uv_selected"
                                and (
                                    len(layer_payloads) == 1
                                    or layer_index > 0
                                )
                            )
                        ),
                        apply_surface_texture_bake=(
                            resolved_appearance_mode
                            in {
                                "surface_texture_bake",
                                "stacked_surface_texture_bake",
                                "screen_texture_bake",
                            }
                            and layer_index == 0
                        ),
                        apply_screen_texture_bake=(
                            resolved_appearance_mode
                            == "screen_texture_bake"
                        ),
                        use_rendered_reference_colors=False,
                        texture_shared_world_shift=(
                            (
                                layer_reports[0].get("world_shift_x"),
                                layer_reports[0].get("world_shift_y"),
                                layer_reports[0].get("world_shift_z"),
                            )
                            if layer_index > 0 and layer_reports
                            else None
                        ),
                        texture_shared_iris_radius=(
                            layer_reports[0].get("iris_position_radius_3d")
                            if layer_index > 0 and layer_reports
                            else None
                        ),
                        processed_geometry_components=(
                            processed_geometry_components
                            if resolved_appearance_mode
                            in {"rotate_eye", "translate_eye"}
                            else None
                        ),
                        geometry_topology_indices=(
                            geometry_topology_indices
                            if resolved_appearance_mode
                            in {"rotate_eye", "translate_eye"}
                            else None
                        ),
                        geometry_coordinate_cache=(
                            geometry_coordinate_cache
                            if resolved_appearance_mode
                            in {"rotate_eye", "translate_eye"}
                            else None
                        ),
                        geometry_surface_offset_ratio=float(
                            eye_gaze_uv_warp.get(
                                "geometry_surface_offset_ratio",
                                0.0,
                            )
                        ),
                        remove_geometry_component=(
                            use_packed_texture_warp
                            and resolved_appearance_mode == "surface_texture"
                            and layer_index > 0
                        ),
                    )
                )
                layer_reports[-1]["texture_cleanup_only"] = bool(
                    layer_payload.get("eye_gaze_texture_cleanup_only", False)
                )
            for mesh, coordinates in geometry_coordinate_cache.values():
                mesh.vertices.foreach_set("co", coordinates.reshape(-1))
                mesh.update(calc_edges=True)
                if hasattr(mesh, "calc_normals"):
                    mesh.calc_normals()
            texture_report = None
            if use_packed_texture_warp:
                texture_report = None
                if resolved_appearance_mode in {
                    "hybrid",
                    "hybrid_copy",
                    "texture_only",
                    "surface_texture",
                    "segmented_texture",
                    "rigid_texture_layers",
                    "rigid_texture_cleanup_layers",
                    "stacked_texture_layers",
                    "color_segmented_texture",
                }:
                    if resolved_appearance_mode == "color_segmented_texture":
                        ordered_texture_reports = sorted(
                            layer_reports,
                            key=lambda report: bool(
                                report.get("texture_cleanup_only", False)
                            ),
                        )
                        texture_report = apply_fbx_eye_gaze_color_texture_warp(
                            mesh_objects=mesh_objects,
                            layer_reports=ordered_texture_reports,
                            eye_name=eye_gaze_landmark_spec(action_unit_id).eye,
                            mask_cache_dir=eye_gaze_uv_warp.get(
                                "texture_mask_cache_dir"
                            ),
                            mask_debug_dir=eye_gaze_uv_warp.get(
                                "texture_mask_debug_dir"
                            ),
                        )
                    else:
                        texture_report = apply_fbx_eye_gaze_texture_warp(
                        mesh_objects=mesh_objects,
                        layer_reports=(
                            layer_reports
                            if resolved_appearance_mode
                            in {
                                "segmented_texture",
                                "rigid_texture_layers",
                                "rigid_texture_cleanup_layers",
                                "stacked_texture_layers",
                            }
                            else layer_reports[:1]
                        ),
                        erase_radius_scale=(
                            1.08
                            if appearance_mode == "auto"
                            and resolved_appearance_mode == "hybrid"
                            else max(
                                1.60,
                                float(
                                    eye_gaze_uv_warp.get(
                                        "texture_erase_radius_scale",
                                        1.45,
                                    )
                                ),
                            )
                            if resolved_appearance_mode
                            in {
                                "rigid_texture_layers",
                                "rigid_texture_cleanup_layers",
                                "stacked_texture_layers",
                            }
                            else float(
                                eye_gaze_uv_warp.get(
                                    "texture_erase_radius_scale",
                                    1.45,
                                )
                            )
                        ),
                        copy_source_iris=resolved_appearance_mode
                        in {
                            "hybrid_copy",
                            "surface_texture",
                            "segmented_texture",
                            "rigid_texture_layers",
                            "rigid_texture_cleanup_layers",
                            "stacked_texture_layers",
                        },
                        fit_source_iris=(
                            resolved_appearance_mode == "segmented_texture"
                        ),
                        use_source_iris_mask=(
                            resolved_appearance_mode
                            in {
                                "rigid_texture_cleanup_layers",
                                "stacked_texture_layers",
                            }
                        ),
                        copy_source_iris_mask=(
                            resolved_appearance_mode
                            == "stacked_texture_layers"
                        ),
                        copy_radius_scale=(
                            1.08
                            if resolved_appearance_mode
                            in {
                                "rigid_texture_layers",
                                "rigid_texture_cleanup_layers",
                                "stacked_texture_layers",
                            }
                            else 1.0
                        ),
                        draw_target_iris=True,
                        maximum_target_layers=(
                            1
                            if resolved_appearance_mode
                            == "stacked_texture_layers"
                            else None
                        ),
                    )
                eye_gaze_report = {
                    "method": resolved_appearance_mode,
                    "requested_method": requested_appearance_mode,
                    "topology_component_polygon_count": (
                        topology_component_polygon_count
                    ),
                    "layer_count": len(layer_reports),
                    "layers": layer_reports,
                }
                if texture_report is not None:
                    eye_gaze_report["texture"] = texture_report
            else:
                eye_gaze_report = (
                    layer_reports[0]
                    if len(layer_reports) == 1
                    else {
                        "method": "uv_coordinate_warp",
                        "layer_count": len(layer_reports),
                        "layers": layer_reports,
                    }
                )
            geometry_rotated = sum(
                int(report.get("geometry_vertices_rotated", 0))
                for report in layer_reports
            )
            geometry_translated = sum(
                int(report.get("geometry_vertices_translated", 0))
                for report in layer_reports
            )
            geometry_polygons_removed = sum(
                int(report.get("geometry_polygons_removed", 0))
                for report in layer_reports
            )
            uv_modified = any(
                bool(report.get("uv_coordinates_modified", 0))
                for report in layer_reports
            )
            if resolved_appearance_mode == "rotate_eye":
                outcome_method = "rigid_rotation"
            elif resolved_appearance_mode == "translate_eye":
                outcome_method = "local_translation"
            elif resolved_appearance_mode in {
                "color_segmented_texture",
                "texture_only",
                "segmented_texture",
                "rigid_texture_layers",
                "rigid_texture_cleanup_layers",
                "stacked_texture_layers",
            }:
                outcome_method = "texture_fallback"
            else:
                outcome_method = resolved_appearance_mode
            eye_gaze_report = {
                "method": outcome_method,
                "implementation_method": resolved_appearance_mode,
                "requested_method": requested_appearance_mode,
                "geometry_modified": bool(
                    geometry_rotated > 0
                    or geometry_translated > 0
                    or geometry_polygons_removed > 0
                ),
                "topology_modified": bool(geometry_polygons_removed > 0),
                "uv_coordinates_modified": bool(uv_modified),
                "geometry_vertices_rotated": geometry_rotated,
                "geometry_vertices_translated": geometry_translated,
                "geometry_polygons_removed": geometry_polygons_removed,
                "generated_geometry_vertices": 0,
                "generated_geometry_polygons": 0,
                "topology_component_polygon_count": (
                    topology_component_polygon_count
                ),
                "layer_count": len(layer_reports),
                "layers": layer_reports,
            }
            if physical_selection is not None:
                eye_gaze_report["physical_assembly"] = {
                    key: value
                    for key, value in physical_selection.items()
                    if key
                    not in {
                        "layer_payloads",
                        "local_translation_layer_payloads",
                        "texture_layer_payloads",
                    }
                }
            if fallback_reason is not None:
                eye_gaze_report["fallback_reason"] = fallback_reason
            if texture_report is not None:
                eye_gaze_report["texture"] = texture_report

        material_face_removal = None
        if exclude_material_patterns:
            material_face_removal = remove_faces_with_material_patterns(
                mesh_objects,
                exclude_material_patterns,
            )
            if not material_face_removal["applied"]:
                raise ValueError(
                    "No FBX faces matched the requested excluded material "
                    f"patterns: {tuple(exclude_material_patterns)}"
                )

        triangulated_polygon_count = 0
        for obj in mesh_objects:
            triangulated_polygon_count += triangulate_mesh_in_reference_configuration(
                obj.data,
                neutral_object_coordinates[obj],
            )
        if eye_gaze_report is not None:
            eye_gaze_report["neutral_triangulated_polygon_count"] = int(
                triangulated_polygon_count
            )

        prepare_source_materials_for_glb(mesh_objects)
        path.parent.mkdir(parents=True, exist_ok=True)
        result = bpy.ops.export_scene.gltf(
            filepath=str(temporary_path),
            export_format="GLB",
            use_selection=True,
            export_apply=True,
            export_materials="EXPORT",
            export_texcoords=True,
            export_normals=True,
        )
        if "FINISHED" not in result or not temporary_path.is_file():
            raise RuntimeError(f"Blender did not write {path}.")
        temporary_path.replace(path)
        if material_face_removal is not None:
            if eye_gaze_report is None:
                eye_gaze_report = {}
            eye_gaze_report["material_face_removal"] = material_face_removal
        return eye_gaze_report
    finally:
        temporary_path.unlink(missing_ok=True)
        for obj in imported_objects:
            if obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)


def export_fbx_appearance_preserving_blendshape_glb(
    *,
    source_path: Path,
    neutral_vertices: torch.Tensor,
    blendshape_vertices: Mapping[str, torch.Tensor],
    path: Path,
    mesh_object_names: Optional[Sequence[str]] = None,
) -> None:
    """Export one textured GLB with predicted poses as named morph targets."""

    try:
        import bpy
        import numpy as np
        from mathutils import Matrix
    except ImportError as exc:
        raise ImportError(
            "Appearance-preserving blendshape export requires Blender's bpy module."
        ) from exc

    source_path = source_path.expanduser().resolve()
    path = path.expanduser().resolve()
    neutral_vertices = neutral_vertices.detach().cpu().float().contiguous()
    if neutral_vertices.ndim != 2 or neutral_vertices.shape[1] != 3:
        raise ValueError("neutral_vertices must have shape [V, 3].")
    if not blendshape_vertices:
        raise ValueError("At least one blendshape target is required.")
    targets = {
        str(name): vertices.detach().cpu().float().contiguous()
        for name, vertices in blendshape_vertices.items()
    }
    for name, vertices in targets.items():
        if vertices.shape != neutral_vertices.shape:
            raise ValueError(
                f"Blendshape {name!r} has shape {tuple(vertices.shape)}, expected "
                f"{tuple(neutral_vertices.shape)}."
            )

    temporary_path = path.with_name(f".{path.stem}.exporting{path.suffix}")
    temporary_path.unlink(missing_ok=True)
    before_objects = set(bpy.data.objects)
    if source_path.suffix.lower() == ".glb":
        bpy.ops.import_scene.gltf(filepath=str(source_path))
    else:
        bpy.ops.import_scene.fbx(filepath=str(source_path))
    imported_objects = tuple(
        obj for obj in bpy.data.objects if obj not in before_objects
    )
    try:
        mesh_objects = source_mesh_objects_for_export(
            imported_objects,
            mesh_object_names,
        )
        vertex_count = sum(len(obj.data.vertices) for obj in mesh_objects)
        if vertex_count != int(neutral_vertices.shape[0]):
            details = ", ".join(
                f"{obj.name}={len(obj.data.vertices)}" for obj in mesh_objects
            )
            raise ValueError(
                "Source appearance mesh vertex count does not match the retained model "
                f"topology: source={vertex_count}, neutral={neutral_vertices.shape[0]} "
                f"({details})."
            )
        has_uvs = any(len(obj.data.uv_layers) > 0 for obj in mesh_objects)
        has_materials = any(len(obj.material_slots) > 0 for obj in mesh_objects)
        if not has_uvs:
            print(
                "[WARN] Source appearance mesh has no UV layer; exporting geometry and "
                "blendshapes without texture coordinates.",
                flush=True,
            )
        if not has_materials:
            print(
                "[WARN] Source appearance mesh has no material; exporting geometry and "
                "blendshapes without source materials.",
                flush=True,
            )

        neutral_blender = glb_vertices_to_blender_axes(neutral_vertices).numpy()
        target_blender = {
            name: glb_vertices_to_blender_axes(vertices).numpy()
            for name, vertices in targets.items()
        }
        bpy.ops.object.select_all(action="DESELECT")
        vertex_offset = 0
        for obj in mesh_objects:
            count = len(obj.data.vertices)
            vertex_slice = slice(vertex_offset, vertex_offset + count)
            vertex_offset += count

            shape_keys = obj.data.shape_keys
            basis = (
                shape_keys.key_blocks.get("Basis")
                if shape_keys is not None
                else None
            )
            coordinate_source = basis.data if basis is not None else obj.data.vertices
            reference_coordinates = np.empty((count, 3), dtype=np.float64)
            coordinate_source.foreach_get("co", reference_coordinates.reshape(-1))

            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            if obj.data.shape_keys is not None:
                bpy.ops.object.shape_key_remove(all=True, apply_mix=False)
            obj.modifiers.clear()
            obj.parent = None
            obj.matrix_world = Matrix.Identity(4)
            obj.data.vertices.foreach_set(
                "co",
                np.ascontiguousarray(neutral_blender[vertex_slice]).reshape(-1),
            )
            obj.data.update(calc_edges=True)
            triangulate_mesh_in_reference_configuration(
                obj.data,
                reference_coordinates,
            )

            obj.shape_key_add(name="Basis", from_mix=False)
            for name, vertices in target_blender.items():
                key = obj.shape_key_add(name=name, from_mix=False)
                key.interpolation = "KEY_LINEAR"
                key.slider_min = 0.0
                key.slider_max = 1.0
                key.value = 0.0
                key.data.foreach_set(
                    "co",
                    np.ascontiguousarray(vertices[vertex_slice]).reshape(-1),
                )
            obj.data.update(calc_edges=True)

        if has_materials:
            prepare_source_materials_for_glb(mesh_objects)
        path.parent.mkdir(parents=True, exist_ok=True)
        result = bpy.ops.export_scene.gltf(
            filepath=str(temporary_path),
            export_format="GLB",
            use_selection=True,
            export_materials="EXPORT",
            export_texcoords=True,
            export_normals=True,
            export_morph=True,
            export_morph_normal=False,
            export_morph_tangent=False,
            export_animations=False,
        )
        if "FINISHED" not in result or not temporary_path.is_file():
            raise RuntimeError(f"Blender did not write {path}.")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
        for obj in imported_objects:
            if obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)


def export_fbx_color_segmented_gaze_batch(
    *,
    source_path: Path,
    vertices: torch.Tensor,
    outputs: Mapping[int, Path],
    motions: Mapping[int, Sequence[float]],
    landmark_payload: Mapping[str, Any],
    mesh_object_names: Optional[Sequence[str]] = None,
    texture_mask_cache_dir: Optional[Path] = None,
    texture_mask_debug_dir: Optional[Path] = None,
) -> dict[int, dict[str, Any]]:
    """Import one FBX once and export every requested procedural gaze AU."""

    import bpy
    import numpy as np

    source_path = Path(source_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    requested_action_units = tuple(sorted(int(value) for value in outputs))
    if not requested_action_units:
        raise ValueError("At least one gaze action unit must be requested.")
    if set(requested_action_units) != {int(value) for value in motions}:
        raise ValueError("Every batch gaze output must have exactly one motion vector.")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [V, 3].")
    for action_unit_id in requested_action_units:
        if eye_gaze_landmark_spec(action_unit_id) is None:
            raise ValueError(f"AU{action_unit_id} is not an eye-gaze action unit.")

    before_objects = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(source_path))
    imported_objects = tuple(
        obj for obj in bpy.context.scene.objects if obj not in before_objects
    )
    temporary_paths: list[Path] = []
    try:
        mesh_objects = source_mesh_objects_for_export(
            imported_objects,
            mesh_object_names,
        )
        vertex_count = sum(len(obj.data.vertices) for obj in mesh_objects)
        if vertex_count != int(vertices.shape[0]):
            raise ValueError(
                "Source FBX mesh vertex count does not match the retained topology: "
                f"source={vertex_count}, prediction={vertices.shape[0]}."
            )
        if not any(len(obj.data.uv_layers) > 0 for obj in mesh_objects):
            raise ValueError("Source FBX does not contain a UV layer.")

        bpy.ops.object.select_all(action="DESELECT")
        for obj in mesh_objects:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            if obj.data.shape_keys is not None:
                bpy.ops.object.shape_key_remove(all=True, apply_mix=False)
            obj.modifiers.clear()

        source_images = []
        for obj in mesh_objects:
            for slot in obj.material_slots:
                material = slot.material
                if material is None or not material.use_nodes or material.node_tree is None:
                    continue
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE" and node.image is not None:
                        if node.image not in source_images:
                            source_images.append(node.image)
        image_snapshots = {}
        for image in source_images:
            width, height = int(image.size[0]), int(image.size[1])
            if width < 2 or height < 2:
                continue
            pixels = np.empty(width * height * 4, dtype=np.float32)
            image.pixels.foreach_get(pixels)
            image_snapshots[image] = pixels
        if not image_snapshots:
            raise EyeGazeUVWarpValidationError(
                "Color-selected gaze requires a packed material texture."
            )

        def restore_images() -> None:
            for image, pixels in image_snapshots.items():
                image.pixels.foreach_set(pixels)
                image.update()

        prepare_source_materials_for_glb(mesh_objects)
        reports: dict[int, dict[str, Any]] = {}
        for action_unit_id in requested_action_units:
            restore_images()
            output_path = Path(outputs[action_unit_id]).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = output_path.with_name(
                f".{output_path.stem}.exporting{output_path.suffix}"
            )
            temporary_paths.append(temporary_path)
            temporary_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            spec = eye_gaze_landmark_spec(action_unit_id)
            try:
                layer_payloads = eye_gaze_layer_payloads(
                    landmark_payload,
                    action_unit_id,
                )[:1]
                layer_reports = []
                for layer_index, layer_payload in enumerate(layer_payloads):
                    layer_report = apply_fbx_eye_gaze_uv_warp(
                        mesh_objects=mesh_objects,
                        landmark_payload=layer_payload,
                        action_unit_id=action_unit_id,
                        motion=torch.as_tensor(motions[action_unit_id]).float(),
                        apply_uv_coordinates=False,
                        texture_seed_only=bool(
                            layer_payload.get("eye_gaze_geometry_seed_only", False)
                        ),
                        texture_shared_world_shift=(
                            (
                                layer_reports[0]["world_shift_x"],
                                layer_reports[0]["world_shift_y"],
                                layer_reports[0]["world_shift_z"],
                            )
                            if layer_index > 0
                            else None
                        ),
                        texture_shared_iris_radius=(
                            layer_reports[0]["iris_position_radius_3d"]
                            if layer_index > 0
                            else None
                        ),
                    )
                    layer_report["texture_cleanup_only"] = bool(
                        layer_payload.get("eye_gaze_texture_cleanup_only", False)
                    )
                    layer_reports.append(layer_report)
                texture_report = apply_fbx_eye_gaze_color_texture_warp(
                    mesh_objects=mesh_objects,
                    layer_reports=layer_reports,
                    eye_name=spec.eye,
                    mask_cache_dir=texture_mask_cache_dir,
                    mask_debug_dir=texture_mask_debug_dir,
                )
            except EyeGazeUVWarpValidationError as exc:
                reports[action_unit_id] = {
                    "status": "skipped",
                    "reason": str(exc),
                }
                continue

            bpy.ops.object.select_all(action="DESELECT")
            for obj in mesh_objects:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = mesh_objects[0]
            result = bpy.ops.export_scene.gltf(
                filepath=str(temporary_path),
                export_format="GLB",
                use_selection=True,
                export_apply=True,
                export_materials="EXPORT",
                export_texcoords=True,
                export_normals=True,
            )
            if "FINISHED" not in result or not temporary_path.is_file():
                raise RuntimeError(f"Blender did not write {output_path}.")
            temporary_path.replace(output_path)
            reports[action_unit_id] = {
                "status": "exported",
                "warp": {
                    "method": "color_segmented_texture",
                    "requested_method": "color_segmented_texture",
                    "layer_count": len(layer_reports),
                    "layers": layer_reports,
                    "texture": texture_report,
                },
            }
        restore_images()
        return reports
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        for obj in imported_objects:
            if obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)


def eye_gaze_layer_payloads(
    landmark_payload: Mapping[str, Any],
    action_unit_id: int,
) -> list[Mapping[str, Any]]:
    spec = eye_gaze_landmark_spec(action_unit_id)
    if spec is None:
        return [landmark_payload]
    raw_selection = landmark_payload.get("surface_anchor_selection")
    raw_anchors = landmark_payload.get("surface_anchors")
    if not isinstance(raw_selection, Mapping) or not isinstance(raw_anchors, Mapping):
        return [landmark_payload]
    eye_selection = raw_selection.get(spec.eye)
    if not isinstance(eye_selection, Mapping):
        return [landmark_payload]
    visible_layers = eye_selection.get("visible_layers")
    if not isinstance(visible_layers, Sequence) or isinstance(
        visible_layers, (str, bytes)
    ):
        return [landmark_payload]
    visible_layers = [
        layer for layer in visible_layers if isinstance(layer, Mapping)
    ]
    if visible_layers:
        first_depth_range = visible_layers[0].get("ray_depth_range")
        first_eye_width = visible_layers[0].get("eye_width_3d")
        if (
            isinstance(first_depth_range, Sequence)
            and len(first_depth_range) == 2
            and first_eye_width is not None
        ):
            front_depth = (
                float(first_depth_range[0]) + float(first_depth_range[1])
            ) * 0.5
            maximum_depth = front_depth + float(first_eye_width) * 0.60
            visible_layers = [
                layer for layer in visible_layers
                if not isinstance(layer.get("ray_depth_range"), Sequence)
                or len(layer["ray_depth_range"]) != 2
                or (
                    float(layer["ray_depth_range"][0])
                    + float(layer["ray_depth_range"][1])
                )
                * 0.5
                <= maximum_depth
            ]

    payloads: list[Mapping[str, Any]] = []
    for raw_layer in visible_layers:
        if not isinstance(raw_layer, Mapping):
            continue
        layer_anchors = raw_layer.get("surface_anchors")
        valid_iris_ids = raw_layer.get("valid_iris_ids")
        if not isinstance(layer_anchors, Mapping) or not isinstance(
            valid_iris_ids, Sequence
        ):
            continue
        merged_anchors = dict(raw_anchors)
        merged_anchors.update(layer_anchors)
        merged_eye_selection = {
            key: value
            for key, value in eye_selection.items()
            if key not in {"visible_layers", "surface_anchors"}
        }
        merged_eye_selection.update(
            {
                key: value
                for key, value in raw_layer.items()
                if key != "surface_anchors"
            }
        )
        merged_selection = dict(raw_selection)
        merged_selection[spec.eye] = merged_eye_selection
        layer_payload = dict(landmark_payload)
        layer_payload["surface_anchors"] = merged_anchors
        layer_payload["surface_anchor_selection"] = merged_selection
        payloads.append(layer_payload)
    return payloads or [landmark_payload]


def eye_gaze_geometry_layer_payloads(
    landmark_payload: Mapping[str, Any],
    action_unit_id: int,
) -> list[Mapping[str, Any]]:
    """Select the visible iris shell and its immediately occluded rear pair."""

    primary_payload = eye_gaze_layer_payloads(landmark_payload, action_unit_id)[0]
    spec = eye_gaze_landmark_spec(action_unit_id)
    if spec is None:
        return [primary_payload]
    raw_anchors = landmark_payload.get("surface_anchors")
    raw_selection = landmark_payload.get("surface_anchor_selection")
    raw_candidates = landmark_payload.get("iris_surface_anchor_candidates")
    if not all(
        isinstance(value, Mapping)
        for value in (raw_anchors, raw_selection, raw_candidates)
    ):
        return [primary_payload]
    eye_selection = raw_selection.get(spec.eye)
    if not isinstance(eye_selection, Mapping):
        return [primary_payload]
    eye_width = eye_selection.get("eye_width_3d")
    center_candidates = raw_candidates.get(str(spec.center_id))
    if eye_width is None or not isinstance(center_candidates, Sequence):
        return [primary_payload]

    primary_anchors = primary_payload.get("surface_anchors")
    if not isinstance(primary_anchors, Mapping):
        return [primary_payload]
    primary_center = primary_anchors.get(str(spec.center_id))
    if not isinstance(primary_center, Mapping):
        return [primary_payload]
    try:
        front_depth = float(primary_center["ray_depth"])
        eye_width = float(eye_width)
    except (KeyError, TypeError, ValueError):
        return [primary_payload]
    primary_polygon = int(primary_center.get("polygon_index", -1))
    maximum_depth = (
        front_depth + eye_width * MAX_PHYSICAL_EYE_LAYER_DEPTH_EYE_WIDTHS
    )
    maximum_visible_shell_depth = (
        front_depth + eye_width * MAX_VISIBLE_IRIS_SHELL_DEPTH_EYE_WIDTHS
    )

    selected_candidates = []
    for candidate in center_candidates:
        if not isinstance(candidate, Mapping):
            continue
        try:
            depth = float(candidate["ray_depth"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            front_depth - 1.0e-5 <= depth <= maximum_depth
        ):
            selected_candidates.append(candidate)
    selected_candidates.sort(key=lambda candidate: float(candidate["ray_depth"]))
    selected_candidates = selected_candidates[:MAX_PHYSICAL_EYE_LAYERS]

    payloads: list[Mapping[str, Any]] = [primary_payload]
    for candidate in selected_candidates:
        if int(candidate.get("polygon_index", -1)) == primary_polygon:
            continue
        merged_anchors = dict(raw_anchors)
        merged_anchors[str(spec.center_id)] = candidate
        layer_payload = dict(landmark_payload)
        layer_payload["surface_anchors"] = merged_anchors
        layer_payload["eye_gaze_geometry_seed_only"] = True
        if float(candidate["ray_depth"]) > maximum_visible_shell_depth:
            layer_payload["eye_gaze_texture_cleanup_only"] = True
        payloads.append(layer_payload)
    return payloads


def eye_gaze_texture_fallback_layer_payloads(
    landmark_payload: Mapping[str, Any],
    action_unit_id: int,
) -> list[Mapping[str, Any]]:
    """Select the visible iris plus only its immediately adjacent duplicates."""

    candidates = eye_gaze_geometry_layer_payloads(
        landmark_payload,
        action_unit_id,
    )
    if not candidates:
        return []
    payloads: list[Mapping[str, Any]] = [candidates[0]]
    for candidate in candidates[1:]:
        if candidate.get("eye_gaze_texture_cleanup_only", False):
            break
        duplicate = dict(candidate)
        duplicate["eye_gaze_texture_cleanup_only"] = True
        payloads.append(duplicate)
    return payloads


def eye_gaze_physical_layer_belongs_to_assembly(
    record: Mapping[str, Any],
) -> bool:
    """Accept a bounded ray-confirmed layer without requiring it to be spherical.

    Sphere and ellipsoid checks select the pivot. Iris discs, corneas, and rear
    duplicate layers can be non-spherical but must still follow that pivot.
    """

    return bool(
        not record.get("main_face_rejected", False)
        # Low-poly pupils and corneal caps can be extremely small components.
        # They still have to follow the eyeball when a center ray confirms that
        # they sit inside the bounded eye-layer depth.  Requiring 16 vertices
        # left a 13-vertex pupil stationary on otherwise valid assets.
        and int(record.get("vertex_count", 0)) >= 3
        and 0 < int(record.get("polygon_count", 0)) <= 20_000
    )


def eye_gaze_is_bounded_iris_side_layer(
    record: Mapping[str, Any],
    *,
    center_iris_id: int,
    visible_depth: float,
    visible_center: Sequence[float],
    eye_width: float,
) -> bool:
    """Accept a compact non-center iris layer without admitting the socket."""

    import numpy as np

    if int(record.get("source_iris_id", center_iris_id)) == int(center_iris_id):
        return False
    if not bool(record.get("plausible", False)):
        return False
    luminance = float(record.get("center_luminance", float("nan")))
    radius_ratio = float(record.get("radius_eye_width_ratio", 0.0))
    depth = float(record.get("ray_depth", float("nan")))
    center = np.asarray(record.get("sphere_center"), dtype=np.float64)
    carrier_center = np.asarray(visible_center, dtype=np.float64)
    return bool(
        np.isfinite(luminance)
        and 0.12 <= luminance <= 0.55
        and 0.40 <= radius_ratio <= 0.90
        and np.isfinite(depth)
        and 0.0 <= depth - float(visible_depth) <= float(eye_width) * 0.09
        and center.shape == (3,)
        and carrier_center.shape == (3,)
        and np.isfinite(center).all()
        and np.isfinite(carrier_center).all()
        and np.linalg.norm(center - carrier_center) <= float(eye_width) * 0.20
    )


def eye_gaze_is_visible_pivot_candidate(
    record: Mapping[str, Any],
    *,
    horizontal_iris_ids: set[int],
    front_depth_by_iris_id: Mapping[int, float],
    eye_width: float,
) -> bool:
    """Reject tiny ray fragments before refining the fitted eyeball pivot."""

    import numpy as np

    source_iris_id = int(record.get("source_iris_id", -1))
    record_depth = float(record.get("ray_depth", float("nan")))
    record_rgb = np.asarray(record.get("center_rgb"), dtype=np.float64)
    return bool(
        source_iris_id in horizontal_iris_ids
        and source_iris_id in front_depth_by_iris_id
        and bool(record.get("plausible", False))
        and int(record.get("vertex_count", 0)) >= 32
        and int(record.get("polygon_count", 0)) >= 8
        and float(record.get("closedness", 0.0)) >= 0.62
        and np.isfinite(record_depth)
        and record_depth
        <= float(front_depth_by_iris_id[source_iris_id])
        + float(eye_width) * MAX_VISIBLE_IRIS_SHELL_DEPTH_EYE_WIDTHS
        and record_rgb.shape == (3,)
        and np.isfinite(record_rgb).all()
        and float(np.max(record_rgb) - np.min(record_rgb)) <= 0.12
        and float(np.dot(record_rgb, (0.2126, 0.7152, 0.0722))) >= 0.55
        and 0.30
        <= float(record.get("radius_eye_width_ratio", 0.0))
        <= 0.90
        and float(record.get("sphere_residual_ratio", float("inf")))
        <= 0.10
    )


def eye_gaze_rigid_assembly_strategy_supported(
    assembly_strategy: str,
    valid_iris_ids: Sequence[int],
) -> bool:
    """Require an identified visible carrier before using rigid rotation."""

    return bool(
        len(valid_iris_ids) >= 3
        and assembly_strategy
        in {
            "visible_dark_carrier",
            "bright_spherical_carrier",
        }
    )


def eye_gaze_is_compact_spherical_eye_layer(
    record: Mapping[str, Any],
) -> bool:
    """Recognize a small independent pupil/iris shell that must follow the eye."""

    return bool(
        not record.get("main_face_rejected", False)
        and 12 <= int(record.get("vertex_count", 0)) <= 512
        and 8 <= int(record.get("polygon_count", 0)) <= 1_024
        and 0.25 <= float(record.get("radius_eye_width_ratio", 0.0)) <= 0.95
        and float(record.get("sphere_residual_ratio", float("inf"))) <= 0.08
        and float(record.get("closedness", 0.0)) >= 0.65
    )


def eye_gaze_is_foreground_non_eye_iris_record(
    record: Mapping[str, Any],
    *,
    center_iris_id: int,
    selected_front_depth: float,
    eye_width: float,
) -> bool:
    """Detect a stationary painted iris obstruction ahead of the eye assembly."""

    import numpy as np

    depth = float(record.get("ray_depth", float("nan")))
    luminance = float(record.get("center_luminance", float("nan")))
    return bool(
        int(record.get("source_iris_id", center_iris_id))
        == int(center_iris_id)
        and np.isfinite(depth)
        and depth <= float(selected_front_depth) + float(eye_width) * 0.03
        and np.isfinite(luminance)
        and luminance <= 0.12
        and not eye_gaze_is_compact_spherical_eye_layer(record)
        and (
            int(record.get("vertex_count", 0)) > 512
            or float(record.get("radius_eye_width_ratio", 0.0)) > 1.25
            or bool(record.get("main_face_rejected", False))
        )
    )


def eye_gaze_auto_rigid_visibility_supported(
    landmark_payload: Mapping[str, Any],
    action_unit_id: int,
) -> bool:
    """Return whether cached landmarks prove a front movable iris surface.

    A sphere-like component alone is insufficient: Pixel3D assets can contain
    concentric eye-colored shells behind a stationary painted iris.  Those
    shells rotate numerically but do not translate the visible iris boundary.
    Auto mode therefore requires a complete MediaPipe iris consensus on the
    foremost ray layer before it reports physical gaze.
    """

    spec = eye_gaze_landmark_spec(action_unit_id)
    if spec is None:
        return False
    raw_selection = landmark_payload.get("surface_anchor_selection")
    eye_selection = (
        raw_selection.get(spec.eye)
        if isinstance(raw_selection, Mapping)
        else None
    )
    if not isinstance(eye_selection, Mapping):
        return False
    if eye_selection.get("status") != "selected_visible_consensus":
        return False
    raw_ranks = eye_selection.get("depth_ranks")
    if not isinstance(raw_ranks, Sequence) or isinstance(raw_ranks, (str, bytes)):
        return False
    try:
        ranks = [int(value) for value in raw_ranks]
    except (TypeError, ValueError):
        return False
    return bool(len(ranks) >= 4 and ranks and max(ranks) == 0)


def eye_gaze_component_is_face_like(
    *,
    polygon_count: int,
    object_polygon_count: int,
    bbox_extent: Sequence[float],
    eye_width: float,
) -> bool:
    """Reject the face or eye-socket patch before scoring rigid-eye pivots."""

    import numpy as np

    extent = np.asarray(bbox_extent, dtype=np.float64)
    oversized_patch = bool(
        extent.shape == (3,)
        and np.isfinite(extent).all()
        and float(eye_width) > 1.0e-8
        and float(extent.max()) > float(eye_width) * 1.70
    )
    return bool(
        int(polygon_count) > 20_000
        or (
            int(object_polygon_count) > 10_000
            and int(polygon_count) > int(object_polygon_count) * 0.35
        )
        or oversized_patch
    )


def eye_gaze_physical_assembly_selection(
    mesh_objects: Sequence[Any],
    landmark_payload: Mapping[str, Any],
    action_unit_id: int,
) -> Optional[dict[str, Any]]:
    """Select independent eye layers and a sphere-like shared rotation pivot."""

    import os
    import numpy as np

    debug_selection = os.environ.get("TOPORIG_DEBUG_EYE_SELECTION") == "1"

    def debug(message: str) -> None:
        if debug_selection:
            print(f"eye-gaze physical selection: {message}", flush=True)

    spec = eye_gaze_landmark_spec(action_unit_id)
    if spec is None:
        return None
    cached_capabilities = landmark_payload.get("eye_geometry_capability")
    if (
        isinstance(cached_capabilities, Mapping)
        and cached_capabilities.get(spec.eye) == "texture_only"
    ):
        return None
    raw_selection = landmark_payload.get("surface_anchor_selection")
    eye_selection = (
        raw_selection.get(spec.eye)
        if isinstance(raw_selection, Mapping)
        else None
    )
    try:
        eye_width = float(eye_selection["eye_width_3d"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(eye_width) or eye_width <= 1.0e-8:
        return None

    objects_by_name: dict[str, Any] = {}
    for obj in mesh_objects:
        objects_by_name[obj.name] = obj
        objects_by_name.setdefault(base_blender_object_name(obj.name), obj)
    raw_anchors = landmark_payload.get("surface_anchors")
    raw_candidates = landmark_payload.get("iris_surface_anchor_candidates")
    physical_candidates: list[tuple[float, int, Mapping[str, Any]]] = []
    if isinstance(raw_candidates, Mapping):
        for iris_id in spec.iris_ids:
            iris_candidates = raw_candidates.get(str(iris_id))
            if not isinstance(iris_candidates, Sequence) or isinstance(
                iris_candidates, (str, bytes)
            ):
                continue
            for candidate in iris_candidates:
                if not isinstance(candidate, Mapping):
                    continue
                try:
                    depth = float(candidate["ray_depth"])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(depth):
                    physical_candidates.append((depth, int(iris_id), candidate))
    physical_candidates.sort(key=lambda value: value[0])
    center_depths = [
        depth
        for depth, iris_id, _candidate in physical_candidates
        if iris_id == spec.center_id
    ]
    if center_depths:
        front_depth = min(center_depths)
        maximum_depth = (
            front_depth + eye_width * MAX_PHYSICAL_EYE_LAYER_DEPTH_EYE_WIDTHS
        )
        physical_candidates = [
            value
            for value in physical_candidates
            if value[0] <= maximum_depth
        ][:MAX_PHYSICAL_EYE_LAYERS]
    else:
        physical_candidates = []

    raw_payloads: list[Mapping[str, Any]] = []
    if isinstance(raw_anchors, Mapping):
        for depth, source_iris_id, candidate in physical_candidates:
            merged_anchors = dict(raw_anchors)
            merged_anchors[str(spec.center_id)] = candidate
            payload = dict(landmark_payload)
            payload["surface_anchors"] = merged_anchors
            payload["eye_gaze_physical_ray_depth"] = depth
            payload["eye_gaze_physical_source_iris_id"] = source_iris_id
            payload["eye_gaze_physical_center_luminance"] = candidate.get(
                "base_color_luminance"
            )
            payload["eye_gaze_physical_center_rgb"] = candidate.get(
                "base_color_rgb"
            )
            raw_payloads.append(payload)
    if not raw_payloads:
        raw_payloads = eye_gaze_geometry_layer_payloads(
            landmark_payload,
            action_unit_id,
        )
    debug(
        f"eye={spec.eye} raw_payloads={len(raw_payloads)} "
        f"physical_candidates={len(physical_candidates)}"
    )
    records = []
    seen_components: set[tuple[str, int]] = set()
    # Full-resolution Pixel3D objects can contain hundreds of thousands of
    # polygons. Build their vertex-to-polygon index once and reuse it for all
    # nearby ray candidates instead of allocating that index per candidate.
    topology_indices: dict[int, dict[str, Any]] = {}

    def component_key_for_anchor(
        anchor: Mapping[str, Any],
    ) -> Optional[tuple[str, int]]:
        object_name = str(anchor.get("object_name", ""))
        obj = objects_by_name.get(
            object_name,
            objects_by_name.get(base_blender_object_name(object_name)),
        )
        if obj is None:
            return None
        try:
            polygon_index = int(anchor["polygon_index"])
        except (KeyError, TypeError, ValueError):
            return None
        if not 0 <= polygon_index < len(obj.data.polygons):
            return None
        topology_index = topology_indices.get(id(obj.data))
        if topology_index is None:
            topology_index = build_eye_gaze_topology_index(obj.data)
            topology_indices[id(obj.data)] = topology_index
        polygon_ids, _vertex_ids = eye_gaze_topology_component(
            obj.data,
            polygon_index,
            topology_index=topology_index,
        )
        return (str(obj.name), int(polygon_ids.min()))

    for payload_index, payload in enumerate(raw_payloads):
        anchors = payload.get("surface_anchors")
        center_anchor = (
            anchors.get(str(spec.center_id))
            if isinstance(anchors, Mapping)
            else None
        )
        if not isinstance(center_anchor, Mapping):
            continue
        object_name = str(center_anchor.get("object_name", ""))
        obj = objects_by_name.get(
            object_name,
            objects_by_name.get(base_blender_object_name(object_name)),
        )
        if obj is None:
            continue
        try:
            seed_polygon = int(center_anchor["polygon_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= seed_polygon < len(obj.data.polygons):
            continue
        topology_index = topology_indices.get(id(obj.data))
        if topology_index is None:
            topology_index = build_eye_gaze_topology_index(obj.data)
            topology_indices[id(obj.data)] = topology_index
        polygon_ids, vertex_ids = eye_gaze_topology_component(
            obj.data,
            seed_polygon,
            topology_index=topology_index,
        )
        component_key = (str(obj.name), int(polygon_ids.min()))
        if component_key in seen_components:
            continue
        seen_components.add(component_key)
        object_polygon_count = len(obj.data.polygons)
        coordinates = np.stack(
            [
                np.asarray(obj.data.vertices[int(vertex_id)].co, dtype=np.float64)
                for vertex_id in vertex_ids
            ]
        )
        triangles = np.asarray(
            [
                tuple(int(value) for value in obj.data.polygons[int(index)].vertices)
                for index in polygon_ids
                if len(obj.data.polygons[int(index)].vertices) == 3
            ],
            dtype=np.int64,
        )
        sphere_center, sphere_radius, sphere_residual = fit_sphere(coordinates)
        (
            _ellipsoid_center,
            _ellipsoid_axes,
            ellipsoid_residual,
            ellipsoid_axis_ratio,
        ) = fit_ellipsoid(coordinates)
        # Blender 4.2 may bundle NumPy 1.x, where ndarray.amin/amax are not
        # available even though the top-level functions are.
        bbox_center = (
            np.min(coordinates, axis=0) + np.max(coordinates, axis=0)
        ) * 0.5
        bbox_extent = np.max(coordinates, axis=0) - np.min(coordinates, axis=0)
        main_face_like = eye_gaze_component_is_face_like(
            polygon_count=len(polygon_ids),
            object_polygon_count=object_polygon_count,
            bbox_extent=bbox_extent,
            eye_width=eye_width,
        )
        try:
            ray_depth = float(center_anchor["ray_depth"])
        except (KeyError, TypeError, ValueError):
            ray_depth = float("nan")
        try:
            center_luminance = float(center_anchor["base_color_luminance"])
        except (KeyError, TypeError, ValueError):
            center_luminance = float("nan")
        try:
            center_rgb = np.asarray(
                payload.get("eye_gaze_physical_center_rgb"),
                dtype=np.float64,
            )
        except (TypeError, ValueError):
            center_rgb = np.full(3, np.nan, dtype=np.float64)
        if center_rgb.shape != (3,):
            center_rgb = np.full(3, np.nan, dtype=np.float64)
        anchor_position = np.sum(
            coordinates[:1] * 0.0,
            axis=0,
        )
        try:
            anchor_vertex_ids = np.asarray(
                center_anchor["vertex_ids"],
                dtype=np.int64,
            )
            object_offsets = landmark_payload.get("mesh_object_offsets", {})
            offset = int(
                object_offsets.get(
                    object_name,
                    object_offsets.get(base_blender_object_name(object_name), 0),
                )
            )
            local_ids = anchor_vertex_ids - offset
            weights = np.asarray(
                center_anchor["barycentric_weights"],
                dtype=np.float64,
            )
            anchor_position = np.sum(
                np.stack(
                    [
                        np.asarray(obj.data.vertices[int(value)].co, dtype=np.float64)
                        for value in local_ids
                    ]
                )
                * weights[:, None],
                axis=0,
            )
        except (KeyError, TypeError, ValueError, IndexError):
            anchor_position = bbox_center
        radius_ratio = sphere_radius / eye_width
        surface_residual = (
            abs(float(np.linalg.norm(anchor_position - sphere_center)) - sphere_radius)
            / max(sphere_radius, 1.0e-8)
            if np.isfinite(sphere_center).all() and np.isfinite(sphere_radius)
            else float("inf")
        )
        closedness = component_closedness(triangles)
        shape_score = min(
            sphere_residual / 0.32,
            ellipsoid_residual / 0.24
            + 0.2 * max(ellipsoid_axis_ratio - 1.0, 0.0),
        )
        score = (
            2.5 * shape_score
            + abs(float(np.log(max(radius_ratio, 1.0e-8) / 0.5)))
            + 1.5 * min(surface_residual, 4.0)
            + (1.0 - closedness)
        )
        plausible = bool(
            not main_face_like
            and len(vertex_ids) >= 64
            and len(polygon_ids) >= 8
            and 0.12 <= radius_ratio <= 1.25
            and surface_residual <= 0.80
            and (
                sphere_residual <= 0.32
                or (
                    ellipsoid_residual <= 0.24
                    and ellipsoid_axis_ratio <= 2.5
                )
            )
        )
        records.append(
            {
                "payload": payload,
                "payload_index": payload_index,
                "source_iris_id": int(
                    payload.get(
                        "eye_gaze_physical_source_iris_id", spec.center_id
                    )
                ),
                "component_key": component_key,
                "polygon_count": int(len(polygon_ids)),
                "vertex_count": int(len(vertex_ids)),
                "bbox_center": bbox_center,
                "anchor_position": anchor_position,
                "sphere_center": sphere_center,
                "sphere_radius": float(sphere_radius),
                "sphere_residual_ratio": float(sphere_residual),
                "ellipsoid_residual_ratio": float(ellipsoid_residual),
                "ellipsoid_axis_ratio": float(ellipsoid_axis_ratio),
                "closedness": float(closedness),
                "radius_eye_width_ratio": float(radius_ratio),
                "iris_surface_residual_ratio": float(surface_residual),
                "ray_depth": ray_depth,
                "center_luminance": center_luminance,
                "center_rgb": center_rgb.tolist(),
                "visible_iris_candidate": bool(
                    np.isfinite(center_luminance)
                    and center_luminance
                    <= MAX_PHYSICAL_EYE_LAYER_CENTER_LUMINANCE
                ),
                "main_face_rejected": main_face_like,
                "score": float(score),
                "plausible": plausible,
            }
        )
    plausible_records = [record for record in records if record["plausible"]]
    visible_iris_records = [
        record
        for record in records
        if record["visible_iris_candidate"]
        and eye_gaze_physical_layer_belongs_to_assembly(record)
    ]
    debug(
        f"records={len(records)} plausible={len(plausible_records)} "
        f"visible={len(visible_iris_records)}"
    )
    if debug_selection:
        for record in records:
            debug(
                f"record={record['component_key']} source={record['source_iris_id']} "
                f"depth={record['ray_depth']:.6f} "
                f"luminance={record['center_luminance']:.3f} "
                f"plausible={record['plausible']} "
                f"visible={record['visible_iris_candidate']}"
            )
    if not plausible_records or not visible_iris_records:
        return None
    finite_layer_depths = [
        float(record["ray_depth"])
        for record in records
        if np.isfinite(float(record["ray_depth"]))
    ]
    if not finite_layer_depths:
        return None
    front_layer_depth = min(finite_layer_depths)
    center_layer_depths = [
        float(record["ray_depth"])
        for record in records
        if int(record.get("source_iris_id", spec.center_id)) == spec.center_id
        and np.isfinite(float(record["ray_depth"]))
    ]
    if not center_layer_depths:
        return None
    front_center_depth = min(center_layer_depths)
    pivot_candidates = [
        record
        for record in plausible_records
        if (
            not np.isfinite(float(record["ray_depth"]))
            or front_layer_depth - 1.0e-5
            <= float(record["ray_depth"])
            <= front_layer_depth
            + eye_width * MAX_PHYSICAL_EYE_LAYER_DEPTH_EYE_WIDTHS
        )
    ]
    if not pivot_candidates:
        return None
    pivot = min(pivot_candidates, key=lambda record: record["score"])
    debug(
        f"initial_pivot={pivot['component_key']} score={pivot['score']:.4f}"
    )
    pivot_center = np.asarray(pivot["sphere_center"], dtype=np.float64)
    pivot_radius = float(pivot["sphere_radius"])

    assembly_records = [
        record
        for record in records
        if eye_gaze_physical_layer_belongs_to_assembly(record)
    ]
    front_depth_by_iris_id: dict[int, float] = {}
    # Depth ordering must include rejected face/socket hits.  They are never
    # moved, but omitting them made a hidden white rear shell look "foremost"
    # and incorrectly attach to the rigid eye assembly.
    for record in records:
        source_iris_id = int(record.get("source_iris_id", spec.center_id))
        depth = float(record["ray_depth"])
        if not np.isfinite(depth):
            continue
        front_depth_by_iris_id[source_iris_id] = min(
            depth,
            front_depth_by_iris_id.get(source_iris_id, float("inf")),
        )

    def movable_eye_layer(record: Mapping[str, Any]) -> bool:
        source_iris_id = int(record.get("source_iris_id", spec.center_id))
        record_depth = float(record.get("ray_depth", float("nan")))
        record_rgb = np.asarray(record.get("center_rgb"), dtype=np.float64)
        neutral_eye_surface = bool(
            record_rgb.shape == (3,)
            and np.isfinite(record_rgb).all()
            and float(np.max(record_rgb) - np.min(record_rgb)) <= 0.12
        )
        foremost_iris_surface = bool(
            np.isfinite(record_depth)
            and source_iris_id != int(spec.bottom_iris_id)
            and source_iris_id in front_depth_by_iris_id
            and record_depth
            <= front_depth_by_iris_id[source_iris_id]
            + eye_width * MAX_VISIBLE_IRIS_SHELL_DEPTH_EYE_WIDTHS
            and neutral_eye_surface
            and float(np.dot(record_rgb, (0.2126, 0.7152, 0.0722))) >= 0.55
            and bool(record.get("plausible", False))
        )
        spherical_eye_shell = bool(
            source_iris_id == spec.center_id
            and record.get("plausible", False)
            and 0.40 <= float(record.get("radius_eye_width_ratio", 0.0)) <= 1.25
            and float(record.get("sphere_residual_ratio", float("inf"))) <= 0.08
        )
        return foremost_iris_surface or spherical_eye_shell

    linked = [
        record
        for record in records
        if eye_gaze_physical_layer_belongs_to_assembly(record)
        and (
            not np.isfinite(float(record["ray_depth"]))
            or front_layer_depth - 1.0e-5
            <= float(record["ray_depth"])
            <= front_layer_depth
            + eye_width * MAX_PHYSICAL_EYE_LAYER_DEPTH_EYE_WIDTHS
        )
    ]
    visible_linked = [
        record for record in visible_iris_records if record in linked
    ]
    if not visible_linked:
        # Hidden spherical shells do not make gaze physical when every dark
        # visible center sample belongs to the face or another unlinked layer.
        return None
    debug(
        f"linked={len(linked)} visible_linked={len(visible_linked)}"
    )
    linked.sort(
        key=lambda record: (
            float(record["ray_depth"])
            if np.isfinite(float(record["ray_depth"]))
            else float("inf")
        )
    )
    center_linked = [
        record
        for record in linked
        if int(record.get("source_iris_id", spec.center_id)) == spec.center_id
    ]
    if not center_linked:
        return None
    front_record = min(
        center_linked, key=lambda record: float(record["ray_depth"])
    )

    def matched_iris_payload(
        record: Mapping[str, Any],
    ) -> tuple[Optional[dict[str, Any]], list[int]]:
        """Build coherent iris anchors on one connected component."""

        record_depth = float(record["ray_depth"])
        matched_anchors = (
            dict(raw_anchors) if isinstance(raw_anchors, Mapping) else {}
        )
        valid_iris_ids = []
        for iris_id in spec.iris_ids:
            candidate_values = []
            if isinstance(raw_candidates, Mapping):
                values = raw_candidates.get(str(iris_id))
                if isinstance(values, Sequence) and not isinstance(
                    values, (str, bytes)
                ):
                    candidate_values.extend(
                        value for value in values if isinstance(value, Mapping)
                    )
            if isinstance(raw_anchors, Mapping):
                default_anchor = raw_anchors.get(str(iris_id))
                if isinstance(default_anchor, Mapping):
                    candidate_values.append(default_anchor)
            matches = [
                value
                for value in candidate_values
                if component_key_for_anchor(value) == record["component_key"]
            ]
            if not matches:
                continue
            matches.sort(
                key=lambda value: abs(
                    float(value.get("ray_depth", record_depth)) - record_depth
                )
            )
            matched_anchors[str(iris_id)] = matches[0]
            valid_iris_ids.append(int(iris_id))
        horizontal_edge_ids = {int(spec.iris_ids[1]), int(spec.iris_ids[3])}
        vertical_edge_ids = {int(spec.iris_ids[2]), int(spec.iris_ids[4])}
        if (
            spec.center_id not in valid_iris_ids
            or not horizontal_edge_ids.intersection(valid_iris_ids)
            or not vertical_edge_ids.intersection(valid_iris_ids)
        ):
            return None, []
        matched_selection = (
            dict(raw_selection) if isinstance(raw_selection, Mapping) else {}
        )
        matched_eye_selection = dict(eye_selection)
        matched_eye_selection["valid_iris_ids"] = valid_iris_ids
        matched_eye_selection.pop("visible_layers", None)
        matched_selection[spec.eye] = matched_eye_selection
        payload = dict(landmark_payload)
        payload["surface_anchors"] = matched_anchors
        payload["surface_anchor_selection"] = matched_selection
        return payload, valid_iris_ids

    driving_record = None
    driving_payload = None
    driving_valid_iris_ids: list[int] = []
    for record in visible_linked:
        payload, valid_iris_ids = matched_iris_payload(record)
        debug(
            f"driving_candidate={record['component_key']} "
            f"valid_iris_ids={valid_iris_ids}"
        )
        if payload is None:
            continue
        driving_record = record
        driving_payload = payload
        driving_valid_iris_ids = valid_iris_ids
        break
    if (
        driving_record is None
        or driving_payload is None
    ):
        return None
    if driving_valid_iris_ids:
        horizontal_iris_ids = {
            int(spec.iris_ids[1]),
            int(spec.iris_ids[3]),
        }
        visible_pivot_candidates = []
        for record in linked:
            if eye_gaze_is_visible_pivot_candidate(
                record,
                horizontal_iris_ids=horizontal_iris_ids,
                front_depth_by_iris_id=front_depth_by_iris_id,
                eye_width=eye_width,
            ):
                visible_pivot_candidates.append(record)
        if visible_pivot_candidates:
            pivot = min(
                visible_pivot_candidates,
                key=lambda record: (
                    float(record["ray_depth"]),
                    float(record["score"]),
                ),
            )
            pivot_center = np.asarray(pivot["sphere_center"], dtype=np.float64)
            pivot_radius = float(pivot["sphere_radius"])
    # Layered Pixel3D files often contain several concentric pupil/iris shells.
    # Rotating all of them exposes the hidden copies as crescents.  Prefer the
    # foremost complete, very-dark sphere hit by the centre iris ray: it is the
    # independently rotating visible carrier.  Keep the separately fitted
    # bright sphere as the pivot, but do not rotate that hidden shell.
    dark_center_carriers = [
        record
        for record in linked
        if (
            int(record.get("source_iris_id", spec.center_id)) == spec.center_id
            and bool(record.get("plausible", False))
            and np.isfinite(float(record.get("center_luminance", float("nan"))))
            and float(record["center_luminance"]) <= 0.12
            and 0.30
            <= float(record.get("radius_eye_width_ratio", 0.0))
            <= 0.90
            and float(record.get("sphere_residual_ratio", float("inf")))
            <= 0.06
            and float(record.get("closedness", 0.0)) >= 0.70
        )
    ]
    dark_center_carriers.sort(
        key=lambda record: (
            float(record["ray_depth"])
            if np.isfinite(float(record["ray_depth"]))
            else float("inf"),
            float(record["score"]),
        )
    )
    visible_carrier = dark_center_carriers[0] if dark_center_carriers else None
    selected_carrier = None
    iris_side_layers: list[Mapping[str, Any]] = []
    occluded_duplicate_components: list[tuple[str, int]] = []
    ambiguous_occluded_duplicate = False
    if visible_carrier is not None:
        selected_carrier = visible_carrier
        visible_radius = float(visible_carrier["sphere_radius"])
        visible_depth = float(visible_carrier["ray_depth"])
        visible_center = np.asarray(
            visible_carrier["sphere_center"], dtype=np.float64
        )
        visible_polygon_count = max(
            int(visible_carrier.get("polygon_count", 0)), 1
        )
        occluded_duplicate_records = [
            record
            for record in dark_center_carriers[1:]
            if (
                str(record["component_key"][0])
                == str(visible_carrier["component_key"][0])
                and np.isfinite(float(record["ray_depth"]))
                and 0.0
                < float(record["ray_depth"]) - visible_depth
                <= eye_width * 0.04
                and abs(float(record["sphere_radius"]) - visible_radius)
                <= visible_radius * 0.12
                and np.linalg.norm(
                    np.asarray(record["sphere_center"], dtype=np.float64)
                    - visible_center
                )
                <= eye_width * 0.06
                and 0.70
                <= int(record.get("polygon_count", 0))
                / visible_polygon_count
                <= 1.40
            )
        ]
        occluded_duplicate_components = [
            record["component_key"] for record in occluded_duplicate_records
        ]
        ambiguous_occluded_duplicate = any(
            int(record.get("polygon_count", 0)) <= visible_polygon_count
            for record in occluded_duplicate_records
        )
        assembly_strategy = "visible_dark_carrier"
        ordered_records = [driving_record]
        if visible_carrier is not driving_record:
            ordered_records.append(visible_carrier)
        iris_side_layers = [
            record
            for record in linked
            if (
                record not in ordered_records
                and eye_gaze_is_bounded_iris_side_layer(
                    record,
                    center_iris_id=spec.center_id,
                    visible_depth=visible_depth,
                    visible_center=visible_center,
                    eye_width=eye_width,
                )
            )
        ]
        iris_side_layers.sort(key=lambda record: float(record["ray_depth"]))
        ordered_records.extend(iris_side_layers)
    elif len(driving_valid_iris_ids) >= 4:
        # Low-poly assets can place the visible iris texture on a small bright
        # sphere hit by a horizontal iris ray.  The nearest such fitted shell
        # moves cleanly; rotating the dark foreground cap exposes its white
        # backing as a crescent.
        bright_spherical_carriers = [
            record
            for record in linked
            if (
                np.isfinite(
                    float(record.get("center_luminance", float("nan")))
                )
                and float(record["center_luminance"]) >= 0.90
                and 0.30
                <= float(record.get("radius_eye_width_ratio", 0.0))
                <= 0.90
                and float(record.get("sphere_residual_ratio", float("inf")))
                <= 0.04
                and float(record.get("closedness", 0.0)) >= 0.62
                and int(record.get("vertex_count", 0)) >= 32
            )
        ]
        bright_spherical_carriers.sort(
            key=lambda record: (
                float(record["ray_depth"])
                if np.isfinite(float(record["ray_depth"]))
                else float("inf"),
                float(record["score"]),
            )
        )
        bright_carrier = (
            bright_spherical_carriers[0]
            if bright_spherical_carriers
            else pivot
        )
        selected_carrier = bright_carrier
        assembly_strategy = "bright_spherical_carrier"
        ordered_records = [driving_record]
        if bright_carrier is not driving_record:
            ordered_records.append(bright_carrier)
    else:
        # With only a partial ray consensus, retain the small bounded set of
        # ray-confirmed layers.  This is needed for split iris/cornea assets.
        assembly_strategy = "partial_consensus_assembly"
        ordered_records = [driving_record] + [
            record
            for record in linked
            if record is not driving_record and movable_eye_layer(record)
        ]
        if pivot not in ordered_records:
            ordered_records.append(pivot)
    if len(driving_valid_iris_ids) < 3:
        for record in visible_linked:
            if record not in ordered_records:
                ordered_records.append(record)
        for record in linked:
            source_iris_id = int(record.get("source_iris_id", spec.center_id))
            record_depth = float(record.get("ray_depth", float("nan")))
            record_rgb = np.asarray(record.get("center_rgb"), dtype=np.float64)
            foreground_neutral_cap = bool(
                np.isfinite(record_depth)
                and source_iris_id in front_depth_by_iris_id
                and record_depth
                <= front_depth_by_iris_id[source_iris_id]
                + eye_width * MAX_VISIBLE_IRIS_SHELL_DEPTH_EYE_WIDTHS
                and record_rgb.shape == (3,)
                and np.isfinite(record_rgb).all()
                and float(np.max(record_rgb) - np.min(record_rgb)) <= 0.12
            )
            if foreground_neutral_cap and record not in ordered_records:
                ordered_records.append(record)

    selected_front_depth = min(
        float(record["ray_depth"])
        for record in ordered_records
        if np.isfinite(float(record.get("ray_depth", float("nan"))))
    )
    foreground_non_eye_iris_records = [
        record
        for record in records
        if (
            record not in ordered_records
            and eye_gaze_is_foreground_non_eye_iris_record(
                record,
                center_iris_id=spec.center_id,
                selected_front_depth=selected_front_depth,
                eye_width=eye_width,
            )
        )
    ]

    nearby_layer_payloads: list[Mapping[str, Any]] = []
    nearby_component_keys: list[tuple[str, int]] = []
    nearby_component_diagnostics: list[Mapping[str, Any]] = []
    nearby_component_cache_hit = False

    def nearby_component_payload(
        obj: Any,
        polygon_ids: Any,
    ) -> Optional[Mapping[str, Any]]:
        seed_polygon = obj.data.polygons[int(polygon_ids[0])]
        seed_vertex_ids = [int(value) for value in seed_polygon.vertices[:3]]
        if len(seed_vertex_ids) != 3:
            return None
        raw_offsets = landmark_payload.get("mesh_object_offsets", {})
        offset_value = raw_offsets.get(obj.name)
        if offset_value is None:
            base_name = base_blender_object_name(obj.name)
            matching_offsets = [
                value
                for name, value in raw_offsets.items()
                if base_blender_object_name(str(name)) == base_name
            ]
            if len(matching_offsets) != 1:
                return None
            offset_value = matching_offsets[0]
        offset = int(offset_value)
        source_anchors = driving_payload.get("surface_anchors")
        if not isinstance(source_anchors, Mapping):
            return None
        source_center_anchor = source_anchors.get(str(spec.center_id))
        if not isinstance(source_center_anchor, Mapping):
            return None
        seed_anchor = dict(source_center_anchor)
        seed_anchor.update(
            {
                "object_name": str(obj.name),
                "polygon_index": int(polygon_ids[0]),
                "vertex_ids": [int(value + offset) for value in seed_vertex_ids],
                "barycentric_weights": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            }
        )
        nearby_anchors = dict(source_anchors)
        nearby_anchors[str(spec.center_id)] = seed_anchor
        nearby_payload = dict(driving_payload)
        nearby_payload["surface_anchors"] = nearby_anchors
        nearby_payload["eye_gaze_geometry_seed_only"] = True
        return nearby_payload

    if driving_valid_iris_ids:
        # Iris rays can miss narrow wedges between their five sample points.
        # Enumerate the remaining connected components once and attach bounded
        # pieces that lie on the selected eyeball shell.  Components already
        # hit by a ray retain the stricter colour/face classification above.
        known_component_keys = {
            tuple(record["component_key"]) for record in records
        }
        assembly_cache_path = None
        assembly_cache = {}
        raw_landmark_cache_path = landmark_payload.get("_landmark_cache_path")
        if raw_landmark_cache_path:
            landmark_cache_path = Path(str(raw_landmark_cache_path))
            assembly_cache_path = landmark_cache_path.with_name(
                landmark_cache_path.stem + "_eye_assembly_v7.json"
            )
            if assembly_cache_path.is_file():
                try:
                    loaded_cache = json.loads(
                        assembly_cache_path.read_text(encoding="utf-8")
                    )
                    if loaded_cache.get("version") == 7:
                        assembly_cache = loaded_cache
                except (OSError, ValueError, TypeError):
                    assembly_cache = {}
        cached_eye = assembly_cache.get("eyes", {}).get(spec.eye)
        cached_component_keys = (
            cached_eye.get("nearby_components")
            if isinstance(cached_eye, Mapping)
            else None
        )
        cached_component_diagnostics = (
            cached_eye.get("nearby_component_diagnostics")
            if isinstance(cached_eye, Mapping)
            else None
        )
        if isinstance(cached_component_diagnostics, Sequence) and not isinstance(
            cached_component_diagnostics, (str, bytes)
        ):
            nearby_component_diagnostics.extend(
                value
                for value in cached_component_diagnostics
                if isinstance(value, Mapping)
            )
        if isinstance(cached_component_keys, Sequence) and not isinstance(
            cached_component_keys, (str, bytes)
        ):
            for raw_component_key in cached_component_keys:
                if (
                    not isinstance(raw_component_key, Sequence)
                    or len(raw_component_key) != 2
                ):
                    continue
                object_name, seed_polygon = raw_component_key
                obj = objects_by_name.get(
                    str(object_name),
                    objects_by_name.get(base_blender_object_name(str(object_name))),
                )
                if obj is None:
                    continue
                topology_index = topology_indices.get(id(obj.data))
                if topology_index is None:
                    topology_index = build_eye_gaze_topology_index(obj.data)
                    topology_indices[id(obj.data)] = topology_index
                try:
                    polygon_ids, _vertex_ids = eye_gaze_topology_component(
                        obj.data,
                        int(seed_polygon),
                        topology_index=topology_index,
                    )
                except (EyeGazeUVWarpValidationError, ValueError):
                    continue
                payload = nearby_component_payload(obj, polygon_ids)
                if payload is None:
                    continue
                component_key = (str(obj.name), int(polygon_ids.min()))
                nearby_layer_payloads.append(payload)
                nearby_component_keys.append(component_key)
            nearby_component_cache_hit = True
        pivot_axis = np.asarray(
            front_record["anchor_position"], dtype=np.float64
        ) - pivot_center
        pivot_axis_length = float(np.linalg.norm(pivot_axis))
        if pivot_axis_length > 1.0e-8 and not nearby_component_cache_hit:
            pivot_axis /= pivot_axis_length
            image_pixel_cache: dict[int, Any] = {}

            def component_base_color(
                obj: Any, polygon_ids: Any
            ) -> Any:
                samples = []
                uv_layer = obj.data.uv_layers.active
                if uv_layer is None:
                    return np.full(3, np.nan, dtype=np.float64)
                stride = max(1, int(len(polygon_ids) // 12))
                for polygon_id in polygon_ids[::stride][:12]:
                    polygon = obj.data.polygons[int(polygon_id)]
                    material = (
                        obj.material_slots[int(polygon.material_index)].material
                        if 0 <= int(polygon.material_index) < len(obj.material_slots)
                        else None
                    )
                    if material is None or not material.use_nodes:
                        continue
                    principled = next(
                        (
                            node
                            for node in material.node_tree.nodes
                            if node.type == "BSDF_PRINCIPLED"
                        ),
                        None,
                    )
                    if principled is None:
                        continue
                    base_color = principled.inputs.get("Base Color")
                    image = None
                    if base_color is not None:
                        for link in base_color.links:
                            if (
                                link.from_node.type == "TEX_IMAGE"
                                and link.from_node.image is not None
                            ):
                                image = link.from_node.image
                                break
                    if image is None:
                        if base_color is not None:
                            samples.append(
                                np.asarray(
                                    base_color.default_value[:3],
                                    dtype=np.float64,
                                )
                            )
                        continue
                    width, height = int(image.size[0]), int(image.size[1])
                    if width < 2 or height < 2:
                        continue
                    cache_key = int(image.as_pointer())
                    pixels = image_pixel_cache.get(cache_key)
                    if pixels is None:
                        flat_pixels = np.empty(width * height * 4, dtype=np.float32)
                        image.pixels.foreach_get(flat_pixels)
                        pixels = flat_pixels.reshape(height, width, 4)
                        image_pixel_cache[cache_key] = pixels
                    loop_ids = [int(value) for value in polygon.loop_indices]
                    if not loop_ids:
                        continue
                    uv = np.mean(
                        np.stack(
                            [
                                np.asarray(
                                    uv_layer.data[loop_id].uv,
                                    dtype=np.float64,
                                )
                                for loop_id in loop_ids
                            ]
                        ),
                        axis=0,
                    )
                    x = max(0, min(width - 1, int((float(uv[0]) % 1.0) * width)))
                    y = max(0, min(height - 1, int((float(uv[1]) % 1.0) * height)))
                    samples.append(np.asarray(pixels[y, x, :3], dtype=np.float64))
                return (
                    np.median(np.stack(samples), axis=0)
                    if samples
                    else np.full(3, np.nan, dtype=np.float64)
                )

            for obj in mesh_objects:
                topology_index = topology_indices.get(id(obj.data))
                if topology_index is None:
                    topology_index = build_eye_gaze_topology_index(obj.data)
                    topology_indices[id(obj.data)] = topology_index
                component_labels = topology_index["component_labels"]
                for polygon_index in range(len(obj.data.polygons)):
                    if int(component_labels[polygon_index]) >= 0:
                        continue
                    polygon_ids, vertex_ids = eye_gaze_topology_component(
                        obj.data,
                        polygon_index,
                        topology_index=topology_index,
                    )
                    component_key = (str(obj.name), int(polygon_ids.min()))
                    if component_key in known_component_keys:
                        continue
                    if not 12 <= len(vertex_ids) or not 8 <= len(polygon_ids) <= 20_000:
                        continue
                    coordinates = np.stack(
                        [
                            np.asarray(
                                obj.data.vertices[int(vertex_id)].co,
                                dtype=np.float64,
                            )
                            for vertex_id in vertex_ids
                        ]
                    )
                    bbox_extent = (
                        np.max(coordinates, axis=0)
                        - np.min(coordinates, axis=0)
                    )
                    if eye_gaze_component_is_face_like(
                        polygon_count=len(polygon_ids),
                        object_polygon_count=len(obj.data.polygons),
                        bbox_extent=bbox_extent,
                        eye_width=eye_width,
                    ):
                        continue
                    deltas = coordinates - pivot_center[None, :]
                    radial_distances = np.linalg.norm(deltas, axis=1)
                    tangent_deltas = (
                        deltas
                        - (deltas @ pivot_axis)[:, None] * pivot_axis[None, :]
                    )
                    tangent_distances = np.linalg.norm(tangent_deltas, axis=1)
                    shared_shell_residual = float(
                        np.median(np.abs(radial_distances - pivot_radius))
                        / max(pivot_radius, 1.0e-8)
                    )
                    component_center = coordinates.mean(axis=0)
                    center_delta = component_center - pivot_center
                    center_tangent = center_delta - np.dot(
                        center_delta, pivot_axis
                    ) * pivot_axis
                    center_axial = float(np.dot(center_delta, pivot_axis))
                    sampled_rgb = component_base_color(obj, polygon_ids)
                    sampled_luminance = float(
                        np.dot(sampled_rgb, (0.2126, 0.7152, 0.0722))
                    )
                    chroma = float(np.max(sampled_rgb) - np.min(sampled_rgb))
                    nearby_sphere_center, nearby_radius, nearby_sphere_residual = (
                        fit_sphere(coordinates)
                    )
                    nearby_radius_ratio = float(nearby_radius / eye_width)
                    triangles = np.asarray(
                        [
                            tuple(
                                int(value)
                                for value in obj.data.polygons[
                                    int(polygon_id)
                                ].vertices
                            )
                            for polygon_id in polygon_ids
                            if len(obj.data.polygons[int(polygon_id)].vertices) == 3
                        ],
                        dtype=np.int64,
                    )
                    nearby_closedness = component_closedness(triangles)
                    if (
                        float(np.linalg.norm(center_delta))
                        > eye_width * 1.25
                        or float(np.linalg.norm(center_tangent))
                        > eye_width * 0.36
                        or float(np.min(tangent_distances))
                        > eye_width * 0.32
                        or float(np.max(bbox_extent)) > eye_width * 0.85
                        or center_axial < pivot_radius * 0.45
                        or center_axial > pivot_radius * 1.40
                        or shared_shell_residual > 0.16
                        or not 0.08 <= nearby_radius_ratio <= 0.90
                        or nearby_sphere_residual > 0.18
                        or nearby_closedness < 0.62
                        or (
                            np.isfinite(sampled_rgb).all()
                            and chroma > 0.18
                            and sampled_luminance > 0.38
                        )
                    ):
                        continue
                    nearby_payload = nearby_component_payload(obj, polygon_ids)
                    if nearby_payload is None:
                        continue
                    nearby_layer_payloads.append(nearby_payload)
                    nearby_component_keys.append(component_key)
                    nearby_component_diagnostics.append(
                        {
                            "component_key": [
                                str(component_key[0]),
                                int(component_key[1]),
                            ],
                            "polygon_count": int(len(polygon_ids)),
                            "vertex_count": int(len(vertex_ids)),
                            "bbox_eye_width_ratio": float(
                                np.max(bbox_extent) / eye_width
                            ),
                            "center_distance_eye_width_ratio": float(
                                np.linalg.norm(center_delta) / eye_width
                            ),
                            "center_tangent_eye_width_ratio": float(
                                np.linalg.norm(center_tangent) / eye_width
                            ),
                            "center_axial_pivot_radius_ratio": float(
                                center_axial / max(pivot_radius, 1.0e-8)
                            ),
                            "shared_shell_residual_ratio": shared_shell_residual,
                            "sphere_radius_eye_width_ratio": nearby_radius_ratio,
                            "sphere_residual_ratio": float(
                                nearby_sphere_residual
                            ),
                            "closedness": float(nearby_closedness),
                            "center_rgb": sampled_rgb.tolist(),
                            "center_luminance": sampled_luminance,
                        }
                    )
                    known_component_keys.add(component_key)
            if assembly_cache_path is not None:
                eyes = dict(assembly_cache.get("eyes", {}))
                eyes[spec.eye] = {
                    "nearby_components": [
                        [str(name), int(seed)]
                        for name, seed in nearby_component_keys
                    ],
                    "nearby_component_diagnostics": (
                        nearby_component_diagnostics
                    ),
                    "evaluated_component_count": int(
                        sum(len(value["components"]) for value in topology_indices.values())
                    ),
                }
                cache_payload = {"version": 7, "eyes": eyes}
                try:
                    assembly_cache_path.write_text(
                        json.dumps(cache_payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                except OSError:
                    pass
    calibration_only_record = None
    driving_transform_payload = dict(driving_payload)
    if (
        ordered_records[0]
        is not (
            selected_carrier if selected_carrier is not None else driving_record
        )
        and not eye_gaze_is_compact_spherical_eye_layer(driving_record)
    ):
        calibration_only_record = driving_record
        driving_transform_payload["eye_gaze_geometry_calibration_only"] = True
    layer_payloads = [driving_transform_payload]
    for record in ordered_records[1:]:
        follower = dict(record["payload"])
        follower["eye_gaze_geometry_seed_only"] = True
        layer_payloads.append(follower)
    if assembly_strategy == "partial_consensus_assembly":
        layer_payloads.extend(nearby_layer_payloads)
    local_translation_records = list(visible_linked)
    if len(driving_valid_iris_ids) < 3:
        # A dark-only ray consensus can point at an occluded pupil shell.  Add
        # the exact foremost neutral caps from the other iris rays so the
        # visible front iris moves too.  The upper iris ray is excluded because
        # it commonly lands on the upper eyelid on layered Pixel3D assets.
        for record in linked:
            source_iris_id = int(record.get("source_iris_id", spec.center_id))
            record_depth = float(record.get("ray_depth", float("nan")))
            record_rgb = np.asarray(record.get("center_rgb"), dtype=np.float64)
            foreground_neutral_cap = bool(
                np.isfinite(record_depth)
                and source_iris_id in front_depth_by_iris_id
                and record_depth
                <= front_depth_by_iris_id[source_iris_id]
                + eye_width * MAX_VISIBLE_IRIS_SHELL_DEPTH_EYE_WIDTHS
                and record_rgb.shape == (3,)
                and np.isfinite(record_rgb).all()
                and float(np.max(record_rgb) - np.min(record_rgb)) <= 0.12
            )
            if foreground_neutral_cap and record not in local_translation_records:
                local_translation_records.append(record)
    ordered_local_records = [driving_record] + [
        record
        for record in local_translation_records
        if record is not driving_record
    ]
    local_translation_layer_payloads = [driving_payload]
    for record in ordered_local_records[1:]:
        follower = dict(record["payload"])
        follower["eye_gaze_geometry_seed_only"] = True
        local_translation_layer_payloads.append(follower)
    local_translation_layer_payloads.extend(nearby_layer_payloads)
    # Texture fallback evaluates the same complete bounded eye assembly.  The
    # coherent dark layer supplies the calibrated shift; each remaining layer
    # uses its local triangle Jacobian.  Moving every successfully segmented
    # iris island prevents an opaque duplicate from leaving a neutral iris on
    # top of a correctly moved rear layer.
    texture_layer_payloads = [
        *local_translation_layer_payloads,
        dict(landmark_payload),
    ]
    confidence = float(np.clip(np.exp(-float(pivot["score"]) / 7.5), 0.0, 1.0))
    # The component search above has already re-evaluated every ray layer and
    # rejected the face.  A complete dark-iris consensus may therefore be
    # rotated around the independently fitted sphere pivot even when the old
    # landmark cache selected a deeper rank, or when iris and sclera are
    # separate connected components.
    rigid_recommended = eye_gaze_rigid_assembly_strategy_supported(
        assembly_strategy,
        driving_valid_iris_ids,
    )
    recommended_method = (
        "rigid_rotation" if rigid_recommended else "local_translation"
    )
    auto_rigid_recommended = bool(
        rigid_recommended
        and eye_gaze_auto_rigid_visibility_supported(
            landmark_payload, action_unit_id
        )
        and driving_record["component_key"] == pivot["component_key"]
    )
    auto_recommended_method = (
        "rigid_rotation" if auto_rigid_recommended else "texture_fallback"
    )
    visible_geometry_proven = assembly_strategy == "visible_dark_carrier"
    physical_capability = (
        "independent_visible_eye_geometry"
        if visible_geometry_proven
        else (
            "independent_eye_geometry_visibility_unproven"
            if assembly_strategy == "bright_spherical_carrier"
            else "local_translation_only"
        )
    )
    return {
        "layer_payloads": layer_payloads,
        "local_translation_layer_payloads": local_translation_layer_payloads,
        "texture_layer_payloads": texture_layer_payloads,
        "rotation_center": pivot_center.tolist(),
        "rotation_radius": pivot_radius,
        "rotation_iris_position": np.asarray(
            front_record["anchor_position"], dtype=np.float64
        ).tolist(),
        "front_layer_component": front_record["component_key"],
        "pivot_component": pivot["component_key"],
        "visible_iris_component": driving_record["component_key"],
        "visible_carrier_component": (
            selected_carrier["component_key"]
            if selected_carrier is not None
            else None
        ),
        "assembly_strategy": assembly_strategy,
        "occluded_duplicate_components": occluded_duplicate_components,
        "occluded_duplicate_policy": "excluded_from_rigid_assembly",
        "iris_side_layer_components": [
            record["component_key"] for record in iris_side_layers
        ],
        "foreground_non_eye_iris_components": [
            record["component_key"]
            for record in foreground_non_eye_iris_records
        ],
        "rigid_assembly_components": [
            record["component_key"]
            for record in ordered_records
            if record is not calibration_only_record
        ]
        + (
            nearby_component_keys
            if assembly_strategy == "partial_consensus_assembly"
            else []
        ),
        "ambiguous_occluded_duplicate": ambiguous_occluded_duplicate,
        "calibration_only_component": (
            calibration_only_record["component_key"]
            if calibration_only_record is not None
            else None
        ),
        "visible_iris_ray_depth": float(driving_record["ray_depth"]),
        "front_layer_ray_depth": float(front_record["ray_depth"]),
        "visible_iris_landmark_ids": driving_valid_iris_ids,
        "nearby_component_count": len(nearby_component_keys),
        "nearby_components": nearby_component_keys,
        "nearby_component_diagnostics": nearby_component_diagnostics,
        "nearby_component_cache_hit": nearby_component_cache_hit,
        "recommended_method": recommended_method,
        "auto_recommended_method": auto_recommended_method,
        "physical_capability": physical_capability,
        "visible_geometry_proven": visible_geometry_proven,
        "safe_for_auto_rigid_rotation": auto_rigid_recommended,
        "confidence": confidence,
        "candidates": [
            {
                key: (
                    value.tolist()
                    if isinstance(value, np.ndarray)
                    else (
                        None
                        if isinstance(value, float) and not np.isfinite(value)
                        else value
                    )
                )
                for key, value in record.items()
                if key
                not in {
                    "payload",
                    "bbox_center",
                }
            }
            for record in records
        ],
    }


def eye_gaze_landmark_screen_projection(
    *,
    landmark_positions: Mapping[int, Sequence[float]],
    landmark_screen_positions: Mapping[int, Sequence[float]],
) -> tuple[Any, float]:
    """Fit the cached orthographic mapping from mesh space to render space."""

    import numpy as np

    projection_ids = sorted(
        set(landmark_positions) & set(landmark_screen_positions)
    )
    if len(projection_ids) < 6:
        raise EyeGazeUVWarpValidationError(
            "Eye screen calibration needs at least six raycast landmark pairs."
        )
    projection_positions = np.stack(
        [landmark_positions[value] for value in projection_ids]
    )
    projection_screens = np.stack(
        [landmark_screen_positions[value] for value in projection_ids]
    )
    projection_design = np.concatenate(
        (
            projection_positions,
            np.ones((len(projection_ids), 1), dtype=np.float64),
        ),
        axis=1,
    )
    projection = np.linalg.lstsq(
        projection_design,
        projection_screens,
        rcond=None,
    )[0]
    projected_landmarks = projection_design @ projection
    error = float(
        np.linalg.norm(projected_landmarks - projection_screens, axis=1).max()
    )
    if not np.isfinite(projection).all() or not np.isfinite(error):
        raise EyeGazeUVWarpValidationError(
            "Eye landmark screen projection is not finite."
        )
    return projection, error


def tangent_world_shift_for_screen_shift(
    *,
    position_to_screen: Sequence[Sequence[float]],
    source_direction: Sequence[float],
    screen_shift: Sequence[float],
) -> Any:
    """Solve a screen displacement while remaining tangent to the eyeball."""

    import numpy as np

    projection = np.asarray(position_to_screen, dtype=np.float64)
    direction = np.asarray(source_direction, dtype=np.float64)
    target = np.asarray(screen_shift, dtype=np.float64)
    if (
        projection.shape != (3, 2)
        or direction.shape != (3,)
        or target.shape != (2,)
        or not np.isfinite(projection).all()
        or not np.isfinite(direction).all()
        or not np.isfinite(target).all()
    ):
        raise EyeGazeUVWarpValidationError(
            "Tangent screen calibration requires finite matching 3D/2D samples."
        )
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1.0e-8:
        raise EyeGazeUVWarpValidationError(
            "Tangent screen calibration has no stable eye direction."
        )
    source_unit = direction / direction_norm
    tangent_projector = np.eye(3, dtype=np.float64) - np.outer(
        source_unit, source_unit
    )
    tangent_to_screen = tangent_projector @ projection
    if np.linalg.matrix_rank(tangent_to_screen.T) < 2:
        raise EyeGazeUVWarpValidationError(
            "Tangent screen calibration is rank deficient."
        )
    world_shift = np.linalg.lstsq(
        tangent_to_screen.T,
        target,
        rcond=None,
    )[0]
    projected = world_shift @ projection
    maximum_error = max(float(np.linalg.norm(target)) * 0.10, 1.0e-6)
    if (
        not np.isfinite(world_shift).all()
        or float(np.linalg.norm(projected - target)) > maximum_error
    ):
        raise EyeGazeUVWarpValidationError(
            "Tangent screen calibration cannot reproduce the requested gaze shift."
        )
    return world_shift


def rigid_eye_rotation_from_sampling_shift(
    *,
    iris_position: Sequence[float],
    rotation_center: Sequence[float],
    sampling_world_shift: Sequence[float],
    rotation_radius: Optional[float] = None,
) -> tuple[Any, Any]:
    """Return a physical eye rotation from a texture-sampling displacement.

    The calibrated UV path stores the displacement used to sample the source
    texture, which is opposite the visible feature motion. A rigid eyeball must
    therefore target the inverse displacement.
    """

    import numpy as np

    iris_position = np.asarray(iris_position, dtype=np.float64)
    rotation_center = np.asarray(rotation_center, dtype=np.float64)
    sampling_world_shift = np.asarray(sampling_world_shift, dtype=np.float64)
    if (
        iris_position.shape != (3,)
        or rotation_center.shape != (3,)
        or sampling_world_shift.shape != (3,)
        or not np.isfinite(iris_position).all()
        or not np.isfinite(rotation_center).all()
        or not np.isfinite(sampling_world_shift).all()
    ):
        raise EyeGazeUVWarpValidationError(
            "Rigid eye rotation requires finite three-dimensional inputs."
        )

    source_direction = iris_position - rotation_center
    source_distance = float(np.linalg.norm(source_direction))
    if source_distance <= 1.0e-8:
        raise EyeGazeUVWarpValidationError(
            "Eye geometry rotation has an invalid center direction."
        )
    source_unit = source_direction / source_distance
    physical_world_shift = -sampling_world_shift
    # A calibrated surface displacement may include a large radial component,
    # especially on layered or strongly curved eyes. Radial motion cannot be
    # produced by a rigid rotation and previously made opposite gaze requests
    # yield asymmetric, sometimes enormous angles. Only the tangent-plane
    # component describes visible iris travel around the fitted eye center.
    physical_world_shift = physical_world_shift - (
        float(np.dot(physical_world_shift, source_unit)) * source_unit
    )
    effective_radius = source_distance
    if rotation_radius is not None:
        try:
            parsed_radius = float(rotation_radius)
        except (TypeError, ValueError):
            parsed_radius = float("nan")
        if np.isfinite(parsed_radius) and parsed_radius > 1.0e-8:
            effective_radius = parsed_radius
    maximum_shift = effective_radius * np.tan(
        np.deg2rad(MAX_PHYSICAL_EYE_ROTATION_DEGREES)
    )
    shift_norm = float(np.linalg.norm(physical_world_shift))
    if shift_norm > maximum_shift:
        physical_world_shift *= maximum_shift / shift_norm
    target_direction = source_unit * effective_radius + physical_world_shift
    target_distance = float(np.linalg.norm(target_direction))
    if target_distance <= 1.0e-8:
        raise EyeGazeUVWarpValidationError(
            "Eye geometry rotation has an invalid center direction."
        )
    source_direction = source_unit
    target_direction /= target_distance
    cross = np.cross(source_direction, target_direction)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source_direction, target_direction), -1.0, 1.0))
    if sine <= 1.0e-10:
        rotation_matrix = np.eye(3, dtype=np.float64)
    else:
        skew = np.asarray(
            (
                (0.0, -cross[2], cross[1]),
                (cross[2], 0.0, -cross[0]),
                (-cross[1], cross[0], 0.0),
            ),
            dtype=np.float64,
        )
        rotation_matrix = (
            np.eye(3, dtype=np.float64)
            + skew
            + skew @ skew * ((1.0 - cosine) / (sine * sine))
        )
    return rotation_matrix, physical_world_shift


def physical_eye_screen_shift_from_sampling_shift(
    sampling_screen_shift: Sequence[float],
) -> Any:
    """Convert calibrated source-sampling motion to visible iris motion."""

    import numpy as np

    shift = np.asarray(sampling_screen_shift, dtype=np.float64)
    if shift.shape != (2,) or not np.isfinite(shift).all():
        raise EyeGazeUVWarpValidationError(
            "Physical eye screen shift requires a finite two-dimensional input."
        )
    return -shift


def eye_gaze_screen_shift_from_eye_width(
    motion: Sequence[float],
    eye_width: float,
) -> Any:
    """Scale both gaze axes in eye-width units, including vertical gaze."""

    import numpy as np

    parsed_motion = np.asarray(motion, dtype=np.float64)
    parsed_width = float(eye_width)
    if (
        parsed_motion.shape != (2,)
        or not np.isfinite(parsed_motion).all()
        or not np.isfinite(parsed_width)
        or parsed_width <= 0.0
    ):
        raise EyeGazeUVWarpValidationError(
            "Eye screen motion requires a finite positive eye width."
        )
    return parsed_motion * parsed_width


def eye_gaze_topology_component_polygon_count(
    mesh_objects: Sequence[Any],
    landmark_payload: Mapping[str, Any],
    action_unit_id: int,
) -> int:
    spec = eye_gaze_landmark_spec(action_unit_id)
    if spec is None:
        raise EyeGazeUVWarpValidationError(
            f"AU{action_unit_id} is not an eye-gaze action unit."
        )
    anchors = landmark_payload.get("surface_anchors")
    if not isinstance(anchors, Mapping):
        raise EyeGazeUVWarpValidationError(
            "Eye-gaze topology selection requires surface anchors."
        )
    anchor = anchors.get(str(spec.center_id), anchors.get(spec.center_id))
    if not isinstance(anchor, Mapping):
        raise EyeGazeUVWarpValidationError(
            f"Missing iris-center surface anchor {spec.center_id}."
        )
    object_name = str(anchor.get("object_name", ""))
    eye_object = next(
        (obj for obj in mesh_objects if obj.name == object_name),
        None,
    )
    if eye_object is None:
        raise EyeGazeUVWarpValidationError(
            f"Surface-anchor object {object_name!r} was not imported."
        )
    seed_polygon = int(anchor["polygon_index"])
    mesh = eye_object.data
    if not 0 <= seed_polygon < len(mesh.polygons):
        raise EyeGazeUVWarpValidationError(
            f"Iris-center polygon {seed_polygon} is outside the source mesh."
        )
    vertex_to_polygons: list[list[int]] = [[] for _ in mesh.vertices]
    for polygon in mesh.polygons:
        for vertex_id in polygon.vertices:
            vertex_to_polygons[int(vertex_id)].append(int(polygon.index))
    selected = {seed_polygon}
    pending = [seed_polygon]
    while pending:
        polygon_index = pending.pop()
        for vertex_id in mesh.polygons[polygon_index].vertices:
            for neighbor in vertex_to_polygons[int(vertex_id)]:
                if neighbor not in selected:
                    selected.add(neighbor)
                    pending.append(neighbor)
    return len(selected)


def eye_gaze_appearance_mode_for_component_polygon_count(
    polygon_count: int,
) -> str:
    if int(polygon_count) < 1:
        raise ValueError("Eye topology component polygon count must be positive.")
    if int(polygon_count) <= MAX_COPY_TEXTURE_EYE_COMPONENT_POLYGONS:
        return "segmented_texture"
    return "hybrid"


def eye_gaze_has_visible_texture_layers(
    landmark_payload: Mapping[str, Any],
    action_unit_id: int,
) -> bool:
    spec = eye_gaze_landmark_spec(action_unit_id)
    if spec is None:
        return False
    selection = landmark_payload.get("surface_anchor_selection")
    if not isinstance(selection, Mapping):
        return False
    eye_selection = selection.get(spec.eye)
    if not isinstance(eye_selection, Mapping):
        return False
    layers = eye_selection.get("visible_layers")
    return (
        isinstance(layers, Sequence)
        and not isinstance(layers, (str, bytes))
        and len(layers) > 0
    )


def apply_fbx_eye_gaze_texture_warp(
    *,
    mesh_objects: Sequence[Any],
    layer_reports: Sequence[Mapping[str, float]],
    erase_radius_scale: float = 1.08,
    copy_source_iris: bool = False,
    fit_source_iris: bool = False,
    use_source_iris_mask: bool = False,
    copy_source_iris_mask: bool = False,
    copy_radius_scale: float = 1.0,
    draw_target_iris: bool = True,
    maximum_target_layers: Optional[int] = None,
) -> dict[str, Any]:
    import numpy as np

    if layer_reports and all(
        abs(float(layer.get("visual_shift_u", 0.0))) <= 1.0e-12
        and abs(float(layer.get("visual_shift_v", 0.0))) <= 1.0e-12
        for layer in layer_reports
    ):
        return {"images": [], "skipped_zero_motion": True}

    images = []
    for obj in mesh_objects:
        for slot in obj.material_slots:
            material = slot.material
            if material is None or not material.use_nodes or material.node_tree is None:
                continue
            for node in material.node_tree.nodes:
                if node.type != "BSDF_PRINCIPLED":
                    continue
                base_color = node.inputs.get("Base Color")
                if base_color is None:
                    continue
                for link in base_color.links:
                    image_node = link.from_node
                    if (
                        image_node.type == "TEX_IMAGE"
                        and image_node.image is not None
                        and image_node.image not in images
                    ):
                        images.append(image_node.image)
    if not images:
        raise EyeGazeUVWarpValidationError(
            "Packed-texture eye gaze requires at least one material image."
        )

    image_reports = []
    for image in images:
        width, height = int(image.size[0]), int(image.size[1])
        if width < 2 or height < 2:
            raise EyeGazeUVWarpValidationError(
                f"Packed texture {image.name!r} has invalid dimensions."
            )
        flat_pixels = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(flat_pixels)
        pixels = flat_pixels.reshape(height, width, 4)
        source_pixels = pixels.copy()
        translated_pixels = 0
        fitted_iris_scales = []
        source_iris_masks = []

        for layer_index, layer in enumerate(layer_reports):
            center_u = float(layer["center_u"])
            center_v = float(layer["center_v"])
            iris_radius_uv = float(layer["iris_radius_uv"])
            iris_horizontal_radius_uv = float(
                layer.get("iris_horizontal_radius_uv", iris_radius_uv)
            )
            iris_vertical_radius_uv = float(
                layer.get("iris_vertical_radius_uv", iris_radius_uv)
            )
            horizontal_axis = np.asarray(
                (
                    float(layer.get("iris_horizontal_axis_u", 1.0)),
                    float(layer.get("iris_horizontal_axis_v", 0.0)),
                ),
                dtype=np.float64,
            )
            vertical_axis = np.asarray(
                (
                    float(layer.get("iris_vertical_axis_u", 0.0)),
                    float(layer.get("iris_vertical_axis_v", 1.0)),
                ),
                dtype=np.float64,
            )
            shift_u = float(layer["visual_shift_u"])
            shift_v = float(layer["visual_shift_v"])
            center_x = int(round(center_u * (width - 1)))
            center_y = int(round(center_v * (height - 1)))
            shift_x = int(round(shift_u * width))
            shift_y = int(round(shift_v * height))
            iris_radius_pixels = max(
                2,
                int(
                    round(
                        max(
                            iris_horizontal_radius_uv,
                            iris_vertical_radius_uv,
                        )
                        * max(width, height)
                    )
                ),
            )
            erase_scale = max(1.0, float(erase_radius_scale))
            sample_radius = int(round(iris_radius_pixels * erase_scale * 2.0))
            x0 = max(0, center_x - sample_radius)
            x1 = min(width, center_x + sample_radius + 1)
            y0 = max(0, center_y - sample_radius)
            y1 = min(height, center_y + sample_radius + 1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                raise EyeGazeUVWarpValidationError(
                    "Pupil texture patch is too small to translate."
                )

            yy, xx = np.mgrid[y0:y1, x0:x1]
            delta_u = (xx - center_x) / max(width - 1, 1)
            delta_v = (yy - center_y) / max(height - 1, 1)
            horizontal_coordinate = (
                delta_u * horizontal_axis[0] + delta_v * horizontal_axis[1]
            ) / iris_horizontal_radius_uv
            vertical_coordinate = (
                delta_u * vertical_axis[0] + delta_v * vertical_axis[1]
            ) / iris_vertical_radius_uv
            landmark_distance = np.sqrt(
                horizontal_coordinate**2 + vertical_coordinate**2
            )
            source_patch = source_pixels[y0:y1, x0:x1].copy()
            luminance = (
                source_patch[..., 0] * 0.2126
                + source_patch[..., 1] * 0.7152
                + source_patch[..., 2] * 0.0722
            )
            sclera_candidates = (
                (landmark_distance <= erase_scale * 1.8)
                & (luminance >= 0.70)
            )
            if int(np.count_nonzero(sclera_candidates)) < 16:
                raise EyeGazeUVWarpValidationError(
                    "Pupil texture patch does not contain enough sclera pixels."
                )
            sclera_color = np.median(
                source_patch[..., :3][sclera_candidates],
                axis=0,
            )
            horizontal_iris_scale = 1.0
            vertical_iris_scale = 1.0
            segmented_iris_mask = None
            source_erase_mask = None
            source_iris_mask_report = None
            if copy_source_iris and use_source_iris_mask:
                sclera_luminance = float(
                    sclera_color[0] * 0.2126
                    + sclera_color[1] * 0.7152
                    + sclera_color[2] * 0.0722
                )
                center_local = (center_y - y0, center_x - x0)
                center_luminance = float(luminance[center_local])
                maximum_luminance_limit = min(
                    0.46,
                    max(0.24, center_luminance + 0.42),
                    sclera_luminance - 0.15,
                )
                minimum_luminance_limit = min(
                    maximum_luminance_limit,
                    max(0.18, center_luminance + 0.16),
                )
                expected_pixels = max(
                    1.0,
                    np.pi
                    * iris_horizontal_radius_uv
                    * width
                    * iris_vertical_radius_uv
                    * height,
                )

                def source_iris_component(luminance_limit: float) -> Any:
                    candidates = (
                        (landmark_distance <= 2.60)
                        & (luminance <= luminance_limit)
                    )
                    component = np.zeros_like(candidates, dtype=bool)
                    if not candidates[center_local]:
                        return component
                    component[center_local] = True
                    pending = [center_local]
                    while pending:
                        row, column = pending.pop()
                        for row_offset in (-1, 0, 1):
                            for column_offset in (-1, 0, 1):
                                if row_offset == 0 and column_offset == 0:
                                    continue
                                neighbor_row = row + row_offset
                                neighbor_column = column + column_offset
                                if (
                                    0 <= neighbor_row < component.shape[0]
                                    and 0 <= neighbor_column < component.shape[1]
                                    and candidates[neighbor_row, neighbor_column]
                                    and not component[neighbor_row, neighbor_column]
                                ):
                                    component[neighbor_row, neighbor_column] = True
                                    pending.append((neighbor_row, neighbor_column))
                    if int(np.count_nonzero(component)) < 16:
                        return component

                    exterior = np.zeros_like(component, dtype=bool)
                    pending = []
                    for row in (0, component.shape[0] - 1):
                        for column in range(component.shape[1]):
                            if (
                                not component[row, column]
                                and not exterior[row, column]
                            ):
                                exterior[row, column] = True
                                pending.append((row, column))
                    for column in (0, component.shape[1] - 1):
                        for row in range(component.shape[0]):
                            if (
                                not component[row, column]
                                and not exterior[row, column]
                            ):
                                exterior[row, column] = True
                                pending.append((row, column))
                    while pending:
                        row, column = pending.pop()
                        for row_offset, column_offset in (
                            (-1, 0),
                            (1, 0),
                            (0, -1),
                            (0, 1),
                        ):
                            neighbor_row = row + row_offset
                            neighbor_column = column + column_offset
                            if (
                                0 <= neighbor_row < exterior.shape[0]
                                and 0 <= neighbor_column < exterior.shape[1]
                                and not component[neighbor_row, neighbor_column]
                                and not exterior[neighbor_row, neighbor_column]
                            ):
                                exterior[neighbor_row, neighbor_column] = True
                                pending.append((neighbor_row, neighbor_column))
                    component |= ~component & ~exterior

                    padded = np.pad(
                        component,
                        1,
                        mode="constant",
                        constant_values=False,
                    )
                    expanded = np.zeros_like(component)
                    for row_offset in range(3):
                        for column_offset in range(3):
                            expanded |= padded[
                                row_offset : row_offset + component.shape[0],
                                column_offset : column_offset + component.shape[1],
                            ]
                    return expanded & (landmark_distance <= 2.70)

                selected_component = None
                selected_report = None
                rejected_report = None
                for iris_luminance_limit in np.linspace(
                    maximum_luminance_limit,
                    minimum_luminance_limit,
                    17,
                ):
                    component = source_iris_component(float(iris_luminance_limit))
                    component_pixels = int(np.count_nonzero(component))
                    if component_pixels < 16:
                        continue
                    area_ratio = float(component_pixels / expected_pixels)
                    maximum_radius = float(landmark_distance[component].max())
                    rejected_report = {
                        "area_ratio": area_ratio,
                        "luminance_limit": float(iris_luminance_limit),
                        "maximum_radius": maximum_radius,
                        "pixels": component_pixels,
                    }
                    maximum_area_ratio = (
                        5.0 if copy_source_iris_mask else 2.50
                    )
                    if (
                        0.45 <= area_ratio <= maximum_area_ratio
                        and maximum_radius <= 2.68
                    ):
                        selected_component = component
                        selected_report = rejected_report
                        break
                if selected_component is None or selected_report is None:
                    source_iris_mask_report = {
                        "status": "geometric_fallback",
                        "last_candidate": rejected_report,
                    }
                else:
                    source_erase_mask = selected_component
                    if copy_source_iris_mask:
                        segmented_iris_mask = selected_component
                    source_iris_mask_report = {
                        "status": "selected_for_source_cleanup",
                        **selected_report,
                    }
            elif copy_source_iris and fit_source_iris:
                sclera_luminance = float(
                    sclera_color[0] * 0.2126
                    + sclera_color[1] * 0.7152
                    + sclera_color[2] * 0.0722
                )
                color_distance = np.linalg.norm(
                    source_patch[..., :3] - sclera_color,
                    axis=-1,
                )
                iris_candidates = (
                    (landmark_distance <= 1.25)
                    & (
                        (luminance <= sclera_luminance - 0.12)
                        | (color_distance >= 0.22)
                    )
                )
                center_local = (center_y - y0, center_x - x0)
                seed = center_local
                if not iris_candidates[seed]:
                    nearby = np.argwhere(
                        iris_candidates & (landmark_distance <= 0.65)
                    )
                    if len(nearby) > 0:
                        distances = np.sum(
                            (nearby - np.asarray(center_local)) ** 2,
                            axis=1,
                        )
                        seed = tuple(int(value) for value in nearby[distances.argmin()])
                if iris_candidates[seed]:
                    component = np.zeros_like(iris_candidates, dtype=bool)
                    component[seed] = True
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
                                    0 <= neighbor_row < component.shape[0]
                                    and 0 <= neighbor_column < component.shape[1]
                                    and iris_candidates[neighbor_row, neighbor_column]
                                    and not component[neighbor_row, neighbor_column]
                                ):
                                    component[neighbor_row, neighbor_column] = True
                                    pending.append((neighbor_row, neighbor_column))
                    if int(np.count_nonzero(component)) >= 16:
                        segmented_iris_mask = component
                        horizontal_iris_scale = float(
                            np.quantile(
                                np.abs(horizontal_coordinate[component]),
                                0.98,
                            )
                        )
                        vertical_iris_scale = float(
                            np.quantile(
                                np.abs(vertical_coordinate[component]),
                                0.98,
                            )
                        )
                        horizontal_iris_scale = float(
                            np.clip(horizontal_iris_scale, 1.0, 1.8)
                        )
                        vertical_iris_scale = float(
                            np.clip(vertical_iris_scale, 1.0, 1.8)
                        )
            fitted_iris_scales.append(
                {
                    "horizontal": horizontal_iris_scale,
                    "vertical": vertical_iris_scale,
                }
            )
            source_iris_masks.append(source_iris_mask_report)
            normalized_distance = np.sqrt(
                (horizontal_coordinate / horizontal_iris_scale) ** 2
                + (vertical_coordinate / vertical_iris_scale) ** 2
            )
            iris_disk = (
                segmented_iris_mask
                if segmented_iris_mask is not None
                else normalized_distance <= 1.0
            )
            erase_component = (
                source_erase_mask
                if source_erase_mask is not None
                else segmented_iris_mask
            )
            if erase_component is None:
                feather_width = max(erase_scale * 0.10, 1.0e-6)
                outer_erase_radius = erase_scale + feather_width
                erase_disk = normalized_distance <= outer_erase_radius
                edge_feather = np.clip(
                    (outer_erase_radius - normalized_distance) / feather_width,
                    0.0,
                    1.0,
                )
            else:
                def dilate(mask: Any) -> Any:
                    padded = np.pad(mask, 1, mode="constant", constant_values=False)
                    expanded = np.zeros_like(mask)
                    for row_offset in range(3):
                        for column_offset in range(3):
                            expanded |= padded[
                                row_offset : row_offset + mask.shape[0],
                                column_offset : column_offset + mask.shape[1],
                            ]
                    return expanded

                erase_disk = erase_component.copy()
                edge_feather = erase_component.astype(np.float64)
                feather_steps = max(2, int(round(iris_radius_pixels * 0.08)))
                previous = erase_component.copy()
                for step in range(feather_steps):
                    expanded = dilate(previous)
                    ring = expanded & ~previous
                    edge_feather[ring] = 1.0 - (step + 1) / (feather_steps + 1)
                    previous = expanded
                erase_disk = previous
            pupil_samples = iris_disk & (luminance <= 0.20)
            if int(np.count_nonzero(pupil_samples)) < 16:
                if not copy_source_iris:
                    raise EyeGazeUVWarpValidationError(
                        "Pupil texture patch does not contain a dark pupil region."
                    )
                pupil_color = source_patch[center_y - y0, center_x - x0, :3]
            else:
                pupil_color = np.median(
                    source_patch[..., :3][pupil_samples],
                    axis=0,
                )

            erase_alpha = edge_feather[..., None] * erase_disk[..., None]
            current_patch = pixels[y0:y1, x0:x1, :3]
            current_patch[:] = (
                current_patch * (1.0 - erase_alpha)
                + sclera_color * erase_alpha
            )

            draw_this_target = draw_target_iris and (
                maximum_target_layers is None
                or layer_index < maximum_target_layers
            )
            if draw_this_target:
                target_x = xx + shift_x
                target_y = yy + shift_y
                if segmented_iris_mask is not None:
                    pupil_feather = segmented_iris_mask.astype(np.float64)
                    previous = segmented_iris_mask.copy()
                    for step, alpha in enumerate((0.66, 0.33)):
                        expanded = dilate(previous)
                        pupil_feather[expanded & ~previous] = alpha
                        previous = expanded
                else:
                    pupil_core_radius = (
                        max(1.0, float(copy_radius_scale))
                        if copy_source_iris
                        else 0.95
                    )
                    pupil_feather_width = max(
                        pupil_core_radius * 0.08,
                        1.0e-6,
                    )
                    pupil_outer_radius = (
                        pupil_core_radius + pupil_feather_width
                    )
                    pupil_feather = np.clip(
                        (pupil_outer_radius - normalized_distance)
                        / pupil_feather_width,
                        0.0,
                        1.0,
                    )
                pupil_mask = pupil_feather >= 0.01
                valid = (
                    pupil_mask
                    & (target_x >= 0)
                    & (target_x < width)
                    & (target_y >= 0)
                    & (target_y < height)
                )
                source_y, source_x = np.nonzero(valid)
                destination_x = target_x[valid]
                destination_y = target_y[valid]
                moved_alpha = pupil_feather[source_y, source_x, None]
                destination = pixels[destination_y, destination_x, :3]
                moved_color = (
                    source_patch[source_y, source_x, :3]
                    if copy_source_iris
                    else pupil_color
                )
                pixels[destination_y, destination_x, :3] = (
                    destination * (1.0 - moved_alpha)
                    + moved_color * moved_alpha
                )
                translated_pixels += int(len(source_x))

        image.pixels.foreach_set(pixels.reshape(-1))
        image.update()
        image.pack()
        image_reports.append(
            {
                "image": image.name,
                "width": width,
                "height": height,
                "translated_pixels": translated_pixels,
                "fitted_iris_scales": fitted_iris_scales,
                "source_iris_masks": source_iris_masks,
            }
        )
    return {"images": image_reports}


def eye_gaze_texture_copy_shift(
    layer: Mapping[str, float],
    *,
    width: int,
    height: int,
) -> tuple[int, int]:
    """Convert UV sampling motion to the opposite physical texture motion."""

    return (
        -int(round(float(layer["visual_shift_u"]) * width)),
        -int(round(float(layer["visual_shift_v"]) * height)),
    )


def apply_fbx_eye_gaze_color_texture_warp(
    *,
    mesh_objects: Sequence[Any],
    layer_reports: Sequence[Mapping[str, float]],
    eye_name: str,
    mask_cache_dir: Optional[Path] = None,
    mask_debug_dir: Optional[Path] = None,
    reference_sclera_colors: Optional[Mapping[str, Sequence[float]]] = None,
    cleanup_only: bool = False,
) -> dict[str, Any]:
    """Move the existing color-selected iris without changing mesh geometry."""

    import numpy as np

    if not layer_reports:
        raise EyeGazeUVWarpValidationError(
            "Color-selected gaze requires at least one visible iris layer."
        )
    if len(layer_reports) > 1:
        shared_sclera = None
        per_layer = []
        skipped_layers = []
        for layer_index, layer_report in enumerate(layer_reports):
            layer_cleanup_only = bool(
                layer_report.get("texture_cleanup_only", False)
            )
            try:
                result = apply_fbx_eye_gaze_color_texture_warp(
                    mesh_objects=mesh_objects,
                    layer_reports=[layer_report],
                    eye_name=f"{eye_name}_layer{layer_index}",
                    mask_cache_dir=mask_cache_dir,
                    mask_debug_dir=mask_debug_dir,
                    reference_sclera_colors=(
                        shared_sclera if layer_cleanup_only else None
                    ),
                    cleanup_only=layer_cleanup_only,
                )
            except EyeGazeUVWarpValidationError as exc:
                skipped_layers.append(
                    {"layer_index": layer_index, "reason": str(exc)}
                )
                continue
            per_layer.append(result)
            if shared_sclera is None:
                shared_sclera = {
                    str(image_report["image"]): image_report["selection"][
                        "sclera_color"
                    ]
                    for image_report in result["images"]
                }
        if not per_layer:
            reasons = "; ".join(
                str(record["reason"]) for record in skipped_layers
            )
            raise EyeGazeUVWarpValidationError(
                "Color iris selection failed on every bounded eye layer: "
                + reasons
            )
        return {
            "method": "color_segmented_texture",
            "geometry_modified": False,
            "layer_count": len(per_layer),
            "skipped_layers": skipped_layers,
            "layers": per_layer,
            "images": [
                image_report
                for layer_report in per_layer
                for image_report in layer_report["images"]
            ],
        }
    layer = layer_reports[0]
    images = []
    for obj in mesh_objects:
        for slot in obj.material_slots:
            material = slot.material
            if material is None or not material.use_nodes or material.node_tree is None:
                continue
            for node in material.node_tree.nodes:
                if node.type != "BSDF_PRINCIPLED":
                    continue
                base_color = node.inputs.get("Base Color")
                if base_color is None:
                    continue
                for link in base_color.links:
                    image_node = link.from_node
                    if (
                        image_node.type == "TEX_IMAGE"
                        and image_node.image is not None
                        and image_node.image not in images
                    ):
                        images.append(image_node.image)
    if not images:
        raise EyeGazeUVWarpValidationError(
            "Color-selected gaze requires a packed base-color texture."
        )

    image_reports = []
    for image_index, image in enumerate(images):
        width, height = int(image.size[0]), int(image.size[1])
        if width < 2 or height < 2:
            raise EyeGazeUVWarpValidationError(
                f"Packed texture {image.name!r} has invalid dimensions."
            )
        flat_pixels = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(flat_pixels)
        pixels = flat_pixels.reshape(height, width, 4)
        center_xy = (
            float(layer["center_u"]) * (width - 1),
            float(layer["center_v"]) * (height - 1),
        )
        horizontal_radius_uv = float(layer["iris_horizontal_radius_uv"])
        vertical_radius_uv = float(layer["iris_vertical_radius_uv"])
        horizontal_axis_uv = (
            float(layer["iris_horizontal_axis_u"]),
            float(layer["iris_horizontal_axis_v"]),
        )
        vertical_axis_uv = (
            float(layer["iris_vertical_axis_u"]),
            float(layer["iris_vertical_axis_v"]),
        )
        reference_sclera_color = None
        if reference_sclera_colors is not None:
            raw_reference = reference_sclera_colors.get(image.name)
            if raw_reference is not None:
                reference_sclera_color = tuple(float(value) for value in raw_reference)
        if cleanup_only:
            if reference_sclera_color is None:
                raise EyeGazeUVWarpValidationError(
                    "Rear iris cleanup requires the visible shell's sclera color."
                )
            try:
                cleanup_report = apply_iris_occlusion_cleanup(
                    pixels,
                    center_xy=center_xy,
                    horizontal_axis_uv=horizontal_axis_uv,
                    vertical_axis_uv=vertical_axis_uv,
                    horizontal_radius_uv=horizontal_radius_uv,
                    vertical_radius_uv=vertical_radius_uv,
                    sclera_color=reference_sclera_color,
                )
            except ValueError as exc:
                raise EyeGazeUVWarpValidationError(
                    f"Rear iris cleanup failed for {eye_name} eye: {exc}"
                ) from exc
            image.pixels.foreach_set(pixels.reshape(-1))
            image.update()
            image.pack()
            image_reports.append(
                {
                    "image": image.name,
                    "width": width,
                    "height": height,
                    "eye": eye_name,
                    "cache_path": None,
                    "cache_hit": True,
                    "shift_x": 0,
                    "shift_y": 0,
                    "selection": {
                        "method": "rear_occlusion_cleanup",
                        "sclera_color": list(reference_sclera_color),
                    },
                    **cleanup_report,
                }
            )
            continue
        safe_image_name = "".join(
            value if value.isalnum() or value in {"-", "_"} else "_"
            for value in image.name
        ).strip("_") or f"image_{image_index}"
        cache_path = (
            Path(mask_cache_dir) / f"{eye_name}_{image_index}_{safe_image_name}.npz"
            if mask_cache_dir is not None
            else None
        )
        cache_hit = False
        selection = None
        if cache_path is not None and cache_path.is_file():
            try:
                candidate = load_iris_texture_selection(cache_path)
            except (OSError, ValueError, KeyError):
                candidate = None
            if candidate is not None and selection_matches(
                candidate,
                width=width,
                height=height,
                center_xy=center_xy,
                horizontal_radius_uv=horizontal_radius_uv,
                vertical_radius_uv=vertical_radius_uv,
                reference_sclera_color=reference_sclera_color,
            ):
                selection = candidate
                cache_hit = True
        if selection is None:
            try:
                selection = segment_iris_texture(
                    pixels,
                    center_xy=center_xy,
                    horizontal_axis_uv=horizontal_axis_uv,
                    vertical_axis_uv=vertical_axis_uv,
                    horizontal_radius_uv=horizontal_radius_uv,
                    vertical_radius_uv=vertical_radius_uv,
                    reference_sclera_color=reference_sclera_color,
                    cache_metadata={
                        "eye": eye_name,
                        "image": image.name,
                    },
                )
            except ValueError as exc:
                raise EyeGazeUVWarpValidationError(
                    f"Color iris selection failed for {eye_name} eye: {exc}"
                ) from exc
            if cache_path is not None:
                save_iris_texture_selection(cache_path, selection)
        if mask_debug_dir is not None:
            debug_path = (
                Path(mask_debug_dir)
                / f"{eye_name}_{image_index}_{safe_image_name}.png"
            )
            if not debug_path.is_file() or not cache_hit:
                write_iris_selection_debug(debug_path, pixels, selection)

        shift_xy = eye_gaze_texture_copy_shift(
            layer,
            width=width,
            height=height,
        )
        iris_radius_pixels = max(
            horizontal_radius_uv * width,
            vertical_radius_uv * height,
        )
        if (
            float(np.linalg.norm(shift_xy))
            > iris_radius_pixels * MAX_TEXTURE_GAZE_SHIFT_IRIS_RADII
        ):
            raise EyeGazeUVWarpValidationError(
                "Color-selected iris movement exceeds the safe landmark radius: "
                f"shift={shift_xy}, radius={iris_radius_pixels:.3f}."
            )
        if shift_xy == (0, 0):
            translation_report = {
                "source_pixels": int(np.count_nonzero(selection.mask)),
                "translated_pixels": 0,
                "clipped_pixels": 0,
            }
        else:
            translation_report = apply_iris_texture_translation(
                pixels,
                selection,
                shift_xy=shift_xy,
            )
            minimum_visible_pixels = max(
                16,
                int(translation_report["source_pixels"] * 0.35),
            )
            if translation_report["translated_pixels"] < minimum_visible_pixels:
                raise EyeGazeUVWarpValidationError(
                    "The destination sclera clips too much of the moved iris: "
                    f"visible={translation_report['translated_pixels']}, "
                    f"minimum={minimum_visible_pixels}."
                )
            image.pixels.foreach_set(pixels.reshape(-1))
            image.update()
            image.pack()
        image_reports.append(
            {
                "image": image.name,
                "width": width,
                "height": height,
                "eye": eye_name,
                "cache_path": str(cache_path) if cache_path is not None else None,
                "cache_hit": cache_hit,
                "shift_x": shift_xy[0],
                "shift_y": shift_xy[1],
                "selection": selection.metadata,
                **translation_report,
            }
        )
    return {
        "method": "color_segmented_texture",
        "geometry_modified": False,
        "images": image_reports,
    }


def eye_gaze_topology_component(
    mesh: Any,
    seed_polygon: int,
    *,
    topology_index: Optional[dict[str, Any]] = None,
) -> tuple[Any, Any]:
    import numpy as np

    if not 0 <= int(seed_polygon) < len(mesh.polygons):
        raise EyeGazeUVWarpValidationError(
            f"Iris-center polygon {seed_polygon} is outside the source mesh."
        )
    if topology_index is None:
        topology_index = build_eye_gaze_topology_index(mesh)
    elif topology_index.get("mesh") is not mesh:
        raise ValueError("The eye topology index belongs to a different mesh.")
    vertex_to_polygons = topology_index["vertex_to_polygons"]
    component_labels = topology_index["component_labels"]
    components = topology_index["components"]
    existing_label = int(component_labels[int(seed_polygon)])
    if existing_label >= 0:
        return components[existing_label]

    component_label = len(components)
    component_labels[int(seed_polygon)] = component_label
    pending = [int(seed_polygon)]
    polygon_values: list[int] = []
    vertex_values: set[int] = set()
    while pending:
        polygon_index = pending.pop()
        polygon_values.append(polygon_index)
        for vertex_id in mesh.polygons[polygon_index].vertices:
            vertex_id = int(vertex_id)
            vertex_values.add(vertex_id)
            for neighbor in vertex_to_polygons[vertex_id]:
                if int(component_labels[neighbor]) < 0:
                    component_labels[neighbor] = component_label
                    pending.append(neighbor)
    polygon_ids = np.asarray(sorted(polygon_values), dtype=np.int64)
    vertex_ids = np.fromiter(
        sorted(vertex_values),
        dtype=np.int64,
        count=len(vertex_values),
    )
    components.append((polygon_ids, vertex_ids))
    return polygon_ids, vertex_ids


def build_eye_gaze_topology_index(mesh: Any) -> dict[str, Any]:
    """Build reusable connectivity state for seeded eye-component queries."""

    import numpy as np

    vertex_to_polygons: list[list[int]] = [[] for _ in mesh.vertices]
    for polygon in mesh.polygons:
        polygon_index = int(polygon.index)
        for vertex_id in polygon.vertices:
            vertex_to_polygons[int(vertex_id)].append(polygon_index)
    return {
        "mesh": mesh,
        "vertex_to_polygons": vertex_to_polygons,
        "component_labels": np.full(len(mesh.polygons), -1, dtype=np.int32),
        "components": [],
    }


def triangulate_mesh_in_reference_configuration(
    mesh: Any,
    reference_coordinates: Any,
) -> int:
    """Triangulate using neutral coordinates, then restore final deformation.

    Blender's n-gon tessellation can choose different diagonals after a mesh is
    deformed. GLB supports triangles only, so fixing the diagonals in the source
    configuration keeps exported connectivity invariant across gaze poses.
    """

    import bmesh
    import numpy as np

    polygon_count = sum(1 for polygon in mesh.polygons if len(polygon.vertices) > 3)
    if polygon_count == 0:
        return 0
    reference = np.asarray(reference_coordinates, dtype=np.float64)
    if reference.shape != (len(mesh.vertices), 3):
        raise ValueError("Reference coordinates must match the mesh vertices.")
    final_coordinates = np.empty_like(reference)
    mesh.vertices.foreach_get("co", final_coordinates.reshape(-1))
    mesh.vertices.foreach_set("co", reference.reshape(-1))
    mesh.update(calc_edges=True)

    vertex_count = len(mesh.vertices)
    editable = bmesh.new()
    try:
        editable.from_mesh(mesh)
        faces = [face for face in editable.faces if len(face.verts) > 3]
        if faces:
            bmesh.ops.triangulate(
                editable,
                faces=faces,
                quad_method="FIXED",
                ngon_method="EAR_CLIP",
            )
            editable.to_mesh(mesh)
    finally:
        editable.free()
    if len(mesh.vertices) != vertex_count:
        raise RuntimeError("Neutral triangulation unexpectedly changed vertex count.")
    mesh.vertices.foreach_set("co", final_coordinates.reshape(-1))
    mesh.update(calc_edges=True)
    return int(polygon_count)


def apply_shared_eye_geometry_transform(
    *,
    eye_object: Any,
    seed_polygon: int,
    geometry_rotation_center: Optional[Sequence[float]],
    geometry_rotation_matrix: Optional[Sequence[Sequence[float]]],
    geometry_shared_translation: Optional[Sequence[float]],
    geometry_translation_center: Optional[Sequence[float]],
    geometry_translation_normal: Optional[Sequence[float]],
    geometry_translation_inner_radius: Optional[float],
    geometry_translation_outer_radius: Optional[float],
    report_as_translation: bool,
    processed_geometry_components: Optional[set[tuple[str, int]]],
    topology_index: Optional[dict[str, Any]] = None,
    coordinate_cache: Optional[dict[int, tuple[Any, Any]]] = None,
) -> dict[str, Any]:
    import numpy as np

    mesh = eye_object.data
    polygon_ids, vertex_ids = eye_gaze_topology_component(
        mesh,
        seed_polygon,
        topology_index=topology_index,
    )
    if len(polygon_ids) > 20_000:
        raise EyeGazeUVWarpValidationError(
            "Eye geometry component is too large to transform safely: "
            f"polygons={len(polygon_ids)}, maximum=20000."
        )
    component_key = (str(eye_object.name), int(polygon_ids.min()))
    duplicate = bool(
        processed_geometry_components is not None
        and component_key in processed_geometry_components
    )
    transformed_vertex_count = 0
    if not duplicate:
        cached_coordinates = (
            coordinate_cache.get(id(mesh))
            if coordinate_cache is not None
            else None
        )
        if cached_coordinates is None:
            coordinates = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
            mesh.vertices.foreach_get("co", coordinates)
            coordinates = coordinates.reshape(-1, 3)
            if coordinate_cache is not None:
                coordinate_cache[id(mesh)] = (mesh, coordinates)
        else:
            _cached_mesh, coordinates = cached_coordinates
        component_coordinates = coordinates[vertex_ids]
        if geometry_shared_translation is not None:
            translation = np.asarray(
                geometry_shared_translation,
                dtype=np.float64,
            )
            if translation.shape != (3,):
                raise EyeGazeUVWarpValidationError(
                    "Shared eye geometry translation has invalid dimensions."
                )
            if geometry_translation_center is None:
                transformed = component_coordinates + translation
                transformed_vertex_count = int(len(vertex_ids))
            else:
                center = np.asarray(
                    geometry_translation_center,
                    dtype=np.float64,
                )
                inner_radius = float(geometry_translation_inner_radius or 0.0)
                outer_radius = float(geometry_translation_outer_radius or 0.0)
                if (
                    center.shape != (3,)
                    or inner_radius <= 0.0
                    or outer_radius <= inner_radius
                ):
                    raise EyeGazeUVWarpValidationError(
                        "Stacked eye translation has an invalid taper region."
                    )
                deltas = component_coordinates - center
                if geometry_translation_normal is not None:
                    normal = np.asarray(
                        geometry_translation_normal, dtype=np.float64
                    )
                    normal_length = float(np.linalg.norm(normal))
                    if normal.shape != (3,) or normal_length <= 1.0e-8:
                        raise EyeGazeUVWarpValidationError(
                            "Stacked eye translation has an invalid surface normal."
                        )
                    normal = normal / normal_length
                    deltas = (
                        deltas
                        - (deltas @ normal)[:, None] * normal[None, :]
                    )
                distances = np.linalg.norm(deltas, axis=1)
                weights = np.clip(
                    (outer_radius - distances)
                    / (outer_radius - inner_radius),
                    0.0,
                    1.0,
                )
                weights = weights * weights * (3.0 - 2.0 * weights)
                transformed = (
                    component_coordinates + weights[:, None] * translation
                )
                transformed_vertex_count = int(
                    np.count_nonzero(weights > 1.0e-4)
                )
        else:
            if geometry_rotation_center is None or geometry_rotation_matrix is None:
                raise EyeGazeUVWarpValidationError(
                    "Stacked eye geometry requires a shared rotation or translation."
                )
            rotation_center = np.asarray(
                geometry_rotation_center,
                dtype=np.float64,
            )
            rotation_matrix = np.asarray(
                geometry_rotation_matrix,
                dtype=np.float64,
            )
            if rotation_center.shape != (3,) or rotation_matrix.shape != (3, 3):
                raise EyeGazeUVWarpValidationError(
                    "Shared eye geometry rotation has invalid dimensions."
                )
            transformed = (
                (component_coordinates - rotation_center) @ rotation_matrix.T
                + rotation_center
            )
            transformed_vertex_count = int(len(vertex_ids))
        coordinates[vertex_ids] = transformed
        if coordinate_cache is None:
            mesh.vertices.foreach_set("co", coordinates.reshape(-1))
            mesh.update(calc_edges=True)
            if hasattr(mesh, "calc_normals"):
                mesh.calc_normals()
        if processed_geometry_components is not None:
            processed_geometry_components.add(component_key)
    return {
        "warped_loop_count": 0.0,
        "maximum_weight": 0.0,
        "geometry_vertices_translated": float(
            transformed_vertex_count if report_as_translation else 0
        ),
        "geometry_vertices_rotated": float(
            transformed_vertex_count if not report_as_translation else 0
        ),
        "geometry_component_polygon_count": float(len(polygon_ids)),
        "geometry_component_min_polygon": float(component_key[1]),
        "geometry_component_duplicate": float(duplicate),
        "geometry_rotation_center": geometry_rotation_center,
        "geometry_rotation_matrix": geometry_rotation_matrix,
        "geometry_shared_translation": geometry_shared_translation,
        "geometry_rotation_tapered": float(
            geometry_shared_translation is not None
            and geometry_translation_center is not None
        ),
        "geometry_translation_center": geometry_translation_center,
        "geometry_translation_normal": geometry_translation_normal,
        "geometry_translation_inner_radius": geometry_translation_inner_radius,
        "geometry_translation_outer_radius": geometry_translation_outer_radius,
        "normals_recomputed": float(transformed_vertex_count > 0),
        "geometry_seed_only": 1.0,
    }


def apply_spatial_eye_uv_sampling_warp(
    *,
    eye_object: Any,
    seed_polygon: int,
    center_position: Sequence[float],
    iris_radius_3d: float,
    eye_width_3d: float,
    world_shift: Sequence[float],
    maximum_warped_loop_count: int,
) -> dict[str, float]:
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    mesh = eye_object.data
    uv_layer = mesh.uv_layers.active
    if uv_layer is None:
        raise EyeGazeUVWarpValidationError(
            f"Stacked iris object {eye_object.name!r} has no active UV layer."
        )
    loop_uvs = np.stack(
        [np.asarray(value.uv, dtype=np.float64) for value in uv_layer.data]
    )
    all_loop_vertex_ids = np.fromiter(
        (int(loop.vertex_index) for loop in mesh.loops),
        dtype=np.int64,
        count=len(mesh.loops),
    )
    polygon_loop_indices = [
        tuple(int(value) for value in polygon.loop_indices)
        for polygon in mesh.polygons
    ]
    component_polygons, component_loop_mask = uv_connected_polygon_component(
        polygon_loop_indices=polygon_loop_indices,
        loop_vertex_ids=all_loop_vertex_ids,
        loop_uvs=loop_uvs,
        seed_polygon_index=int(seed_polygon),
    )
    polygon_ids = np.flatnonzero(component_polygons)
    polygon_id_set = set(int(index) for index in polygon_ids)
    component_loops = np.flatnonzero(component_loop_mask)
    loop_vertex_ids = np.asarray(
        [int(mesh.loops[int(index)].vertex_index) for index in component_loops],
        dtype=np.int64,
    )
    positions = np.stack(
        [np.asarray(mesh.vertices[int(index)].co) for index in loop_vertex_ids]
    )
    center = np.asarray(center_position, dtype=np.float64)
    shift = np.asarray(world_shift, dtype=np.float64)
    if (
        center.shape != (3,)
        or shift.shape != (3,)
        or not np.isfinite(center).all()
        or not np.isfinite(shift).all()
        or iris_radius_3d <= 1.0e-8
        or eye_width_3d <= iris_radius_3d
    ):
        raise EyeGazeUVWarpValidationError(
            "Stacked iris UV sampling warp has invalid spatial dimensions."
        )
    if float(np.linalg.norm(shift)) <= 1.0e-12:
        return {
            "spatial_uv_warped_loop_count": 0.0,
            "spatial_uv_inner_radius_3d": 0.0,
            "spatial_uv_outer_radius_3d": 0.0,
        }
    inner_radius = max(iris_radius_3d * 1.15, eye_width_3d * 0.25)
    outer_radius = max(inner_radius * 1.6, eye_width_3d * 0.48)
    distances = np.linalg.norm(positions - center, axis=1)
    weights = np.clip(
        (outer_radius - distances) / max(outer_radius - inner_radius, 1.0e-8),
        0.0,
        1.0,
    )
    weights = weights * weights * (3.0 - 2.0 * weights)
    active = weights > 1.0e-4
    warped_loop_count = int(np.count_nonzero(active))
    if warped_loop_count < 3:
        raise EyeGazeUVWarpValidationError(
            "Stacked iris UV sampling warp found fewer than three active loops."
        )
    if warped_loop_count > int(maximum_warped_loop_count):
        raise EyeGazeUVWarpValidationError(
            "Stacked iris UV sampling warp exceeds the safe loop limit: "
            f"loops={warped_loop_count}, maximum={maximum_warped_loop_count}."
        )

    mesh.calc_loop_triangles()
    source_triangles = [
        triangle
        for triangle in mesh.loop_triangles
        if int(triangle.polygon_index) in polygon_id_set
    ]
    if not source_triangles:
        raise EyeGazeUVWarpValidationError(
            "Stacked iris UV sampling warp found no source triangles."
        )
    vertex_positions = np.stack(
        [np.asarray(vertex.co, dtype=np.float64) for vertex in mesh.vertices]
    )
    triangle_vertex_ids = np.asarray(
        [tuple(int(value) for value in triangle.vertices) for triangle in source_triangles],
        dtype=np.int64,
    )
    triangle_positions = vertex_positions[triangle_vertex_ids]
    triangle_area_squared = np.sum(
        np.cross(
            triangle_positions[:, 1] - triangle_positions[:, 0],
            triangle_positions[:, 2] - triangle_positions[:, 0],
        )
        ** 2,
        axis=1,
    )
    valid_triangles = triangle_area_squared > 1.0e-16
    source_triangles = [
        triangle
        for triangle, valid in zip(source_triangles, valid_triangles)
        if bool(valid)
    ]
    triangle_vertex_ids = triangle_vertex_ids[valid_triangles]
    triangle_positions = triangle_positions[valid_triangles]
    if not source_triangles:
        raise EyeGazeUVWarpValidationError(
            "Stacked iris UV sampling warp found no nondegenerate source triangles."
        )
    triangle_uvs = np.stack(
        [
            np.stack(
                [
                    np.asarray(uv_layer.data[int(loop_index)].uv, dtype=np.float64)
                    for loop_index in triangle.loops
                ]
            )
            for triangle in source_triangles
        ]
    )
    source_tree = BVHTree.FromPolygons(
        [Vector(value) for value in vertex_positions],
        [tuple(int(index) for index in values) for values in triangle_vertex_ids],
        all_triangles=True,
    )
    maximum_projection_distance = max(eye_width_3d * 0.12, 1.0e-6)
    remapped_uvs: dict[int, np.ndarray] = {}
    for local_index in np.flatnonzero(active):
        loop_index = int(component_loops[int(local_index)])
        desired_position = (
            positions[int(local_index)] - shift * weights[int(local_index)]
        )
        nearest = source_tree.find_nearest(Vector(desired_position))
        if nearest is None:
            raise EyeGazeUVWarpValidationError(
                "Stacked iris UV sampling warp could not project a source point."
            )
        location, _normal, triangle_index, projection_distance = nearest
        if (
            triangle_index is None
            or projection_distance is None
            or float(projection_distance) > maximum_projection_distance
        ):
            raise EyeGazeUVWarpValidationError(
                "Stacked iris UV sampling warp projected outside its source sheet."
            )
        triangle_index = int(triangle_index)
        first, second, third = triangle_positions[triangle_index]
        point = np.asarray(location, dtype=np.float64)
        edge_first = second - first
        edge_second = third - first
        relative = point - first
        dot00 = float(np.dot(edge_first, edge_first))
        dot01 = float(np.dot(edge_first, edge_second))
        dot11 = float(np.dot(edge_second, edge_second))
        dot20 = float(np.dot(relative, edge_first))
        dot21 = float(np.dot(relative, edge_second))
        denominator = dot00 * dot11 - dot01 * dot01
        if abs(denominator) <= 1.0e-16:
            raise EyeGazeUVWarpValidationError(
                "Stacked iris UV sampling warp found a degenerate source triangle."
            )
        second_weight = (dot11 * dot20 - dot01 * dot21) / denominator
        third_weight = (dot00 * dot21 - dot01 * dot20) / denominator
        barycentric = np.asarray(
            (1.0 - second_weight - third_weight, second_weight, third_weight),
            dtype=np.float64,
        )
        if not np.isfinite(barycentric).all() or float(barycentric.min()) < -1.0e-3:
            raise EyeGazeUVWarpValidationError(
                "Stacked iris UV sampling warp produced invalid barycentric weights."
            )
        remapped_uvs[loop_index] = barycentric @ triangle_uvs[triangle_index]
    for loop_index, uv in remapped_uvs.items():
        uv_layer.data[loop_index].uv = tuple(float(value) for value in uv)
    mesh.update()
    return {
        "spatial_uv_warped_loop_count": float(warped_loop_count),
        "spatial_uv_inner_radius_3d": float(inner_radius),
        "spatial_uv_outer_radius_3d": float(outer_radius),
        "spatial_uv_source_triangle_count": float(len(source_triangles)),
        "spatial_uv_source_polygon_count": float(len(polygon_ids)),
        "spatial_uv_maximum_projection_distance": float(
            maximum_projection_distance
        ),
    }


def apply_spatial_eye_texture_bake(
    *,
    eye_object: Any,
    component_polygons: Any,
    center_position: Sequence[float],
    iris_radius_3d: float,
    eye_width_3d: float,
    world_shift: Sequence[float],
    screen_projection: Any = None,
    screen_depth_direction: Optional[Sequence[float]] = None,
    screen_shift: Optional[Sequence[float]] = None,
    screen_center: Optional[Sequence[float]] = None,
    screen_iris_radius: Optional[float] = None,
    screen_iris_basis: Any = None,
    screen_eye_width: Optional[float] = None,
    screen_eye_contour: Any = None,
    screen_reference_image_path: Optional[str] = None,
    use_rendered_reference_colors: bool = False,
    bake_margin_pixels: int = 6,
    maximum_baked_texel_count: int = 500_000,
) -> dict[str, float]:
    if use_rendered_reference_colors:
        raise EyeGazeUVWarpValidationError(
            "Generated screen-render eye overlays are disabled."
        )

    import bpy
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    from mathutils.geometry import closest_point_on_tri

    mesh = eye_object.data
    uv_layer = mesh.uv_layers.active
    if uv_layer is None:
        raise EyeGazeUVWarpValidationError(
            f"Surface texture object {eye_object.name!r} has no active UV layer."
        )
    center = np.asarray(center_position, dtype=np.float64)
    shift = np.asarray(world_shift, dtype=np.float64)
    if (
        center.shape != (3,)
        or shift.shape != (3,)
        or not np.isfinite(center).all()
        or not np.isfinite(shift).all()
        or iris_radius_3d <= 1.0e-8
        or eye_width_3d <= iris_radius_3d
    ):
        raise EyeGazeUVWarpValidationError(
            "Surface texture bake has invalid spatial dimensions."
        )
    if float(np.linalg.norm(shift)) <= 1.0e-12:
        return {
            "surface_texture_baked_texel_count": 0.0,
            "surface_texture_source_triangle_count": 0.0,
        }
    if int(bake_margin_pixels) != bake_margin_pixels or bake_margin_pixels < 0:
        raise EyeGazeUVWarpValidationError(
            "Surface texture bake margin must be a nonnegative integer."
        )
    bake_margin_pixels = int(bake_margin_pixels)
    use_screen_rays = screen_projection is not None
    parsed_screen_shift = None
    parsed_screen_projection = None
    parsed_screen_depth_direction = None
    parsed_screen_center = None
    parsed_screen_iris_radius = None
    parsed_screen_iris_basis = None
    parsed_screen_eye_width = None
    parsed_screen_eye_contour = None
    if use_screen_rays:
        parsed_screen_shift = np.asarray(screen_shift, dtype=np.float64)
        parsed_screen_projection = np.asarray(screen_projection, dtype=np.float64)
        parsed_screen_depth_direction = np.asarray(
            screen_depth_direction, dtype=np.float64
        )
        parsed_screen_center = np.asarray(screen_center, dtype=np.float64)
        parsed_screen_iris_radius = float(screen_iris_radius or 0.0)
        parsed_screen_iris_basis = np.asarray(
            screen_iris_basis, dtype=np.float64
        )
        parsed_screen_eye_width = float(screen_eye_width or 0.0)
        if screen_eye_contour is not None:
            parsed_screen_eye_contour = np.asarray(
                screen_eye_contour,
                dtype=np.float64,
            )
        if (
            parsed_screen_shift.shape != (2,)
            or parsed_screen_projection.shape != (4, 2)
            or parsed_screen_depth_direction.shape != (3,)
            or not np.isfinite(parsed_screen_shift).all()
            or not np.isfinite(parsed_screen_projection).all()
            or not np.isfinite(parsed_screen_depth_direction).all()
            or parsed_screen_center.shape != (2,)
            or not np.isfinite(parsed_screen_center).all()
            or parsed_screen_iris_radius <= 1.0e-8
            or parsed_screen_iris_basis.shape != (2, 2)
            or not np.isfinite(parsed_screen_iris_basis).all()
            or abs(float(np.linalg.det(parsed_screen_iris_basis))) <= 1.0e-10
            or parsed_screen_eye_width <= parsed_screen_iris_radius
            or float(np.linalg.norm(parsed_screen_depth_direction)) <= 1.0e-8
            or (
                parsed_screen_eye_contour is not None
                and (
                    parsed_screen_eye_contour.ndim != 2
                    or parsed_screen_eye_contour.shape[0] < 6
                    or parsed_screen_eye_contour.shape[1] != 2
                    or not np.isfinite(parsed_screen_eye_contour).all()
                    or not point_in_polygon_2d(
                        parsed_screen_center,
                        parsed_screen_eye_contour,
                    )
                )
            )
        ):
            basis_determinant = (
                float(np.linalg.det(parsed_screen_iris_basis))
                if parsed_screen_iris_basis.shape == (2, 2)
                else float("nan")
            )
            contour_contains_center = (
                point_in_polygon_2d(
                    parsed_screen_center, parsed_screen_eye_contour
                )
                if parsed_screen_center.shape == (2,)
                and parsed_screen_eye_contour is not None
                and parsed_screen_eye_contour.ndim == 2
                and parsed_screen_eye_contour.shape[0] >= 3
                and parsed_screen_eye_contour.shape[1] == 2
                else None
            )
            raise EyeGazeUVWarpValidationError(
                "Screen texture bake requires a finite projection, depth direction, "
                "and iris frame: "
                f"shift_shape={parsed_screen_shift.shape}, "
                f"projection_shape={parsed_screen_projection.shape}, "
                f"depth_shape={parsed_screen_depth_direction.shape}, "
                f"center_shape={parsed_screen_center.shape}, "
                f"iris_radius={parsed_screen_iris_radius:.8f}, "
                f"iris_basis_shape={parsed_screen_iris_basis.shape}, "
                f"iris_basis_determinant={basis_determinant:.12f}, "
                f"eye_width={parsed_screen_eye_width:.8f}, "
                f"contour_contains_center={contour_contains_center}."
            )
        parsed_screen_depth_direction /= np.linalg.norm(
            parsed_screen_depth_direction
        )
        screen_inner_radius = max(
            parsed_screen_iris_radius * 1.15,
            parsed_screen_eye_width * 0.26,
        )
        screen_outer_radius = max(
            screen_inner_radius * 1.55,
            parsed_screen_eye_width * 0.48,
        )
        screen_iris_bake_scale = 1.05 if use_rendered_reference_colors else 1.35
        screen_iris_cleanup_scale = screen_iris_bake_scale * (
            1.15 if use_rendered_reference_colors else 1.18
        )
        screen_iris_feather_scale = screen_iris_bake_scale * 0.12
        screen_iris_bake_radius = (
            parsed_screen_iris_radius * screen_iris_bake_scale
        )
        screen_iris_cleanup_radius = (
            parsed_screen_iris_radius * screen_iris_cleanup_scale
        )
        screen_iris_feather_radius = (
            parsed_screen_iris_radius * screen_iris_feather_scale
        )

        def screen_iris_normalized_distance(
            screen_position: Any,
            iris_center: Any,
        ) -> float:
            coordinates = np.linalg.solve(
                parsed_screen_iris_basis,
                np.asarray(screen_position, dtype=np.float64)
                - np.asarray(iris_center, dtype=np.float64),
            )
            return float(np.linalg.norm(coordinates))

        def screen_iris_mask_weight(
            screen_position: Any,
            iris_center: Any,
            radius_scale: float,
        ) -> float:
            distance = screen_iris_normalized_distance(
                screen_position, iris_center
            )
            weight = float(
                np.clip(
                    (
                        radius_scale + screen_iris_feather_scale - distance
                    )
                    / (2.0 * screen_iris_feather_scale),
                    0.0,
                    1.0,
                )
            )
            return weight * weight * (3.0 - 2.0 * weight)

        screen_capsule_minimum = (
            np.minimum(
                parsed_screen_center,
                parsed_screen_center + parsed_screen_shift,
            )
            - screen_outer_radius
        )
        screen_capsule_maximum = (
            np.maximum(
                parsed_screen_center,
                parsed_screen_center + parsed_screen_shift,
            )
            + screen_outer_radius
        )
    elif use_rendered_reference_colors:
        raise EyeGazeUVWarpValidationError(
            "Rendered-reference eye bake requires screen-space calibration."
        )

    def base_color_image(material_index: int) -> Any:
        if not 0 <= material_index < len(eye_object.material_slots):
            return None
        material = eye_object.material_slots[material_index].material
        if material is None or not material.use_nodes or material.node_tree is None:
            return None
        for node in material.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            base_color = node.inputs.get("Base Color")
            if base_color is None:
                continue
            for link in base_color.links:
                image_node = link.from_node
                if image_node.type == "TEX_IMAGE" and image_node.image is not None:
                    return image_node.image
        return None

    mesh.calc_loop_triangles()
    vertex_positions = np.stack(
        [np.asarray(vertex.co, dtype=np.float64) for vertex in mesh.vertices]
    )
    if use_screen_rays:
        all_triangles = list(mesh.loop_triangles)
        all_triangle_vertex_ids = np.asarray(
            [
                tuple(int(value) for value in triangle.vertices)
                for triangle in all_triangles
            ],
            dtype=np.int64,
        )
        all_triangle_positions = vertex_positions[all_triangle_vertex_ids]
        neighborhood_radius = eye_width_3d * 0.70
        triangle_minimum = all_triangle_positions.min(axis=1)
        triangle_maximum = all_triangle_positions.max(axis=1)
        box_distance = np.maximum(
            np.maximum(triangle_minimum - center, center - triangle_maximum),
            0.0,
        )
        neighborhood_mask = (
            np.linalg.norm(box_distance, axis=1) <= neighborhood_radius
        )
        raw_triangles = [
            triangle
            for triangle, keep in zip(all_triangles, neighborhood_mask)
            if bool(keep)
        ]
        triangle_vertex_ids = all_triangle_vertex_ids[neighborhood_mask]
        triangle_positions = all_triangle_positions[neighborhood_mask]
        if len(raw_triangles) > 100_000:
            raise EyeGazeUVWarpValidationError(
                "Screen texture bake eye neighborhood exceeds the safe triangle limit: "
                f"triangles={len(raw_triangles)}."
            )
    else:
        selected_polygon_ids = set(
            int(index) for index in np.flatnonzero(component_polygons)
        )
        raw_triangles = [
            triangle
            for triangle in mesh.loop_triangles
            if int(triangle.polygon_index) in selected_polygon_ids
        ]
        triangle_vertex_ids = np.asarray(
            [
                tuple(int(value) for value in triangle.vertices)
                for triangle in raw_triangles
            ],
            dtype=np.int64,
        )
        triangle_positions = vertex_positions[triangle_vertex_ids]
    triangle_area_squared = np.sum(
        np.cross(
            triangle_positions[:, 1] - triangle_positions[:, 0],
            triangle_positions[:, 2] - triangle_positions[:, 0],
        )
        ** 2,
        axis=1,
    )
    valid = triangle_area_squared > 1.0e-16
    source_triangles = [
        triangle for triangle, keep in zip(raw_triangles, valid) if bool(keep)
    ]
    triangle_vertex_ids = triangle_vertex_ids[valid]
    triangle_positions = triangle_positions[valid]
    if not source_triangles:
        raise EyeGazeUVWarpValidationError(
            "Surface texture bake found no nondegenerate source triangles."
        )
    triangle_uvs = np.stack(
        [
            np.stack(
                [
                    np.asarray(uv_layer.data[int(loop_index)].uv, dtype=np.float64)
                    for loop_index in triangle.loops
                ]
            )
            for triangle in source_triangles
        ]
    )
    triangle_images = [
        base_color_image(int(mesh.polygons[int(triangle.polygon_index)].material_index))
        for triangle in source_triangles
    ]
    source_tree = BVHTree.FromPolygons(
        [Vector(value) for value in vertex_positions],
        [tuple(int(index) for index in values) for values in triangle_vertex_ids],
        all_triangles=True,
    )
    iris_source_triangle_indices = np.asarray(
        [
            index
            for index, triangle in enumerate(source_triangles)
            if component_polygons[int(triangle.polygon_index)]
        ],
        dtype=np.int64,
    )
    if len(iris_source_triangle_indices) < 8:
        raise EyeGazeUVWarpValidationError(
            "Surface texture bake iris component has too few source triangles."
        )
    iris_source_tree = BVHTree.FromPolygons(
        [Vector(value) for value in vertex_positions],
        [
            tuple(int(index) for index in triangle_vertex_ids[source_index])
            for source_index in iris_source_triangle_indices
        ],
        all_triangles=True,
    )
    visible_polygon_ids = None
    if use_screen_rays:
        homogeneous_positions = np.concatenate(
            (
                triangle_positions,
                np.ones((*triangle_positions.shape[:2], 1), dtype=np.float64),
            ),
            axis=-1,
        )
        triangle_screen_positions = homogeneous_positions @ parsed_screen_projection
        polygon_triangle_indices: dict[int, list[int]] = {}
        for triangle_index, triangle in enumerate(source_triangles):
            polygon_triangle_indices.setdefault(
                int(triangle.polygon_index), []
            ).append(triangle_index)
        object_tree = BVHTree.FromObject(
            eye_object,
            bpy.context.evaluated_depsgraph_get(),
        )
        projection_inverse = np.linalg.pinv(parsed_screen_projection[:3])
        projection_offset = parsed_screen_projection[3]
        ray_extent = max(eye_width_3d * 50.0, 1.0)

        def screen_object_hit(screen_position: Any) -> Any:
            ray_base = (
                np.asarray(screen_position, dtype=np.float64) - projection_offset
            ) @ projection_inverse
            ray_origin = ray_base + parsed_screen_depth_direction * ray_extent
            return object_tree.ray_cast(
                Vector(ray_origin),
                Vector(-parsed_screen_depth_direction),
                ray_extent * 2.0,
            )

        raster_resolution = 1024
        raster_minimum = np.floor(
            np.clip(screen_capsule_minimum, 0.0, 1.0)
            * (raster_resolution - 1)
        ).astype(int)
        raster_maximum = np.ceil(
            np.clip(screen_capsule_maximum, 0.0, 1.0)
            * (raster_resolution - 1)
        ).astype(int)
        visible_polygon_ids = set()
        for row in range(raster_minimum[1], raster_maximum[1] + 1):
            for column in range(raster_minimum[0], raster_maximum[0] + 1):
                screen_position = np.asarray(
                    (column, row), dtype=np.float64
                ) / (raster_resolution - 1)
                hit = screen_object_hit(screen_position)
                polygon_index = hit[2] if hit is not None else None
                if (
                    polygon_index is not None
                    and int(polygon_index) in polygon_triangle_indices
                ):
                    visible_polygon_ids.add(int(polygon_index))
        if not visible_polygon_ids:
            raise EyeGazeUVWarpValidationError(
                "Screen texture bake found no front-visible eye triangles."
            )

        def screen_surface_hit(screen_position: Any) -> Any:
            hit = screen_object_hit(screen_position)
            if hit is None or hit[0] is None or hit[2] is None:
                return None
            location, normal, polygon_index, distance = hit
            for candidate_index in polygon_triangle_indices.get(
                int(polygon_index), ()
            ):
                weights = barycentric_3d(
                    np.asarray(location, dtype=np.float64),
                    triangle_positions[candidate_index],
                )
                if (
                    weights is None
                    or not np.isfinite(weights).all()
                    or float(weights.min()) < -1.0e-5
                ):
                    continue
                return location, normal, candidate_index, distance
            return None

        def screen_iris_surface_hit(screen_position: Any) -> Any:
            ray_base = (
                np.asarray(screen_position, dtype=np.float64) - projection_offset
            ) @ projection_inverse
            ray_origin = ray_base + parsed_screen_depth_direction * ray_extent
            hit = iris_source_tree.ray_cast(
                Vector(ray_origin),
                Vector(-parsed_screen_depth_direction),
                ray_extent * 2.0,
            )
            if hit is None or hit[0] is None or hit[2] is None:
                return None
            location, normal, component_index, distance = hit
            source_index = int(iris_source_triangle_indices[int(component_index)])
            return location, normal, source_index, distance

    images = []
    for image in triangle_images:
        if image is not None and image not in images:
            images.append(image)
    if not images:
        raise EyeGazeUVWarpValidationError(
            "Surface texture bake found no packed base-color image."
        )
    image_buffers = {}
    for image in images:
        width, height = int(image.size[0]), int(image.size[1])
        if width < 2 or height < 2:
            raise EyeGazeUVWarpValidationError(
                f"Packed texture {image.name!r} has invalid dimensions."
            )
        flat = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(flat)
        source = flat.reshape(height, width, 4)
        image_buffers[image.name] = (
            image,
            source.copy(),
            source.copy(),
            np.zeros((height, width), dtype=bool),
        )
    baked_polygon_ids_by_image = {image.name: set() for image in images}
    screen_reference_buffer = None
    screen_reference_colorspace = None
    if use_screen_rays and screen_reference_image_path:
        reference_path = Path(screen_reference_image_path)
        if not reference_path.is_file():
            raise EyeGazeUVWarpValidationError(
                f"Rendered gaze reference does not exist: {reference_path}."
            )
        reference_image = bpy.data.images.load(
            str(reference_path),
            check_existing=True,
        )
        screen_reference_colorspace = reference_image.colorspace_settings.name
        reference_width = int(reference_image.size[0])
        reference_height = int(reference_image.size[1])
        if reference_width < 2 or reference_height < 2:
            raise EyeGazeUVWarpValidationError(
                "Rendered gaze reference has invalid dimensions."
            )
        reference_flat = np.empty(
            reference_width * reference_height * 4,
            dtype=np.float32,
        )
        reference_image.pixels.foreach_get(reference_flat)
        screen_reference_buffer = reference_flat.reshape(
            reference_height,
            reference_width,
            4,
        )
    if use_rendered_reference_colors and screen_reference_buffer is None:
        raise EyeGazeUVWarpValidationError(
            "Rendered-reference eye bake requires the textured landmark render."
        )

    def barycentric_2d(points: Any, triangle: Any) -> Any:
        first, second, third = triangle
        edge_first = second - first
        edge_second = third - first
        relative = points - first
        denominator = (
            edge_first[0] * edge_second[1]
            - edge_first[1] * edge_second[0]
        )
        if abs(float(denominator)) <= 1.0e-12:
            return None
        second_weights = (
            relative[..., 0] * edge_second[1]
            - relative[..., 1] * edge_second[0]
        ) / denominator
        third_weights = (
            edge_first[0] * relative[..., 1]
            - edge_first[1] * relative[..., 0]
        ) / denominator
        return np.stack(
            (1.0 - second_weights - third_weights, second_weights, third_weights),
            axis=-1,
        )

    def barycentric_3d(point: Any, triangle: Any) -> Any:
        first, second, third = triangle
        edge_first = second - first
        edge_second = third - first
        relative = point - first
        dot00 = float(np.dot(edge_first, edge_first))
        dot01 = float(np.dot(edge_first, edge_second))
        dot11 = float(np.dot(edge_second, edge_second))
        dot20 = float(np.dot(relative, edge_first))
        dot21 = float(np.dot(relative, edge_second))
        denominator = dot00 * dot11 - dot01 * dot01
        if abs(denominator) <= 1.0e-16:
            return None
        second_weight = (dot11 * dot20 - dot01 * dot21) / denominator
        third_weight = (dot00 * dot21 - dot01 * dot20) / denominator
        return np.asarray(
            (1.0 - second_weight - third_weight, second_weight, third_weight),
            dtype=np.float64,
        )

    def sample_image(image: Any, uv: Any) -> Any:
        _image, source, _output, _mask = image_buffers[image.name]
        height, width = source.shape[:2]
        x = float(np.clip(uv[0], 0.0, 1.0)) * (width - 1)
        y = float(np.clip(uv[1], 0.0, 1.0)) * (height - 1)
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = min(x0 + 1, width - 1)
        y1 = min(y0 + 1, height - 1)
        tx = x - x0
        ty = y - y0
        return (
            source[y0, x0] * (1.0 - tx) * (1.0 - ty)
            + source[y0, x1] * tx * (1.0 - ty)
            + source[y1, x0] * (1.0 - tx) * ty
            + source[y1, x1] * tx * ty
        )

    def screen_reference_color(screen_position: Any) -> Any:
        if screen_reference_buffer is None:
            return None
        height, width = screen_reference_buffer.shape[:2]
        x = float(np.clip(screen_position[0], 0.0, 1.0)) * (width - 1)
        y = (1.0 - float(np.clip(screen_position[1], 0.0, 1.0))) * (
            height - 1
        )
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = min(x0 + 1, width - 1)
        y1 = min(y0 + 1, height - 1)
        tx = x - x0
        ty = y - y0
        return (
            screen_reference_buffer[y0, x0] * (1.0 - tx) * (1.0 - ty)
            + screen_reference_buffer[y0, x1] * tx * (1.0 - ty)
            + screen_reference_buffer[y1, x0] * (1.0 - tx) * ty
            + screen_reference_buffer[y1, x1] * tx * ty
        )

    def screen_hit_color(hit: Any) -> Any:
        if hit is None:
            return None
        location, _normal, source_index, _projection_distance = hit
        if source_index is None:
            return None
        source_index = int(source_index)
        source_weights = barycentric_3d(
            np.asarray(location, dtype=np.float64),
            triangle_positions[source_index],
        )
        source_image = triangle_images[source_index]
        if (
            source_weights is None
            or source_image is None
            or not np.isfinite(source_weights).all()
            or float(source_weights.min()) < -1.0e-3
        ):
            return None
        source_uv = source_weights @ triangle_uvs[source_index]
        return sample_image(source_image, source_uv)

    sclera_source_positions = np.empty((0, 2), dtype=np.float64)
    sclera_source_colors = np.empty((0, 4), dtype=np.float64)
    sclera_candidate_count = 0
    iris_sprite_source_positions = np.empty((0, 2), dtype=np.float64)
    iris_sprite_source_colors = np.empty((0, 4), dtype=np.float64)
    iris_sprite_candidate_count = 0
    if use_screen_rays and parsed_screen_eye_contour is not None:
        contour_minimum = parsed_screen_eye_contour.min(axis=0)
        contour_maximum = parsed_screen_eye_contour.max(axis=0)
        boundary_inset = max(
            parsed_screen_eye_width * 0.018,
            parsed_screen_iris_radius * 0.08,
        )
        def contour_boundary_distance(point: Any) -> float:
            distances = []
            for index, first in enumerate(parsed_screen_eye_contour):
                second = parsed_screen_eye_contour[
                    (index + 1) % len(parsed_screen_eye_contour)
                ]
                edge = second - first
                denominator = float(np.dot(edge, edge))
                if denominator <= 1.0e-16:
                    continue
                parameter = float(
                    np.clip(np.dot(point - first, edge) / denominator, 0.0, 1.0)
                )
                distances.append(float(np.linalg.norm(point - (first + parameter * edge))))
            return min(distances, default=0.0)

        raw_sclera_candidates = []
        for screen_y in np.linspace(contour_minimum[1], contour_maximum[1], 19):
            for screen_x in np.linspace(contour_minimum[0], contour_maximum[0], 37):
                candidate = np.asarray((screen_x, screen_y), dtype=np.float64)
                if not point_in_polygon_2d(candidate, parsed_screen_eye_contour):
                    continue
                if contour_boundary_distance(candidate) < boundary_inset:
                    continue
                if screen_iris_normalized_distance(
                    candidate, parsed_screen_center
                ) < screen_iris_cleanup_scale:
                    continue
                hit = screen_surface_hit(candidate)
                reference_color = screen_reference_color(candidate)
                source_color = screen_hit_color(hit)
                if source_color is None or not np.isfinite(source_color).all():
                    continue
                segmentation_color = reference_color
                if segmentation_color is None:
                    segmentation_color = source_color
                if not np.isfinite(segmentation_color).all():
                    continue
                rgb = np.clip(
                    np.asarray(segmentation_color[:3], dtype=np.float64),
                    0.0,
                    1.0,
                )
                luminance = float(np.dot(rgb, (0.2126, 0.7152, 0.0722)))
                chroma = float(rgb.max() - rgb.min())
                raw_sclera_candidates.append(
                    (
                        candidate,
                        np.asarray(source_color, dtype=np.float64),
                        luminance - 0.35 * chroma,
                        np.asarray(segmentation_color, dtype=np.float64),
                    )
                )
        sclera_candidate_count = len(raw_sclera_candidates)
        if sclera_candidate_count < 8:
            raise EyeGazeUVWarpValidationError(
                "Screen texture bake found too few visible sclera samples: "
                f"count={sclera_candidate_count}."
            )
        retained_candidates = []
        for sign in (-1.0, 1.0):
            side_candidates = [
                value
                for value in raw_sclera_candidates
                if (value[0][0] - parsed_screen_center[0]) * sign >= 0.0
            ]
            if len(side_candidates) < 3:
                raise EyeGazeUVWarpValidationError(
                    "Screen texture bake needs sclera samples on both sides of the iris."
                )
            side_scores = np.asarray(
                [value[2] for value in side_candidates], dtype=np.float64
            )
            score_threshold = float(np.quantile(side_scores, 0.65))
            retained_candidates.extend(
                value for value in side_candidates if value[2] >= score_threshold
            )
        sclera_source_positions = np.stack(
            [value[0] for value in retained_candidates]
        )
        sclera_source_colors = np.stack(
            [value[1] for value in retained_candidates]
        )

        def screen_sclera_color(target_screen: Any) -> Any:
            distances = np.linalg.norm(
                sclera_source_positions - np.asarray(target_screen, dtype=np.float64),
                axis=1,
            )
            neighbor_count = min(8, len(distances))
            nearest_indices = np.argpartition(
                distances,
                neighbor_count - 1,
            )[:neighbor_count]
            local_distances = distances[nearest_indices]
            regularization = max(parsed_screen_eye_width * 0.025, 1.0e-8)
            weights = 1.0 / (local_distances**2 + regularization**2)
            weights /= weights.sum()
            return weights @ sclera_source_colors[nearest_indices]

        sclera_segmentation_colors = np.stack(
            [value[3] for value in retained_candidates]
        )

        def screen_reference_sclera_color(target_screen: Any) -> Any:
            distances = np.linalg.norm(
                sclera_source_positions
                - np.asarray(target_screen, dtype=np.float64),
                axis=1,
            )
            neighbor_count = min(8, len(distances))
            nearest_indices = np.argpartition(
                distances,
                neighbor_count - 1,
            )[:neighbor_count]
            local_distances = distances[nearest_indices]
            regularization = max(parsed_screen_eye_width * 0.025, 1.0e-8)
            weights = 1.0 / (local_distances**2 + regularization**2)
            weights /= weights.sum()
            return weights @ sclera_segmentation_colors[nearest_indices]

        sclera_luminances = sclera_segmentation_colors[:, :3] @ np.asarray(
            (0.2126, 0.7152, 0.0722), dtype=np.float64
        )
        bright_sclera_threshold = float(np.quantile(sclera_luminances, 0.70))
        bright_sclera_reference = np.median(
            sclera_segmentation_colors[
                sclera_luminances >= bright_sclera_threshold
            ],
            axis=0,
        )
        sclera_luminance_weights = np.asarray(
            (0.2126, 0.7152, 0.0722),
            dtype=np.float64,
        )
        bright_sclera_luminance = float(
            np.dot(bright_sclera_reference[:3], sclera_luminance_weights)
        )
        if bright_sclera_luminance < 0.92:
            bright_sclera_reference[:3] = np.clip(
                bright_sclera_reference[:3]
                + (0.92 - bright_sclera_luminance),
                0.0,
                1.0,
            )
        sclera_fit_scale = max(parsed_screen_eye_width, 1.0e-8)
        sclera_fit_coordinates = (
            sclera_source_positions - parsed_screen_center
        ) / sclera_fit_scale
        sclera_fit_design = np.column_stack(
            (
                sclera_fit_coordinates,
                np.ones(len(sclera_fit_coordinates), dtype=np.float64),
            )
        )
        sclera_fit_coefficients = np.linalg.lstsq(
            sclera_fit_design,
            sclera_segmentation_colors,
            rcond=None,
        )[0]
        sclera_fit_predictions = sclera_fit_design @ sclera_fit_coefficients
        sclera_fit_rms = float(
            np.sqrt(
                np.mean(
                    (sclera_fit_predictions[:, :3]
                    - sclera_segmentation_colors[:, :3])
                    ** 2
                )
            )
        )
        sclera_fit_minimum = np.quantile(
            sclera_segmentation_colors, 0.05, axis=0
        )
        sclera_fit_maximum = np.quantile(
            sclera_segmentation_colors, 0.95, axis=0
        )

        def screen_fitted_sclera_color(target_screen: Any) -> Any:
            coordinate = (
                np.asarray(target_screen, dtype=np.float64)
                - parsed_screen_center
            ) / sclera_fit_scale
            prediction = (
                np.asarray((coordinate[0], coordinate[1], 1.0))
                @ sclera_fit_coefficients
            )
            return np.clip(
                prediction,
                sclera_fit_minimum,
                sclera_fit_maximum,
            )

        def screen_color_is_iris(color: Any, screen_position: Any) -> bool:
            rgb = np.clip(
                np.asarray(color[:3], dtype=np.float64),
                0.0,
                1.0,
            )
            local_sclera = screen_fitted_sclera_color(screen_position)[:3]
            luminance_weights = np.asarray(
                (0.2126, 0.7152, 0.0722),
                dtype=np.float64,
            )
            luminance = float(np.dot(rgb, luminance_weights))
            sclera_luminance = float(
                np.dot(local_sclera, luminance_weights)
            )
            color_distance = float(np.linalg.norm(rgb - local_sclera))
            return bool(
                luminance <= sclera_luminance - 0.12
                or color_distance >= 0.20
            )

        raw_iris_sprite_candidates = []
        for screen_y in np.linspace(
            parsed_screen_center[1] - screen_iris_bake_radius,
            parsed_screen_center[1] + screen_iris_bake_radius,
            35,
        ):
            for screen_x in np.linspace(
                parsed_screen_center[0] - screen_iris_bake_radius,
                parsed_screen_center[0] + screen_iris_bake_radius,
                35,
            ):
                candidate = np.asarray((screen_x, screen_y), dtype=np.float64)
                if (
                    screen_iris_normalized_distance(
                        candidate, parsed_screen_center
                    )
                    > screen_iris_bake_scale
                    or not point_in_polygon_2d(
                        candidate,
                        parsed_screen_eye_contour,
                    )
                    or contour_boundary_distance(candidate) < boundary_inset
                ):
                    continue
                hit = screen_surface_hit(candidate)
                if hit is None or hit[2] is None:
                    continue
                source_index = int(hit[2])
                source_polygon = int(
                    source_triangles[source_index].polygon_index
                )
                if (
                    not component_polygons[source_polygon]
                    and screen_iris_normalized_distance(
                        candidate, parsed_screen_center
                    )
                    > 0.95
                ):
                    continue
                reference_color = screen_reference_color(candidate)
                source_color = (
                    reference_color
                    if use_rendered_reference_colors
                    else screen_hit_color(hit)
                )
                if source_color is None or not np.isfinite(source_color).all():
                    continue
                segmentation_color = reference_color
                if segmentation_color is None:
                    segmentation_color = source_color
                if not np.isfinite(segmentation_color).all():
                    continue
                if not screen_color_is_iris(segmentation_color, candidate):
                    continue
                raw_iris_sprite_candidates.append(
                    (candidate, np.asarray(source_color, dtype=np.float64))
                )
        iris_sprite_candidate_count = len(raw_iris_sprite_candidates)
        if iris_sprite_candidate_count < 16:
            raise EyeGazeUVWarpValidationError(
                "Screen texture bake found too few visible iris sprite samples: "
                f"count={iris_sprite_candidate_count}."
            )
        iris_sprite_source_positions = np.stack(
            [value[0] for value in raw_iris_sprite_candidates]
        )
        iris_sprite_source_colors = np.stack(
            [value[1] for value in raw_iris_sprite_candidates]
        )

        def screen_iris_sprite_color(source_screen: Any) -> Any:
            distances = np.linalg.norm(
                iris_sprite_source_positions
                - np.asarray(source_screen, dtype=np.float64),
                axis=1,
            )
            neighbor_count = min(8, len(distances))
            nearest_indices = np.argpartition(
                distances,
                neighbor_count - 1,
            )[:neighbor_count]
            local_distances = distances[nearest_indices]
            regularization = max(screen_iris_bake_radius * 0.025, 1.0e-8)
            weights = 1.0 / (local_distances**2 + regularization**2)
            weights /= weights.sum()
            return weights @ iris_sprite_source_colors[nearest_indices]

        def screen_reference_iris_color(source_screen: Any) -> Any:
            if screen_reference_buffer is None:
                return None
            source_screen = np.asarray(source_screen, dtype=np.float64)
            if (
                screen_iris_normalized_distance(
                    source_screen, parsed_screen_center
                )
                > screen_iris_bake_scale
                or not point_in_polygon_2d(
                    source_screen,
                    parsed_screen_eye_contour,
                )
                or contour_boundary_distance(source_screen) < boundary_inset
            ):
                return None
            color = screen_reference_color(source_screen)
            if color is None or not np.isfinite(color).all():
                return None
            if not screen_color_is_iris(color, source_screen):
                return None
            return np.asarray(color, dtype=np.float64)

    screen_overlay_report = None
    inner_radius = max(iris_radius_3d * 1.15, eye_width_3d * 0.25)
    outer_radius = max(inner_radius * 1.6, eye_width_3d * 0.48)
    if use_screen_rays:
        outer_radius = max(outer_radius, eye_width_3d * 0.65)
    maximum_projection_distance = max(eye_width_3d * 0.12, 1.0e-6)
    baked_texels = 0
    intersecting_triangles = 0
    aperture_target_texels_skipped = 0
    aperture_sclera_source_samples = 0
    direct_reference_iris_source_samples = 0
    sprite_iris_source_samples = 0
    for triangle_index, triangle in enumerate(source_triangles):
        target_image = triangle_images[triangle_index]
        if target_image is None:
            continue
        if (
            visible_polygon_ids is not None
            and int(triangle.polygon_index) not in visible_polygon_ids
        ):
            continue
        positions = triangle_positions[triangle_index]
        if use_screen_rays:
            screen_triangle = triangle_screen_positions[triangle_index]
            if np.any(screen_triangle.max(axis=0) < screen_capsule_minimum) or np.any(
                screen_triangle.min(axis=0) > screen_capsule_maximum
            ):
                continue
        closest_position = closest_point_on_tri(
            Vector(center),
            Vector(positions[0]),
            Vector(positions[1]),
            Vector(positions[2]),
        )
        if float(
            np.linalg.norm(np.asarray(closest_position, dtype=np.float64) - center)
        ) > outer_radius:
            continue
        intersecting_triangles += 1
        _image, _source, output, baked_mask = image_buffers[target_image.name]
        height, width = output.shape[:2]
        uv_pixels = triangle_uvs[triangle_index] * np.asarray(
            (width - 1, height - 1), dtype=np.float64
        )
        minimum = np.maximum(np.floor(uv_pixels.min(axis=0)).astype(int), 0)
        maximum = np.minimum(
            np.ceil(uv_pixels.max(axis=0)).astype(int),
            np.asarray((width - 1, height - 1)),
        )
        if np.any(maximum < minimum):
            continue
        columns, rows = np.meshgrid(
            np.arange(minimum[0], maximum[0] + 1),
            np.arange(minimum[1], maximum[1] + 1),
        )
        points = np.stack((columns, rows), axis=-1).astype(np.float64)
        barycentric = barycentric_2d(points, uv_pixels)
        if barycentric is None:
            continue
        inside = np.all(barycentric >= -1.0e-6, axis=-1)
        for row_index, column_index in np.argwhere(inside):
            weights = barycentric[row_index, column_index]
            target_position = weights @ positions
            translated_iris_weight = 0.0
            vacated_iris_weight = 0.0
            if use_screen_rays:
                target_screen = (
                    np.append(target_position, 1.0) @ parsed_screen_projection
                )
                if (
                    parsed_screen_eye_contour is not None
                    and not point_in_polygon_2d(
                        target_screen,
                        parsed_screen_eye_contour,
                    )
                ):
                    aperture_target_texels_skipped += 1
                    continue
                neutral_iris_weight = screen_iris_mask_weight(
                    target_screen,
                    parsed_screen_center,
                    screen_iris_cleanup_scale,
                )
                translated_iris_weight = screen_iris_mask_weight(
                    target_screen,
                    parsed_screen_center + parsed_screen_shift,
                    screen_iris_bake_scale,
                )
                if use_rendered_reference_colors:
                    # The overlay carries the translated iris. This packed-image
                    # bake only restores sclera across the old iris footprint.
                    vacated_iris_weight = neutral_iris_weight
                    translated_iris_weight = 0.0
                else:
                    vacated_iris_weight = neutral_iris_weight * (
                        1.0 - translated_iris_weight
                    )
                if max(translated_iris_weight, vacated_iris_weight) <= 1.0e-4:
                    continue
                spatial_weight = 1.0
            else:
                distance = float(np.linalg.norm(target_position - center))
                weight_inner_radius = inner_radius
                weight_outer_radius = outer_radius
                spatial_weight = float(
                    np.clip(
                        (weight_outer_radius - distance)
                        / max(weight_outer_radius - weight_inner_radius, 1.0e-8),
                        0.0,
                        1.0,
                    )
                )
                spatial_weight = spatial_weight * spatial_weight * (
                    3.0 - 2.0 * spatial_weight
                )
            if spatial_weight <= 1.0e-4:
                continue
            if use_screen_rays:
                fallback_color = None
                if vacated_iris_weight > 1.0e-4:
                    fallback_color = screen_sclera_color(target_screen)
                    aperture_sclera_source_samples += 1
                translated_color = None
                if (
                    not use_rendered_reference_colors
                    and translated_iris_weight > 1.0e-4
                ):
                    desired_screen = target_screen - parsed_screen_shift
                    translated_color = screen_iris_sprite_color(desired_screen)
                    sprite_iris_source_samples += 1
                if (
                    translated_iris_weight <= 1.0e-4
                    and vacated_iris_weight <= 1.0e-4
                ):
                    continue
                row = int(rows[row_index, column_index])
                column = int(columns[row_index, column_index])
                composite = np.asarray(output[row, column], dtype=np.float64)
                if fallback_color is not None:
                    composite = (
                        composite * (1.0 - vacated_iris_weight)
                        + fallback_color * vacated_iris_weight
                    )
                if translated_color is not None:
                    composite = (
                        composite * (1.0 - translated_iris_weight)
                        + translated_color * translated_iris_weight
                    )
                output[row, column] = composite
            else:
                desired_position = target_position - shift * spatial_weight
                nearest = source_tree.find_nearest(Vector(desired_position))
                if nearest is None:
                    continue
                location, _normal, source_index, projection_distance = nearest
                if (
                    source_index is None
                    or projection_distance is None
                    or float(projection_distance) > maximum_projection_distance
                ):
                    continue
                source_index = int(source_index)
                source_weights = barycentric_3d(
                    np.asarray(location, dtype=np.float64),
                    triangle_positions[source_index],
                )
                source_image = triangle_images[source_index]
                if (
                    source_weights is None
                    or source_image is None
                    or not np.isfinite(source_weights).all()
                    or float(source_weights.min()) < -1.0e-3
                ):
                    continue
                source_uv = source_weights @ triangle_uvs[source_index]
                row = int(rows[row_index, column_index])
                column = int(columns[row_index, column_index])
                output[row, column] = sample_image(source_image, source_uv)
            baked_mask[row, column] = True
            baked_polygon_ids_by_image[target_image.name].add(
                int(triangle.polygon_index)
            )
            baked_texels += 1
            if baked_texels > int(maximum_baked_texel_count):
                raise EyeGazeUVWarpValidationError(
                    "Surface texture bake exceeds the safe texel limit: "
                    f"texels={baked_texels}, maximum={maximum_baked_texel_count}."
                )

    if baked_texels < 16:
        raise EyeGazeUVWarpValidationError(
            f"Surface texture bake changed only {baked_texels} texels."
        )
    isolated_material_count = 0
    isolated_polygon_ids = set()
    for image, _source, output, baked_mask in image_buffers.values():
        for _step in range(bake_margin_pixels):
            padded_mask = np.pad(
                baked_mask,
                1,
                mode="constant",
                constant_values=False,
            )
            padded_output = np.pad(
                output,
                ((1, 1), (1, 1), (0, 0)),
                mode="edge",
            )
            color_sum = np.zeros_like(output)
            neighbor_count = np.zeros(baked_mask.shape, dtype=np.float32)
            for row_offset in range(3):
                for column_offset in range(3):
                    if row_offset == 1 and column_offset == 1:
                        continue
                    neighbor_mask = padded_mask[
                        row_offset : row_offset + baked_mask.shape[0],
                        column_offset : column_offset + baked_mask.shape[1],
                    ]
                    color_sum += (
                        padded_output[
                            row_offset : row_offset + baked_mask.shape[0],
                            column_offset : column_offset + baked_mask.shape[1],
                        ]
                        * neighbor_mask[..., None]
                    )
                    neighbor_count += neighbor_mask
            margin = (~baked_mask) & (neighbor_count > 0.0)
            if not np.any(margin):
                break
            output[margin] = color_sum[margin] / neighbor_count[margin][:, None]
            baked_mask[margin] = True
        if use_screen_rays:
            polygon_ids = baked_polygon_ids_by_image[image.name]
            if not polygon_ids:
                continue
            baked_image = bpy.data.images.new(
                name=f"{image.name}_gaze_bake",
                width=int(image.size[0]),
                height=int(image.size[1]),
                alpha=True,
                float_buffer=bool(image.is_float),
            )
            try:
                baked_image.colorspace_settings.name = image.colorspace_settings.name
            except TypeError:
                pass
            baked_image.alpha_mode = image.alpha_mode
            baked_image.pixels.foreach_set(output.reshape(-1))
            baked_image.update()
            baked_image.pack()

            isolated_material_indices = {}
            for polygon_id in sorted(polygon_ids):
                polygon = mesh.polygons[polygon_id]
                source_material_index = int(polygon.material_index)
                isolated_material_index = isolated_material_indices.get(
                    source_material_index
                )
                if isolated_material_index is None:
                    if not 0 <= source_material_index < len(eye_object.material_slots):
                        raise EyeGazeUVWarpValidationError(
                            "Screen texture bake polygon has no source material."
                        )
                    source_material = eye_object.material_slots[
                        source_material_index
                    ].material
                    if source_material is None:
                        raise EyeGazeUVWarpValidationError(
                            "Screen texture bake polygon has an empty material slot."
                        )
                    isolated_material = source_material.copy()
                    replaced_image = False
                    if (
                        isolated_material.use_nodes
                        and isolated_material.node_tree is not None
                    ):
                        for node in isolated_material.node_tree.nodes:
                            if node.type == "TEX_IMAGE" and node.image is image:
                                node.image = baked_image
                                replaced_image = True
                    if not replaced_image:
                        raise EyeGazeUVWarpValidationError(
                            "Screen texture bake could not isolate the packed image material."
                        )
                    eye_object.data.materials.append(isolated_material)
                    isolated_material_index = len(eye_object.data.materials) - 1
                    isolated_material_indices[
                        source_material_index
                    ] = isolated_material_index
                    isolated_material_count += 1
                polygon.material_index = isolated_material_index
                isolated_polygon_ids.add(polygon_id)
        else:
            image.pixels.foreach_set(output.reshape(-1))
            image.update()
            image.pack()
    report = {
        "surface_texture_baked_texel_count": float(baked_texels),
        "surface_texture_source_triangle_count": float(len(source_triangles)),
        "surface_texture_iris_source_triangle_count": float(
            len(iris_source_triangle_indices)
        ),
        "surface_texture_intersecting_triangle_count": float(
            intersecting_triangles
        ),
        "surface_texture_inner_radius_3d": float(inner_radius),
        "surface_texture_outer_radius_3d": float(outer_radius),
        "surface_texture_bake_margin_pixels": float(bake_margin_pixels),
        "surface_texture_screen_raycast": float(use_screen_rays),
        "surface_texture_visible_polygon_count": (
            float(len(visible_polygon_ids))
            if visible_polygon_ids is not None
            else None
        ),
        "surface_texture_screen_inner_radius": (
            float(screen_inner_radius) if use_screen_rays else None
        ),
        "surface_texture_screen_outer_radius": (
            float(screen_outer_radius) if use_screen_rays else None
        ),
        "surface_texture_aperture_mask": float(
            parsed_screen_eye_contour is not None
        ),
        "surface_texture_aperture_target_texels_skipped": float(
            aperture_target_texels_skipped
        ),
        "surface_texture_aperture_sclera_source_samples": float(
            aperture_sclera_source_samples
        ),
        "surface_texture_sclera_candidate_count": float(sclera_candidate_count),
        "surface_texture_sclera_retained_count": float(
            len(sclera_source_positions)
        ),
        "surface_texture_iris_sprite_candidate_count": float(
            iris_sprite_candidate_count
        ),
        "surface_texture_sprite_iris_source_samples": float(
            sprite_iris_source_samples
        ),
        "surface_texture_direct_reference_iris_source_samples": float(
            direct_reference_iris_source_samples
        ),
        "surface_texture_rigid_iris_composite": float(use_screen_rays),
        "surface_texture_screen_iris_bake_radius": (
            float(screen_iris_bake_radius) if use_screen_rays else None
        ),
        "surface_texture_screen_iris_landmark_radius": (
            float(parsed_screen_iris_radius) if use_screen_rays else None
        ),
        "surface_texture_rendered_reference_source": float(
            screen_reference_buffer is not None
        ),
        "surface_texture_screen_eye_width": (
            float(parsed_screen_eye_width) if use_screen_rays else None
        ),
        "surface_texture_screen_iris_cleanup_radius": (
            float(screen_iris_cleanup_radius) if use_screen_rays else None
        ),
        "surface_texture_isolated_material_count": float(
            isolated_material_count
        ),
        "surface_texture_isolated_emissive_material_count": 0.0,
        "surface_texture_isolated_polygon_count": float(
            len(isolated_polygon_ids)
        ),
    }
    if screen_overlay_report is not None:
        report.update(screen_overlay_report)
    return report


def apply_fbx_eye_gaze_uv_warp(
    *,
    mesh_objects: Sequence[Any],
    landmark_payload: Mapping[str, Any],
    action_unit_id: int,
    motion: torch.Tensor,
    rigid_uv_translation: bool = False,
    maximum_eye_width_uv: float = MAX_EYE_GAZE_UV_WIDTH,
    maximum_warped_loop_count: int = MAX_EYE_GAZE_WARPED_LOOPS,
    apply_uv_coordinates: bool = True,
    apply_geometry_translation: bool = False,
    rotate_geometry_component: bool = False,
    force_local_eye_translation: bool = False,
    geometry_rotation_center_hint: Optional[Sequence[float]] = None,
    geometry_rotation_iris_position_hint: Optional[Sequence[float]] = None,
    geometry_rotation_radius_hint: Optional[float] = None,
    geometry_rotation_center: Optional[Sequence[float]] = None,
    geometry_rotation_matrix: Optional[Sequence[Sequence[float]]] = None,
    geometry_shared_translation: Optional[Sequence[float]] = None,
    geometry_translation_center: Optional[Sequence[float]] = None,
    geometry_translation_normal: Optional[Sequence[float]] = None,
    geometry_translation_inner_radius: Optional[float] = None,
    geometry_translation_outer_radius: Optional[float] = None,
    geometry_seed_only: bool = False,
    geometry_calibration_only: bool = False,
    texture_seed_only: bool = False,
    spatial_uv_seed_only: bool = False,
    surface_texture_bake_seed_only: bool = False,
    apply_spatial_uv_sampling: bool = False,
    apply_surface_texture_bake: bool = False,
    apply_screen_texture_bake: bool = False,
    use_rendered_reference_colors: bool = False,
    texture_shared_world_shift: Optional[Sequence[float]] = None,
    texture_shared_iris_radius: Optional[float] = None,
    processed_geometry_components: Optional[set[tuple[str, int]]] = None,
    geometry_topology_indices: Optional[dict[int, dict[str, Any]]] = None,
    geometry_coordinate_cache: Optional[dict[int, tuple[Any, Any]]] = None,
    remove_geometry_component: bool = False,
    geometry_surface_offset_ratio: float = 0.0,
) -> dict[str, Any]:
    """Apply an anchor-localized UV warp without moving facial geometry."""

    if use_rendered_reference_colors:
        raise EyeGazeUVWarpValidationError(
            "Generated eye overlays are disabled."
        )

    import numpy as np

    spec = eye_gaze_landmark_spec(action_unit_id)
    if spec is None:
        raise ValueError(f"AU{action_unit_id} is not an eye-gaze action unit.")
    motion = torch.as_tensor(motion, dtype=torch.float32).flatten()
    if motion.shape != (2,) or not torch.isfinite(motion).all():
        raise ValueError("Eye-gaze UV motion must contain two finite values.")
    if maximum_eye_width_uv <= 0.0:
        raise ValueError("maximum_eye_width_uv must be positive.")
    if maximum_warped_loop_count < 1:
        raise ValueError("maximum_warped_loop_count must be positive.")

    raw_anchors = landmark_payload.get("surface_anchors")
    raw_offsets = landmark_payload.get("mesh_object_offsets")
    if not isinstance(raw_anchors, Mapping) or not isinstance(raw_offsets, Mapping):
        raise EyeGazeUVWarpValidationError(
            "Eye-gaze UV warp requires raycast surface anchors and mesh object offsets."
        )

    objects_by_name: dict[str, Any] = {}
    for obj in mesh_objects:
        objects_by_name[obj.name] = obj
        objects_by_name.setdefault(base_blender_object_name(obj.name), obj)

    def anchor_record(mediapipe_id: int) -> dict[str, Any]:
        raw = raw_anchors.get(str(int(mediapipe_id)))
        if raw is None:
            raw = raw_anchors.get(int(mediapipe_id))
        if not isinstance(raw, Mapping):
            raise EyeGazeUVWarpValidationError(
                f"MediaPipe landmark {mediapipe_id} has no raycast surface anchor."
            )
        object_name = str(raw.get("object_name", ""))
        obj = objects_by_name.get(object_name)
        if obj is None:
            obj = objects_by_name.get(base_blender_object_name(object_name))
        if obj is None:
            raise EyeGazeUVWarpValidationError(
                f"Surface anchor {mediapipe_id} references missing object {object_name!r}."
            )
        uv_layer = obj.data.uv_layers.active
        if uv_layer is None:
            raise EyeGazeUVWarpValidationError(
                f"Surface anchor object {obj.name!r} has no active UV layer."
            )

        try:
            polygon_index = int(raw["polygon_index"])
            global_vertex_ids = tuple(int(value) for value in raw["vertex_ids"])
            weights = np.asarray(raw["barycentric_weights"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise EyeGazeUVWarpValidationError(
                f"Surface anchor {mediapipe_id} is malformed."
            ) from exc
        if len(global_vertex_ids) != 3 or weights.shape != (3,):
            raise EyeGazeUVWarpValidationError(
                f"Surface anchor {mediapipe_id} must contain one triangle."
            )
        if (
            not np.isfinite(weights).all()
            or (weights < -1.0e-4).any()
            or not np.isclose(weights.sum(), 1.0, atol=1.0e-4)
        ):
            raise EyeGazeUVWarpValidationError(
                f"Surface anchor {mediapipe_id} has invalid barycentric weights."
            )
        if not 0 <= polygon_index < len(obj.data.polygons):
            raise EyeGazeUVWarpValidationError(
                f"Surface anchor {mediapipe_id} has an invalid polygon index."
            )

        offset_value = raw_offsets.get(object_name)
        if offset_value is None:
            base_name = base_blender_object_name(object_name)
            matching_offsets = [
                value
                for name, value in raw_offsets.items()
                if base_blender_object_name(str(name)) == base_name
            ]
            if len(matching_offsets) != 1:
                raise EyeGazeUVWarpValidationError(
                    f"Surface anchor {mediapipe_id} has no unambiguous object offset."
                )
            offset_value = matching_offsets[0]
        object_offset = int(offset_value)
        local_vertex_ids = tuple(value - object_offset for value in global_vertex_ids)
        polygon = obj.data.polygons[polygon_index]
        loops_by_vertex = {
            int(obj.data.loops[loop_index].vertex_index): int(loop_index)
            for loop_index in polygon.loop_indices
        }
        try:
            triangle_loops = tuple(loops_by_vertex[value] for value in local_vertex_ids)
        except KeyError as exc:
            raise EyeGazeUVWarpValidationError(
                f"Surface anchor {mediapipe_id} triangle does not match its polygon."
            ) from exc
        triangle_uvs = np.stack(
            [
                np.asarray(uv_layer.data[loop_index].uv, dtype=np.float64)
                for loop_index in triangle_loops
            ]
        )
        triangle_positions = np.stack(
            [
                np.asarray(obj.data.vertices[vertex_id].co, dtype=np.float64)
                for vertex_id in local_vertex_ids
            ]
        )
        raw_screen_position = raw.get("screen_position")
        screen_position = None
        if isinstance(raw_screen_position, Mapping):
            try:
                screen_position = np.asarray(
                    (
                        float(raw_screen_position["x"]),
                        float(raw_screen_position["y"]),
                    ),
                    dtype=np.float64,
                )
            except (KeyError, TypeError, ValueError):
                screen_position = None
        return {
            "object": obj,
            "polygon_index": polygon_index,
            "uv": (triangle_uvs * weights[:, None]).sum(axis=0),
            "position": (triangle_positions * weights[:, None]).sum(axis=0),
            "triangle_uvs": triangle_uvs,
            "triangle_positions": triangle_positions,
            "screen_position": screen_position,
        }

    if geometry_seed_only:
        if not rotate_geometry_component:
            raise EyeGazeUVWarpValidationError(
                "A stacked eye-geometry seed requires physical eye motion."
            )
        center_anchor = anchor_record(spec.center_id)
        raw_selection = landmark_payload.get("surface_anchor_selection")
        eye_selection = (
            raw_selection.get(spec.eye)
            if isinstance(raw_selection, Mapping)
            else None
        )
        try:
            spatial_eye_width = float(eye_selection["eye_width_3d"])
        except (KeyError, TypeError, ValueError):
            spatial_eye_width = 0.0
        translation_center = geometry_translation_center
        translation_inner_radius = geometry_translation_inner_radius
        translation_outer_radius = geometry_translation_outer_radius
        if geometry_shared_translation is not None:
            if spatial_eye_width <= 1.0e-8:
                raise EyeGazeUVWarpValidationError(
                    "Stacked eye translation requires a valid spatial eye width."
                )
            if translation_center is None:
                translation_center = center_anchor["position"]
            if translation_inner_radius is None:
                translation_inner_radius = spatial_eye_width * 0.30
            if translation_outer_radius is None:
                translation_outer_radius = spatial_eye_width * 0.90
        topology_index = None
        if geometry_topology_indices is not None:
            topology_index = geometry_topology_indices.get(id(center_anchor["object"].data))
            if topology_index is None:
                topology_index = build_eye_gaze_topology_index(
                    center_anchor["object"].data
                )
                geometry_topology_indices[id(center_anchor["object"].data)] = (
                    topology_index
                )
        return apply_shared_eye_geometry_transform(
            eye_object=center_anchor["object"],
            seed_polygon=int(center_anchor["polygon_index"]),
            geometry_rotation_center=geometry_rotation_center,
            geometry_rotation_matrix=geometry_rotation_matrix,
            geometry_shared_translation=geometry_shared_translation,
            geometry_translation_center=translation_center,
            geometry_translation_normal=geometry_translation_normal,
            geometry_translation_inner_radius=translation_inner_radius,
            geometry_translation_outer_radius=translation_outer_radius,
            report_as_translation=force_local_eye_translation,
            processed_geometry_components=processed_geometry_components,
            topology_index=topology_index,
            coordinate_cache=geometry_coordinate_cache,
        )

    if texture_seed_only or spatial_uv_seed_only or surface_texture_bake_seed_only:
        center_anchor = anchor_record(spec.center_id)
        world_shift = np.asarray(texture_shared_world_shift, dtype=np.float64)
        iris_radius_3d = float(texture_shared_iris_radius or 0.0)
        if (
            world_shift.shape != (3,)
            or not np.isfinite(world_shift).all()
            or iris_radius_3d <= 1.0e-8
        ):
            raise EyeGazeUVWarpValidationError(
                "Stacked iris texture motion requires a valid shared 3D shift."
            )
        triangle_positions = center_anchor["triangle_positions"]
        triangle_uvs = center_anchor["triangle_uvs"]
        position_deltas = triangle_positions[1:] - triangle_positions[0]
        uv_deltas = triangle_uvs[1:] - triangle_uvs[0]
        position_to_uv = np.linalg.lstsq(
            position_deltas,
            uv_deltas,
            rcond=None,
        )[0]
        _world_axes, singular_values, uv_axes = np.linalg.svd(
            position_to_uv,
            full_matrices=False,
        )
        radii = singular_values * iris_radius_3d
        visual_shift = world_shift @ position_to_uv
        if (
            radii.shape != (2,)
            or not np.isfinite(radii).all()
            or float(radii.min()) <= 1.0e-6
            or float(radii.max()) > MAX_EYE_GAZE_UV_WIDTH
            or not np.isfinite(visual_shift).all()
        ):
            raise EyeGazeUVWarpValidationError(
                "Stacked iris texture layer has an unstable local UV transform."
            )
        center_uv = center_anchor["uv"]
        report = {
            "warped_loop_count": 0.0,
            "maximum_weight": 0.0,
            "center_u": float(center_uv[0]),
            "center_v": float(center_uv[1]),
            "iris_radius_uv": float(radii.max()),
            "iris_horizontal_radius_uv": float(radii[0]),
            "iris_vertical_radius_uv": float(radii[1]),
            "iris_horizontal_axis_u": float(uv_axes[0, 0]),
            "iris_horizontal_axis_v": float(uv_axes[0, 1]),
            "iris_vertical_axis_u": float(uv_axes[1, 0]),
            "iris_vertical_axis_v": float(uv_axes[1, 1]),
            "visual_shift_u": float(visual_shift[0]),
            "visual_shift_v": float(visual_shift[1]),
            "world_shift_x": float(world_shift[0]),
            "world_shift_y": float(world_shift[1]),
            "world_shift_z": float(world_shift[2]),
            "iris_position_radius_3d": iris_radius_3d,
            "screen_calibrated": 1.0,
            "uv_coordinates_modified": 0.0,
            "geometry_vertices_rotated": 0.0,
            "geometry_vertices_translated": 0.0,
            "texture_seed_only": 1.0,
        }
        if spatial_uv_seed_only:
            raw_selection = landmark_payload.get("surface_anchor_selection")
            eye_selection = (
                raw_selection.get(spec.eye)
                if isinstance(raw_selection, Mapping)
                else None
            )
            try:
                spatial_eye_width = float(eye_selection["eye_width_3d"])
            except (KeyError, TypeError, ValueError):
                spatial_eye_width = 0.0
            report.update(
                apply_spatial_eye_uv_sampling_warp(
                    eye_object=center_anchor["object"],
                    seed_polygon=int(center_anchor["polygon_index"]),
                    center_position=center_anchor["position"],
                    iris_radius_3d=iris_radius_3d,
                    eye_width_3d=spatial_eye_width,
                    world_shift=world_shift,
                    maximum_warped_loop_count=max(
                        maximum_warped_loop_count,
                        MAX_EYE_GAZE_SPATIAL_UV_WARPED_LOOPS,
                    ),
                )
            )
            report["uv_coordinates_modified"] = 1.0
            report["spatial_uv_seed_only"] = 1.0
        if surface_texture_bake_seed_only:
            raw_selection = landmark_payload.get("surface_anchor_selection")
            eye_selection = (
                raw_selection.get(spec.eye)
                if isinstance(raw_selection, Mapping)
                else None
            )
            try:
                spatial_eye_width = float(eye_selection["eye_width_3d"])
            except (KeyError, TypeError, ValueError):
                spatial_eye_width = 0.0
            eye_object = center_anchor["object"]
            mesh = eye_object.data
            uv_layer = mesh.uv_layers.active
            loop_uvs = np.stack(
                [np.asarray(value.uv, dtype=np.float64) for value in uv_layer.data]
            )
            loop_vertex_ids = np.fromiter(
                (int(loop.vertex_index) for loop in mesh.loops),
                dtype=np.int64,
                count=len(mesh.loops),
            )
            polygon_loop_indices = [
                tuple(int(value) for value in polygon.loop_indices)
                for polygon in mesh.polygons
            ]
            component_polygons, _component_loops = uv_connected_polygon_component(
                polygon_loop_indices=polygon_loop_indices,
                loop_vertex_ids=loop_vertex_ids,
                loop_uvs=loop_uvs,
                seed_polygon_index=int(center_anchor["polygon_index"]),
            )
            report.update(
                apply_spatial_eye_texture_bake(
                    eye_object=eye_object,
                    component_polygons=component_polygons,
                    center_position=center_anchor["position"],
                    iris_radius_3d=iris_radius_3d,
                    eye_width_3d=spatial_eye_width,
                    world_shift=world_shift,
                )
            )
            report["surface_texture_bake_seed_only"] = 1.0
        return report

    raw_selection = landmark_payload.get("surface_anchor_selection", {})
    eye_selection = (
        raw_selection.get(spec.eye, {})
        if isinstance(raw_selection, Mapping)
        else {}
    )
    raw_valid_iris_ids = (
        eye_selection.get("valid_iris_ids")
        if isinstance(eye_selection, Mapping)
        else None
    )
    if raw_valid_iris_ids is None:
        valid_iris_ids = tuple(int(value) for value in spec.iris_ids)
    else:
        try:
            valid_iris_ids = tuple(int(value) for value in raw_valid_iris_ids)
        except (TypeError, ValueError) as exc:
            raise EyeGazeUVWarpValidationError(
                "Eye-gaze visible-iris landmark selection is malformed."
            ) from exc
    valid_iris_ids = tuple(
        value for value in spec.iris_ids if value in set(valid_iris_ids)
    )
    if spec.center_id not in valid_iris_ids or len(valid_iris_ids) < 3:
        raise EyeGazeUVWarpValidationError(
            "Eye-gaze visible-iris selection needs the center and two edge landmarks."
        )

    required_ids = set((*valid_iris_ids, *spec.corner_ids))
    landmark_anchors = {
        mediapipe_id: anchor_record(mediapipe_id)
        for mediapipe_id in required_ids
    }
    eye_object = landmark_anchors[spec.center_id]["object"]
    if any(value["object"] is not eye_object for value in landmark_anchors.values()):
        raise EyeGazeUVWarpValidationError(
            "Required eye-gaze surface anchors are split across mesh objects."
        )
    uv_layer = eye_object.data.uv_layers.active
    landmark_uv = {
        mediapipe_id: value["uv"]
        for mediapipe_id, value in landmark_anchors.items()
    }
    landmark_positions = {
        mediapipe_id: value["position"]
        for mediapipe_id, value in landmark_anchors.items()
    }
    landmark_screen_positions = {
        mediapipe_id: value["screen_position"]
        for mediapipe_id, value in landmark_anchors.items()
        if value["screen_position"] is not None
    }
    for mediapipe_id in spec.contour_ids:
        try:
            record = anchor_record(mediapipe_id)
        except EyeGazeUVWarpValidationError:
            continue
        if record["object"] is eye_object:
            landmark_anchors[mediapipe_id] = record
            landmark_uv[mediapipe_id] = record["uv"]
            landmark_positions[mediapipe_id] = record["position"]
            if record["screen_position"] is not None:
                landmark_screen_positions[mediapipe_id] = record["screen_position"]

    # UV visibility controls the mesh anchor set, but the rendered iris ellipse
    # can use every valid screen landmark, including points occluded in 3D.
    for mediapipe_id in spec.iris_ids:
        if mediapipe_id in landmark_screen_positions:
            continue
        try:
            record = anchor_record(mediapipe_id)
        except EyeGazeUVWarpValidationError:
            continue
        if record["screen_position"] is not None:
            landmark_screen_positions[mediapipe_id] = record["screen_position"]

    center_uv = landmark_uv[spec.center_id]
    top_iris_id = spec.iris_ids[2]
    left_iris_id = spec.iris_ids[3]
    if spec.right_iris_id in landmark_uv:
        horizontal_raw = landmark_uv[spec.right_iris_id] - center_uv
    elif left_iris_id in landmark_uv:
        horizontal_raw = center_uv - landmark_uv[left_iris_id]
    else:
        raise EyeGazeUVWarpValidationError(
            "Eye-gaze visible iris has no horizontal edge landmark."
        )
    if spec.bottom_iris_id in landmark_uv:
        vertical_raw = landmark_uv[spec.bottom_iris_id] - center_uv
    elif top_iris_id in landmark_uv:
        vertical_raw = center_uv - landmark_uv[top_iris_id]
    else:
        raise EyeGazeUVWarpValidationError(
            "Eye-gaze visible iris has no vertical edge landmark."
        )
    horizontal_norm = float(np.linalg.norm(horizontal_raw))
    if horizontal_norm <= 1.0e-8:
        raise ValueError("Eye-gaze iris UVs have zero horizontal extent.")
    horizontal_axis = horizontal_raw / horizontal_norm
    vertical_axis = vertical_raw - horizontal_axis * float(
        np.dot(vertical_raw, horizontal_axis)
    )
    vertical_norm = float(np.linalg.norm(vertical_axis))
    if vertical_norm <= 1.0e-8:
        vertical_axis = np.asarray((-horizontal_axis[1], horizontal_axis[0]))
        if float(np.dot(vertical_axis, vertical_raw)) < 0.0:
            vertical_axis *= -1.0
    else:
        vertical_axis /= vertical_norm

    iris_uvs = [
        landmark_uv[value]
        for value in valid_iris_ids
        if value in landmark_uv
    ]
    iris_relative_uvs = np.stack(iris_uvs) - center_uv
    iris_horizontal_radius = float(
        np.abs(iris_relative_uvs @ horizontal_axis).max()
    )
    iris_vertical_radius = float(
        np.abs(iris_relative_uvs @ vertical_axis).max()
    )
    iris_radius = max(
        float(np.linalg.norm(value - center_uv)) for value in iris_uvs
    )
    if (
        not np.isfinite(iris_radius)
        or iris_radius <= 1.0e-6
        or not np.isfinite(iris_horizontal_radius)
        or iris_horizontal_radius <= 1.0e-6
        or not np.isfinite(iris_vertical_radius)
        or iris_vertical_radius <= 1.0e-6
    ):
        raise EyeGazeUVWarpValidationError(
            "Eye-gaze iris anchors have invalid UV radii."
        )
    corner_width = float(
        np.linalg.norm(
            landmark_uv[spec.corner_ids[1]] - landmark_uv[spec.corner_ids[0]]
        )
    )
    center_position = landmark_positions[spec.center_id]
    iris_position_radius = max(
        float(np.linalg.norm(landmark_positions[value] - center_position))
        for value in valid_iris_ids
    )
    corner_positions = [landmark_positions[value] for value in spec.corner_ids]
    spatial_eye_width = float(
        np.linalg.norm(corner_positions[1] - corner_positions[0])
    )
    if not np.isfinite(iris_position_radius) or iris_position_radius <= 1.0e-8:
        raise EyeGazeUVWarpValidationError(
            "Eye-gaze iris anchors have an invalid 3D radius."
        )
    eye_to_iris_ratio = spatial_eye_width / iris_position_radius
    minimum_eye_to_iris_ratio = (
        1.5
        if rotate_geometry_component
        and geometry_rotation_center_hint is not None
        else 2.5
    )
    if not minimum_eye_to_iris_ratio <= eye_to_iris_ratio <= 8.0:
        raise EyeGazeUVWarpValidationError(
            "Eye corner and iris anchors do not define a consistent 3D scale: "
            f"eye_width={spatial_eye_width:.6f}, "
            f"iris_radius={iris_position_radius:.6f}, "
            f"ratio={eye_to_iris_ratio:.6f}, "
            f"minimum={minimum_eye_to_iris_ratio:.6f}."
        )
    eye_width_uv = iris_radius * eye_to_iris_ratio
    if not 1.0e-4 <= eye_width_uv <= float(maximum_eye_width_uv):
        raise EyeGazeUVWarpValidationError(
            "Eye-gaze UV width is outside the safe range: "
            f"width={eye_width_uv:.6f}, maximum={maximum_eye_width_uv:.6f}."
        )

    spatial_radius = max(
        spatial_eye_width * 0.9,
        1.0e-6,
    )
    loop_uvs = np.stack(
        [np.asarray(value.uv, dtype=np.float64) for value in uv_layer.data]
    )
    loop_vertex_ids = np.fromiter(
        (int(loop.vertex_index) for loop in eye_object.data.loops),
        dtype=np.int64,
        count=len(eye_object.data.loops),
    )
    loop_positions = np.stack(
        [
            np.asarray(
                eye_object.data.vertices[loop.vertex_index].co,
                dtype=np.float64,
            )
            for loop in eye_object.data.loops
        ]
    )
    spatial_mask = np.linalg.norm(loop_positions - center_position, axis=1) <= spatial_radius
    polygon_loop_indices = [
        tuple(int(value) for value in polygon.loop_indices)
        for polygon in eye_object.data.polygons
    ]
    candidate_polygons = np.asarray(
        [
            bool(np.any(spatial_mask[np.asarray(loop_indices, dtype=np.int64)]))
            for loop_indices in polygon_loop_indices
        ],
        dtype=bool,
    )
    for mediapipe_id in valid_iris_ids:
        candidate_polygons[landmark_anchors[mediapipe_id]["polygon_index"]] = True
    component_polygons, component_loops = uv_connected_polygon_component(
        polygon_loop_indices=polygon_loop_indices,
        loop_vertex_ids=loop_vertex_ids,
        loop_uvs=loop_uvs,
        seed_polygon_index=landmark_anchors[spec.center_id]["polygon_index"],
        candidate_polygons=candidate_polygons,
    )
    disconnected = [
        mediapipe_id
        for mediapipe_id in valid_iris_ids
        if not component_polygons[landmark_anchors[mediapipe_id]["polygon_index"]]
    ]
    if disconnected:
        raise EyeGazeUVWarpValidationError(
            "Required eye-gaze anchors are not on the iris UV component: "
            f"{sorted(disconnected)}."
        )

    contour_uvs = [
        landmark_uv[value]
        for value in spec.contour_ids
        if value in landmark_uv
        and component_polygons[landmark_anchors[value]["polygon_index"]]
        and float(np.linalg.norm(landmark_uv[value] - center_uv))
        <= eye_width_uv * 1.25
    ]
    contour_relative = np.stack(contour_uvs) - center_uv if contour_uvs else None
    horizontal_radius = eye_width_uv * EYE_GAZE_WARP_RADIUS_EYE_WIDTHS
    vertical_radius = eye_width_uv * EYE_GAZE_WARP_RADIUS_EYE_WIDTHS
    if contour_relative is not None:
        horizontal_radius = max(
            horizontal_radius,
            float(np.abs(contour_relative @ horizontal_axis).max()) * 1.05,
        )
        vertical_radius = max(
            vertical_radius,
            float(np.abs(contour_relative @ vertical_axis).max()) * 1.10,
        )
    horizontal_radius = min(horizontal_radius, eye_width_uv * 0.75)
    vertical_radius = min(vertical_radius, eye_width_uv * 0.75)
    active_mask = spatial_mask & component_loops
    visual_shift = (
        horizontal_axis * float(motion[0])
        + vertical_axis * float(motion[1])
    ) * eye_width_uv
    screen_calibrated = False
    screen_shift = None
    screen_projection = None
    screen_projection_error = None
    screen_depth_direction = None
    screen_center = None
    screen_iris_radius = None
    screen_iris_basis = None
    screen_eye_width = None
    screen_eye_contour = None
    if all(value in landmark_screen_positions for value in spec.contour_ids):
        screen_eye_contour = np.stack(
            [landmark_screen_positions[value] for value in spec.contour_ids]
        )
    world_shift = np.zeros(3, dtype=np.float64)
    if (
        spec.center_id in landmark_screen_positions
        and all(value in landmark_screen_positions for value in spec.corner_ids)
    ):
        screen_center = landmark_screen_positions[spec.center_id]
        screen_deltas = []
        uv_deltas = []
        position_deltas = []
        for mediapipe_id in valid_iris_ids:
            if (
                mediapipe_id == spec.center_id
                or mediapipe_id not in landmark_screen_positions
            ):
                continue
            screen_deltas.append(
                landmark_screen_positions[mediapipe_id] - screen_center
            )
            uv_deltas.append(landmark_uv[mediapipe_id] - center_uv)
            position_deltas.append(
                landmark_positions[mediapipe_id] - center_position
            )
        if len(screen_deltas) >= 2:
            screen_matrix = np.stack(screen_deltas)
            uv_matrix = np.stack(uv_deltas)
            if np.linalg.matrix_rank(screen_matrix) >= 2:
                screen_motion = -np.asarray(
                    (float(motion[0]), float(motion[1])), dtype=np.float64
                )
                screen_to_uv = np.linalg.lstsq(
                    screen_matrix,
                    uv_matrix,
                    rcond=None,
                )[0]
                screen_eye_width = float(
                    np.linalg.norm(
                        landmark_screen_positions[spec.corner_ids[1]]
                        - landmark_screen_positions[spec.corner_ids[0]]
                    )
                )
                requested_screen_shift = eye_gaze_screen_shift_from_eye_width(
                    screen_motion,
                    screen_eye_width,
                )
                calibrated_shift = (
                    requested_screen_shift
                ) @ screen_to_uv
                if np.isfinite(calibrated_shift).all():
                    visual_shift = calibrated_shift
                    position_matrix = np.stack(position_deltas)
                    screen_to_position = np.linalg.lstsq(
                        screen_matrix,
                        position_matrix,
                        rcond=None,
                    )[0]
                    world_shift = requested_screen_shift @ screen_to_position
                    if (
                        rotate_geometry_component
                        and geometry_rotation_center_hint is not None
                    ):
                        rotation_center_hint = np.asarray(
                            geometry_rotation_center_hint,
                            dtype=np.float64,
                        )
                        rotation_iris_hint = np.asarray(
                            geometry_rotation_iris_position_hint
                            if geometry_rotation_iris_position_hint is not None
                            else center_position,
                            dtype=np.float64,
                        )
                        physical_projection, projection_error = (
                            eye_gaze_landmark_screen_projection(
                                landmark_positions=landmark_positions,
                                landmark_screen_positions=(
                                    landmark_screen_positions
                                ),
                            )
                        )
                        if projection_error > 0.005:
                            raise EyeGazeUVWarpValidationError(
                                "Raycast landmark projection is not reliably "
                                "orthographic for physical gaze: "
                                f"maximum_error={projection_error:.6f}."
                            )
                        world_shift = tangent_world_shift_for_screen_shift(
                            position_to_screen=physical_projection[:3],
                            source_direction=(
                                rotation_iris_hint - rotation_center_hint
                            ),
                            screen_shift=requested_screen_shift,
                        )
                    screen_calibrated = True
                    screen_shift = requested_screen_shift
                screen_iris_radius = float(
                    np.linalg.norm(screen_matrix, axis=1).max()
                )
                if (
                    spec.right_iris_id in landmark_screen_positions
                    and spec.bottom_iris_id in landmark_screen_positions
                ):
                    screen_iris_basis = np.column_stack(
                        (
                            landmark_screen_positions[spec.right_iris_id]
                            - screen_center,
                            landmark_screen_positions[spec.bottom_iris_id]
                            - screen_center,
                        )
                    )

    if apply_screen_texture_bake:
        if screen_eye_contour is None:
            raise EyeGazeUVWarpValidationError(
                "Screen texture bake needs the complete rendered eyelid contour."
            )
        screen_projection, screen_projection_error = (
            eye_gaze_landmark_screen_projection(
                landmark_positions=landmark_positions,
                landmark_screen_positions=landmark_screen_positions,
            )
        )
        if screen_projection_error > 0.005:
            raise EyeGazeUVWarpValidationError(
                "Raycast landmark projection is not reliably orthographic: "
                f"maximum_error={screen_projection_error:.6f}."
            )
        view_directions = {
            "neg_y": (0.0, -1.0, 0.0),
            "pos_y": (0.0, 1.0, 0.0),
            "neg_x": (-1.0, 0.0, 0.0),
            "pos_x": (1.0, 0.0, 0.0),
            "neg_z": (0.0, 0.0, -1.0),
            "pos_z": (0.0, 0.0, 1.0),
        }
        selected_view = str(landmark_payload.get("selected_view", ""))
        if selected_view not in view_directions:
            raise EyeGazeUVWarpValidationError(
                f"Unknown cached landmark view {selected_view!r}."
            )
        source_direction = np.asarray(
            view_directions[selected_view], dtype=np.float64
        )
        yaw = np.deg2rad(float(landmark_payload.get("yaw_offset", 0.0)))
        source_direction = source_direction @ np.asarray(
            (
                (np.cos(yaw), -np.sin(yaw), 0.0),
                (np.sin(yaw), np.cos(yaw), 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        ).T
        screen_depth_direction = np.asarray(
            (source_direction[0], -source_direction[2], source_direction[1]),
            dtype=np.float64,
        )

    spatial_uv_report = None
    if apply_spatial_uv_sampling:
        spatial_uv_report = apply_spatial_eye_uv_sampling_warp(
            eye_object=eye_object,
            seed_polygon=int(landmark_anchors[spec.center_id]["polygon_index"]),
            center_position=center_position,
            iris_radius_3d=iris_position_radius,
            eye_width_3d=spatial_eye_width,
            world_shift=world_shift,
            maximum_warped_loop_count=max(
                maximum_warped_loop_count,
                MAX_EYE_GAZE_SPATIAL_UV_WARPED_LOOPS,
            ),
        )

    surface_texture_bake_report = None
    if apply_surface_texture_bake:
        if not screen_calibrated:
            raise EyeGazeUVWarpValidationError(
                "Surface texture bake requires screen-calibrated landmarks."
            )
        surface_texture_bake_report = apply_spatial_eye_texture_bake(
            eye_object=eye_object,
            component_polygons=component_polygons,
            center_position=center_position,
            iris_radius_3d=iris_position_radius,
            eye_width_3d=spatial_eye_width,
            world_shift=world_shift,
            screen_projection=screen_projection,
            screen_depth_direction=screen_depth_direction,
            screen_shift=(
                physical_eye_screen_shift_from_sampling_shift(screen_shift)
                if apply_screen_texture_bake and screen_shift is not None
                else screen_shift
            ),
            screen_center=screen_center,
            screen_iris_radius=screen_iris_radius,
            screen_iris_basis=screen_iris_basis,
            screen_eye_width=screen_eye_width,
            screen_eye_contour=(
                screen_eye_contour if apply_screen_texture_bake else None
            ),
            screen_reference_image_path=landmark_payload.get(
                "_rendered_reference_image"
            ),
            use_rendered_reference_colors=use_rendered_reference_colors,
            bake_margin_pixels=(0 if apply_screen_texture_bake else 6),
        )

    if apply_uv_coordinates:
        source_envelope = iris_relative_uvs * 1.20
        swept_envelope = np.concatenate(
            (source_envelope, source_envelope + visual_shift[None, :]),
            axis=0,
        )

        def normalized_envelope_radius() -> float:
            horizontal = (swept_envelope @ horizontal_axis) / horizontal_radius
            vertical = (swept_envelope @ vertical_axis) / vertical_radius
            return float(np.sqrt(horizontal**2 + vertical**2).max())

        swept_radius = normalized_envelope_radius()
        target_swept_radius = 0.58
        if swept_radius > target_swept_radius:
            expansion = swept_radius / target_swept_radius
            maximum_radius = eye_width_uv
            horizontal_radius = min(
                horizontal_radius * expansion,
                maximum_radius,
            )
            vertical_radius = min(
                vertical_radius * expansion,
                maximum_radius,
            )
            swept_radius = normalized_envelope_radius()
        if swept_radius >= 0.82:
            raise EyeGazeUVWarpValidationError(
                "Eye-gaze UV component cannot preserve the full translated iris: "
                f"swept_radius={swept_radius:.6f}."
            )
        rigid_iris_radius = max(
            EYE_GAZE_WARP_INNER_RADIUS,
            swept_radius * 1.03,
        )
    else:
        rigid_iris_radius = min(
            0.85,
            max(
                EYE_GAZE_WARP_INNER_RADIUS,
                iris_horizontal_radius / horizontal_radius,
                iris_vertical_radius / vertical_radius,
            )
            * 1.05,
        )
    normalized_shift = float(
        np.linalg.norm(
            np.asarray(
                (
                    float(np.dot(visual_shift, horizontal_axis)) / horizontal_radius,
                    float(np.dot(visual_shift, vertical_axis)) / vertical_radius,
                )
            )
        )
    )
    maximum_safe_shift = (1.0 - rigid_iris_radius) / 1.5 * 0.9
    if (
        apply_uv_coordinates
        and not rigid_uv_translation
        and normalized_shift >= maximum_safe_shift
    ):
        raise EyeGazeUVWarpValidationError(
            "Eye-gaze UV shift would fold the local texture map: "
            f"normalized_shift={normalized_shift:.6f}, "
            f"maximum={maximum_safe_shift:.6f}."
        )
    if rigid_uv_translation:
        if float(np.linalg.norm(visual_shift)) > iris_radius * 0.90:
            raise EyeGazeUVWarpValidationError(
                "Rigid eye UV translation exceeds the mapped iris radius: "
                f"shift={float(np.linalg.norm(visual_shift)):.6f}, "
                f"radius={iris_radius:.6f}."
            )
        weights = component_loops.astype(np.float32)
        warped_uvs = loop_uvs - weights[:, None] * visual_shift[None, :]
        active_warped_uvs = warped_uvs[component_loops]
        if (
            not np.isfinite(active_warped_uvs).all()
            or float(active_warped_uvs.min()) < -1.0e-5
            or float(active_warped_uvs.max()) > 1.0 + 1.0e-5
        ):
            raise EyeGazeUVWarpValidationError(
                "Rigid eye UV translation leaves the packed texture atlas."
            )
    else:
        warped_uvs, weights = tapered_eye_uv_warp(
            torch.from_numpy(loop_uvs).float(),
            center=torch.from_numpy(center_uv).float(),
            horizontal_axis=torch.from_numpy(horizontal_axis).float(),
            vertical_axis=torch.from_numpy(vertical_axis).float(),
            horizontal_radius=horizontal_radius,
            vertical_radius=vertical_radius,
            visual_shift=torch.from_numpy(visual_shift).float(),
            active_mask=torch.from_numpy(active_mask),
            inner_radius=rigid_iris_radius,
        )
        warped_uvs = warped_uvs.numpy()
        weights = weights.numpy()
    warped_loop_count = int(np.count_nonzero(weights > 1.0e-4))
    if apply_uv_coordinates and warped_loop_count < 3:
        raise EyeGazeUVWarpValidationError(
            f"Eye-gaze UV component contains only {warped_loop_count} warped loops."
        )
    if (
        apply_uv_coordinates
        and warped_loop_count > int(maximum_warped_loop_count)
    ):
        raise EyeGazeUVWarpValidationError(
            "Eye-gaze UV component exceeds the safe loop limit: "
            f"loops={warped_loop_count}, maximum={maximum_warped_loop_count}."
        )
    if apply_uv_coordinates:
        for index in np.flatnonzero(weights > 0.0):
            uv_layer.data[int(index)].uv = tuple(
                float(value) for value in warped_uvs[index]
            )
        eye_object.data.update()
    geometry_vertex_count = 0
    geometry_rotated_vertex_count = 0
    removed_polygon_count = 0
    geometry_surface_offset = np.zeros(3, dtype=np.float64)
    output_rotation_center = None
    output_rotation_matrix = None
    output_shared_translation = None
    output_translation_center = None
    output_translation_normal = None
    output_translation_inner_radius = None
    output_translation_outer_radius = None
    geometry_target_shift = None
    taper_geometry_rotation = False
    geometry_component_key = None
    geometry_component_polygon_count = 0
    if apply_geometry_translation:
        component_loop_count = int(np.count_nonzero(component_loops))
        if component_loop_count > 5_000:
            raise EyeGazeUVWarpValidationError(
                "Pupil geometry component is too large to translate safely: "
                f"loops={component_loop_count}, maximum=5000."
            )
        if not screen_calibrated or not np.isfinite(world_shift).all():
            raise EyeGazeUVWarpValidationError(
                "Pupil geometry translation requires screen-calibrated landmarks."
            )
        world_shift_norm = float(np.linalg.norm(world_shift))
        if world_shift_norm > spatial_eye_width * 0.25:
            raise EyeGazeUVWarpValidationError(
                "Pupil geometry translation exceeds one quarter eye width: "
                f"shift={world_shift_norm:.6f}, eye_width={spatial_eye_width:.6f}."
            )
        component_vertex_ids = np.unique(loop_vertex_ids[component_loops])
        object_coordinates = np.empty(
            len(eye_object.data.vertices) * 3,
            dtype=np.float64,
        )
        eye_object.data.vertices.foreach_get("co", object_coordinates)
        object_coordinates = object_coordinates.reshape(-1, 3)
        if geometry_surface_offset_ratio > 0.0:
            component_normals = np.stack(
                [
                    np.asarray(eye_object.data.vertices[int(index)].normal)
                    for index in component_vertex_ids
                ]
            )
            mean_normal = component_normals.mean(axis=0)
            mean_normal_norm = float(np.linalg.norm(mean_normal))
            if mean_normal_norm <= 1.0e-8:
                raise EyeGazeUVWarpValidationError(
                    "Pupil geometry component has no stable outward normal."
                )
            geometry_surface_offset = (
                mean_normal
                / mean_normal_norm
                * spatial_eye_width
                * float(geometry_surface_offset_ratio)
            )
        object_coordinates[component_vertex_ids] += (
            world_shift + geometry_surface_offset
        )
        eye_object.data.vertices.foreach_set("co", object_coordinates.reshape(-1))
        eye_object.data.update(calc_edges=True)
        geometry_vertex_count = int(len(component_vertex_ids))
    if rotate_geometry_component:
        mesh = eye_object.data
        seed_polygon = int(landmark_anchors[spec.center_id]["polygon_index"])
        topology_polygon_ids, topology_vertex_ids = eye_gaze_topology_component(
            mesh,
            seed_polygon,
        )
        if len(topology_polygon_ids) > 20_000:
            raise EyeGazeUVWarpValidationError(
                "Eye geometry component is too large to rotate safely: "
                f"polygons={len(topology_polygon_ids)}, maximum=20000."
            )
        geometry_component_key = (
            str(eye_object.name),
            int(topology_polygon_ids.min()),
        )
        geometry_component_polygon_count = int(len(topology_polygon_ids))
        object_coordinates = np.empty(
            len(mesh.vertices) * 3,
            dtype=np.float64,
        )
        mesh.vertices.foreach_get("co", object_coordinates)
        object_coordinates = object_coordinates.reshape(-1, 3)
        component_coordinates = object_coordinates[topology_vertex_ids]
        shared_translation = None

        if geometry_shared_translation is not None:
            shared_translation = np.asarray(
                geometry_shared_translation,
                dtype=np.float64,
            )
            if shared_translation.shape != (3,):
                raise EyeGazeUVWarpValidationError(
                    "Shared eye geometry translation has invalid dimensions."
                )
            rotation_center = None
            rotation_matrix = None
        elif geometry_rotation_center is None or geometry_rotation_matrix is None:
            if not screen_calibrated or not np.isfinite(world_shift).all():
                raise EyeGazeUVWarpValidationError(
                    "Eye component rotation requires screen-calibrated landmarks."
                )
            if geometry_rotation_center_hint is not None:
                rotation_center = np.asarray(
                    geometry_rotation_center_hint,
                    dtype=np.float64,
                )
                if rotation_center.shape != (3,) or not np.isfinite(
                    rotation_center
                ).all():
                    raise EyeGazeUVWarpValidationError(
                        "Selected eye rotation center is invalid."
                    )
            else:
                rotation_center, sphere_radius, sphere_residual_ratio = fit_sphere(
                    component_coordinates
                )
                if (
                    not np.isfinite(sphere_radius)
                    or sphere_radius <= 1.0e-6
                    or sphere_residual_ratio > 0.25
                ):
                    raise EyeGazeUVWarpValidationError(
                        "Eye geometry component does not fit a stable sphere: "
                        f"radius={sphere_radius:.6f}, "
                        f"residual_ratio={sphere_residual_ratio:.6f}."
                    )
            taper_geometry_rotation = force_local_eye_translation
            rotation_matrix, geometry_target_shift = (
                rigid_eye_rotation_from_sampling_shift(
                    iris_position=(
                        geometry_rotation_iris_position_hint
                        if geometry_rotation_iris_position_hint is not None
                        else center_position
                    ),
                    rotation_center=rotation_center,
                    sampling_world_shift=world_shift,
                    rotation_radius=geometry_rotation_radius_hint,
                )
            )
            if force_local_eye_translation:
                shared_translation = geometry_target_shift.copy()
        else:
            rotation_center = np.asarray(
                geometry_rotation_center,
                dtype=np.float64,
            )
            rotation_matrix = np.asarray(
                geometry_rotation_matrix,
                dtype=np.float64,
            )
            if rotation_center.shape != (3,) or rotation_matrix.shape != (3, 3):
                raise EyeGazeUVWarpValidationError(
                    "Shared eye geometry rotation has invalid dimensions."
                )
        if shared_translation is not None and not taper_geometry_rotation:
            transformed_coordinates = component_coordinates + shared_translation
            geometry_rotated_vertex_count = int(len(topology_vertex_ids))
        elif shared_translation is not None:
            inner_radius = max(
                iris_position_radius * 1.25,
                spatial_eye_width * 0.25,
            )
            outer_radius = max(
                inner_radius * 1.5,
                spatial_eye_width * 0.90,
            )
            translation_normal = np.asarray(
                center_position - rotation_center,
                dtype=np.float64,
            )
            translation_normal_length = float(np.linalg.norm(translation_normal))
            if translation_normal_length <= 1.0e-8:
                raise EyeGazeUVWarpValidationError(
                    "Local eye translation has no stable surface normal."
                )
            translation_normal /= translation_normal_length
            translation_deltas = component_coordinates - center_position
            tangent_deltas = (
                translation_deltas
                - (translation_deltas @ translation_normal)[:, None]
                * translation_normal[None, :]
            )
            distance_from_iris = np.linalg.norm(tangent_deltas, axis=1)
            transform_weights = np.clip(
                (outer_radius - distance_from_iris)
                / max(outer_radius - inner_radius, 1.0e-8),
                0.0,
                1.0,
            )
            transform_weights = (
                transform_weights
                * transform_weights
                * (3.0 - 2.0 * transform_weights)
            )
            transformed_coordinates = (
                component_coordinates
                + transform_weights[:, None] * shared_translation
            )
            output_translation_center = center_position.tolist()
            output_translation_normal = translation_normal.tolist()
            output_translation_inner_radius = float(inner_radius)
            output_translation_outer_radius = float(outer_radius)
            geometry_rotated_vertex_count = int(
                np.count_nonzero(transform_weights > 1.0e-4)
            )
        else:
            transformed_coordinates = (
                (component_coordinates - rotation_center) @ rotation_matrix.T
                + rotation_center
            )
            geometry_rotated_vertex_count = int(len(topology_vertex_ids))
        if geometry_calibration_only:
            geometry_rotated_vertex_count = 0
        else:
            object_coordinates[topology_vertex_ids] = transformed_coordinates
            mesh.vertices.foreach_set("co", object_coordinates.reshape(-1))
            mesh.update(calc_edges=True)
            if hasattr(mesh, "calc_normals"):
                mesh.calc_normals()
            if processed_geometry_components is not None:
                processed_geometry_components.add(geometry_component_key)
        if rotation_center is not None and rotation_matrix is not None:
            output_rotation_center = rotation_center.tolist()
            output_rotation_matrix = rotation_matrix.tolist()
        if shared_translation is not None:
            output_shared_translation = shared_translation.tolist()
    if remove_geometry_component:
        component_loop_count = int(np.count_nonzero(component_loops))
        if component_loop_count > 5_000:
            raise EyeGazeUVWarpValidationError(
                "Pupil geometry component is too large to remove safely: "
                f"loops={component_loop_count}, maximum=5000."
            )
        import bmesh

        mesh = eye_object.data
        component_face_indices = np.flatnonzero(component_polygons)
        bm = bmesh.new()
        try:
            bm.from_mesh(mesh)
            bm.faces.ensure_lookup_table()
            faces = [bm.faces[int(index)] for index in component_face_indices]
            bmesh.ops.delete(bm, geom=faces, context="FACES")
            bm.to_mesh(mesh)
        finally:
            bm.free()
        mesh.update(calc_edges=True)
        removed_polygon_count = int(len(component_face_indices))
    report = {
        "warped_loop_count": float(warped_loop_count),
        "component_loop_count": float(np.count_nonzero(component_loops)),
        "surface_anchor_uv": 1.0,
        "visible_iris_landmark_count": float(len(valid_iris_ids)),
        "maximum_weight": float(weights.max(initial=0.0)),
        "eye_width_uv": float(eye_width_uv),
        "iris_radius_uv": float(iris_radius),
        "iris_position_radius_3d": float(iris_position_radius),
        "iris_horizontal_radius_uv": float(iris_horizontal_radius),
        "iris_vertical_radius_uv": float(iris_vertical_radius),
        "iris_horizontal_axis_u": float(horizontal_axis[0]),
        "iris_horizontal_axis_v": float(horizontal_axis[1]),
        "iris_vertical_axis_u": float(vertical_axis[0]),
        "iris_vertical_axis_v": float(vertical_axis[1]),
        "center_u": float(center_uv[0]),
        "center_v": float(center_uv[1]),
        "corner_width_uv": float(corner_width),
        "eye_to_iris_3d_ratio": float(eye_to_iris_ratio),
        "horizontal_radius_uv": float(horizontal_radius),
        "vertical_radius_uv": float(vertical_radius),
        "rigid_iris_radius": float(rigid_iris_radius),
        "normalized_shift": float(normalized_shift),
        "visual_shift_u": float(visual_shift[0]),
        "visual_shift_v": float(visual_shift[1]),
        "screen_calibrated": float(screen_calibrated),
        "rigid_uv_translation": float(rigid_uv_translation),
        "uv_coordinates_modified": float(
            bool(apply_uv_coordinates or spatial_uv_report is not None)
        ),
        "geometry_vertices_translated": float(
            geometry_vertex_count
            + (geometry_rotated_vertex_count if force_local_eye_translation else 0)
        ),
        "geometry_vertices_rotated": float(
            0 if force_local_eye_translation else geometry_rotated_vertex_count
        ),
        "geometry_component_polygon_count": float(
            geometry_component_polygon_count
        ),
        "geometry_component_min_polygon": (
            float(geometry_component_key[1])
            if geometry_component_key is not None
            else None
        ),
        "geometry_rotation_center": output_rotation_center,
        "geometry_rotation_matrix": output_rotation_matrix,
        "geometry_shared_translation": output_shared_translation,
        "geometry_translation_center": output_translation_center,
        "geometry_translation_normal": output_translation_normal,
        "geometry_translation_inner_radius": output_translation_inner_radius,
        "geometry_translation_outer_radius": output_translation_outer_radius,
        "geometry_rotation_tapered": float(taper_geometry_rotation),
        "normals_recomputed": float(geometry_rotated_vertex_count > 0),
        "geometry_polygons_removed": float(removed_polygon_count),
        "world_shift_x": float(
            geometry_target_shift[0]
            if geometry_target_shift is not None
            else world_shift[0]
        ),
        "world_shift_y": float(
            geometry_target_shift[1]
            if geometry_target_shift is not None
            else world_shift[1]
        ),
        "world_shift_z": float(
            geometry_target_shift[2]
            if geometry_target_shift is not None
            else world_shift[2]
        ),
        "geometry_surface_offset_x": float(geometry_surface_offset[0]),
        "geometry_surface_offset_y": float(geometry_surface_offset[1]),
        "geometry_surface_offset_z": float(geometry_surface_offset[2]),
    }
    if spatial_uv_report is not None:
        report.update(spatial_uv_report)
    if surface_texture_bake_report is not None:
        report.update(surface_texture_bake_report)
    return report


def prepare_source_materials_for_glb(mesh_objects: Sequence[Any]) -> None:
    materials = []
    for obj in mesh_objects:
        for slot in obj.material_slots:
            material = slot.material
            if material is not None and material not in materials:
                materials.append(material)

    for material in materials:
        if not material.use_nodes or material.node_tree is None:
            material.blend_method = "OPAQUE"
            continue
        principled = next(
            (
                node
                for node in material.node_tree.nodes
                if node.type == "BSDF_PRINCIPLED"
            ),
            None,
        )
        if principled is None:
            material.blend_method = "OPAQUE"
            continue

        base_color = principled.inputs.get("Base Color")
        alpha = principled.inputs.get("Alpha")
        base_link = base_color.links[0] if base_color is not None and base_color.links else None
        base_node = base_link.from_node if base_link is not None else None
        alpha_link = alpha.links[0] if alpha is not None and alpha.links else None
        alpha_node = alpha_link.from_node if alpha_link is not None else None
        texture_node = base_node if base_node is not None else alpha_node
        if (
            alpha is None
            or texture_node is None
            or texture_node.type != "TEX_IMAGE"
            or texture_node.image is None
            or texture_node.outputs.get("Alpha") is None
        ):
            material.blend_method = "OPAQUE"
            continue

        for link in tuple(alpha.links):
            material.node_tree.links.remove(link)
        material.node_tree.links.new(texture_node.outputs["Alpha"], alpha)
        for node in tuple(material.node_tree.nodes):
            if node is texture_node or node.type != "TEX_IMAGE":
                continue
            if not any(output.links for output in node.outputs):
                material.node_tree.nodes.remove(node)
        material.blend_method = "CLIP"
        material.alpha_threshold = 0.5


def source_mesh_objects_for_export(
    imported_objects: Sequence[Any],
    mesh_object_names: Optional[Sequence[str]],
) -> tuple[Any, ...]:
    mesh_objects = tuple(obj for obj in imported_objects if obj.type == "MESH")
    if not mesh_objects:
        raise ValueError("No mesh objects were found in the source FBX.")

    requested_names = tuple(str(name) for name in (mesh_object_names or ()))
    if not requested_names:
        keyed = tuple(obj for obj in mesh_objects if obj.data.shape_keys is not None)
        candidates = keyed or mesh_objects
        return tuple(
            sorted(candidates, key=lambda obj: len(obj.data.vertices), reverse=True)
        )

    by_name: dict[str, Any] = {}
    for obj in mesh_objects:
        by_name[obj.name] = obj
        by_name.setdefault(base_blender_object_name(obj.name), obj)
    selected = []
    for name in requested_names:
        obj = by_name.get(name)
        if obj is None:
            raise ValueError(f"Source FBX is missing expected mesh object {name!r}.")
        if obj not in selected:
            selected.append(obj)
    return tuple(selected)


def base_blender_object_name(name: str) -> str:
    suffix = name.rpartition(".")[2]
    if len(suffix) == 3 and suffix.isdigit():
        return name[:-4]
    return name


def glb_vertices_to_blender_axes(vertices: torch.Tensor) -> torch.Tensor:
    """Convert desired glTF coordinates to Blender coordinates before export."""
    return torch.stack(
        (
            vertices[..., 0],
            -vertices[..., 2],
            vertices[..., 1],
        ),
        dim=-1,
    ).contiguous()


def export_mesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    normals: torch.Tensor,
    path: Path,
    vertex_colors: Optional[torch.Tensor] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".obj":
        save_obj(vertices, faces, path)
        return
    if suffix != ".glb":
        raise ValueError("Only GLB and OBJ outputs are supported.")

    try:
        import numpy as np
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "GLB export requires trimesh and numpy. Install them or pass "
            "--output-format obj."
        ) from exc

    mesh = trimesh.Trimesh(
        vertices=vertices.detach().cpu().float().numpy(),
        faces=faces.detach().cpu().long().numpy(),
        vertex_normals=normals.detach().cpu().float().numpy(),
        process=False,
    )
    if vertex_colors is None:
        colors = train.validation_vertex_colors(vertices.unsqueeze(0))[0]
    else:
        colors = vertex_colors.detach().cpu().float()
        if colors.shape != vertices.shape:
            raise ValueError("vertex_colors must have shape [V, 3] matching vertices.")
    rgb = (colors.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).round()
    alpha = np.full((rgb.shape[0], 1), 255.0)
    mesh.visual.vertex_colors = np.concatenate([rgb, alpha], axis=1).astype(np.uint8)
    mesh.export(path)


def save_obj(vertices: torch.Tensor, faces: torch.Tensor, path: Path) -> None:
    vertices = vertices.detach().cpu().float()
    faces = faces.detach().cpu().long()
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Generated by TopoRig demo.py\n")
        for vertex in vertices:
            handle.write(
                f"v {vertex[0].item():.8f} "
                f"{vertex[1].item():.8f} "
                f"{vertex[2].item():.8f}\n"
            )
        for face in faces:
            handle.write(
                f"f {face[0].item() + 1} "
                f"{face[1].item() + 1} "
                f"{face[2].item() + 1}\n"
            )


def recompute_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    vertices = vertices.detach().float()
    faces = faces.detach().long()
    triangles = vertices[faces]
    face_normals = torch.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=-1,
    )

    normals = torch.zeros_like(vertices)
    for corner in range(3):
        normals.index_add_(0, faces[:, corner], face_normals)
    return F.normalize(normals, dim=-1, eps=1.0e-6).contiguous()


if __name__ == "__main__":
    main()
