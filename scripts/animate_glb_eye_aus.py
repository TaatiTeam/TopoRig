#!/usr/bin/env python3
"""Find physical eyes in a textured GLB and animate them from gaze AUs.

The script has two stages:

1. ``map_mediapipe_landmarks.py`` renders the original GLB materials and maps
   refined MediaPipe eye/iris landmarks back to surface ray hits.
2. A Blender worker finds sphere-like connected components behind those iris
   hits, gives each eye a pivot, keys AU-driven rotations, and exports a GLB.

Examples
--------
Animate the left eye looking inward (neutral to full AU14 over one second)::

    python scripts/animate_glb_eye_aus.py \
        --input character.glb --output character_au14.glb \
        --au 14 --intensity 1

Animate a timeline::

    python scripts/animate_glb_eye_aus.py \
        --input character.glb --output character_eye_motion.glb \
        --timeline eye_motion.json

Timeline JSON may be a list of keyframes or an object with ``fps`` and
``keyframes``. Each keyframe accepts ``time`` (seconds) or ``frame`` and either
an ``aus`` mapping or one ``au``/``intensity`` pair::

    {
      "fps": 30,
      "keyframes": [
        {"time": 0.0, "aus": {}},
        {"time": 0.4, "aus": {"14": 1.0, "15": 1.0}},
        {"time": 0.8, "aus": {"18": 0.7, "19": 0.7}},
        {"time": 1.2, "aus": {}}
      ]
    }

Supported project gaze AUs are 12-19:
left/right down, left/right in, left/right out, and left/right up.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPER = ROOT / "map_mediapipe_landmarks.py"
DEFAULT_CACHE_DIR = ROOT / ".cache" / "glb_eye_au_landmarks"
LANDMARK_PIPELINE_VERSION = 5
GLB_LANDMARK_VIEWS = ("neg_z", "pos_z", "neg_y", "pos_y", "neg_x", "pos_x")
GLB_LANDMARK_ATTEMPTS = tuple(
    (view, roll)
    for view in GLB_LANDMARK_VIEWS
    for roll in ((180.0, 0.0) if view in {"neg_z", "pos_z"} else (0.0, 180.0))
)


@dataclass(frozen=True)
class AUSpec:
    eye: str
    motion_x: float
    motion_y: float
    name: str


AU_SPECS: dict[int, AUSpec] = {
    12: AUSpec("left", 0.0, 1.0, "eyeLookDown_L"),
    13: AUSpec("right", 0.0, 1.0, "eyeLookDown_R"),
    14: AUSpec("left", -1.0, 0.0, "eyeLookIn_L"),
    15: AUSpec("right", 1.0, 0.0, "eyeLookIn_R"),
    16: AUSpec("left", 1.0, 0.0, "eyeLookOut_L"),
    17: AUSpec("right", -1.0, 0.0, "eyeLookOut_R"),
    18: AUSpec("left", 0.0, -1.0, "eyeLookUp_L"),
    19: AUSpec("right", 0.0, -1.0, "eyeLookUp_R"),
}

AU_ALIASES: dict[str, int] = {}
for _au_id, _spec in AU_SPECS.items():
    for _alias in (
        str(_au_id),
        f"au{_au_id}",
        _spec.name,
        _spec.name.lower(),
        _spec.name.replace("_", "").lower(),
    ):
        AU_ALIASES[_alias.lower()] = _au_id


EYE_LANDMARKS = {
    "left": {
        "center": 473,
        "iris": (473, 474, 475, 476, 477),
        "corners": (263, 362),
    },
    "right": {
        "center": 468,
        "iris": (468, 469, 470, 471, 472),
        "corners": (33, 133),
    },
}


@dataclass(frozen=True)
class MotionKeyframe:
    time: float
    aus: dict[int, float]


def parse_au_id(value: object) -> int:
    """Parse a numeric AU id or one of the project ARKit-style AU names."""

    normalized = str(value).strip().lower()
    result = AU_ALIASES.get(normalized)
    if result is None:
        supported = ", ".join(
            f"AU{au_id} ({spec.name})" for au_id, spec in AU_SPECS.items()
        )
        raise ValueError(f"Unsupported eye-gaze AU {value!r}. Supported: {supported}.")
    return result


def _finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric; got {value!r}.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite; got {value!r}.")
    return result


def _parse_au_values(raw: object) -> dict[int, float]:
    if raw is None:
        return {}
    values: dict[int, float] = {}
    if isinstance(raw, Mapping):
        entries: Iterable[tuple[object, object]] = raw.items()
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        unpacked: list[tuple[object, object]] = []
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping) or "au" not in item:
                raise ValueError(
                    f"aus[{index}] must contain an 'au' and optional 'intensity'."
                )
            unpacked.append((item["au"], item.get("intensity", 1.0)))
        entries = unpacked
    else:
        raise ValueError("'aus' must be a mapping or a list of AU/intensity objects.")

    for raw_au, raw_intensity in entries:
        au_id = parse_au_id(raw_au)
        intensity = _finite_float(raw_intensity, f"AU{au_id} intensity")
        if abs(intensity) > 1.0:
            raise ValueError(
                f"AU{au_id} intensity must be between -1 and 1; got {intensity}."
            )
        values[au_id] = intensity
    return values


def normalize_timeline_payload(
    payload: object,
    *,
    default_fps: float,
) -> tuple[float, list[MotionKeyframe]]:
    """Normalize the documented timeline formats into sorted keyframes."""

    fps = _finite_float(default_fps, "fps")
    if isinstance(payload, Mapping):
        fps = _finite_float(payload.get("fps", fps), "fps")
        raw_keyframes = payload.get("keyframes", payload.get("frames"))
    else:
        raw_keyframes = payload
    if fps <= 0.0:
        raise ValueError("fps must be positive.")
    if not isinstance(raw_keyframes, Sequence) or isinstance(
        raw_keyframes, (str, bytes)
    ):
        raise ValueError("Timeline JSON must contain a keyframe list.")
    if not raw_keyframes:
        raise ValueError("Timeline JSON contains no keyframes.")

    keyframes: list[MotionKeyframe] = []
    for index, item in enumerate(raw_keyframes):
        if not isinstance(item, Mapping):
            raise ValueError(f"Keyframe {index} must be an object.")
        if "time" in item:
            time = _finite_float(item["time"], f"keyframe {index} time")
        elif "frame" in item:
            frame = _finite_float(item["frame"], f"keyframe {index} frame")
            time = frame / fps
        else:
            raise ValueError(f"Keyframe {index} needs 'time' or 'frame'.")
        if time < 0.0:
            raise ValueError(f"Keyframe {index} time cannot be negative.")

        if "aus" in item:
            aus = _parse_au_values(item["aus"])
        elif "au" in item:
            au_id = parse_au_id(item["au"])
            aus = _parse_au_values({au_id: item.get("intensity", 1.0)})
        else:
            aus = {}
        keyframes.append(MotionKeyframe(time=time, aus=aus))

    keyframes.sort(key=lambda keyframe: keyframe.time)
    for first, second in zip(keyframes, keyframes[1:]):
        if math.isclose(first.time, second.time, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(f"Timeline contains duplicate keyframe time {first.time}.")
    if keyframes[0].time > 0.0:
        keyframes.insert(0, MotionKeyframe(time=0.0, aus={}))
    return fps, keyframes


def single_au_timeline(
    au: object,
    *,
    intensity: float,
    duration: float,
    fps: float,
) -> tuple[float, list[MotionKeyframe]]:
    au_id = parse_au_id(au)
    intensity_value = _finite_float(intensity, "intensity")
    duration_value = _finite_float(duration, "duration")
    if not -1.0 <= intensity_value <= 1.0:
        raise ValueError("intensity must be between -1 and 1.")
    if duration_value <= 0.0:
        raise ValueError("duration must be positive.")
    fps_value = _finite_float(fps, "fps")
    if fps_value <= 0.0:
        raise ValueError("fps must be positive.")
    return fps_value, [
        MotionKeyframe(time=0.0, aus={}),
        MotionKeyframe(time=duration_value, aus={au_id: intensity_value}),
    ]


def serialize_timeline(fps: float, keyframes: Sequence[MotionKeyframe]) -> dict[str, Any]:
    return {
        "fps": float(fps),
        "keyframes": [
            {
                "time": float(keyframe.time),
                "aus": {str(key): float(value) for key, value in keyframe.aus.items()},
            }
            for keyframe in keyframes
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect physical eyes from textured GLB iris landmarks and export "
            "AU-driven eye rotations as GLB animation."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Input .glb file.")
    parser.add_argument("--output", type=Path, required=True, help="Animated output .glb.")
    motion = parser.add_mutually_exclusive_group(required=True)
    motion.add_argument("--au", help="AU id or name for a neutral-to-AU animation.")
    motion.add_argument("--timeline", type=Path, help="JSON AU animation timeline.")
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--max-angle",
        type=float,
        default=25.0,
        help="Rotation in degrees produced by AU intensity 1 (default: 25).",
    )
    parser.add_argument(
        "--landmarks",
        type=Path,
        help="Existing versioned surface-landmark JSON; skips landmark detection.",
    )
    parser.add_argument("--mapper", type=Path, default=DEFAULT_MAPPER)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--refresh-landmarks", action="store_true")
    parser.add_argument("--surface-hits-only", action="store_true",
                        help="Skip projected-vertex fallback during eye landmark detection.")
    parser.add_argument("--blender", help="Blender executable; defaults to PATH.")
    parser.add_argument("--blender-threads", type=int)
    parser.add_argument(
        "--left-eye-object",
        help="Optional Blender object-name override for the left physical eye.",
    )
    parser.add_argument(
        "--right-eye-object",
        help="Optional Blender object-name override for the right physical eye.",
    )
    parser.add_argument(
        "--sclera-backing",
        choices=("auto", "always", "off"),
        default="auto",
        help=(
            "Add a slightly inset off-white sphere behind open/partial eye "
            "shells. 'always' also infers a spherical pivot from a visible outer "
            "cap when no complete eyeball exists and builds opaque sclera, iris, "
            "and pupil material regions on one complete spherical mesh. 'auto' "
            "is the safe default; use 'off' to preserve only source geometry."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep landmark debug renders and include component metrics in the report.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="JSON report path (default: <output>.eye_animation.json).",
    )
    return parser.parse_args(argv)


def _resolve_blender(value: str | None) -> str:
    if value:
        candidate = Path(value).expanduser().resolve()
        if candidate.is_file():
            return str(candidate)
        executable = shutil.which(value)
        if executable is not None:
            return executable
        raise FileNotFoundError(f"Blender executable not found: {value}")
    executable = shutil.which("blender")
    if executable is None:
        raise FileNotFoundError(
            "Blender was not found on PATH. Pass --blender /path/to/blender."
        )
    return executable


def _run_checked(command: list[str], label: str) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    if process.returncode != 0:
        details = "\n".join(
            part.strip() for part in (process.stdout, process.stderr) if part.strip()
        )
        raise RuntimeError(f"{label} failed.\n{details}")
    return process


def _landmark_cache_path(mesh: Path, cache_dir: Path) -> Path:
    stat = mesh.stat()
    fingerprint = "|".join(
        (
            str(LANDMARK_PIPELINE_VERSION),
            str(mesh.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
        )
    )
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{mesh.stem}_{digest}.eye_landmarks.json"


def _validate_landmarks(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Landmark file is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Landmark file must contain an object: {path}")
    if not isinstance(payload.get("surface_anchors"), dict):
        raise ValueError(
            f"{path} has no surface_anchors. Generate it with "
            "map_mediapipe_landmarks.py --output-surface-anchors."
        )
    anchors = payload["surface_anchors"]
    required = {
        str(value)
        for eye in EYE_LANDMARKS.values()
        for value in (*eye["iris"], *eye["corners"])
    }
    missing = sorted(value for value in required if anchors.get(value) is None)
    selections = payload.get("surface_anchor_selection")
    selected_eyes = {
        side
        for side in EYE_LANDMARKS
        if isinstance(selections, Mapping)
        and isinstance(selections.get(side), Mapping)
    }
    if missing and len(selected_eyes) < 2:
        raise ValueError(
            f"{path} is missing usable iris/corner anchors {missing} and does not "
            "contain selected visible eye layers."
        )
    return payload


def _landmark_eye_quality(payload: Mapping[str, Any]) -> tuple[float, bool]:
    """Score whether a rendered view found two coherent textured irises.

    MediaPipe can occasionally report a face on a head viewed from above.  A
    large 2D face box alone cannot reject that false positive, but two dark,
    anatomically scaled iris layers at coherent ray depths can.
    """

    selections = payload.get("surface_anchor_selection", {})
    if not isinstance(selections, Mapping):
        return 0.0, False
    score = 0.0
    resolved = 0
    visible_consensus = 0
    for side in ("left", "right"):
        selection = selections.get(side, {})
        if not isinstance(selection, Mapping):
            continue
        status = str(selection.get("status", ""))
        if not status.startswith("selected"):
            continue
        try:
            ratio = float(selection["eye_to_iris_ratio"])
            eye_width = float(selection["eye_width_3d"])
            iris_radius = float(selection["iris_radius_3d"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (
            math.isfinite(ratio)
            and math.isfinite(eye_width)
            and math.isfinite(iris_radius)
            and 2.5 <= ratio <= 8.0
            and eye_width > 0.0
            and iris_radius > 0.0
        ):
            continue
        resolved += 1
        score += 20.0 - abs(ratio - 4.5)
        if status == "selected_visible_consensus":
            visible_consensus += 1
            score += 50.0
            layers = selection.get("visible_layers", [])
            if isinstance(layers, Sequence):
                score += min(len(layers), 4) * 2.0
    return score, resolved == 2 and visible_consensus == 2


def ensure_landmarks(args: argparse.Namespace, blender: str, mesh: Path) -> Path:
    if args.landmarks is not None:
        path = args.landmarks.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        _validate_landmarks(path)
        return path

    mapper = args.mapper.expanduser().resolve()
    if not mapper.is_file():
        raise FileNotFoundError(mapper)
    cache_dir = args.cache_dir.expanduser().resolve()
    path = _landmark_cache_path(mesh, cache_dir)
    if path.is_file() and not args.refresh_landmarks:
        _validate_landmarks(path)
        print(f"Using cached eye landmarks: {path}", flush=True)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print("Detecting face and iris landmarks from the textured GLB...", flush=True)
    attempts: list[str] = []
    usable: list[tuple[float, Path, str, str, bool]] = []
    # Most meshes should use MediaPipe's normal threshold. Stylized or
    # upside-down heads can still be valid but score just below it, so retry
    # only after the complete normal pass. The two-eye landmark-quality gate
    # and Blender geometry compatibility check remain mandatory.
    landmark_passes = ((None, False), (0.10, True))
    for min_detection_confidence, low_confidence in landmark_passes:
      for view, roll_offset in GLB_LANDMARK_ATTEMPTS:
        roll_label = f"roll{int(roll_offset):03d}"
        confidence_label = "/lowconf" if low_confidence else ""
        attempt_label = f"{view}/{roll_label}{confidence_label}"
        candidate_path = path.with_name(
            f".{path.stem}.{view}.{roll_label}"
            f"{'.lowconf' if low_confidence else ''}.candidate.json"
        )
        candidate_path.unlink(missing_ok=True)
        command = [
            sys.executable,
            str(mapper),
            "--mesh",
            str(mesh),
            "--output",
            str(candidate_path),
            "--views",
            view,
            "--blender",
            blender,
            "--refine-landmarks",
            "--preserve-materials",
            "--output-surface-anchors",
            "--no-debug-glb",
            "--roll-offset",
            str(roll_offset),
        ]
        if args.blender_threads is not None:
            command.extend(("--blender-threads", str(args.blender_threads)))
        if min_detection_confidence is not None:
            command.extend(
                (
                    "--min-detection-confidence",
                    str(min_detection_confidence),
                )
            )
        if args.debug:
            command.append("--debug")
        if getattr(args, "surface_hits_only", False):
            command.append("--surface-hits-only")
        print(f"  trying landmark view {attempt_label}...", flush=True)
        try:
            process = _run_checked(command, f"Eye landmark detection ({view})")
            payload = _validate_landmarks(candidate_path)
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            attempts.append(f"{attempt_label}: {exc}")
            candidate_path.unlink(missing_ok=True)
            continue
        quality, strong = _landmark_eye_quality(payload)
        selected_view = str(payload.get("selected_view", view))
        usable.append(
            (
                quality,
                candidate_path,
                selected_view,
                process.stdout.strip(),
                low_confidence,
            )
        )
        if strong:
            candidate_path.replace(path)
            for _quality, other_path, _view, _stdout, _low in usable[:-1]:
                other_path.unlink(missing_ok=True)
            if process.stdout.strip():
                print(process.stdout.strip(), flush=True)
            return path

    if usable:
        eligible = [
            item
            for item in usable
            if item[0] >= (38.0 if item[4] else 40.0)
        ]
        best = max(eligible, key=lambda item: item[0]) if eligible else None
        for _quality, candidate_path, _view, _stdout, _low in usable:
            if best is not None and candidate_path == best[1]:
                continue
            candidate_path.unlink(missing_ok=True)
        if best is not None:
            quality, best_path, selected_view, stdout, low_confidence = best
            best_path.replace(path)
            if stdout:
                print(stdout, flush=True)
            confidence_note = "low-confidence " if low_confidence else ""
            print(
                f"Warning: using {confidence_note}non-texture-validated "
                f"landmark view {selected_view}; physical eye fitting will "
                "perform the final safety check.",
                flush=True,
            )
            return path
    details = "\n".join(attempts[-6:])
    raise RuntimeError(
        "No cardinal view produced coherent landmarks for both eyes."
        + (f"\n{details}" if details else "")
    )


def _worker_parser(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--_blender-worker", action="store_true")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--worker-report", type=Path, required=True)
    parser.add_argument("--max-angle", type=float, required=True)
    parser.add_argument("--left-eye-object")
    parser.add_argument("--right-eye-object")
    parser.add_argument(
        "--sclera-backing", choices=("auto", "always", "off"), required=True
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def _worker_command_prefix(blender: str, threads: int | None) -> list[str]:
    command = [blender, "--background"]
    if threads is not None:
        if threads < 1:
            raise ValueError("--blender-threads must be positive.")
        command.extend(("--threads", str(threads)))
    return command


def run_blender_worker(
    args: argparse.Namespace,
    *,
    blender: str,
    mesh: Path,
    output: Path,
    landmarks: Path,
    fps: float,
    keyframes: Sequence[MotionKeyframe],
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="toporig_eye_au_") as temp_name:
        temp_dir = Path(temp_name)
        timeline_path = temp_dir / "timeline.json"
        worker_report = temp_dir / "worker_report.json"
        timeline_path.write_text(
            json.dumps(serialize_timeline(fps, keyframes), indent=2) + "\n",
            encoding="utf-8",
        )
        command = [
            *_worker_command_prefix(blender, args.blender_threads),
            "--python",
            str(Path(__file__).resolve()),
            "--",
            "--_blender-worker",
            "--input",
            str(mesh),
            "--output",
            str(output),
            "--landmarks",
            str(landmarks),
            "--timeline",
            str(timeline_path),
            "--worker-report",
            str(worker_report),
            "--max-angle",
            str(args.max_angle),
            "--sclera-backing",
            str(args.sclera_backing),
        ]
        if args.left_eye_object:
            command.extend(("--left-eye-object", args.left_eye_object))
        if args.right_eye_object:
            command.extend(("--right-eye-object", args.right_eye_object))
        if args.debug:
            command.append("--debug")
        print("Fitting eyeballs and writing AU rotation animation...", flush=True)
        process = _run_checked(command, "Blender eye animation export")
        if process.stdout.strip():
            lines = [
                line
                for line in process.stdout.splitlines()
                if line.startswith("[eye-au]")
            ]
            if lines:
                print("\n".join(lines), flush=True)
        if not worker_report.is_file():
            raise RuntimeError("Blender finished without writing its eye-animation report.")
        return json.loads(worker_report.read_text(encoding="utf-8"))


def outer_main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    mesh = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not mesh.is_file():
        raise FileNotFoundError(mesh)
    if mesh.suffix.lower() != ".glb":
        raise ValueError(f"Input must be a .glb file; got {mesh.suffix!r}.")
    if output.suffix.lower() != ".glb":
        raise ValueError(f"Output must be a .glb file; got {output.suffix!r}.")
    if mesh == output:
        raise ValueError("Input and output paths must be different.")
    if not math.isfinite(args.max_angle) or not 0.0 < args.max_angle <= 35.0:
        raise ValueError("--max-angle must be greater than 0 and no more than 35 degrees.")

    if args.timeline is not None:
        timeline_path = args.timeline.expanduser().resolve()
        if not timeline_path.is_file():
            raise FileNotFoundError(timeline_path)
        payload = json.loads(timeline_path.read_text(encoding="utf-8"))
        fps, keyframes = normalize_timeline_payload(payload, default_fps=args.fps)
    else:
        fps, keyframes = single_au_timeline(
            args.au,
            intensity=args.intensity,
            duration=args.duration,
            fps=args.fps,
        )

    blender = _resolve_blender(args.blender)
    landmarks = ensure_landmarks(args, blender, mesh)
    worker_report = run_blender_worker(
        args,
        blender=blender,
        mesh=mesh,
        output=output,
        landmarks=landmarks,
        fps=fps,
        keyframes=keyframes,
    )
    if not output.is_file():
        raise RuntimeError(f"Blender did not write the requested output: {output}")

    report_path = (
        args.report.expanduser().resolve()
        if args.report is not None
        else output.with_suffix(".eye_animation.json")
    )
    report = {
        "status": "exported",
        "input": str(mesh),
        "output": str(output),
        "landmarks": str(landmarks),
        "fps": fps,
        "max_angle_degrees": args.max_angle,
        "keyframes": serialize_timeline(fps, keyframes)["keyframes"],
        **worker_report,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote animated GLB: {output}")
    print(f"Wrote detection report: {report_path}")


# Everything below this line runs inside Blender. Keeping bpy imports local lets
# the command-line/timeline code be tested with ordinary Python.


@dataclass
class _Component:
    obj: Any
    polygon_ids: tuple[int, ...]
    vertex_ids: tuple[int, ...]
    seed_polygon: int


@dataclass
class _Candidate:
    component: _Component
    center: Any
    radius: float
    extent: float
    residual: float
    closedness: float
    shell_error: float
    radius_ratio: float
    score: float
    plausible: bool
    reason: str
    ray_hits: tuple[tuple[int, float, float, tuple[float, float, float]], ...] = ()
    pivot_inferred: bool = False


class _ComponentIndex:
    def __init__(self, obj: Any):
        self.obj = obj
        mesh = obj.data
        self.vertex_to_polygons: list[list[int]] = [[] for _ in mesh.vertices]
        for polygon in mesh.polygons:
            for vertex_id in polygon.vertices:
                self.vertex_to_polygons[int(vertex_id)].append(int(polygon.index))
        self.polygon_to_component: dict[int, _Component] = {}
        self.components: list[_Component] = []
        self.fully_indexed = False

    def component(self, seed_polygon: int) -> _Component:
        existing = self.polygon_to_component.get(int(seed_polygon))
        if existing is not None:
            return existing
        mesh = self.obj.data
        if seed_polygon < 0 or seed_polygon >= len(mesh.polygons):
            raise ValueError(
                f"Polygon {seed_polygon} is out of range for {self.obj.name!r}."
            )
        selected: set[int] = {int(seed_polygon)}
        vertices: set[int] = set()
        pending = [int(seed_polygon)]
        while pending:
            polygon_id = pending.pop()
            polygon = mesh.polygons[polygon_id]
            for raw_vertex_id in polygon.vertices:
                vertex_id = int(raw_vertex_id)
                vertices.add(vertex_id)
                for neighbor in self.vertex_to_polygons[vertex_id]:
                    if neighbor not in selected:
                        selected.add(neighbor)
                        pending.append(neighbor)
        result = _Component(
            obj=self.obj,
            polygon_ids=tuple(sorted(selected)),
            vertex_ids=tuple(sorted(vertices)),
            seed_polygon=int(seed_polygon),
        )
        self.components.append(result)
        for polygon_id in selected:
            self.polygon_to_component[polygon_id] = result
        return result

    def all_components(self) -> Iterable[_Component]:
        if not self.fully_indexed:
            for polygon_id in range(len(self.obj.data.polygons)):
                if polygon_id not in self.polygon_to_component:
                    self.component(polygon_id)
            self.fully_indexed = True
        yield from self.components


def _base_name(name: str) -> str:
    prefix, separator, suffix = name.rpartition(".")
    return prefix if separator and len(suffix) == 3 and suffix.isdigit() else name


def _object_lookup(objects: Sequence[Any], name: str) -> Any | None:
    for obj in objects:
        if obj.name == name or _base_name(obj.name) == _base_name(name):
            return obj
    return None


def _anchor_vector(anchor: object) -> Any | None:
    from mathutils import Vector

    if not isinstance(anchor, Mapping):
        return None
    position = anchor.get("position")
    if not isinstance(position, Mapping):
        return None
    try:
        result = Vector(
            (float(position["x"]), float(position["y"]), float(position["z"]))
        )
    except (KeyError, TypeError, ValueError):
        return None
    return result if all(math.isfinite(value) for value in result) else None


def _eye_selection(payload: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    selections = payload.get("surface_anchor_selection", {})
    selection = selections.get(side, {}) if isinstance(selections, Mapping) else {}
    return selection if isinstance(selection, Mapping) else {}


def _preferred_eye_anchors(
    payload: Mapping[str, Any], side: str
) -> dict[str, Mapping[str, Any]]:
    selection = _eye_selection(payload, side)
    layers = selection.get("visible_layers", [])
    if isinstance(layers, Sequence):
        for layer in layers:
            if isinstance(layer, Mapping) and isinstance(
                layer.get("surface_anchors"), Mapping
            ):
                return dict(layer["surface_anchors"])
    raw = payload.get("surface_anchors", {})
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(landmark_id): raw[str(landmark_id)]
        for landmark_id in EYE_LANDMARKS[side]["iris"]
        if isinstance(raw.get(str(landmark_id)), Mapping)
    }


def _eye_geometry(payload: Mapping[str, Any], side: str) -> dict[str, Any]:
    from mathutils import Vector

    center_id = str(EYE_LANDMARKS[side]["center"])
    raw_candidates = payload.get("iris_surface_anchor_candidates", {})
    center_candidates = (
        raw_candidates.get(center_id, [])
        if isinstance(raw_candidates, Mapping)
        else []
    )
    # Candidate lists are ordered front-to-back along the render ray.  The
    # front hit is the physical visible iris/cornea surface.  Choosing only the
    # darkest coherent layer can select a duplicated iris texture on the rear
    # of some Pixel3D heads (notably identity 00007).
    iris_center = (
        _anchor_vector(center_candidates[0])
        if isinstance(center_candidates, Sequence) and center_candidates
        else None
    )
    anchors = _preferred_eye_anchors(payload, side)
    if iris_center is None:
        iris_center = _anchor_vector(anchors.get(center_id))
    if iris_center is None:
        positions = [
            point
            for point in (_anchor_vector(value) for value in anchors.values())
            if point is not None
        ]
        if not positions:
            raise ValueError(f"No usable {side} iris surface positions were found.")
        iris_center = sum(positions, Vector()) / len(positions)

    raw_anchors = payload.get("surface_anchors", {})
    corner_points = [
        _anchor_vector(raw_anchors.get(str(landmark_id)))
        for landmark_id in EYE_LANDMARKS[side]["corners"]
    ]
    if any(point is None for point in corner_points):
        eye_width = _finite_float(
            _eye_selection(payload, side).get("eye_width_3d"),
            f"{side} eye width",
        )
    else:
        eye_width = (corner_points[0] - corner_points[1]).length
    if not math.isfinite(eye_width) or eye_width <= 1.0e-8:
        raise ValueError(f"The detected {side} eye width is invalid.")
    iris_edge_points = []
    for landmark_id in EYE_LANDMARKS[side]["iris"][1:]:
        candidate_records = (
            raw_candidates.get(str(landmark_id), [])
            if isinstance(raw_candidates, Mapping)
            else []
        )
        point = (
            _anchor_vector(candidate_records[0])
            if isinstance(candidate_records, Sequence) and candidate_records
            else _anchor_vector(anchors.get(str(landmark_id)))
        )
        if point is not None:
            iris_edge_points.append(point)
    iris_radii = sorted(
        (point - iris_center).length for point in iris_edge_points
    )
    if iris_radii:
        midpoint = len(iris_radii) // 2
        iris_radius = (
            iris_radii[midpoint]
            if len(iris_radii) % 2
            else (iris_radii[midpoint - 1] + iris_radii[midpoint]) * 0.5
        )
    else:
        iris_radius = eye_width * 0.22
    iris_radius = min(max(float(iris_radius), eye_width * 0.08), eye_width * 0.35)
    return {
        "iris_center": iris_center,
        "iris_radius": iris_radius,
        "eye_width": float(eye_width),
    }


def _validate_landmark_geometry_compatibility(
    objects: Sequence[Any], geometry: Mapping[str, Mapping[str, Any]]
) -> None:
    """Reject landmark coordinates that are outside the imported GLB bounds."""

    from mathutils import Vector

    corners = [
        obj.matrix_world @ Vector(corner)
        for obj in objects
        if obj.type == "MESH"
        for corner in obj.bound_box
    ]
    if not corners:
        raise ValueError("The imported GLB has no mesh bounds for landmark validation.")
    minimum = Vector(
        tuple(min(float(corner[index]) for corner in corners) for index in range(3))
    )
    maximum = Vector(
        tuple(max(float(corner[index]) for corner in corners) for index in range(3))
    )
    maximum_extent = max(
        float(maximum.x - minimum.x),
        float(maximum.y - minimum.y),
        float(maximum.z - minimum.z),
        1.0e-6,
    )
    margin = maximum_extent * 0.06
    incompatible = []
    for side in ("left", "right"):
        center = geometry[side]["iris_center"]
        if any(
            float(center[index]) < float(minimum[index]) - margin
            or float(center[index]) > float(maximum[index]) + margin
            for index in range(3)
        ):
            incompatible.append(side)
    if incompatible:
        raise ValueError(
            "Landmark coordinates do not match the imported GLB for "
            f"{', '.join(incompatible)} eye(s). The landmark file likely came "
            "from an FBX or GLB with a different axis/topology conversion. "
            "Remove --landmarks and let this GLB be detected directly."
        )


def _seed_anchor_records(
    payload: Mapping[str, Any], side: str
) -> list[tuple[int, Mapping[str, Any]]]:
    results: list[tuple[int, Mapping[str, Any]]] = []
    seen: set[tuple[int, str, int, float]] = set()

    def add(landmark_id: object, anchor: object) -> None:
        if not isinstance(anchor, Mapping):
            return
        try:
            iris_id = int(landmark_id)
            key = (
                iris_id,
                str(anchor["object_name"]),
                int(anchor["polygon_index"]),
                float(anchor.get("ray_depth", float("nan"))),
            )
        except (KeyError, TypeError, ValueError):
            return
        if key not in seen:
            seen.add(key)
            results.append((iris_id, anchor))

    for landmark_id, anchor in _preferred_eye_anchors(payload, side).items():
        add(landmark_id, anchor)
    candidates = payload.get("iris_surface_anchor_candidates", {})
    if isinstance(candidates, Mapping):
        for landmark_id in EYE_LANDMARKS[side]["iris"]:
            raw = candidates.get(str(landmark_id), [])
            if isinstance(raw, Sequence):
                for anchor in raw:
                    add(landmark_id, anchor)
    raw_anchors = payload.get("surface_anchors", {})
    if isinstance(raw_anchors, Mapping):
        for landmark_id in EYE_LANDMARKS[side]["iris"]:
            add(landmark_id, raw_anchors.get(str(landmark_id)))
    return results


def _seed_anchors(payload: Mapping[str, Any], side: str) -> list[Mapping[str, Any]]:
    return [anchor for _landmark_id, anchor in _seed_anchor_records(payload, side)]


def _anchor_ray_hit(
    landmark_id: int, anchor: Mapping[str, Any]
) -> tuple[int, float, float, tuple[float, float, float]]:
    try:
        depth = float(anchor.get("ray_depth", float("nan")))
    except (TypeError, ValueError):
        depth = float("nan")
    try:
        luminance = float(anchor.get("base_color_luminance", float("nan")))
    except (TypeError, ValueError):
        luminance = float("nan")
    raw_rgb = anchor.get("base_color_rgb", ())
    try:
        rgb = tuple(float(raw_rgb[index]) for index in range(3))
    except (IndexError, TypeError, ValueError):
        rgb = (float("nan"),) * 3
    return int(landmark_id), depth, luminance, rgb


def _fit_sphere(points: Any) -> tuple[Any, float, float]:
    import numpy as np
    from mathutils import Vector

    values = np.asarray(points, dtype=np.float64)
    if len(values) > 20_000:
        sample_ids = np.linspace(0, len(values) - 1, 20_000, dtype=np.int64)
        values = values[sample_ids]
    if values.ndim != 2 or values.shape[0] < 6 or values.shape[1] != 3:
        return Vector((float("nan"),) * 3), float("nan"), float("inf")
    matrix = np.concatenate(
        (2.0 * values, np.ones((values.shape[0], 1), dtype=np.float64)), axis=1
    )
    try:
        solution, _residuals, rank, _singular = np.linalg.lstsq(
            matrix, np.sum(values * values, axis=1), rcond=None
        )
    except np.linalg.LinAlgError:
        rank = 0
    if rank < 4:
        return Vector((float("nan"),) * 3), float("nan"), float("inf")
    center_values = solution[:3]
    radii = np.linalg.norm(values - center_values, axis=1)
    radius = float(np.median(radii))
    residual = float(
        np.sqrt(np.mean(np.square(radii - radius))) / max(radius, 1.0e-12)
    )
    return Vector(center_values.tolist()), radius, residual


def _closedness(component: _Component) -> float:
    edge_counts: dict[tuple[int, int], int] = {}
    mesh = component.obj.data
    for polygon_id in component.polygon_ids:
        vertices = [int(value) for value in mesh.polygons[polygon_id].vertices]
        for index, first in enumerate(vertices):
            second = vertices[(index + 1) % len(vertices)]
            edge = (first, second) if first < second else (second, first)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    if not edge_counts:
        return 0.0
    invalid = sum(1 for count in edge_counts.values() if count != 2)
    return max(0.0, 1.0 - invalid / len(edge_counts))


def _score_component(
    component: _Component,
    *,
    iris_center: Any,
    eye_width: float,
    forced: bool,
    center_hint: Any | None = None,
    radius_hint: float | None = None,
) -> _Candidate:
    import numpy as np

    matrix = component.obj.matrix_world
    points = np.asarray(
        [tuple(matrix @ component.obj.data.vertices[index].co) for index in component.vertex_ids],
        dtype=np.float64,
    )
    if len(points) < 6:
        extent = 0.0
    else:
        extent = float(np.ptp(points, axis=0).max())
    if not forced and (extent < eye_width * 0.12 or extent > eye_width * 2.2):
        return _Candidate(
            component,
            iris_center.copy(),
            0.0,
            extent,
            float("inf"),
            0.0,
            float("inf"),
            0.0,
            float("inf"),
            False,
            "component extent is inconsistent with an eyeball",
        )

    center, radius, residual = _fit_sphere(points)
    if not math.isfinite(radius) or radius <= 1.0e-10:
        return _Candidate(
            component,
            center,
            radius,
            extent,
            residual,
            0.0,
            float("inf"),
            0.0,
            float("inf"),
            False,
            "sphere fit is degenerate",
        )
    closedness = _closedness(component)
    radius_ratio = radius / eye_width
    shell_error = abs((iris_center - center).length - radius) / radius
    center_hint_error = (
        (center - center_hint).length / eye_width
        if center_hint is not None
        else float("inf")
    )
    name = f"{component.obj.name} {component.obj.data.name}".lower()
    name_bonus = 0.0
    if "eye" in name:
        name_bonus += 0.35
    if any(token in name for token in ("ball", "sclera", "cornea")):
        name_bonus += 0.25
    if any(token in name for token in ("lash", "lid", "brow")):
        name_bonus -= 0.75
    if center_hint is None:
        score = (
            residual * 4.0
            + abs(math.log(max(radius_ratio, 1.0e-8) / 0.38))
            + shell_error * 2.5
            + (1.0 - closedness) * 1.5
            - name_bonus
        )
    else:
        expected_radius = max(float(radius_hint or eye_width * 0.75), 1.0e-10)
        score = (
            residual * 4.0
            + abs(math.log(max(radius, 1.0e-10) / expected_radius))
            + center_hint_error * 2.0
            + (1.0 - closedness) * 1.5
            - name_bonus
        )
    closed_surface_safe = closedness >= 0.60 or (
        closedness >= 0.30 and residual <= 0.08 and shell_error <= 0.18
    )
    radius_safe = (
        0.50 <= radius / max(float(radius_hint), 1.0e-10) <= 1.50
        if center_hint is not None and radius_hint is not None
        else 0.12 <= radius_ratio <= 1.0
    )
    location_safe = center_hint_error <= 0.75 if center_hint is not None else shell_error <= 0.65
    plausible = (
        (forced or radius_safe)
        and residual <= (0.45 if forced else 0.30)
        and (forced or closed_surface_safe)
        and (forced or location_safe)
        and len(component.polygon_ids) >= 4
    )
    reasons = []
    if not radius_safe:
        reasons.append(f"radius/eye-width={radius_ratio:.3f}")
    if residual > (0.45 if forced else 0.30):
        reasons.append(f"sphere residual={residual:.3f}")
    if not closed_surface_safe:
        reasons.append(f"closedness={closedness:.3f}")
    if center_hint is not None and center_hint_error > 0.75:
        reasons.append(f"bilateral center error={center_hint_error:.3f}")
    elif center_hint is None and shell_error > 0.65:
        reasons.append(f"iris shell error={shell_error:.3f}")
    if len(component.polygon_ids) < 4:
        reasons.append(f"polygon count={len(component.polygon_ids)}")
    return _Candidate(
        component=component,
        center=center,
        radius=radius,
        extent=extent,
        residual=residual,
        closedness=closedness,
        shell_error=shell_error,
        radius_ratio=radius_ratio,
        score=float(score),
        plausible=bool(plausible),
        reason=", ".join(reasons) if reasons else "accepted",
    )


def _find_eye_candidates(
    *,
    objects: Sequence[Any],
    indices: dict[str, _ComponentIndex],
    payload: Mapping[str, Any],
    side: str,
    geometry: Mapping[str, Any],
    object_override: str | None,
) -> list[_Candidate]:
    components: dict[tuple[str, int], _Component] = {}
    component_hits: dict[
        tuple[str, int],
        list[tuple[int, float, float, tuple[float, float, float]]],
    ] = {}
    forced_names: set[str] = set()
    if object_override:
        obj = _object_lookup(objects, object_override)
        if obj is None:
            available = ", ".join(sorted(item.name for item in objects))
            raise ValueError(
                f"Requested {side} eye object {object_override!r} was not found. "
                f"Mesh objects: {available}."
            )
        forced_names.add(obj.name)
        index = indices.setdefault(obj.name, _ComponentIndex(obj))
        for component in index.all_components():
            components[(obj.name, component.seed_polygon)] = component
    else:
        for landmark_id, anchor in _seed_anchor_records(payload, side):
            obj = _object_lookup(objects, str(anchor.get("object_name", "")))
            if obj is None:
                continue
            try:
                polygon_id = int(anchor["polygon_index"])
                index = indices.setdefault(obj.name, _ComponentIndex(obj))
                component = index.component(polygon_id)
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            key = (obj.name, component.seed_polygon)
            components[key] = component
            hit = _anchor_ray_hit(landmark_id, anchor)
            if hit not in component_hits.setdefault(key, []):
                component_hits[key].append(hit)

        # Named eye objects are useful when transparent corneas prevent a ray
        # layer from being retained in the landmark payload.
        side_tokens = ("left", "_l", ".l") if side == "left" else ("right", "_r", ".r")
        for obj in objects:
            lowered = obj.name.lower()
            if "eye" not in lowered or not any(token in lowered for token in side_tokens):
                continue
            index = indices.setdefault(obj.name, _ComponentIndex(obj))
            for component in index.all_components():
                components[(obj.name, component.seed_polygon)] = component

    candidates = []
    for key, component in components.items():
        candidate = _score_component(
            component,
            iris_center=geometry["iris_center"],
            eye_width=float(geometry["eye_width"]),
            forced=component.obj.name in forced_names,
        )
        candidate.ray_hits = tuple(component_hits.get(key, ()))
        candidates.append(candidate)
    candidates.sort(key=lambda candidate: candidate.score)
    return candidates


def _eye_assembly_is_complete(candidates: Sequence[_Candidate]) -> bool:
    if not candidates:
        return False
    pivot = candidates[0]
    return (
        pivot.plausible
        and len(pivot.component.polygon_ids) >= 20
        and len(pivot.component.vertex_ids) >= 16
        and pivot.closedness >= 0.55
        and pivot.residual <= 0.12
    )


def _bilateral_center_hint(
    *,
    source_center: Any,
    payload: Mapping[str, Any],
    geometry: Mapping[str, Mapping[str, Any]],
) -> Any:
    """Reflect a fitted eye center across the cardinal-view face midline."""

    view = str(payload.get("selected_view", "neg_z"))
    axis_index = 1 if view in {"neg_x", "pos_x"} else 0
    midpoint = (
        geometry["left"]["iris_center"] + geometry["right"]["iris_center"]
    ) * 0.5
    result = source_center.copy()
    result[axis_index] = 2.0 * midpoint[axis_index] - source_center[axis_index]
    return result


def _find_bilateral_eye_candidates(
    *,
    objects: Sequence[Any],
    indices: dict[str, _ComponentIndex],
    payload: Mapping[str, Any],
    side: str,
    geometry: Mapping[str, Any],
    center_hint: Any,
    radius_hint: float,
    existing: Sequence[_Candidate],
) -> list[_Candidate]:
    """Find all compact components around a center mirrored from the other eye."""

    import numpy as np

    existing_keys = {
        (candidate.component.obj.name, candidate.component.seed_polygon)
        for candidate in existing
    }
    object_names = {
        str(anchor.get("object_name"))
        for anchor in _seed_anchors(payload, side)
        if anchor.get("object_name") is not None
    }
    search_objects = [
        obj for obj in objects if not object_names or obj.name in object_names or _base_name(obj.name) in {_base_name(value) for value in object_names}
    ]
    search_radius = max(float(geometry["eye_width"]) * 1.8, radius_hint * 1.8)
    recovered: list[_Candidate] = []
    for obj in search_objects:
        index = indices.setdefault(obj.name, _ComponentIndex(obj))
        for component in index.all_components():
            key = (component.obj.name, component.seed_polygon)
            if key in existing_keys:
                continue
            matrix = component.obj.matrix_world
            points = np.asarray(
                [
                    tuple(matrix @ component.obj.data.vertices[vertex_id].co)
                    for vertex_id in component.vertex_ids
                ],
                dtype=np.float64,
            )
            if points.size == 0:
                continue
            bbox_center = (points.min(axis=0) + points.max(axis=0)) * 0.5
            if float(np.linalg.norm(bbox_center - np.asarray(center_hint))) > search_radius:
                continue
            recovered.append(
                _score_component(
                    component,
                    iris_center=geometry["iris_center"],
                    eye_width=float(geometry["eye_width"]),
                    forced=False,
                    center_hint=center_hint,
                    radius_hint=radius_hint,
                )
            )
    return recovered


def _infer_full_ball_pivot(
    *,
    candidate: _Candidate,
    payload: Mapping[str, Any],
    geometry: Mapping[str, Any],
    opposite_geometry: Mapping[str, Any] | None,
    opposite_pivot: _Candidate | None,
) -> dict[str, Any]:
    """Infer a spherical pivot behind a ray-confirmed outer eye cap."""

    from mathutils import Matrix, Vector

    eye_width = float(geometry["eye_width"])
    if opposite_pivot is not None and opposite_geometry is not None:
        forward = opposite_geometry["iris_center"] - opposite_pivot.center
        radius = opposite_pivot.radius * eye_width / max(
            float(opposite_geometry["eye_width"]), 1.0e-10
        )
        source = "mirrored_from_opposite_eye"
    else:
        view_directions = {
            "neg_y": (0.0, -1.0, 0.0),
            "pos_y": (0.0, 1.0, 0.0),
            "neg_x": (-1.0, 0.0, 0.0),
            "pos_x": (1.0, 0.0, 0.0),
            "neg_z": (0.0, 0.0, -1.0),
            "pos_z": (0.0, 0.0, 1.0),
        }
        view = str(payload.get("selected_view", "neg_z"))
        forward = Vector(view_directions.get(view, (0.0, 0.0, -1.0)))
        try:
            yaw_offset = float(payload.get("yaw_offset", 0.0))
        except (TypeError, ValueError):
            yaw_offset = 0.0
        if yaw_offset:
            forward.rotate(Matrix.Rotation(math.radians(yaw_offset), 4, "Z"))
        radius = eye_width * 0.55
        source = "landmark_view_and_eye_width"
    if forward.length <= 1.0e-8:
        raise ValueError("Could not infer the procedural eyeball forward direction.")
    forward.normalize()
    radius = min(max(float(radius), eye_width * 0.38), eye_width * 0.78)
    original_center = candidate.center.copy()
    original_radius = float(candidate.radius)
    candidate.center = geometry["iris_center"] - forward * radius
    candidate.radius = radius
    candidate.radius_ratio = radius / eye_width
    candidate.shell_error = 0.0
    candidate.pivot_inferred = True
    candidate.reason = f"full-ball pivot inferred via {source}"
    return {
        "method": source,
        "center": [float(value) for value in candidate.center],
        "radius": float(radius),
        "source_fit_center": [float(value) for value in original_center],
        "source_fit_radius": original_radius,
    }


def _outer_cap_sphere_is_usable(
    candidate: _Candidate, eye_width: float
) -> bool:
    return bool(
        math.isfinite(candidate.radius)
        and candidate.radius > 1.0e-10
        and all(math.isfinite(float(value)) for value in candidate.center)
        # An anatomical eyeball radius is roughly half the eyelid-opening
        # width.  Larger fitted values are usually eyelid/socket caps rather
        # than the eye itself and visibly protrude after reconstruction.
        and 0.38 <= candidate.radius / max(float(eye_width), 1.0e-10) <= 0.68
        and candidate.residual <= 0.18
        and len(candidate.component.vertex_ids) >= 6
        and len(candidate.component.polygon_ids) >= 4
    )


def _use_outer_cap_sphere_pivot(
    candidate: _Candidate, eye_width: float
) -> dict[str, Any]:
    """Promote the cap's own fitted sphere to the full-eyeball pivot."""

    if not _outer_cap_sphere_is_usable(candidate, eye_width):
        raise ValueError("The outer eye cap does not provide a usable sphere fit.")
    candidate.radius_ratio = candidate.radius / max(float(eye_width), 1.0e-10)
    candidate.pivot_inferred = True
    candidate.reason = "full-ball surface fitted directly from outer eye cap"
    return {
        "method": "outer_cap_sphere_fit",
        "center": [float(value) for value in candidate.center],
        "radius": float(candidate.radius),
        "sphere_residual_ratio": float(candidate.residual),
        "source_component": [
            candidate.component.obj.name,
            int(candidate.component.seed_polygon),
        ],
        "source_geometry_modified": False,
    }


def _select_eye_assembly_legacy(
    candidates: Sequence[_Candidate], side: str
) -> list[_Candidate]:
    accepted = [candidate for candidate in candidates if candidate.plausible]
    if not accepted:
        details = "; ".join(
            f"{candidate.component.obj.name}/p{candidate.component.seed_polygon}: "
            f"score={candidate.score:.3f}, radius/width={candidate.radius_ratio:.3f}, "
            f"residual={candidate.residual:.3f}, closedness={candidate.closedness:.3f}, "
            f"iris-shell={candidate.shell_error:.3f}, "
            f"center=({candidate.center.x:.4g},{candidate.center.y:.4g},"
            f"{candidate.center.z:.4g}), radius={candidate.radius:.4g} "
            f"({candidate.reason})"
            for candidate in candidates[:8]
        )
        suffix = f" Candidates: {details}." if details else " No ray-hit components were found."
        raise ValueError(
            f"Could not find an independent sphere-like {side} eyeball.{suffix} "
            f"If the eye is a separate mesh, pass --{side}-eye-object OBJECT_NAME. "
            "Painted-only or face-fused eyes cannot be rigidly rotated."
        )
    best = accepted[0]
    assembly = [best]
    for candidate in accepted[1:]:
        center_distance = (candidate.center - best.center).length
        radius_ratio = candidate.radius / best.radius
        if center_distance <= max(best.radius, candidate.radius) * 0.35 and 0.68 <= radius_ratio <= 1.35:
            assembly.append(candidate)

    # Coarse Pixel3D eyes can store the sclera, iris, pupil, and cornea as
    # separate components with only a handful of polygons.  Those small layers
    # do not individually contain enough curvature for a useful sphere fit,
    # but they must follow the fitted eyeball.  Admit only ray-seeded compact
    # components whose vertices lie on the selected sphere; large eyelid/face
    # patches are excluded by the extent guard.
    selected_keys = {
        (candidate.component.obj.name, candidate.component.seed_polygon)
        for candidate in assembly
    }
    for candidate in candidates:
        key = (candidate.component.obj.name, candidate.component.seed_polygon)
        if key in selected_keys or candidate.extent > best.radius * 2.6:
            continue
        matrix = candidate.component.obj.matrix_world
        radial_errors = [
            abs(
                (matrix @ candidate.component.obj.data.vertices[vertex_id].co - best.center).length
                - best.radius
            )
            / max(best.radius, 1.0e-10)
            for vertex_id in candidate.component.vertex_ids
        ]
        if radial_errors and min(radial_errors) <= 0.25:
            assembly.append(candidate)
            selected_keys.add(key)
    return assembly


def _finite_hit_depths(candidate: _Candidate) -> list[float]:
    return [
        depth
        for _landmark_id, depth, _luminance, _rgb in candidate.ray_hits
        if math.isfinite(depth)
    ]


def _bright_horizontal_hits(
    candidate: _Candidate, side: str
) -> list[tuple[int, float, float, tuple[float, float, float]]]:
    iris_ids = EYE_LANDMARKS[side]["iris"]
    horizontal_ids = {int(iris_ids[1]), int(iris_ids[3])}
    results = []
    for hit in candidate.ray_hits:
        landmark_id, depth, luminance, rgb = hit
        finite_rgb = all(math.isfinite(value) for value in rgb)
        neutral = not finite_rgb or max(rgb) - min(rgb) <= 0.12
        if (
            landmark_id in horizontal_ids
            and math.isfinite(depth)
            and math.isfinite(luminance)
            and luminance >= 0.90
            and neutral
        ):
            results.append(hit)
    return results


def _coherent_visible_carrier(
    candidates: Sequence[_Candidate], side: str
) -> _Candidate | None:
    iris_ids = EYE_LANDMARKS[side]["iris"]
    center_id = int(iris_ids[0])
    horizontal_ids = {int(iris_ids[1]), int(iris_ids[3])}
    vertical_ids = {int(iris_ids[2]), int(iris_ids[4])}
    coherent = []
    for candidate in candidates:
        hit_ids = {hit[0] for hit in candidate.ray_hits}
        depths = _finite_hit_depths(candidate)
        if (
            depths
            and center_id in hit_ids
            and hit_ids.intersection(horizontal_ids)
            and hit_ids.intersection(vertical_ids)
        ):
            coherent.append(candidate)
    if not coherent:
        return None
    # The ray list is ordered from the rendered surface toward the back of the
    # head.  A complete iris consensus on the nearest component is the visible
    # carrier; a deeper white sphere is the rotation pivot, not the iris.
    return min(
        coherent,
        key=lambda candidate: (min(_finite_hit_depths(candidate)), candidate.score),
    )


def _select_outer_eye_cap(
    candidates: Sequence[_Candidate], side: str, eye_width: float
) -> _Candidate | None:
    """Choose a compact visible cap for forced full-ball reconstruction."""

    coherent = _coherent_visible_carrier(candidates, side)
    bounded = [
        candidate
        for candidate in candidates
        if candidate.ray_hits
        and 4 <= len(candidate.component.polygon_ids) <= 2_000
        and 4 <= len(candidate.component.vertex_ids) <= 1_500
        and eye_width * 0.05 <= candidate.extent <= eye_width * 1.70
        # Forced reconstruction may accept an open cap, but its fitted sphere
        # must still place the detected iris near that sphere.  This rejects
        # stale FBX landmarks whose polygon IDs happen to index unrelated GLB
        # face/hair components after an axis or topology conversion.
        and math.isfinite(candidate.shell_error)
        and candidate.shell_error <= 0.75
    ]
    if coherent in bounded:
        return coherent
    if not bounded:
        return None
    return min(
        bounded,
        key=lambda candidate: (
            min(_finite_hit_depths(candidate))
            if _finite_hit_depths(candidate)
            else float("inf"),
            candidate.score,
        ),
    )


def _visible_front_eye_layers(
    candidates: Sequence[_Candidate], eye_width: float
) -> list[_Candidate]:
    """Return compact fragments occupying the rendered front eye surface."""

    eligible = [
        candidate
        for candidate in candidates
        if _finite_hit_depths(candidate)
        and _candidate_has_eye_colored_hit(candidate)
        and candidate.plausible
        and 4 <= len(candidate.component.polygon_ids) <= 512
        and 4 <= len(candidate.component.vertex_ids) <= 512
        and candidate.extent <= eye_width * 1.10
        and math.isfinite(candidate.radius)
        and candidate.radius > 1.0e-10
        and 0.28 <= candidate.radius / max(eye_width, 1.0e-10) <= 1.05
        and candidate.residual <= 0.15
    ]
    if not eligible:
        return []
    front_depth = min(
        min(_finite_hit_depths(candidate)) for candidate in eligible
    )
    return [
        candidate
        for candidate in eligible
        if min(_finite_hit_depths(candidate)) <= front_depth + eye_width * 0.55
    ]


def _candidate_has_eye_colored_hit(candidate: _Candidate) -> bool:
    """Reject skin-colored components that merely cross an iris landmark ray."""

    for _landmark_id, _depth, luminance, rgb in candidate.ray_hits:
        if not math.isfinite(float(luminance)) or len(rgb) < 3:
            continue
        channels = tuple(float(value) for value in rgb[:3])
        if not all(math.isfinite(value) for value in channels):
            continue
        chroma = max(channels) - min(channels)
        # Sclera and gray stylized irises are moderately bright and nearly
        # neutral. Pupils may be very dark, where texture quantization makes a
        # fixed small chroma allowance more reliable than a ratio alone.
        if (
            float(luminance) >= 0.32
            and chroma <= 0.18
        ) or (
            float(luminance) < 0.32
            and chroma <= max(0.06, float(luminance) * 0.55)
        ):
            return True
    return False


def _select_eye_assembly(candidates: Sequence[_Candidate], side: str) -> list[_Candidate]:
    """Select a ray-confirmed visible carrier and every rigid eye follower.

    Pixel3D eyes are commonly split into disconnected pupil, iris, sclera, and
    corneal shells.  Ranking each component as an independent sphere picks one
    fragment; accepting every nearby sphere picks eyelid and socket fragments.
    The horizontal iris rays provide the missing discriminator: neutral bright
    layers on those rays are physical eye shells, while a coherent five-point
    front layer is the visible carrier.
    """

    visible = _coherent_visible_carrier(candidates, side)
    if visible is None:
        return _select_eye_assembly_legacy(candidates, side)

    pivot_options = [
        candidate
        for candidate in candidates
        if candidate is not visible
        and candidate.plausible
        and len(candidate.component.vertex_ids) >= 32
        and len(candidate.component.polygon_ids) >= 8
        and candidate.closedness >= 0.62
        and candidate.residual <= 0.10
        and 0.30 <= candidate.radius_ratio <= 0.90
        and _bright_horizontal_hits(candidate, side)
    ]
    if pivot_options:
        pivot = min(
            pivot_options,
            key=lambda candidate: (
                min(hit[1] for hit in _bright_horizontal_hits(candidate, side)),
                candidate.score,
            ),
        )
    else:
        pivot = visible

    assembly = [pivot]
    if visible is not pivot:
        assembly.append(visible)

    visible_depths = _finite_hit_depths(visible)
    eye_width = (
        pivot.radius / pivot.radius_ratio
        if pivot.radius_ratio > 1.0e-10
        else visible.radius / max(visible.radius_ratio, 1.0e-10)
    )
    maximum_front_depth = min(visible_depths) + eye_width * 0.15
    selected_keys = {
        (candidate.component.obj.name, candidate.component.seed_polygon)
        for candidate in assembly
    }
    front_bright_followers = []
    for candidate in candidates:
        key = (candidate.component.obj.name, candidate.component.seed_polygon)
        bright_hits = _bright_horizontal_hits(candidate, side)
        if key in selected_keys or not bright_hits:
            continue
        if min(hit[1] for hit in bright_hits) > maximum_front_depth:
            continue
        if not (
            12 <= len(candidate.component.vertex_ids) <= 512
            and 8 <= len(candidate.component.polygon_ids) <= 1_024
            and candidate.closedness >= 0.55
            and candidate.residual <= 0.12
            and 0.25 <= candidate.radius_ratio <= 0.95
        ):
            continue
        front_bright_followers.append(candidate)
    substantial_followers = [
        candidate
        for candidate in front_bright_followers
        if len(candidate.component.vertex_ids) >= 32
    ]
    if substantial_followers:
        front_bright_followers = substantial_followers
    elif front_bright_followers:
        # Several almost-identical tiny concentric shells can sit a fraction
        # of a millimetre apart. Move only the foremost when no substantial
        # bright shell exists; otherwise a rear duplicate becomes a crescent.
        front_bright_followers = [
            min(
                front_bright_followers,
                key=lambda candidate: min(
                    hit[1] for hit in _bright_horizontal_hits(candidate, side)
                ),
            )
        ]
    for candidate in front_bright_followers:
        assembly.append(candidate)
        selected_keys.add(
            (candidate.component.obj.name, candidate.component.seed_polygon)
        )
    return assembly


def _component_base_color(
    component: _Component, image_pixel_cache: dict[int, Any]
) -> Any:
    """Sample a component's texture without rendering it."""

    import numpy as np

    obj = component.obj
    uv_layer = obj.data.uv_layers.active
    samples = []
    stride = max(1, len(component.polygon_ids) // 12)
    for polygon_id in component.polygon_ids[::stride][:12]:
        polygon = obj.data.polygons[int(polygon_id)]
        material = (
            obj.material_slots[int(polygon.material_index)].material
            if 0 <= int(polygon.material_index) < len(obj.material_slots)
            else None
        )
        if material is None or not material.use_nodes:
            continue
        principled = next(
            (node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"),
            None,
        )
        if principled is None:
            continue
        base_color = principled.inputs.get("Base Color")
        image = None
        if base_color is not None:
            for link in base_color.links:
                if link.from_node.type == "TEX_IMAGE" and link.from_node.image is not None:
                    image = link.from_node.image
                    break
        if image is None or uv_layer is None:
            if base_color is not None:
                samples.append(np.asarray(base_color.default_value[:3], dtype=np.float64))
            continue
        width, height = int(image.size[0]), int(image.size[1])
        if width < 2 or height < 2:
            continue
        cache_key = int(image.as_pointer())
        pixels = image_pixel_cache.get(cache_key)
        if pixels is None:
            flat = np.empty(width * height * 4, dtype=np.float32)
            image.pixels.foreach_get(flat)
            pixels = flat.reshape(height, width, 4)
            image_pixel_cache[cache_key] = pixels
        loop_ids = [int(value) for value in polygon.loop_indices]
        if not loop_ids:
            continue
        uv = np.mean(
            np.stack(
                [np.asarray(uv_layer.data[loop_id].uv, dtype=np.float64) for loop_id in loop_ids]
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


def _find_nearby_eye_followers(
    *,
    objects: Sequence[Any],
    indices: dict[str, _ComponentIndex],
    geometry: Mapping[str, Any],
    candidates: Sequence[_Candidate],
    assembly: Sequence[_Candidate],
) -> list[_Candidate]:
    """Find compact eye wedges missed by the five MediaPipe iris rays."""

    import numpy as np

    if not assembly:
        return []
    pivot = assembly[0]
    eye_width = float(geometry["eye_width"])
    axis = geometry["iris_center"] - pivot.center
    if axis.length <= 1.0e-8:
        return []
    axis.normalize()
    known_keys = {
        (candidate.component.obj.name, id(candidate.component)) for candidate in candidates
    }
    selected_components = {id(candidate.component) for candidate in assembly}
    object_names = {candidate.component.obj.name for candidate in candidates}
    image_pixel_cache: dict[int, Any] = {}
    followers: list[_Candidate] = []
    for obj in objects:
        if object_names and obj.name not in object_names:
            continue
        index = indices.setdefault(obj.name, _ComponentIndex(obj))
        for component in index.all_components():
            if (
                (component.obj.name, id(component)) in known_keys
                or id(component) in selected_components
                or not 12 <= len(component.vertex_ids)
                or not 8 <= len(component.polygon_ids) <= 20_000
            ):
                continue
            matrix = component.obj.matrix_world
            coordinates = np.asarray(
                [
                    tuple(matrix @ component.obj.data.vertices[vertex_id].co)
                    for vertex_id in component.vertex_ids
                ],
                dtype=np.float64,
            )
            bbox_extent = np.ptp(coordinates, axis=0)
            if float(bbox_extent.max()) > eye_width * 0.85:
                continue
            deltas = coordinates - np.asarray(pivot.center)[None, :]
            radial_distances = np.linalg.norm(deltas, axis=1)
            tangent_deltas = deltas - (deltas @ np.asarray(axis))[:, None] * np.asarray(axis)[None, :]
            tangent_distances = np.linalg.norm(tangent_deltas, axis=1)
            component_center = coordinates.mean(axis=0)
            center_delta = component_center - np.asarray(pivot.center)
            center_tangent = center_delta - np.dot(center_delta, np.asarray(axis)) * np.asarray(axis)
            center_axial = float(np.dot(center_delta, np.asarray(axis)))
            shared_shell_residual = float(
                np.median(np.abs(radial_distances - pivot.radius))
                / max(pivot.radius, 1.0e-8)
            )
            if (
                float(np.linalg.norm(center_delta)) > eye_width * 1.25
                or float(np.linalg.norm(center_tangent)) > eye_width * 0.36
                or float(np.min(tangent_distances)) > eye_width * 0.32
                or center_axial < pivot.radius * 0.45
                or center_axial > pivot.radius * 1.40
                or shared_shell_residual > 0.16
            ):
                continue
            _near_center, near_radius, near_residual = _fit_sphere(coordinates)
            radius_ratio = near_radius / eye_width
            closedness = _closedness(component)
            if (
                not 0.12 <= radius_ratio <= 0.90
                or near_residual > 0.18
                or closedness < 0.62
            ):
                continue
            sampled_rgb = _component_base_color(component, image_pixel_cache)
            if np.isfinite(sampled_rgb).all():
                luminance = float(np.dot(sampled_rgb, (0.2126, 0.7152, 0.0722)))
                chroma = float(sampled_rgb.max() - sampled_rgb.min())
                if chroma > 0.18 and luminance > 0.38:
                    continue
            candidate = _score_component(
                component,
                iris_center=geometry["iris_center"],
                eye_width=eye_width,
                forced=False,
            )
            candidate.reason = "accepted as nearby eye-shell follower"
            followers.append(candidate)
    followers.sort(key=lambda candidate: candidate.component.seed_polygon)
    return followers


def _candidate_report(candidate: _Candidate) -> dict[str, Any]:
    def json_number(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    return {
        "object": candidate.component.obj.name,
        "seed_polygon": candidate.component.seed_polygon,
        "polygon_count": len(candidate.component.polygon_ids),
        "vertex_count": len(candidate.component.vertex_ids),
        "center": [float(value) for value in candidate.center],
        "radius": json_number(candidate.radius),
        "extent": json_number(candidate.extent),
        "radius_eye_width_ratio": json_number(candidate.radius_ratio),
        "sphere_residual_ratio": json_number(candidate.residual),
        "closedness": json_number(candidate.closedness),
        "iris_shell_error_ratio": json_number(candidate.shell_error),
        "score": json_number(candidate.score),
        "plausible": candidate.plausible,
        "reason": candidate.reason,
        "pivot_inferred": candidate.pivot_inferred,
        "ray_hits": [
            {
                "landmark_id": landmark_id,
                "ray_depth": json_number(depth),
                "luminance": json_number(luminance),
                "rgb": [json_number(value) for value in rgb],
            }
            for landmark_id, depth, luminance, rgb in candidate.ray_hits
        ],
    }


def _reset_and_import(path: Path) -> list[Any]:
    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        # The GLB importer will recreate what it needs. Only delete orphaned data.
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)
    before = set(bpy.context.scene.objects)
    result = bpy.ops.import_scene.gltf(filepath=str(path))
    if "FINISHED" not in result:
        raise RuntimeError(f"Blender could not import {path}.")
    objects = [
        obj
        for obj in bpy.context.scene.objects
        if obj not in before and obj.type == "MESH" and len(obj.data.polygons) > 0
    ]
    if not objects:
        raise ValueError(f"The GLB contains no non-empty mesh objects: {path}")
    return objects


def _delete_polygons_exact(mesh: Any, polygon_ids: set[int]) -> None:
    import bmesh

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()
        faces = [face for face in bm.faces if int(face.index) in polygon_ids]
        if faces:
            bmesh.ops.delete(bm, geom=faces, context="FACES")
        loose_vertices = [vertex for vertex in bm.verts if not vertex.link_faces]
        if loose_vertices:
            bmesh.ops.delete(bm, geom=loose_vertices, context="VERTS")
        bm.to_mesh(mesh)
        mesh.update()
    finally:
        bm.free()


def _separate_selected_polygons(obj: Any, polygon_ids: set[int], label: str) -> Any:
    if not polygon_ids:
        raise ValueError(f"No polygons were selected for {label}.")
    if len(polygon_ids) == len(obj.data.polygons):
        return obj
    if obj.data.shape_keys is not None:
        raise ValueError(
            f"{obj.name!r} contains morph targets and also contains non-eye geometry; "
            "Blender cannot safely separate its eye components. Supply a GLB with "
            "separate physical eye meshes."
        )
    source_polygon_count = len(obj.data.polygons)
    expected_polygon_count = len(polygon_ids)

    # Blender's edit-mode separate operator is vertex-selection based. A GLB
    # can contain a degenerate polygon that shares every vertex with an eye
    # component, causing that adjacent polygon to be swept into the eye even
    # when its face is not selected. Work directly from immutable face indices
    # so the operation is exact and cannot tear a strip from the head.
    separated = obj.copy()
    separated.data = obj.data.copy()
    collection = obj.users_collection[0] if obj.users_collection else None
    if collection is None:
        raise RuntimeError(f"{obj.name!r} is not linked to a Blender collection.")
    collection.objects.link(separated)

    all_polygon_ids = set(range(source_polygon_count))
    _delete_polygons_exact(separated.data, all_polygon_ids - polygon_ids)
    if obj.data.users > 1:
        obj.data = obj.data.copy()
    _delete_polygons_exact(obj.data, polygon_ids)

    if (
        len(separated.data.polygons) != expected_polygon_count
        or len(obj.data.polygons) != source_polygon_count - expected_polygon_count
    ):
        # The copied object is safe to remove because it has not yet been used
        # as a parent or animation target.
        import bpy

        bpy.data.objects.remove(separated, do_unlink=True)
        raise RuntimeError(
            f"Exact {label} separation did not preserve polygon counts; "
            "export was stopped to avoid tearing the head mesh."
        )
    return separated


def _copy_selected_polygons(obj: Any, polygon_ids: set[int], label: str) -> Any:
    """Copy an exact face set without changing the source object."""

    import bpy

    source_polygons = [obj.data.polygons[index] for index in sorted(polygon_ids)]
    source_vertex_ids = sorted(
        {int(vertex_id) for polygon in source_polygons for vertex_id in polygon.vertices}
    )
    vertex_map = {
        source_vertex_id: new_vertex_id
        for new_vertex_id, source_vertex_id in enumerate(source_vertex_ids)
    }
    mesh = bpy.data.meshes.new(f"{label}Mesh")
    mesh.from_pydata(
        [tuple(obj.data.vertices[index].co) for index in source_vertex_ids],
        [],
        [
            [vertex_map[int(vertex_id)] for vertex_id in polygon.vertices]
            for polygon in source_polygons
        ],
    )
    mesh.update()
    for material in obj.data.materials:
        mesh.materials.append(material)
    for target_polygon, source_polygon in zip(mesh.polygons, source_polygons):
        target_polygon.material_index = int(source_polygon.material_index)
        target_polygon.use_smooth = bool(source_polygon.use_smooth)
    for source_layer in obj.data.uv_layers:
        target_layer = mesh.uv_layers.new(name=source_layer.name)
        target_layer.active_render = bool(source_layer.active_render)
        for target_polygon, source_polygon in zip(mesh.polygons, source_polygons):
            for target_loop, source_loop in zip(
                target_polygon.loop_indices,
                source_polygon.loop_indices,
            ):
                target_layer.data[int(target_loop)].uv = source_layer.data[
                    int(source_loop)
                ].uv
    copied = bpy.data.objects.new(label, mesh)
    copied.matrix_world = obj.matrix_world.copy()
    collection = obj.users_collection[0] if obj.users_collection else None
    if collection is None:
        raise RuntimeError(f"{obj.name!r} is not linked to a Blender collection.")
    collection.objects.link(copied)
    return copied


def _split_chunk_by_eye(chunk: Any, left_center: Any, right_center: Any) -> dict[str, list[Any]]:
    left_polygons: set[int] = set()
    right_polygons: set[int] = set()
    matrix = chunk.matrix_world
    for polygon in chunk.data.polygons:
        center = matrix @ polygon.center
        target = left_polygons if (center - left_center).length <= (center - right_center).length else right_polygons
        target.add(int(polygon.index))
    if not left_polygons:
        return {"left": [], "right": [chunk]}
    if not right_polygons:
        return {"left": [chunk], "right": []}
    left_obj = _separate_selected_polygons(chunk, left_polygons, "left eye")
    return {"left": [left_obj], "right": [chunk]}


def _isolate_eye_objects(
    objects: Sequence[Any],
    assemblies: Mapping[str, Sequence[_Candidate]],
    geometry: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[Any]]:
    selected: dict[str, dict[str, set[int]]] = {"left": {}, "right": {}}
    object_by_name = {obj.name: obj for obj in objects}
    for side, candidates in assemblies.items():
        for candidate in candidates:
            selected[side].setdefault(candidate.component.obj.name, set()).update(
                candidate.component.polygon_ids
            )

    overlap = set(selected["left"]) & set(selected["right"])
    for object_name in overlap:
        duplicate = selected["left"][object_name] & selected["right"][object_name]
        if duplicate:
            raise ValueError(
                f"The left and right eye selectors chose the same component in "
                f"{object_name!r}; refusing to rotate ambiguous geometry."
            )

    result: dict[str, list[Any]] = {"left": [], "right": []}
    for object_name in sorted(set(selected["left"]) | set(selected["right"])):
        obj = object_by_name[object_name]
        left_ids = selected["left"].get(object_name, set())
        right_ids = selected["right"].get(object_name, set())
        union = left_ids | right_ids
        if left_ids and right_ids:
            chunk = _separate_selected_polygons(obj, union, "left/right eye assembly")
            split = _split_chunk_by_eye(
                chunk,
                geometry["left"]["iris_center"],
                geometry["right"]["iris_center"],
            )
            result["left"].extend(split["left"])
            result["right"].extend(split["right"])
        else:
            side = "left" if left_ids else "right"
            chunk = _separate_selected_polygons(obj, union, f"{side} eye assembly")
            result[side].append(chunk)

    for side in ("left", "right"):
        if not result[side]:
            raise RuntimeError(f"No isolated {side} eye objects remain after selection.")
        for index, obj in enumerate(result[side]):
            obj.name = f"EyeAU_{side}_{index:02d}"
    return result


def _smooth_replacement_eye_boundaries(
    obj: Any,
    assemblies: Mapping[str, Sequence[_Candidate]],
    geometry: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    """Relax only the open mesh boundaries surrounding replaced eyeballs."""

    import bmesh

    face_horizontal = (
        geometry["left"]["iris_center"] - geometry["right"]["iris_center"]
    )
    if face_horizontal.length <= 1.0e-8:
        return {"left": 0, "right": 0}
    face_horizontal.normalize()
    frames = {}
    for side in ("left", "right"):
        pivot = assemblies[side][0]
        forward = geometry[side]["iris_center"] - pivot.center
        if forward.length <= 1.0e-8:
            continue
        forward.normalize()
        horizontal = face_horizontal - forward * face_horizontal.dot(forward)
        if horizontal.length <= 1.0e-8:
            continue
        horizontal.normalize()
        vertical = forward.cross(horizontal)
        if vertical.length <= 1.0e-8:
            continue
        vertical.normalize()
        frames[side] = {
            "center": pivot.center,
            "iris_center": geometry[side]["iris_center"],
            "forward": forward,
            "horizontal": horizontal,
            "vertical": vertical,
            "half_width": float(geometry[side]["eye_width"]) * 0.50,
            "half_height": float(geometry[side]["iris_radius"]) * 1.30,
        }

    counts = {"left": 0, "right": 0}
    if not frames:
        return counts
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.edges.ensure_lookup_table()
        matrix = obj.matrix_world
        inverse = matrix.inverted_safe()
        side_vertices: dict[str, set[Any]] = {"left": set(), "right": set()}
        side_edges: dict[str, set[Any]] = {"left": set(), "right": set()}
        for edge in bm.edges:
            if len(edge.link_faces) != 1:
                continue
            midpoint = matrix @ ((edge.verts[0].co + edge.verts[1].co) * 0.5)
            for side, frame in frames.items():
                relative = midpoint - frame["iris_center"]
                horizontal_value = relative.dot(frame["horizontal"]) / max(
                    frame["half_width"], 1.0e-8
                )
                vertical_value = relative.dot(frame["vertical"]) / max(
                    frame["half_height"], 1.0e-8
                )
                ellipse_radius = math.hypot(horizontal_value, vertical_value)
                forward_depth = (midpoint - frame["center"]).dot(frame["forward"])
                if 0.62 <= ellipse_radius <= 1.48 and forward_depth > 0.0:
                    side_edges[side].add(edge)
                    side_vertices[side].update(edge.verts)
                    break

        for side, vertices in side_vertices.items():
            edges = side_edges[side]
            if len(vertices) < 3 or len(edges) < 2:
                continue
            frame = frames[side]
            for _iteration in range(3):
                targets = {}
                for vertex in vertices:
                    neighbors = [
                        edge.other_vert(vertex)
                        for edge in vertex.link_edges
                        if edge in edges and edge.other_vert(vertex) in vertices
                    ]
                    if len(neighbors) < 2:
                        continue
                    point = matrix @ vertex.co
                    average = sum(
                        (matrix @ neighbor.co for neighbor in neighbors),
                        point.copy() * 0.0,
                    ) / len(neighbors)
                    delta = average - point
                    # Preserve socket depth; relax only within the eye plane.
                    delta -= frame["forward"] * delta.dot(frame["forward"])
                    targets[vertex] = inverse @ (point + delta * 0.32)
                for vertex, target in targets.items():
                    vertex.co = target
            for edge in edges:
                for face in edge.link_faces:
                    face.smooth = True
            counts[side] = len(vertices)
        bm.to_mesh(obj.data)
        obj.data.update()
    finally:
        bm.free()
    return counts


def _remove_replaced_eye_polygons(
    objects: Sequence[Any],
    assemblies: Mapping[str, Sequence[_Candidate]],
    geometry: Mapping[str, Mapping[str, Any]],
    indices: Mapping[str, _ComponentIndex] | None = None,
) -> dict[str, int]:
    """Remove source eye faces when a procedural sphere replaces them."""

    import bpy
    import numpy as np

    selected: dict[str, set[int]] = {}
    object_by_name = {obj.name: obj for obj in objects}
    side_faces: dict[str, dict[str, set[int]]] = {"left": {}, "right": {}}
    selected_components: dict[str, set[tuple[str, int]]] = {
        "left": set(),
        "right": set(),
    }
    aperture_face_counts = {"left": 0, "right": 0}
    smoothed_boundary_counts = {"left": 0, "right": 0}
    # Remove disconnected source-eye fragments missed by the five iris rays.
    # Keep fused face clipping disabled: deleting an ellipse from a connected
    # head mesh can split the eyelids and was the cause of the earlier jagged
    # half-head failure.
    cleanup_unselected_components = True
    cleanup_fused_sources = True
    image_pixel_cache: dict[int, Any] = {}
    for side, candidates in assemblies.items():
        for candidate in candidates:
            name = candidate.component.obj.name
            side_faces[side].setdefault(name, set()).update(
                candidate.component.polygon_ids
            )
            selected.setdefault(name, set()).update(candidate.component.polygon_ids)
            selected_components[side].add(
                (name, int(candidate.component.seed_polygon))
            )

    # Pixel3D GLBs can contain hundreds of tiny disconnected triangles and
    # caps around each eye. Landmark rays do not hit every one of them, so an
    # index-only removal leaves coplanar fragments that flicker as the view
    # changes. Add every compact component inside the fitted front eyeball
    # envelope, while retaining outer eyelids and the much larger face mesh.
    for obj in objects if cleanup_unselected_components else ():
        index = (
            indices.get(obj.name)
            if indices is not None and obj.name in indices
            else _ComponentIndex(obj)
        )
        matrix = obj.matrix_world
        for component in index.all_components():
            key = (obj.name, int(component.seed_polygon))
            if any(key in selected_components[side] for side in ("left", "right")):
                continue
            points = np.asarray(
                [
                    tuple(matrix @ obj.data.vertices[vertex_id].co)
                    for vertex_id in component.vertex_ids
                ],
                dtype=np.float64,
            )
            if len(points) == 0:
                continue
            # Fragmented Pixel3D heads can place small disconnected skin
            # wedges on the same approximate sphere as the eye. Geometry
            # alone therefore is not enough: deleting those wedges produces
            # gray holes below the lids. Only extend cleanup to components
            # whose source albedo is sclera/pupil-like. Components explicitly
            # selected by iris rays were already included above.
            sampled_rgb = _component_base_color(component, image_pixel_cache)
            if not np.isfinite(sampled_rgb).all():
                continue
            luminance = float(
                np.dot(sampled_rgb, (0.2126, 0.7152, 0.0722))
            )
            chroma = float(sampled_rgb.max() - sampled_rgb.min())
            sclera_like = luminance >= 0.32 and chroma <= 0.18
            dark_eye_like = (
                luminance < 0.32
                and chroma <= max(0.06, luminance * 0.55)
            )
            if not (sclera_like or dark_eye_like):
                continue
            matches = []
            for side in ("left", "right"):
                pivot = assemblies[side][0]
                radius = max(float(pivot.radius), 1.0e-8)
                center = np.asarray(tuple(pivot.center), dtype=np.float64)
                forward = np.asarray(
                    tuple(geometry[side]["iris_center"] - pivot.center),
                    dtype=np.float64,
                )
                forward_length = float(np.linalg.norm(forward))
                if forward_length <= 1.0e-8:
                    continue
                forward /= forward_length
                deltas = points - center[None, :]
                radial = np.linalg.norm(deltas, axis=1)
                safe_radial = np.maximum(radial, 1.0e-10)
                front_fraction = float(
                    np.mean((deltas @ forward) / safe_radial > 0.20)
                )
                extent = float(np.ptp(points, axis=0).max())
                centroid_delta = points.mean(axis=0) - center
                tangent = centroid_delta - np.dot(centroid_delta, forward) * forward
                median_ratio = float(np.median(radial)) / radius
                shell_fraction = float(
                    np.mean(np.abs(radial - radius) <= radius * 0.22)
                )
                if (
                    float(radial.min()) <= radius * 1.25
                    and median_ratio <= 1.30
                    and float(radial.max()) <= radius * 1.65
                    and extent <= radius * 2.40
                    and float(np.linalg.norm(tangent)) <= radius * 1.25
                    and shell_fraction >= 0.65
                    and (
                        front_fraction >= 0.55
                        or float(radial.max()) <= radius * 1.55
                    )
                ):
                    matches.append((median_ratio, side))
            if not matches:
                continue
            _score, side = min(matches)
            side_faces[side].setdefault(obj.name, set()).update(
                component.polygon_ids
            )
            selected.setdefault(obj.name, set()).update(component.polygon_ids)
            selected_components[side].add(key)

    # A source eye is not always a self-contained connected component. Some
    # Pixel3D identities weld eye-cap triangles to much larger face fragments,
    # so deleting only whole components leaves pieces in front of the generated
    # ball. Clip polygon centers only inside the detected eye opening. The
    # conservative ellipse stays clear of the eyelids: eye corners determine
    # its width and the four iris-edge landmarks determine its height.
    face_horizontal = (
        geometry["left"]["iris_center"] - geometry["right"]["iris_center"]
    )
    if cleanup_fused_sources and face_horizontal.length > 1.0e-8:
        face_horizontal.normalize()
        for side in ("left", "right"):
            pivot = assemblies[side][0]
            forward = geometry[side]["iris_center"] - pivot.center
            if forward.length <= 1.0e-8:
                continue
            forward.normalize()
            horizontal = face_horizontal - forward * face_horizontal.dot(forward)
            if horizontal.length <= 1.0e-8:
                continue
            horizontal.normalize()
            vertical = forward.cross(horizontal)
            if vertical.length <= 1.0e-8:
                continue
            vertical.normalize()
            half_width = float(geometry[side]["eye_width"]) * 0.50
            half_height = float(geometry[side]["iris_radius"]) * 1.30
            iris_center = geometry[side]["iris_center"]
            if half_width <= 1.0e-8 or half_height <= 1.0e-8:
                continue
            for obj in objects:
                object_faces = side_faces[side].setdefault(obj.name, set())
                before = len(object_faces)
                matrix = obj.matrix_world
                normal_matrix = matrix.to_3x3()
                for polygon in obj.data.polygons:
                    point = matrix @ polygon.center
                    pivot_delta = point - pivot.center
                    radial_distance = pivot_delta.length
                    shell_error = abs(radial_distance - float(pivot.radius)) / max(
                        float(pivot.radius), 1.0e-8
                    )
                    polygon_points = [
                        matrix @ obj.data.vertices[int(vertex_id)].co
                        for vertex_id in polygon.vertices
                    ]
                    if not _polygon_intersects_eye_replacement_aperture(
                        points=polygon_points,
                        iris_center=iris_center,
                        pivot_center=pivot.center,
                        horizontal=horizontal,
                        vertical=vertical,
                        forward=forward,
                        half_width=half_width,
                        half_height=half_height,
                    ):
                        continue
                    component = _Component(
                        obj=obj,
                        polygon_ids=(int(polygon.index),),
                        vertex_ids=tuple(int(value) for value in polygon.vertices),
                        seed_polygon=int(polygon.index),
                    )
                    sampled_rgb = _component_base_color(
                        component, image_pixel_cache
                    )
                    has_sample = bool(np.isfinite(sampled_rgb).all())
                    luminance = (
                        float(np.dot(sampled_rgb, (0.2126, 0.7152, 0.0722)))
                        if has_sample
                        else float("nan")
                    )
                    chroma = (
                        float(sampled_rgb.max() - sampled_rgb.min())
                        if has_sample
                        else float("nan")
                    )
                    sclera_like = bool(
                        has_sample and luminance >= 0.35 and chroma <= 0.18
                    )
                    eye_colored = bool(
                        has_sample and (sclera_like or chroma <= 0.14)
                    )
                    if len(obj.data.polygons) > 2_500 and not eye_colored:
                        continue
                    # Bright neutral sclera fragments can sit behind the front
                    # cap, so allow them a wider radial envelope. Dark
                    # iris/pupil faces retain the strict shell constraint.
                    if shell_error > (1.10 if sclera_like else 0.24):
                        continue
                    radial_direction = pivot_delta.normalized()
                    world_normal = normal_matrix @ polygon.normal
                    if world_normal.length <= 1.0e-8:
                        continue
                    world_normal.normalize()
                    if abs(float(world_normal.dot(radial_direction))) < (
                        0.05 if sclera_like else 0.48
                    ):
                        continue
                    object_faces.add(int(polygon.index))
                aperture_face_counts[side] += len(object_faces) - before
                selected.setdefault(obj.name, set()).update(object_faces)

        # Polygon centers and vertices can all lie outside the eye opening for
        # a large triangle that nevertheless spans across it.  Cast a dense
        # grid of view-aligned rays through the aperture and remove every
        # neutral eye-colored layer near the fitted eyeball. This directly
        # catches the visible source faces while brown skin hits are retained.
        from mathutils.bvhtree import BVHTree

        ray_surfaces = []
        for obj in objects:
            world_vertices = [
                obj.matrix_world @ vertex.co for vertex in obj.data.vertices
            ]
            polygons = [
                tuple(int(value) for value in polygon.vertices)
                for polygon in obj.data.polygons
            ]
            if world_vertices and polygons:
                ray_surfaces.append(
                    (
                        obj,
                        BVHTree.FromPolygons(
                            world_vertices, polygons, all_triangles=False
                        ),
                    )
                )
        for side in ("left", "right"):
            pivot = assemblies[side][0]
            radius = max(float(pivot.radius), 1.0e-8)
            forward = geometry[side]["iris_center"] - pivot.center
            if forward.length <= 1.0e-8:
                continue
            forward.normalize()
            horizontal = face_horizontal - forward * face_horizontal.dot(forward)
            if horizontal.length <= 1.0e-8:
                continue
            horizontal.normalize()
            vertical = forward.cross(horizontal)
            if vertical.length <= 1.0e-8:
                continue
            vertical.normalize()
            half_width = float(geometry[side]["eye_width"]) * 0.51
            half_height = float(geometry[side]["iris_radius"]) * 1.32
            iris_center = geometry[side]["iris_center"]
            for vertical_index in range(-6, 7):
                normalized_vertical = vertical_index / 6.0
                for horizontal_index in range(-10, 11):
                    normalized_horizontal = horizontal_index / 10.0
                    if (
                        normalized_horizontal * normalized_horizontal
                        + normalized_vertical * normalized_vertical
                        > 1.0
                    ):
                        continue
                    target = (
                        iris_center
                        + horizontal * half_width * normalized_horizontal
                        + vertical * half_height * normalized_vertical
                    )
                    ray_direction = -forward
                    for obj, tree in ray_surfaces:
                        cursor = target + forward * radius * 3.0
                        remaining = radius * 6.0
                        for _hit_index in range(12):
                            location, _normal, polygon_index, distance = tree.ray_cast(
                                cursor, ray_direction, remaining
                            )
                            if (
                                location is None
                                or polygon_index is None
                                or distance is None
                            ):
                                break
                            advance = float(distance) + radius * 0.002
                            remaining -= advance
                            if remaining <= 0.0:
                                break
                            cursor = location + ray_direction * radius * 0.002
                            radial_ratio = (location - pivot.center).length / radius
                            if radial_ratio > 2.20:
                                continue
                            polygon = obj.data.polygons[int(polygon_index)]
                            component = _Component(
                                obj=obj,
                                polygon_ids=(int(polygon_index),),
                                vertex_ids=tuple(
                                    int(value) for value in polygon.vertices
                                ),
                                seed_polygon=int(polygon_index),
                            )
                            sampled_rgb = _component_base_color(
                                component, image_pixel_cache
                            )
                            if not np.isfinite(sampled_rgb).all():
                                continue
                            luminance = float(
                                np.dot(
                                    sampled_rgb,
                                    (0.2126, 0.7152, 0.0722),
                                )
                            )
                            chroma = float(
                                sampled_rgb.max() - sampled_rgb.min()
                            )
                            sclera_like = luminance >= 0.35 and chroma <= 0.18
                            dark_eye_like = (
                                luminance < 0.35
                                and chroma <= max(0.045, luminance * 0.65)
                            )
                            if not (sclera_like or dark_eye_like):
                                continue
                            object_faces = side_faces[side].setdefault(
                                obj.name, set()
                            )
                            before = len(object_faces)
                            object_faces.add(int(polygon_index))
                            aperture_face_counts[side] += len(object_faces) - before
                            selected.setdefault(obj.name, set()).add(
                                int(polygon_index)
                            )
    for name in set(side_faces["left"]) & set(side_faces["right"]):
        if side_faces["left"][name] & side_faces["right"][name]:
            raise ValueError(
                f"The left and right procedural eyes overlap in {name!r}."
            )

    for object_name, polygon_ids in selected.items():
        obj = object_by_name[object_name]
        source_polygon_count = len(obj.data.polygons)
        if not polygon_ids:
            continue
        if len(polygon_ids) == source_polygon_count:
            bpy.data.objects.remove(obj, do_unlink=True)
            continue
        if obj.data.shape_keys is not None:
            raise ValueError(
                f"{obj.name!r} contains morph targets and non-eye geometry; "
                "the source eye faces cannot be replaced safely."
            )
        if obj.data.users > 1:
            obj.data = obj.data.copy()
        _delete_polygons_exact(obj.data, polygon_ids)
        if len(obj.data.polygons) != source_polygon_count - len(polygon_ids):
            raise RuntimeError(
                f"Exact procedural-eye replacement failed for {obj.name!r}."
            )
        if cleanup_fused_sources:
            smoothed = _smooth_replacement_eye_boundaries(
                obj, assemblies, geometry
            )
            for side in ("left", "right"):
                smoothed_boundary_counts[side] += smoothed[side]
    print(
        "[eye-au] clipped fused source-eye faces: "
        + ", ".join(
            f"{side}={aperture_face_counts[side]}" for side in ("left", "right")
        ),
        flush=True,
    )
    print(
        "[eye-au] smoothed eye-opening boundary vertices: "
        + ", ".join(
            f"{side}={smoothed_boundary_counts[side]}"
            for side in ("left", "right")
        ),
        flush=True,
    )
    return {
        side: len(selected_components[side]) for side in ("left", "right")
    }


def _inside_eye_replacement_aperture(
    *,
    horizontal_offset: float,
    vertical_offset: float,
    forward_depth: float,
    half_width: float,
    half_height: float,
) -> bool:
    """Return whether a source polygon center is inside the open-eye ellipse."""

    if forward_depth <= 0.0 or half_width <= 0.0 or half_height <= 0.0:
        return False
    normalized_horizontal = horizontal_offset / half_width
    normalized_vertical = vertical_offset / half_height
    return normalized_horizontal**2 + normalized_vertical**2 <= 1.0


def _polygon_intersects_eye_replacement_aperture(
    *,
    points: Sequence[Any],
    iris_center: Any,
    pivot_center: Any,
    horizontal: Any,
    vertical: Any,
    forward: Any,
    half_width: float,
    half_height: float,
) -> bool:
    """Return whether a source polygon overlaps the projected eye opening."""

    if len(points) < 3 or half_width <= 0.0 or half_height <= 0.0:
        return False
    if max(float((point - pivot_center).dot(forward)) for point in points) <= 0.0:
        return False
    projected = [
        (
            float((point - iris_center).dot(horizontal)) / half_width,
            float((point - iris_center).dot(vertical)) / half_height,
        )
        for point in points
    ]
    if any(x * x + y * y <= 1.0 for x, y in projected):
        return True

    # The eye center can lie inside a large triangle even when all of the
    # triangle's vertices lie outside the aperture.
    inside = False
    previous_x, previous_y = projected[-1]
    for current_x, current_y in projected:
        if (current_y > 0.0) != (previous_y > 0.0):
            crossing_x = previous_x + (-previous_y) * (
                current_x - previous_x
            ) / (current_y - previous_y)
            if crossing_x > 0.0:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    if inside:
        return True

    # Finally test polygon edges against the unit-circle aperture.
    previous_x, previous_y = projected[-1]
    for current_x, current_y in projected:
        edge_x = current_x - previous_x
        edge_y = current_y - previous_y
        length_squared = edge_x * edge_x + edge_y * edge_y
        if length_squared > 1.0e-12:
            amount = min(
                max(
                    -(previous_x * edge_x + previous_y * edge_y)
                    / length_squared,
                    0.0,
                ),
                1.0,
            )
            closest_x = previous_x + edge_x * amount
            closest_y = previous_y + edge_y * amount
            if closest_x * closest_x + closest_y * closest_y <= 1.0:
                return True
        previous_x, previous_y = current_x, current_y
    return False


def _landmark_position(payload: Mapping[str, Any], landmark_id: int) -> Any | None:
    anchors = payload.get("surface_anchors", {})
    if not isinstance(anchors, Mapping):
        return None
    return _anchor_vector(anchors.get(str(landmark_id)))


def _face_axes(
    payload: Mapping[str, Any],
    geometry: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, Any]:
    from mathutils import Vector

    horizontal = geometry["left"]["iris_center"] - geometry["right"]["iris_center"]
    if horizontal.length <= 1.0e-8:
        raise ValueError("The two detected iris centers coincide.")
    horizontal.normalize()

    forehead = _landmark_position(payload, 10) or _landmark_position(payload, 9)
    chin = _landmark_position(payload, 152)
    if forehead is not None and chin is not None:
        vertical = forehead - chin
    else:
        left_anchors = _preferred_eye_anchors(payload, "left")
        right_anchors = _preferred_eye_anchors(payload, "right")
        top_points = [
            _anchor_vector(left_anchors.get("475")),
            _anchor_vector(right_anchors.get("470")),
        ]
        bottom_points = [
            _anchor_vector(left_anchors.get("477")),
            _anchor_vector(right_anchors.get("472")),
        ]
        top_points = [point for point in top_points if point is not None]
        bottom_points = [point for point in bottom_points if point is not None]
        if not top_points or not bottom_points:
            raise ValueError("Could not infer the face-up direction from landmarks.")
        vertical = sum(top_points, Vector()) / len(top_points) - sum(
            bottom_points, Vector()
        ) / len(bottom_points)
    vertical -= horizontal * vertical.dot(horizontal)
    if vertical.length <= 1.0e-8:
        raise ValueError("Detected face horizontal and vertical axes are collinear.")
    vertical.normalize()
    return horizontal, vertical


def _motion_for_eye(aus: Mapping[int, float], side: str) -> tuple[float, float]:
    motion_x = 0.0
    motion_y = 0.0
    for au_id, intensity in aus.items():
        spec = AU_SPECS[int(au_id)]
        if spec.eye == side:
            motion_x += spec.motion_x * float(intensity)
            motion_y += spec.motion_y * float(intensity)
    magnitude = math.hypot(motion_x, motion_y)
    if magnitude > 1.0:
        motion_x /= magnitude
        motion_y /= magnitude
    return motion_x, motion_y


def _assembly_needs_sclera_backing(
    assembly: Sequence[_Candidate], iris_center: Any
) -> bool:
    """Return whether the fitted pivot is an open shell rather than a full ball."""

    if not assembly:
        return False
    pivot = assembly[0]
    iris_radius_ratio = (iris_center - pivot.center).length / max(
        pivot.radius, 1.0e-10
    )
    return bool(
        pivot.plausible
        and 0.75 <= iris_radius_ratio <= 1.25
        and (
            pivot.closedness < 0.98
            or pivot.extent < pivot.radius * 1.75
        )
    )


def _create_sclera_backing(
    *,
    side: str,
    center: Any,
    pivot_radius: float,
    eye_width: float,
    iris_center: Any,
    cap_surface_aligned: bool = False,
) -> Any:
    """Create a smooth, inset eyeball behind the source iris/cornea shells."""

    import bpy
    from mathutils import Vector

    iris_distance = (iris_center - center).length
    # Stay behind both the fitted sphere and visible iris surface. The inset
    # prevents z-fighting and keeps the generated white material from covering
    # the original textured iris/cornea at neutral pose.
    if cap_surface_aligned:
        # Match the visible cap instead of expanding past it.  The former 7.5%
        # occlusion margin pushed the replacement sphere in front of the
        # eyelids.  A tiny inset avoids z-fighting while retaining the fitted
        # pivot and a complete sphere behind the opening.
        maximum_radius = float(eye_width) * 0.62
        minimum_radius = float(eye_width) * 0.40
        radius = min(
            float(pivot_radius), float(iris_distance), maximum_radius
        ) * 0.995
        radius = min(max(radius, minimum_radius), maximum_radius)
    else:
        radius = min(float(pivot_radius) * 0.94, float(iris_distance) * 0.96)
        radius = max(radius, float(pivot_radius) * 0.10)
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=128 if cap_surface_aligned else 32,
        ring_count=64 if cap_surface_aligned else 16,
        radius=radius,
        location=center,
    )
    backing = bpy.context.object
    backing.name = f"EyeAU_ScleraBacking_{side}"
    backing.data.name = f"EyeAU_ScleraBackingMesh_{side}"
    if cap_surface_aligned:
        iris_direction = iris_center - center
        if iris_direction.length <= 1.0e-8:
            raise RuntimeError(f"The detected {side} iris direction is degenerate.")
        iris_direction.normalize()
        # Rotate mesh coordinates rather than the object/pivot so local +Z is
        # the detected iris direction. UV-sphere latitude rings then become
        # exact concentric iris/pupil boundaries. This is safe because forced
        # full-ball mode uses polygon materials and does not depend on UVs.
        surface_rotation = Vector((0.0, 0.0, 1.0)).rotation_difference(
            iris_direction
        )
        for vertex in backing.data.vertices:
            vertex.co = surface_rotation @ vertex.co
        backing.data.update()
        backing["toporig_concentric_iris_topology"] = True
        backing["toporig_surface_rotation_wxyz"] = [
            float(surface_rotation.w),
            float(surface_rotation.x),
            float(surface_rotation.y),
            float(surface_rotation.z),
        ]
    for polygon in backing.data.polygons:
        polygon.use_smooth = True

    material = bpy.data.materials.new(f"EyeAU_ScleraMaterial_{side}")
    material.diffuse_color = (0.82, 0.82, 0.78, 1.0)
    material.use_nodes = True
    principled = next(
        (
            node
            for node in material.node_tree.nodes
            if node.type == "BSDF_PRINCIPLED"
        ),
        None,
    )
    if principled is not None:
        base_color = principled.inputs.get("Base Color")
        roughness = principled.inputs.get("Roughness")
        metallic = principled.inputs.get("Metallic")
        if base_color is not None:
            base_color.default_value = (0.82, 0.82, 0.78, 1.0)
        if roughness is not None:
            roughness.default_value = 0.48
        if metallic is not None:
            metallic.default_value = 0.0
    backing.data.materials.append(material)
    backing["toporig_generated_eye_backing"] = True
    backing["toporig_eye_side"] = side
    backing["toporig_backing_radius"] = float(radius)
    return backing


def _create_original_iris_patch(
    *,
    side: str,
    source_objects: Sequence[Any],
    backing: Any,
    sphere_center: Any,
    sphere_radius: float,
    iris_direction: Any,
    iris_angle: float,
) -> tuple[list[Any], dict[str, Any] | None]:
    """Attach the complete original eye-layer stack to the generated ball.

    Pixel3D eyes are commonly assembled from several small, overlapping mesh
    components.  A single component can look like a perforated iris even
    though the complete stack renders correctly.  Preserve every source layer
    that intersects the iris disk, along with the full ray-confirmed outer cap,
    so its original materials, UVs, and depth ordering move with the new ball.
    """

    import numpy as np

    image_pixel_cache: dict[int, Any] = {}
    candidates = []
    for source_index, source in enumerate(source_objects):
        if source is None or source.type != "MESH":
            continue
        matrix = source.matrix_world
        polygon_ids = []
        radial_errors = []
        for polygon in source.data.polygons:
            point = matrix @ polygon.center
            direction = point - sphere_center
            distance = direction.length
            if distance <= 1.0e-8:
                continue
            direction /= distance
            angle = math.acos(
                max(-1.0, min(1.0, float(direction.dot(iris_direction))))
            )
            if angle <= iris_angle * 1.12:
                polygon_ids.append(int(polygon.index))
                radial_errors.append(
                    abs(distance - sphere_radius) / max(sphere_radius, 1.0e-8)
                )
        if len(polygon_ids) < 2:
            continue
        vertex_ids = sorted(
            {
                int(vertex_id)
                for polygon_id in polygon_ids
                for vertex_id in source.data.polygons[polygon_id].vertices
            }
        )
        component = _Component(
            obj=source,
            polygon_ids=tuple(polygon_ids),
            vertex_ids=tuple(vertex_ids),
            seed_polygon=polygon_ids[0],
        )
        sampled_rgb = _component_base_color(component, image_pixel_cache)
        luminance = (
            float(np.dot(sampled_rgb, (0.2126, 0.7152, 0.0722)))
            if np.isfinite(sampled_rgb).all()
            else 1.0
        )
        radial_error = float(np.median(np.asarray(radial_errors)))
        selected_fraction = len(polygon_ids) / max(len(source.data.polygons), 1)
        # The first component is the ray-confirmed visible outer cap.  Other
        # compact components are already isolated eye layers, so retain their
        # native boundaries when most of the layer belongs to the iris stack.
        use_native_component = bool(
            len(source.data.polygons) <= 512
            and (source_index == 0 or selected_fraction >= 0.35)
        )
        preserved_ids = (
            set(range(len(source.data.polygons)))
            if use_native_component
            else set(polygon_ids)
        )
        candidates.append(
            (
                source_index,
                source,
                preserved_ids,
                luminance,
                radial_error,
                use_native_component,
                selected_fraction,
            )
        )
    if not candidates:
        return [], None

    patches = []
    layer_reports = []
    # Lift the whole native stack together instead of flattening every layer
    # onto one radius.  This prevents z-fighting with the generated sphere while
    # preserving the original occlusion/depth order that completes the iris.
    surface_offset_ratio = 0.012
    surface_offset = float(sphere_radius) * surface_offset_ratio
    for (
        source_index,
        source,
        polygon_ids,
        luminance,
        radial_error,
        used_native_component,
        selected_fraction,
    ) in candidates:
        patch = _copy_selected_polygons(
            source,
            polygon_ids,
            f"EyeAU_OriginalEyeLayer_{side}_{source_index:02d}",
        )
        inverse = patch.matrix_world.inverted_safe()
        for vertex in patch.data.vertices:
            point = patch.matrix_world @ vertex.co
            direction = point - sphere_center
            if direction.length <= 1.0e-8:
                direction = iris_direction.copy()
            else:
                direction.normalize()
            vertex.co = inverse @ (point + direction * surface_offset)
        for polygon in patch.data.polygons:
            polygon.use_smooth = True
        patch.data.update()

        patch_world = patch.matrix_world.copy()
        patch.parent = backing
        patch.matrix_parent_inverse = backing.matrix_world.inverted_safe()
        patch.matrix_world = patch_world
        patch["toporig_original_eye_layer"] = True
        patch["toporig_original_iris_patch"] = True
        patch["toporig_eye_side"] = side
        patch["toporig_source_layer_index"] = int(source_index)
        patch["toporig_surface_offset_ratio"] = surface_offset_ratio
        patches.append(patch)
        layer_reports.append(
            {
                "object": patch.name,
                "source_object": source.name,
                "source_index": int(source_index),
                "polygon_count": len(patch.data.polygons),
                "sampled_luminance": float(luminance),
                "median_radial_error_ratio": float(radial_error),
                "selected_polygon_fraction": float(selected_fraction),
                "used_native_component_boundary": bool(used_native_component),
            }
        )
    return patches, {
        "objects": [patch.name for patch in patches],
        "layer_count": len(patches),
        "polygon_count": sum(len(patch.data.polygons) for patch in patches),
        "surface_offset_ratio": surface_offset_ratio,
        "preserves_original_materials_and_uvs": True,
        "preserves_native_depth_order": True,
        "layers": layer_reports,
    }


def _estimate_single_iris_palette(
    payload: Mapping[str, Any], side: str
) -> dict[str, Any]:
    """Estimate a stable iris palette from the visible landmark ray hits.

    The center and four edge rays are produced from the neutral textured mesh.
    Pixel3D meshes can put those colors on several overlapping, incomplete
    components, so transferring the component topology is not reliable.  The
    foremost sample on each ray is view-visible and identity-specific.  A
    lower-three-of-four consensus rejects an occasional sclera hit without
    turning consistently light gray/blue irises into the dark fallback.
    """

    raw_candidates = payload.get("iris_surface_anchor_candidates", {})
    if not isinstance(raw_candidates, Mapping):
        raw_candidates = {}

    samples = []
    for landmark_id in EYE_LANDMARKS[side]["iris"]:
        records = raw_candidates.get(str(landmark_id), [])
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            continue
        selected = None
        for record in records:
            if not isinstance(record, Mapping):
                continue
            raw_rgb = record.get("base_color_rgb", ())
            try:
                rgb = tuple(float(raw_rgb[index]) for index in range(3))
            except (IndexError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in rgb):
                continue
            try:
                luminance = float(record.get("base_color_luminance"))
            except (TypeError, ValueError):
                luminance = sum(
                    value * weight
                    for value, weight in zip(rgb, (0.2126, 0.7152, 0.0722))
                )
            if not math.isfinite(luminance):
                continue
            selected = {
                "landmark_id": int(landmark_id),
                "rgb": rgb,
                "luminance": luminance,
            }
            break
        if selected is not None:
            samples.append(selected)

    fallback = (0.075, 0.075, 0.085)

    def median_rgb(records: Sequence[Mapping[str, Any]]) -> tuple[float, float, float]:
        if not records:
            return fallback
        result = []
        for channel in range(3):
            values = sorted(float(record["rgb"][channel]) for record in records)
            midpoint = len(values) // 2
            result.append(
                values[midpoint]
                if len(values) % 2
                else (values[midpoint - 1] + values[midpoint]) * 0.5
            )
        return tuple(result)

    center_id = int(EYE_LANDMARKS[side]["center"])
    center_sample = next(
        (sample for sample in samples if sample["landmark_id"] == center_id),
        None,
    )
    edge_samples = [
        sample for sample in samples if sample["landmark_id"] != center_id
    ]
    if not edge_samples:
        edge_samples = list(samples)
    ordered_edges = sorted(edge_samples, key=lambda sample: sample["luminance"])
    keep_count = min(
        len(ordered_edges),
        max(2, int(math.ceil(len(ordered_edges) * 0.75))),
    )
    consensus = ordered_edges[:keep_count]
    middle = tuple(
        min(max(value, 0.006), 0.72) for value in median_rgb(consensus)
    )
    center = (
        tuple(float(value) for value in center_sample["rgb"])
        if center_sample is not None
        else tuple(value * 0.55 for value in middle)
    )
    center_luminance = (
        float(center_sample["luminance"])
        if center_sample is not None
        else sum(
            value * weight
            for value, weight in zip(center, (0.2126, 0.7152, 0.0722))
        )
    )
    agreement_limit = min(max(center_luminance + 0.16, 0.18), 0.55)
    edge_center_agreement_count = sum(
        float(sample["luminance"]) <= agreement_limit for sample in edge_samples
    )
    inner = tuple(
        min(max(middle[index] * 0.72 + center[index] * 0.28, 0.004), 0.72)
        for index in range(3)
    )
    # A slightly brighter middle band and a darker limbal ring preserve the
    # perceived original iris color while giving the remeshed disk its depth.
    middle = tuple(min(max(value * 1.06, 0.006), 0.72) for value in middle)
    outer = tuple(min(max(value * 0.72, 0.004), 0.60) for value in middle)
    pupil = tuple(
        min(max(min(center[index] * 0.35, middle[index] * 0.14), 0.001), 0.025)
        for index in range(3)
    )
    return {
        "iris_inner": inner,
        "iris_middle": middle,
        "iris_outer": outer,
        "pupil": pupil,
        "sample_count": len(samples),
        "edge_consensus_count": len(consensus),
        "edge_center_agreement_count": int(edge_center_agreement_count),
        "discarded_bright_edge_count": max(len(edge_samples) - len(consensus), 0),
        "samples": [
            {
                "landmark_id": int(sample["landmark_id"]),
                "luminance": float(sample["luminance"]),
                "rgb": [float(value) for value in sample["rgb"]],
            }
            for sample in samples
        ],
    }


def _estimate_original_iris_palette(
    payload: Mapping[str, Any], side: str
) -> dict[str, Any]:
    """Estimate one eye and use its mate only when this side is unreliable."""

    palette = _estimate_single_iris_palette(payload, side)
    opposite = "right" if side == "left" else "left"
    opposite_palette = _estimate_single_iris_palette(payload, opposite)
    palette["palette_source_side"] = side
    palette["borrowed_from_opposite_eye"] = False
    palette["opposite_edge_center_agreement_count"] = int(
        opposite_palette["edge_center_agreement_count"]
    )
    if (
        palette["edge_center_agreement_count"] < 2
        and opposite_palette["edge_center_agreement_count"] >= 2
    ):
        for key in ("iris_inner", "iris_middle", "iris_outer", "pupil"):
            palette[key] = opposite_palette[key]
        palette["palette_source_side"] = opposite
        palette["borrowed_from_opposite_eye"] = True
        palette["palette_source_samples"] = opposite_palette["samples"]
    return palette


def _render_original_texture_reference(
    objects: Sequence[Any], payload: Mapping[str, Any], size: int = 1024
) -> Any:
    """Render the untouched GLB from the landmark camera into an RGBA array."""

    import bpy
    import numpy as np
    from mathutils import Matrix, Quaternion, Vector

    corners = [
        obj.matrix_world @ Vector(corner)
        for obj in objects
        if obj.type == "MESH"
        for corner in obj.bound_box
    ]
    if not corners:
        raise RuntimeError("The imported GLB has no renderable mesh bounds.")
    minimum = Vector(
        tuple(min(float(corner[index]) for corner in corners) for index in range(3))
    )
    maximum = Vector(
        tuple(max(float(corner[index]) for corner in corners) for index in range(3))
    )
    center = (minimum + maximum) * 0.5
    extents = maximum - minimum
    maximum_extent = max(float(extents.x), float(extents.y), float(extents.z), 1.0e-3)
    view_directions = {
        "neg_y": (0.0, -1.0, 0.0),
        "pos_y": (0.0, 1.0, 0.0),
        "neg_x": (-1.0, 0.0, 0.0),
        "pos_x": (1.0, 0.0, 0.0),
        "neg_z": (0.0, 0.0, -1.0),
        "pos_z": (0.0, 0.0, 1.0),
    }
    direction = Vector(
        view_directions.get(str(payload.get("selected_view", "neg_z")), (0.0, 0.0, -1.0))
    ).normalized()
    yaw_offset = float(payload.get("yaw_offset", 0.0) or 0.0)
    if yaw_offset:
        direction.rotate(Matrix.Rotation(math.radians(yaw_offset), 4, "Z"))

    camera_data = bpy.data.cameras.new("EyeAU_TextureReferenceCamera")
    camera = bpy.data.objects.new("EyeAU_TextureReferenceCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = center + direction * (maximum_extent * 3.0 + 1.0)
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    roll_offset = float(payload.get("roll_offset", 0.0) or 0.0)
    if roll_offset:
        camera.rotation_mode = "QUATERNION"
        camera.rotation_quaternion = (
            camera.rotation_euler.to_quaternion()
            @ Quaternion((0.0, 0.0, 1.0), math.radians(roll_offset))
        )
    camera_data.type = "ORTHO"
    camera_data.clip_start = 0.001
    camera_data.clip_end = maximum_extent * 20.0 + 100.0
    bpy.context.scene.camera = camera
    bpy.context.view_layer.update()
    inverse_camera = camera.matrix_world.inverted()
    local_corners = [inverse_camera @ corner for corner in corners]
    span_x = max(
        max(float(corner.x) for corner in local_corners)
        - min(float(corner.x) for corner in local_corners),
        1.0e-3,
    )
    span_y = max(
        max(float(corner.y) for corner in local_corners)
        - min(float(corner.y) for corner in local_corners),
        1.0e-3,
    )
    camera_data.ortho_scale = max(span_y, span_x) * 1.18

    scene = bpy.context.scene
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.world = scene.world or bpy.data.worlds.new("EyeAU_TextureReferenceWorld")
    scene.world.color = (0.43, 0.45, 0.47)
    scene.render.engine = "BLENDER_WORKBENCH"
    # FLAT keeps the source texture's albedo.  STUDIO lighting baked shadows
    # into the atlas and then the emissive eye preserved those gray shadows
    # from every viewing direction.
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "TEXTURE"
    scene.display.shading.show_shadows = False
    scene.display.shading.show_cavity = False
    scene.display.shading.show_specular_highlight = False
    scene.display.shading.background_type = "VIEWPORT"
    scene.display.shading.background_color = (0.43, 0.45, 0.47)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    descriptor, render_name = tempfile.mkstemp(
        prefix="toporig_iris_reference_", suffix=".png"
    )
    os.close(descriptor)
    render_path = Path(render_name)
    try:
        scene.render.filepath = str(render_path)
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        bpy.ops.render.render(write_still=True)
        render = bpy.data.images.load(str(render_path), check_existing=False)
        width, height = int(render.size[0]), int(render.size[1])
        flat = np.empty(width * height * 4, dtype=np.float32)
        render.pixels.foreach_get(flat)
        pixels = flat.reshape(height, width, 4).copy()
        bpy.data.images.remove(render)
    finally:
        render_path.unlink(missing_ok=True)
    bpy.data.objects.remove(camera, do_unlink=True)
    return pixels


def _fit_smooth_sclera_field(
    sample_coordinates: Any,
    sample_colors: Any,
    query_coordinates: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """Fit a robust, smoothly varying sclera color field.

    Extending the closest visible sclera pixel over the back of an eyeball
    creates long Voronoi wedges.  A low-order surface preserves the original
    eye's tint and broad lighting variation without baking the eyelid outline
    or those view-dependent wedges into the replacement sphere.
    """

    import numpy as np

    coordinates = np.asarray(sample_coordinates, dtype=np.float64).reshape(-1, 2)
    colors = np.asarray(sample_colors, dtype=np.float64).reshape(-1, 4)
    queries = np.asarray(query_coordinates, dtype=np.float64).reshape(-1, 2)
    finite = np.all(np.isfinite(coordinates), axis=1) & np.all(
        np.isfinite(colors), axis=1
    )
    coordinates = coordinates[finite]
    colors = np.clip(colors[finite], 0.0, 1.0)
    fallback = np.asarray((0.82, 0.82, 0.78, 1.0), dtype=np.float64)
    if len(colors):
        fallback = np.median(colors, axis=0)
        fallback[3] = 1.0

    if len(colors) < 12:
        fitted = np.repeat(fallback[None, :], len(queries), axis=0)
        return fitted, fallback, {
            "model": "median_fallback",
            "sample_count": int(len(colors)),
            "inlier_count": int(len(colors)),
        }

    luminance = (
        colors[:, 0] * 0.2126
        + colors[:, 1] * 0.7152
        + colors[:, 2] * 0.0722
    )
    chroma = np.ptp(colors[:, :3], axis=1)
    neutral_brightness = luminance - chroma * 0.65
    # Prefer bright, low-chroma pixels.  This removes brown skin and lashes
    # even when either occupies more than a simple dark-pixel percentile.
    keep = (
        neutral_brightness >= np.percentile(neutral_brightness, 45.0)
    ) & (luminance >= np.percentile(luminance, 30.0))
    if int(np.count_nonzero(keep)) < 12:
        keep = np.ones(len(colors), dtype=bool)

    scale = np.maximum(
        np.percentile(np.abs(coordinates[keep]), 90.0, axis=0), 1.0
    )

    def features(values: Any) -> Any:
        normalized = np.asarray(values, dtype=np.float64) / scale
        x = normalized[:, 0]
        y = normalized[:, 1]
        return np.column_stack((np.ones(len(normalized)), x, y, x * y, x * x, y * y))

    coefficients = None
    for _iteration in range(3):
        design = features(coordinates[keep])
        coefficients, _residuals, _rank, _singular = np.linalg.lstsq(
            design, colors[keep, :3], rcond=None
        )
        predicted = features(coordinates) @ coefficients
        error = np.linalg.norm(predicted - colors[:, :3], axis=1)
        candidate_errors = error[keep]
        cutoff = float(np.percentile(candidate_errors, 88.0))
        revised = keep & (error <= max(cutoff, 0.015))
        if int(np.count_nonzero(revised)) < 12 or np.array_equal(revised, keep):
            break
        keep = revised

    assert coefficients is not None
    predicted_rgb = features(queries) @ coefficients
    lower = np.percentile(colors[keep, :3], 5.0, axis=0)
    upper = np.percentile(colors[keep, :3], 95.0, axis=0)
    predicted_rgb = np.clip(predicted_rgb, lower, upper)
    fitted = np.column_stack((predicted_rgb, np.ones(len(queries))))
    fill = np.median(colors[keep], axis=0)
    fill[3] = 1.0
    fill_luminance = float(
        fill[0] * 0.2126 + fill[1] * 0.7152 + fill[2] * 0.0722
    )
    fill_chroma = float(np.ptp(fill[:3]))
    if fill_luminance < 0.48 or fill_chroma > 0.22:
        # If even the neutral cluster looks like skin/iris, the contour or
        # camera is unreliable.  A conservative warm sclera is preferable to
        # baking a skin-colored or charcoal eyeball.
        fill = np.asarray((0.72, 0.70, 0.67, 1.0), dtype=np.float64)
        fitted = np.repeat(fill[None, :], len(queries), axis=0)
        return fitted, fill, {
            "model": "safe_neutral_fallback",
            "sample_count": int(len(colors)),
            "inlier_count": int(np.count_nonzero(keep)),
            "rejected_luminance": fill_luminance,
            "rejected_chroma": fill_chroma,
        }
    return fitted, fill, {
        "model": "robust_quadratic",
        "sample_count": int(len(colors)),
        "inlier_count": int(np.count_nonzero(keep)),
        "coordinate_scale": [float(value) for value in scale],
        "selection": "bright_low_chroma",
    }


def _fit_radial_iris_field(
    sample_radial: Any,
    sample_colors: Any,
    query_radial: Any,
) -> tuple[Any | None, dict[str, Any]]:
    """Complete eyelid-hidden iris sectors from visible source iris pixels."""

    import numpy as np

    radial = np.asarray(sample_radial, dtype=np.float64).reshape(-1)
    colors = np.asarray(sample_colors, dtype=np.float64).reshape(-1, 4)
    queries = np.asarray(query_radial, dtype=np.float64).reshape(-1)
    valid = (
        np.isfinite(radial)
        & np.all(np.isfinite(colors), axis=1)
        & (radial >= 0.0)
        & (radial <= 1.02)
    )
    radial = radial[valid]
    colors = np.clip(colors[valid], 0.0, 1.0)
    if len(colors) < 24:
        return None, {"model": "geometry_fallback", "sample_count": int(len(colors))}

    bin_count = 48
    centers = (np.arange(bin_count, dtype=np.float64) + 0.5) / bin_count
    bands = np.full((bin_count, 4), np.nan, dtype=np.float64)
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        selected = (radial >= lower) & (radial < upper)
        if np.any(selected):
            bands[index] = np.median(colors[selected], axis=0)
    available = np.flatnonzero(np.all(np.isfinite(bands), axis=1))
    if len(available) < 6:
        return None, {"model": "geometry_fallback", "sample_count": int(len(colors))}
    completed = np.empty_like(bands)
    for channel in range(4):
        completed[:, channel] = np.interp(
            centers,
            centers[available],
            bands[available, channel],
        )
    completed[:, 3] = 1.0
    indices = np.clip(
        np.floor(np.clip(queries, 0.0, 1.0) * bin_count).astype(np.int64),
        0,
        bin_count - 1,
    )
    return completed[indices], {
        "model": "visible_iris_radial_median",
        "sample_count": int(len(colors)),
        "populated_radial_bins": int(len(available)),
    }


def _bake_original_iris_texture(
    *,
    side: str,
    payload: Mapping[str, Any],
    reference_pixels: Any | None,
    sources: Sequence[Any],
    backing: Any,
    sphere_center: Any,
    surface_radius: float,
    eye_width: float,
    iris_direction: Any,
    iris_angle: float,
    iris_palette: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Bake the original layered iris appearance into an embedded UV texture.

    Vertex-color attributes are not consistently reconnected to their shader
    after a Blender -> GLB -> Blender round trip.  We still use the sphere
    vertices as dense source samples, then rasterize those samples into a
    packed image and give the front of the sphere a stable, iris-local UV map.
    """

    import bpy
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    from mathutils.interpolate import poly_3d_calc
    from mathutils.kdtree import KDTree

    image_cache: dict[int, tuple[Any, int, int]] = {}
    surface_indices = []
    for source in sources:
        world_vertices = [
            source.matrix_world @ vertex.co for vertex in source.data.vertices
        ]
        polygons = [
            tuple(int(value) for value in polygon.vertices)
            for polygon in source.data.polygons
        ]
        if world_vertices and polygons:
            surface_indices.append(
                (
                    source,
                    BVHTree.FromPolygons(
                        world_vertices, polygons, all_triangles=False
                    ),
                )
            )
    if not surface_indices:
        raise RuntimeError(f"No {side} source surfaces can carry iris texture.")

    def upstream_image(node: Any, visited: set[int] | None = None) -> Any | None:
        if visited is None:
            visited = set()
        pointer = int(node.as_pointer())
        if pointer in visited:
            return None
        visited.add(pointer)
        if node.type == "TEX_IMAGE" and getattr(node, "image", None) is not None:
            return node.image
        for socket in getattr(node, "inputs", ()):
            if not socket.is_linked:
                continue
            for link in socket.links:
                image = upstream_image(link.from_node, visited)
                if image is not None:
                    return image
        return None

    def image_pixels(image: Any) -> tuple[Any, int, int] | None:
        key = int(image.as_pointer())
        cached = image_cache.get(key)
        if cached is not None:
            return cached
        width, height = int(image.size[0]), int(image.size[1])
        if width < 1 or height < 1:
            return None
        flat = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(flat)
        cached = (flat.reshape(height, width, 4), width, height)
        image_cache[key] = cached
        return cached

    def image_color(image: Any, uv: Any) -> Any | None:
        cached = image_pixels(image)
        if cached is None:
            return None
        pixels, width, height = cached
        x = (float(uv.x) % 1.0) * (width - 1)
        y = (float(uv.y) % 1.0) * (height - 1)
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
        tx, ty = x - x0, y - y0
        top = pixels[y0, x0] * (1.0 - tx) + pixels[y0, x1] * tx
        bottom = pixels[y1, x0] * (1.0 - tx) + pixels[y1, x1] * tx
        return np.asarray(top * (1.0 - ty) + bottom * ty, dtype=np.float64)

    def surface_uv(source: Any, polygon_index: int, world_location: Any) -> Any | None:
        uv_layer = source.data.uv_layers.active
        if uv_layer is None:
            return None
        polygon = source.data.polygons[polygon_index]
        local_location = source.matrix_world.inverted_safe() @ world_location
        coordinates = [
            source.data.vertices[int(vertex_id)].co.copy()
            for vertex_id in polygon.vertices
        ]
        try:
            weights = poly_3d_calc(coordinates, local_location)
        except (RuntimeError, ValueError):
            return None
        uv = Vector((0.0, 0.0))
        total = 0.0
        for loop_index, weight in zip(polygon.loop_indices, weights):
            value = float(weight)
            uv += uv_layer.data[int(loop_index)].uv * value
            total += value
        return uv / total if abs(total) > 1.0e-8 else None

    def material_color(
        source: Any, polygon_index: int, world_location: Any
    ) -> Any | None:
        polygon = source.data.polygons[polygon_index]
        if polygon.material_index >= len(source.data.materials):
            return None
        material = source.data.materials[polygon.material_index]
        if material is None:
            return None
        fallback = np.asarray(material.diffuse_color, dtype=np.float64)
        if fallback.size < 4:
            fallback = np.append(fallback[:3], 1.0)
        if not material.use_nodes or material.node_tree is None:
            return np.clip(fallback[:4], 0.0, 1.0)
        principled = next(
            (
                node
                for node in material.node_tree.nodes
                if node.type == "BSDF_PRINCIPLED"
            ),
            None,
        )
        if principled is None:
            return np.clip(fallback[:4], 0.0, 1.0)
        base_socket = principled.inputs.get("Base Color")
        alpha_socket = principled.inputs.get("Alpha")
        base = (
            np.asarray(base_socket.default_value, dtype=np.float64)
            if base_socket is not None
            else fallback
        )
        alpha = (
            float(alpha_socket.default_value)
            if alpha_socket is not None
            else float(base[3] if base.size >= 4 else 1.0)
        )
        image = None
        if base_socket is not None:
            for link in base_socket.links:
                image = upstream_image(link.from_node)
                if image is not None:
                    break
        uv = surface_uv(source, polygon_index, world_location)
        sampled = image_color(image, uv) if image is not None and uv is not None else None
        if sampled is not None:
            rgb = sampled[:3]
            if sampled.size >= 4:
                alpha *= float(sampled[3])
        else:
            rgb = base[:3]
        return np.asarray(
            (*np.clip(rgb, 0.0, 1.0), min(max(alpha, 0.0), 1.0)),
            dtype=np.float64,
        )

    def palette_fallback(radial: float) -> Any:
        pupil_limit = 0.48
        if radial <= pupil_limit:
            rgb = iris_palette["pupil"]
        elif radial <= 0.75:
            amount = (radial - pupil_limit) / (0.75 - pupil_limit)
            rgb = tuple(
                float(iris_palette["iris_inner"][index]) * (1.0 - amount)
                + float(iris_palette["iris_middle"][index]) * amount
                for index in range(3)
            )
        else:
            amount = (radial - 0.75) / 0.25
            rgb = tuple(
                float(iris_palette["iris_middle"][index]) * (1.0 - amount)
                + float(iris_palette["iris_outer"][index]) * amount
                for index in range(3)
            )
        return np.asarray((*rgb, 1.0), dtype=np.float64)

    sclera = np.asarray((0.82, 0.82, 0.78, 1.0), dtype=np.float64)
    directions = []
    colors = [sclera.copy() for _vertex in backing.data.vertices]
    sampled_indices = []
    source_hit_counts: dict[str, int] = {}
    ray_origin_radius = surface_radius * 2.4
    ray_distance = surface_radius * 3.0
    sample_limit = iris_angle * 1.04
    for vertex in backing.data.vertices:
        direction = backing.matrix_world @ vertex.co - sphere_center
        if direction.length <= 1.0e-8:
            direction = iris_direction.copy()
        else:
            direction.normalize()
        directions.append(direction.copy())
        angle = math.acos(
            max(-1.0, min(1.0, float(direction.dot(iris_direction))))
        )
        if angle > sample_limit:
            continue
        radial = min(angle / max(iris_angle, 1.0e-8), 1.0)
        inside_iris = angle <= iris_angle
        if inside_iris:
            colors[int(vertex.index)] = palette_fallback(radial)
        # Reproduce the neutral visible eye rather than shooting radial rays
        # toward the fitted center.  The source eye is a stack of open caps;
        # radial rays can strike the side of an opaque sclera cap and hide the
        # iris layer that is visible from the frontal landmark camera.  Keep
        # the target vertex's tangent-plane coordinate and cast parallel to the
        # detected iris/view direction instead.
        world_point = backing.matrix_world @ vertex.co
        point_delta = world_point - sphere_center
        tangent = point_delta - iris_direction * point_delta.dot(iris_direction)
        origin = sphere_center + tangent + iris_direction * ray_origin_radius
        hits = []
        for source, tree in surface_indices:
            location, _normal, polygon_index, distance = tree.ray_cast(
                origin, -iris_direction, ray_distance
            )
            if location is None or polygon_index is None or distance is None:
                continue
            hit_radius = (location - sphere_center).length
            if not surface_radius * 0.45 <= hit_radius <= surface_radius * 1.75:
                continue
            rgba = material_color(source, int(polygon_index), location)
            if rgba is None or float(rgba[3]) <= 0.01:
                continue
            hits.append((float(distance), source, rgba))
        if not hits:
            continue
        hits.sort(key=lambda item: item[0])
        # The Pixel3D eye is commonly a stack of partial caps.  A ray through
        # the iris can therefore hit an opaque white sclera fragment before a
        # valid iris fragment.  Compare hits with the identity-specific color
        # measured at the five iris landmarks and skip only those conspicuous
        # white occluders.  This keeps genuine light irises possible while
        # repairing holes in dark irises without a hard-coded eye color.
        expected = colors[int(vertex.index)][:3]
        expected_luminance = float(
            0.2126 * expected[0] + 0.7152 * expected[1] + 0.0722 * expected[2]
        )
        maximum_iris_luminance = min(
            0.92, expected_luminance * 3.0 + 0.08
        )
        if inside_iris:
            iris_hits = [
                hit
                for hit in hits
                if float(
                    0.2126 * hit[2][0]
                    + 0.7152 * hit[2][1]
                    + 0.0722 * hit[2][2]
                )
                <= maximum_iris_luminance
            ]
            if not iris_hits:
                # Keep the landmark-derived color already assigned above.
                # It closes gaps between source triangles rather than cutting
                # white holes into the transferred iris.
                continue
            hits = iris_hits
        accumulated = np.zeros(3, dtype=np.float64)
        accumulated_alpha = 0.0
        used_sources = set()
        for _distance, source, rgba in hits:
            contribution = float(rgba[3]) * (1.0 - accumulated_alpha)
            accumulated += rgba[:3] * contribution
            accumulated_alpha += contribution
            used_sources.add(source.name)
            if accumulated_alpha >= 0.995:
                break
        if accumulated_alpha <= 0.05:
            continue
        fallback = colors[int(vertex.index)][:3]
        colors[int(vertex.index)] = np.asarray(
            (*(
                accumulated + fallback * (1.0 - accumulated_alpha)
            ), 1.0),
            dtype=np.float64,
        )
        sampled_indices.append(int(vertex.index))
        for name in used_sources:
            source_hit_counts[name] = source_hit_counts.get(name, 0) + 1

    # Fill gaps between disconnected source layers with the nearest actual
    # sample, but retain a circular mask from the landmark-derived iris angle.
    if sampled_indices:
        tree = KDTree(len(sampled_indices))
        for vertex_index in sampled_indices:
            tree.insert(directions[vertex_index], vertex_index)
        tree.balance()
        for vertex in backing.data.vertices:
            direction = directions[int(vertex.index)]
            angle = math.acos(
                max(-1.0, min(1.0, float(direction.dot(iris_direction))))
            )
            if angle > iris_angle or int(vertex.index) in sampled_indices:
                continue
            _location, nearest_index, _distance = tree.find(direction)
            colors[int(vertex.index)] = colors[int(nearest_index)].copy()

    # Give the iris most of a square atlas instead of squeezing it into a
    # small patch of an equirectangular eye texture.  The UVs remain attached
    # to the sphere, so the transferred appearance follows every eye rotation.
    atlas_size = 512
    iris_patch_extent = math.sin(
        min(max(iris_angle * 1.16, iris_angle + math.radians(1.5)), 1.2)
    )
    # Allocate the atlas to the complete eye opening, not just the iris.  A
    # sphere projects non-linearly near its silhouette, so use tangent-plane
    # distance and leave a small margin beyond the two eye corners.
    eye_patch_extent = min(
        max(float(eye_width) * 0.56 / max(surface_radius, 1.0e-8), 0.25),
        0.94,
    )
    patch_extent = max(iris_patch_extent, eye_patch_extent, 1.0e-4)
    reference = min(
        (Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0))),
        key=lambda axis: abs(float(axis.dot(iris_direction))),
    )
    tangent_u = reference - iris_direction * reference.dot(iris_direction)
    tangent_u.normalize()
    tangent_v = iris_direction.cross(tangent_u)
    tangent_v.normalize()

    sample_tree = KDTree(len(directions))
    for vertex_index, direction in enumerate(directions):
        sample_tree.insert(direction, vertex_index)
    sample_tree.balance()
    atlas = np.empty((atlas_size, atlas_size, 4), dtype=np.float32)
    for y in range(atlas_size):
        tangent_y = ((y + 0.5) / atlas_size * 2.0 - 1.0) * patch_extent
        for x in range(atlas_size):
            tangent_x = ((x + 0.5) / atlas_size * 2.0 - 1.0) * patch_extent
            tangent_squared = tangent_x * tangent_x + tangent_y * tangent_y
            if tangent_squared >= 0.999:
                atlas[y, x] = sclera
                continue
            direction = (
                iris_direction * math.sqrt(max(1.0 - tangent_squared, 0.0))
                + tangent_u * tangent_x
                + tangent_v * tangent_y
            )
            direction.normalize()
            _location, nearest_index, _distance = sample_tree.find(direction)
            atlas[y, x] = np.clip(colors[int(nearest_index)], 0.0, 1.0)

    reference_sample_count = 0
    iris_reference_sample_count = 0
    sclera_fill_color = sclera.copy()
    sclera_extension_report: dict[str, Any] = {"model": "geometry_fallback"}
    iris_completion_report: dict[str, Any] = {"model": "geometry_fallback"}
    iris_visual_scale = 0.82
    if reference_pixels is not None:
        candidates = payload.get("iris_surface_anchor_candidates", {})
        anchors = payload.get("surface_anchors", {})

        def screen_position(landmark_id: int) -> Any | None:
            records = []
            for source in (anchors, candidates):
                raw = (
                    source.get(str(landmark_id))
                    if isinstance(source, Mapping)
                    else None
                )
                if isinstance(raw, Mapping):
                    records.append(raw)
                elif isinstance(raw, Sequence):
                    records.extend(raw)
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                position = record.get("screen_position")
                if not isinstance(position, Mapping):
                    continue
                try:
                    value = np.asarray(
                        (float(position["x"]), float(position["y"])),
                        dtype=np.float64,
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if np.all(np.isfinite(value)):
                    return value
            return None

        eye_contour_ids = {
            "left": (
                362, 382, 381, 380, 374, 373, 390, 249,
                263, 466, 388, 387, 386, 385, 384, 398,
            ),
            "right": (
                33, 7, 163, 144, 145, 153, 154, 155,
                133, 173, 157, 158, 159, 160, 161, 246,
            ),
        }

        def inside_polygon(point: Any, polygon: Sequence[Any]) -> bool:
            inside = False
            previous = polygon[-1]
            for current in polygon:
                if (float(current[1]) > float(point[1])) != (
                    float(previous[1]) > float(point[1])
                ):
                    crossing_x = float(previous[0]) + (
                        float(point[1]) - float(previous[1])
                    ) * (float(current[0]) - float(previous[0])) / (
                        float(current[1]) - float(previous[1])
                    )
                    if float(point[0]) < crossing_x:
                        inside = not inside
                previous = current
            return inside

        iris_ids = EYE_LANDMARKS[side]["iris"]
        screen_points = [screen_position(int(value)) for value in iris_ids]
        contour_points = [
            screen_position(int(value)) for value in eye_contour_ids[side]
        ]
        if all(value is not None for value in (*screen_points, *contour_points)):
            center_screen, right_screen, top_screen, left_screen, bottom_screen = screen_points
            horizontal_screen = (right_screen - left_screen) * 0.5
            vertical_down_screen = (bottom_screen - top_screen) * 0.5
            reference_height, reference_width = reference_pixels.shape[:2]

            def reference_color(screen: Any) -> Any:
                source_x = min(max(float(screen[0]), 0.0), 1.0) * (
                    reference_width - 1
                )
                # Landmark coordinates use a top-left origin; Blender image
                # buffers use a bottom-left origin.
                source_y = (1.0 - min(max(float(screen[1]), 0.0), 1.0)) * (
                    reference_height - 1
                )
                x0, y0 = int(math.floor(source_x)), int(math.floor(source_y))
                x1 = min(x0 + 1, reference_width - 1)
                y1 = min(y0 + 1, reference_height - 1)
                tx, ty = source_x - x0, source_y - y0
                lower = (
                    reference_pixels[y0, x0] * (1.0 - tx)
                    + reference_pixels[y0, x1] * tx
                )
                upper = (
                    reference_pixels[y1, x0] * (1.0 - tx)
                    + reference_pixels[y1, x1] * tx
                )
                return lower * (1.0 - ty) + upper * ty

            iris_extent = max(math.sin(iris_angle), 1.0e-4)
            target_iris_extent = iris_extent * iris_visual_scale
            # Keep a complete, circular iris fallback for portions hidden by
            # the neutral eyelids.  The visible source render replaces it
            # wherever real iris pixels exist.
            geometry_atlas = atlas.copy()
            axis_coordinates = (
                (np.arange(atlas_size, dtype=np.float64) + 0.5)
                / atlas_size
                * 2.0
                - 1.0
            ) * patch_extent
            tangent_x_grid, tangent_y_grid = np.meshgrid(
                axis_coordinates, axis_coordinates
            )
            iris_x_grid = tangent_x_grid / target_iris_extent
            iris_y_grid = tangent_y_grid / target_iris_extent
            query_coordinates = np.column_stack(
                (iris_x_grid.reshape(-1), iris_y_grid.reshape(-1))
            )
            reference_colors = np.zeros(
                (atlas_size, atlas_size, 4), dtype=np.float32
            )
            reference_mask = np.zeros((atlas_size, atlas_size), dtype=bool)
            visible_sclera_coordinates = []
            visible_sclera_colors = []
            for y in range(atlas_size):
                iris_y = float(iris_y_grid[y, 0])
                for x in range(atlas_size):
                    iris_x = float(iris_x_grid[y, x])
                    source_screen = (
                        center_screen
                        + horizontal_screen * iris_x
                        - vertical_down_screen * iris_y
                    )
                    if not inside_polygon(source_screen, contour_points):
                        continue
                    reference_colors[y, x] = np.clip(
                        reference_color(source_screen), 0.0, 1.0
                    )
                    reference_colors[y, x, 3] = 1.0
                    reference_mask[y, x] = True
                    reference_sample_count += 1
                    radial_squared = iris_x * iris_x + iris_y * iris_y
                    if radial_squared >= 1.16**2:
                        visible_sclera_coordinates.append((iris_x, iris_y))
                        visible_sclera_colors.append(
                            reference_colors[y, x].copy()
                        )

            if visible_sclera_colors:
                smooth_sclera, sclera_fill_color, sclera_extension_report = (
                    _fit_smooth_sclera_field(
                        visible_sclera_coordinates,
                        visible_sclera_colors,
                        query_coordinates,
                    )
                )
                atlas = smooth_sclera.reshape(
                    atlas_size, atlas_size, 4
                ).astype(np.float32)
                radial = np.sqrt(iris_x_grid * iris_x_grid + iris_y_grid * iris_y_grid)
                # Exact original pixels define the visible iris and pupil.
                # The geometry-derived circular fallback fills only the iris
                # sectors hidden by the neutral lids, so newly revealed areas
                # remain iris rather than turning white during rotation.
                source_x = np.clip(
                    np.rint(
                        (
                            tangent_x_grid / iris_visual_scale / patch_extent
                            + 1.0
                        )
                        * 0.5
                        * atlas_size
                        - 0.5
                    ).astype(np.int64),
                    0,
                    atlas_size - 1,
                )
                source_y = np.clip(
                    np.rint(
                        (
                            tangent_y_grid / iris_visual_scale / patch_extent
                            + 1.0
                        )
                        * 0.5
                        * atlas_size
                        - 0.5
                    ).astype(np.int64),
                    0,
                    atlas_size - 1,
                )
                iris_layer = geometry_atlas[source_y, source_x]
                radial_iris, iris_completion_report = _fit_radial_iris_field(
                    radial[reference_mask & (radial <= 1.0)],
                    reference_colors[reference_mask & (radial <= 1.0)],
                    radial.reshape(-1),
                )
                if radial_iris is not None:
                    iris_layer = radial_iris.reshape(
                        atlas_size, atlas_size, 4
                    ).astype(np.float32)
                visible_iris = reference_mask & (radial <= 1.08)
                iris_layer[visible_iris] = reference_colors[visible_iris]
                iris_reference_sample_count = int(
                    np.count_nonzero(visible_iris & (radial <= 1.0))
                )
                blend = np.clip((1.08 - radial) / 0.08, 0.0, 1.0)
                blend = blend * blend * (3.0 - 2.0 * blend)
                atlas[:, :, :3] = (
                    iris_layer[:, :, :3] * blend[:, :, None]
                    + atlas[:, :, :3] * (1.0 - blend[:, :, None])
                )
                atlas[:, :, 3] = 1.0

    uv_layer = backing.data.uv_layers.active
    if uv_layer is None:
        uv_layer = backing.data.uv_layers.new(name="EyeAU_IrisUV")
    else:
        uv_layer.name = "EyeAU_IrisUV"
    for polygon in backing.data.polygons:
        for loop_index in polygon.loop_indices:
            vertex = backing.data.vertices[backing.data.loops[loop_index].vertex_index]
            direction = backing.matrix_world @ vertex.co - sphere_center
            if direction.length <= 1.0e-8:
                uv = (0.0, 0.0)
            else:
                direction.normalize()
                if float(direction.dot(iris_direction)) <= 0.0:
                    uv = (0.0, 0.0)
                else:
                    uv = (
                        min(max(0.5 + 0.5 * float(direction.dot(tangent_u)) / patch_extent, 0.0), 1.0),
                        min(max(0.5 + 0.5 * float(direction.dot(tangent_v)) / patch_extent, 0.0), 1.0),
                    )
            uv_layer.data[loop_index].uv = uv

    image = bpy.data.images.new(
        f"EyeAU_OriginalEyeTexture_{side}",
        width=atlas_size,
        height=atlas_size,
        alpha=True,
        float_buffer=False,
    )
    image.pixels.foreach_set(atlas.reshape(-1))
    image.pack()

    old_materials = list(backing.data.materials)
    backing.data.materials.clear()
    material = bpy.data.materials.new(f"EyeAU_OriginalEyeBake_{side}")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    image_texture = nodes.new("ShaderNodeTexImage")
    image_texture.image = image
    image_texture.interpolation = "Linear"
    image_texture.extension = "EXTEND"
    material.node_tree.links.new(
        image_texture.outputs["Color"], emission.inputs["Color"]
    )
    material.node_tree.links.new(
        emission.outputs["Emission"], output.inputs["Surface"]
    )
    emission.inputs["Strength"].default_value = 1.0
    backing.data.materials.append(material)
    for polygon in backing.data.polygons:
        polygon.material_index = 0
    backing.data.update()
    for old_material in old_materials:
        if old_material is not None and old_material.users == 0:
            bpy.data.materials.remove(old_material)
    return material, {
        "mode": "smooth_original_complete_eye_uv_texture_bake",
        "sampled_vertex_count": len(sampled_indices),
        "total_vertex_count": len(backing.data.vertices),
        "source_hit_counts": source_hit_counts,
        "uses_vertex_colors": False,
        "uses_image_texture": True,
        "image_size": [atlas_size, atlas_size],
        "uv_layout": "iris_local_planar",
        "preserves_source_uv_samples": True,
        "sampling_projection": "neutral_parallel_view",
        "visible_reference_sample_count": reference_sample_count,
        "visible_reference_transfer": bool(reference_sample_count),
        "visible_reference_scope": "iris_detail_and_full_eye_color_model",
        "visible_iris_reference_sample_count": iris_reference_sample_count,
        "iris_visual_scale": iris_visual_scale,
        "iris_completion": iris_completion_report,
        "sclera_extension": sclera_extension_report,
        "sclera_fill_color": [float(value) for value in sclera_fill_color],
        "view_independent_emissive_texture": True,
    }


def _build_opaque_eye_material_regions(
    *,
    side: str,
    payload: Mapping[str, Any],
    reference_pixels: Any | None,
    source_objects: Sequence[Any],
    backing: Any,
    pivot_radius: float,
    eye_width: float,
    iris_center: Any,
    iris_radius: float,
) -> tuple[Any, dict[str, Any]]:
    """Build one opaque sphere and bake the original visible iris into it.

    The texture is embedded in the GLB, uses an iris-local UV layout, and is
    connected as emission so the transferred pixels do not depend on a
    viewer's light placement.  Procedural material regions remain only as a
    fallback while the source appearance is sampled.
    """

    import bpy

    sources = [
        obj
        for obj in source_objects
        if obj is not None and obj.type == "MESH" and len(obj.data.polygons) > 0
    ]
    if not sources:
        raise RuntimeError(f"No {side} source eye surfaces remain for replacement.")

    sphere_center = backing.matrix_world.translation
    desired_direction = iris_center - sphere_center
    if desired_direction.length <= 1.0e-8:
        raise RuntimeError(f"The detected {side} iris direction is degenerate.")
    desired_direction.normalize()
    surface_radius = float(
        backing.get("toporig_backing_radius", float(pivot_radius))
    )
    iris_angle = math.asin(
        min(max(float(iris_radius) / max(surface_radius, 1.0e-8), 0.08), 0.72)
    )
    pupil_angle = iris_angle * 0.48

    iris_palette = _estimate_original_iris_palette(payload, side)

    old_materials = list(backing.data.materials)
    backing.data.materials.clear()
    for attribute in list(backing.data.color_attributes):
        backing.data.color_attributes.remove(attribute)

    def opaque_material(
        name: str,
        color: tuple[float, float, float, float],
        roughness_value: float,
    ) -> Any:
        material = bpy.data.materials.new(name)
        material.diffuse_color = color
        material.use_nodes = True
        principled = next(
            (
                node
                for node in material.node_tree.nodes
                if node.type == "BSDF_PRINCIPLED"
            ),
            None,
        )
        if principled is not None:
            base_color = principled.inputs.get("Base Color")
            alpha = principled.inputs.get("Alpha")
            roughness = principled.inputs.get("Roughness")
            metallic = principled.inputs.get("Metallic")
            if base_color is not None:
                base_color.default_value = color
            if alpha is not None:
                alpha.default_value = 1.0
            if roughness is not None:
                roughness.default_value = roughness_value
            if metallic is not None:
                metallic.default_value = 0.0
        return material

    sclera_material = opaque_material(
        f"EyeAU_ScleraMaterial_{side}", (0.82, 0.82, 0.78, 1.0), 0.48
    )

    def mix_rgb(
        first: Sequence[float], second: Sequence[float], amount: float
    ) -> tuple[float, float, float, float]:
        return tuple(
            float(first[index]) * (1.0 - amount)
            + float(second[index]) * amount
            for index in range(3)
        ) + (1.0,)

    iris_band_count = 6
    iris_materials = []
    for band_index in range(iris_band_count):
        radial = (band_index + 0.5) / iris_band_count
        if radial <= 0.58:
            color = mix_rgb(
                iris_palette["iris_inner"],
                iris_palette["iris_middle"],
                radial / 0.58,
            )
        else:
            color = mix_rgb(
                iris_palette["iris_middle"],
                iris_palette["iris_outer"],
                (radial - 0.58) / 0.42,
            )
        iris_materials.append(
            opaque_material(
                f"EyeAU_IrisMaterial_{side}_{band_index:02d}", color, 0.34
            )
        )
    pupil_material = opaque_material(
        f"EyeAU_PupilMaterial_{side}",
        tuple(float(value) for value in iris_palette["pupil"]) + (1.0,),
        0.38,
    )
    materials = (sclera_material, *iris_materials, pupil_material)
    for material in materials:
        backing.data.materials.append(material)

    region_counts = {"sclera": 0, "iris": 0, "pupil": 0}
    for polygon in backing.data.polygons:
        direction = backing.matrix_world @ polygon.center - sphere_center
        if direction.length <= 1.0e-8:
            polygon.material_index = 0
            region_counts["sclera"] += 1
            continue
        direction.normalize()
        angle = math.acos(
            max(-1.0, min(1.0, float(direction.dot(desired_direction))))
        )
        if angle <= pupil_angle:
            polygon.material_index = len(materials) - 1
            region_counts["pupil"] += 1
        elif angle <= iris_angle:
            iris_radial = (angle - pupil_angle) / max(
                iris_angle - pupil_angle, 1.0e-8
            )
            band_index = min(
                int(max(iris_radial, 0.0) * iris_band_count),
                iris_band_count - 1,
            )
            polygon.material_index = 1 + band_index
            region_counts["iris"] += 1
        else:
            polygon.material_index = 0
            region_counts["sclera"] += 1
        polygon.use_smooth = True

    if region_counts["iris"] < 4 or region_counts["pupil"] < 4:
        raise RuntimeError(
            f"The generated {side} eye has insufficient iris/pupil polygons: "
            f"{region_counts}."
        )
    backing.data.update()
    baked_material, texture_transfer_report = _bake_original_iris_texture(
        side=side,
        payload=payload,
        reference_pixels=reference_pixels,
        sources=sources,
        backing=backing,
        sphere_center=sphere_center,
        surface_radius=surface_radius,
        eye_width=float(eye_width),
        iris_direction=desired_direction,
        iris_angle=iris_angle,
        iris_palette=iris_palette,
    )
    for material in old_materials:
        if material is not None and material.users == 0:
            bpy.data.materials.remove(material)
    source_names = sorted(obj.name for obj in sources)
    for obj in sources:
        bpy.data.objects.remove(obj, do_unlink=True)

    backing.name = f"EyeAU_CompleteBall_{side}"
    backing.data.name = f"EyeAU_CompleteBallMesh_{side}"
    backing["toporig_complete_eye_ball"] = True
    backing["toporig_surface_projection"] = False
    backing["toporig_eye_surface_mode"] = (
        "smooth_original_complete_eye_uv_texture_bake"
    )
    backing["toporig_source_eye_objects"] = json.dumps(source_names)
    backing["toporig_original_iris_transfer"] = (
        "smooth_original_complete_eye_uv_texture_bake"
    )
    backing["toporig_iris_palette"] = json.dumps(
        {
            key: [float(value) for value in iris_palette[key]]
            for key in ("iris_inner", "iris_middle", "iris_outer", "pupil")
        }
    )
    return backing, {
        "method": "smooth_original_complete_eye_uv_texture_bake",
        "connected_spherical_surface": True,
        "source_shells_removed": True,
        "uses_image_texture": True,
        "uses_vertex_colors": False,
        "uses_alpha": False,
        "materials": [baked_material.name],
        "region_polygon_counts": region_counts,
        "original_iris_transfer": {
            **texture_transfer_report,
            "preserves_fragmented_source_topology": False,
            "palette": {
                key: [float(value) for value in iris_palette[key]]
                for key in ("iris_inner", "iris_middle", "iris_outer", "pupil")
            },
            "sample_count": int(iris_palette["sample_count"]),
            "edge_consensus_count": int(iris_palette["edge_consensus_count"]),
            "edge_center_agreement_count": int(
                iris_palette["edge_center_agreement_count"]
            ),
            "discarded_bright_edge_count": int(
                iris_palette["discarded_bright_edge_count"]
            ),
            "palette_source_side": iris_palette["palette_source_side"],
            "borrowed_from_opposite_eye": bool(
                iris_palette["borrowed_from_opposite_eye"]
            ),
            "samples": iris_palette["samples"],
        },
        "procedural_iris_radius": float(iris_radius),
        "procedural_iris_angle_degrees": float(math.degrees(iris_angle)),
        "source_objects": source_names,
    }


def _project_eye_appearance_onto_ball_legacy_texture(
    *,
    side: str,
    source_objects: Sequence[Any],
    backing: Any,
    pivot_radius: float,
    iris_center: Any,
    iris_radius: float,
) -> tuple[Any, dict[str, Any]]:
    """Bake the nearest visible eye-cap color onto a true sphere.

    Joining an open cap to a sphere still leaves two loose surfaces.  This
    routine instead keeps only the generated sphere and transfers the original
    cap's texture into a point-domain color attribute on the sphere. Avoiding
    material and UV seams keeps the exported GLB topologically connected. The
    resulting eye has one spherical surface and no exterior cap.
    """

    import bpy
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    from mathutils.interpolate import poly_3d_calc
    from mathutils.kdtree import KDTree

    sources = [
        obj
        for obj in source_objects
        if obj is not None and obj.type == "MESH" and len(obj.data.polygons) > 0
    ]
    if not sources:
        raise RuntimeError(f"No {side} source eye surfaces remain for projection.")

    surface_indices = {}
    for obj in sources:
        world_vertices = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
        polygons = [
            tuple(int(value) for value in polygon.vertices)
            for polygon in obj.data.polygons
        ]
        surface_indices[int(obj.as_pointer())] = BVHTree.FromPolygons(
            world_vertices,
            polygons,
            all_triangles=False,
        )

    def closest_sample(world_point: Any, objects: Sequence[Any]) -> Any | None:
        best = None
        for obj in objects:
            tree = surface_indices[int(obj.as_pointer())]
            world_location, _normal, polygon_index, distance = tree.find_nearest(
                world_point
            )
            if world_location is None or polygon_index is None or distance is None:
                continue
            location = obj.matrix_world.inverted_safe() @ world_location
            if best is None or distance < best[0]:
                best = (distance, obj, location, int(polygon_index))
        return best

    def surface_uv(obj: Any, polygon_index: int, location: Any) -> Any | None:
        uv_layer = obj.data.uv_layers.active
        if uv_layer is None:
            return None
        polygon = obj.data.polygons[polygon_index]
        coordinates = [
            obj.data.vertices[int(vertex_id)].co.copy()
            for vertex_id in polygon.vertices
        ]
        try:
            weights = poly_3d_calc(coordinates, location)
        except (RuntimeError, ValueError):
            return None
        uv = Vector((0.0, 0.0))
        total = 0.0
        for loop_index, weight in zip(polygon.loop_indices, weights):
            value = float(weight)
            uv += uv_layer.data[int(loop_index)].uv * value
            total += value
        if abs(total) <= 1.0e-8:
            return None
        return uv / total

    material_counts: dict[int, tuple[int, Any]] = {}
    for obj in sources:
        for polygon in obj.data.polygons:
            if polygon.material_index >= len(obj.data.materials):
                continue
            material = obj.data.materials[polygon.material_index]
            if material is None:
                continue
            key = int(material.as_pointer())
            count, _existing = material_counts.get(key, (0, material))
            material_counts[key] = (count + 1, material)
    if not material_counts:
        raise RuntimeError(f"The {side} appearance carrier has no material.")
    _count, appearance_material = max(material_counts.values(), key=lambda item: item[0])

    principled = (
        next(
            (
                node
                for node in appearance_material.node_tree.nodes
                if node.type == "BSDF_PRINCIPLED"
            ),
            None,
        )
        if appearance_material.use_nodes
        else None
    )
    base_color = principled.inputs.get("Base Color") if principled is not None else None
    base_image = None
    if base_color is not None:
        base_image = next(
            (
                link.from_node.image
                for link in base_color.links
                if link.from_node.type == "TEX_IMAGE"
                and link.from_node.image is not None
            ),
            None,
        )
    if base_image is None:
        raise RuntimeError(
            f"The {side} eye material has no directly linked base-color image; "
            "a single textured spherical surface cannot be built safely."
        )
    width, height = int(base_image.size[0]), int(base_image.size[1])
    if width < 2 or height < 2:
        raise RuntimeError(f"The {side} eye base-color image is empty.")
    flat_pixels = np.empty(width * height * 4, dtype=np.float32)
    base_image.pixels.foreach_get(flat_pixels)
    pixels = flat_pixels.reshape(height, width, 4)
    rgb = pixels[:, :, :3]
    neutral_score = np.min(rgb, axis=2) - 0.5 * (
        np.max(rgb, axis=2) - np.min(rgb, axis=2)
    )
    neutral_score = np.where(pixels[:, :, 3] >= 0.5, neutral_score, -np.inf)
    white_pixel_index = int(np.argmax(neutral_score))
    white_y, white_x = divmod(white_pixel_index, width)
    sclera_color = np.asarray(pixels[white_y, white_x], dtype=np.float64)
    sclera_color[3] = 1.0

    def texture_color(uv: Any) -> np.ndarray:
        x = (float(uv.x) % 1.0) * (width - 1)
        y = (float(uv.y) % 1.0) * (height - 1)
        x0 = int(math.floor(x))
        y0 = int(math.floor(y))
        x1 = min(x0 + 1, width - 1)
        y1 = min(y0 + 1, height - 1)
        tx = x - x0
        ty = y - y0
        top = pixels[y0, x0] * (1.0 - tx) + pixels[y0, x1] * tx
        bottom = pixels[y1, x0] * (1.0 - tx) + pixels[y1, x1] * tx
        color = np.asarray(top * (1.0 - ty) + bottom * ty, dtype=np.float64)
        # Image texture nodes decode sRGB before shading, while glTF vertex
        # colors are linear. Match the original texture-node appearance.
        color[:3] = np.where(
            color[:3] <= 0.04045,
            color[:3] / 12.92,
            np.power((color[:3] + 0.055) / 1.055, 2.4),
        )
        return color

    # Blender's glTF exporter/importer standardizes COLOR_0 to this name.
    color_name = "Color"
    for attribute in list(backing.data.color_attributes):
        backing.data.color_attributes.remove(attribute)
    color_attribute = backing.data.color_attributes.new(
        name=color_name,
        type="FLOAT_COLOR",
        domain="POINT",
    )
    backing.data.color_attributes.active_color_index = 0
    backing.data.color_attributes.render_color_index = 0
    generated_materials = list(backing.data.materials)
    backing.data.materials.clear()
    material = bpy.data.materials.new(f"EyeAU_SphericalMaterial_{side}")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    material_principled = next(
        (node for node in nodes if node.type == "BSDF_PRINCIPLED"),
        None,
    )
    if material_principled is None:
        material_principled = nodes.new("ShaderNodeBsdfPrincipled")
    vertex_color = nodes.new("ShaderNodeVertexColor")
    vertex_color.layer_name = color_name
    material.node_tree.links.new(
        vertex_color.outputs["Color"],
        material_principled.inputs["Base Color"],
    )
    material.node_tree.links.new(
        vertex_color.outputs["Alpha"],
        material_principled.inputs["Alpha"],
    )
    roughness = material_principled.inputs.get("Roughness")
    metallic = material_principled.inputs.get("Metallic")
    if roughness is not None:
        roughness.default_value = 0.48
    if metallic is not None:
        metallic.default_value = 0.0
    backing.data.materials.append(material)
    for polygon in backing.data.polygons:
        polygon.material_index = 0
    for old_material in generated_materials:
        if old_material is not None and old_material.users == 0:
            bpy.data.materials.remove(old_material)

    max_surface_distance = max(float(pivot_radius) * 0.25, 1.0e-6)
    projected_vertices = 0
    source_names: set[str] = set()
    baked_colors = []
    projected_mask = []
    for vertex in backing.data.vertices:
        color = sclera_color
        world_point = backing.matrix_world @ vertex.co
        samples = []
        for source in sources:
            sample = closest_sample(world_point, [source])
            if sample is None or sample[0] > max_surface_distance:
                continue
            distance, _source, location, source_polygon_index = sample
            source_polygon = source.data.polygons[source_polygon_index]
            if source_polygon.material_index >= len(source.data.materials):
                continue
            source_material = source.data.materials[source_polygon.material_index]
            if source_material != appearance_material:
                continue
            uv = surface_uv(source, source_polygon_index, location)
            if uv is None:
                continue
            sampled_color = texture_color(uv)
            luminance = float(
                np.dot(sampled_color[:3], (0.2126, 0.7152, 0.0722))
            )
            # Eye layers commonly overlap: sclera, iris and pupil can each be
            # disconnected caps. The darkest valid layer is the visible iris/
            # pupil appearance, with distance as a small tie breaker.
            samples.append((luminance + distance / pivot_radius * 0.01, sampled_color, source))
        if not samples:
            baked_colors.append(np.asarray(color, dtype=np.float64))
            projected_mask.append(False)
            continue
        _score, color, source = min(samples, key=lambda item: item[0])
        baked_colors.append(np.asarray(color, dtype=np.float64))
        projected_mask.append(True)
        source_names.add(source.name)
        projected_vertices += 1

    if projected_vertices < 4:
        raise RuntimeError(
            f"Only {projected_vertices} {side} sphere vertices received the source eye "
            "appearance; refusing to export an untextured procedural eye."
        )

    sphere_center = backing.matrix_world.translation
    directions = [
        (backing.matrix_world @ vertex.co - sphere_center).normalized()
        for vertex in backing.data.vertices
    ]
    projected_luminance = [
        float(np.dot(color[:3], (0.2126, 0.7152, 0.0722)))
        for color, projected in zip(baked_colors, projected_mask)
        if projected
    ]
    dark_limit = min(
        0.18,
        float(np.quantile(np.asarray(projected_luminance), 0.20)) + 0.03,
    )
    weights = [
        max(
            0.0,
            dark_limit
            - float(np.dot(color[:3], (0.2126, 0.7152, 0.0722))),
        )
        if projected
        else 0.0
        for color, projected in zip(baked_colors, projected_mask)
    ]
    appearance_alignment_degrees = 0.0
    if sum(weights) > 1.0e-8:
        dark_direction = sum(
            (direction * weight for direction, weight in zip(directions, weights)),
            Vector(),
        )
        desired_direction = iris_center - sphere_center
        if dark_direction.length > 1.0e-8 and desired_direction.length > 1.0e-8:
            dark_direction.normalize()
            desired_direction.normalize()
            alignment = dark_direction.rotation_difference(desired_direction)
            appearance_alignment_degrees = math.degrees(alignment.angle)
            if appearance_alignment_degrees > 0.05:
                tree = KDTree(len(directions))
                for index, direction in enumerate(directions):
                    tree.insert(direction, index)
                tree.balance()
                inverse_alignment = alignment.inverted()
                baked_colors = [
                    baked_colors[
                        tree.find(inverse_alignment @ direction)[1]
                    ]
                    for direction in directions
                ]

    desired_direction = iris_center - sphere_center
    if desired_direction.length <= 1.0e-8:
        raise RuntimeError(f"The detected {side} iris direction is degenerate.")
    desired_direction.normalize()
    iris_angle = math.asin(
        min(max(float(iris_radius) / max(float(pivot_radius), 1.0e-8), 0.08), 0.72)
    )
    pupil_angle = iris_angle * 0.48
    edge_softness = max(math.pi / 128.0, iris_angle * 0.06)
    iris_outer = np.asarray((0.075, 0.075, 0.085, 1.0), dtype=np.float64)
    iris_inner = np.asarray((0.025, 0.025, 0.032, 1.0), dtype=np.float64)
    pupil_color = np.asarray((0.002, 0.002, 0.002, 1.0), dtype=np.float64)
    for index, direction in enumerate(directions):
        angle = math.acos(max(-1.0, min(1.0, direction.dot(desired_direction))))
        if angle >= iris_angle + edge_softness:
            continue
        radial = min(max(angle / max(iris_angle, 1.0e-8), 0.0), 1.0)
        iris_color = iris_inner * (1.0 - radial) + iris_outer * radial
        if angle <= pupil_angle:
            eye_color = pupil_color
        else:
            eye_color = iris_color
        edge_mix = min(
            max((iris_angle + edge_softness - angle) / (2.0 * edge_softness), 0.0),
            1.0,
        )
        baked_colors[index] = (
            baked_colors[index] * (1.0 - edge_mix) + eye_color * edge_mix
        )

    texture_width = 1024
    texture_height = 512
    u = (np.arange(texture_width, dtype=np.float64) + 0.5) / texture_width
    v = (np.arange(texture_height, dtype=np.float64) + 0.5) / texture_height
    longitude = (u - 0.5) * (2.0 * math.pi)
    latitude = (v - 0.5) * math.pi
    cos_latitude = np.cos(latitude)[:, None]
    texture_directions = np.stack(
        np.broadcast_arrays(
            cos_latitude * np.cos(longitude)[None, :],
            cos_latitude * np.sin(longitude)[None, :],
            np.sin(latitude)[:, None],
        ),
        axis=2,
    )
    desired = np.asarray(
        (1.0, 0.0, 0.0)
        if bool(backing.get("toporig_iris_uv_reoriented", False))
        else tuple(desired_direction),
        dtype=np.float64,
    )
    texture_angles = np.arccos(
        np.clip(texture_directions @ desired, -1.0, 1.0)
    )
    texture_pixels = np.empty(
        (texture_height, texture_width, 4), dtype=np.float32
    )
    texture_pixels[:, :, :] = np.asarray((0.82, 0.82, 0.78, 1.0), dtype=np.float32)
    radial = np.clip(texture_angles / max(iris_angle, 1.0e-8), 0.0, 1.0)
    procedural_iris = (
        iris_inner[None, None, :] * (1.0 - radial[:, :, None])
        + iris_outer[None, None, :] * radial[:, :, None]
    )
    procedural_eye = np.where(
        (texture_angles <= pupil_angle)[:, :, None],
        pupil_color[None, None, :],
        procedural_iris,
    )
    edge_mix = np.clip(
        (iris_angle + edge_softness - texture_angles) / (2.0 * edge_softness),
        0.0,
        1.0,
    )
    texture_pixels = (
        texture_pixels * (1.0 - edge_mix[:, :, None])
        + procedural_eye * edge_mix[:, :, None]
    ).astype(np.float32)
    image = bpy.data.images.new(
        f"EyeAU_SphericalTexture_{side}",
        width=texture_width,
        height=texture_height,
        alpha=True,
    )
    image.pixels.foreach_set(texture_pixels.reshape(-1))
    image.pack()

    nodes = material.node_tree.nodes
    nodes.clear()
    output_node = nodes.new("ShaderNodeOutputMaterial")
    material_principled = nodes.new("ShaderNodeBsdfPrincipled")
    texture_node = nodes.new("ShaderNodeTexImage")
    texture_node.image = image
    texture_node.interpolation = "Linear"
    material.node_tree.links.new(
        texture_node.outputs["Color"], material_principled.inputs["Base Color"]
    )
    material.node_tree.links.new(
        texture_node.outputs["Alpha"], material_principled.inputs["Alpha"]
    )
    material.node_tree.links.new(
        material_principled.outputs["BSDF"], output_node.inputs["Surface"]
    )
    roughness = material_principled.inputs.get("Roughness")
    metallic = material_principled.inputs.get("Metallic")
    if roughness is not None:
        roughness.default_value = 0.48
    if metallic is not None:
        metallic.default_value = 0.0
    for attribute in list(backing.data.color_attributes):
        backing.data.color_attributes.remove(attribute)
    backing.data.update()
    for obj in sources:
        bpy.data.objects.remove(obj, do_unlink=True)
    backing.name = f"EyeAU_CompleteBall_{side}"
    backing.data.name = f"EyeAU_CompleteBallMesh_{side}"
    backing["toporig_complete_eye_ball"] = True
    backing["toporig_surface_projection"] = True
    backing["toporig_projected_vertex_count"] = int(projected_vertices)
    backing["toporig_source_eye_objects"] = json.dumps(sorted(source_names))
    return backing, {
        "method": "landmark_aligned_spherical_texture",
        "connected_spherical_surface": True,
        "source_shells_removed": True,
        "projected_vertex_count": int(projected_vertices),
        "maximum_projection_distance": float(max_surface_distance),
        "single_material": material.name,
        "texture": image.name,
        "texture_size": [texture_width, texture_height],
        "sclera_color": [float(value) for value in sclera_color],
        "appearance_alignment_degrees": float(appearance_alignment_degrees),
        "procedural_iris_radius": float(iris_radius),
        "procedural_iris_angle_degrees": float(math.degrees(iris_angle)),
        "source_objects": sorted(source_names),
    }


def _create_pivot(side: str, center: Any, eye_objects: Sequence[Any]) -> Any:
    import bpy
    from mathutils import Matrix

    pivot = bpy.data.objects.new(f"EyeAU_Pivot_{side}", None)
    bpy.context.scene.collection.objects.link(pivot)
    pivot.empty_display_type = "PLAIN_AXES"
    pivot.empty_display_size = max(
        max((obj.dimensions.length for obj in eye_objects), default=1.0) * 0.25,
        1.0e-4,
    )
    pivot.matrix_world = Matrix.Translation(center)
    pivot["toporig_eye_side"] = side
    pivot["toporig_driver"] = "AU12-AU19"
    for obj in eye_objects:
        world = obj.matrix_world.copy()
        obj.parent = pivot
        obj.matrix_parent_inverse = pivot.matrix_world.inverted()
        obj.matrix_world = world
    pivot.rotation_mode = "QUATERNION"
    return pivot


def _key_eye_animation(
    *,
    pivot: Any,
    side: str,
    center: Any,
    iris_center: Any,
    screen_horizontal: Any,
    screen_vertical: Any,
    keyframes: Sequence[MotionKeyframe],
    fps: float,
    max_angle_degrees: float,
) -> None:
    from mathutils import Quaternion

    forward = iris_center - center
    if forward.length <= 1.0e-8:
        raise ValueError(f"The fitted {side} eye center equals its iris center.")
    forward.normalize()
    horizontal = screen_horizontal - forward * screen_horizontal.dot(forward)
    if horizontal.length <= 1.0e-8:
        raise ValueError(f"The {side} eye horizontal tangent is degenerate.")
    horizontal.normalize()
    vertical = screen_vertical - forward * screen_vertical.dot(forward)
    vertical -= horizontal * vertical.dot(horizontal)
    if vertical.length <= 1.0e-8:
        raise ValueError(f"The {side} eye vertical tangent is degenerate.")
    vertical.normalize()
    if vertical.dot(screen_vertical) < 0.0:
        vertical.negate()

    for keyframe in keyframes:
        motion_x, motion_y = _motion_for_eye(keyframe.aus, side)
        horizontal_angle = math.radians(max_angle_degrees * motion_x)
        downward_angle = math.radians(max_angle_degrees * motion_y)
        target = (
            forward
            + horizontal * math.tan(horizontal_angle)
            - vertical * math.tan(downward_angle)
        )
        target.normalize()
        rotation = forward.rotation_difference(target)
        pivot.rotation_quaternion = Quaternion(rotation)
        frame = 1.0 + keyframe.time * fps
        pivot.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    if pivot.animation_data and pivot.animation_data.action:
        pivot.animation_data.action.name = f"Eye_AU_Motion_{side}"
        for curve in pivot.animation_data.action.fcurves:
            for point in curve.keyframe_points:
                point.interpolation = "LINEAR"


def _export_glb(path: Path, frame_end: int) -> None:
    import bpy

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.exporting.glb")
    if temporary.exists():
        temporary.unlink()
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = max(1, frame_end)
    scene.frame_set(1)
    kwargs: dict[str, Any] = {
        "filepath": str(temporary),
        "export_format": "GLB",
        "export_yup": True,
        "export_animations": True,
        "export_frame_range": True,
        "export_force_sampling": True,
        "export_materials": "EXPORT",
        "export_texcoords": True,
        "export_normals": True,
        "export_extras": True,
    }
    properties = set(bpy.ops.export_scene.gltf.get_rna_type().properties.keys())
    if "export_animation_mode" in properties:
        kwargs["export_animation_mode"] = "SCENE"
    kwargs = {key: value for key, value in kwargs.items() if key in properties}
    result = bpy.ops.export_scene.gltf(**kwargs)
    if "FINISHED" not in result or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Blender did not export {path}.")
    os.replace(temporary, path)


def blender_worker_main(args: argparse.Namespace) -> None:
    import bpy

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    payload = json.loads(args.landmarks.read_text(encoding="utf-8"))
    timeline_payload = json.loads(args.timeline.read_text(encoding="utf-8"))
    fps, keyframes = normalize_timeline_payload(timeline_payload, default_fps=30.0)

    objects = _reset_and_import(input_path)
    indices: dict[str, _ComponentIndex] = {}
    geometry = {
        side: _eye_geometry(payload, side) for side in ("left", "right")
    }
    _validate_landmark_geometry_compatibility(objects, geometry)
    overrides = {
        "left": args.left_eye_object,
        "right": args.right_eye_object,
    }
    candidates = {
        side: _find_eye_candidates(
            objects=objects,
            indices=indices,
            payload=payload,
            side=side,
            geometry=geometry[side],
            object_override=overrides[side],
        )
        for side in ("left", "right")
    }
    assemblies: dict[str, list[_Candidate]] = {}
    for side in ("left", "right"):
        try:
            assemblies[side] = _select_eye_assembly(candidates[side], side)
        except ValueError:
            if args.sclera_backing != "always":
                raise
            outer_cap = _select_outer_eye_cap(
                candidates[side], side, float(geometry[side]["eye_width"])
            )
            if outer_cap is None:
                raise ValueError(
                    f"The {side} eye has no independent ray-confirmed outer cap; "
                    "a procedural ball would be hidden by or tear the face mesh."
                )
            assemblies[side] = [outer_cap]
    complete = {
        side: _eye_assembly_is_complete(assemblies[side])
        for side in ("left", "right")
    }
    for side, opposite in (("left", "right"), ("right", "left")):
        if complete[side] or not complete[opposite]:
            continue
        opposite_pivot = assemblies[opposite][0]
        center_hint = _bilateral_center_hint(
            source_center=opposite_pivot.center,
            payload=payload,
            geometry=geometry,
        )
        recovered = _find_bilateral_eye_candidates(
            objects=objects,
            indices=indices,
            payload=payload,
            side=side,
            geometry=geometry[side],
            center_hint=center_hint,
            radius_hint=opposite_pivot.radius,
            existing=candidates[side],
        )
        if recovered:
            candidates[side].extend(recovered)
            candidates[side].sort(key=lambda candidate: candidate.score)
            try:
                assemblies[side] = _select_eye_assembly(candidates[side], side)
            except ValueError:
                if args.sclera_backing != "always":
                    raise
            complete[side] = _eye_assembly_is_complete(assemblies[side])

    physically_complete = dict(complete)
    inferred_pivots: dict[str, dict[str, Any] | None] = {
        "left": None,
        "right": None,
    }
    if args.sclera_backing != "always":
        for side in ("left", "right"):
            if complete[side]:
                continue
            raise ValueError(
                f"The {side} eye fit selected only a partial component and "
                "bilateral recovery could not find the complete eyeball assembly."
            )

    for side in ("left", "right"):
        if not physically_complete[side]:
            continue
        followers = _find_nearby_eye_followers(
            objects=objects,
            indices=indices,
            geometry=geometry[side],
            candidates=candidates[side],
            assembly=assemblies[side],
        )
        if followers:
            assemblies[side].extend(followers)
            candidates[side].extend(followers)

    if args.sclera_backing == "always":
        outer_caps: dict[str, _Candidate] = {}
        full_ball_ready = {"left": False, "right": False}
        for side in ("left", "right"):
            outer_cap = _select_outer_eye_cap(
                candidates[side], side, float(geometry[side]["eye_width"])
            )
            if outer_cap is None:
                raise ValueError(
                    f"The {side} eye has no independent ray-confirmed outer cap; "
                    "a full procedural ball cannot be placed safely."
                )
            outer_caps[side] = outer_cap
            # Forced reconstruction needs only ray-visible, sphere-like source
            # layers.  Do not carry the legacy assembly wholesale: on some
            # identities it contains a large face/eyelid component that merely
            # touches the fitted eye shell and would tear the socket when
            # deleted.
            assemblies[side] = [outer_cap]
            eye_width = float(geometry[side]["eye_width"])
            existing_components = {id(outer_cap.component)}
            for candidate in _visible_front_eye_layers(
                candidates[side], eye_width
            ):
                if id(candidate.component) in existing_components:
                    continue
                candidate.reason = "accepted as safe compact eye follower"
                assemblies[side].append(candidate)
                existing_components.add(id(candidate.component))
            if _outer_cap_sphere_is_usable(outer_cap, eye_width):
                inferred_pivots[side] = _use_outer_cap_sphere_pivot(
                    outer_cap, eye_width
                )
                full_ball_ready[side] = True

        for side, opposite in (("left", "right"), ("right", "left")):
            if full_ball_ready[side]:
                continue
            opposite_pivot = (
                assemblies[opposite][0] if full_ball_ready[opposite] else None
            )
            inferred_pivots[side] = _infer_full_ball_pivot(
                candidate=outer_caps[side],
                payload=payload,
                geometry=geometry[side],
                opposite_geometry=(
                    geometry[opposite] if opposite_pivot is not None else None
                ),
                opposite_pivot=opposite_pivot,
            )
            full_ball_ready[side] = True

    for side in ("left", "right"):
        best = assemblies[side][0]
        print(
            f"[eye-au] {side}: {best.component.obj.name}, "
            f"radius={best.radius:.6g}, residual={best.residual:.4f}, "
            f"layers={len(assemblies[side])}",
            flush=True,
        )

    texture_reference_pixels = None
    if args.sclera_backing == "always":
        try:
            texture_reference_pixels = _render_original_texture_reference(
                objects, payload
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            print(
                f"[eye-au] warning: visible iris texture render failed; "
                f"using surface UV samples ({exc})",
                flush=True,
            )

    appearance_sources: dict[str, list[Any]] = {"left": [], "right": []}
    if args.sclera_backing == "always":
        for side in ("left", "right"):
            seen_components: set[tuple[str, int]] = set()
            for index, candidate in enumerate(assemblies[side]):
                component = candidate.component
                key = (component.obj.name, int(component.seed_polygon))
                if key in seen_components:
                    continue
                seen_components.add(key)
                appearance_sources[side].append(
                    _copy_selected_polygons(
                        component.obj,
                        set(component.polygon_ids),
                        f"EyeAU_AppearanceLayer_{side}_{index:02d}",
                    )
                )

    removed_eye_components: dict[str, int] | None = None
    if args.sclera_backing == "always":
        removed_eye_components = _remove_replaced_eye_polygons(
            objects, assemblies, geometry, indices
        )
        eye_objects: dict[str, list[Any]] = {"left": [], "right": []}
    else:
        eye_objects = _isolate_eye_objects(objects, assemblies, geometry)
    sclera_backings: dict[str, Any | None] = {"left": None, "right": None}
    surface_projections: dict[str, dict[str, Any] | None] = {
        "left": None,
        "right": None,
    }
    for side in ("left", "right"):
        best = assemblies[side][0]
        should_add_backing = args.sclera_backing == "always" or (
            args.sclera_backing == "auto"
            and _assembly_needs_sclera_backing(
                assemblies[side], geometry[side]["iris_center"]
            )
        )
        if should_add_backing:
            backing = _create_sclera_backing(
                side=side,
                center=best.center,
                pivot_radius=best.radius,
                eye_width=float(geometry[side]["eye_width"]),
                iris_center=geometry[side]["iris_center"],
                cap_surface_aligned=args.sclera_backing == "always",
            )
            eye_objects[side].append(backing)
            sclera_backings[side] = backing
            if args.sclera_backing == "always":
                (
                    complete_ball,
                    surface_projections[side],
                ) = _build_opaque_eye_material_regions(
                    side=side,
                    payload=payload,
                    reference_pixels=texture_reference_pixels,
                    source_objects=appearance_sources[side],
                    backing=backing,
                    pivot_radius=best.radius,
                    eye_width=float(geometry[side]["eye_width"]),
                    iris_center=geometry[side]["iris_center"],
                    iris_radius=float(geometry[side]["iris_radius"]),
                )
                eye_objects[side] = [complete_ball]
    horizontal, vertical = _face_axes(payload, geometry)
    pivots = {}
    for side in ("left", "right"):
        best = assemblies[side][0]
        pivot = _create_pivot(side, best.center, eye_objects[side])
        pivot["toporig_fit_radius"] = float(best.radius)
        # Blender's glTF extras encoder rejects NaN/Infinity.  A planar cap
        # can legitimately have an undefined sphere-fit residual when its
        # pivot was inferred from landmarks, so store a finite sentinel while
        # retaining the explicit inferred flag below.
        pivot["toporig_fit_residual"] = (
            float(best.residual) if math.isfinite(best.residual) else -1.0
        )
        pivot["toporig_pivot_inferred"] = bool(best.pivot_inferred)
        _key_eye_animation(
            pivot=pivot,
            side=side,
            center=best.center,
            iris_center=geometry[side]["iris_center"],
            screen_horizontal=horizontal,
            screen_vertical=vertical,
            keyframes=keyframes,
            fps=fps,
            max_angle_degrees=float(args.max_angle),
        )
        pivots[side] = pivot

    bpy.context.scene.render.fps = max(1, int(round(fps)))
    frame_end = int(math.ceil(1.0 + keyframes[-1].time * fps))
    _export_glb(output_path, frame_end)
    report = {
        "animation_name": "Eye_AU_Motion",
        "frame_start": 1,
        "frame_end": frame_end,
        "eye_detection": {
            side: {
                "eye_width": float(geometry[side]["eye_width"]),
                "iris_center": [
                    float(value) for value in geometry[side]["iris_center"]
                ],
                "selected_components": [
                    _candidate_report(candidate) for candidate in assemblies[side]
                ],
                "isolated_objects": [obj.name for obj in eye_objects[side]],
                "sclera_backing": (
                    {
                        "object": sclera_backings[side].name,
                        "radius": float(
                            sclera_backings[side]["toporig_backing_radius"]
                        ),
                        "generated": True,
                    }
                    if sclera_backings[side] is not None
                    else None
                ),
                "inferred_full_ball_pivot": inferred_pivots[side],
                "removed_source_eye_component_count": (
                    removed_eye_components[side]
                    if removed_eye_components is not None
                    else None
                ),
                "spherical_surface_projection": surface_projections[side],
                "outer_cap_surface_alignment": (
                    {
                        "surface_component_defines_ball": True,
                        "source_geometry_modified": False,
                        **inferred_pivots[side],
                    }
                    if (
                        args.sclera_backing == "always"
                        and inferred_pivots[side] is not None
                    )
                    else None
                ),
                "pivot": pivots[side].name,
                "all_candidates": (
                    [_candidate_report(candidate) for candidate in candidates[side]]
                    if args.debug
                    else None
                ),
            }
            for side in ("left", "right")
        },
    }
    args.worker_report.parent.mkdir(parents=True, exist_ok=True)
    args.worker_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _argv_after_separator() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


if __name__ == "__main__":
    worker_argv = _argv_after_separator()
    try:
        if "--_blender-worker" in worker_argv:
            blender_worker_main(_worker_parser(worker_argv))
        else:
            outer_main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
