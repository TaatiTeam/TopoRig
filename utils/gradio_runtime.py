"""Mesh preparation and inference shared by the Gradio callbacks."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import threading
import uuid

import numpy as np
import torch

import demo
from dataset.image_dataset import _read_glb
from utils.paths import runtime_path


ROOT = Path(__file__).resolve().parents[1]
MESH_FORMATS = {".glb", ".fbx", ".obj"}


def find_blender(value: str | None = None) -> str:
    candidates = [value, os.environ.get("TOPORIG_BLENDER"), shutil.which("blender")]
    candidates += [str(p) for p in sorted((ROOT / "third_party").glob("blender-*/blender"))]
    for candidate in candidates:
        if candidate:
            path = Path(candidate).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path.resolve())
    raise FileNotFoundError("Blender was not found. Follow the Blender installation in README.md "
                            "or pass --blender to gradio_app.py.")


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(command: list[str], *, timeout: int = 600) -> None:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Mesh processing failed:\n{(result.stdout + result.stderr)[-4000:]}")


def convert_mesh(source: Path, output: Path, blender: str, *, frame: int | None = None) -> None:
    temporary = output.with_name(f".{uuid.uuid4().hex}.glb")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [blender, "--background", "--factory-startup", "--threads", "2",
               "--python-exit-code", "1", "--python", str(ROOT / "scripts/prepare_demo_mesh.py"),
               "--", "--input", str(source), "--output", str(temporary)]
    if frame is not None:
        command += ["--frame", str(frame)]
    try:
        run_command(command)
        if not temporary.is_file():
            raise RuntimeError("Blender did not produce a GLB file.")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def write_deformed_glb(source: Path, output: Path, vertices: torch.Tensor) -> None:
    """Replace positions and normals while retaining the embedded materials/UVs."""
    document, original_binary = _read_glb(source)
    binary = bytearray(original_binary)
    vertices = vertices.detach().cpu().float()
    if not torch.isfinite(vertices).all():
        raise ValueError("The prediction contains non-finite coordinates.")
    source_mesh = demo.load_mesh_for_model(source)
    if vertices.shape != source_mesh["vertices"].shape:
        raise ValueError("Prediction and source vertex counts differ.")
    normals = demo.recompute_vertex_normals(vertices, source_mesh["faces"])

    def append_attribute(values: torch.Tensor, *, bounds: bool) -> int:
        while len(binary) % 4:
            binary.append(0)
        data = np.asarray(values.numpy(), dtype="<f4").tobytes()
        view = len(document["bufferViews"])
        document["bufferViews"].append({"buffer": 0, "byteOffset": len(binary),
                                        "byteLength": len(data), "target": 34962})
        binary.extend(data)
        accessor = {"bufferView": view, "componentType": 5126,
                    "count": len(values), "type": "VEC3"}
        if bounds:
            accessor.update(min=values.amin(0).tolist(), max=values.amax(0).tolist())
        index = len(document["accessors"])
        document["accessors"].append(accessor)
        return index

    offset = 0
    for mesh in document["meshes"]:
        for primitive in mesh["primitives"]:
            attributes = primitive["attributes"]
            count = document["accessors"][attributes["POSITION"]]["count"]
            attributes["POSITION"] = append_attribute(vertices[offset:offset + count], bounds=True)
            attributes["NORMAL"] = append_attribute(normals[offset:offset + count], bounds=False)
            offset += count
    if offset != len(vertices):
        raise ValueError("GLB primitive counts differ from the prediction.")
    document["buffers"][0]["byteLength"] = len(binary)
    payload = json.dumps(document, separators=(",", ":")).encode()
    payload += b" " * (-len(payload) % 4)
    binary += b"\x00" * (-len(binary) % 4)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, 28 + len(payload) + len(binary)))
        handle.write(struct.pack("<I4s", len(payload), b"JSON"))
        handle.write(payload)
        handle.write(struct.pack("<I4s", len(binary), b"BIN\x00"))
        handle.write(binary)


@dataclass
class PreparedMesh:
    source: Path
    original: dict
    model_mesh: dict
    vertex_map: torch.Tensor
    model_to_source: torch.Tensor
    model_offset: torch.Tensor
    landmarks: Path


class RigRuntime:
    def __init__(self, checkpoint: Path | None = None, device: str = "auto",
                 blender: str | None = None, cache_root: Path | None = None):
        self.checkpoint = checkpoint
        self.device_name = device
        self.blender_option = blender
        self.cache = (cache_root or runtime_path("cache", "gradio")).resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._model = None
        self._reference = None
        self._prepared: OrderedDict[str, PreparedMesh] = OrderedDict()
        self._mouth: OrderedDict[str, object] = OrderedDict()
        self._predictions: OrderedDict[tuple, torch.Tensor] = OrderedDict()

    @property
    def blender(self) -> str:
        return find_blender(self.blender_option)

    def prepare_upload(self, source: Path) -> Path:
        source = source.expanduser().resolve()
        if not source.is_file() or source.suffix.lower() not in MESH_FORMATS:
            raise ValueError("Choose a GLB, FBX or OBJ mesh file.")
        target = self.cache / "uploads_v2" / fingerprint(source) / "neutral.glb"
        with self.lock:
            if not target.is_file():
                pending = target.with_name(f".{uuid.uuid4().hex}.glb")
                try:
                    convert_mesh(source, pending, self.blender)
                    self.validate_mesh(pending)
                    self.orient_for_viewer(pending)
                    pending.replace(target)
                finally:
                    pending.unlink(missing_ok=True)
        return target

    def sample_thumbnail(self, source: Path) -> Path:
        target = self.cache / "thumbnails" / f"{fingerprint(source)}.png"
        with self.lock:
            if not target.is_file():
                prepared = self.prepare_upload(source)
                temporary = target.with_name(f".{uuid.uuid4().hex}.png")
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    run_command([self.blender, "--background", "--factory-startup", "--threads", "2",
                                 "--python-exit-code", "1", "--python", str(ROOT / "scripts/prepare_demo_mesh.py"),
                                 "--", "--input", str(prepared), "--output", str(temporary), "--thumbnail"])
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
        return target

    @staticmethod
    def validate_mesh(path: Path) -> dict:
        mesh = demo.load_mesh_for_model(path)
        vertices, faces = mesh["vertices"], mesh["faces"]
        if len(vertices) < 3 or not len(faces) or not torch.isfinite(vertices).all():
            raise ValueError("The file must contain a finite triangle mesh.")
        if len(vertices) > 2_000_000:
            raise ValueError("Please simplify the mesh to fewer than two million vertices.")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("The mesh contains invalid triangle indices.")
        return mesh

    def load_model(self) -> None:
        if self._model is None:
            checkpoint = demo.resolve_checkpoint_path(self.checkpoint)
            payload = demo.load_checkpoint(checkpoint)
            self.config = demo.checkpoint_config(payload, None)
            self.device = demo.resolve_device(self.device_name, self.config)
            self._model = demo.load_model(payload, self.config, self.device)

    def map_landmarks(self, source: Path) -> Path:
        target = self.cache / "landmarks_v2" / f"{fingerprint(source)}.json"
        if target.is_file():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{uuid.uuid4().hex}.json")
        mesh = demo.load_mesh_for_model(source)
        document, _ = _read_glb(source)
        from scipy.spatial import cKDTree
        tree = cKDTree(mesh["vertices"].numpy())
        material_modes = (True, False) if document.get("images") else (False,)
        last_error = None
        try:
            for roll in (0, 180):
                for preserve_materials in material_modes:
                    command = [sys.executable, str(demo.DEFAULT_MEDIAPIPE_MAPPER),
                               "--mesh", str(source), "--output", str(temporary),
                               "--blender", self.blender, "--blender-threads", "2",
                               "--views", "neg_y,pos_y,neg_z,pos_z,neg_x,pos_x",
                               "--roll-offset", str(roll), "--output-surface-anchors",
                               "--surface-hits-only", "--no-debug-glb"]
                    if preserve_materials:
                        command.append("--preserve-materials")
                    try:
                        run_command(command)
                    except RuntimeError as exc:
                        last_error = exc
                        continue
                    payload = json.loads(temporary.read_text())
                    # Blender can reorder glTF vertices. Match world-space hits
                    # back to GLB accessors, using the standard Y-up conversion.
                    mapping = {}
                    for key, anchor in payload["surface_anchors"].items():
                        if not isinstance(anchor, dict) or not isinstance(anchor.get("position"), dict):
                            continue
                        point = anchor["position"]
                        _, index = tree.query([point["x"], point["z"], -point["y"]])
                        mapping[int(key)] = int(index)
                    if len(mapping) >= 32 and {10, 152, 33, 263}.issubset(mapping):
                        temporary.write_text(json.dumps(mapping))
                        temporary.replace(target)
                        return target
            raise ValueError("Could not locate enough facial landmarks. Use a clearly visible neutral face.") from last_error
        finally:
            temporary.unlink(missing_ok=True)

    def orient_for_viewer(self, source: Path) -> None:
        """Frame the face upright, looking toward +Z, in the input/output viewers."""
        mapping = demo.load_landmark_mapping(self.map_landmarks(source))
        vertices = self.validate_mesh(source)["vertices"]
        required = {10, 152, 33, 263}
        if not required.issubset(mapping):
            raise ValueError("Could not identify the eyes, forehead and chin for face alignment.")
        right = vertices[mapping[263]] - vertices[mapping[33]]
        up = vertices[mapping[10]] - vertices[mapping[152]]
        right = torch.nn.functional.normalize(right, dim=0)
        up = torch.nn.functional.normalize(up - torch.dot(up, right) * right, dim=0)
        front = torch.linalg.cross(right, up)
        if front.norm() < 0.9:
            raise ValueError("Facial landmarks do not define a valid face orientation.")
        rotation = torch.stack((right, up, front), dim=1)
        center = (vertices.amin(0) + vertices.amax(0)) * 0.5
        aligned = (vertices - center) @ rotation
        target = source.with_name(f".{uuid.uuid4().hex}.glb")
        try:
            write_deformed_glb(source, target, aligned)
            target.replace(source)
        finally:
            target.unlink(missing_ok=True)

    def reference_mesh(self) -> tuple[dict, dict]:
        if self._reference is None:
            reference = demo.metahuman_sample_mesh(runtime_path("reference", "metahuman"))
            converted = self.cache / "reference" / fingerprint(reference) / "neutral.glb"
            if not converted.is_file():
                convert_mesh(reference, converted, self.blender)
            mesh = self.validate_mesh(converted)
            # Undo the FBX preview rotation and GLB's Y-up conversion to recover
            # the same reference coordinate frame as the command-line demo.
            for name in ("vertices", "normals"):
                values = mesh[name]
                mesh[name] = torch.stack((-values[:, 0], values[:, 2], values[:, 1]), dim=-1)
            mapping = demo.load_landmark_mapping(self.map_landmarks(converted))
            self._reference = mesh, mapping
        return self._reference

    def prepare_model_mesh(self, source: Path) -> PreparedMesh:
        key = fingerprint(source)
        if key in self._prepared:
            self._prepared.move_to_end(key)
            return self._prepared[key]
        self.load_model()
        raw = self.validate_mesh(source)
        landmark_path = self.map_landmarks(source)
        mapping = demo.load_landmark_mapping(landmark_path)
        reference, reference_mapping = self.reference_mesh()
        oriented = dict(raw)
        for name in ("vertices", "normals"):
            values = raw[name]
            oriented[name] = torch.stack((values[:, 0], -values[:, 2], values[:, 1]), dim=-1)
        original = demo.preprocess_mesh_for_model(
            oriented, self.config, input_convention="fbx", input_landmarks=mapping,
            reference_mesh=reference, reference_landmarks=reference_mapping,
        )
        tolerance, snap_ids = demo.inference_weld_settings(self.config)
        welded, vertex_map = demo.weld_mesh_for_inference(
            original, tolerance=tolerance, snap_mediapipe_ids=snap_ids,
        )
        # Alignment is a similarity transform. Recover its linear part so only
        # displacement changes; exported units, pose and materials stay intact.
        indices = torch.linspace(0, len(raw["vertices"]) - 1, min(1024, len(raw["vertices"]))).long()
        points = raw["vertices"][indices].double()
        basis = torch.cat((points, torch.ones((len(points), 1), dtype=torch.float64)), dim=1)
        fitted = torch.linalg.lstsq(basis, original["vertices"][indices].double()).solution
        if not torch.allclose(basis @ fitted, original["vertices"][indices].double(), atol=1e-5):
            raise ValueError("Unable to preserve the mesh coordinate frame during alignment.")
        inverse = torch.linalg.inv(fitted[:3]).float()
        offset = raw["vertices"].mean(0) - original["vertices"].mean(0) @ inverse
        prepared = PreparedMesh(source, raw, welded, vertex_map, inverse, offset, landmark_path)
        self._prepared[key] = prepared
        while len(self._prepared) > 4:
            self._prepared.popitem(last=False)
        return prepared

    @staticmethod
    def replacement_eye_mask(source: Path) -> torch.Tensor:
        document, _ = _read_glb(source)
        masks = []
        for mesh in document["meshes"]:
            for primitive in mesh["primitives"]:
                material = document.get("materials", [])[primitive["material"]] if "material" in primitive else {}
                is_eye = material.get("name", "").startswith("EyeAU_")
                count = document["accessors"][primitive["attributes"]["POSITION"]]["count"]
                masks.append(torch.full((count,), is_eye, dtype=torch.bool))
        return torch.cat(masks)

    def export_eyes(self, source: Path, output: Path, action_unit: int, intensity: float) -> None:
        animated = output.with_name("gaze_animation.glb")
        report_path = output.with_suffix(".eye_animation.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            command = demo.complete_eye_animation_command(
                source_glb=source, output_path=animated, action_unit_id=action_unit,
                report_path=report_path,
                intensity=intensity, duration=1.0, fps=24.0, max_angle_degrees=25.0,
                landmarks=None, mapper_path=demo.DEFAULT_MEDIAPIPE_MAPPER,
                landmark_cache_dir=self.cache / "gaze", blender=self.blender,
                blender_threads=2, sclera_backing="always", refresh_landmarks=False, debug=False,
            )
            run_command(command + ["--surface-hits-only"])
            report = json.loads(report_path.read_text())
            if report.get("status") != "exported":
                raise RuntimeError("Eye replacement did not export a valid mesh.")
            convert_mesh(animated, output, self.blender, frame=int(report["frame_end"]))
        except RuntimeError as exc:
            raise ValueError("Eyes postprocessing could not fit the eyeballs. Use a textured face "
                             "with clearly visible eyes, or turn off Eyes postprocessing.") from exc
        finally:
            animated.unlink(missing_ok=True)

    def prepare_eyes(self, source: Path) -> Path:
        target = self.cache / "postprocessing" / "eyes" / fingerprint(source) / "neutral.glb"
        if not target.is_file():
            self.export_eyes(source, target, 14, 0.0)
        return target

    def predict_mouth(self, source: Path, action_unit: int, intensity: float, output: Path) -> None:
        from utils.gradio_mouth import prepare_mouth, predict_mouth
        asset = runtime_path("reference", "ict/ict_oral_assembly.npz")
        if not asset.is_file():
            raise FileNotFoundError("Mouth postprocessing needs ict_oral_assembly.npz. "
                                    "Download the demo assets as described in README.md.")
        prepared = self.prepare_model_mesh(source)
        key = fingerprint(source)
        if key not in self._mouth:
            self._mouth[key] = prepare_mouth(self, prepared)
            while len(self._mouth) > 4:
                self._mouth.popitem(last=False)
        self._mouth.move_to_end(key)
        predict_mouth(self, prepared, self._mouth[key], action_unit, intensity, output, asset)

    def predict(self, source: Path, action_unit: int, intensity: float, *,
                mouth_postprocessing: bool = False, eyes_postprocessing: bool = False) -> Path:
        source = source.resolve()
        if action_unit not in demo.AU_NAME:
            raise ValueError("Choose an action unit from the list.")
        if not math.isfinite(intensity) or not 0 <= intensity <= 1:
            raise ValueError("Intensity must be between 0 and 1.")
        is_gaze = action_unit in demo.complete_eye_au.AU_SPECS
        if is_gaze and not eyes_postprocessing:
            raise ValueError("Enable Eyes postprocessing to use eye-gaze action units.")
        output = self.cache / "results" / uuid.uuid4().hex / f"{demo.AU_NAME[action_unit]}_{intensity:.2f}.glb"
        output.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, torch.inference_mode():
            if is_gaze:
                if mouth_postprocessing:
                    neutral = output.with_name("mouth_neutral.glb")
                    self.predict_mouth(source, 26, 0.0, neutral)
                    source = neutral
                self.export_eyes(source, output, action_unit, intensity)
                return output
            if eyes_postprocessing:
                source = self.prepare_eyes(source)
            if mouth_postprocessing:
                self.predict_mouth(source, action_unit, intensity, output)
                return output
            if intensity == 0:
                shutil.copyfile(source, output)
                return output
            prepared = self.prepare_model_mesh(source)
            cache_key = (fingerprint(source), action_unit)
            if cache_key not in self._predictions:
                _, delta = demo.run_model(self._model, prepared.model_mesh,
                                          action_unit, self.config, self.device)
                self._predictions[cache_key] = (
                    demo.map_welded_values_to_original(delta, prepared.vertex_map)
                    @ prepared.model_to_source
                )
                self._predictions[cache_key][self.replacement_eye_mask(source)] = 0
                while len(self._predictions) > 8:
                    self._predictions.popitem(last=False)
            self._predictions.move_to_end(cache_key)
            vertices = prepared.original["vertices"] + intensity * self._predictions[cache_key]
            write_deformed_glb(source, output, vertices)
        return output
