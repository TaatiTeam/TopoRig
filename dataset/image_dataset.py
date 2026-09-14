from __future__ import annotations

import csv
import json
import re
import struct
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from dataset.mesh_dataset import (
    DEFAULT_LANDMARK_MAPPER,
    _empty_landmark_payload,
    _load_landmark_payload_or_regenerate,
    _publish_mediapipe_mapping,
    _temporary_mediapipe_mapping_path,
    _transform_model_sample,
    _unlink_if_exists,
)
from utils.displacement_2d import compute_displacement_2d
from utils.eye_gaze_surface_anchors import load_eye_gaze_surface_anchor_payload
from utils.welded_eyelid_landmarks import VALIDATED_WELDED_LANDMARK_VERSION
from utils.fbx_to_tensor import AU_NAME, fbx_to_tensor
from dataset.hf_assets import (
    NEUTRAL_IMAGE_RE, EXPRESSED_IMAGE_RE, canonical_identity,
    person_id as normalize_person_id, prepare_assets, asset_cache_namespace, versioned_cache_dir,
)


PathLike = Union[str, Path]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEUTRAL_DIR = ROOT / "data" / "neutral_imgs"
DEFAULT_EXPRESSED_DIR = ROOT / "data" / "expressed_imgs"
DEFAULT_NEUTRAL_MESH_DIR = (
    ROOT / "data" / "meshes_custom"
)
DEFAULT_CACHE_DIR = ROOT / ".cache" / "image_dataset"

VALID_SPLITS = {"train", "val", "test", "all"}

_COMPONENT_DTYPES = {
    5120: torch.int8,
    5121: torch.uint8,
    5122: torch.int16,
    5123: torch.uint16,
    5125: torch.uint32,
    5126: torch.float32,
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


@dataclass(frozen=True)
class ImageSample:
    person_id: str
    action_unit_id: int
    neutral_image_path: Path
    expressed_image_path: Path
    neutral_mesh_path: Path


class ImageDataset(Dataset):
    def __init__(
        self,
        neutral_dir: PathLike = DEFAULT_NEUTRAL_DIR,
        expressed_dir: PathLike = DEFAULT_EXPRESSED_DIR,
        neutral_mesh_dir: PathLike = DEFAULT_NEUTRAL_MESH_DIR,
        neutral_mesh_path: Optional[PathLike] = None,
        neutral_mesh_filename_template: str = "mesh_{person_id}.glb",
        neutral_mesh_filename_fallback_template: Optional[str] = None,
        cache_dir: PathLike = DEFAULT_CACHE_DIR,
        neutral_mesh_tensor_dir: Optional[PathLike] = None,
        split: str = "all",
        split_csv: Optional[PathLike] = None,
        action_units: Optional[Sequence[Union[int, str]]] = None,
        person_ids: Optional[Sequence[Union[int, str]]] = None,
        exclude_person_ids: Optional[Sequence[Union[int, str]]] = None,
        exclude_person_ids_file: Optional[PathLike] = None,
        use_mediapipe_landmarks: bool = False,
        generate_mediapipe_landmarks: bool = False,
        require_mediapipe_landmarks: bool = False,
        min_mediapipe_landmarks: int = 1,
        mediapipe_landmark_dir: Optional[PathLike] = None,
        mediapipe_landmark_filename_template: str = "mesh_{person_id}.json",
        mediapipe_landmark_filename_fallback_template: Optional[str] = None,
        displacement_2d_cache_device: str = "cpu",
        displacement_2d_cache_batch_size: int = 1,
        displacement_2d_cache_image_size: Optional[Sequence[int]] = None,
        mediapipe_mapper_path: PathLike = DEFAULT_LANDMARK_MAPPER,
        mediapipe_mapper_args: Optional[Sequence[str]] = None,
        mesh_up_axis: str = "z",
        mesh_front_axis: str = "-y",
        normalize_on_get: bool = True,
        normalized_extent: float = 2.0,
        neutral_mesh_cache_size: Optional[int] = None,
        displacement_cache_size: Optional[int] = None,
        max_neutral_mesh_vertices: Optional[int] = None,
        max_neutral_mesh_faces: Optional[int] = None,
        weld_coincident_vertices: bool = False,
        weld_tolerance: float = 1.0e-6,
        welded_neutral_mesh_tensor_dir: Optional[PathLike] = None,
        weld_landmark_ids_to_largest_component: Optional[Sequence[int]] = None,
        welded_landmark_validation: Optional[Mapping[str, Any]] = None,
        use_gaze_surface_anchors: bool = False,
        require_gaze_surface_anchors: bool = False,
        gaze_surface_anchor_dir: Optional[PathLike] = None,
        gaze_surface_anchor_filename_template: str = "mesh_{person_id}.pt",
        gaze_surface_anchor_filename_fallback_template: str = (
            "mesh_{person_id}_fallback.pt"
        ),
        data_root: Optional[PathLike] = None,
        asset_cache_dir: Optional[PathLike] = None,
        identity_profile: str = "public",
        precompute_displacement_2d: bool = True,
    ) -> None:
        self.data_root = Path(data_root).expanduser() if data_root is not None else None
        if identity_profile not in {"public", "historical"}:
            raise ValueError("identity_profile must be public or historical")
        self.identity_profile = identity_profile
        self.precompute_displacement_2d = bool(precompute_displacement_2d)
        if self.data_root is not None:
            namespace = asset_cache_namespace(self.data_root)
            source = prepare_assets(
                self.data_root,
                asset_cache_dir or Path(cache_dir).expanduser().parent / "hf_assets",
                ("neutral_imgs", "expressed_imgs", "meshes_custom"),
                identities=_normalize_person_id_filter(person_ids),
                action_units=_normalize_action_unit_ids(action_units),
            )
            cache_dir = versioned_cache_dir(cache_dir, self.data_root, "image")
            neutral_dir, expressed_dir = source / "neutral_imgs", source / "expressed_imgs"
            neutral_mesh_dir = source / "meshes_custom"
            neutral_mesh_filename_template = "{identity}.fbx"
            neutral_mesh_filename_fallback_template = None
            if generate_mediapipe_landmarks and mediapipe_landmark_dir is None:
                raise ValueError("With data_root, generate landmarks into an explicit writable mediapipe_landmark_dir")
            mediapipe_landmark_dir = mediapipe_landmark_dir or neutral_mesh_dir
            mediapipe_landmark_filename_template = "{identity}.json"
            mediapipe_landmark_filename_fallback_template = None
            for name, value in (("neutral_mesh_tensor_dir", neutral_mesh_tensor_dir),
                                ("welded_neutral_mesh_tensor_dir", welded_neutral_mesh_tensor_dir)):
                if value is not None:
                    raise ValueError(f"With data_root, use cache_dir instead of {name}; caches are versioned automatically")
        self.neutral_dir = Path(neutral_dir).expanduser()
        self.expressed_dir = Path(expressed_dir).expanduser()
        from dataset.layout import assert_dataset_ready
        assert_dataset_ready(self.neutral_dir)
        assert_dataset_ready(self.expressed_dir)
        self.neutral_mesh_dir = Path(neutral_mesh_dir).expanduser()
        self.neutral_mesh_path = (
            Path(neutral_mesh_path).expanduser()
            if neutral_mesh_path is not None
            else None
        )
        self.neutral_mesh_filename_template = str(neutral_mesh_filename_template)
        self.neutral_mesh_filename_fallback_template = (
            str(neutral_mesh_filename_fallback_template)
            if neutral_mesh_filename_fallback_template not in (None, "")
            else None
        )
        self.cache_dir = Path(cache_dir).expanduser()
        self.neutral_mesh_tensor_dir = (
            Path(neutral_mesh_tensor_dir).expanduser()
            if neutral_mesh_tensor_dir is not None
            else self.cache_dir / "neutral_mesh_tensors"
        )
        self.displacement_2d_dir = self.cache_dir / "displacement_2d"
        self.split = split.lower()
        self.split_csv = Path(split_csv).expanduser() if split_csv is not None else None
        self.action_unit_ids = _normalize_action_unit_ids(action_units)
        self.person_id_filter = _normalize_person_id_filter(person_ids)
        self.exclude_person_ids_file = (
            Path(exclude_person_ids_file).expanduser()
            if exclude_person_ids_file is not None
            else None
        )
        excluded_person_ids = _normalize_person_id_filter(exclude_person_ids) or set()
        if self.exclude_person_ids_file is not None:
            excluded_person_ids.update(
                _load_person_id_filter_file(self.exclude_person_ids_file)
            )
        self.excluded_person_ids = excluded_person_ids or None
        self.use_mediapipe_landmarks = bool(use_mediapipe_landmarks)
        self.generate_mediapipe_landmarks = bool(generate_mediapipe_landmarks)
        self.require_mediapipe_landmarks = bool(require_mediapipe_landmarks)
        self.min_mediapipe_landmarks = int(min_mediapipe_landmarks)
        self.displacement_2d_cache_device = str(displacement_2d_cache_device)
        self.displacement_2d_cache_batch_size = int(displacement_2d_cache_batch_size)
        self.displacement_2d_cache_image_size = _normalize_optional_image_size(
            displacement_2d_cache_image_size,
            "displacement_2d_cache_image_size",
        )
        self.mediapipe_landmark_dir = (
            Path(mediapipe_landmark_dir).expanduser()
            if mediapipe_landmark_dir is not None
            else self.cache_dir / "mediapipe_landmarks"
        )
        self.mediapipe_landmark_filename_template = str(
            mediapipe_landmark_filename_template
        )
        self.mediapipe_landmark_filename_fallback_template = (
            str(mediapipe_landmark_filename_fallback_template)
            if mediapipe_landmark_filename_fallback_template not in (None, "")
            else None
        )
        self.mediapipe_mapper_path = Path(mediapipe_mapper_path).expanduser()
        self.mediapipe_mapper_args = tuple(
            str(arg) for arg in (mediapipe_mapper_args or ())
        )
        self.mesh_up_axis = str(mesh_up_axis)
        self.mesh_front_axis = str(mesh_front_axis)
        self.normalize_on_get = bool(normalize_on_get)
        self.normalized_extent = float(normalized_extent)
        self.neutral_mesh_cache_size = _normalize_memory_cache_size(
            neutral_mesh_cache_size,
            "neutral_mesh_cache_size",
        )
        self.displacement_cache_size = _normalize_memory_cache_size(
            displacement_cache_size,
            "displacement_cache_size",
        )
        self.max_neutral_mesh_vertices = _normalize_optional_positive_int(
            max_neutral_mesh_vertices,
            "max_neutral_mesh_vertices",
        )
        self.max_neutral_mesh_faces = _normalize_optional_positive_int(
            max_neutral_mesh_faces,
            "max_neutral_mesh_faces",
        )
        self.weld_coincident_vertices = bool(weld_coincident_vertices)
        self.weld_tolerance = float(weld_tolerance)
        self.welded_neutral_mesh_tensor_dir = (
            Path(welded_neutral_mesh_tensor_dir).expanduser()
            if welded_neutral_mesh_tensor_dir is not None
            else self.neutral_mesh_tensor_dir / "welded"
        )
        self.weld_landmark_ids_to_largest_component = {
            int(value)
            for value in (weld_landmark_ids_to_largest_component or ())
        }
        self.welded_landmark_validation = dict(welded_landmark_validation or {})
        self.use_gaze_surface_anchors = bool(use_gaze_surface_anchors)
        self.require_gaze_surface_anchors = bool(require_gaze_surface_anchors)
        self.gaze_surface_anchor_dir = (
            Path(gaze_surface_anchor_dir).expanduser()
            if gaze_surface_anchor_dir is not None
            else self.cache_dir / "eye_gaze_surface_anchors"
        )
        self.gaze_surface_anchor_filename_template = str(
            gaze_surface_anchor_filename_template
        )
        self.gaze_surface_anchor_filename_fallback_template = str(
            gaze_surface_anchor_filename_fallback_template
        )
        self.validated_welded_landmark_dir = Path(
            str(
                self.welded_landmark_validation.get(
                    "cache_dir",
                    self.welded_neutral_mesh_tensor_dir.parent
                    / "validated_eyelid_landmarks_v1",
                )
            )
        ).expanduser()
        self._mesh_cache: OrderedDict[Path, Dict[str, torch.Tensor]] = OrderedDict()
        self._displacement_cache: OrderedDict[
            Path,
            Dict[str, torch.Tensor],
        ] = OrderedDict()

        if self.split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}.")
        if self.split != "all" and self.split_csv is None:
            raise ValueError("split_csv is required unless split='all'.")
        if self.displacement_2d_cache_batch_size < 1:
            raise ValueError("displacement_2d_cache_batch_size must be at least 1.")
        if self.min_mediapipe_landmarks < 1:
            raise ValueError("min_mediapipe_landmarks must be at least 1.")
        if self.weld_tolerance <= 0.0:
            raise ValueError("weld_tolerance must be positive.")
        if self.use_gaze_surface_anchors and not self.weld_coincident_vertices:
            raise ValueError(
                "use_gaze_surface_anchors requires weld_coincident_vertices=true."
            )

        self.neutral_mesh_tensor_dir.mkdir(parents=True, exist_ok=True)
        if self.weld_coincident_vertices:
            self.welded_neutral_mesh_tensor_dir.mkdir(parents=True, exist_ok=True)
        if self._validated_welded_landmarks_enabled():
            self.validated_welded_landmark_dir.mkdir(parents=True, exist_ok=True)
        self.displacement_2d_dir.mkdir(parents=True, exist_ok=True)
        if self.use_mediapipe_landmarks:
            self.mediapipe_landmark_dir.mkdir(parents=True, exist_ok=True)
        if self.use_gaze_surface_anchors:
            self.gaze_surface_anchor_dir.mkdir(parents=True, exist_ok=True)

        neutral_images = self._list_neutral_images()
        if self.identity_profile == "historical":
            neutral_images = {key: path for key, path in neutral_images.items() if not key.startswith("v2_")}
        if self.person_id_filter is not None:
            neutral_images = {
                person_id: path
                for person_id, path in neutral_images.items()
                if person_id in self.person_id_filter
            }
        if self.excluded_person_ids is not None:
            neutral_images = {
                person_id: path
                for person_id, path in neutral_images.items()
                if person_id not in self.excluded_person_ids
            }
        if self.split != "all":
            allowed_person_ids = self._load_split_person_ids()
            neutral_images = {
                person_id: path
                for person_id, path in neutral_images.items()
                if person_id in allowed_person_ids
            }

        expressed_images = self._list_expressed_images(set(neutral_images))
        neutral_images = {
            person_id: path
            for person_id, path in neutral_images.items()
            if expressed_images.get(person_id)
        }
        self.person_ids = sorted(neutral_images)
        self._validate_neutral_meshes(self.person_ids)
        neutral_images = self._filter_neutral_images_by_mesh_size(neutral_images)
        neutral_images = self._filter_neutral_images_by_validated_landmarks(
            neutral_images
        )
        self.person_ids = sorted(neutral_images)
        expressed_images = {
            person_id: expressed_images[person_id]
            for person_id in self.person_ids
        }
        self.samples = self._build_samples(neutral_images, expressed_images)
        self._cache_neutral_mesh_tensors()
        if self.precompute_displacement_2d:
            self._cache_displacement_2d()

        print(
            f"[INFO] Found {len(self.person_ids)} face identities "
            f"for split {self.split}."
        )
        print(
            f"[INFO] Found {len(self.samples)} expressed images "
            f"for split {self.split}."
        )

    def _filter_neutral_images_by_mesh_size(
        self,
        neutral_images: Dict[str, Path],
    ) -> Dict[str, Path]:
        if (
            self.max_neutral_mesh_vertices is None
            and self.max_neutral_mesh_faces is None
        ):
            return neutral_images

        kept: Dict[str, Path] = {}
        dropped: list[tuple[str, int, int]] = []
        for person_id, image_path in sorted(neutral_images.items()):
            vertex_count, face_count = self._neutral_mesh_size(person_id)
            too_many_vertices = (
                self.max_neutral_mesh_vertices is not None
                and vertex_count > self.max_neutral_mesh_vertices
            )
            too_many_faces = (
                self.max_neutral_mesh_faces is not None
                and face_count > self.max_neutral_mesh_faces
            )
            if too_many_vertices or too_many_faces:
                dropped.append((person_id, vertex_count, face_count))
                continue
            kept[person_id] = image_path

        if dropped:
            shown = ", ".join(
                f"{person_id}(v={vertex_count},f={face_count})"
                for person_id, vertex_count, face_count in dropped[:10]
            )
            suffix = "..." if len(dropped) > 10 else ""
            print(
                "[INFO] Filtered "
                f"{len(dropped)} neutral image mesh(es) above size limits "
                f"(max_vertices={self.max_neutral_mesh_vertices}, "
                f"max_faces={self.max_neutral_mesh_faces}): "
                f"{shown}{suffix}"
            )
        if neutral_images and not kept:
            raise ValueError(
                "max_neutral_mesh_vertices/max_neutral_mesh_faces filtered every "
                "neutral image mesh."
            )
        return kept

    def _filter_neutral_images_by_validated_landmarks(
        self,
        neutral_images: Dict[str, Path],
    ) -> Dict[str, Path]:
        if not self._validated_welded_landmarks_enabled():
            return neutral_images
        if not self.weld_coincident_vertices:
            raise ValueError(
                "welded_landmark_validation requires weld_coincident_vertices=true."
            )

        require_cache = bool(
            self.welded_landmark_validation.get("require_cache", True)
        )
        skip_invalid = bool(
            self.welded_landmark_validation.get("skip_invalid", True)
        )
        kept: Dict[str, Path] = {}
        skipped: list[tuple[str, str]] = []
        missing: list[tuple[str, Path]] = []
        for person_id, image_path in sorted(neutral_images.items()):
            cache_path = self._validated_welded_landmark_path(person_id)
            if not cache_path.is_file():
                if require_cache:
                    missing.append((person_id, cache_path))
                    continue
                kept[person_id] = image_path
                continue
            with cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected {cache_path} to contain a validation mapping."
                )
            version = int(payload.get("version", -1))
            if version != VALIDATED_WELDED_LANDMARK_VERSION:
                raise ValueError(
                    f"Validated landmark cache {cache_path} has version {version}; "
                    f"expected {VALIDATED_WELDED_LANDMARK_VERSION}."
                )
            status = str(payload.get("status", "invalid"))
            if status in {"valid", "repaired"}:
                kept[person_id] = image_path
                continue
            reasons = payload.get("reasons", ())
            reason = ",".join(str(value) for value in reasons) or "invalid"
            if skip_invalid:
                skipped.append((person_id, reason))
                continue
            raise ValueError(
                f"Validated landmark cache {cache_path} is invalid: {reason}"
            )

        if missing:
            shown = ", ".join(
                f"{person_id} ({path})" for person_id, path in missing[:10]
            )
            suffix = "..." if len(missing) > 10 else ""
            raise FileNotFoundError(
                f"Missing {len(missing)} required validated welded landmark "
                f"cache(s): {shown}{suffix}"
            )
        if skipped:
            shown = ", ".join(
                f"{person_id}({reason})" for person_id, reason in skipped[:10]
            )
            suffix = "..." if len(skipped) > 10 else ""
            print(
                "[INFO] Skipped "
                f"{len(skipped)} image identity/identities with invalid welded "
                f"eyelid mappings: {shown}{suffix}"
            )
        if neutral_images and not kept:
            raise ValueError("Validated eyelid mappings filtered every image identity.")
        return kept

    def _neutral_mesh_size(self, person_id: str) -> tuple[int, int]:
        cache_path = self._neutral_mesh_tensor_path(person_id)
        if cache_path.is_file():
            neutral_mesh = _load_tensor_dict(cache_path)
        else:
            neutral_mesh = _load_neutral_mesh_for_model(
                self._neutral_mesh_path(person_id),
                mesh_up_axis=self.mesh_up_axis,
                mesh_front_axis=self.mesh_front_axis,
                normalize_on_get=self.normalize_on_get,
                normalized_extent=self.normalized_extent,
            )
            torch.save(neutral_mesh, cache_path)
        vertices = neutral_mesh.get("vertices")
        faces = neutral_mesh.get("faces")
        if not isinstance(vertices, torch.Tensor) or vertices.dim() != 2:
            raise ValueError("Neutral image mesh vertices must have shape [V, 3].")
        if not isinstance(faces, torch.Tensor) or faces.dim() != 2:
            raise ValueError("Neutral image mesh faces must have shape [F, 3].")
        return int(vertices.shape[0]), int(faces.shape[0])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, Any], Dict[str, torch.Tensor]]:
        sample = self.samples[index]
        neutral_mesh = self._load_neutral_mesh(sample.person_id)
        if self.weld_coincident_vertices:
            source_to_welded = neutral_mesh.pop("source_vertex_to_welded")
            component_ids = neutral_mesh.pop("component_ids")
            largest_component_id = int(
                neutral_mesh.pop("largest_component_id").item()
            )
            if self._validated_welded_landmarks_enabled():
                landmarks_3d = self._load_validated_welded_landmarks(
                    sample.person_id,
                    neutral_mesh["vertices"],
                )
            else:
                source_vertices = neutral_mesh["vertices"].new_zeros(
                    (source_to_welded.shape[0], 3)
                )
                landmarks_3d = self._load_or_create_landmarks(
                    sample.person_id,
                    source_vertices,
                )
                landmarks_3d = _remap_landmarks_to_welded_mesh(
                    landmarks=landmarks_3d,
                    welded_vertices=neutral_mesh["vertices"],
                    source_to_welded=source_to_welded,
                    component_ids=component_ids,
                    largest_component_id=largest_component_id,
                    snap_mediapipe_ids=self.weld_landmark_ids_to_largest_component,
                )
        else:
            landmarks_3d = self._load_or_create_landmarks(
                sample.person_id,
                neutral_mesh["vertices"],
            )
        gaze_anchors = self._load_gaze_surface_anchors(
            sample.person_id,
            vertex_count=int(neutral_mesh["vertices"].shape[0]),
        )
        landmarks_3d.update(gaze_anchors)
        displacement_2d = self._load_displacement_2d(sample)
        return neutral_mesh, sample.action_unit_id, displacement_2d, landmarks_3d

    def _list_neutral_images(self) -> Dict[str, Path]:
        if not self.neutral_dir.is_dir():
            raise FileNotFoundError(self.neutral_dir)

        neutral_images: Dict[str, Path] = {}
        for path in sorted(self.neutral_dir.iterdir()):
            if not path.is_file():
                continue
            # Old numeric-template recipes historically omitted this namespace.
            # Preserve their selection; portable releases include v2 explicitly.
            if (self.data_root is None and path.name.startswith("v2_")
                    and path.name.endswith("_2.png")):
                continue
            match = NEUTRAL_IMAGE_RE.match(path.name)
            if match is None:
                continue
            identity = _normalize_person_id(match.group(1))
            if identity in neutral_images:
                raise ValueError(f"Duplicate neutral identity {identity}: {neutral_images[identity]} and {path}")
            neutral_images[identity] = path
        return neutral_images

    def _load_split_person_ids(self) -> set[str]:
        assert self.split_csv is not None
        if not self.split_csv.is_file():
            raise FileNotFoundError(self.split_csv)

        with self.split_csv.open(newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise ValueError(f"{self.split_csv} is empty.")
            missing_columns = {"person_id", "split"} - set(reader.fieldnames)
            if missing_columns:
                raise ValueError(
                    f"{self.split_csv} is missing columns: {sorted(missing_columns)}"
                )

            person_ids = set()
            for row in reader:
                row_split = row["split"].strip().lower()
                if row_split == self.split:
                    person_ids.add(_normalize_person_id(row["person_id"]))
        return person_ids

    def _validate_neutral_meshes(self, person_ids: list[str]) -> None:
        if self.neutral_mesh_path is not None:
            if not self.neutral_mesh_path.is_file():
                raise FileNotFoundError(self.neutral_mesh_path)
            return

        missing_mesh_ids = [
            person_id
            for person_id in person_ids
            if not self._neutral_mesh_path(person_id).is_file()
        ]
        if missing_mesh_ids:
            shown = ", ".join(missing_mesh_ids[:10])
            suffix = "..." if len(missing_mesh_ids) > 10 else ""
            templates = [self.neutral_mesh_filename_template]
            if self.neutral_mesh_filename_fallback_template is not None:
                templates.append(self.neutral_mesh_filename_fallback_template)
            raise FileNotFoundError(
                "Missing neutral mesh files for configured template(s) "
                f"{templates}: {shown}{suffix}"
            )

    def _list_expressed_images(
        self,
        person_ids: set[str],
    ) -> Dict[str, Tuple[Tuple[int, Path], ...]]:
        if not self.expressed_dir.is_dir():
            raise FileNotFoundError(self.expressed_dir)

        expressed_by_person: Dict[str, list[Tuple[int, Path]]] = {
            person_id: [] for person_id in person_ids
        }
        for path in sorted(self.expressed_dir.iterdir()):
            if not path.is_file():
                continue
            match = EXPRESSED_IMAGE_RE.match(path.name)
            if match is None:
                continue
            person_id = _normalize_person_id(match.group(1))
            if person_id not in expressed_by_person:
                continue
            action_unit_id = int(match.group(2))
            if action_unit_id not in self.action_unit_ids:
                continue
            if any(au == action_unit_id for au, _ in expressed_by_person[person_id]):
                raise ValueError(f"Duplicate expressed image for {person_id}, AU {action_unit_id}: {path}")
            expressed_by_person[person_id].append((action_unit_id, path))

        return {
            person_id: tuple(sorted(paths))
            for person_id, paths in expressed_by_person.items()
        }

    def _build_samples(
        self,
        neutral_images: Dict[str, Path],
        expressed_images: Dict[str, Tuple[Tuple[int, Path], ...]],
    ) -> list[ImageSample]:
        samples: list[ImageSample] = []

        for person_id in sorted(neutral_images):
            expressed_paths = expressed_images.get(person_id, ())
            if not expressed_paths:
                continue

            for action_unit_id, expressed_path in expressed_paths:
                samples.append(
                    ImageSample(
                        person_id=person_id,
                        action_unit_id=action_unit_id,
                        neutral_image_path=neutral_images[person_id],
                        expressed_image_path=expressed_path,
                        neutral_mesh_path=self._neutral_mesh_path(person_id),
                    )
                )

        return samples

    def _cache_neutral_mesh_tensors(self) -> None:
        for person_id in self.person_ids:
            cache_path = self._neutral_mesh_tensor_path(person_id)
            if not cache_path.is_file():
                neutral_mesh = _load_neutral_mesh_for_model(
                    self._neutral_mesh_path(person_id),
                    mesh_up_axis=self.mesh_up_axis,
                    mesh_front_axis=self.mesh_front_axis,
                    normalize_on_get=self.normalize_on_get,
                    normalized_extent=self.normalized_extent,
                )
                torch.save(neutral_mesh, cache_path)
            if not self.weld_coincident_vertices:
                continue
            welded_cache_path = self._welded_neutral_mesh_tensor_path(person_id)
            if welded_cache_path.is_file():
                continue
            neutral_mesh = _load_tensor_dict(cache_path)
            welded_mesh = _weld_neutral_mesh_payload(
                neutral_mesh,
                tolerance=self.weld_tolerance,
            )
            torch.save(welded_mesh, welded_cache_path)

    def _cache_displacement_2d(self) -> None:
        missing_samples = [
            sample
            for sample in self.samples
            if not self._displacement_2d_path(sample).is_file()
        ]
        if not missing_samples:
            return

        device = self._resolve_displacement_2d_cache_device()
        batch_size = self.displacement_2d_cache_batch_size
        print(
            "[INFO] Caching "
            f"{len(missing_samples)} 2D displacement target(s) on {device} "
            f"with batch size {batch_size}."
        )
        if batch_size == 1:
            self._cache_displacement_2d_single(missing_samples, device)
            return

        for start in range(0, len(missing_samples), batch_size):
            batch = missing_samples[start : start + batch_size]
            neutral_images = torch.stack(
                [
                    self._load_displacement_cache_image(sample.neutral_image_path)
                    for sample in batch
                ],
                dim=0,
            ).to(device=device)
            expressed_images = torch.stack(
                [
                    self._load_displacement_cache_image(sample.expressed_image_path)
                    for sample in batch
                ],
                dim=0,
            ).to(device=device)
            with torch.no_grad():
                displacements, masks = compute_displacement_2d(
                    neutral_images,
                    expressed_images,
                    return_mask=True,
                )
            displacements = displacements.detach().cpu()
            masks = masks.detach().cpu()
            for index, sample in enumerate(batch):
                self._save_displacement_2d(
                    sample,
                    displacements[index],
                    masks[index],
                )

    def _cache_displacement_2d_single(
        self,
        samples: Sequence[ImageSample],
        device: torch.device,
    ) -> None:
        for sample in samples:
            cache_path = self._displacement_2d_path(sample)
            if cache_path.is_file():
                continue

            neutral_image = self._load_displacement_cache_image(
                sample.neutral_image_path
            ).to(device)
            expressed_image = self._load_displacement_cache_image(
                sample.expressed_image_path
            ).to(device)
            with torch.no_grad():
                displacement, mask = compute_displacement_2d(
                    neutral_image,
                    expressed_image,
                    return_mask=True,
                )
            self._save_displacement_2d(sample, displacement, mask)

    def _save_displacement_2d(
        self,
        sample: ImageSample,
        displacement: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        torch.save(
            {
                "displacement": displacement.detach().cpu().clone(),
                "mask": mask.detach().cpu().clone(),
            },
            self._displacement_2d_path(sample),
        )

    def _resolve_displacement_2d_cache_device(self) -> torch.device:
        requested = self.displacement_2d_cache_device.strip().lower()
        if requested in {"", "cpu", "none"}:
            return torch.device("cpu")
        if requested == "gpu":
            requested = "cuda"
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "displacement_2d_cache_device requested CUDA, but CUDA is not "
                "available in this process."
            )
        return device

    def _load_displacement_cache_image(self, path: Path) -> torch.Tensor:
        image = _load_rgb_tensor(path)
        if self.displacement_2d_cache_image_size is None:
            return image
        return F.interpolate(
            image.unsqueeze(0),
            size=self.displacement_2d_cache_image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _load_neutral_mesh(self, person_id: str) -> Dict[str, torch.Tensor]:
        cache_path = (
            self._welded_neutral_mesh_tensor_path(person_id)
            if self.weld_coincident_vertices
            else self._neutral_mesh_tensor_path(person_id)
        )
        neutral_mesh = self._load_cached_tensor_dict(
            self._mesh_cache,
            cache_path,
            self.neutral_mesh_cache_size,
        )
        return {
            name: tensor.clone()
            for name, tensor in neutral_mesh.items()
        }

    def _load_displacement_2d(self, sample: ImageSample) -> Dict[str, Any]:
        if not self.precompute_displacement_2d:
            return {"neutral_image_path": str(sample.neutral_image_path),
                    "expressed_image_path": str(sample.expressed_image_path)}
        cache_path = self._displacement_2d_path(sample)
        cached_displacement = self._load_cached_tensor_dict(
            self._displacement_cache,
            cache_path,
            self.displacement_cache_size,
        )
        displacement_2d: Dict[str, Any] = {
            name: tensor.clone()
            for name, tensor in cached_displacement.items()
        }
        displacement_2d["neutral_image_path"] = str(sample.neutral_image_path)
        displacement_2d["expressed_image_path"] = str(sample.expressed_image_path)
        return displacement_2d

    def clear_memory_caches(self) -> None:
        self._mesh_cache.clear()
        self._displacement_cache.clear()

    @staticmethod
    def _load_cached_tensor_dict(
        cache: OrderedDict[Path, Dict[str, torch.Tensor]],
        cache_path: Path,
        max_items: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        if max_items == 0:
            return _load_tensor_dict(cache_path)

        cached = cache.get(cache_path)
        if cached is not None:
            cache.move_to_end(cache_path)
            return cached

        cached = _load_tensor_dict(cache_path)
        cache[cache_path] = cached
        if max_items is not None:
            while len(cache) > max_items:
                cache.popitem(last=False)
        return cached

    def _load_or_create_landmarks(
        self,
        person_id: str,
        vertices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not self.use_mediapipe_landmarks:
            return _empty_landmark_payload(vertices)

        mapping_path = self._mediapipe_landmark_path(person_id)
        if not mapping_path.is_file():
            if not self.generate_mediapipe_landmarks:
                if self.require_mediapipe_landmarks:
                    raise FileNotFoundError(
                        "Required MediaPipe landmark mapping is missing for image "
                        f"identity {person_id}: {mapping_path}"
                    )
                return _empty_landmark_payload(vertices)
            self._generate_mediapipe_mapping(person_id, mapping_path)
        landmarks = _load_landmark_payload_or_regenerate(
            mapping_path,
            vertices,
            (
                lambda: self._generate_mediapipe_mapping(person_id, mapping_path)
            )
            if self.generate_mediapipe_landmarks
            else None,
        )
        landmark_ids = landmarks.get("vertex_ids")
        landmark_count = (
            int(landmark_ids.numel())
            if isinstance(landmark_ids, torch.Tensor)
            else 0
        )
        if (
            self.require_mediapipe_landmarks
            and landmark_count < self.min_mediapipe_landmarks
        ):
            raise ValueError(
                f"Required MediaPipe landmark mapping {mapping_path} has "
                f"{landmark_count} valid landmarks; expected at least "
                f"{self.min_mediapipe_landmarks}."
            )
        return landmarks

    def _load_validated_welded_landmarks(
        self,
        person_id: str,
        welded_vertices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cache_path = self._validated_welded_landmark_path(person_id)
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Required validated welded landmark cache is missing: {cache_path}"
            )
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        status = (
            str(payload.get("status", "invalid"))
            if isinstance(payload, dict)
            else "invalid"
        )
        if status not in {"valid", "repaired"}:
            raise ValueError(
                f"Validated welded landmark cache {cache_path} has status {status}."
            )
        return _load_landmark_payload_or_regenerate(
            cache_path,
            welded_vertices,
            None,
        )

    def _validated_welded_landmarks_enabled(self) -> bool:
        return bool(self.welded_landmark_validation.get("enabled", False))

    def _load_gaze_surface_anchors(
        self,
        person_id: str,
        *,
        vertex_count: int,
    ) -> Dict[str, torch.Tensor]:
        if not self.use_gaze_surface_anchors:
            return {}
        path = self._gaze_surface_anchor_path(person_id)
        if not path.is_file():
            if self.require_gaze_surface_anchors:
                raise FileNotFoundError(
                    f"Required eye-gaze surface-anchor cache is missing: {path}"
                )
            return {}
        payload = load_eye_gaze_surface_anchor_payload(
            path,
            vertex_count=vertex_count,
        )
        return {
            "gaze_mediapipe_ids": payload["mediapipe_ids"],
            "gaze_face_vertex_ids": payload["face_vertex_ids"],
            "gaze_barycentric_weights": payload["barycentric_weights"],
            "gaze_neutral_positions": payload["neutral_positions"],
            "gaze_rigid_vertex_ids": payload["rigid_vertex_ids"],
            "gaze_roi_vertex_ids": payload["roi_vertex_ids"],
            "gaze_component_id": payload["component_id"],
        }

    def _mediapipe_landmark_path(self, person_id: str) -> Path:
        _mesh_path, using_fallback = self._neutral_mesh_path_and_variant(person_id)
        return self._mediapipe_landmark_variant_path(person_id, using_fallback)

    def _gaze_surface_anchor_path(self, person_id: str) -> Path:
        _mesh_path, using_fallback = self._neutral_mesh_path_and_variant(person_id)
        template = (
            self.gaze_surface_anchor_filename_fallback_template
            if using_fallback
            else self.gaze_surface_anchor_filename_template
        )
        filename = _format_person_id_template(
            template,
            person_id,
            "gaze_surface_anchor_filename_template",
        )
        path = Path(filename).expanduser()
        return path if path.is_absolute() else self.gaze_surface_anchor_dir / path

    def _mediapipe_landmark_variant_path(
        self,
        person_id: str,
        using_fallback: bool,
    ) -> Path:
        template = self.mediapipe_landmark_filename_template
        template_name = "mediapipe_landmark_filename_template"
        if using_fallback and self.mediapipe_landmark_filename_fallback_template:
            template = self.mediapipe_landmark_filename_fallback_template
            template_name = "mediapipe_landmark_filename_fallback_template"
        filename = _format_person_id_template(
            template,
            person_id,
            template_name,
        )
        path = Path(filename).expanduser()
        if path.is_absolute():
            return path
        return self.mediapipe_landmark_dir / path

    def _generate_mediapipe_mapping(
        self,
        person_id: str,
        mapping_path: Path,
    ) -> None:
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = _temporary_mediapipe_mapping_path(mapping_path)
        mesh_path = self._neutral_mesh_path(person_id)
        command = [
            sys.executable,
            str(self.mediapipe_mapper_path),
            "--mesh",
            str(mesh_path),
            "--output",
            str(tmp_path),
            *self.mediapipe_mapper_args,
        ]
        try:
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
                    f"{mesh_path} with {self.mediapipe_mapper_path}."
                    f"\n{details}"
                )
            _publish_mediapipe_mapping(tmp_path, mapping_path)
        finally:
            _unlink_if_exists(tmp_path)

    def _neutral_mesh_path(self, person_id: str) -> Path:
        return self._neutral_mesh_path_and_variant(person_id)[0]

    def _neutral_mesh_path_and_variant(self, person_id: str) -> tuple[Path, bool]:
        if self.neutral_mesh_path is not None:
            return self.neutral_mesh_path, False
        filename = _format_person_id_template(
            self.neutral_mesh_filename_template,
            person_id,
            "neutral_mesh_filename_template",
        )
        path = Path(filename).expanduser()
        if path.is_absolute():
            primary_path = path
        else:
            primary_path = self.neutral_mesh_dir / path
        if self.neutral_mesh_filename_fallback_template is None:
            return primary_path, False

        fallback_filename = _format_person_id_template(
            self.neutral_mesh_filename_fallback_template,
            person_id,
            "neutral_mesh_filename_fallback_template",
        )
        fallback_path = Path(fallback_filename).expanduser()
        if not fallback_path.is_absolute():
            fallback_path = self.neutral_mesh_dir / fallback_path
        if (
            self.use_mediapipe_landmarks
            and primary_path.is_file()
            and fallback_path.is_file()
            and self.mediapipe_landmark_filename_fallback_template is not None
        ):
            primary_landmarks = self._mediapipe_landmark_variant_path(
                person_id,
                False,
            )
            fallback_landmarks = self._mediapipe_landmark_variant_path(
                person_id,
                True,
            )
            if not primary_landmarks.is_file() and fallback_landmarks.is_file():
                return fallback_path, True
        if primary_path.is_file():
            return primary_path, False
        if fallback_path.is_file():
            return fallback_path, True
        return primary_path, False

    def _neutral_mesh_tensor_path(self, person_id: str) -> Path:
        _mesh_path, using_fallback = self._neutral_mesh_path_and_variant(person_id)
        variant = "_fallback" if using_fallback else ""
        return self.neutral_mesh_tensor_dir / f"mesh_{person_id}{variant}.pt"

    def _welded_neutral_mesh_tensor_path(self, person_id: str) -> Path:
        return self.welded_neutral_mesh_tensor_dir / self._neutral_mesh_tensor_path(
            person_id
        ).name

    def _validated_welded_landmark_path(self, person_id: str) -> Path:
        filename = Path(self._neutral_mesh_tensor_path(person_id).name).with_suffix(
            ".json"
        )
        return self.validated_welded_landmark_dir / filename

    def _displacement_2d_path(self, sample: ImageSample) -> Path:
        return self.displacement_2d_dir / f"{sample.expressed_image_path.stem}.pt"


def _weld_neutral_mesh_payload(
    mesh: Dict[str, torch.Tensor],
    *,
    tolerance: float,
) -> Dict[str, torch.Tensor]:
    vertices = mesh.get("vertices")
    faces = mesh.get("faces")
    if not isinstance(vertices, torch.Tensor) or vertices.dim() != 2:
        raise ValueError("Neutral mesh vertices must have shape [V, 3].")
    if not isinstance(faces, torch.Tensor) or faces.dim() != 2:
        raise ValueError("Neutral mesh faces must have shape [F, 3].")
    if tolerance <= 0.0:
        raise ValueError("Mesh weld tolerance must be positive.")

    vertices_cpu = vertices.detach().cpu().float().contiguous()
    quantized = torch.round(vertices_cpu / tolerance).to(dtype=torch.int64).numpy()
    _unique_positions, source_to_welded_np = np.unique(
        quantized,
        axis=0,
        return_inverse=True,
    )
    source_to_welded = torch.from_numpy(source_to_welded_np).long()
    welded_vertex_count = int(source_to_welded.amax().item()) + 1
    welded_vertices = vertices_cpu.new_zeros((welded_vertex_count, 3))
    welded_vertices.index_add_(0, source_to_welded, vertices_cpu)
    group_counts = torch.bincount(
        source_to_welded,
        minlength=welded_vertex_count,
    ).to(dtype=welded_vertices.dtype)
    welded_vertices = welded_vertices / group_counts.unsqueeze(-1).clamp_min(1.0)

    welded_faces = source_to_welded.index_select(
        0,
        faces.detach().cpu().long().reshape(-1),
    ).reshape(-1, 3)
    nondegenerate = (
        (welded_faces[:, 0] != welded_faces[:, 1])
        & (welded_faces[:, 1] != welded_faces[:, 2])
        & (welded_faces[:, 2] != welded_faces[:, 0])
    )
    welded_faces = welded_faces[nondegenerate].contiguous()
    component_ids, largest_component_id = _mesh_connected_components(
        welded_vertex_count,
        welded_faces,
    )
    return {
        "vertices": welded_vertices.contiguous(),
        "faces": welded_faces,
        "normals": _compute_vertex_normals(welded_vertices, welded_faces),
        "source_vertex_to_welded": source_to_welded.contiguous(),
        "component_ids": component_ids.contiguous(),
        "largest_component_id": torch.tensor(
            largest_component_id,
            dtype=torch.long,
        ),
    }


def _mesh_connected_components(
    vertex_count: int,
    faces: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError as exc:
        raise ImportError(
            "Welding image meshes requires scipy for connected components."
        ) from exc

    faces_np = faces.detach().cpu().long().numpy()
    rows = np.concatenate(
        (faces_np[:, 0], faces_np[:, 1], faces_np[:, 2]),
    )
    columns = np.concatenate(
        (faces_np[:, 1], faces_np[:, 2], faces_np[:, 0]),
    )
    graph = coo_matrix(
        (np.ones(rows.shape[0], dtype=np.uint8), (rows, columns)),
        shape=(vertex_count, vertex_count),
    )
    _component_count, labels = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )
    component_sizes = np.bincount(labels, minlength=int(labels.max()) + 1)
    largest_component_id = int(component_sizes.argmax())
    return torch.from_numpy(labels).long(), largest_component_id


def _remap_landmarks_to_welded_mesh(
    *,
    landmarks: Dict[str, torch.Tensor],
    welded_vertices: torch.Tensor,
    source_to_welded: torch.Tensor,
    component_ids: torch.Tensor,
    largest_component_id: int,
    snap_mediapipe_ids: set[int],
) -> Dict[str, torch.Tensor]:
    mediapipe_ids = landmarks.get("mediapipe_ids")
    source_vertex_ids = landmarks.get("vertex_ids")
    if not isinstance(mediapipe_ids, torch.Tensor) or not isinstance(
        source_vertex_ids,
        torch.Tensor,
    ):
        raise ValueError("Landmark payload must contain mediapipe_ids and vertex_ids.")
    if source_vertex_ids.numel() == 0:
        return _empty_landmark_payload(welded_vertices)

    source_to_welded = source_to_welded.detach().cpu().long()
    source_vertex_ids = source_vertex_ids.detach().cpu().long()
    welded_vertex_ids = source_to_welded.index_select(0, source_vertex_ids)
    component_ids = component_ids.detach().cpu().long()
    if snap_mediapipe_ids:
        snap_mask = torch.tensor(
            [int(value) in snap_mediapipe_ids for value in mediapipe_ids.tolist()],
            dtype=torch.bool,
        )
        off_surface = component_ids.index_select(
            0,
            welded_vertex_ids,
        ) != int(largest_component_id)
        repair_indices = torch.where(snap_mask & off_surface)[0]
        if repair_indices.numel() > 0:
            skin_vertex_ids = torch.where(
                component_ids == int(largest_component_id)
            )[0]
            source_points = welded_vertices.detach().cpu().float().index_select(
                0,
                welded_vertex_ids.index_select(0, repair_indices),
            )
            skin_points = welded_vertices.detach().cpu().float().index_select(
                0,
                skin_vertex_ids,
            )
            nearest_skin = torch.cdist(source_points, skin_points).argmin(dim=1)
            welded_vertex_ids[repair_indices] = skin_vertex_ids.index_select(
                0,
                nearest_skin,
            )

    return {
        "mediapipe_ids": mediapipe_ids.detach().cpu().long().contiguous(),
        "vertex_ids": welded_vertex_ids.contiguous(),
        "neutral_positions": welded_vertices.detach()
        .cpu()
        .float()
        .index_select(0, welded_vertex_ids)
        .contiguous(),
    }


def _normalize_person_id(value: str) -> str:
    return normalize_person_id(value)


def _format_person_id_template(template: str, person_id: str, name: str) -> str:
    try:
        return str(template).format(person_id=person_id, identity=canonical_identity(person_id))
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"{name} may only reference {{person_id}} or {{identity}}.") from exc


def _normalize_person_id_filter(
    values: Optional[Sequence[Union[int, str]]],
) -> Optional[set[str]]:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raw_value = values.decode() if isinstance(values, bytes) else values
        raw_values: Sequence[Union[int, str]] = [raw_value]
    else:
        raw_values = values

    person_ids = {_normalize_person_id(str(value)) for value in raw_values}
    if not person_ids:
        raise ValueError("person_ids must not be empty when provided.")
    return person_ids


def _load_person_id_filter_file(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    person_ids = {
        _normalize_person_id(line.split("#", 1)[0].strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    }
    if not person_ids:
        raise ValueError(f"Person ID exclusion file is empty: {path}")
    return person_ids


def _normalize_action_unit_ids(
    values: Optional[Sequence[Union[int, str]]],
) -> Tuple[int, ...]:
    if values is None:
        return tuple(sorted(AU_NAME))
    if isinstance(values, (str, bytes)):
        raw_value = values.decode() if isinstance(values, bytes) else values
        raw_values: Sequence[Union[int, str]] = [raw_value]
    else:
        raw_values = values

    name_to_id = {name: action_unit_id for action_unit_id, name in AU_NAME.items()}
    action_unit_ids = []
    for value in raw_values:
        if isinstance(value, str) and not value.isdigit():
            if value not in name_to_id:
                raise ValueError(f"Unknown action unit name: {value!r}.")
            action_unit_id = name_to_id[value]
        else:
            action_unit_id = int(value)
        if action_unit_id not in AU_NAME:
            raise ValueError(f"Unknown action unit id: {action_unit_id}.")
        action_unit_ids.append(action_unit_id)

    if not action_unit_ids:
        raise ValueError("action_units must not be empty when provided.")
    return tuple(action_unit_ids)


def _normalize_optional_image_size(
    image_size: Optional[Sequence[int]],
    name: str,
) -> Optional[Tuple[int, int]]:
    if image_size is None:
        return None
    if len(image_size) != 2:
        raise ValueError(f"{name} must contain exactly two values: [height, width].")
    height = int(image_size[0])
    width = int(image_size[1])
    if height < 1 or width < 1:
        raise ValueError(f"{name} values must be positive.")
    return height, width


def _normalize_memory_cache_size(value: Any, name: str) -> Optional[int]:
    if value in (None, ""):
        return None
    size = int(value)
    if size < 0:
        raise ValueError(f"{name} must be greater than or equal to 0.")
    return size


def _normalize_optional_positive_int(value: Any, name: str) -> Optional[int]:
    if value in (None, ""):
        return None
    size = int(value)
    if size < 1:
        raise ValueError(f"{name} must be positive.")
    return size


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    raw = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return raw.view(image.height, image.width, 3).permute(2, 0, 1).float() / 255.0


def _load_tensor_dict(path: Path) -> Dict[str, torch.Tensor]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")

    if not isinstance(loaded, dict):
        raise ValueError(f"Expected tensor cache at {path} to contain a dictionary.")
    for key, value in loaded.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError(
                f"Expected tensor cache at {path} to map strings to tensors."
            )
    return loaded


def _load_glb_for_model(path: Path) -> Dict[str, torch.Tensor]:
    gltf, binary_blob = _read_glb(path)

    vertices_by_primitive = []
    normals_by_primitive = []
    faces_by_primitive = []
    vertex_offset = 0

    for mesh in gltf.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if primitive.get("mode", 4) != 4:
                raise ValueError(f"{path} contains a non-triangle GLB primitive.")

            attributes = primitive.get("attributes", {})
            if "POSITION" not in attributes:
                continue

            vertices = _read_accessor(gltf, binary_blob, attributes["POSITION"]).float()
            faces = _read_primitive_faces(gltf, binary_blob, primitive, vertices.shape[0])
            vertices_by_primitive.append(vertices)
            faces_by_primitive.append(faces + vertex_offset)

            if "NORMAL" in attributes:
                normals = _read_accessor(gltf, binary_blob, attributes["NORMAL"]).float()
                normals_by_primitive.append(F.normalize(normals, dim=-1, eps=1.0e-6))
            else:
                normals_by_primitive.append(None)

            vertex_offset += vertices.shape[0]

    if not vertices_by_primitive:
        raise ValueError(f"No triangle mesh vertices were found in {path}.")

    vertices = torch.cat(vertices_by_primitive, dim=0).contiguous()
    faces = torch.cat(faces_by_primitive, dim=0).long().contiguous()
    if all(normals is not None for normals in normals_by_primitive):
        normals = torch.cat(normals_by_primitive, dim=0).contiguous()
    else:
        normals = _compute_vertex_normals(vertices, faces)

    return {
        "vertices": vertices,
        "faces": faces,
        "normals": normals,
    }


def _load_neutral_mesh_for_model(
    path: Path,
    mesh_up_axis: str,
    mesh_front_axis: str,
    normalize_on_get: bool,
    normalized_extent: float,
) -> Dict[str, torch.Tensor]:
    suffix = path.suffix.lower()
    if suffix == ".glb":
        return _load_glb_for_model(path)
    if suffix != ".fbx":
        raise ValueError(
            "neutral_mesh_path must point to a .glb or .fbx file; "
            f"got {path}."
        )

    faces, vertices, _blendshapes, _metadata = fbx_to_tensor(
        path,
        include_metadata=True,
    )
    mesh = {
        "vertices": vertices.float().contiguous(),
        "faces": faces.long().contiguous(),
        "normals": _compute_vertex_normals(vertices.float(), faces.long()),
    }
    zero_delta = torch.zeros_like(mesh["vertices"])
    mesh, _zero_delta, _landmarks = _transform_model_sample(
        mesh=mesh,
        delta_vertices=zero_delta,
        landmarks_3d=_empty_landmark_payload(mesh["vertices"]),
        mesh_up_axis=mesh_up_axis,
        mesh_front_axis=mesh_front_axis,
        normalize_on_get=normalize_on_get,
        normalized_extent=normalized_extent,
    )
    return mesh


def _read_primitive_faces(
    gltf: Dict[str, Any],
    binary_blob: bytes,
    primitive: Dict[str, Any],
    num_vertices: int,
) -> torch.Tensor:
    if "indices" in primitive:
        indices = _read_accessor(gltf, binary_blob, primitive["indices"]).long().flatten()
    else:
        indices = torch.arange(num_vertices, dtype=torch.long)

    if indices.numel() % 3 != 0:
        raise ValueError("Triangle index count must be divisible by 3.")
    return indices.view(-1, 3)


def _compute_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    triangles = vertices[faces]
    face_normals = torch.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=-1,
    )

    normals = torch.zeros_like(vertices)
    for corner in range(3):
        normals.index_add_(0, faces[:, corner], face_normals)
    return F.normalize(normals, dim=-1, eps=1.0e-6)


def _read_glb(path: Path) -> Tuple[Dict[str, Any], bytes]:
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or length != len(data):
        raise ValueError(f"{path} is not a valid GLB 2.0 file.")

    chunks: Dict[int, bytes] = {}
    offset = 12
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunks[chunk_type] = data[offset : offset + chunk_length]
        offset += chunk_length

    json_chunk = chunks[0x4E4F534A].decode("utf-8").rstrip("\x00 ")
    binary_blob = chunks[0x004E4942]
    return json.loads(json_chunk), binary_blob


def _read_accessor(
    gltf: Dict[str, Any],
    binary_blob: bytes,
    accessor_index: int,
) -> torch.Tensor:
    accessor = gltf["accessors"][accessor_index]
    if "sparse" in accessor:
        raise ValueError("Sparse GLB accessors are not supported.")

    buffer_view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = _COMPONENT_DTYPES[accessor["componentType"]]
    components = _TYPE_COMPONENTS[accessor["type"]]
    count = accessor["count"]
    element_size = torch.empty((), dtype=dtype).element_size()
    packed_stride = element_size * components
    stride = buffer_view.get("byteStride", packed_stride)
    byte_offset = buffer_view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    buffer = bytearray(binary_blob)

    if stride == packed_stride:
        tensor = torch.frombuffer(
            buffer,
            dtype=dtype,
            count=count * components,
            offset=byte_offset,
        ).clone()
        return tensor.view(count, components)

    rows = []
    for row in range(count):
        rows.append(
            torch.frombuffer(
                buffer,
                dtype=dtype,
                count=components,
                offset=byte_offset + row * stride,
            ).clone()
        )
    return torch.stack(rows, dim=0)
