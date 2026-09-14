from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import multiprocessing
import os
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.fbx_to_tensor import (
    AU_NAME,
    CANONICAL_BLENDSHAPE_NAMES,
    fbx_to_tensor,
)
from utils.mouth_landmark_validation import VALIDATED_MOUTH_LANDMARK_VERSION
from dataset.hf_assets import canonical_identity, prepare_assets, asset_cache_namespace, versioned_cache_dir

try:
    import h5py
except ImportError:  # pragma: no cover - h5py is optional unless requested.
    h5py = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    tqdm = None


PathLike = Union[str, Path]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MESH_DIR = ROOT / "data" / "meshes_ict"
DEFAULT_CACHE_DIR = ROOT / ".cache" / "mesh_dataset"
DEFAULT_LANDMARK_MAPPER = ROOT / "map_mediapipe_landmarks.py"
VALID_SPLITS = {"train", "val", "test", "all"}
DEFAULT_CACHE_WORKERS = 4
ACTION_UNIT_IDS = tuple(sorted(AU_NAME))
ACTION_UNIT_NAMES = {
    action_unit_id: AU_NAME[action_unit_id] for action_unit_id in ACTION_UNIT_IDS
}


@dataclass(frozen=True)
class MeshSample:
    mesh_id: str
    action_unit_id: int


@dataclass(frozen=True)
class MeshCacheTask:
    mesh_id: str
    mesh_path: Path
    cache_path: Path
    use_mediapipe_landmarks: bool
    generate_mediapipe_landmarks: bool
    mediapipe_landmark_dir: Optional[Path]
    mediapipe_mapper_path: Path
    mediapipe_mapper_args: Tuple[str, ...]
    drop_mouth_probability: float
    cut_eyeballs_probability: float
    cache_random_seed: int
    cache_format: str
    cache_compression: Optional[str]
    cache_compression_level: int
    hdf5_blendshape_chunk_vertices: int
    blendshape_name_mapping: str = "metahuman"


class MeshDataset(Dataset):
    def __init__(
        self,
        mesh_dir: PathLike = DEFAULT_MESH_DIR,
        split: str = "all",
        split_csv: Optional[PathLike] = None,
        exclude_mesh_ids: Optional[Sequence[str]] = None,
        exclude_mesh_ids_file: Optional[PathLike] = None,
        identity_fraction: float = 1.0,
        identity_fraction_seed: int = 7,
        cache_dir: PathLike = DEFAULT_CACHE_DIR,
        action_units: Optional[Sequence[Union[int, str]]] = None,
        exclude_action_units: Optional[Sequence[Union[int, str]]] = None,
        use_mediapipe_landmarks: bool = True,
        generate_mediapipe_landmarks: bool = False,
        mediapipe_landmark_dir: Optional[PathLike] = None,
        mediapipe_mapper_path: PathLike = DEFAULT_LANDMARK_MAPPER,
        mediapipe_mapper_args: Optional[Sequence[str]] = None,
        cache_workers: Optional[int] = None,
        trust_existing_cache: bool = False,
        cache_format: str = "torch",
        cache_compression: Optional[str] = None,
        cache_compression_level: int = 1,
        hdf5_blendshape_chunk_vertices: int = 8192,
        blendshape_name_mapping: str = "metahuman",
        rigged_mesh_cache_size: Optional[int] = None,
        drop_teeth_probability: float = 0.0,
        drop_mouth_probability: Optional[float] = None,
        cut_eyeballs_probability: float = 0.0,
        cache_random_seed: int = 0,
        normalize_on_get: bool = True,
        mesh_up_axis: str = "z",
        mesh_front_axis: str = "-y",
        normalized_extent: float = 2.0,
        topology_augmentation_probability: float = 0.0,
        topology_augmentation_mode: str = "face_split",
        topology_augmentation_face_split_probability: float = 0.25,
        topology_augmentation_subdivision_levels: int = 1,
        topology_augmentation_max_vertices: Optional[int] = None,
        topology_augmentation_train_only: bool = True,
        shape_augmentation_probability: float = 0.0,
        shape_augmentation_scale: float = 0.08,
        shape_augmentation_num_anchors: int = 24,
        shape_augmentation_smoothness: float = 0.35,
        shape_augmentation_train_only: bool = True,
        mouth_landmark_validation: Optional[Mapping[str, object]] = None,
        data_root: Optional[PathLike] = None,
        mesh_source: str = "ict",
        asset_cache_dir: Optional[PathLike] = None,
    ) -> None:
        self.data_root = Path(data_root).expanduser() if data_root is not None else None
        if self.data_root is not None:
            if mesh_source not in {"ict", "custom"}:
                raise ValueError("mesh_source must be 'ict' or 'custom'")
            component = f"meshes_{mesh_source}"
            source = prepare_assets(
                self.data_root, asset_cache_dir or Path(cache_dir).expanduser().parent / "hf_assets",
                (component,),
            )
            mesh_dir = source / component
            cache_dir = versioned_cache_dir(cache_dir, self.data_root, component)
            if generate_mediapipe_landmarks and mediapipe_landmark_dir is None:
                raise ValueError("With data_root, generate landmarks into an explicit writable mediapipe_landmark_dir")
        self.mesh_dir = Path(mesh_dir).expanduser()
        from dataset.layout import assert_dataset_ready
        assert_dataset_ready(self.mesh_dir)
        self.split = split.lower()
        self.split_csv = Path(split_csv).expanduser() if split_csv is not None else None
        self.exclude_mesh_ids_file = (
            Path(exclude_mesh_ids_file).expanduser()
            if exclude_mesh_ids_file is not None
            else None
        )
        self.identity_fraction = float(identity_fraction)
        self.identity_fraction_seed = int(identity_fraction_seed)
        self.cache_dir = Path(cache_dir).expanduser()
        if drop_mouth_probability is None:
            drop_mouth_probability = drop_teeth_probability
        self.drop_mouth_probability = float(drop_mouth_probability)
        self.cut_eyeballs_probability = float(cut_eyeballs_probability)
        self.cache_random_seed = int(cache_random_seed)
        self.blendshape_name_mapping = _normalize_blendshape_name_mapping(
            blendshape_name_mapping
        )
        self.normalize_on_get = bool(normalize_on_get)
        self.mesh_up_axis = _normalize_mesh_up_axis(mesh_up_axis)
        self.mesh_front_axis = _normalize_mesh_front_axis(mesh_front_axis)
        self.normalized_extent = float(normalized_extent)
        self.topology_augmentation_probability = float(
            topology_augmentation_probability
        )
        self.topology_augmentation_mode = _normalize_topology_augmentation_mode(
            topology_augmentation_mode,
        )
        self.topology_augmentation_face_split_probability = float(
            topology_augmentation_face_split_probability
        )
        self.topology_augmentation_subdivision_levels = int(
            topology_augmentation_subdivision_levels
        )
        self.topology_augmentation_max_vertices = (
            int(topology_augmentation_max_vertices)
            if topology_augmentation_max_vertices is not None
            else None
        )
        self.topology_augmentation_train_only = bool(topology_augmentation_train_only)
        self.shape_augmentation_probability = float(shape_augmentation_probability)
        self.shape_augmentation_scale = float(shape_augmentation_scale)
        self.shape_augmentation_num_anchors = int(shape_augmentation_num_anchors)
        self.shape_augmentation_smoothness = float(shape_augmentation_smoothness)
        self.shape_augmentation_train_only = bool(shape_augmentation_train_only)
        if not 0.0 <= self.drop_mouth_probability <= 1.0:
            raise ValueError("drop_mouth_probability must be between 0.0 and 1.0.")
        if not 0.0 < self.identity_fraction <= 1.0:
            raise ValueError("identity_fraction must be in (0, 1].")
        if not 0.0 <= self.cut_eyeballs_probability <= 1.0:
            raise ValueError("cut_eyeballs_probability must be between 0.0 and 1.0.")
        if not 0.0 <= self.topology_augmentation_probability <= 1.0:
            raise ValueError(
                "topology_augmentation_probability must be between 0.0 and 1.0."
            )
        if not 0.0 <= self.topology_augmentation_face_split_probability <= 1.0:
            raise ValueError(
                "topology_augmentation_face_split_probability must be between "
                "0.0 and 1.0."
            )
        if self.topology_augmentation_subdivision_levels < 0:
            raise ValueError("topology_augmentation_subdivision_levels must be >= 0.")
        if (
            self.topology_augmentation_max_vertices is not None
            and self.topology_augmentation_max_vertices <= 0
        ):
            raise ValueError("topology_augmentation_max_vertices must be positive.")
        if not 0.0 <= self.shape_augmentation_probability <= 1.0:
            raise ValueError(
                "shape_augmentation_probability must be between 0.0 and 1.0."
            )
        if self.shape_augmentation_scale < 0.0:
            raise ValueError("shape_augmentation_scale must be >= 0.")
        if self.shape_augmentation_num_anchors < 1:
            raise ValueError("shape_augmentation_num_anchors must be at least 1.")
        if self.shape_augmentation_smoothness <= 0.0:
            raise ValueError("shape_augmentation_smoothness must be greater than 0.")
        if self.normalized_extent <= 0.0:
            raise ValueError("normalized_extent must be greater than 0.")
        self.rigged_mesh_dir = self._rigged_mesh_cache_dir()
        self.use_mediapipe_landmarks = bool(use_mediapipe_landmarks)
        self.generate_mediapipe_landmarks = bool(generate_mediapipe_landmarks)
        self.mediapipe_landmark_dir = (
            Path(mediapipe_landmark_dir).expanduser()
            if mediapipe_landmark_dir is not None
            else None
        )
        self.mediapipe_mapper_path = Path(mediapipe_mapper_path).expanduser()
        self.mediapipe_mapper_args = tuple(
            str(arg) for arg in (mediapipe_mapper_args or ())
        )
        self.cache_workers = cache_workers
        self.trust_existing_cache = bool(trust_existing_cache)
        self.cache_format = _normalize_cache_format(cache_format)
        self.cache_compression = _normalize_cache_compression(cache_compression)
        self.cache_compression_level = _normalize_cache_compression_level(
            cache_compression_level,
        )
        self.hdf5_blendshape_chunk_vertices = (
            _normalize_hdf5_blendshape_chunk_vertices(
                hdf5_blendshape_chunk_vertices,
            )
        )
        self.rigged_mesh_cache_size = _normalize_rigged_mesh_cache_size(
            rigged_mesh_cache_size,
        )
        self.mouth_landmark_validation = dict(mouth_landmark_validation or {})
        self.validated_mouth_landmark_dir = Path(
            str(
                self.mouth_landmark_validation.get(
                    "cache_dir",
                    self.cache_dir / "validated_mouth_landmarks_v1",
                )
            )
        ).expanduser()
        self._validated_mouth_landmarks: dict[
            str, Dict[str, torch.Tensor]
        ] = {}
        self.action_unit_ids = _exclude_action_unit_ids(
            _normalize_action_unit_ids(action_units),
            exclude_action_units,
        )
        self.action_unit_names = {
            action_unit_id: AU_NAME[action_unit_id]
            for action_unit_id in self.action_unit_ids
        }
        self._rigged_mesh_cache: OrderedDict[
            Path, Dict[str, object]
        ] = OrderedDict()

        if self.split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}.")
        if self.split != "all" and self.split_csv is None:
            raise ValueError("split_csv is required unless split='all'.")

        self.rigged_mesh_dir.mkdir(parents=True, exist_ok=True)
        if self.mediapipe_landmark_dir is not None:
            self.mediapipe_landmark_dir.mkdir(parents=True, exist_ok=True)

        mesh_paths = self._list_mesh_paths()
        if self.split != "all":
            allowed_mesh_ids = self._load_split_mesh_ids()
            mesh_paths = {
                mesh_id: path
                for mesh_id, path in mesh_paths.items()
                if mesh_id in allowed_mesh_ids
            }
        excluded_mesh_ids = _load_excluded_mesh_ids(
            exclude_mesh_ids,
            self.exclude_mesh_ids_file,
        )
        if self.data_root is not None:
            excluded_mesh_ids = {canonical_identity(value) for value in excluded_mesh_ids}
        if excluded_mesh_ids:
            mesh_paths = {
                mesh_id: path
                for mesh_id, path in mesh_paths.items()
                if mesh_id not in excluded_mesh_ids
            }
        unscaled_identity_count = len(mesh_paths)
        mesh_paths = _identity_fraction_subset(
            mesh_paths,
            fraction=self.identity_fraction,
            seed=self.identity_fraction_seed,
        )
        if len(mesh_paths) != unscaled_identity_count:
            print(
                f"[INFO] Identity-scale subset retained {len(mesh_paths)}/"
                f"{unscaled_identity_count} mesh(es) "
                f"(fraction={self.identity_fraction:g}, "
                f"seed={self.identity_fraction_seed})."
            )

        self.mesh_ids = sorted(mesh_paths)
        self.mesh_paths = {mesh_id: mesh_paths[mesh_id] for mesh_id in self.mesh_ids}
        self.samples = self._build_samples()
        self._cache_rigged_meshes()
        self._load_validated_mouth_landmark_caches()

        print(
            f"[INFO] Found {len(self.mesh_ids)} rigged meshes "
            f"for split {self.split}."
        )
        print(
            f"[INFO] Found {len(self.samples)} mesh/AU samples "
            f"for split {self.split}."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[Dict[str, torch.Tensor], int, torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[index]
        rigged_mesh = self._load_rigged_mesh_sample(
            sample.mesh_id,
            sample.action_unit_id,
        )
        validated_landmarks = self._validated_mouth_landmarks.get(sample.mesh_id)
        if validated_landmarks is not None:
            rigged_mesh = dict(rigged_mesh)
            rigged_mesh["landmarks_3d"] = _landmark_payload_from_validated_mapping(
                rigged_mesh,
                validated_landmarks,
            )
        mesh = _mesh_for_model(rigged_mesh)
        delta_vertices = _delta_vertices_for_action_unit(
            rigged_mesh,
            sample.action_unit_id,
        )
        landmarks_3d = _landmarks_for_action_unit(rigged_mesh, delta_vertices)
        if "vertex_groups" in rigged_mesh:
            landmarks_3d["vertex_groups"] = _clone_vertex_groups(
                rigged_mesh.get("vertex_groups")
            )
        mesh, delta_vertices, landmarks_3d = _augment_model_sample(
            mesh=mesh,
            delta_vertices=delta_vertices,
            landmarks_3d=landmarks_3d,
            vertex_groups=rigged_mesh.get("vertex_groups"),
            drop_mouth_probability=self.drop_mouth_probability,
            cut_eyeballs_probability=self.cut_eyeballs_probability,
            mesh_front_axis=self.mesh_front_axis,
        )
        if self._shape_augmentation_enabled():
            mesh, delta_vertices, landmarks_3d = _augment_shape_sample(
                mesh=mesh,
                delta_vertices=delta_vertices,
                landmarks_3d=landmarks_3d,
                probability=self.shape_augmentation_probability,
                warp_scale=self.shape_augmentation_scale,
                num_anchors=self.shape_augmentation_num_anchors,
                smoothness=self.shape_augmentation_smoothness,
                mesh_up_axis=self.mesh_up_axis,
                mesh_front_axis=self.mesh_front_axis,
            )
        if self._topology_augmentation_enabled():
            mesh, delta_vertices, landmarks_3d = _augment_topology_sample(
                mesh=mesh,
                delta_vertices=delta_vertices,
                landmarks_3d=landmarks_3d,
                probability=self.topology_augmentation_probability,
                mode=self.topology_augmentation_mode,
                face_split_probability=(
                    self.topology_augmentation_face_split_probability
                ),
                subdivision_levels=self.topology_augmentation_subdivision_levels,
                max_vertices=self.topology_augmentation_max_vertices,
            )
        mesh, delta_vertices, landmarks_3d = _transform_model_sample(
            mesh=mesh,
            delta_vertices=delta_vertices,
            landmarks_3d=landmarks_3d,
            mesh_up_axis=self.mesh_up_axis,
            mesh_front_axis=self.mesh_front_axis,
            normalize_on_get=self.normalize_on_get,
            normalized_extent=self.normalized_extent,
        )
        return mesh, sample.action_unit_id, delta_vertices, landmarks_3d

    def _topology_augmentation_enabled(self) -> bool:
        if self.topology_augmentation_probability <= 0.0:
            return False
        if self.topology_augmentation_train_only and self.split in {"val", "test"}:
            return False
        return True

    def _shape_augmentation_enabled(self) -> bool:
        if self.shape_augmentation_probability <= 0.0:
            return False
        if self.shape_augmentation_train_only and self.split in {"val", "test"}:
            return False
        return True

    def _list_mesh_paths(self) -> Dict[str, Path]:
        if self.mesh_dir.is_file():
            if self.mesh_dir.suffix.lower() != ".fbx":
                raise ValueError(
                    "mesh_dir must be a directory of .fbx files or one .fbx file."
                )
            return {self.mesh_dir.stem: self.mesh_dir}

        if not self.mesh_dir.is_dir():
            raise FileNotFoundError(self.mesh_dir)

        mesh_paths: Dict[str, Path] = {}
        for path in sorted(self.mesh_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() != ".fbx":
                continue
            mesh_paths[path.stem] = path
        return mesh_paths

    def _load_split_mesh_ids(self) -> set[str]:
        assert self.split_csv is not None
        if not self.split_csv.is_file():
            raise FileNotFoundError(self.split_csv)

        with self.split_csv.open(newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise ValueError(f"{self.split_csv} is empty.")
            missing_columns = {"mesh_id", "split"} - set(reader.fieldnames)
            if missing_columns:
                raise ValueError(
                    f"{self.split_csv} is missing columns: {sorted(missing_columns)}"
                )

            mesh_ids = set()
            for row in reader:
                row_split = row["split"].strip().lower()
                if row_split == self.split:
                    normalize = canonical_identity if self.data_root is not None else _normalize_mesh_id
                    mesh_ids.add(normalize(row["mesh_id"]))
        return mesh_ids

    def _build_samples(self) -> list[MeshSample]:
        return [
            MeshSample(mesh_id=mesh_id, action_unit_id=action_unit_id)
            for mesh_id in self.mesh_ids
            for action_unit_id in self.action_unit_ids
        ]

    def _cache_rigged_meshes(self) -> None:
        tasks = [self._cache_task(mesh_id) for mesh_id in self.mesh_ids]
        if self.trust_existing_cache:
            missing_tasks = [task for task in tasks if not task.cache_path.is_file()]
            reused_count = len(tasks) - len(missing_tasks)
            if reused_count:
                print(
                    f"[INFO] Reusing {reused_count} existing rigged mesh caches "
                    "without revalidation."
                )
            tasks = missing_tasks

        if not tasks:
            return

        worker_count = _cache_worker_count(self.cache_workers, len(tasks))
        print(
            f"[INFO] Caching {len(tasks)} rigged meshes with "
            f"{worker_count} worker processes."
        )

        if worker_count == 1:
            for task in _progress(tasks, total=len(tasks), desc="cache meshes"):
                _cache_rigged_mesh_task(task)
            return

        context = multiprocessing.get_context(_cache_process_start_method())
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
            ) as executor:
                list(
                    _progress(
                        executor.map(_cache_rigged_mesh_task, tasks),
                        total=len(tasks),
                        desc="cache meshes",
                    )
                )
        except BrokenProcessPool as exc:
            raise RuntimeError(
                "A mesh cache worker process crashed while importing FBX files. "
                "This usually points to Blender/bpy native code or the node running "
                "out of memory. Retry with fewer workers, for example "
                "TOPORIG_MESH_CACHE_WORKERS=2."
            ) from exc

    def _cache_task(self, mesh_id: str) -> MeshCacheTask:
        return MeshCacheTask(
            mesh_id=mesh_id,
            mesh_path=self.mesh_paths[mesh_id],
            cache_path=self._rigged_mesh_path(mesh_id),
            use_mediapipe_landmarks=self.use_mediapipe_landmarks,
            generate_mediapipe_landmarks=self.generate_mediapipe_landmarks,
            mediapipe_landmark_dir=self.mediapipe_landmark_dir,
            mediapipe_mapper_path=self.mediapipe_mapper_path,
            mediapipe_mapper_args=self.mediapipe_mapper_args,
            drop_mouth_probability=self.drop_mouth_probability,
            cut_eyeballs_probability=self.cut_eyeballs_probability,
            cache_random_seed=self.cache_random_seed,
            cache_format=self.cache_format,
            cache_compression=self.cache_compression,
            cache_compression_level=self.cache_compression_level,
            hdf5_blendshape_chunk_vertices=self.hdf5_blendshape_chunk_vertices,
            blendshape_name_mapping=self.blendshape_name_mapping,
        )

    def _load_validated_mouth_landmark_caches(self) -> None:
        if not bool(self.mouth_landmark_validation.get("enabled", False)):
            return
        require_cache = bool(
            self.mouth_landmark_validation.get("require_cache", True)
        )
        skip_invalid = bool(
            self.mouth_landmark_validation.get("skip_invalid", True)
        )
        missing: list[tuple[str, Path]] = []
        invalid_meshes: dict[str, str] = {}
        invalid_action_units: dict[str, set[int]] = {}
        checked_action_units: dict[str, set[int]] = {}

        for mesh_id in self.mesh_ids:
            cache_path = self._validated_mouth_landmark_path(mesh_id)
            if not cache_path.is_file():
                if require_cache:
                    missing.append((mesh_id, cache_path))
                continue
            with cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected {cache_path} to contain a validation mapping."
                )
            version = int(payload.get("version", -1))
            if version != VALIDATED_MOUTH_LANDMARK_VERSION:
                raise ValueError(
                    f"Validated mouth landmark cache {cache_path} has version "
                    f"{version}; expected {VALIDATED_MOUTH_LANDMARK_VERSION}."
                )
            status = str(payload.get("status", "invalid"))
            if status not in {"valid", "repaired"}:
                reasons = payload.get("reasons", ())
                invalid_meshes[mesh_id] = (
                    ",".join(str(value) for value in reasons) or status
                )
                continue
            raw_mapping = payload.get("mapping")
            if not isinstance(raw_mapping, Mapping):
                raise ValueError(
                    f"Validated mouth landmark cache {cache_path} has no mapping."
                )
            ordered = sorted(
                (int(mediapipe_id), int(vertex_id))
                for mediapipe_id, vertex_id in raw_mapping.items()
            )
            self._validated_mouth_landmarks[mesh_id] = {
                "mediapipe_ids": torch.tensor(
                    [value[0] for value in ordered],
                    dtype=torch.long,
                ),
                "vertex_ids": torch.tensor(
                    [value[1] for value in ordered],
                    dtype=torch.long,
                ),
            }
            invalid_action_units[mesh_id] = {
                int(value) for value in payload.get("invalid_action_units", ())
            }
            checked_action_units[mesh_id] = {
                int(value) for value in payload.get("checked_action_units", ())
            }

        if missing:
            shown = ", ".join(
                f"{mesh_id} ({path})" for mesh_id, path in missing[:10]
            )
            suffix = "..." if len(missing) > 10 else ""
            raise FileNotFoundError(
                f"Missing {len(missing)} required validated mouth landmark "
                f"cache(s): {shown}{suffix}"
            )
        if invalid_meshes and not skip_invalid:
            shown = ", ".join(
                f"{mesh_id}({reason})"
                for mesh_id, reason in list(invalid_meshes.items())[:10]
            )
            raise ValueError(f"Invalid mouth landmark mappings: {shown}")

        original_sample_count = len(self.samples)
        unchecked: list[tuple[str, int]] = []
        kept_samples = []
        for sample in self.samples:
            if sample.mesh_id in invalid_meshes:
                continue
            checked = checked_action_units.get(sample.mesh_id)
            if checked is not None and sample.action_unit_id not in checked:
                unchecked.append((sample.mesh_id, sample.action_unit_id))
                continue
            if sample.action_unit_id in invalid_action_units.get(sample.mesh_id, set()):
                continue
            kept_samples.append(sample)
        if unchecked and require_cache:
            shown = ", ".join(
                f"{mesh_id}:AU{action_unit}" for mesh_id, action_unit in unchecked[:10]
            )
            raise ValueError(
                "Validated mouth landmark caches did not audit every requested "
                f"mesh/AU sample: {shown}"
            )
        self.samples = kept_samples
        kept_mesh_ids = {sample.mesh_id for sample in self.samples}
        self.mesh_ids = [
            mesh_id for mesh_id in self.mesh_ids if mesh_id in kept_mesh_ids
        ]
        removed_count = original_sample_count - len(self.samples)
        repaired_count = sum(
            1
            for mesh_id in self.mesh_ids
            if self._validated_mouth_landmark_status(mesh_id) == "repaired"
        )
        print(
            "[INFO] Validated MediaPipe mouth mappings before training: "
            f"identities={len(self.mesh_ids)} repaired={repaired_count} "
            f"skipped_samples={removed_count}."
        )
        if not self.samples:
            raise ValueError("Mouth landmark validation filtered every mesh sample.")

    def _validated_mouth_landmark_path(self, mesh_id: str) -> Path:
        return self.validated_mouth_landmark_dir / f"{mesh_id}.json"

    def _validated_mouth_landmark_status(self, mesh_id: str) -> str:
        path = self._validated_mouth_landmark_path(mesh_id)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return str(payload.get("status", "invalid"))

    def _load_rigged_mesh_sample(
        self,
        mesh_id: str,
        action_unit_id: int,
    ) -> Dict[str, object]:
        if self.cache_format == "hdf5":
            return _load_hdf5_rigged_mesh_sample(
                self._rigged_mesh_path(mesh_id),
                action_unit_id,
            )
        return self._load_rigged_mesh(mesh_id)

    def _load_rigged_mesh(self, mesh_id: str) -> Dict[str, object]:
        cache_path = self._rigged_mesh_path(mesh_id)
        if self.rigged_mesh_cache_size == 0:
            return _load_rigged_mesh(cache_path)

        cached = self._rigged_mesh_cache.get(cache_path)
        if cached is not None:
            self._rigged_mesh_cache.move_to_end(cache_path)
            return cached

        cached = _load_rigged_mesh(cache_path)
        self._rigged_mesh_cache[cache_path] = cached
        if self.rigged_mesh_cache_size is not None:
            while len(self._rigged_mesh_cache) > self.rigged_mesh_cache_size:
                self._rigged_mesh_cache.popitem(last=False)
        return cached

    def vertex_groups(self, mesh_id: str) -> Dict[str, Dict[str, torch.Tensor]]:
        if mesh_id not in self.mesh_paths:
            raise KeyError(f"Unknown mesh_id: {mesh_id!r}.")
        rigged_mesh = self._load_rigged_mesh(mesh_id)
        return _clone_vertex_groups(rigged_mesh.get("vertex_groups"))

    def _rigged_mesh_cache_dir(self) -> Path:
        return self.cache_dir / "rigged_meshes"

    def _rigged_mesh_path(self, mesh_id: str) -> Path:
        return _rigged_mesh_cache_path(
            self.rigged_mesh_dir,
            mesh_id,
            self.cache_format,
        )

    def _cache_missing_landmarks(self, mesh_id: str, cache_path: Path) -> None:
        rigged_mesh = _load_rigged_mesh(cache_path)
        if "landmarks_3d" in rigged_mesh and not self._should_refresh_landmarks(
            mesh_id,
            rigged_mesh["landmarks_3d"],
        ):
            return

        vertices = _expect_tensor(rigged_mesh, "vertices", cache_path)
        rigged_mesh["landmarks_3d"] = self._load_or_create_landmarks(
            mesh_id,
            vertices,
        )
        torch.save(rigged_mesh, cache_path)

    def _load_or_create_landmarks(
        self,
        mesh_id: str,
        vertices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not self.use_mediapipe_landmarks:
            return _empty_landmark_payload(vertices)

        mapping_path = self._mediapipe_landmark_path(mesh_id)
        if not mapping_path.is_file():
            if not self.generate_mediapipe_landmarks:
                return _empty_landmark_payload(vertices)
            self._generate_mediapipe_mapping(mesh_id, mapping_path)
        return _load_landmark_payload_or_regenerate(
            mapping_path,
            vertices,
            (
                lambda: self._generate_mediapipe_mapping(mesh_id, mapping_path)
            )
            if self.generate_mediapipe_landmarks
            else None,
        )

    def _mediapipe_landmark_path(self, mesh_id: str) -> Path:
        if self.mediapipe_landmark_dir is not None:
            return self.mediapipe_landmark_dir / f"{mesh_id}.json"
        return self.mesh_paths[mesh_id].with_suffix(".json")

    def _generate_mediapipe_mapping(self, mesh_id: str, mapping_path: Path) -> None:
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = _temporary_mediapipe_mapping_path(mapping_path)
        command = [
            sys.executable,
            str(self.mediapipe_mapper_path),
            "--mesh",
            str(self.mesh_paths[mesh_id]),
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
                    f"{self.mesh_paths[mesh_id]} with {self.mediapipe_mapper_path}."
                    f"\n{details}"
                )
            _publish_mediapipe_mapping(tmp_path, mapping_path)
        finally:
            _unlink_if_exists(tmp_path)

    def _should_refresh_landmarks(self, mesh_id: str, landmarks: object) -> bool:
        if not isinstance(landmarks, dict):
            return True
        vertex_ids = landmarks.get("vertex_ids")
        if not isinstance(vertex_ids, torch.Tensor):
            return True
        mapping_path = self._mediapipe_landmark_path(mesh_id)
        if mapping_path.is_file() and _is_path_newer(
            mapping_path,
            self._rigged_mesh_path(mesh_id),
        ):
            return True
        if vertex_ids.numel() > 0:
            return False
        return self.generate_mediapipe_landmarks or self._mediapipe_landmark_path(
            mesh_id,
        ).is_file()


def _normalize_mesh_id(value: str) -> str:
    mesh_id = str(value).strip()
    if mesh_id.lower().endswith(".fbx"):
        mesh_id = mesh_id[:-4]
    if not mesh_id:
        raise ValueError("mesh_id values must be non-empty.")
    return mesh_id


def _identity_fraction_subset(
    mesh_paths: Mapping[str, Path],
    *,
    fraction: float,
    seed: int,
) -> Dict[str, Path]:
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("identity_fraction must be in (0, 1].")
    if fraction == 1.0 or not mesh_paths:
        return dict(mesh_paths)
    ranked_ids = sorted(
        mesh_paths,
        key=lambda mesh_id: (
            hashlib.sha256(
                f"{int(seed)}:{_identity_fraction_group_id(mesh_id)}".encode("utf-8")
            ).digest(),
            mesh_id,
        ),
    )
    keep_count = max(1, int(len(ranked_ids) * fraction))
    selected = set(ranked_ids[:keep_count])
    return {
        mesh_id: mesh_paths[mesh_id]
        for mesh_id in sorted(mesh_paths)
        if mesh_id in selected
    }


def _identity_fraction_group_id(mesh_id: str) -> str:
    # Preserve the historical hash input across filename cleanup. Both sources
    # previously ranked identities using the ICT-style *_2_fit stem.
    try:
        return canonical_identity(mesh_id) + "_2_fit"
    except ValueError:
        pass
    suffix = "_100k_faces"
    return mesh_id[: -len(suffix)] if mesh_id.endswith(suffix) else mesh_id


def _load_excluded_mesh_ids(
    values: Optional[Sequence[str]],
    file_path: Optional[Path],
) -> set[str]:
    excluded: set[str] = set()
    if values:
        excluded.update(_normalize_mesh_id(value) for value in values)
    if file_path is not None:
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        for raw_line in file_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            excluded.add(_normalize_mesh_id(line))
    return excluded


def _normalize_action_unit_ids(
    values: Optional[Sequence[Union[int, str]]],
) -> Tuple[int, ...]:
    if values is None:
        return ACTION_UNIT_IDS
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


def _exclude_action_unit_ids(
    action_unit_ids: Tuple[int, ...],
    excluded_values: Optional[Sequence[Union[int, str]]],
) -> Tuple[int, ...]:
    if excluded_values is None:
        return action_unit_ids
    if not excluded_values:
        return action_unit_ids
    excluded = set(_normalize_action_unit_ids(excluded_values))
    if not excluded:
        return action_unit_ids
    filtered = tuple(
        action_unit_id
        for action_unit_id in action_unit_ids
        if action_unit_id not in excluded
    )
    if not filtered:
        raise ValueError("exclude_action_units removed every action unit.")
    return filtered


def _normalize_mesh_up_axis(value: str) -> str:
    axis = str(value).strip().lower()
    if axis not in {"z", "y", "auto"}:
        raise ValueError("mesh_up_axis must be one of: 'z', 'y', or 'auto'.")
    return axis


def _normalize_mesh_front_axis(value: str) -> str:
    axis = str(value).strip().lower()
    if axis.startswith("+"):
        axis = axis[1:]
    if axis not in {"y", "-y", "x", "-x"}:
        raise ValueError(
            "mesh_front_axis must be one of: 'y', '-y', 'x', or '-x'."
        )
    return axis


def _normalize_topology_augmentation_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "centroid": "face_split",
        "centroid_split": "face_split",
        "split": "face_split",
        "subdivision": "subdivide",
        "midpoint_subdivide": "subdivide",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"face_split", "subdivide"}:
        raise ValueError(
            "topology_augmentation_mode must be one of: "
            "'face_split' or 'subdivide'."
        )
    return mode


def _normalize_cache_format(value: str) -> str:
    cache_format = str(value).strip().lower().replace("-", "_")
    aliases = {
        "pt": "torch",
        "pth": "torch",
        "pytorch": "torch",
        "torch_save": "torch",
        "h5": "hdf5",
    }
    cache_format = aliases.get(cache_format, cache_format)
    if cache_format not in {"torch", "hdf5"}:
        raise ValueError("cache_format must be one of: torch, hdf5.")
    if cache_format == "hdf5" and h5py is None:
        raise ImportError("cache_format='hdf5' requires h5py to be installed.")
    return cache_format


def _normalize_blendshape_name_mapping(value: str) -> str:
    mapping = str(value).strip().lower().replace("-", "_")
    if mapping not in {"metahuman", "canonical"}:
        raise ValueError(
            "blendshape_name_mapping must be one of: metahuman, canonical."
        )
    return mapping


def _normalize_cache_compression(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    compression = str(value).strip().lower()
    if compression in {"", "none", "false", "off", "raw"}:
        return None
    if compression in {"gzip", "gz"}:
        return "gzip"
    raise ValueError("cache_compression must be one of: none, gzip.")


def _normalize_cache_compression_level(value: int) -> int:
    level = int(value)
    if level < 1 or level > 9:
        raise ValueError("cache_compression_level must be between 1 and 9.")
    return level


def _normalize_hdf5_blendshape_chunk_vertices(value: int) -> int:
    chunk_vertices = int(value)
    if chunk_vertices < 1:
        raise ValueError("hdf5_blendshape_chunk_vertices must be at least 1.")
    return chunk_vertices


def _normalize_rigged_mesh_cache_size(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    cache_size = int(value)
    if cache_size < 0:
        raise ValueError("rigged_mesh_cache_size must be non-negative or null.")
    return cache_size


def _rigged_mesh_cache_path(
    rigged_mesh_dir: Path,
    mesh_id: str,
    cache_format: str,
) -> Path:
    suffix = ".h5" if cache_format == "hdf5" else ".pt"
    return rigged_mesh_dir / f"{mesh_id}{suffix}"


def _augment_model_sample(
    mesh: Dict[str, torch.Tensor],
    delta_vertices: torch.Tensor,
    landmarks_3d: Dict[str, torch.Tensor],
    vertex_groups: object,
    drop_mouth_probability: float,
    cut_eyeballs_probability: float,
    mesh_front_axis: str,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
    vertices = mesh["vertices"]
    keep_mask = torch.ones(vertices.shape[0], dtype=torch.bool)

    if _sample_probability(drop_mouth_probability):
        keep_mask &= _keep_without_mouth_region(vertex_groups, vertices.shape[0])

    if _sample_probability(cut_eyeballs_probability):
        keep_mask &= _keep_front_half_of_eyeballs(
            vertices,
            vertex_groups,
            mesh_front_axis,
        )

    if bool(keep_mask.all().item()):
        return mesh, delta_vertices, landmarks_3d
    return _slice_model_sample(mesh, delta_vertices, landmarks_3d, keep_mask)


def _augment_topology_sample(
    mesh: Dict[str, torch.Tensor],
    delta_vertices: torch.Tensor,
    landmarks_3d: Dict[str, torch.Tensor],
    probability: float,
    mode: str,
    face_split_probability: float,
    subdivision_levels: int,
    max_vertices: Optional[int],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
    if probability <= 0.0 or not _sample_probability(probability):
        return mesh, delta_vertices, landmarks_3d
    if mesh["faces"].numel() == 0:
        return mesh, delta_vertices, landmarks_3d

    if mode == "face_split":
        return _split_faces_for_topology_augmentation(
            mesh,
            delta_vertices,
            landmarks_3d,
            face_split_probability,
            max_vertices,
        )
    if mode == "subdivide":
        return _subdivide_for_topology_augmentation(
            mesh,
            delta_vertices,
            landmarks_3d,
            subdivision_levels,
            max_vertices,
        )
    raise ValueError(f"Unsupported topology augmentation mode: {mode!r}.")


def _augment_shape_sample(
    mesh: Dict[str, torch.Tensor],
    delta_vertices: torch.Tensor,
    landmarks_3d: Dict[str, torch.Tensor],
    probability: float,
    warp_scale: float,
    num_anchors: int,
    smoothness: float,
    mesh_up_axis: str = "z",
    mesh_front_axis: str = "-y",
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
    if probability <= 0.0 or warp_scale <= 0.0 or not _sample_probability(probability):
        return mesh, delta_vertices, landmarks_3d

    vertices = mesh["vertices"]
    if vertices.shape[0] == 0:
        return mesh, delta_vertices, landmarks_3d

    shape_params = _identity_shape_warp_parameters(vertices, warp_scale)
    anchors, offsets, radius = _shape_warp_controls(
        vertices,
        warp_scale=warp_scale * 0.25,
        num_anchors=num_anchors,
        smoothness=smoothness,
    )
    target_vertices = vertices + delta_vertices
    warped_vertices = _apply_identity_shape_warp(
        vertices,
        shape_params,
        mesh_up_axis=mesh_up_axis,
        mesh_front_axis=mesh_front_axis,
    )
    warped_vertices = _apply_shape_warp(warped_vertices, anchors, offsets, radius)
    warped_target_vertices = _apply_identity_shape_warp(
        target_vertices,
        shape_params,
        mesh_up_axis=mesh_up_axis,
        mesh_front_axis=mesh_front_axis,
    )
    warped_target_vertices = _apply_shape_warp(
        warped_target_vertices,
        anchors,
        offsets,
        radius,
    )
    warped_delta = (warped_target_vertices - warped_vertices).contiguous()

    warped_mesh = dict(mesh)
    warped_mesh["vertices"] = warped_vertices.contiguous()
    warped_mesh["normals"] = _compute_vertex_normals(
        warped_mesh["vertices"],
        warped_mesh["faces"],
    )
    warped_landmarks = _refresh_landmarks_from_mesh_delta(
        landmarks_3d,
        warped_mesh["vertices"],
        warped_delta,
    )
    return warped_mesh, warped_delta, warped_landmarks


def _identity_shape_warp_parameters(
    vertices: torch.Tensor,
    warp_scale: float,
) -> Dict[str, torch.Tensor]:
    dtype = vertices.dtype

    def sample(factor: float) -> torch.Tensor:
        magnitude = vertices.new_empty(()).uniform_(0.35, 1.0)
        sign = vertices.new_tensor(1.0 if torch.rand(()).item() >= 0.5 else -1.0)
        return sign * magnitude * float(warp_scale) * float(factor)

    return {
        "head_width": sample(0.75).to(dtype),
        "face_height": sample(0.55).to(dtype),
        "face_depth": sample(0.55).to(dtype),
        "jaw_width": sample(1.35).to(dtype),
        "chin_height": sample(0.55).to(dtype),
        "forehead_width": sample(0.80).to(dtype),
        "cheek_width": sample(0.90).to(dtype),
        "neck_width": sample(1.10).to(dtype),
        "asymmetry": sample(0.20).to(dtype),
    }


def _apply_identity_shape_warp(
    positions: torch.Tensor,
    params: Mapping[str, torch.Tensor],
    mesh_up_axis: str,
    mesh_front_axis: str,
) -> torch.Tensor:
    up_axis = _resolved_mesh_up_axis(positions, mesh_up_axis)
    lateral_axis = _lateral_axis(up_axis, mesh_front_axis)
    up_idx = _axis_index(up_axis)
    lateral_idx = _axis_index(lateral_axis)
    front_idx = _axis_index(mesh_front_axis)
    front_sign = _axis_sign(mesh_front_axis)

    bounds_min = positions.amin(dim=0)
    bounds_max = positions.amax(dim=0)
    center = (bounds_min + bounds_max) * 0.5
    spans = (bounds_max - bounds_min).clamp_min(1.0e-8)
    max_span = spans.amax()

    lateral = positions[:, lateral_idx] - center[lateral_idx]
    up = positions[:, up_idx] - center[up_idx]
    front = positions[:, front_idx] * front_sign - center[front_idx] * front_sign
    lateral_norm = lateral / (spans[lateral_idx] * 0.5).clamp_min(1.0e-8)
    up_norm = up / (spans[up_idx] * 0.5).clamp_min(1.0e-8)
    front_norm = front / (spans[front_idx] * 0.5).clamp_min(1.0e-8)

    head_mask = torch.sigmoid((up_norm + 0.78) * 10.0)
    lower_face = torch.exp(-((up_norm + 0.42) / 0.32).square()) * head_mask
    chin = (
        torch.exp(-((up_norm + 0.70) / 0.20).square())
        * torch.exp(-(lateral_norm / 0.48).square())
        * head_mask
    )
    forehead = torch.exp(-((up_norm - 0.38) / 0.42).square()) * head_mask
    cheeks = (
        torch.exp(-((up_norm + 0.05) / 0.30).square())
        * torch.exp(-((lateral_norm.abs() - 0.48) / 0.32).square())
        * head_mask
    )
    neck = torch.sigmoid(-(up_norm + 0.62) * 12.0)
    face_center = (
        torch.exp(-((up_norm + 0.02) / 0.48).square())
        * torch.exp(-(lateral_norm / 0.52).square())
        * head_mask
    )

    lateral_scale = (
        params["head_width"] * head_mask
        + params["jaw_width"] * lower_face
        + params["forehead_width"] * forehead
        + params["cheek_width"] * cheeks
        + params["neck_width"] * neck
        + params["asymmetry"] * lateral_norm * head_mask
    )
    up_scale = params["face_height"] * head_mask
    front_scale = params["face_depth"] * (0.35 * head_mask + face_center)

    displacement = torch.zeros_like(positions)
    displacement[:, lateral_idx] += lateral * lateral_scale
    displacement[:, up_idx] += up * up_scale
    displacement[:, up_idx] += max_span * params["chin_height"] * 0.20 * chin
    displacement[:, front_idx] += front_sign * front * front_scale
    displacement[:, front_idx] += (
        front_sign * max_span * params["face_depth"] * 0.10 * face_center
    )
    return positions + displacement


def _axis_index(axis: str) -> int:
    axis_name = axis[-1].lower()
    if axis_name == "x":
        return 0
    if axis_name == "y":
        return 1
    if axis_name == "z":
        return 2
    raise ValueError(f"Unsupported axis: {axis!r}.")


def _axis_sign(axis: str) -> float:
    return -1.0 if str(axis).startswith("-") else 1.0


def _lateral_axis(up_axis: str, front_axis: str) -> str:
    used = {_axis_index(up_axis), _axis_index(front_axis)}
    for index, axis in enumerate(("x", "y", "z")):
        if index not in used:
            return axis
    return "x"


def _shape_warp_controls(
    vertices: torch.Tensor,
    warp_scale: float,
    num_anchors: int,
    smoothness: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vertex_count = int(vertices.shape[0])
    anchor_count = min(max(int(num_anchors), 1), vertex_count)
    if anchor_count == vertex_count:
        anchor_ids = torch.arange(vertex_count, dtype=torch.long)
    else:
        anchor_ids = torch.randperm(vertex_count)[:anchor_count].long()
    anchors = vertices.index_select(0, anchor_ids)

    spans = vertices.amax(dim=0) - vertices.amin(dim=0)
    max_span = spans.amax().clamp_min(1.0e-8)
    offsets = torch.randn(anchor_count, 3, dtype=vertices.dtype)
    offsets = offsets - offsets.mean(dim=0, keepdim=True)
    offsets = offsets * (max_span * float(warp_scale))
    radius = max_span * float(smoothness)
    return anchors, offsets, radius.clamp_min(1.0e-8)


def _apply_shape_warp(
    positions: torch.Tensor,
    anchors: torch.Tensor,
    offsets: torch.Tensor,
    radius: torch.Tensor,
) -> torch.Tensor:
    distances = torch.cdist(positions, anchors)
    weights = torch.exp(-0.5 * (distances / radius).square())
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return positions + weights.matmul(offsets)


def _refresh_landmarks_from_mesh_delta(
    landmarks: Dict[str, torch.Tensor],
    vertices: torch.Tensor,
    delta_vertices: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    vertex_ids = landmarks.get("vertex_ids")
    if not isinstance(vertex_ids, torch.Tensor) or vertex_ids.numel() == 0:
        return _clone_landmarks_metadata(landmarks)
    if int(vertex_ids.max().item()) >= vertices.shape[0]:
        return _clone_landmarks_metadata(landmarks)

    refreshed = dict(landmarks)
    neutral_positions = vertices.index_select(0, vertex_ids).contiguous()
    target_delta = delta_vertices.index_select(0, vertex_ids).contiguous()
    refreshed["neutral_positions"] = neutral_positions
    refreshed["target_delta"] = target_delta
    refreshed["target_positions"] = neutral_positions + target_delta
    return refreshed


def _clone_landmarks_metadata(
    landmarks: Mapping[str, object],
) -> Dict[str, object]:
    return {str(key): _clone_metadata_value(value) for key, value in landmarks.items()}


def _clone_metadata_value(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, Mapping):
        return {str(key): _clone_metadata_value(item) for key, item in value.items()}
    return value


def _split_faces_for_topology_augmentation(
    mesh: Dict[str, torch.Tensor],
    delta_vertices: torch.Tensor,
    landmarks_3d: Dict[str, torch.Tensor],
    face_split_probability: float,
    max_vertices: Optional[int],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
    faces = mesh["faces"]
    face_count = faces.shape[0]
    if face_count == 0 or face_split_probability <= 0.0:
        return mesh, delta_vertices, landmarks_3d

    if face_split_probability >= 1.0:
        split_indices = torch.arange(face_count, dtype=torch.long)
    else:
        split_mask = torch.rand(face_count) < float(face_split_probability)
        split_indices = torch.nonzero(split_mask, as_tuple=False).flatten().long()
    if split_indices.numel() == 0:
        return mesh, delta_vertices, landmarks_3d

    vertex_count = mesh["vertices"].shape[0]
    if max_vertices is not None:
        allowed_new_vertices = int(max_vertices) - int(vertex_count)
        if allowed_new_vertices <= 0:
            return mesh, delta_vertices, landmarks_3d
        split_indices = split_indices[:allowed_new_vertices]
        if split_indices.numel() == 0:
            return mesh, delta_vertices, landmarks_3d

    split_mask = torch.zeros(face_count, dtype=torch.bool)
    split_mask[split_indices] = True
    selected_faces = faces.index_select(0, split_indices)
    selected_vertices = mesh["vertices"].index_select(
        0,
        selected_faces.reshape(-1),
    ).reshape(-1, 3, 3)
    selected_delta = delta_vertices.index_select(
        0,
        selected_faces.reshape(-1),
    ).reshape(-1, 3, 3)
    centroid_vertices = selected_vertices.mean(dim=1)
    centroid_delta = selected_delta.mean(dim=1)
    centroid_ids = torch.arange(
        vertex_count,
        vertex_count + split_indices.numel(),
        dtype=torch.long,
    )

    a, b, c = selected_faces.unbind(dim=-1)
    split_faces = torch.stack(
        (
            torch.stack((a, b, centroid_ids), dim=-1),
            torch.stack((b, c, centroid_ids), dim=-1),
            torch.stack((c, a, centroid_ids), dim=-1),
        ),
        dim=1,
    ).reshape(-1, 3)
    updated_faces = torch.cat((faces[~split_mask], split_faces), dim=0).contiguous()
    updated_vertices = torch.cat(
        (mesh["vertices"], centroid_vertices),
        dim=0,
    ).contiguous()
    updated_delta = torch.cat((delta_vertices, centroid_delta), dim=0).contiguous()

    updated_mesh = dict(mesh)
    updated_mesh["vertices"] = updated_vertices
    updated_mesh["faces"] = updated_faces
    updated_mesh["normals"] = _compute_vertex_normals(updated_vertices, updated_faces)
    return updated_mesh, updated_delta, dict(landmarks_3d)


def _subdivide_for_topology_augmentation(
    mesh: Dict[str, torch.Tensor],
    delta_vertices: torch.Tensor,
    landmarks_3d: Dict[str, torch.Tensor],
    subdivision_levels: int,
    max_vertices: Optional[int],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
    if subdivision_levels <= 0:
        return mesh, delta_vertices, landmarks_3d

    updated_mesh = mesh
    updated_delta = delta_vertices
    for _level in range(int(subdivision_levels)):
        estimated_max_vertices = (
            updated_mesh["vertices"].shape[0] + updated_mesh["faces"].numel()
        )
        if max_vertices is not None and estimated_max_vertices > int(max_vertices):
            break
        updated_mesh, updated_delta = _subdivide_once_with_delta(
            updated_mesh,
            updated_delta,
        )
    return updated_mesh, updated_delta, dict(landmarks_3d)


def _subdivide_once_with_delta(
    mesh: Dict[str, torch.Tensor],
    delta_vertices: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    vertices = mesh["vertices"]
    faces = mesh["faces"]
    vertex_chunks = [vertices]
    delta_chunks = [delta_vertices]
    midpoint_ids: dict[tuple[int, int], int] = {}
    split_faces: list[list[int]] = []
    next_vertex_id = int(vertices.shape[0])

    def midpoint_id(first: int, second: int) -> int:
        nonlocal next_vertex_id
        key = (first, second) if first <= second else (second, first)
        cached = midpoint_ids.get(key)
        if cached is not None:
            return cached

        midpoint_ids[key] = next_vertex_id
        vertex_chunks.append(
            (vertices[first : first + 1] + vertices[second : second + 1]) * 0.5
        )
        delta_chunks.append(
            (delta_vertices[first : first + 1] + delta_vertices[second : second + 1])
            * 0.5
        )
        next_vertex_id += 1
        return midpoint_ids[key]

    for raw_face in faces.tolist():
        a, b, c = (int(raw_face[0]), int(raw_face[1]), int(raw_face[2]))
        ab = midpoint_id(a, b)
        bc = midpoint_id(b, c)
        ca = midpoint_id(c, a)
        split_faces.extend(
            (
                [a, ab, ca],
                [ab, b, bc],
                [ca, bc, c],
                [ab, bc, ca],
            )
        )

    updated_vertices = torch.cat(vertex_chunks, dim=0).contiguous()
    updated_delta = torch.cat(delta_chunks, dim=0).contiguous()
    updated_faces = faces.new_tensor(split_faces)
    updated_mesh = dict(mesh)
    updated_mesh["vertices"] = updated_vertices
    updated_mesh["faces"] = updated_faces
    updated_mesh["normals"] = _compute_vertex_normals(updated_vertices, updated_faces)
    return updated_mesh, updated_delta


def _sample_probability(probability: float) -> bool:
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    return bool(torch.rand(()).item() < probability)


def _keep_without_mouth_region(
    vertex_groups: object,
    vertex_count: int,
) -> torch.Tensor:
    keep_mask = torch.ones(vertex_count, dtype=torch.bool)
    mouth_ids = _region_vertex_ids(
        vertex_groups,
        section_names=("mouth", "teeth"),
        object_name_tokens=("mouth", "lip", "tongue", "gum", "teeth", "tooth"),
    )
    if mouth_ids.numel() == 0:
        return keep_mask
    keep_mask[mouth_ids] = False
    if not bool(keep_mask.any().item()):
        return torch.ones(vertex_count, dtype=torch.bool)
    return keep_mask


def _region_vertex_ids(
    vertex_groups: object,
    section_names: Sequence[str],
    object_name_tokens: Sequence[str],
) -> torch.Tensor:
    if not isinstance(vertex_groups, Mapping):
        return torch.empty(0, dtype=torch.long)

    collected = []
    sections = vertex_groups.get("sections")
    if isinstance(sections, Mapping):
        wanted_sections = {name.lower() for name in section_names}
        for name, vertex_ids in sections.items():
            if str(name).lower() in wanted_sections and isinstance(
                vertex_ids,
                torch.Tensor,
            ):
                collected.append(vertex_ids.detach().cpu().long().flatten())

    objects = vertex_groups.get("objects")
    if isinstance(objects, Mapping):
        tokens = tuple(token.lower() for token in object_name_tokens)
        for name, vertex_ids in objects.items():
            object_name = str(name).lower()
            if any(token in object_name for token in tokens) and isinstance(
                vertex_ids,
                torch.Tensor,
            ):
                collected.append(vertex_ids.detach().cpu().long().flatten())

    if not collected:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(collected, dim=0).unique(sorted=True).contiguous()


def _slice_model_sample(
    mesh: Dict[str, torch.Tensor],
    delta_vertices: torch.Tensor,
    landmarks_3d: Dict[str, torch.Tensor],
    keep_mask: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
    index_map = _slice_index_map(keep_mask)
    sliced_faces = _slice_faces(mesh["faces"], keep_mask, index_map)
    sliced_vertices = mesh["vertices"][keep_mask].contiguous()

    sliced_mesh = dict(mesh)
    sliced_mesh["vertices"] = sliced_vertices
    sliced_mesh["faces"] = sliced_faces
    sliced_mesh["normals"] = _compute_vertex_normals(sliced_vertices, sliced_faces)

    sliced_delta = delta_vertices[keep_mask].contiguous()
    sliced_landmarks = _slice_landmarks(landmarks_3d, keep_mask, index_map)
    return sliced_mesh, sliced_delta, sliced_landmarks


def _slice_index_map(keep_mask: torch.Tensor) -> torch.Tensor:
    index_map = torch.full((keep_mask.shape[0],), -1, dtype=torch.long)
    index_map[keep_mask] = torch.arange(int(keep_mask.sum().item()), dtype=torch.long)
    return index_map


def _slice_faces(
    faces: torch.Tensor,
    keep_mask: torch.Tensor,
    index_map: torch.Tensor,
) -> torch.Tensor:
    valid_faces = (
        keep_mask.index_select(0, faces.reshape(-1))
        .reshape_as(faces)
        .all(dim=-1)
    )
    return index_map.index_select(
        0,
        faces[valid_faces].reshape(-1),
    ).reshape(-1, 3).contiguous()


def _slice_landmarks(
    landmarks: Dict[str, torch.Tensor],
    keep_mask: torch.Tensor,
    index_map: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    vertex_ids = landmarks.get("vertex_ids")
    if not isinstance(vertex_ids, torch.Tensor) or vertex_ids.numel() == 0:
        sliced = _clone_landmarks_metadata(landmarks)
        vertex_groups = sliced.get("vertex_groups")
        if isinstance(vertex_groups, Mapping):
            sliced["vertex_groups"] = _slice_vertex_groups(
                vertex_groups,
                keep_mask,
                index_map,
            )
        return sliced

    kept_landmarks = keep_mask.index_select(0, vertex_ids)
    sliced = dict(landmarks)
    sliced["vertex_ids"] = index_map.index_select(
        0,
        vertex_ids[kept_landmarks],
    ).contiguous()

    for key in ("mediapipe_ids",):
        value = sliced.get(key)
        if isinstance(value, torch.Tensor) and value.shape[:1] == vertex_ids.shape[:1]:
            sliced[key] = value[kept_landmarks].contiguous()

    for key in ("neutral_positions", "target_delta", "target_positions"):
        value = sliced.get(key)
        if isinstance(value, torch.Tensor) and value.shape[:1] == vertex_ids.shape[:1]:
            sliced[key] = value[kept_landmarks].contiguous()

    vertex_groups = sliced.get("vertex_groups")
    if isinstance(vertex_groups, Mapping):
        sliced["vertex_groups"] = _slice_vertex_groups(
            vertex_groups,
            keep_mask,
            index_map,
        )

    return sliced


def _slice_vertex_groups(
    vertex_groups: Mapping[str, object],
    keep_mask: torch.Tensor,
    index_map: torch.Tensor,
) -> Dict[str, Dict[str, torch.Tensor]]:
    sliced: Dict[str, Dict[str, torch.Tensor]] = {}
    for namespace, groups in vertex_groups.items():
        if not isinstance(groups, Mapping):
            continue
        namespace_groups: Dict[str, torch.Tensor] = {}
        for name, vertex_ids in groups.items():
            if not isinstance(vertex_ids, torch.Tensor):
                continue
            ids = vertex_ids.detach().cpu().long().flatten()
            if ids.numel() == 0:
                namespace_groups[str(name)] = ids.clone()
                continue
            valid_ids = ids[
                (ids >= 0)
                & (ids < keep_mask.shape[0])
                & keep_mask.index_select(0, ids.clamp(0, keep_mask.shape[0] - 1))
            ]
            if valid_ids.numel() == 0:
                namespace_groups[str(name)] = torch.empty(0, dtype=torch.long)
                continue
            namespace_groups[str(name)] = (
                index_map.index_select(0, valid_ids).unique(sorted=True).contiguous()
            )
        if namespace_groups:
            sliced[str(namespace)] = namespace_groups
    return sliced


def _transform_model_sample(
    mesh: Dict[str, torch.Tensor],
    delta_vertices: torch.Tensor,
    landmarks_3d: Dict[str, torch.Tensor],
    mesh_up_axis: str,
    mesh_front_axis: str,
    normalize_on_get: bool,
    normalized_extent: float,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
    vertices = mesh["vertices"]
    up_axis = _resolved_mesh_up_axis(vertices, mesh_up_axis)

    transformed_mesh = dict(mesh)
    transformed_mesh["vertices"] = _orient_positions_to_model_frame(
        vertices,
        up_axis,
        mesh_front_axis,
    )
    transformed_mesh["normals"] = F.normalize(
        _orient_vectors_to_model_frame(mesh["normals"], up_axis, mesh_front_axis),
        dim=-1,
        eps=1.0e-8,
    )
    transformed_delta = _orient_vectors_to_model_frame(
        delta_vertices,
        up_axis,
        mesh_front_axis,
    )
    transformed_landmarks = _orient_landmarks_to_model_frame(
        landmarks_3d,
        up_axis,
        mesh_front_axis,
    )

    if normalize_on_get:
        center, scale = _normalization_center_and_scale(
            transformed_mesh["vertices"],
            normalized_extent,
        )
        transformed_mesh["vertices"] = (transformed_mesh["vertices"] - center) * scale
        transformed_delta = transformed_delta * scale
        transformed_landmarks = _normalize_landmarks(
            transformed_landmarks,
            center,
            scale,
        )
        transformed_mesh["normals"] = _compute_vertex_normals(
            transformed_mesh["vertices"],
            transformed_mesh["faces"],
        )

    return transformed_mesh, transformed_delta, transformed_landmarks


def _resolved_mesh_up_axis(vertices: torch.Tensor, mesh_up_axis: str) -> str:
    if mesh_up_axis != "auto":
        return mesh_up_axis
    spans = (vertices.amax(dim=0) - vertices.amin(dim=0)).detach()
    return "y" if float(spans[1]) > float(spans[2]) else "z"


def _orient_positions_to_model_frame(
    values: torch.Tensor,
    up_axis: str,
    front_axis: str,
) -> torch.Tensor:
    return _orient_vectors_to_model_frame(values, up_axis, front_axis)


def _orient_vectors_to_model_frame(
    values: torch.Tensor,
    up_axis: str,
    front_axis: str,
) -> torch.Tensor:
    z_up_values = _rotate_vectors_to_z_up(values, up_axis)
    return _rotate_vectors_to_y_front(z_up_values, front_axis)


def _rotate_vectors_to_z_up(values: torch.Tensor, up_axis: str) -> torch.Tensor:
    if up_axis == "z":
        return values.clone()
    if up_axis == "y":
        return torch.stack(
            (
                values[..., 0],
                -values[..., 2],
                values[..., 1],
            ),
            dim=-1,
        ).contiguous()
    raise ValueError(f"Unsupported mesh up axis: {up_axis!r}.")


def _rotate_vectors_to_y_front(values: torch.Tensor, front_axis: str) -> torch.Tensor:
    if front_axis == "y":
        return values.clone()
    if front_axis == "-y":
        return torch.stack(
            (
                -values[..., 0],
                -values[..., 1],
                values[..., 2],
            ),
            dim=-1,
        ).contiguous()
    if front_axis == "x":
        return torch.stack(
            (
                -values[..., 1],
                values[..., 0],
                values[..., 2],
            ),
            dim=-1,
        ).contiguous()
    if front_axis == "-x":
        return torch.stack(
            (
                values[..., 1],
                -values[..., 0],
                values[..., 2],
            ),
            dim=-1,
        ).contiguous()
    raise ValueError(f"Unsupported mesh front axis: {front_axis!r}.")


def _orient_landmarks_to_model_frame(
    landmarks: Dict[str, torch.Tensor],
    up_axis: str,
    front_axis: str,
) -> Dict[str, torch.Tensor]:
    rotated = dict(landmarks)
    for key in ("neutral_positions", "target_positions", "target_delta"):
        value = rotated.get(key)
        if isinstance(value, torch.Tensor) and value.shape[-1:] == (3,):
            rotated[key] = _orient_vectors_to_model_frame(value, up_axis, front_axis)
    return rotated


def _normalization_center_and_scale(
    vertices: torch.Tensor,
    normalized_extent: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bounds_min = vertices.amin(dim=0)
    bounds_max = vertices.amax(dim=0)
    center = (bounds_min + bounds_max) * 0.5
    max_span = (bounds_max - bounds_min).amax().clamp_min(1.0e-8)
    scale = vertices.new_tensor(float(normalized_extent)) / max_span
    return center, scale


def _normalize_landmarks(
    landmarks: Dict[str, torch.Tensor],
    center: torch.Tensor,
    scale: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    normalized = dict(landmarks)
    for key in ("neutral_positions", "target_positions"):
        value = normalized.get(key)
        if isinstance(value, torch.Tensor) and value.shape[-1:] == (3,):
            normalized[key] = (value - center) * scale
    target_delta = normalized.get("target_delta")
    if isinstance(target_delta, torch.Tensor) and target_delta.shape[-1:] == (3,):
        normalized["target_delta"] = target_delta * scale
    return normalized


def _cache_worker_count(configured_workers: Optional[int], mesh_count: int) -> int:
    if mesh_count <= 1:
        return 1

    raw_workers = configured_workers
    if raw_workers is None:
        env_workers = os.environ.get("TOPORIG_MESH_CACHE_WORKERS")
        raw_workers = int(env_workers) if env_workers else DEFAULT_CACHE_WORKERS

    worker_count = int(raw_workers or 1)
    if worker_count < 1:
        raise ValueError("cache_workers must be at least 1.")
    return min(worker_count, mesh_count, os.cpu_count() or 1)


def _cache_process_start_method() -> str:
    start_methods = multiprocessing.get_all_start_methods()
    if "fork" in start_methods:
        return "fork"
    return "spawn"


def _progress(iterable, *, total: int, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="mesh")


def _cache_rigged_mesh_task(task: MeshCacheTask) -> None:
    if task.cache_path.is_file():
        if _should_refresh_cache_compression_for_task(task):
            _write_rigged_mesh_cache_for_task(task)
            return
        if _should_refresh_blendshape_name_mapping_for_task(task):
            _write_rigged_mesh_cache_for_task(task)
            return
        if _should_refresh_vertex_group_cache_for_task(task):
            _write_rigged_mesh_cache_for_task(task)
            return
        if task.use_mediapipe_landmarks:
            _cache_missing_landmarks_for_task(task)
        return

    _write_rigged_mesh_cache_for_task(task)


def _should_refresh_cache_compression_for_task(task: MeshCacheTask) -> bool:
    return (
        task.cache_format == "torch"
        and task.cache_compression == "gzip"
        and not _is_gzip_file(task.cache_path)
    )


def _should_refresh_blendshape_name_mapping_for_task(task: MeshCacheTask) -> bool:
    metadata: object = None
    if task.cache_format == "hdf5" and _is_hdf5_file(task.cache_path):
        assert h5py is not None
        with h5py.File(task.cache_path, "r") as handle:
            metadata = _read_hdf5_blob(handle, "cache_metadata_blob")
    else:
        loaded = _load_rigged_mesh(task.cache_path)
        metadata = loaded.get("cache_metadata")

    cached_mapping = (
        metadata.get("blendshape_name_mapping")
        if isinstance(metadata, Mapping)
        else None
    )
    if cached_mapping is None:
        # Caches predating this field were created with the MetaHuman mapping.
        return task.blendshape_name_mapping != "metahuman"
    return str(cached_mapping) != task.blendshape_name_mapping


def _write_rigged_mesh_cache_for_task(task: MeshCacheTask) -> None:
    faces, vertices, blendshapes, metadata = _fbx_to_tensor_for_cache(task)
    landmarks_3d = _load_or_create_landmarks_for_task(task, vertices)
    rigged_mesh = _rigged_mesh_payload(
        faces=faces,
        vertices=vertices,
        blendshapes=blendshapes,
        landmarks_3d=landmarks_3d,
        vertex_groups=metadata.get("vertex_groups"),
        mesh_object_names=metadata.get("mesh_object_names"),
        cache_metadata=_cache_metadata_for_task(task),
    )
    _save_rigged_mesh(
        rigged_mesh,
        task.cache_path,
        cache_format=task.cache_format,
        compression=task.cache_compression,
        compression_level=task.cache_compression_level,
        hdf5_blendshape_chunk_vertices=task.hdf5_blendshape_chunk_vertices,
    )


def _should_refresh_vertex_group_cache_for_task(task: MeshCacheTask) -> bool:
    if task.drop_mouth_probability <= 0.0 and task.cut_eyeballs_probability <= 0.0:
        return False
    rigged_mesh = _load_rigged_mesh(task.cache_path)
    return not _has_vertex_group_metadata(rigged_mesh.get("vertex_groups"))


def _has_vertex_group_metadata(vertex_groups: object) -> bool:
    if not isinstance(vertex_groups, Mapping):
        return False
    objects = vertex_groups.get("objects")
    sections = vertex_groups.get("sections")
    return isinstance(objects, Mapping) and bool(objects) and isinstance(
        sections,
        Mapping,
    ) and bool(sections)


def _fbx_to_tensor_for_cache(
    task: MeshCacheTask,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, object]]:
    exclude_patterns = _cache_excluded_mesh_patterns(task)
    blendshape_kwargs = {}
    if task.blendshape_name_mapping == "canonical":
        blendshape_kwargs["blendshape_name_map"] = CANONICAL_BLENDSHAPE_NAMES
    try:
        loaded = fbx_to_tensor(
            task.mesh_path,
            include_metadata=True,
            exclude_mesh_name_patterns=exclude_patterns,
            **blendshape_kwargs,
        )
    except TypeError as exc:
        if "unexpected keyword" not in str(exc):
            raise
        faces, vertices, blendshapes = fbx_to_tensor(task.mesh_path)
        return faces, vertices, blendshapes, {}

    if len(loaded) != 4:
        faces, vertices, blendshapes = loaded
        return faces, vertices, blendshapes, {}
    faces, vertices, blendshapes, metadata = loaded
    return faces, vertices, blendshapes, metadata


def _cache_excluded_mesh_patterns(_task: MeshCacheTask) -> Tuple[str, ...]:
    return ()


def _keep_front_half_of_eyeballs(
    vertices: torch.Tensor,
    vertex_groups: object,
    front_axis: str,
) -> torch.Tensor:
    keep_mask = torch.ones(vertices.shape[0], dtype=torch.bool)
    if not isinstance(vertex_groups, Mapping):
        return keep_mask

    object_groups = vertex_groups.get("objects")
    if not isinstance(object_groups, Mapping):
        return keep_mask

    for object_name, vertex_ids in object_groups.items():
        if "eye" not in str(object_name).lower():
            continue
        if not isinstance(vertex_ids, torch.Tensor) or vertex_ids.numel() < 2:
            continue
        ids = vertex_ids.detach().cpu().long().flatten()
        front_values = _front_axis_values(vertices.index_select(0, ids), front_axis)
        back_to_front_order = torch.argsort(front_values, descending=False)
        remove_count = int(ids.numel()) // 2
        if remove_count <= 0:
            continue
        removed_ids = ids.index_select(0, back_to_front_order[:remove_count])
        keep_mask[removed_ids] = False
    return keep_mask


def _front_axis_values(values: torch.Tensor, front_axis: str) -> torch.Tensor:
    if front_axis == "y":
        return values[..., 1]
    if front_axis == "-y":
        return -values[..., 1]
    if front_axis == "x":
        return values[..., 0]
    if front_axis == "-x":
        return -values[..., 0]
    raise ValueError(f"Unsupported mesh front axis: {front_axis!r}.")


def _cache_metadata_for_task(task: MeshCacheTask) -> Dict[str, object]:
    excluded_patterns = _cache_excluded_mesh_patterns(task)
    return {
        "cached_full_mesh": True,
        "cache_random_seed": int(task.cache_random_seed),
        "cache_format": task.cache_format,
        "cache_compression": task.cache_compression,
        "cache_compression_level": int(task.cache_compression_level),
        "excluded_mesh_name_patterns": tuple(excluded_patterns),
        "blendshape_name_mapping": task.blendshape_name_mapping,
    }


def _cache_missing_landmarks_for_task(task: MeshCacheTask) -> None:
    if task.cache_format == "hdf5" and _is_hdf5_file(task.cache_path):
        _cache_missing_hdf5_landmarks_for_task(task)
        return

    rigged_mesh = _load_rigged_mesh(task.cache_path)
    if "landmarks_3d" in rigged_mesh and not _should_refresh_landmarks_for_task(
        task,
        rigged_mesh["landmarks_3d"],
    ):
        return

    vertices = _expect_tensor(rigged_mesh, "vertices", task.cache_path)
    rigged_mesh["landmarks_3d"] = _load_or_create_landmarks_for_task(task, vertices)
    _save_rigged_mesh(
        rigged_mesh,
        task.cache_path,
        cache_format=task.cache_format,
        compression=task.cache_compression,
        compression_level=task.cache_compression_level,
        hdf5_blendshape_chunk_vertices=task.hdf5_blendshape_chunk_vertices,
    )


def _cache_missing_hdf5_landmarks_for_task(task: MeshCacheTask) -> None:
    assert h5py is not None
    with h5py.File(task.cache_path, "r") as handle:
        vertices = _hdf5_array_to_tensor(handle["vertices"], dtype=torch.float32)
        existing_landmarks = _load_hdf5_landmarks(handle, vertices)

    if not _should_refresh_landmarks_for_task(task, existing_landmarks):
        return

    landmarks = _load_or_create_landmarks_for_task(task, vertices)
    tensor_kwargs = _hdf5_compression_kwargs(
        task.cache_compression,
        task.cache_compression_level,
    )
    with h5py.File(task.cache_path, "r+") as handle:
        tmp_group_name = f"landmarks_3d_tmp_{os.getpid()}"
        if tmp_group_name in handle:
            del handle[tmp_group_name]
        _write_hdf5_tensor_group(
            handle.create_group(tmp_group_name),
            _landmark_payload_for_cache(landmarks),
            **tensor_kwargs,
        )
        if "landmarks_3d" in handle:
            del handle["landmarks_3d"]
        handle.move(tmp_group_name, "landmarks_3d")


def _load_or_create_landmarks_for_task(
    task: MeshCacheTask,
    vertices: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if not task.use_mediapipe_landmarks:
        return _empty_landmark_payload(vertices)

    mapping_path = _mediapipe_landmark_path_for_task(task)
    if not mapping_path.is_file():
        if not task.generate_mediapipe_landmarks:
            return _empty_landmark_payload(vertices)
        _generate_mediapipe_mapping_for_task(task, mapping_path)
    return _load_landmark_payload_or_regenerate(
        mapping_path,
        vertices,
        (
            lambda: _generate_mediapipe_mapping_for_task(task, mapping_path)
        )
        if task.generate_mediapipe_landmarks
        else None,
    )


def _mediapipe_landmark_path_for_task(task: MeshCacheTask) -> Path:
    if task.mediapipe_landmark_dir is not None:
        return task.mediapipe_landmark_dir / f"{task.mesh_id}.json"
    return task.mesh_path.with_suffix(".json")


def _generate_mediapipe_mapping_for_task(
    task: MeshCacheTask,
    mapping_path: Path,
) -> None:
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _temporary_mediapipe_mapping_path(mapping_path)
    command = [
        sys.executable,
        str(task.mediapipe_mapper_path),
        "--mesh",
        str(task.mesh_path),
        "--output",
        str(tmp_path),
        *task.mediapipe_mapper_args,
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
                f"{task.mesh_path} with {task.mediapipe_mapper_path}."
                f"\n{details}"
            )
        _publish_mediapipe_mapping(tmp_path, mapping_path)
    finally:
        _unlink_if_exists(tmp_path)


def _should_refresh_landmarks_for_task(
    task: MeshCacheTask,
    landmarks: object,
) -> bool:
    if not isinstance(landmarks, dict):
        return True
    vertex_ids = landmarks.get("vertex_ids")
    if not isinstance(vertex_ids, torch.Tensor):
        return True
    mapping_path = _mediapipe_landmark_path_for_task(task)
    if mapping_path.is_file() and _is_path_newer(mapping_path, task.cache_path):
        return True
    if vertex_ids.numel() > 0:
        return False
    return task.generate_mediapipe_landmarks or _mediapipe_landmark_path_for_task(
        task,
    ).is_file()


def _is_path_newer(path: Path, reference_path: Path) -> bool:
    try:
        return path.stat().st_mtime > reference_path.stat().st_mtime
    except FileNotFoundError:
        return True


def _rigged_mesh_payload(
    faces: torch.Tensor,
    vertices: torch.Tensor,
    blendshapes: Dict[str, torch.Tensor],
    landmarks_3d: Optional[Dict[str, torch.Tensor]] = None,
    vertex_groups: Optional[object] = None,
    mesh_object_names: Optional[object] = None,
    cache_metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    vertices = vertices.detach().cpu().float().contiguous()
    faces = faces.detach().cpu().long().contiguous()
    normals = _compute_vertex_normals(vertices, faces)
    if landmarks_3d is None:
        landmarks_3d = _empty_landmark_payload(vertices)

    payload: Dict[str, object] = {
        "vertices": vertices,
        "faces": faces,
        "normals": normals,
        "landmarks_3d": _landmark_payload_for_cache(landmarks_3d),
        "blendshapes": {
            action_unit_name: blendshapes[action_unit_name]
            .detach()
            .cpu()
            .float()
            .contiguous()
            for action_unit_name in (AU_NAME[index] for index in ACTION_UNIT_IDS)
        },
    }
    payload["vertex_groups"] = _vertex_groups_for_cache(vertex_groups, vertices)
    if mesh_object_names is not None:
        payload["mesh_object_names"] = tuple(str(name) for name in mesh_object_names)
    if cache_metadata is not None:
        payload["cache_metadata"] = dict(cache_metadata)
    return payload


def _mesh_for_model(rigged_mesh: Dict[str, object]) -> Dict[str, torch.Tensor]:
    return {
        "vertices": _clone_tensor(rigged_mesh, "vertices"),
        "faces": _clone_tensor(rigged_mesh, "faces"),
        "normals": _clone_tensor(rigged_mesh, "normals"),
    }


def _delta_vertices_for_action_unit(
    rigged_mesh: Dict[str, object],
    action_unit_id: int,
) -> torch.Tensor:
    action_unit_name = AU_NAME[action_unit_id]
    blendshapes = rigged_mesh["blendshapes"]
    if not isinstance(blendshapes, dict):
        raise ValueError("Rigged mesh cache has invalid blendshapes.")

    delta_vertices = blendshapes[action_unit_name]
    if not isinstance(delta_vertices, torch.Tensor):
        raise ValueError(f"Blendshape {action_unit_name!r} is not a tensor.")
    return delta_vertices.clone()


def _landmarks_for_action_unit(
    rigged_mesh: Dict[str, object],
    delta_vertices: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    landmarks = rigged_mesh.get("landmarks_3d")
    if not isinstance(landmarks, dict):
        return _empty_landmark_payload(delta_vertices)

    payload = {
        "mediapipe_ids": _clone_landmark_tensor(
            landmarks,
            "mediapipe_ids",
            torch.long,
        ),
        "vertex_ids": _clone_landmark_tensor(landmarks, "vertex_ids", torch.long),
        "neutral_positions": _clone_landmark_tensor(
            landmarks,
            "neutral_positions",
            delta_vertices.dtype,
        ),
    }
    vertex_ids = payload["vertex_ids"]
    if vertex_ids.numel() == 0:
        payload["target_delta"] = delta_vertices.new_zeros((0, 3))
        payload["target_positions"] = delta_vertices.new_zeros((0, 3))
        return payload

    if vertex_ids.max().item() >= delta_vertices.shape[0]:
        raise ValueError("Cached MediaPipe landmark vertex id is out of range.")
    target_delta = delta_vertices.index_select(0, vertex_ids)
    payload["target_delta"] = target_delta.clone()
    payload["target_positions"] = payload["neutral_positions"] + target_delta
    return payload


def _landmark_payload_from_validated_mapping(
    rigged_mesh: Mapping[str, object],
    mapping: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    vertices = rigged_mesh.get("vertices")
    mediapipe_ids = mapping.get("mediapipe_ids")
    vertex_ids = mapping.get("vertex_ids")
    if not isinstance(vertices, torch.Tensor):
        raise ValueError("Rigged mesh vertices must be a tensor.")
    if not isinstance(mediapipe_ids, torch.Tensor) or not isinstance(
        vertex_ids,
        torch.Tensor,
    ):
        raise ValueError("Validated mouth mapping must contain landmark tensors.")
    if mediapipe_ids.shape != vertex_ids.shape or mediapipe_ids.dim() != 1:
        raise ValueError("Validated mouth landmark IDs must have matching shape [L].")
    if vertex_ids.numel() and (
        int(vertex_ids.min().item()) < 0
        or int(vertex_ids.max().item()) >= vertices.shape[0]
    ):
        raise ValueError("Validated mouth landmark vertex ID is out of range.")
    return {
        "mediapipe_ids": mediapipe_ids.clone(),
        "vertex_ids": vertex_ids.clone(),
        "neutral_positions": vertices.index_select(0, vertex_ids).clone(),
    }


def _load_rigged_mesh(path: Path) -> Dict[str, object]:
    if _is_hdf5_file(path):
        loaded = _load_hdf5_rigged_mesh(path)
    elif _is_gzip_file(path):
        with gzip.open(path, "rb") as handle:
            loaded = _torch_load_rigged_mesh(handle)
    else:
        loaded = _torch_load_rigged_mesh(path)

    if not isinstance(loaded, dict):
        raise ValueError(
            f"Expected rigged mesh cache at {path} to contain a dictionary."
        )

    required_keys = {"vertices", "faces", "normals", "blendshapes"}
    missing_keys = required_keys - set(loaded)
    if missing_keys:
        raise ValueError(
            f"Rigged mesh cache at {path} is missing keys: {sorted(missing_keys)}"
        )

    vertices = _expect_tensor(loaded, "vertices", path)
    faces = _expect_tensor(loaded, "faces", path)
    normals = _expect_tensor(loaded, "normals", path)
    blendshapes = loaded["blendshapes"]
    if not isinstance(blendshapes, dict):
        raise ValueError(f"Rigged mesh cache at {path} has invalid blendshapes.")

    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError(f"vertices in {path} must have shape [V, 3].")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(f"faces in {path} must have shape [F, 3].")
    if normals.shape != vertices.shape:
        raise ValueError(f"normals in {path} must have the same shape as vertices.")
    if "landmarks_3d" in loaded:
        _validate_landmark_payload(loaded["landmarks_3d"], vertices, path)
    if "vertex_groups" in loaded:
        _validate_vertex_groups(loaded["vertex_groups"], vertices, path)

    for action_unit_id in ACTION_UNIT_IDS:
        action_unit_name = AU_NAME[action_unit_id]
        if action_unit_name not in blendshapes:
            raise ValueError(
                f"Rigged mesh cache at {path} is missing blendshape "
                f"{action_unit_name!r}."
            )
        delta_vertices = blendshapes[action_unit_name]
        if not isinstance(delta_vertices, torch.Tensor):
            raise ValueError(
                f"Blendshape {action_unit_name!r} in {path} is not a tensor."
            )
        if delta_vertices.shape != vertices.shape:
            raise ValueError(
                f"Blendshape {action_unit_name!r} in {path} must have shape "
                f"{tuple(vertices.shape)}."
            )

    return loaded


def _save_rigged_mesh(
    rigged_mesh: Dict[str, object],
    path: Path,
    *,
    cache_format: str,
    compression: Optional[str],
    compression_level: int,
    hdf5_blendshape_chunk_vertices: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if cache_format == "hdf5":
        _save_hdf5_rigged_mesh(
            rigged_mesh,
            path,
            compression=compression,
            compression_level=compression_level,
            blendshape_chunk_vertices=hdf5_blendshape_chunk_vertices,
        )
        return
    if compression == "gzip":
        with gzip.open(path, "wb", compresslevel=compression_level) as handle:
            torch.save(rigged_mesh, handle)
        return
    torch.save(rigged_mesh, path)


def _torch_load_rigged_mesh(source: object) -> object:
    try:
        return torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(source, map_location="cpu")


def _save_hdf5_rigged_mesh(
    rigged_mesh: Dict[str, object],
    path: Path,
    *,
    compression: Optional[str],
    compression_level: int,
    blendshape_chunk_vertices: int,
) -> None:
    assert h5py is not None
    vertices = _expect_tensor(rigged_mesh, "vertices", path).detach().cpu().float()
    faces = _expect_tensor(rigged_mesh, "faces", path).detach().cpu().long()
    normals = _expect_tensor(rigged_mesh, "normals", path).detach().cpu().float()
    blendshapes = rigged_mesh.get("blendshapes")
    if not isinstance(blendshapes, Mapping):
        raise ValueError(f"Rigged mesh cache at {path} has invalid blendshapes.")

    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with h5py.File(tmp_path, "w") as handle:
            handle.attrs["schema_version"] = 1
            handle.attrs["cache_format"] = "hdf5"
            handle.attrs["action_unit_ids"] = np.asarray(ACTION_UNIT_IDS, dtype=np.int16)
            handle.attrs["action_unit_names_json"] = json.dumps(
                [AU_NAME[action_unit_id] for action_unit_id in ACTION_UNIT_IDS]
            )

            tensor_kwargs = _hdf5_compression_kwargs(compression, compression_level)
            _create_hdf5_dataset(handle, "vertices", vertices, **tensor_kwargs)
            _create_hdf5_dataset(handle, "faces", faces, **tensor_kwargs)
            _create_hdf5_dataset(handle, "normals", normals, **tensor_kwargs)

            vertex_count = int(vertices.shape[0])
            chunk_vertices = min(int(blendshape_chunk_vertices), vertex_count)
            blendshape_dataset = handle.create_dataset(
                "blendshapes",
                shape=(len(ACTION_UNIT_IDS), vertex_count, 3),
                dtype=np.float32,
                chunks=(1, chunk_vertices, 3),
                **tensor_kwargs,
            )
            for index, action_unit_id in enumerate(ACTION_UNIT_IDS):
                action_unit_name = AU_NAME[action_unit_id]
                delta_vertices = blendshapes.get(action_unit_name)
                if not isinstance(delta_vertices, torch.Tensor):
                    raise ValueError(
                        f"Blendshape {action_unit_name!r} in {path} is not a tensor."
                    )
                if delta_vertices.shape != vertices.shape:
                    raise ValueError(
                        f"Blendshape {action_unit_name!r} in {path} must have shape "
                        f"{tuple(vertices.shape)}."
                    )
                blendshape_dataset[index, :, :] = (
                    delta_vertices.detach().cpu().float().contiguous().numpy()
                )

            landmarks = rigged_mesh.get("landmarks_3d")
            if isinstance(landmarks, Mapping):
                _write_hdf5_tensor_group(
                    handle.create_group("landmarks_3d"),
                    _landmark_payload_for_cache(landmarks),
                    **tensor_kwargs,
                )

            vertex_groups = rigged_mesh.get("vertex_groups")
            if vertex_groups:
                _write_hdf5_blob(
                    handle,
                    "vertex_groups_blob",
                    vertex_groups,
                    **tensor_kwargs,
                )
            if "mesh_object_names" in rigged_mesh:
                _write_hdf5_blob(
                    handle,
                    "mesh_object_names_blob",
                    rigged_mesh["mesh_object_names"],
                    **tensor_kwargs,
                )
            if "cache_metadata" in rigged_mesh:
                _write_hdf5_blob(
                    handle,
                    "cache_metadata_blob",
                    rigged_mesh["cache_metadata"],
                    **tensor_kwargs,
                )
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _load_hdf5_rigged_mesh(path: Path) -> Dict[str, object]:
    assert h5py is not None
    with h5py.File(path, "r") as handle:
        rigged_mesh = _load_hdf5_base_mesh(handle)
        action_unit_ids = _hdf5_action_unit_ids(handle)
        blendshape_dataset = handle["blendshapes"]
        rigged_mesh["blendshapes"] = {
            AU_NAME[action_unit_id]: _hdf5_array_to_tensor(
                blendshape_dataset[index, :, :],
                dtype=torch.float32,
            )
            for index, action_unit_id in enumerate(action_unit_ids)
        }
        return rigged_mesh


def _load_hdf5_rigged_mesh_sample(
    path: Path,
    action_unit_id: int,
) -> Dict[str, object]:
    assert h5py is not None
    with h5py.File(path, "r") as handle:
        rigged_mesh = _load_hdf5_base_mesh(handle)
        action_unit_ids = _hdf5_action_unit_ids(handle)
        try:
            blendshape_index = action_unit_ids.index(int(action_unit_id))
        except ValueError as exc:
            raise ValueError(
                f"Rigged mesh HDF5 cache at {path} is missing action unit "
                f"{action_unit_id}."
            ) from exc
        delta_vertices = _hdf5_array_to_tensor(
            handle["blendshapes"][blendshape_index, :, :],
            dtype=torch.float32,
        )
        vertices = _expect_tensor(rigged_mesh, "vertices", path)
        if delta_vertices.shape != vertices.shape:
            raise ValueError(
                f"Blendshape {action_unit_id} in {path} must have shape "
                f"{tuple(vertices.shape)}."
            )
        rigged_mesh["blendshapes"] = {AU_NAME[int(action_unit_id)]: delta_vertices}
        return rigged_mesh


def _load_hdf5_base_mesh(handle: object) -> Dict[str, object]:
    path = Path(str(handle.filename))
    vertices = _hdf5_array_to_tensor(handle["vertices"], dtype=torch.float32)
    rigged_mesh: Dict[str, object] = {
        "vertices": vertices,
        "faces": _hdf5_array_to_tensor(handle["faces"], dtype=torch.long),
        "normals": _hdf5_array_to_tensor(handle["normals"], dtype=torch.float32),
        "landmarks_3d": _load_hdf5_landmarks(handle, vertices),
    }
    vertex_groups = _read_hdf5_blob(handle, "vertex_groups_blob")
    if vertex_groups is not None:
        _validate_vertex_groups(vertex_groups, vertices, path)
        rigged_mesh["vertex_groups"] = vertex_groups
    mesh_object_names = _read_hdf5_blob(handle, "mesh_object_names_blob")
    if mesh_object_names is not None:
        rigged_mesh["mesh_object_names"] = mesh_object_names
    cache_metadata = _read_hdf5_blob(handle, "cache_metadata_blob")
    if cache_metadata is not None:
        rigged_mesh["cache_metadata"] = cache_metadata
    return rigged_mesh


def _hdf5_action_unit_ids(handle: object) -> list[int]:
    if "action_unit_ids" not in handle.attrs:
        return list(ACTION_UNIT_IDS)
    return [int(value) for value in handle.attrs["action_unit_ids"]]


def _load_hdf5_landmarks(
    handle: object,
    vertices: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if "landmarks_3d" not in handle:
        return _empty_landmark_payload(vertices)
    group = handle["landmarks_3d"]
    landmarks = {
        "mediapipe_ids": _hdf5_array_to_tensor(
            group["mediapipe_ids"],
            dtype=torch.long,
        ),
        "vertex_ids": _hdf5_array_to_tensor(group["vertex_ids"], dtype=torch.long),
        "neutral_positions": _hdf5_array_to_tensor(
            group["neutral_positions"],
            dtype=torch.float32,
        ),
    }
    _validate_landmark_payload(landmarks, vertices, Path(str(handle.filename)))
    return landmarks


def _write_hdf5_tensor_group(group: object, payload: Mapping[str, torch.Tensor], **kwargs) -> None:
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            _create_hdf5_dataset(group, key, value, **kwargs)


def _create_hdf5_dataset(group: object, name: str, value: torch.Tensor, **kwargs) -> None:
    group.create_dataset(
        name,
        data=value.detach().cpu().contiguous().numpy(),
        **kwargs,
    )


def _write_hdf5_blob(group: object, name: str, value: object, **kwargs) -> None:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    group.create_dataset(name, data=data, **kwargs)


def _read_hdf5_blob(group: object, name: str) -> object:
    if name not in group:
        return None
    data = np.asarray(group[name][()], dtype=np.uint8).tobytes()
    buffer = io.BytesIO(data)
    try:
        return torch.load(buffer, map_location="cpu", weights_only=False)
    except TypeError:
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu")


def _hdf5_array_to_tensor(value: object, *, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(value[()])).to(dtype=dtype)
    return tensor.contiguous()


def _hdf5_compression_kwargs(
    compression: Optional[str],
    compression_level: int,
) -> Dict[str, object]:
    if compression is None:
        return {}
    if compression != "gzip":
        raise ValueError("HDF5 mesh cache only supports gzip compression.")
    return {"compression": "gzip", "compression_opts": int(compression_level)}


def _is_gzip_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(2) == b"\x1f\x8b"
    except FileNotFoundError:
        raise
    except OSError:
        return False


def _is_hdf5_file(path: Path) -> bool:
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return True
    try:
        with path.open("rb") as handle:
            return handle.read(8) == b"\x89HDF\r\n\x1a\n"
    except FileNotFoundError:
        raise
    except OSError:
        return False


def _temporary_mediapipe_mapping_path(mapping_path: Path) -> Path:
    return mapping_path.with_name(
        f".{mapping_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )


def _publish_mediapipe_mapping(tmp_path: Path, mapping_path: Path) -> None:
    if not tmp_path.is_file():
        raise RuntimeError(
            "MediaPipe landmark mapper finished but did not write "
            f"{tmp_path}."
        )
    with tmp_path.open("r", encoding="utf-8") as handle:
        json.load(handle)
    os.replace(tmp_path, mapping_path)


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _load_landmark_payload_or_regenerate(
    mapping_path: Path,
    vertices: torch.Tensor,
    regenerate: Optional[Callable[[], None]] = None,
) -> Dict[str, torch.Tensor]:
    try:
        return _load_landmark_payload(mapping_path, vertices)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        if regenerate is None:
            raise
        print(
            "[WARN] Failed to load MediaPipe landmark mapping "
            f"{mapping_path}: {type(exc).__name__}: {exc}. Regenerating.",
            file=sys.stderr,
            flush=True,
        )
        regenerate()
        return _load_landmark_payload(mapping_path, vertices)


def _load_landmark_payload(
    mapping_path: Path,
    vertices: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    with mapping_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    if isinstance(loaded, dict) and isinstance(loaded.get("mapping"), dict):
        mapping = loaded["mapping"]
    elif isinstance(loaded, dict):
        mapping = loaded
    else:
        raise ValueError(f"Expected {mapping_path} to contain a landmark mapping.")

    landmark_pairs: list[tuple[int, int]] = []
    skipped_out_of_range = 0
    for raw_mediapipe_id, raw_vertex_id in mapping.items():
        if raw_vertex_id is None:
            continue
        mediapipe_id = int(raw_mediapipe_id)
        vertex_id = int(raw_vertex_id)
        if vertex_id < 0 or vertex_id >= vertices.shape[0]:
            skipped_out_of_range += 1
            continue
        landmark_pairs.append((mediapipe_id, vertex_id))
    if skipped_out_of_range:
        print(
            f"[WARN] Skipped {skipped_out_of_range} out-of-range MediaPipe "
            f"landmark(s) in {mapping_path}.",
            file=sys.stderr,
        )

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
    }


def _empty_landmark_payload(reference: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "mediapipe_ids": torch.empty(0, dtype=torch.long),
        "vertex_ids": torch.empty(0, dtype=torch.long),
        "neutral_positions": torch.empty(0, 3, dtype=torch.float32),
        "target_delta": torch.empty(0, 3, dtype=torch.float32),
        "target_positions": torch.empty(0, 3, dtype=torch.float32),
    }


def _landmark_payload_for_cache(
    landmarks: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        "mediapipe_ids": landmarks["mediapipe_ids"].detach().cpu().long().contiguous(),
        "vertex_ids": landmarks["vertex_ids"].detach().cpu().long().contiguous(),
        "neutral_positions": landmarks["neutral_positions"]
        .detach()
        .cpu()
        .float()
        .contiguous(),
    }


def _vertex_groups_for_cache(
    vertex_groups: Optional[object],
    vertices: torch.Tensor,
) -> Dict[str, Dict[str, torch.Tensor]]:
    if not isinstance(vertex_groups, Mapping):
        return {}

    cached: Dict[str, Dict[str, torch.Tensor]] = {}
    for namespace, groups in vertex_groups.items():
        if not isinstance(groups, Mapping):
            continue
        namespace_groups: Dict[str, torch.Tensor] = {}
        for name, vertex_ids in groups.items():
            if not isinstance(vertex_ids, torch.Tensor):
                continue
            ids = vertex_ids.detach().cpu().long().flatten().unique(sorted=True)
            if ids.numel() > 0 and (
                int(ids.min().item()) < 0 or int(ids.max().item()) >= vertices.shape[0]
            ):
                raise ValueError(
                    f"vertex group {namespace}.{name} contains out-of-range ids "
                    f"for a mesh with {vertices.shape[0]} vertices."
                )
            namespace_groups[str(name)] = ids.contiguous()
        if namespace_groups:
            cached[str(namespace)] = namespace_groups
    return cached


def _clone_vertex_groups(vertex_groups: object) -> Dict[str, Dict[str, torch.Tensor]]:
    if not isinstance(vertex_groups, Mapping):
        return {}
    cloned: Dict[str, Dict[str, torch.Tensor]] = {}
    for namespace, groups in vertex_groups.items():
        if not isinstance(groups, Mapping):
            continue
        namespace_groups = {
            str(name): vertex_ids.clone()
            for name, vertex_ids in groups.items()
            if isinstance(vertex_ids, torch.Tensor)
        }
        if namespace_groups:
            cloned[str(namespace)] = namespace_groups
    return cloned


def _validate_landmark_payload(
    landmarks: object,
    vertices: torch.Tensor,
    path: Path,
) -> None:
    if not isinstance(landmarks, dict):
        raise ValueError(f"landmarks_3d in {path} must be a dictionary.")
    mediapipe_ids = _expect_landmark_tensor(landmarks, "mediapipe_ids", path)
    vertex_ids = _expect_landmark_tensor(landmarks, "vertex_ids", path)
    neutral_positions = _expect_landmark_tensor(landmarks, "neutral_positions", path)

    if mediapipe_ids.ndim != 1 or vertex_ids.ndim != 1:
        raise ValueError(f"landmark ids in {path} must have shape [L].")
    if mediapipe_ids.shape != vertex_ids.shape:
        raise ValueError(f"landmark id tensors in {path} must have matching shapes.")
    if neutral_positions.shape != (vertex_ids.numel(), 3):
        raise ValueError(
            f"neutral landmark positions in {path} must have shape [L, 3]."
        )
    if vertex_ids.numel() > 0:
        if int(vertex_ids.min()) < 0 or int(vertex_ids.max()) >= vertices.shape[0]:
            raise ValueError(f"landmark vertex ids in {path} are out of range.")


def _validate_vertex_groups(
    vertex_groups: object,
    vertices: torch.Tensor,
    path: Path,
) -> None:
    if not isinstance(vertex_groups, Mapping):
        raise ValueError(f"vertex_groups in {path} must be a mapping.")
    for namespace, groups in vertex_groups.items():
        if not isinstance(groups, Mapping):
            raise ValueError(f"vertex_groups.{namespace} in {path} must be a mapping.")
        for name, vertex_ids in groups.items():
            if not isinstance(vertex_ids, torch.Tensor):
                raise ValueError(
                    f"vertex group {namespace}.{name} in {path} must be a tensor."
                )
            if vertex_ids.ndim != 1:
                raise ValueError(
                    f"vertex group {namespace}.{name} in {path} must have shape [N]."
                )
            if vertex_ids.numel() == 0:
                continue
            if int(vertex_ids.min().item()) < 0 or int(vertex_ids.max().item()) >= (
                vertices.shape[0]
            ):
                raise ValueError(
                    f"vertex group {namespace}.{name} in {path} is out of range."
                )


def _expect_tensor(
    payload: Dict[str, object],
    key: str,
    path: Path,
) -> torch.Tensor:
    value = payload[key]
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{key} in {path} must be a tensor.")
    return value


def _expect_landmark_tensor(
    payload: Mapping[str, object],
    key: str,
    path: Path,
) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"landmarks_3d.{key} in {path} must be a tensor.")
    return value


def _clone_tensor(payload: Dict[str, object], key: str) -> torch.Tensor:
    value = payload[key]
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{key} must be a tensor.")
    return value.clone()


def _clone_landmark_tensor(
    payload: Mapping[str, object],
    key: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"landmarks_3d.{key} must be a tensor.")
    return value.to(dtype=dtype).clone()


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
    return F.normalize(normals, dim=-1, eps=1.0e-6).contiguous()
