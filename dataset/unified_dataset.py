from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from dataset.image_dataset import ImageDataset
from dataset.mesh_dataset import MeshDataset
from dataset.layout import canonical_identity


DatasetArgs = Optional[Mapping[str, Any]]
TensorDict = Mapping[str, torch.Tensor]


class UnifiedDataset(Dataset):
    def __init__(
        self,
        image_args: DatasetArgs = None,
        mesh_args: DatasetArgs = None,
        extra_mesh_args: Optional[Any] = None,
        split: str = "all",
        split_csv: Optional[Any] = None,
        pair_image_with_mesh_by_action_unit: bool = False,
        data_root: Optional[Any] = None,
        asset_cache_dir: Optional[Any] = None,
    ) -> None:
        extra_mesh_specs = _extra_mesh_specs(extra_mesh_args)
        def with_shared_root(args):
            if args is None:
                return None
            args = dict(args)
            for key, value in (("data_root", data_root), ("asset_cache_dir", asset_cache_dir)):
                if value is not None:
                    args.setdefault(key, value)
            return args
        image_args, mesh_args = with_shared_root(image_args), with_shared_root(mesh_args)
        for spec in extra_mesh_specs:
            spec["args"] = with_shared_root(spec["args"])
        if image_args is None and mesh_args is None and not extra_mesh_specs:
            raise ValueError(
                "At least one of image_args, mesh_args, or extra_mesh_args "
                "must be provided."
            )

        self.image_dataset = (
            ImageDataset(**_dataset_args(image_args, split, split_csv))
            if image_args is not None
            else None
        )
        self.mesh_dataset = (
            MeshDataset(**_dataset_args(mesh_args, split, split_csv))
            if mesh_args is not None
            else None
        )
        self.extra_mesh_datasets = [
            {
                "name": spec["name"],
                "loss_weight_key": spec["loss_weight_key"],
                "dataset": MeshDataset(**_dataset_args(spec["args"], split, split_csv)),
            }
            for spec in extra_mesh_specs
        ]
        self.image_length = (
            len(self.image_dataset) if self.image_dataset is not None else 0
        )
        self.mesh_length = (
            len(self.mesh_dataset) if self.mesh_dataset is not None else 0
        )
        self.extra_mesh_lengths = [
            len(spec["dataset"]) for spec in self.extra_mesh_datasets
        ]
        self.extra_mesh_total_length = sum(self.extra_mesh_lengths)
        self._image_reference: Optional[dict[str, Any]] = None
        self._mesh_reference: Optional[dict[str, Any]] = None
        self._visualization_mesh_target_index: Optional[
            dict[tuple[str, int], list[tuple[Dataset, int]]]
        ] = None
        self.pair_image_with_mesh_by_action_unit = (
            bool(pair_image_with_mesh_by_action_unit)
            and self.image_dataset is not None
            and self.mesh_dataset is not None
        )
        self._paired_image_indices: list[Optional[int]] = []
        self._unpaired_image_indices: list[int] = []
        if self.pair_image_with_mesh_by_action_unit:
            self._prepare_action_unit_pairs()
            self._paired_length = self.mesh_length + len(self._unpaired_image_indices)
            self._length = self._paired_length + self.extra_mesh_total_length
        else:
            self._paired_length = 0
            self._length = (
                self.image_length + self.mesh_length + self.extra_mesh_total_length
            )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._length == 0:
            raise IndexError(index)
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)

        if self.pair_image_with_mesh_by_action_unit and index < self._paired_length:
            return self._paired_item(index)
        if self.pair_image_with_mesh_by_action_unit:
            return self._extra_mesh_item(index - self._paired_length)

        if index < self.image_length:
            assert self.image_dataset is not None
            image = _image_record(self.image_dataset[index])
            return {
                "image": image,
                "mesh": self._zero_mesh_record(image["mesh"]),
                "is_img": True,
                "is_mesh": False,
            }

        mesh_index = index - self.image_length
        if mesh_index < self.mesh_length:
            assert self.mesh_dataset is not None
            mesh = _mesh_record(self.mesh_dataset[mesh_index])
            return {
                "image": self._zero_image_record(mesh["mesh"]),
                "mesh": mesh,
                "is_img": False,
                "is_mesh": True,
                "mesh_stream": "primary",
                "mesh_loss_weight_key": "mesh_3d",
            }
        return self._extra_mesh_item(mesh_index - self.mesh_length)

    def _extra_mesh_item(self, index: int) -> dict[str, Any]:
        for spec, length in zip(self.extra_mesh_datasets, self.extra_mesh_lengths):
            if index >= length:
                index -= length
                continue
            dataset = spec["dataset"]
            mesh = _mesh_record(dataset[index])
            return {
                "image": self._zero_image_record(mesh["mesh"]),
                "mesh": mesh,
                "is_img": False,
                "is_mesh": True,
                "mesh_stream": spec["name"],
                "mesh_loss_weight_key": spec["loss_weight_key"],
            }
        raise IndexError(index)

    def _primary_mesh_item(
        self,
        mesh: dict[str, Any],
        image: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if image is None:
            image = self._zero_image_record(mesh["mesh"])
        return {
            "image": image,
            "mesh": mesh,
            "is_img": image.get("action_unit_id") != -1,
            "is_mesh": True,
            "mesh_stream": "primary",
            "mesh_loss_weight_key": "mesh_3d",
        }

    def _prepare_action_unit_pairs(self) -> None:
        assert self.image_dataset is not None
        assert self.mesh_dataset is not None

        image_indices_by_au: dict[int, list[int]] = defaultdict(list)
        for image_index in range(self.image_length):
            action_unit_id = _dataset_action_unit_id(self.image_dataset, image_index)
            image_indices_by_au[action_unit_id].append(image_index)

        used_image_indices: set[int] = set()
        next_image_offset_by_au: dict[int, int] = defaultdict(int)
        for mesh_index in range(self.mesh_length):
            action_unit_id = _dataset_action_unit_id(self.mesh_dataset, mesh_index)
            image_indices = image_indices_by_au.get(action_unit_id, ())
            if not image_indices:
                self._paired_image_indices.append(None)
                continue
            offset = next_image_offset_by_au[action_unit_id] % len(image_indices)
            image_index = image_indices[offset]
            next_image_offset_by_au[action_unit_id] += 1
            used_image_indices.add(image_index)
            self._paired_image_indices.append(image_index)

        self._unpaired_image_indices = [
            image_index
            for image_index in range(self.image_length)
            if image_index not in used_image_indices
        ]

    def _paired_item(self, index: int) -> dict[str, Any]:
        assert self.image_dataset is not None
        assert self.mesh_dataset is not None

        if index < self.mesh_length:
            mesh = _mesh_record(self.mesh_dataset[index])
            image_index = self._paired_image_indices[index]
            if image_index is None:
                return self._primary_mesh_item(mesh)
            image = _image_record(self.image_dataset[image_index])
            return self._primary_mesh_item(mesh, image=image)

        image_index = self._unpaired_image_indices[index - self.mesh_length]
        image = _image_record(self.image_dataset[image_index])
        return {
            "image": image,
            "mesh": self._zero_mesh_record(image["mesh"]),
            "is_img": True,
            "is_mesh": False,
        }

    def _zero_image_record(self, fallback_mesh: TensorDict) -> dict[str, Any]:
        reference = self._image_reference_record()
        if reference is None:
            return {
                "mesh": _zero_tensor_dict_like(fallback_mesh),
                "action_unit_id": -1,
                "displacement_2d": _empty_displacement_2d(),
                "landmarks_3d": _empty_landmarks_3d(),
            }
        return _zero_image_record_like(reference)

    def _zero_mesh_record(self, fallback_mesh: TensorDict) -> dict[str, Any]:
        reference = self._mesh_reference_record()
        if reference is None:
            return {
                "mesh": _zero_tensor_dict_like(fallback_mesh),
                "action_unit_id": -1,
                "delta_vertices": _zero_vertices_like(fallback_mesh),
                "landmarks_3d": _empty_landmarks_3d(),
            }
        return _zero_mesh_record_like(reference)

    def _image_reference_record(self) -> Optional[dict[str, Any]]:
        if self.image_dataset is None or self.image_length == 0:
            return None
        if self._image_reference is None:
            self._image_reference = _image_record(self.image_dataset[0])
        return self._image_reference

    def _mesh_reference_record(self) -> Optional[dict[str, Any]]:
        if self.mesh_dataset is None or self.mesh_length == 0:
            for spec in self.extra_mesh_datasets:
                dataset = spec["dataset"]
                if len(dataset) > 0:
                    return _mesh_record(dataset[0])
            return None
        if self._mesh_reference is None:
            self._mesh_reference = _mesh_record(self.mesh_dataset[0])
        return self._mesh_reference

    def visualization_target_candidate_records(
        self,
        base_index: int,
        action_units: Sequence[int],
    ) -> Optional[dict[int, Iterator[dict[str, Any]]]]:
        source = self._visualization_record_source(base_index)
        if source is None:
            return None
        source_dataset, source_index = source
        identity = _dataset_sample_identity(source_dataset, source_index)
        if identity is None:
            return None

        target_index = self._visualization_mesh_targets()
        candidates: dict[int, Iterator[dict[str, Any]]] = {}
        mesh_sources = self._visualization_mesh_sources()
        if source_dataset in mesh_sources:
            mesh_sources = [source_dataset] + [
                dataset for dataset in mesh_sources if dataset is not source_dataset
            ]
        elif source_dataset is self.image_dataset:
            mesh_sources = [
                spec["dataset"] for spec in self.extra_mesh_datasets
            ] + ([self.mesh_dataset] if self.mesh_dataset is not None else [])
        source_order = {
            id(dataset): index for index, dataset in enumerate(mesh_sources)
        }

        for action_unit in action_units:
            references = list(target_index.get((identity, int(action_unit)), ()))
            references.sort(
                key=lambda reference: source_order.get(
                    id(reference[0]), len(source_order)
                )
            )
            if references:
                candidates[int(action_unit)] = _mesh_record_candidates(references)
        return candidates

    def _visualization_record_source(
        self,
        index: int,
    ) -> Optional[tuple[Dataset, int]]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)

        if self.pair_image_with_mesh_by_action_unit:
            if index < self.mesh_length:
                assert self.mesh_dataset is not None
                return self.mesh_dataset, index
            if index < self._paired_length:
                assert self.image_dataset is not None
                image_index = self._unpaired_image_indices[index - self.mesh_length]
                return self.image_dataset, image_index
            return self._visualization_extra_mesh_source(index - self._paired_length)

        if index < self.image_length:
            assert self.image_dataset is not None
            return self.image_dataset, index
        index -= self.image_length
        if index < self.mesh_length:
            assert self.mesh_dataset is not None
            return self.mesh_dataset, index
        return self._visualization_extra_mesh_source(index - self.mesh_length)

    def _visualization_extra_mesh_source(
        self,
        index: int,
    ) -> Optional[tuple[Dataset, int]]:
        for spec, length in zip(self.extra_mesh_datasets, self.extra_mesh_lengths):
            if index < length:
                return spec["dataset"], index
            index -= length
        return None

    def _visualization_mesh_sources(self) -> list[Dataset]:
        sources = []
        if self.mesh_dataset is not None:
            sources.append(self.mesh_dataset)
        sources.extend(spec["dataset"] for spec in self.extra_mesh_datasets)
        return sources

    def _visualization_mesh_targets(
        self,
    ) -> dict[tuple[str, int], list[tuple[Dataset, int]]]:
        if self._visualization_mesh_target_index is not None:
            return self._visualization_mesh_target_index

        target_index: dict[tuple[str, int], list[tuple[Dataset, int]]] = defaultdict(
            list
        )
        for dataset in self._visualization_mesh_sources():
            samples = getattr(dataset, "samples", None)
            if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
                continue
            for index, sample in enumerate(samples):
                identity = _sample_identity(sample)
                action_unit = _sample_action_unit_id(sample)
                if identity is None or action_unit is None:
                    continue
                target_index[(identity, action_unit)].append((dataset, index))
        self._visualization_mesh_target_index = dict(target_index)
        return self._visualization_mesh_target_index


def _dataset_args(
    dataset_args: Mapping[str, Any],
    split: str,
    split_csv: Optional[Any],
) -> dict[str, Any]:
    args = dict(dataset_args)
    args.setdefault("split", split)
    if split_csv is not None:
        args.setdefault("split_csv", split_csv)
    return args


def _extra_mesh_specs(extra_mesh_args: Optional[Any]) -> list[dict[str, Any]]:
    if extra_mesh_args is None:
        return []
    if isinstance(extra_mesh_args, Mapping):
        if "mesh_dir" in extra_mesh_args:
            raw_specs: list[tuple[str, Mapping[str, Any]]] = [
                ("extra_mesh", extra_mesh_args)
            ]
        else:
            raw_specs = [
                (str(name), args)
                for name, args in extra_mesh_args.items()
                if not (
                    isinstance(args, Mapping)
                    and bool(args.get("enabled", True)) is False
                )
            ]
    elif isinstance(extra_mesh_args, Sequence) and not isinstance(
        extra_mesh_args,
        (str, bytes),
    ):
        raw_specs = []
        for index, args in enumerate(extra_mesh_args):
            if not isinstance(args, Mapping):
                raise ValueError("extra_mesh_args entries must be mappings.")
            if bool(args.get("enabled", True)) is False:
                continue
            raw_specs.append((str(args.get("name", f"extra_mesh_{index}")), args))
    else:
        raise ValueError("extra_mesh_args must be a mapping or list of mappings.")

    specs: list[dict[str, Any]] = []
    for name, args in raw_specs:
        if not isinstance(args, Mapping):
            raise ValueError("extra_mesh_args entries must be mappings.")
        dataset_args = dict(args)
        dataset_args.pop("enabled", None)
        stream_name = str(dataset_args.pop("name", name))
        loss_weight_key = str(
            dataset_args.pop("loss_weight_key", f"{stream_name}_mesh_3d")
        )
        specs.append(
            {
                "name": stream_name,
                "loss_weight_key": loss_weight_key,
                "args": dataset_args,
            }
        )
    return specs


def _image_record(sample: tuple[TensorDict, int, TensorDict]) -> dict[str, Any]:
    if len(sample) == 3:
        mesh, action_unit_id, displacement_2d = sample
        landmarks_3d = _empty_landmarks_3d()
    elif len(sample) == 4:
        mesh, action_unit_id, displacement_2d, landmarks_3d = sample
    else:
        raise ValueError("ImageDataset samples must have 3 or 4 fields.")

    return {
        "mesh": mesh,
        "action_unit_id": action_unit_id,
        "displacement_2d": displacement_2d,
        "landmarks_3d": landmarks_3d,
    }


def _mesh_record(sample: tuple[Any, ...]) -> dict[str, Any]:
    if len(sample) == 3:
        mesh, action_unit_id, delta_vertices = sample
        landmarks_3d = _empty_landmarks_3d()
    elif len(sample) == 4:
        mesh, action_unit_id, delta_vertices, landmarks_3d = sample
    else:
        raise ValueError("MeshDataset samples must have 3 or 4 fields.")

    record = {
        "mesh": mesh,
        "action_unit_id": action_unit_id,
        "delta_vertices": delta_vertices,
        "landmarks_3d": landmarks_3d,
    }
    return record


def _dataset_action_unit_id(dataset: Dataset, index: int) -> int:
    samples = getattr(dataset, "samples", None)
    if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes)):
        sample = samples[index]
        action_unit_id = _sample_action_unit_id(sample)
        if action_unit_id is not None:
            return action_unit_id

    sample = dataset[index]
    action_unit_id = _sample_action_unit_id(sample)
    if action_unit_id is None:
        raise ValueError(
            "Dataset samples must expose an action_unit_id to pair image and "
            "mesh samples."
        )
    return action_unit_id


def _sample_action_unit_id(sample: Any) -> Optional[int]:
    if hasattr(sample, "action_unit_id"):
        return int(getattr(sample, "action_unit_id"))
    if isinstance(sample, Mapping) and "action_unit_id" in sample:
        return int(sample["action_unit_id"])
    if isinstance(sample, Sequence) and not isinstance(sample, (str, bytes)):
        if len(sample) > 1:
            return int(sample[1])
    return None


def _mesh_record_candidates(
    references: Sequence[tuple[Dataset, int]],
) -> Iterator[dict[str, Any]]:
    for dataset, index in references:
        yield _mesh_record(dataset[index])


def _dataset_sample_identity(dataset: Dataset, index: int) -> Optional[str]:
    samples = getattr(dataset, "samples", None)
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        return None
    return _sample_identity(samples[index])


def _sample_identity(sample: Any) -> Optional[str]:
    for name in ("person_id", "mesh_id"):
        if hasattr(sample, name):
            return _canonical_visualization_identity(str(getattr(sample, name)))
        if isinstance(sample, Mapping) and name in sample:
            return _canonical_visualization_identity(str(sample[name]))
    return None


def _canonical_visualization_identity(value: str) -> str:
    try:
        return canonical_identity(value)
    except ValueError:
        return value  # Arbitrary user-provided mesh names are still supported.


def _zero_image_record_like(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mesh": _zero_tensor_dict_like(record["mesh"]),
        "action_unit_id": -1,
        "displacement_2d": _zero_tensor_dict_like(record["displacement_2d"]),
        "landmarks_3d": _zero_landmarks_3d_like(record.get("landmarks_3d")),
    }


def _zero_mesh_record_like(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mesh": _zero_tensor_dict_like(record["mesh"]),
        "action_unit_id": -1,
        "delta_vertices": torch.zeros_like(record["delta_vertices"]),
        "landmarks_3d": _zero_landmarks_3d_like(record.get("landmarks_3d")),
    }


def _zero_tensor_dict_like(tensors: TensorDict) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros_like(tensor)
        for name, tensor in tensors.items()
        if isinstance(tensor, torch.Tensor)
    }


def _zero_vertices_like(mesh: TensorDict) -> torch.Tensor:
    vertices = mesh.get("vertices")
    if vertices is None:
        return torch.zeros(0)
    return torch.zeros_like(vertices)


def _empty_displacement_2d() -> dict[str, torch.Tensor]:
    return {
        "displacement": torch.zeros(0),
        "mask": torch.zeros(0),
    }


def _empty_landmarks_3d() -> dict[str, torch.Tensor]:
    return {
        "mediapipe_ids": torch.empty(0, dtype=torch.long),
        "vertex_ids": torch.empty(0, dtype=torch.long),
        "neutral_positions": torch.empty(0, 3),
        "target_delta": torch.empty(0, 3),
        "target_positions": torch.empty(0, 3),
    }


def _zero_landmarks_3d_like(landmarks: Any) -> dict[str, torch.Tensor]:
    if not isinstance(landmarks, Mapping):
        return _empty_landmarks_3d()
    return {
        name: torch.zeros_like(tensor)
        for name, tensor in landmarks.items()
        if isinstance(tensor, torch.Tensor)
    }
