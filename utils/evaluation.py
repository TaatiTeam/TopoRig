"""Manifest records and report helpers for the main TopoRig evaluation."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence

from dataset.layout import canonical_identity


STANDARD_HEAD_HEIGHT_MM = 240.0
PAIRED_MANIFEST_FIELDS = (
    "index", "dataset_index", "pair_id", "paired_test_identity", "dataset",
    "mesh_id", "fbx_path", "source_split",
)


@dataclass(frozen=True)
class PairedRecord:
    index: int
    dataset_index: int
    pair_id: str
    paired_test_identity: bool
    dataset: str
    mesh_id: str
    fbx_path: Path
    source_split: str


def read_paired_manifest(path: Path) -> list[PairedRecord]:
    """Read recorded membership without selecting or reshuffling identities.

    File paths may be relative to the manifest. Evaluation resolves files from
    its explicit collection roots, so historical absolute paths remain provenance.
    """
    path = path.expanduser().resolve()
    records = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not set(PAIRED_MANIFEST_FIELDS).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} must contain: {', '.join(PAIRED_MANIFEST_FIELDS)}")
        for row in reader:
            identity = canonical_identity(row["mesh_id"])
            mesh = Path(row["fbx_path"]).expanduser()
            mesh = mesh if mesh.is_absolute() else path.parent / mesh
            paired = row["paired_test_identity"].strip().lower()
            if paired not in {"true", "false"}:
                raise ValueError(f"Invalid paired_test_identity in {path}")
            records.append(PairedRecord(
                index=int(row["index"]), dataset_index=int(row["dataset_index"]),
                pair_id=row["pair_id"].strip(), paired_test_identity=paired == "true",
                dataset=row["dataset"].strip().lower(), mesh_id=identity,
                fbx_path=mesh.resolve(), source_split=row["source_split"].strip().lower(),
            ))
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    if any(record.dataset not in {"ict", "pixel3d"} for record in records):
        raise ValueError(f"Main evaluation requires ICT and Pixel3D records: {path}")
    keys = [(record.dataset, record.mesh_id) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate dataset/mesh records in {path}")
    return records


def q95(values: Sequence[float]) -> float:
    """NumPy-compatible linear 0.95 quantile without requiring NumPy at merge."""

    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    position = 0.95 * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def atomic_write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
