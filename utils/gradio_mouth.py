"""Lip splitting and oral fitting for the interactive demo."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import struct

import numpy as np
import torch

import demo
from dataset.image_dataset import _read_accessor, _read_glb, _read_primitive_faces
from utils.lip_split import (
    detect_jaw_open_aperture_faces,
    detect_stretched_mouth_faces,
    remap_lower_lip_landmarks,
    remap_vertex_mapping,
    repair_mediapipe_lip_landmarks_to_dominant_component,
    retain_aperture_connected_face_removals,
    smooth_mouth_boundary_loops,
    split_lips_along_mediapipe_seam,
)
from utils.metahuman_oral import (
    fit_oral_assembly,
    remove_disconnected_head_mouth_components,
    remove_head_faces_behind_outer_lips,
)


@dataclass
class MouthMesh:
    mesh: dict
    landmarks: dict
    source_face_ids: torch.Tensor
    rigid_vertices: torch.Tensor
    reference_faces: torch.Tensor


def prepare_mouth(runtime, prepared) -> MouthMesh:
    mapping = remap_vertex_mapping(demo.load_landmark_mapping(prepared.landmarks),
                                   prepared.vertex_map)
    mapping, _ = repair_mediapipe_lip_landmarks_to_dominant_component(
        prepared.model_mesh, mapping,
    )
    mesh, report = split_lips_along_mediapipe_seam(
        prepared.model_mesh, mapping, corner_inset_ratio=0.05, preopen_distance=0.002,
    )
    mapping = remap_lower_lip_landmarks(mapping, report)
    mesh["landmarks_3d"] = demo.landmark_payload_from_mapping(mapping, mesh["vertices"])
    # Welding removes only degenerate faces; preserve the remaining face order
    # so every corner keeps its original texture coordinates and material.
    welded_faces = prepared.vertex_map[prepared.original["faces"]]
    valid = ((welded_faces[:, 0] != welded_faces[:, 1])
             & (welded_faces[:, 1] != welded_faces[:, 2])
             & (welded_faces[:, 0] != welded_faces[:, 2]))
    source_face_ids = torch.nonzero(valid).flatten()
    if not torch.equal(welded_faces[valid], prepared.model_mesh["faces"]):
        raise ValueError("Cannot preserve the face textures after mouth preparation.")
    rigid = torch.zeros(len(mesh["vertices"]), dtype=torch.bool)
    rigid[prepared.vertex_map[runtime.replacement_eye_mask(prepared.source)]] = True
    # A jaw-open probe identifies faces spanning the mouth aperture. Do not
    # weld again after the split: that would reconnect the separated lips.
    reference_faces = mesh["faces"].clone()
    for _ in range(4):
        probe, _ = demo.run_model(runtime._model, mesh, 26, runtime.config, runtime.device)
        aperture = detect_jaw_open_aperture_faces(probe, mesh["faces"], mapping)
        stretch = detect_stretched_mouth_faces(mesh["vertices"], probe, mesh["faces"], mapping)
        # The updated demo keeps removals connected to the mouth aperture,
        # preventing separate holes in the cheeks or chin.
        connected = retain_aperture_connected_face_removals(
            mesh["faces"], aperture.removed_face_ids, stretch.removed_face_ids,
        )
        if not connected.removed_face_ids:
            break
        keep = torch.ones(len(mesh["faces"]), dtype=torch.bool)
        keep[connected.removed_face_ids] = False
        mesh["faces"] = mesh["faces"][keep]
        source_face_ids = source_face_ids[keep]
        mesh["normals"] = demo.recompute_vertex_normals(mesh["vertices"], mesh["faces"])
    return MouthMesh(mesh, mapping, source_face_ids, rigid, reference_faces)


def export_mouth(source: Path, output: Path, head_vertices: torch.Tensor,
                 mouth: MouthMesh, oral_vertices: torch.Tensor, oral_faces: torch.Tensor,
                 oral_colors: torch.Tensor) -> None:
    """Export changed topology while copying source UVs and embedded textures."""
    document, original_binary = _read_glb(source)
    binary = bytearray(original_binary)

    def attribute(values, *, indices=False, bounds=False):
        values = values.detach().cpu().contiguous()
        data = np.asarray(values.numpy(), dtype="<u4" if indices else "<f4").tobytes()
        binary.extend(b"\0" * (-len(binary) % 4))
        view_id = len(document["bufferViews"])
        document["bufferViews"].append({"buffer": 0, "byteOffset": len(binary),
                                      "byteLength": len(data), "target": 34963 if indices else 34962})
        binary.extend(data)
        accessor = {"bufferView": view_id, "componentType": 5125 if indices else 5126,
                    "count": len(values), "type": "SCALAR" if indices else f"VEC{values.shape[1]}"}
        if bounds:
            accessor.update(min=values.amin(0).tolist(), max=values.amax(0).tolist())
        document["accessors"].append(accessor)
        return len(document["accessors"]) - 1

    normals = demo.recompute_vertex_normals(head_vertices, mouth.mesh["faces"])
    new_primitives = []
    face_offset = 0
    for mesh in document["meshes"]:
        for primitive in mesh["primitives"]:
            attrs = primitive["attributes"]
            count = document["accessors"][attrs["POSITION"]]["count"]
            faces = _read_primitive_faces(document, original_binary, primitive, count)
            selected = ((mouth.source_face_ids >= face_offset)
                        & (mouth.source_face_ids < face_offset + len(faces)))
            old_corners = faces[mouth.source_face_ids[selected] - face_offset].flatten()
            new_corners = mouth.mesh["faces"][selected].flatten()
            face_offset += len(faces)
            if not len(old_corners):
                continue
            # One vertex per (original UV vertex, split geometry vertex) pair
            # keeps UV seams and the new upper/lower lip seam independent.
            pairs, inverse = np.unique(torch.stack((old_corners, new_corners), 1).numpy(),
                                       axis=0, return_inverse=True)
            old_ids, new_ids = torch.from_numpy(pairs).unbind(1)
            new_attrs = {"POSITION": attribute(head_vertices[new_ids], bounds=True),
                         "NORMAL": attribute(normals[new_ids])}
            for name, accessor_id in attrs.items():
                if name.startswith("TEXCOORD_") or name.startswith("COLOR_"):
                    values = _read_accessor(document, original_binary, accessor_id)[old_ids]
                    original_type = values.dtype
                    values = values.float()
                    if document["accessors"][accessor_id].get("normalized"):
                        values /= torch.iinfo(original_type).max
                    new_attrs[name] = attribute(values)
            new_primitive = {"attributes": new_attrs, "mode": 4,
                             "indices": attribute(torch.from_numpy(inverse), indices=True)}
            if "material" in primitive:
                new_primitive["material"] = primitive["material"]
            new_primitives.append(new_primitive)

    oral_normals = demo.recompute_vertex_normals(oral_vertices, oral_faces)
    oral_attributes = {"POSITION": attribute(oral_vertices, bounds=True),
                       "NORMAL": attribute(oral_normals)}
    # Match the updated textured demo export's enamel, gum and cavity materials.
    references = torch.tensor(((0.91, 0.86, 0.72), (0.58, 0.20, 0.23), (0.34, 0.075, 0.085)))
    face_colors = oral_colors[oral_faces].mean(1)
    assignments = ((face_colors[:, None] - references[None]) ** 2).sum(2).argmin(1)
    materials = (("TopoRig_Teeth_Enamel", (0.91, 0.86, 0.72, 1), 0.30),
                 ("TopoRig_Gums_Tongue", (0.58, 0.20, 0.23, 1), 0.42),
                 ("TopoRig_Oral_Cavity", (0.17, 0.025, 0.032, 1), 0.55))
    for group, (name, color, roughness) in enumerate(materials):
        subset = oral_faces[assignments == group]
        if not len(subset):
            continue
        material_id = len(document.setdefault("materials", []))
        document["materials"].append({"name": name, "doubleSided": True,
                                      "pbrMetallicRoughness": {"baseColorFactor": color,
                                                              "metallicFactor": 0,
                                                              "roughnessFactor": roughness}})
        new_primitives.append({"attributes": oral_attributes,
                               "indices": attribute(subset.flatten(), indices=True),
                               "material": material_id, "mode": 4})
    document["meshes"] = [{"name": "Face with mouth interior", "primitives": new_primitives}]
    document["nodes"] = [{"mesh": 0, "name": "Face with mouth interior"}]
    document["scenes"] = [{"nodes": [0]}]
    document["scene"] = 0
    document.pop("animations", None)
    document.pop("skins", None)
    document["buffers"][0]["byteLength"] = len(binary)
    payload = json.dumps(document, separators=(",", ":")).encode()
    payload += b" " * (-len(payload) % 4)
    binary += b"\0" * (-len(binary) % 4)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, 28 + len(payload) + len(binary)))
        handle.write(struct.pack("<I4s", len(payload), b"JSON") + payload)
        handle.write(struct.pack("<I4s", len(binary), b"BIN\0") + binary)


def predict_mouth(runtime, prepared, mouth: MouthMesh, action_unit: int,
                  intensity: float, output: Path, oral_asset: Path) -> None:
    key = (str(prepared.source), action_unit, "mouth")
    if key not in runtime._predictions:
        _, delta = demo.run_model(runtime._model, mouth.mesh, action_unit,
                                  runtime.config, runtime.device)
        delta[mouth.rigid_vertices] = 0
        runtime._predictions[key] = delta
        while len(runtime._predictions) > 8:
            runtime._predictions.popitem(last=False)
    runtime._predictions.move_to_end(key)
    neutral = mouth.mesh["vertices"]
    full_expression = neutral + runtime._predictions[key]
    neutral, full_expression, _ = smooth_mouth_boundary_loops(
        neutral, full_expression, mouth.mesh["faces"], mouth.landmarks,
        reference_faces=mouth.reference_faces,
        preserve_boundary_junctions=True, arc_length_weighted=True,
    )
    neutral_oral, full_oral, oral_faces, colors, _ = fit_oral_assembly(
        oral_asset, neutral, full_expression, mouth.landmarks,
        depth_offset=0.012, level_target_frame=True, contain_inside_lips=True,
        aperture_crop_scale=1.0, tongue_planar_scale=1.4,
    )
    depth = float(full_oral[:, 1].amax())
    cleaned_faces, aperture = remove_head_faces_behind_outer_lips(
        full_expression, mouth.mesh["faces"], mouth.landmarks, maximum_depth=depth,
    )
    cleaned_faces, _ = remove_disconnected_head_mouth_components(
        full_expression, cleaned_faces, torch.tensor(aperture.aperture_polygon),
        mapping=mouth.landmarks, maximum_depth=depth,
    )
    # Both cleanup routines retain face order. Recover the corresponding source
    # face ids so the textured export follows the same oral-aperture cleanup.
    row_type = np.dtype((np.void, mouth.mesh["faces"].numpy().dtype.itemsize * 3))
    original_rows = mouth.mesh["faces"].numpy().view(row_type).flatten()
    cleaned_rows = cleaned_faces.numpy().view(row_type).flatten()
    keep = torch.from_numpy(np.isin(original_rows, cleaned_rows))
    export_mesh = replace(mouth, mesh={**mouth.mesh, "faces": cleaned_faces},
                          source_face_ids=mouth.source_face_ids[keep])
    vertices = torch.lerp(neutral, full_expression, intensity)
    oral_vertices = torch.lerp(neutral_oral, full_oral, intensity)
    vertices = vertices @ prepared.model_to_source + prepared.model_offset
    oral_vertices = oral_vertices @ prepared.model_to_source + prepared.model_offset
    if not torch.isfinite(vertices).all() or not torch.isfinite(oral_vertices).all():
        raise ValueError("Mouth fitting produced invalid coordinates.")
    export_mouth(prepared.source, output, vertices, export_mesh, oral_vertices, oral_faces, colors)
