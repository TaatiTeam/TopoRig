#!/usr/bin/env python3
"""Export a lip-split TopoRig result with the source Pixel3D appearance.

The lip-split pipeline changes topology, so the retained-topology exporter in
``demo.py`` cannot be used directly.  This exporter consumes the exact
component arrays written by ``test_lip_split_jaw_open.py``, transfers UVs from
the canonical textured demo output onto the neutral split surface, and writes
neutral/jawOpen as an FBX rig.  Teeth, gum/tongue, and cavity faces receive
separate physically plausible materials rather than the head texture.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy
import numpy as np
import torch
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-rig-fbx",
        type=Path,
        required=True,
        help="Canonical, textured retained-topology demo output used as the UV source.",
    )
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-neutral-predictions",
        type=Path,
        help="Prediction NPZ containing the retained-topology neutral_vertices array.",
    )
    parser.add_argument("--source-landmarks", type=Path)
    parser.add_argument("--target-landmarks", type=Path)
    parser.add_argument(
        "--alignment-target-components",
        type=Path,
        help=(
            "Optional component NPZ whose neutral vertices match target-landmark "
            "indices. Useful for aligning through the exact dense canonical frame "
            "while exporting a simplified clean-mouth target."
        ),
    )
    parser.add_argument(
        "--uv-transfer-method",
        choices=(
            "topology-exact",
            "centroid-face",
            "nearest-face",
            "nearest-vertex",
            "baked-color",
        ),
        default="nearest-face",
        help=(
            "topology-exact preserves source face-corner UVs for unchanged "
            "triangles; centroid-face performs a fast barycentric source-face "
            "projection for decimated surfaces; nearest-face uses Blender's "
            "slower spatial transfer; baked-color samples the source diffuse "
            "atlas into seam-safe vertex colours; nearest-vertex is diagnostic only."
        ),
    )
    parser.add_argument(
        "--uv-source-target-faces",
        type=int,
        help=(
            "Optional face budget for a hidden copy of the UV donor before "
            "nearest-face transfer. This never changes the rendered target mesh."
        ),
    )
    parser.add_argument(
        "--include-all-prediction-blendshapes",
        action="store_true",
        help=(
            "Add every target from --source-neutral-predictions to the split "
            "head. The generated lip-split/oral jawOpen target takes precedence."
        ),
    )
    return parser.parse_args()


def load_components(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        result = {key: np.asarray(payload[key]) for key in payload.files}
    neutral = np.asarray(result["neutral_vertices"], dtype=np.float32)
    jaw = np.asarray(result["jaw_open_vertices"], dtype=np.float32)
    faces = np.asarray(result["faces"], dtype=np.int64)
    colors = np.asarray(result["vertex_colors"], dtype=np.float32)
    if neutral.ndim != 2 or neutral.shape[1] != 3:
        raise ValueError(f"Invalid neutral vertices: {neutral.shape}")
    if jaw.shape != neutral.shape or colors.shape != neutral.shape:
        raise ValueError(
            f"Component shape mismatch: neutral={neutral.shape}, jaw={jaw.shape}, "
            f"colors={colors.shape}"
        )
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Invalid faces: {faces.shape}")
    result.update(
        neutral_vertices=neutral,
        jaw_open_vertices=jaw,
        faces=faces,
        vertex_colors=colors,
    )
    return result


def scalar(payload: dict[str, Any], key: str) -> int:
    return int(np.asarray(payload[key]).reshape(()).item())


def load_landmark_mapping(path: Path) -> dict[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Landmark file must contain an object: {path}")
    return {int(key): int(value) for key, value in payload.items()}


def robust_similarity_alignment(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    source_landmarks: dict[int, int],
    target_landmarks: dict[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    common = sorted(set(source_landmarks) & set(target_landmarks))
    if len(common) < 32:
        raise ValueError(f"Only {len(common)} common alignment landmarks were found.")
    source = source_vertices[[source_landmarks[index] for index in common]].astype(np.float64)
    target = target_vertices[[target_landmarks[index] for index in common]].astype(np.float64)
    keep = np.ones(len(common), dtype=bool)
    scale = 1.0
    rotation = np.eye(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    for _iteration in range(4):
        source_fit = source[keep]
        target_fit = target[keep]
        source_center = source_fit.mean(axis=0)
        target_center = target_fit.mean(axis=0)
        source_centered = source_fit - source_center
        target_centered = target_fit - target_center
        u, singular_values, vt = np.linalg.svd(
            (target_centered.T @ source_centered) / len(source_fit)
        )
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        variance = np.mean(np.sum(source_centered * source_centered, axis=1))
        scale = float(singular_values.sum() / variance)
        translation = target_center - scale * (rotation @ source_center)
        predicted = scale * (source @ rotation.T) + translation
        residual = np.linalg.norm(predicted - target, axis=1)
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        keep = residual < median + 4.0 * max(mad, 1.0e-6)
    aligned = scale * (source_vertices.astype(np.float64) @ rotation.T) + translation
    final_residual = np.linalg.norm(
        scale * (source @ rotation.T) + translation - target,
        axis=1,
    )
    report = {
        "common_landmark_count": len(common),
        "inlier_landmark_count": int(np.count_nonzero(keep)),
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "inlier_residual_median": float(np.median(final_residual[keep])),
        "inlier_residual_q95": float(np.quantile(final_residual[keep], 0.95)),
        "inlier_residual_max": float(np.max(final_residual[keep])),
    }
    return aligned.astype(np.float32), report


def load_prediction_bank(
    path: Path,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        neutral = np.asarray(payload["neutral_vertices"], dtype=np.float32)
        targets = np.asarray(payload["target_vertices"], dtype=np.float32)
        names = [str(value) for value in payload["action_unit_names"].tolist()]
    if neutral.ndim != 2 or neutral.shape[1] != 3:
        raise ValueError(f"Invalid prediction neutral shape: {neutral.shape}")
    if targets.shape != (len(names), len(neutral), 3):
        raise ValueError(
            f"Invalid prediction targets {targets.shape}; expected "
            f"({len(names)}, {len(neutral)}, 3)."
        )
    if not np.isfinite(neutral).all() or not np.isfinite(targets).all():
        raise ValueError("Prediction bank contains non-finite coordinates.")
    return neutral, names, targets


def map_prediction_bank_to_split_head(
    *,
    prediction_neutral: np.ndarray,
    prediction_targets: np.ndarray,
    target_neutral: np.ndarray,
    source_landmarks: dict[int, int],
    target_landmarks: dict[int, int],
    alignment_target_vertices: np.ndarray,
    neighbors: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    aligned_neutral, alignment = robust_similarity_alignment(
        prediction_neutral,
        alignment_target_vertices,
        source_landmarks,
        target_landmarks,
    )
    scale = float(alignment["scale"])
    rotation = np.asarray(alignment["rotation"], dtype=np.float64)
    prediction_deltas = (
        (prediction_targets.astype(np.float64) - prediction_neutral[None].astype(np.float64))
        @ rotation.T
        * scale
    ).astype(np.float32)
    neighbor_count = max(1, min(int(neighbors), len(aligned_neutral)))
    distances, nearest = cKDTree(aligned_neutral).query(
        target_neutral,
        k=neighbor_count,
        workers=-1,
    )
    if neighbor_count == 1:
        distances = distances[:, None]
        nearest = nearest[:, None]
    weights = 1.0 / np.maximum(distances, 1.0e-8) ** 2
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-12)
    mapped_deltas = np.sum(
        prediction_deltas[:, nearest, :] * weights[None, :, :, None],
        axis=2,
    )
    diagonal = float(np.linalg.norm(np.ptp(aligned_neutral, axis=0)))
    q99 = float(np.quantile(distances[:, 0], 0.99))
    if not np.isfinite(q99) or diagonal <= 0.0 or q99 > diagonal * 0.05:
        raise ValueError(
            "Unsafe prediction-to-split mapping: "
            f"q99={q99:g}, diagonal={diagonal:g}."
        )
    return (target_neutral[None] + mapped_deltas).astype(np.float32), {
        "alignment": alignment,
        "neighbors": neighbor_count,
        "mapping_distance_median": float(np.median(distances[:, 0])),
        "mapping_distance_q99": q99,
        "mapping_distance_max": float(np.max(distances[:, 0])),
        "prediction_bbox_diagonal": diagonal,
    }


def create_mesh_object(
    name: str,
    neutral: np.ndarray,
    jaw: np.ndarray,
    faces: np.ndarray,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(neutral.tolist(), [], faces.tolist())
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    basis = obj.shape_key_add(name="Basis", from_mix=False)
    basis.interpolation = "KEY_LINEAR"
    jaw_key = obj.shape_key_add(name="jawOpen", from_mix=False)
    jaw_key.interpolation = "KEY_LINEAR"
    jaw_key.slider_min = 0.0
    jaw_key.slider_max = 1.0
    jaw_key.data.foreach_set("co", np.ascontiguousarray(jaw).reshape(-1))
    mesh.update(calc_edges=True)
    return obj


def add_shape_key(
    obj: bpy.types.Object,
    name: str,
    coordinates: np.ndarray,
) -> None:
    key = obj.shape_key_add(name=name, from_mix=False)
    key.interpolation = "KEY_LINEAR"
    key.slider_min = 0.0
    key.slider_max = 1.0
    key.value = 0.0
    key.data.foreach_set("co", np.ascontiguousarray(coordinates).reshape(-1))


def subset_component(
    vertices: np.ndarray,
    jaw: np.ndarray,
    faces: np.ndarray,
    face_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    component_faces = faces[face_mask]
    used = np.unique(component_faces.reshape(-1))
    inverse = np.full(vertices.shape[0], -1, dtype=np.int64)
    inverse[used] = np.arange(used.size, dtype=np.int64)
    return (
        vertices[used],
        jaw[used],
        inverse[component_faces],
        used,
    )


def make_principled_material(
    name: str,
    color: tuple[float, float, float, float],
    *,
    roughness: float,
    specular: float,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
        specular_input = bsdf.inputs.get("Specular IOR Level") or bsdf.inputs.get("Specular")
        if specular_input is not None:
            specular_input.default_value = specular
    return material


def copy_source_materials(
    source: bpy.types.Object,
    target: bpy.types.Object,
) -> None:
    if not source.data.uv_layers:
        raise ValueError(f"UV source {source.name!r} has no UV layers.")
    if not source.material_slots:
        raise ValueError(f"UV source {source.name!r} has no material slots.")
    for slot in source.material_slots:
        if slot.material is not None:
            target.data.materials.append(slot.material)
    if not target.data.materials:
        raise ValueError(f"UV source {source.name!r} has no usable materials.")
    target.data.uv_layers.new(name=source.data.uv_layers.active.name)


def transfer_uvs(source: bpy.types.Object, target: bpy.types.Object) -> None:
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    source.select_set(False)
    modifier = target.modifiers.new(name="SourceUVTransfer", type="DATA_TRANSFER")
    modifier.object = source
    modifier.use_loop_data = True
    modifier.data_types_loops = {"UV"}
    modifier.loop_mapping = "POLYINTERP_NEAREST"
    modifier.layers_uv_select_src = source.data.uv_layers.active.name
    modifier.layers_uv_select_dst = target.data.uv_layers.active.name
    result = bpy.ops.object.modifier_apply(modifier=modifier.name)
    if "FINISHED" not in result:
        raise RuntimeError("Blender could not apply the source UV transfer.")
    uv = target.data.uv_layers.active
    if uv is None or len(uv.data) != len(target.data.loops):
        raise RuntimeError("Transferred UV layer is missing or incomplete.")
    values = np.empty(len(uv.data) * 2, dtype=np.float32)
    uv.data.foreach_get("uv", values)
    if not np.isfinite(values).all() or float(np.ptp(values)) < 1.0e-5:
        raise RuntimeError("Transferred UV coordinates are invalid or constant.")


def reduce_uv_source(
    source: bpy.types.Object, target_faces: int
) -> dict[str, int]:
    if target_faces <= 0:
        raise ValueError("UV-source target faces must be positive.")
    original_faces = len(source.data.polygons)
    if original_faces <= target_faces:
        return {
            "original_face_count": original_faces,
            "result_face_count": original_faces,
        }
    bpy.context.view_layer.objects.active = source
    source.select_set(True)
    modifier = source.modifiers.new(name="UVSourceDecimate", type="DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = float(target_faces) / float(original_faces)
    modifier.use_collapse_triangulate = True
    result = bpy.ops.object.modifier_apply(modifier=modifier.name)
    if "FINISHED" not in result:
        raise RuntimeError("Could not decimate the hidden UV-source copy.")
    if source.data.uv_layers.active is None:
        raise RuntimeError("UV-source decimation removed the active UV layer.")
    return {
        "original_face_count": original_faces,
        "result_face_count": len(source.data.polygons),
    }


def transfer_uvs_nearest_vertex(
    source: bpy.types.Object, target: bpy.types.Object
) -> None:
    """Transfer source corner UVs through a fast nearest-vertex lookup."""

    source_uv = source.data.uv_layers.active
    target_uv = target.data.uv_layers.active
    if source_uv is None or target_uv is None:
        raise ValueError("Nearest-vertex UV transfer requires active UV layers.")
    source_points = np.asarray(
        [tuple(source.matrix_world @ vertex.co) for vertex in source.data.vertices],
        dtype=np.float32,
    )
    target_points = np.asarray(
        [tuple(target.matrix_world @ vertex.co) for vertex in target.data.vertices],
        dtype=np.float32,
    )
    source_loop_uv = np.empty((len(source_uv.data), 2), dtype=np.float32)
    source_uv.data.foreach_get("uv", source_loop_uv.reshape(-1))
    source_loop_vertex = np.empty(len(source.data.loops), dtype=np.int64)
    source.data.loops.foreach_get("vertex_index", source_loop_vertex)
    # The first corner is deliberately retained at a discontinuous seam.  UV
    # averaging would blend unrelated atlas islands and create a dark line.
    first_loop = np.full(len(source.data.vertices), -1, dtype=np.int64)
    first_loop[source_loop_vertex[::-1]] = np.arange(len(source_loop_vertex) - 1, -1, -1)
    referenced = first_loop >= 0
    if not np.any(referenced):
        raise RuntimeError("Source mesh has no UV-referenced vertices.")
    referenced_points = source_points[referenced]
    referenced_uv = source_loop_uv[first_loop[referenced]]
    _distance, nearest = cKDTree(referenced_points).query(target_points, workers=-1)
    target_vertex_uv = referenced_uv[nearest]
    target_loop_vertex = np.empty(len(target.data.loops), dtype=np.int64)
    target.data.loops.foreach_get("vertex_index", target_loop_vertex)
    target_uv.data.foreach_set(
        "uv",
        np.ascontiguousarray(target_vertex_uv[target_loop_vertex]).reshape(-1),
    )
    target.data.update()


def source_base_color_image(source: bpy.types.Object) -> bpy.types.Image:
    """Return the image directly driving the source Principled base colour."""

    for material in source.data.materials:
        if material is None or not material.use_nodes or material.node_tree is None:
            continue
        principled = next(
            (node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"),
            None,
        )
        if principled is None:
            continue
        base_color = principled.inputs.get("Base Color")
        if base_color is None:
            continue
        for link in base_color.links:
            image = getattr(link.from_node, "image", None)
            if link.from_node.type == "TEX_IMAGE" and image is not None:
                return image
    raise RuntimeError("The UV source has no image directly driving Principled Base Color.")


def sample_image_at_uvs(image: bpy.types.Image, uvs: np.ndarray) -> np.ndarray:
    """Bilinearly sample a packed Blender image and return linear RGBA."""

    width, height = int(image.size[0]), int(image.size[1])
    if width < 2 or height < 2:
        raise RuntimeError(f"Texture image {image.name!r} is empty.")
    flat = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(flat)
    pixels = flat.reshape(height, width, 4)
    x = np.mod(uvs[:, 0], 1.0) * (width - 1)
    y = np.mod(uvs[:, 1], 1.0) * (height - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    tx = (x - x0).astype(np.float32)[:, None]
    ty = (y - y0).astype(np.float32)[:, None]
    top = pixels[y0, x0] * (1.0 - tx) + pixels[y0, x1] * tx
    bottom = pixels[y1, x0] * (1.0 - tx) + pixels[y1, x1] * tx
    colors = top * (1.0 - ty) + bottom * ty
    # Keep the packed image's sRGB samples. The publication renderer applies
    # the same sRGB-to-linear transform it uses for retained Pixel3D colours.
    colors[:, 3] = 1.0
    return colors


def transfer_baked_vertex_colors(
    source: bpy.types.Object, target: bpy.types.Object
) -> dict[str, Any]:
    """Sample source texture at UV corners, then map colours spatially.

    Mapping UVs onto a simplified mesh lets triangles interpolate across
    unrelated atlas islands. Mapping the sampled colour instead is seam-safe
    and retains the real source appearance on the clean proxy surface.
    """

    source_uv = source.data.uv_layers.active
    if source_uv is None:
        raise ValueError("Baked-colour transfer requires a source UV layer.")
    image = source_base_color_image(source)
    loop_uvs = np.empty((len(source_uv.data), 2), dtype=np.float32)
    source_uv.data.foreach_get("uv", loop_uvs.reshape(-1))
    loop_vertices = np.empty(len(source.data.loops), dtype=np.int64)
    source.data.loops.foreach_get("vertex_index", loop_vertices)
    source_points = np.asarray(
        [tuple(source.matrix_world @ vertex.co) for vertex in source.data.vertices],
        dtype=np.float32,
    )
    target_points = np.asarray(
        [tuple(target.matrix_world @ vertex.co) for vertex in target.data.vertices],
        dtype=np.float32,
    )
    if any(len(polygon.vertices) != 3 for polygon in source.data.polygons):
        bpy.context.view_layer.objects.active = source
        source.select_set(True)
        triangulate = source.modifiers.new(name="TextureBakeTriangulate", type="TRIANGULATE")
        result = bpy.ops.object.modifier_apply(modifier=triangulate.name)
        if "FINISHED" not in result:
            raise RuntimeError("Could not triangulate the texture-bake source.")
        source_points = np.asarray(
            [tuple(source.matrix_world @ vertex.co) for vertex in source.data.vertices],
            dtype=np.float32,
        )
        source_uv = source.data.uv_layers.active
        loop_uvs = np.empty((len(source_uv.data), 2), dtype=np.float32)
        source_uv.data.foreach_get("uv", loop_uvs.reshape(-1))

    # Query the closest point on the actual textured source surface. Sampling
    # nearest vertices can select a hidden mouth/hair layer even when the
    # visible surface is closer, causing the dark patches seen in bad renders.
    tree = BVHTree.FromObject(source, bpy.context.evaluated_depsgraph_get())
    source_faces = np.asarray(
        [tuple(polygon.vertices) for polygon in source.data.polygons],
        dtype=np.int64,
    )
    source_face_uvs = np.asarray(
        [loop_uvs[polygon.loop_start : polygon.loop_start + 3] for polygon in source.data.polygons],
        dtype=np.float32,
    )
    sampled_uvs = np.empty((len(target_points), 2), dtype=np.float32)
    distances = np.empty(len(target_points), dtype=np.float64)
    for target_index, point in enumerate(target_points):
        location, _normal, polygon_index, distance = tree.find_nearest(Vector(point))
        if location is None or polygon_index is None:
            raise RuntimeError(f"No source-surface match for target vertex {target_index}.")
        triangle = source_points[source_faces[int(polygon_index)]].astype(np.float64)
        relative = np.asarray(location, dtype=np.float64) - triangle[0]
        edge_0 = triangle[1] - triangle[0]
        edge_1 = triangle[2] - triangle[0]
        dot_00 = float(np.dot(edge_0, edge_0))
        dot_01 = float(np.dot(edge_0, edge_1))
        dot_11 = float(np.dot(edge_1, edge_1))
        dot_r0 = float(np.dot(relative, edge_0))
        dot_r1 = float(np.dot(relative, edge_1))
        denominator = dot_00 * dot_11 - dot_01 * dot_01
        if abs(denominator) <= 1.0e-20:
            weights = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        else:
            weight_1 = (dot_11 * dot_r0 - dot_01 * dot_r1) / denominator
            weight_2 = (dot_00 * dot_r1 - dot_01 * dot_r0) / denominator
            weights = np.asarray(
                (1.0 - weight_1 - weight_2, weight_1, weight_2),
                dtype=np.float64,
            )
        sampled_uvs[target_index] = weights @ source_face_uvs[int(polygon_index)]
        distances[target_index] = float(distance)
    target_colors = sample_image_at_uvs(image, sampled_uvs)
    color_name = "TopoRig_Source_Texture"
    for old in list(target.data.color_attributes):
        target.data.color_attributes.remove(old)
    attribute = target.data.color_attributes.new(
        name=color_name,
        type="FLOAT_COLOR",
        domain="POINT",
    )
    attribute.data.foreach_set("color", np.ascontiguousarray(target_colors).reshape(-1))
    target.data.color_attributes.active_color_index = 0
    target.data.color_attributes.render_color_index = 0

    target.data.materials.clear()
    material = bpy.data.materials.new("TopoRig_Pixel3D_Texture_Baked")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    principled = next(
        (node for node in nodes if node.type == "BSDF_PRINCIPLED"), None
    )
    if principled is None:
        principled = nodes.new("ShaderNodeBsdfPrincipled")
    vertex_color = nodes.new("ShaderNodeVertexColor")
    vertex_color.layer_name = color_name
    material.node_tree.links.new(
        vertex_color.outputs["Color"], principled.inputs["Base Color"]
    )
    roughness = principled.inputs.get("Roughness")
    metallic = principled.inputs.get("Metallic")
    if roughness is not None:
        roughness.default_value = 0.46
    if metallic is not None:
        metallic.default_value = 0.0
    target.data.materials.append(material)
    for polygon in target.data.polygons:
        polygon.material_index = 0
    target.data.update()
    diagonal = float(np.linalg.norm(np.ptp(source_points, axis=0)))
    return {
        "source_image": image.name,
        "source_image_size": [int(image.size[0]), int(image.size[1])],
        "target_vertex_count": len(target_points),
        "surface_distance_median": float(np.median(distances)),
        "surface_distance_q99": float(np.quantile(distances, 0.99)),
        "surface_distance_max": float(np.max(distances)),
        "surface_distance_q99_relative_to_diagonal": (
            float(np.quantile(distances, 0.99) / diagonal) if diagonal > 0.0 else None
        ),
    }


def transfer_uvs_topology_exact(
    source: bpy.types.Object, target: bpy.types.Object
) -> dict[str, Any]:
    """Copy exact source corner UVs through coincident-position face keys.

    Lip splitting welds source duplicates, but it leaves almost every surface
    triangle geometrically unchanged.  A face key made from its three spatial
    position ids therefore identifies the original UV corners without
    collapsing atlas seams.  Newly constructed mouth faces fall back to a
    barycentric projection from the nearest source triangle.
    """

    source_uv = source.data.uv_layers.active
    target_uv = target.data.uv_layers.active
    if source_uv is None or target_uv is None:
        raise ValueError("Topology-exact UV transfer requires active UV layers.")
    if len(source.data.loops) != 3 * len(source.data.polygons):
        bpy.context.view_layer.objects.active = source
        source.select_set(True)
        triangulate = source.modifiers.new(name="UVSourceTriangulate", type="TRIANGULATE")
        triangulate.quad_method = "BEAUTY"
        triangulate.ngon_method = "BEAUTY"
        result = bpy.ops.object.modifier_apply(modifier=triangulate.name)
        if "FINISHED" not in result:
            raise RuntimeError("Could not triangulate the UV-source mesh.")
        source_uv = source.data.uv_layers.active
        if source_uv is None:
            raise RuntimeError("Triangulating the UV source removed its active UV layer.")
    if len(source.data.loops) != 3 * len(source.data.polygons):
        raise ValueError("Topology-exact UV transfer requires triangular source faces.")
    if len(target.data.loops) != 3 * len(target.data.polygons):
        raise ValueError("Topology-exact UV transfer requires triangular target faces.")

    source_points = np.asarray(
        [tuple(source.matrix_world @ vertex.co) for vertex in source.data.vertices],
        dtype=np.float64,
    )
    target_points = np.asarray(
        [tuple(target.matrix_world @ vertex.co) for vertex in target.data.vertices],
        dtype=np.float64,
    )
    source_loop_vertex = np.empty(len(source.data.loops), dtype=np.int64)
    source.data.loops.foreach_get("vertex_index", source_loop_vertex)
    target_loop_vertex = np.empty(len(target.data.loops), dtype=np.int64)
    target.data.loops.foreach_get("vertex_index", target_loop_vertex)
    source_faces = source_loop_vertex.reshape(-1, 3)
    target_faces = target_loop_vertex.reshape(-1, 3)
    source_loop_uv = np.empty((len(source_uv.data), 2), dtype=np.float32)
    source_uv.data.foreach_get("uv", source_loop_uv.reshape(-1))
    source_face_uv = source_loop_uv.reshape(-1, 3, 2)

    diagonal = float(np.linalg.norm(np.ptp(source_points, axis=0)))
    tolerance = max(diagonal * 5.0e-7, 1.0e-8)
    quantized = np.rint(source_points / tolerance).astype(np.int64)
    _unique_positions, source_position_ids = np.unique(
        quantized, axis=0, return_inverse=True
    )
    distances, nearest_source = cKDTree(source_points).query(
        target_points, workers=-1
    )
    target_position_ids = source_position_ids[nearest_source]
    valid_target_vertices = distances <= tolerance

    source_face_positions = source_position_ids[source_faces]
    target_face_positions = target_position_ids[target_faces]
    valid_target_faces = np.all(valid_target_vertices[target_faces], axis=1)
    position_count = int(source_position_ids.max()) + 1
    bits = max(1, int(np.ceil(np.log2(position_count + 1))))
    if 3 * bits > 63:
        raise RuntimeError("Too many unique positions for packed triangular face keys.")

    def packed_face_keys(position_ids: np.ndarray) -> np.ndarray:
        ordered = np.sort(position_ids.astype(np.uint64), axis=1)
        return ordered[:, 0] | (ordered[:, 1] << bits) | (ordered[:, 2] << (2 * bits))

    source_keys = packed_face_keys(source_face_positions)
    source_order = np.argsort(source_keys)
    sorted_source_keys = source_keys[source_order]
    matched = np.zeros(len(target_faces), dtype=bool)
    source_face_for_target = np.full(len(target_faces), -1, dtype=np.int64)
    candidate_ids = np.flatnonzero(valid_target_faces)
    candidate_keys = packed_face_keys(target_face_positions[candidate_ids])
    source_positions = np.searchsorted(sorted_source_keys, candidate_keys)
    in_bounds = source_positions < len(sorted_source_keys)
    exact = np.zeros(len(candidate_ids), dtype=bool)
    exact[in_bounds] = (
        sorted_source_keys[source_positions[in_bounds]] == candidate_keys[in_bounds]
    )
    exact_target_ids = candidate_ids[exact]
    exact_source_ids = source_order[source_positions[exact]]
    matched[exact_target_ids] = True
    source_face_for_target[exact_target_ids] = exact_source_ids

    target_face_uv = np.full(
        (len(target_faces), 3, 2), np.nan, dtype=np.float32
    )
    target_codes = target_face_positions[exact_target_ids]
    source_codes = source_face_positions[exact_source_ids]
    equal = target_codes[:, :, None] == source_codes[:, None, :]
    if not np.all(np.any(equal, axis=2)):
        raise RuntimeError("Exact face-key UV transfer found an unmatched corner.")
    source_corner = np.argmax(equal, axis=2)
    target_face_uv[exact_target_ids] = np.take_along_axis(
        source_face_uv[exact_source_ids], source_corner[:, :, None], axis=1
    )

    fallback_ids = np.flatnonzero(~matched)
    if fallback_ids.size:
        # New hole-fill/split triangles do not have a source-face key.  Reuse
        # UVs from exact neighboring target corners instead of projecting onto
        # the spatially closest source triangle, which can be an internal dark
        # mouth surface on an open-jaw asset.
        exact_corner_vertices = target_faces[exact_target_ids].reshape(-1)
        exact_corner_uv = target_face_uv[exact_target_ids].reshape(-1, 2)
        unique_vertices, first_corner = np.unique(
            exact_corner_vertices, return_index=True
        )
        target_vertex_uv = np.full(
            (len(target_points), 2), np.nan, dtype=np.float32
        )
        target_vertex_uv[unique_vertices] = exact_corner_uv[first_corner]
        fallback_vertices = target_faces[fallback_ids]
        missing_vertices = np.unique(
            fallback_vertices[~np.isfinite(target_vertex_uv[fallback_vertices]).all(axis=2)]
        )
        if missing_vertices.size:
            known_vertices = np.flatnonzero(
                np.isfinite(target_vertex_uv).all(axis=1)
            )
            if not known_vertices.size:
                raise RuntimeError("No exact target UV corners exist for fallback repair.")
            _fallback_distance, nearest_known = cKDTree(
                target_points[known_vertices]
            ).query(target_points[missing_vertices], workers=-1)
            target_vertex_uv[missing_vertices] = target_vertex_uv[
                known_vertices[nearest_known]
            ]
        target_face_uv[fallback_ids] = target_vertex_uv[fallback_vertices]

    invalid_uv_faces = ~np.isfinite(target_face_uv).all(axis=(1, 2))
    repaired_invalid_face_count = int(np.count_nonzero(invalid_uv_faces))
    if repaired_invalid_face_count:
        finite_source_faces = np.isfinite(source_face_uv).all(axis=(1, 2))
        finite_source_ids = np.flatnonzero(finite_source_faces)
        if not finite_source_ids.size:
            raise RuntimeError("UV source contains no fully finite triangular faces.")
        finite_triangles = source_points[source_faces[finite_source_ids]]
        finite_centers = finite_triangles.mean(axis=1)
        invalid_target_ids = np.flatnonzero(invalid_uv_faces)
        invalid_centers = target_points[target_faces[invalid_target_ids]].mean(axis=1)
        _repair_distance, repair_nearest = cKDTree(finite_centers).query(
            invalid_centers, workers=-1
        )
        target_face_uv[invalid_target_ids] = source_face_uv[
            finite_source_ids[repair_nearest]
        ]
    if not np.isfinite(target_face_uv).all():
        raise RuntimeError("Topology-exact UV repair left non-finite coordinates.")

    target_uv.data.foreach_set("uv", np.ascontiguousarray(target_face_uv).reshape(-1))
    target.data.update()
    return {
        "exact_face_count": int(np.count_nonzero(matched)),
        "fallback_face_count": int(fallback_ids.size),
        "exact_face_fraction": float(np.mean(matched)),
        "position_tolerance": tolerance,
        "target_vertex_distance_q99": float(np.quantile(distances, 0.99)),
        "target_vertex_distance_max": float(np.max(distances)),
        "repaired_invalid_uv_face_count": repaired_invalid_face_count,
    }


def transfer_uvs_centroid_face(
    source: bpy.types.Object, target: bpy.types.Object
) -> dict[str, Any]:
    """Project target corners through nearby source triangles in bounded chunks."""

    source_uv = source.data.uv_layers.active
    target_uv = target.data.uv_layers.active
    if source_uv is None or target_uv is None:
        raise ValueError("Centroid-face UV transfer requires active UV layers.")
    if len(source.data.loops) != 3 * len(source.data.polygons):
        bpy.context.view_layer.objects.active = source
        source.select_set(True)
        triangulate = source.modifiers.new(name="UVSourceTriangulate", type="TRIANGULATE")
        triangulate.quad_method = "BEAUTY"
        triangulate.ngon_method = "BEAUTY"
        result = bpy.ops.object.modifier_apply(modifier=triangulate.name)
        if "FINISHED" not in result:
            raise RuntimeError("Could not triangulate the UV-source mesh.")
        source_uv = source.data.uv_layers.active
    if source_uv is None or len(source.data.loops) != 3 * len(source.data.polygons):
        raise RuntimeError("Centroid-face UV source was not triangulated correctly.")
    if len(target.data.loops) != 3 * len(target.data.polygons):
        raise ValueError("Centroid-face UV transfer requires triangular target faces.")

    source_points = np.asarray(
        [tuple(source.matrix_world @ vertex.co) for vertex in source.data.vertices],
        dtype=np.float64,
    )
    target_points = np.asarray(
        [tuple(target.matrix_world @ vertex.co) for vertex in target.data.vertices],
        dtype=np.float64,
    )
    source_loop_vertex = np.empty(len(source.data.loops), dtype=np.int64)
    source.data.loops.foreach_get("vertex_index", source_loop_vertex)
    target_loop_vertex = np.empty(len(target.data.loops), dtype=np.int64)
    target.data.loops.foreach_get("vertex_index", target_loop_vertex)
    source_faces = source_loop_vertex.reshape(-1, 3)
    target_faces = target_loop_vertex.reshape(-1, 3)
    source_face_uv = np.empty((len(source.data.polygons), 3, 2), dtype=np.float32)
    source_uv.data.foreach_get("uv", source_face_uv.reshape(-1))

    source_triangles = source_points[source_faces]
    source_centers = source_triangles.mean(axis=1)
    source_tree = cKDTree(source_centers)
    target_triangles = target_points[target_faces]
    target_centers = target_triangles.mean(axis=1)
    corner_points = target_triangles.reshape(-1, 3)
    # A tiny shift toward the target face center makes coincident corners on a
    # source atlas seam choose the source triangle on the correct face side.
    corner_queries = (
        0.98 * target_triangles + 0.02 * target_centers[:, None, :]
    ).reshape(-1, 3)
    output_uv = np.empty((len(corner_points), 2), dtype=np.float32)
    chosen_distances = np.empty(len(corner_points), dtype=np.float64)
    candidate_count = 12
    chunk_size = 20000
    for start in range(0, len(corner_points), chunk_size):
        end = min(start + chunk_size, len(corner_points))
        _centroid_distance, candidate_ids = source_tree.query(
            corner_queries[start:end], k=candidate_count, workers=-1
        )
        candidates = source_triangles[candidate_ids]
        points = corner_points[start:end, None, :]
        edge_0 = candidates[:, :, 1] - candidates[:, :, 0]
        edge_1 = candidates[:, :, 2] - candidates[:, :, 0]
        relative = points - candidates[:, :, 0]
        dot_00 = np.einsum("mkj,mkj->mk", edge_0, edge_0)
        dot_01 = np.einsum("mkj,mkj->mk", edge_0, edge_1)
        dot_11 = np.einsum("mkj,mkj->mk", edge_1, edge_1)
        dot_r0 = np.einsum("mkj,mkj->mk", relative, edge_0)
        dot_r1 = np.einsum("mkj,mkj->mk", relative, edge_1)
        denominator = dot_00 * dot_11 - dot_01 * dot_01
        safe = np.abs(denominator) > 1.0e-20
        denominator = np.where(safe, denominator, 1.0)
        weight_1 = (dot_11 * dot_r0 - dot_01 * dot_r1) / denominator
        weight_2 = (dot_00 * dot_r1 - dot_01 * dot_r0) / denominator
        weight_0 = 1.0 - weight_1 - weight_2
        weights = np.stack((weight_0, weight_1, weight_2), axis=2)
        weights = np.clip(weights, 0.0, 1.0)
        weights /= np.maximum(weights.sum(axis=2, keepdims=True), 1.0e-12)
        projected = np.einsum("mkc,mkcj->mkj", weights, candidates)
        distance_squared = np.sum((projected - points) ** 2, axis=2)
        distance_squared = np.where(safe, distance_squared, np.inf)
        chosen = np.argmin(distance_squared, axis=1)
        rows = np.arange(end - start)
        chosen_faces = candidate_ids[rows, chosen]
        chosen_weights = weights[rows, chosen]
        output_uv[start:end] = np.einsum(
            "mc,mcd->md", chosen_weights, source_face_uv[chosen_faces]
        ).astype(np.float32)
        chosen_distances[start:end] = np.sqrt(distance_squared[rows, chosen])
    if not np.isfinite(output_uv).all():
        raise RuntimeError("Centroid-face transfer produced non-finite UV coordinates.")
    target_uv.data.foreach_set("uv", output_uv.reshape(-1))
    target.data.update()
    return {
        "target_corner_count": len(corner_points),
        "source_triangle_count": len(source_triangles),
        "candidate_count": candidate_count,
        "surface_distance_median": float(np.median(chosen_distances)),
        "surface_distance_q99": float(np.quantile(chosen_distances, 0.99)),
        "surface_distance_max": float(np.max(chosen_distances)),
    }


def assign_oral_materials(
    obj: bpy.types.Object,
    source_colors: np.ndarray,
    source_vertex_ids: np.ndarray,
    component_faces: np.ndarray,
) -> dict[str, int]:
    materials = (
        make_principled_material(
            "TopoRig_Teeth_Enamel", (0.91, 0.86, 0.72, 1.0), roughness=0.30, specular=0.42
        ),
        make_principled_material(
            "TopoRig_Gums_Tongue", (0.58, 0.20, 0.23, 1.0), roughness=0.42, specular=0.30
        ),
        make_principled_material(
            "TopoRig_Oral_Cavity", (0.17, 0.025, 0.032, 1.0), roughness=0.55, specular=0.16
        ),
    )
    for material in materials:
        obj.data.materials.append(material)
    local_colors = source_colors[source_vertex_ids]
    face_colors = local_colors[component_faces].mean(axis=1)
    teeth_reference = np.asarray((0.91, 0.86, 0.72), dtype=np.float32)
    gum_reference = np.asarray((0.58, 0.20, 0.23), dtype=np.float32)
    cavity_reference = np.asarray((0.34, 0.075, 0.085), dtype=np.float32)
    references = np.stack((teeth_reference, gum_reference, cavity_reference))
    assignments = np.square(face_colors[:, None, :] - references[None, :, :]).sum(axis=2).argmin(axis=1)
    for polygon, material_index in zip(obj.data.polygons, assignments.tolist()):
        polygon.material_index = int(material_index)
        polygon.use_smooth = True
    return {
        "teeth_face_count": int(np.count_nonzero(assignments == 0)),
        "gum_tongue_face_count": int(np.count_nonzero(assignments == 1)),
        "cavity_face_count": int(np.count_nonzero(assignments == 2)),
    }


def main() -> None:
    args = parse_args()
    source_path = args.source_rig_fbx.expanduser().resolve()
    component_path = args.components.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not component_path.is_file():
        raise FileNotFoundError(component_path)
    if output_path.suffix.lower() != ".fbx":
        raise ValueError("--output must end in .fbx")
    alignment_arguments = (
        args.source_neutral_predictions,
        args.source_landmarks,
        args.target_landmarks,
    )
    if any(value is not None for value in alignment_arguments) and not all(
        value is not None for value in alignment_arguments
    ):
        raise ValueError(
            "Source-neutral predictions, source landmarks, and target landmarks "
            "must be supplied together."
        )
    if args.include_all_prediction_blendshapes and not all(
        value is not None for value in alignment_arguments
    ):
        raise ValueError(
            "--include-all-prediction-blendshapes requires the prediction and "
            "both landmark files."
        )

    payload = load_components(component_path)
    neutral = demo.glb_vertices_to_blender_axes(
        torch.from_numpy(payload["neutral_vertices"])
    ).cpu().numpy()
    jaw = demo.glb_vertices_to_blender_axes(
        torch.from_numpy(payload["jaw_open_vertices"])
    ).cpu().numpy()
    faces = payload["faces"]
    colors = payload["vertex_colors"]
    surface_end = scalar(payload, "head_vertex_count") + scalar(
        payload, "lip_annulus_vertex_count"
    )
    surface_face_mask = np.all(faces < surface_end, axis=1)
    oral_face_mask = ~surface_face_mask
    if not np.any(surface_face_mask) or not np.any(oral_face_mask):
        raise ValueError("Expected both textured surface and oral component faces.")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    if source_path.suffix.lower() == ".glb":
        bpy.ops.import_scene.gltf(filepath=str(source_path))
    else:
        bpy.ops.import_scene.fbx(filepath=str(source_path))
    source_candidates = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and len(obj.data.polygons) > 0 and obj.data.uv_layers
    ]
    if not source_candidates:
        raise ValueError(f"No textured mesh object found in {source_path}")
    source = max(source_candidates, key=lambda obj: len(obj.data.polygons))
    bpy.context.view_layer.objects.active = source
    source.select_set(True)
    if source.data.shape_keys is not None:
        bpy.ops.object.shape_key_remove(all=True, apply_mix=False)
    source_alignment_report = None
    if all(value is not None for value in alignment_arguments):
        prediction_path = args.source_neutral_predictions.expanduser().resolve()
        with np.load(prediction_path, allow_pickle=False) as predictions:
            source_neutral = np.asarray(
                predictions["neutral_vertices"], dtype=np.float32
            )
        if len(source.data.vertices) != source_neutral.shape[0]:
            raise ValueError(
                "UV-source rig and neutral prediction vertex counts differ: "
                f"{len(source.data.vertices)} != {source_neutral.shape[0]}"
            )
        alignment_target_vertices = payload["neutral_vertices"]
        if args.alignment_target_components is not None:
            alignment_payload = load_components(
                args.alignment_target_components.expanduser().resolve()
            )
            alignment_target_vertices = alignment_payload["neutral_vertices"]
        aligned_source, source_alignment_report = robust_similarity_alignment(
            source_neutral,
            alignment_target_vertices,
            load_landmark_mapping(args.source_landmarks.expanduser().resolve()),
            load_landmark_mapping(args.target_landmarks.expanduser().resolve()),
        )
        aligned_source_blender = demo.glb_vertices_to_blender_axes(
            torch.from_numpy(aligned_source)
        ).cpu().numpy()
        source.parent = None
        source.matrix_world.identity()
        source.data.vertices.foreach_set(
            "co", np.ascontiguousarray(aligned_source_blender).reshape(-1)
        )
        source.data.update(calc_edges=True)
    source.hide_render = True

    head_neutral, head_jaw, head_faces, head_ids = subset_component(
        neutral, jaw, faces, surface_face_mask
    )
    oral_neutral, oral_jaw, oral_faces, oral_ids = subset_component(
        neutral, jaw, faces, oral_face_mask
    )
    prediction_names: list[str] = []
    mapped_prediction_targets = None
    prediction_mapping_report = None
    if args.include_all_prediction_blendshapes:
        prediction_neutral, prediction_names, prediction_targets = load_prediction_bank(
            args.source_neutral_predictions.expanduser().resolve()
        )
        alignment_target_vertices = payload["neutral_vertices"]
        if args.alignment_target_components is not None:
            alignment_payload = load_components(
                args.alignment_target_components.expanduser().resolve()
            )
            alignment_target_vertices = alignment_payload["neutral_vertices"]
        mapped_targets_glb, prediction_mapping_report = (
            map_prediction_bank_to_split_head(
                prediction_neutral=prediction_neutral,
                prediction_targets=prediction_targets,
                target_neutral=payload["neutral_vertices"][head_ids],
                source_landmarks=load_landmark_mapping(
                    args.source_landmarks.expanduser().resolve()
                ),
                target_landmarks=load_landmark_mapping(
                    args.target_landmarks.expanduser().resolve()
                ),
                alignment_target_vertices=alignment_target_vertices,
            )
        )
        mapped_prediction_targets = demo.glb_vertices_to_blender_axes(
            torch.from_numpy(mapped_targets_glb)
        ).cpu().numpy()

    head = create_mesh_object("TopoRig_Textured_Head", head_neutral, head_jaw, head_faces)
    oral = create_mesh_object("TopoRig_Complete_Oral_Assembly", oral_neutral, oral_jaw, oral_faces)
    if mapped_prediction_targets is not None:
        for name, coordinates in zip(
            prediction_names, mapped_prediction_targets, strict=True
        ):
            if name == "jawOpen":
                continue
            add_shape_key(head, name, coordinates)
    for polygon in head.data.polygons:
        polygon.use_smooth = True
    copy_source_materials(source, head)
    uv_transfer_report = None
    uv_source_reduction_report = None
    if args.uv_source_target_faces is not None:
        if args.uv_transfer_method != "nearest-face":
            raise ValueError(
                "--uv-source-target-faces is supported only with nearest-face transfer."
            )
        uv_source_reduction_report = reduce_uv_source(
            source, args.uv_source_target_faces
        )
    if args.uv_transfer_method == "topology-exact":
        uv_transfer_report = transfer_uvs_topology_exact(source, head)
    elif args.uv_transfer_method == "centroid-face":
        uv_transfer_report = transfer_uvs_centroid_face(source, head)
    elif args.uv_transfer_method == "nearest-face":
        transfer_uvs(source, head)
    elif args.uv_transfer_method == "baked-color":
        uv_transfer_report = transfer_baked_vertex_colors(source, head)
    else:
        transfer_uvs_nearest_vertex(source, head)
    oral_report = assign_oral_materials(oral, colors, oral_ids, oral_faces)

    bpy.ops.object.select_all(action="DESELECT")
    head.select_set(True)
    oral.select_set(True)
    bpy.context.view_layer.objects.active = head
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.exporting.fbx")
    temporary_path.unlink(missing_ok=True)
    result = bpy.ops.export_scene.fbx(
        filepath=str(temporary_path),
        use_selection=True,
        object_types={"MESH"},
        use_mesh_modifiers=False,
        add_leaf_bones=False,
        bake_anim=False,
        path_mode="COPY",
        embed_textures=True,
    )
    if "FINISHED" not in result or not temporary_path.is_file():
        raise RuntimeError(f"Blender did not write {temporary_path}")
    temporary_path.replace(output_path)

    report = {
        "status": "complete",
        "source_rig_fbx": str(source_path),
        "components": str(component_path),
        "output_fbx": str(output_path),
        "head_vertex_count": len(head.data.vertices),
        "head_face_count": len(head.data.polygons),
        "oral_vertex_count": len(oral.data.vertices),
        "oral_face_count": len(oral.data.polygons),
        "uv_layer": (
            head.data.uv_layers.active.name
            if head.data.uv_layers.active is not None
            else None
        ),
        "uv_transfer_method": args.uv_transfer_method,
        "uv_transfer": uv_transfer_report,
        "uv_source_reduction": uv_source_reduction_report,
        "source_alignment": source_alignment_report,
        "source_materials": [
            slot.material.name for slot in head.material_slots if slot.material is not None
        ],
        "packed_image_count": len(bpy.data.images),
        "oral_materials": oral_report,
        "prediction_mapping": prediction_mapping_report,
        "blendshapes": (
            prediction_names if prediction_names else ["jawOpen"]
        ),
    }
    report_path = output_path.with_suffix(".fbx_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] Textured lip-split FBX: {output_path}")
    print(f"[DONE] UV loops: {len(head.data.uv_layers.active.data)}")
    print(f"[DONE] Packed images: {len(bpy.data.images)}")


if __name__ == "__main__":
    main()
