#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp


DEFAULT_MESH = "./metaHumanHead_52shapekeys_01.fbx"
DEFAULT_VIEWS = ("neg_y", "pos_y", "neg_x", "pos_x")
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


BLENDER_HELPER = r"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion, Vector


VIEW_DIRECTIONS = {
    "neg_y": (0.0, -1.0, 0.0),
    "pos_y": (0.0, 1.0, 0.0),
    "neg_x": (-1.0, 0.0, 0.0),
    "pos_x": (1.0, 0.0, 0.0),
    "neg_z": (0.0, 0.0, -1.0),
    "pos_z": (0.0, 0.0, 1.0),
}

FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

NON_FACE_OBJECT_TOKENS = (
    "eye",
    "teeth",
    "tooth",
    "tongue",
    "gum",
    "lash",
    "cube",
)
TENSOR_MESH_NAMES = (
    "head_lod0_ORIGINAL",
    "eyeLeft_ORIGINAL",
    "eyeRight_ORIGINAL",
    "teeth_ORIGINAL",
)
EYE_IRIS_GROUPS = {
    "left": {
        "iris_ids": (473, 474, 475, 476, 477),
        "center_id": 473,
        "right_id": 474,
        "top_id": 475,
        "left_id": 476,
        "bottom_id": 477,
        "corner_ids": (263, 362),
    },
    "right": {
        "iris_ids": (468, 469, 470, 471, 472),
        "center_id": 468,
        "right_id": 469,
        "top_id": 470,
        "left_id": 471,
        "bottom_id": 472,
        "corner_ids": (33, 133),
    },
}


def parse_blender_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("render", "raycast"), required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--views", default="neg_y,pos_y,neg_x,pos_x")
    parser.add_argument("--view")
    parser.add_argument("--yaw-offset", type=float, default=0.0)
    parser.add_argument("--yaw-offsets", default="0")
    parser.add_argument("--roll-offset", type=float, default=0.0)
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--landmarks-json")
    parser.add_argument("--output-json")
    parser.add_argument("--debug-glb")
    parser.add_argument("--selection-mode", choices=("raycast", "scored", "depth"), default="raycast")
    parser.add_argument("--preserve-materials", action="store_true")
    parser.add_argument("--surface-hits-only", action="store_true")
    return parser.parse_args(argv)


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_mesh(mesh_path: Path) -> list[bpy.types.Object]:
    reset_scene()
    suffix = mesh_path.suffix.lower()
    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(mesh_path))
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    else:
        raise RuntimeError(f"Unsupported mesh format: {mesh_path.suffix}")

    mesh_objects = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj.data and len(obj.data.vertices) > 0
    ]
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects were imported from {mesh_path}")

    for obj in mesh_objects:
        obj.hide_render = False
        obj.hide_set(False)
        try:
            for poly in obj.data.polygons:
                poly.use_smooth = True
        except Exception:
            pass

    return mesh_objects


def all_bbox_corners(mesh_objects: list[bpy.types.Object]) -> list[Vector]:
    corners: list[Vector] = []
    for obj in mesh_objects:
        world = obj.matrix_world
        corners.extend(world @ Vector(corner) for corner in obj.bound_box)
    return corners


def bounds_from_corners(corners: list[Vector]) -> tuple[Vector, Vector, Vector]:
    min_corner = Vector(
        (
            min(corner.x for corner in corners),
            min(corner.y for corner in corners),
            min(corner.z for corner in corners),
        )
    )
    max_corner = Vector(
        (
            max(corner.x for corner in corners),
            max(corner.y for corner in corners),
            max(corner.z for corner in corners),
        )
    )
    center = (min_corner + max_corner) * 0.5
    return min_corner, max_corner, center


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    if direction.length == 0:
        direction = Vector((0.0, 0.0, -1.0))
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def get_or_create_camera() -> bpy.types.Object:
    camera_data = bpy.data.cameras.new("LandmarkRenderCamera")
    camera = bpy.data.objects.new("LandmarkRenderCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def setup_camera(
    mesh_objects: list[bpy.types.Object],
    view_name: str,
    width: int,
    height: int,
    yaw_offset: float = 0.0,
    roll_offset: float = 0.0,
) -> bpy.types.Object:
    if view_name not in VIEW_DIRECTIONS:
        raise RuntimeError(f"Unknown view '{view_name}'")

    corners = all_bbox_corners(mesh_objects)
    min_corner, max_corner, center = bounds_from_corners(corners)
    extents = max_corner - min_corner
    max_extent = max(extents.x, extents.y, extents.z, 1e-3)
    direction = Vector(VIEW_DIRECTIONS[view_name]).normalized()
    if yaw_offset != 0.0:
        direction.rotate(Matrix.Rotation(math.radians(yaw_offset), 4, "Z"))

    camera = get_or_create_camera()
    camera.location = center + direction * (max_extent * 3.0 + 1.0)
    look_at(camera, center)
    if roll_offset != 0.0:
        # Apply roll in camera-local space.  This must be identical for the
        # texture render and the later landmark ray-cast or the 2D landmarks
        # will be projected onto unrelated mesh surfaces.
        camera.rotation_mode = "QUATERNION"
        camera.rotation_quaternion = (
            camera.rotation_euler.to_quaternion()
            @ Quaternion((0.0, 0.0, 1.0), math.radians(roll_offset))
        )
    camera.data.type = "ORTHO"
    camera.data.clip_start = 0.001
    camera.data.clip_end = max_extent * 20.0 + 100.0

    # Fit the imported scene in the chosen camera frame.
    bpy.context.view_layer.update()
    inverse_camera = camera.matrix_world.inverted()
    local_corners = [inverse_camera @ corner for corner in corners]
    min_x = min(corner.x for corner in local_corners)
    max_x = max(corner.x for corner in local_corners)
    min_y = min(corner.y for corner in local_corners)
    max_y = max(corner.y for corner in local_corners)
    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)
    aspect = max(width / max(height, 1), 1e-3)
    camera.data.ortho_scale = max(span_y, span_x / aspect) * 1.18

    return camera


def configure_render(
    width: int,
    height: int,
    preserve_materials: bool = False,
) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.film_transparent = False
    scene.world = scene.world or bpy.data.worlds.new("World")
    scene.world.color = (0.43, 0.45, 0.47)

    engines = ("BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE")
    for engine in engines:
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue

    try:
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = (
            "TEXTURE" if preserve_materials else "SINGLE"
        )
        scene.display.shading.single_color = (0.72, 0.52, 0.43)
        scene.display.shading.show_cavity = True
        scene.display.shading.cavity_valley_factor = 1.6
        scene.display.shading.cavity_ridge_factor = 0.8
        scene.display.shading.background_type = "VIEWPORT"
        scene.display.shading.background_color = (0.43, 0.45, 0.47)
    except Exception:
        pass

    try:
        scene.eevee.taa_render_samples = 64
        scene.eevee.use_gtao = True
    except Exception:
        pass

    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "Medium High Contrast"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except Exception:
        pass


def material_is_blank(material: bpy.types.Material | None) -> bool:
    if material is None:
        return True
    if material.use_nodes:
        node = material.node_tree.nodes.get("Principled BSDF")
        if node:
            base = node.inputs.get("Base Color")
            alpha = node.inputs.get("Alpha")
            if base and tuple(base.default_value[:3]) == (0.8, 0.8, 0.8):
                return True
            if alpha and alpha.default_value <= 0.01:
                return True
    return False


def ensure_visible_materials(mesh_objects: list[bpy.types.Object]) -> None:
    skin = bpy.data.materials.new("LandmarkNeutralSkin")
    skin.use_nodes = True
    bsdf = skin.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.72, 0.52, 0.43, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.58

    for obj in mesh_objects:
        obj.data.materials.clear()
        obj.data.materials.append(skin)


def setup_light(camera: bpy.types.Object, mesh_objects: list[bpy.types.Object]) -> None:
    for obj in list(bpy.context.scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    corners = all_bbox_corners(mesh_objects)
    min_corner, max_corner, center = bounds_from_corners(corners)
    extents = max_corner - min_corner
    max_extent = max(extents.x, extents.y, extents.z, 1e-3)
    rotation = camera.matrix_world.to_quaternion()
    right = rotation @ Vector((1.0, 0.0, 0.0))
    up = rotation @ Vector((0.0, 1.0, 0.0))

    key_data = bpy.data.lights.new("LandmarkKeyLight", "AREA")
    key = bpy.data.objects.new("LandmarkKeyLight", key_data)
    bpy.context.collection.objects.link(key)
    key.location = camera.location - right * max_extent * 0.75 + up * max_extent * 0.55
    look_at(key, center)
    key.data.energy = 350.0
    key.data.size = max_extent * 2.2

    fill_data = bpy.data.lights.new("LandmarkFillLight", "POINT")
    fill = bpy.data.objects.new("LandmarkFillLight", fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = camera.location + right * max_extent * 0.7
    fill.data.energy = 45.0


def render_candidates(args: argparse.Namespace) -> None:
    mesh_objects = import_mesh(Path(args.mesh))
    bpy.context.scene.frame_set(max(1, int(args.frame)))
    bpy.context.view_layer.update()
    if not args.preserve_materials:
        ensure_visible_materials(mesh_objects)
    configure_render(
        args.width,
        args.height,
        preserve_materials=args.preserve_materials,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    views = [view.strip() for view in args.views.split(",") if view.strip()]
    yaw_offsets = [float(value.strip()) for value in args.yaw_offsets.split(",") if value.strip()]
    rendered = []
    for view in views:
        for yaw_offset in yaw_offsets:
            camera = setup_camera(
                mesh_objects,
                view,
                args.width,
                args.height,
                yaw_offset=yaw_offset,
                roll_offset=args.roll_offset,
            )
            setup_light(camera, mesh_objects)
            yaw_label = f"yaw_{yaw_offset:+.2f}".replace("+", "p").replace("-", "m").replace(".", "_")
            image_path = output_dir / f"candidate_{view}_{yaw_label}.png"
            if abs(yaw_offset) < 1e-9 and len(yaw_offsets) == 1:
                image_path = output_dir / f"candidate_{view}.png"
            bpy.context.scene.render.filepath = str(image_path)
            bpy.ops.render.render(write_still=True)
            rendered.append(
                {
                    "view": view,
                    "yaw_offset": yaw_offset,
                    "roll_offset": args.roll_offset,
                    "image": str(image_path),
                    "width": args.width,
                    "height": args.height,
                }
            )

            bpy.data.objects.remove(camera, do_unlink=True)

    with (output_dir / "views.json").open("w", encoding="utf-8") as handle:
        json.dump({"views": rendered}, handle, indent=2)


def base_blender_name(name: str) -> str:
    suffix = name.rpartition(".")[2]
    if len(suffix) == 3 and suffix.isdigit():
        return name[:-4]
    return name


def tensor_mesh_objects(mesh_objects: list[bpy.types.Object]) -> list[bpy.types.Object]:
    by_name = {base_blender_name(obj.name): obj for obj in mesh_objects}
    by_name.update({obj.name: obj for obj in mesh_objects})
    selected: list[bpy.types.Object] = []
    seen: set[str] = set()
    for name in TENSOR_MESH_NAMES:
        obj = by_name.get(name)
        if obj is not None and obj.name not in seen:
            selected.append(obj)
            seen.add(obj.name)
    if selected:
        return selected

    keyed_meshes = [obj for obj in mesh_objects if obj.data.shape_keys is not None]
    return sorted(
        keyed_meshes or mesh_objects,
        key=lambda obj: len(obj.data.vertices),
        reverse=True,
    )


def build_offsets(mesh_objects: list[bpy.types.Object]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    offset = 0
    for obj in mesh_objects:
        offsets[obj.name] = offset
        offset += len(obj.data.vertices)
    return offsets


def mapping_mesh_objects(mesh_objects: list[bpy.types.Object]) -> list[bpy.types.Object]:
    face_objects = [
        obj
        for obj in mesh_objects
        if not any(token in obj.name.lower() for token in NON_FACE_OBJECT_TOKENS)
    ]
    return face_objects or mesh_objects


def ray_from_camera(camera: bpy.types.Object, scene: bpy.types.Scene, x: float, y: float) -> tuple[Vector, Vector]:
    frame = camera.data.view_frame(scene=scene)
    top_right, bottom_right, bottom_left, top_left = frame
    u = x
    v = 1.0 - y

    bottom = bottom_left.lerp(bottom_right, u)
    top = top_left.lerp(top_right, u)
    local_origin = bottom.lerp(top, v)
    origin = camera.matrix_world @ local_origin

    if camera.data.type == "ORTHO":
        direction = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    else:
        direction = (origin - camera.location).normalized()
        origin = camera.location

    return origin, direction.normalized()


def face_oval_polygon(landmarks: list[dict]) -> list[tuple[float, float]]:
    landmark_by_index = {int(landmark["index"]): landmark for landmark in landmarks}
    polygon: list[tuple[float, float]] = []
    for index in FACE_OVAL_INDICES:
        landmark = landmark_by_index.get(index)
        if landmark is not None:
            polygon.append((float(landmark["x"]), float(landmark["y"])))
    return polygon


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return True

    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            intersect_x = ((xj - xi) * (y - yi) / (yj - yi)) + xi
            if x < intersect_x:
                inside = not inside
        j = i
    return inside


def distance_to_segment(
    x: float,
    y: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-12:
        return math.sqrt((x - ax) * (x - ax) + (y - ay) * (y - ay))
    t = max(0.0, min(1.0, ((x - ax) * abx + (y - ay) * aby) / denom))
    px = ax + t * abx
    py = ay + t * aby
    return math.sqrt((x - px) * (x - px) + (y - py) * (y - py))


def distance_to_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> float:
    if len(polygon) < 2:
        return 0.0
    return min(
        distance_to_segment(
            x,
            y,
            polygon[index][0],
            polygon[index][1],
            polygon[(index + 1) % len(polygon)][0],
            polygon[(index + 1) % len(polygon)][1],
        )
        for index in range(len(polygon))
    )


def polygon_scale(polygon: list[tuple[float, float]]) -> float:
    if not polygon:
        return 1.0
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return max(max(xs) - min(xs), max(ys) - min(ys), 1e-3)


def projected_vertex_candidates(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    mesh_objects: list[bpy.types.Object],
    offsets: dict[str, int],
    face_polygon: list[tuple[float, float]],
) -> list[dict]:
    view_direction = (camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))).normalized()
    candidates: list[dict] = []
    for obj in mesh_objects:
        if obj.name not in offsets:
            continue
        offset = offsets[obj.name]
        for vertex in obj.data.vertices:
            world_co = obj.matrix_world @ vertex.co
            projected = world_to_camera_view(scene, camera, world_co)
            if projected.z < 0:
                continue
            px = projected.x
            py = 1.0 - projected.y
            if px < -0.1 or px > 1.1 or py < -0.1 or py > 1.1:
                continue

            normal = (obj.matrix_world.to_3x3() @ vertex.normal).normalized()
            inside_face = point_in_polygon(px, py, face_polygon)
            candidates.append(
                {
                    "vertex_id": offset + vertex.index,
                    "x": float(px),
                    "y": float(py),
                    "z": float(projected.z),
                    "frontness": float(normal.dot(-view_direction)),
                    "inside_face": inside_face,
                    "face_distance": 0.0 if inside_face else distance_to_polygon(px, py, face_polygon),
                }
            )
    return candidates


def nearest_projected_vertex(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    mesh_objects: list[bpy.types.Object],
    offsets: dict[str, int],
    x: float,
    y: float,
) -> int | None:
    best_distance = float("inf")
    best_vertex: int | None = None
    for obj in mesh_objects:
        if obj.name not in offsets:
            continue
        offset = offsets[obj.name]
        for vertex in obj.data.vertices:
            world_co = obj.matrix_world @ vertex.co
            projected = world_to_camera_view(scene, camera, world_co)
            if projected.z < 0:
                continue
            px = projected.x
            py = 1.0 - projected.y
            if px < -0.25 or px > 1.25 or py < -0.25 or py > 1.25:
                continue
            distance = (px - x) * (px - x) + (py - y) * (py - y)
            if distance < best_distance:
                best_distance = distance
                best_vertex = offset + vertex.index
    return best_vertex


def nearest_scored_vertex(
    candidates: list[dict],
    x: float,
    y: float,
    face_scale: float,
) -> int | None:
    if not candidates:
        return None

    search_radii = [
        max(face_scale * 0.018, 0.006),
        max(face_scale * 0.035, 0.012),
        max(face_scale * 0.070, 0.024),
        max(face_scale * 0.140, 0.050),
    ]
    nearby: list[dict] = []
    for radius in search_radii:
        radius_sq = radius * radius
        nearby = [
            candidate
            for candidate in candidates
            if (candidate["x"] - x) * (candidate["x"] - x)
            + (candidate["y"] - y) * (candidate["y"] - y)
            <= radius_sq
        ]
        if nearby:
            break
    if not nearby:
        nearby = candidates

    best_score = float("inf")
    best_vertex: int | None = None
    for candidate in nearby:
        reprojection_error = math.sqrt(
            (candidate["x"] - x) * (candidate["x"] - x)
            + (candidate["y"] - y) * (candidate["y"] - y)
        )
        outside_penalty = candidate["face_distance"] * 4.0
        grazing_penalty = max(0.0, 0.04 - candidate["frontness"]) * face_scale * 0.18
        score = reprojection_error + outside_penalty + grazing_penalty
        if score < best_score:
            best_score = score
            best_vertex = candidate["vertex_id"]
    return best_vertex


def robust_range(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    ordered = sorted(values)
    if len(ordered) < 5:
        low = ordered[0]
        high = ordered[-1]
    else:
        low = ordered[int((len(ordered) - 1) * 0.05)]
        high = ordered[int((len(ordered) - 1) * 0.95)]
    if abs(high - low) < 1e-9:
        low = ordered[0]
        high = ordered[-1]
    if abs(high - low) < 1e-9:
        high = low + 1.0
    return (low, high)


def normalize_range(value: float, value_range: tuple[float, float]) -> float:
    low, high = value_range
    return max(0.0, min(1.0, (value - low) / (high - low)))


def landmark_z_range(landmarks: list[dict]) -> tuple[float, float]:
    return robust_range([float(landmark.get("z", 0.0)) for landmark in landmarks])


def candidate_z_range(candidates: list[dict]) -> tuple[float, float]:
    face_candidates = [candidate["z"] for candidate in candidates if candidate["inside_face"]]
    return robust_range(face_candidates or [candidate["z"] for candidate in candidates])


def nearest_depth_vertex(
    candidates: list[dict],
    x: float,
    y: float,
    z: float,
    face_scale: float,
    mp_z_range: tuple[float, float],
    mesh_z_range: tuple[float, float],
) -> int | None:
    if not candidates:
        return None

    search_radii = [
        max(face_scale * 0.025, 0.010),
        max(face_scale * 0.050, 0.020),
        max(face_scale * 0.100, 0.040),
        max(face_scale * 0.180, 0.075),
    ]
    nearby: list[dict] = []
    for radius in search_radii:
        radius_sq = radius * radius
        nearby = [
            candidate
            for candidate in candidates
            if (candidate["x"] - x) * (candidate["x"] - x)
            + (candidate["y"] - y) * (candidate["y"] - y)
            <= radius_sq
        ]
        if nearby:
            break
    if not nearby:
        nearby = candidates

    mp_depth = normalize_range(z, mp_z_range)
    depth_weight = max(face_scale * 0.075, 0.025)
    best_score = float("inf")
    best_vertex: int | None = None
    for candidate in nearby:
        reprojection_error = math.sqrt(
            (candidate["x"] - x) * (candidate["x"] - x)
            + (candidate["y"] - y) * (candidate["y"] - y)
        )
        mesh_depth = normalize_range(candidate["z"], mesh_z_range)
        depth_error = abs(mesh_depth - mp_depth)
        outside_penalty = candidate["face_distance"] * 2.0
        grazing_penalty = max(0.0, 0.02 - candidate["frontness"]) * face_scale * 0.10
        score = reprojection_error + depth_error * depth_weight + outside_penalty + grazing_penalty
        if score < best_score:
            best_score = score
            best_vertex = candidate["vertex_id"]
    return best_vertex


def nearest_hit_vertex(
    hit_obj: bpy.types.Object,
    location: Vector,
    face_index: int,
    offsets: dict[str, int],
) -> int | None:
    source_obj = getattr(hit_obj, "original", hit_obj)
    if source_obj.name not in offsets:
        return None
    if face_index < 0 or face_index >= len(source_obj.data.polygons):
        return None

    polygon = source_obj.data.polygons[face_index]
    best_vertex: int | None = None
    best_distance = float("inf")
    for vertex_index in polygon.vertices:
        world_co = source_obj.matrix_world @ source_obj.data.vertices[vertex_index].co
        distance = (world_co - location).length_squared
        if distance < best_distance:
            best_distance = distance
            best_vertex = offsets[source_obj.name] + vertex_index
    return best_vertex


def hit_surface_anchor(
    hit_obj: bpy.types.Object,
    location: Vector,
    face_index: int,
    offsets: dict[str, int],
) -> dict | None:
    source_obj = getattr(hit_obj, "original", hit_obj)
    if source_obj.name not in offsets:
        return None
    if face_index < 0 or face_index >= len(source_obj.data.polygons):
        return None

    polygon = source_obj.data.polygons[face_index]
    polygon_vertices = list(polygon.vertices)
    if len(polygon_vertices) < 3:
        return None

    world_vertices = [
        source_obj.matrix_world @ source_obj.data.vertices[index].co
        for index in polygon_vertices
    ]
    best = None
    best_score = float("inf")
    for triangle_index in range(1, len(polygon_vertices) - 1):
        local_ids = (0, triangle_index, triangle_index + 1)
        first, second, third = (world_vertices[index] for index in local_ids)
        edge_first = second - first
        edge_second = third - first
        point_offset = location - first
        dot00 = edge_first.dot(edge_first)
        dot01 = edge_first.dot(edge_second)
        dot11 = edge_second.dot(edge_second)
        dot20 = point_offset.dot(edge_first)
        dot21 = point_offset.dot(edge_second)
        denominator = dot00 * dot11 - dot01 * dot01
        if abs(denominator) <= 1.0e-20:
            continue
        second_weight = (dot11 * dot20 - dot01 * dot21) / denominator
        third_weight = (dot00 * dot21 - dot01 * dot20) / denominator
        first_weight = 1.0 - second_weight - third_weight
        weights = (first_weight, second_weight, third_weight)
        reconstructed = (
            first * first_weight + second * second_weight + third * third_weight
        )
        outside = sum(max(-float(weight), 0.0) for weight in weights)
        score = outside * outside + (reconstructed - location).length_squared
        if score < best_score:
            best_score = score
            best = (local_ids, weights, reconstructed)

    if best is None:
        return None
    local_ids, weights, reconstructed = best
    offset = offsets[source_obj.name]
    return {
        "vertex_ids": [
            int(offset + polygon_vertices[index]) for index in local_ids
        ],
        "barycentric_weights": [float(weight) for weight in weights],
        "position": {
            "x": float(reconstructed.x),
            "y": float(reconstructed.y),
            "z": float(reconstructed.z),
        },
        "object_name": source_obj.name,
        "polygon_index": int(face_index),
    }


def surface_anchor_uv(
    hit_obj: bpy.types.Object,
    anchor: dict,
    offsets: dict[str, int],
) -> list[float] | None:
    source_obj = getattr(hit_obj, "original", hit_obj)
    if source_obj.name not in offsets or not source_obj.data.uv_layers:
        return None
    polygon_index = int(anchor["polygon_index"])
    if polygon_index < 0 or polygon_index >= len(source_obj.data.polygons):
        return None

    polygon = source_obj.data.polygons[polygon_index]
    polygon_vertices = list(polygon.vertices)
    polygon_loops = list(polygon.loop_indices)
    local_vertex_ids = [
        int(vertex_id) - offsets[source_obj.name]
        for vertex_id in anchor["vertex_ids"]
    ]
    loop_ids: list[int] = []
    for vertex_id in local_vertex_ids:
        try:
            polygon_offset = polygon_vertices.index(vertex_id)
        except ValueError:
            return None
        loop_ids.append(polygon_loops[polygon_offset])

    uv_data = source_obj.data.uv_layers.active.data
    result = Vector((0.0, 0.0))
    for loop_id, weight in zip(loop_ids, anchor["barycentric_weights"]):
        result += Vector(uv_data[loop_id].uv) * float(weight)
    if not math.isfinite(result.x) or not math.isfinite(result.y):
        return None
    return [float(result.x), float(result.y)]


def surface_base_color_sample(
    hit_obj: bpy.types.Object,
    face_index: int,
    uv: list[float],
) -> tuple[float, list[float]] | None:
    source_obj = getattr(hit_obj, "original", hit_obj)
    if face_index < 0 or face_index >= len(source_obj.data.polygons):
        return None
    polygon = source_obj.data.polygons[face_index]
    material_index = int(polygon.material_index)
    if material_index < 0 or material_index >= len(source_obj.material_slots):
        return None
    material = source_obj.material_slots[material_index].material
    if material is None or not material.use_nodes or material.node_tree is None:
        return None

    base_image = None
    for node in material.node_tree.nodes:
        if node.type != "BSDF_PRINCIPLED":
            continue
        base_color = node.inputs.get("Base Color")
        if base_color is None:
            continue
        for link in base_color.links:
            if link.from_node.type == "TEX_IMAGE" and link.from_node.image is not None:
                base_image = link.from_node.image
                break
        if base_image is not None:
            break
    if base_image is None or base_image.size[0] < 1 or base_image.size[1] < 1:
        return None

    width, height = int(base_image.size[0]), int(base_image.size[1])
    x = max(0, min(width - 1, int((float(uv[0]) % 1.0) * width)))
    y = max(0, min(height - 1, int((float(uv[1]) % 1.0) * height)))
    pixel_offset = 4 * (y * width + x)
    red = float(base_image.pixels[pixel_offset])
    green = float(base_image.pixels[pixel_offset + 1])
    blue = float(base_image.pixels[pixel_offset + 2])
    return (
        float(0.2126 * red + 0.7152 * green + 0.0722 * blue),
        [red, green, blue],
    )


def raycast_surface_anchor_candidates(
    scene: bpy.types.Scene,
    depsgraph,
    camera: bpy.types.Object,
    origin: Vector,
    direction: Vector,
    offsets: dict[str, int],
    max_hits: int = 16,
    sample_base_color: bool = False,
    require_uv: bool = True,
) -> list[dict]:
    candidates: list[dict] = []
    cursor = origin.copy()
    unit_direction = direction.normalized()
    max_distance = float(camera.data.clip_end) * 2.0
    travelled = 0.0
    repeated_hits = 0
    previous_key = None
    epsilon = max(float(camera.data.clip_end) * 1.0e-7, 1.0e-6)

    for _ in range(max_hits * 3):
        remaining = max_distance - travelled
        if remaining <= epsilon or len(candidates) >= max_hits:
            break
        hit, location, _normal, face_index, hit_obj, _matrix = scene.ray_cast(
            depsgraph,
            cursor,
            unit_direction,
            distance=remaining,
        )
        if not hit or hit_obj is None:
            break

        step = max(float((location - cursor).dot(unit_direction)), 0.0)
        travelled += step
        source_obj = getattr(hit_obj, "original", hit_obj)
        key = (source_obj.name, int(face_index))
        if key == previous_key:
            repeated_hits += 1
        else:
            repeated_hits = 0
        previous_key = key

        anchor = hit_surface_anchor(hit_obj, location, face_index, offsets)
        if anchor is not None:
            uv = surface_anchor_uv(hit_obj, anchor, offsets)
            if (uv is not None or not require_uv) and not any(
                candidate["object_name"] == anchor["object_name"]
                and candidate["polygon_index"] == anchor["polygon_index"]
                for candidate in candidates
            ):
                if uv is not None:
                    anchor["uv"] = uv
                anchor["ray_depth"] = float((location - origin).dot(unit_direction))
                if sample_base_color and uv is not None:
                    color_sample = surface_base_color_sample(
                        hit_obj,
                        face_index,
                        uv,
                    )
                    if color_sample is not None:
                        (
                            anchor["base_color_luminance"],
                            anchor["base_color_rgb"],
                        ) = color_sample
                candidates.append(anchor)

        advance = epsilon * (4.0 if repeated_hits else 1.0)
        cursor = location + unit_direction * advance
        travelled += advance
        if repeated_hits >= 3:
            break
    return candidates


def anchor_position(anchor: dict) -> Vector:
    position = anchor["position"]
    return Vector((float(position["x"]), float(position["y"]), float(position["z"])))


def validate_eye_iris_candidate(
    anchors: list[dict],
    group: dict,
    surface_anchors: dict[str, dict | None],
) -> dict | None:
    if len({anchor["object_name"] for anchor in anchors}) != 1:
        return None
    corner_anchors = [surface_anchors.get(str(value)) for value in group["corner_ids"]]
    if any(anchor is None for anchor in corner_anchors):
        return None

    by_id = dict(zip(group["iris_ids"], anchors))
    center_anchor = by_id[group["center_id"]]
    center_uv = Vector(center_anchor["uv"])
    center_position = anchor_position(center_anchor)
    edge_ids = [value for value in group["iris_ids"] if value != group["center_id"]]
    uv_radii = [
        (Vector(by_id[value]["uv"]) - center_uv).length for value in edge_ids
    ]
    position_radii = [
        (anchor_position(by_id[value]) - center_position).length for value in edge_ids
    ]
    if min(uv_radii) <= 1.0e-6 or min(position_radii) <= 1.0e-7:
        return None
    if max(uv_radii) > min(uv_radii) * 4.0:
        return None
    if max(position_radii) > min(position_radii) * 4.0:
        return None

    corner_positions = [anchor_position(anchor) for anchor in corner_anchors]
    eye_width = (corner_positions[0] - corner_positions[1]).length
    iris_radius = sum(position_radii) / len(position_radii)
    if eye_width <= 1.0e-7 or iris_radius <= 1.0e-7:
        return None
    eye_to_iris_ratio = eye_width / iris_radius
    if not 2.5 <= eye_to_iris_ratio <= 8.0:
        return None

    uv_iris_radius = sum(uv_radii) / len(uv_radii)
    estimated_uv_eye_width = uv_iris_radius * eye_to_iris_ratio
    if not 1.0e-4 <= estimated_uv_eye_width <= 0.25:
        return None

    horizontal = Vector(by_id[group["right_id"]]["uv"]) - center_uv
    vertical = Vector(by_id[group["bottom_id"]]["uv"]) - center_uv
    determinant = abs(horizontal.x * vertical.y - horizontal.y * vertical.x)
    axis_independence = determinant / max(horizontal.length * vertical.length, 1.0e-12)
    if axis_independence < 0.15:
        return None

    ray_depths = [float(anchor["ray_depth"]) for anchor in anchors]
    if max(ray_depths) - min(ray_depths) > eye_width:
        return None
    return {
        "object_name": anchors[0]["object_name"],
        "eye_width_3d": float(eye_width),
        "iris_radius_3d": float(iris_radius),
        "eye_to_iris_ratio": float(eye_to_iris_ratio),
        "estimated_eye_width_uv": float(estimated_uv_eye_width),
        "axis_independence": float(axis_independence),
        "uv_radius_variation": float(max(uv_radii) / min(uv_radii)),
        "position_radius_variation": float(
            max(position_radii) / min(position_radii)
        ),
        "ray_depth_range": [float(min(ray_depths)), float(max(ray_depths))],
    }


def validate_visible_eye_iris_candidate(
    anchors_by_id: dict[int, dict],
    group: dict,
    surface_anchors: dict[str, dict | None],
    minimum_edge_count: int = 3,
) -> dict | None:
    center_id = group["center_id"]
    if center_id not in anchors_by_id:
        return None
    edge_ids = [
        value for value in group["iris_ids"]
        if value != center_id and value in anchors_by_id
    ]
    if len(edge_ids) < minimum_edge_count:
        return None
    if not ({group["right_id"], group["left_id"]} & set(edge_ids)):
        return None
    if not ({group["top_id"], group["bottom_id"]} & set(edge_ids)):
        return None
    anchors = list(anchors_by_id.values())
    if len({anchor["object_name"] for anchor in anchors}) != 1:
        return None

    corner_anchors = [surface_anchors.get(str(value)) for value in group["corner_ids"]]
    if any(anchor is None for anchor in corner_anchors):
        return None
    center_anchor = anchors_by_id[center_id]
    center_uv = Vector(center_anchor["uv"])
    center_position = anchor_position(center_anchor)
    uv_radii = [
        (Vector(anchors_by_id[value]["uv"]) - center_uv).length
        for value in edge_ids
    ]
    position_radii = [
        (anchor_position(anchors_by_id[value]) - center_position).length
        for value in edge_ids
    ]
    if min(uv_radii) <= 1.0e-6 or min(position_radii) <= 1.0e-7:
        return None
    if max(uv_radii) > min(uv_radii) * 3.0:
        return None
    if max(position_radii) > min(position_radii) * 3.0:
        return None

    corner_positions = [anchor_position(anchor) for anchor in corner_anchors]
    eye_width = (corner_positions[0] - corner_positions[1]).length
    iris_radius = max(position_radii)
    if eye_width <= 1.0e-7 or iris_radius <= 1.0e-7:
        return None
    eye_to_iris_ratio = eye_width / iris_radius
    if not 2.5 <= eye_to_iris_ratio <= 8.0:
        return None
    estimated_uv_eye_width = max(uv_radii) * eye_to_iris_ratio
    if not 1.0e-4 <= estimated_uv_eye_width <= 0.25:
        return None

    if group["right_id"] in anchors_by_id:
        horizontal = Vector(anchors_by_id[group["right_id"]]["uv"]) - center_uv
    else:
        horizontal = center_uv - Vector(anchors_by_id[group["left_id"]]["uv"])
    if group["bottom_id"] in anchors_by_id:
        vertical = Vector(anchors_by_id[group["bottom_id"]]["uv"]) - center_uv
    else:
        vertical = center_uv - Vector(anchors_by_id[group["top_id"]]["uv"])
    determinant = abs(horizontal.x * vertical.y - horizontal.y * vertical.x)
    axis_independence = determinant / max(horizontal.length * vertical.length, 1.0e-12)
    if axis_independence < 0.15:
        return None

    ray_depths = [float(anchor["ray_depth"]) for anchor in anchors]
    if max(ray_depths) - min(ray_depths) > eye_width * 0.35:
        return None
    return {
        "object_name": center_anchor["object_name"],
        "valid_iris_ids": [int(center_id), *[int(value) for value in edge_ids]],
        "eye_width_3d": float(eye_width),
        "iris_radius_3d": float(iris_radius),
        "eye_to_iris_ratio": float(eye_to_iris_ratio),
        "estimated_eye_width_uv": float(estimated_uv_eye_width),
        "axis_independence": float(axis_independence),
        "uv_radius_variation": float(max(uv_radii) / min(uv_radii)),
        "position_radius_variation": float(
            max(position_radii) / min(position_radii)
        ),
        "ray_depth_range": [float(min(ray_depths)), float(max(ray_depths))],
    }


def select_eye_iris_surface_anchors(
    candidates: dict[str, list[dict]],
    surface_anchors: dict[str, dict | None],
    mapping: dict[str, int | None],
) -> dict[str, dict]:
    report: dict[str, dict] = {}
    for eye, group in EYE_IRIS_GROUPS.items():
        candidate_lists = [candidates.get(str(value), []) for value in group["iris_ids"]]
        if not candidate_lists[0] or sum(bool(values) for values in candidate_lists[1:]) < 3:
            report[eye] = {"status": "unresolved", "reason": "missing ray hits"}
            continue

        selected = None
        selected_metrics = None
        selected_ranks = None
        selected_score = float("inf")
        center_candidates = candidates.get(str(group["center_id"]), [])
        corner_anchors = [surface_anchors.get(str(value)) for value in group["corner_ids"]]
        visible_layers: list[dict] = []
        if not any(anchor is None for anchor in corner_anchors):
            corner_positions = [anchor_position(anchor) for anchor in corner_anchors]
            eye_width = (corner_positions[0] - corner_positions[1]).length
            for center_rank, center_anchor in enumerate(center_candidates):
                center_uv = Vector(center_anchor["uv"])
                center_position = anchor_position(center_anchor)
                anchors_by_id = {group["center_id"]: center_anchor}
                ranks_by_id = {group["center_id"]: center_rank}
                for edge_id in group["iris_ids"]:
                    if edge_id == group["center_id"]:
                        continue
                    options = []
                    for rank, anchor in enumerate(candidates.get(str(edge_id), [])):
                        if anchor["object_name"] != center_anchor["object_name"]:
                            continue
                        uv_distance = (Vector(anchor["uv"]) - center_uv).length
                        position_distance = (
                            anchor_position(anchor) - center_position
                        ).length
                        depth_distance = abs(
                            float(anchor["ray_depth"])
                            - float(center_anchor["ray_depth"])
                        )
                        if (
                            1.0e-6 < uv_distance <= 0.08
                            and 1.0e-7 < position_distance <= eye_width * 0.5
                            and depth_distance <= eye_width * 0.25
                        ):
                            options.append(
                                (
                                    depth_distance / max(eye_width, 1.0e-8)
                                    + uv_distance / 0.08,
                                    rank,
                                    anchor,
                                )
                            )
                    if options:
                        _score, rank, anchor = min(options, key=lambda value: value[0])
                        anchors_by_id[edge_id] = anchor
                        ranks_by_id[edge_id] = rank
                coherent_choice = None
                available_edge_ids = [
                    value for value in group["iris_ids"]
                    if value != group["center_id"] and value in anchors_by_id
                ]
                for edge_count in range(len(available_edge_ids), 1, -1):
                    choices = []
                    for edge_subset in itertools.combinations(
                        available_edge_ids,
                        edge_count,
                    ):
                        subset = {
                            group["center_id"]: center_anchor,
                            **{value: anchors_by_id[value] for value in edge_subset},
                        }
                        subset_metrics = validate_visible_eye_iris_candidate(
                            subset,
                            group,
                            surface_anchors,
                            minimum_edge_count=2,
                        )
                        if subset_metrics is not None:
                            choices.append(
                                (
                                    subset_metrics["uv_radius_variation"]
                                    + subset_metrics["position_radius_variation"],
                                    subset,
                                    subset_metrics,
                                )
                            )
                    if choices:
                        coherent_choice = min(choices, key=lambda value: value[0])
                        break
                if coherent_choice is None:
                    metrics = None
                else:
                    _coherence_score, anchors_by_id, metrics = coherent_choice
                center_luminance = center_anchor.get("base_color_luminance")
                if (
                    metrics is not None
                    and center_luminance is not None
                    and float(center_luminance) <= 0.20
                ):
                    valid_ids = metrics["valid_iris_ids"]
                    visible_layers.append(
                        {
                            "center_luminance": float(center_luminance),
                            "depth_ranks": [
                                int(ranks_by_id[value]) for value in valid_ids
                            ],
                            "surface_anchors": {
                                str(value): anchors_by_id[value] for value in valid_ids
                            },
                            **metrics,
                        }
                    )

        if visible_layers:
            front_depth = sum(visible_layers[0]["ray_depth_range"]) * 0.5
            maximum_layer_depth = (
                front_depth + visible_layers[0]["eye_width_3d"] * 0.60
            )
            visible_layers = [
                layer for layer in visible_layers
                if sum(layer["ray_depth_range"]) * 0.5 <= maximum_layer_depth
            ]
            primary_layer = next(
                (
                    layer for layer in visible_layers
                    if len(layer["valid_iris_ids"]) >= 4
                ),
                visible_layers[0],
            )
            selected_metrics = {
                key: value
                for key, value in primary_layer.items()
                if key not in {"surface_anchors", "depth_ranks"}
            }
            selected_ranks = primary_layer["depth_ranks"]
            selected = [
                primary_layer["surface_anchors"][str(value)]
                for value in selected_metrics["valid_iris_ids"]
            ]
            selected_score = float(selected_ranks[0]) * 1.0e-3

        if selected is not None:
            for landmark_id, anchor in zip(
                selected_metrics["valid_iris_ids"], selected
            ):
                surface_anchors[str(landmark_id)] = anchor
                weights = [float(value) for value in anchor["barycentric_weights"]]
                mapping[str(landmark_id)] = int(
                    anchor["vertex_ids"][weights.index(max(weights))]
                )
            report[eye] = {
                "status": "selected_visible_consensus",
                "depth_ranks": [int(value) for value in selected_ranks],
                "selection_score": float(selected_score),
                "visible_layers": visible_layers,
                **selected_metrics,
            }
            continue

        for rank in range(min(len(values) for values in candidate_lists)):
            rank_anchors = [values[rank] for values in candidate_lists]
            metrics = validate_eye_iris_candidate(rank_anchors, group, surface_anchors)
            if metrics is not None:
                selected = rank_anchors
                selected_metrics = metrics
                selected_ranks = [rank] * len(rank_anchors)
                selected_score = float(rank) * 1.0e-3
                break

        if not any(anchor is None for anchor in corner_anchors):
            corner_positions = [anchor_position(anchor) for anchor in corner_anchors]
            eye_width = (corner_positions[0] - corner_positions[1]).length
            for center_rank, center_anchor in enumerate(candidate_lists[0]):
                center_uv = Vector(center_anchor["uv"])
                center_position = anchor_position(center_anchor)
                edge_options: list[list[tuple[int, dict]]] = []
                for values in candidate_lists[1:]:
                    options = []
                    for rank, anchor in enumerate(values):
                        if anchor["object_name"] != center_anchor["object_name"]:
                            continue
                        uv_distance = (Vector(anchor["uv"]) - center_uv).length
                        position_distance = (
                            anchor_position(anchor) - center_position
                        ).length
                        if (
                            1.0e-6 < uv_distance <= 0.08
                            and 1.0e-7 < position_distance <= eye_width * 0.5
                        ):
                            options.append((rank, anchor))
                    if not options:
                        edge_options = []
                        break
                    edge_options.append(options)
                if not edge_options:
                    continue

                for edge_choices in itertools.product(*edge_options):
                    ranks = [center_rank, *[choice[0] for choice in edge_choices]]
                    anchors = [center_anchor, *[choice[1] for choice in edge_choices]]
                    metrics = validate_eye_iris_candidate(
                        anchors,
                        group,
                        surface_anchors,
                    )
                    if metrics is None:
                        continue
                    depth_min, depth_max = metrics["ray_depth_range"]
                    normalized_depth_spread = (
                        (depth_max - depth_min) / metrics["eye_width_3d"]
                    )
                    score = (
                        normalized_depth_spread
                        + 0.08 * (max(ranks) - min(ranks))
                        + 0.04 * (metrics["uv_radius_variation"] - 1.0)
                        + 0.04 * (metrics["position_radius_variation"] - 1.0)
                        + 1.0e-3 * sum(ranks) / len(ranks)
                    )
                    if score < selected_score:
                        selected = anchors
                        selected_metrics = metrics
                        selected_ranks = ranks
                        selected_score = score

        if selected is None:
            report[eye] = {
                "status": "unresolved",
                "reason": "no coherent common ray-depth layer",
                "candidate_counts": [len(values) for values in candidate_lists],
            }
            continue

        for landmark_id, anchor in zip(group["iris_ids"], selected):
            surface_anchors[str(landmark_id)] = anchor
            weights = [float(value) for value in anchor["barycentric_weights"]]
            mapping[str(landmark_id)] = int(anchor["vertex_ids"][weights.index(max(weights))])
        report[eye] = {
            "status": "selected",
            "depth_ranks": [int(value) for value in selected_ranks],
            "selection_score": float(selected_score),
            **selected_metrics,
        }
    return report


def selected_vertex_world_positions(
    mesh_objects: list[bpy.types.Object],
    offsets: dict[str, int],
    selected: set[int],
) -> list[Vector]:
    positions: list[Vector] = []
    for obj in mesh_objects:
        if obj.name not in offsets:
            continue
        offset = offsets[obj.name]
        for vertex_id in selected:
            local_index = vertex_id - offset
            if 0 <= local_index < len(obj.data.vertices):
                positions.append(obj.matrix_world @ obj.data.vertices[local_index].co)
    return positions


def vertex_world_position(
    mesh_objects: list[bpy.types.Object],
    offsets: dict[str, int],
    vertex_id: int,
) -> Vector | None:
    for obj in mesh_objects:
        if obj.name not in offsets:
            continue
        offset = offsets[obj.name]
        local_index = vertex_id - offset
        if 0 <= local_index < len(obj.data.vertices):
            return obj.matrix_world @ obj.data.vertices[local_index].co
    return None


def add_debug_markers(mesh_objects: list[bpy.types.Object], offsets: dict[str, int], selected: set[int]) -> None:
    red = bpy.data.materials.new("DetectedMediapipeVerticesRed")
    red.diffuse_color = (1.0, 0.0, 0.0, 1.0)
    red.use_nodes = True
    bsdf = red.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.45

    corners = all_bbox_corners(mesh_objects)
    min_corner, max_corner, _center = bounds_from_corners(corners)
    extents = max_corner - min_corner
    radius = max(extents.x, extents.y, extents.z, 1e-3) * 0.008

    for index, position in enumerate(selected_vertex_world_positions(mesh_objects, offsets, selected)):
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=8,
            ring_count=4,
            radius=radius,
            location=position,
        )
        marker = bpy.context.object
        marker.name = f"mediapipe_vertex_marker_{index:03d}"
        marker.data.materials.append(red)


def project_mapping(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    mesh_objects: list[bpy.types.Object],
    offsets: dict[str, int],
    mapping: dict[str, int | None],
) -> dict[str, dict[str, float] | None]:
    projections: dict[str, dict[str, float] | None] = {}
    for mp_index, vertex_id in mapping.items():
        if vertex_id is None:
            projections[mp_index] = None
            continue
        position = vertex_world_position(mesh_objects, offsets, vertex_id)
        if position is None:
            projections[mp_index] = None
            continue
        projected = world_to_camera_view(scene, camera, position)
        projections[mp_index] = {
            "x": float(projected.x),
            "y": float(1.0 - projected.y),
            "z": float(projected.z),
        }
    return projections


def vertex_positions(
    mesh_objects: list[bpy.types.Object],
    offsets: dict[str, int],
    mapping: dict[str, int | None],
) -> dict[str, dict[str, float] | None]:
    positions: dict[str, dict[str, float] | None] = {}
    for mp_index, vertex_id in mapping.items():
        if vertex_id is None:
            positions[mp_index] = None
            continue
        position = vertex_world_position(mesh_objects, offsets, vertex_id)
        if position is None:
            positions[mp_index] = None
            continue
        positions[mp_index] = {
            "x": float(position.x),
            "y": float(position.y),
            "z": float(position.z),
        }
    return positions


def export_debug_glb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for obj in list(bpy.context.scene.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    try:
        bpy.ops.export_scene.gltf(
            filepath=str(path),
            export_format="GLB",
            export_materials="EXPORT",
            export_yup=True,
        )
    except TypeError:
        bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB")


def raycast_landmarks(args: argparse.Namespace) -> None:
    mesh_objects = import_mesh(Path(args.mesh))
    configure_render(args.width, args.height)
    camera = setup_camera(
        mesh_objects,
        args.view,
        args.width,
        args.height,
        yaw_offset=args.yaw_offset,
        roll_offset=args.roll_offset,
    )
    bpy.context.view_layer.update()

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    tensor_objects = tensor_mesh_objects(mesh_objects)
    offsets = build_offsets(tensor_objects)

    with Path(args.landmarks_json).open("r", encoding="utf-8") as handle:
        landmark_data = json.load(handle)
    landmarks = landmark_data["landmarks"]
    candidates: list[dict] = []
    face_scale = 1.0
    mp_z_range = (0.0, 1.0)
    mesh_z_range = (0.0, 1.0)
    if args.selection_mode in {"scored", "depth"}:
        face_polygon = face_oval_polygon(landmarks)
        face_scale = polygon_scale(face_polygon)
        candidate_objects = mapping_mesh_objects(tensor_objects)
        candidates = projected_vertex_candidates(scene, camera, candidate_objects, offsets, face_polygon)
        if args.selection_mode == "depth":
            mp_z_range = landmark_z_range(landmarks)
            mesh_z_range = candidate_z_range(candidates)

    mapping: dict[str, int | None] = {}
    surface_anchors: dict[str, dict | None] = {}
    surface_anchor_candidates: dict[str, list[dict]] = {}
    iris_landmark_ids = {
        value
        for group in EYE_IRIS_GROUPS.values()
        for value in group["iris_ids"]
    }
    for landmark in landmarks:
        mp_index = int(landmark["index"])
        x = float(landmark["x"])
        y = float(landmark["y"])
        vertex_id: int | None = None
        surface_anchor: dict | None = None
        if args.selection_mode == "scored":
            vertex_id = nearest_scored_vertex(candidates, x, y, face_scale)
        elif args.selection_mode == "depth":
            vertex_id = nearest_depth_vertex(
                candidates,
                x,
                y,
                float(landmark.get("z", 0.0)),
                face_scale,
                mp_z_range,
                mesh_z_range,
            )
        if vertex_id is None:
            origin, direction = ray_from_camera(camera, scene, x, y)
            ray_candidates = raycast_surface_anchor_candidates(
                scene,
                depsgraph,
                camera,
                origin,
                direction,
                offsets,
                # Physical eye assemblies can be split across the center and
                # four iris-edge rays.  Cache texture luminance on every iris
                # ray so the component selector can reject skin/eyelid patches
                # while retaining dark iris and corneal layers.
                sample_base_color=mp_index in iris_landmark_ids,
                require_uv=not args.surface_hits_only,
            )
            for candidate_anchor in ray_candidates:
                candidate_anchor["screen_position"] = {
                    "x": float(x),
                    "y": float(y),
                }
            if mp_index in iris_landmark_ids:
                surface_anchor_candidates[str(mp_index)] = ray_candidates
            if ray_candidates:
                surface_anchor = ray_candidates[0]
                weights = [float(value) for value in surface_anchor["barycentric_weights"]]
                vertex_id = int(
                    surface_anchor["vertex_ids"][weights.index(max(weights))]
                )
        if vertex_id is None and not args.surface_hits_only:
            vertex_id = nearest_projected_vertex(scene, camera, tensor_objects, offsets, x, y)
        mapping[str(mp_index)] = vertex_id
        surface_anchors[str(mp_index)] = surface_anchor

    surface_anchor_selection = select_eye_iris_surface_anchors(
        surface_anchor_candidates,
        surface_anchors,
        mapping,
    )
    selected = {value for value in mapping.values() if value is not None}
    vertex_projections = project_mapping(scene, camera, mesh_objects, offsets, mapping)
    if args.debug_glb:
        add_debug_markers(mesh_objects, offsets, selected)
        export_debug_glb(Path(args.debug_glb))

    result = {
        "mapping": mapping,
        "surface_anchors": surface_anchors,
        "surface_anchor_selection": surface_anchor_selection,
        "iris_surface_anchor_candidates": surface_anchor_candidates,
        "mesh_object_offsets": offsets,
        "vertex_positions": vertex_positions(mesh_objects, offsets, mapping),
        "vertex_projections": vertex_projections,
        "selection_mode": args.selection_mode,
        "selected_view": args.view,
        "yaw_offset": args.yaw_offset,
        "roll_offset": args.roll_offset,
    }
    with Path(args.output_json).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_blender_args()
    if args.mode == "render":
        render_candidates(args)
    elif args.mode == "raycast":
        raycast_landmarks(args)


if __name__ == "__main__":
    main()
"""


@dataclass(frozen=True)
class DetectionCandidate:
    view: str
    yaw_offset: float
    roll_offset: float
    image_path: Path
    landmarks: list[Any]
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a frontal face image from an FBX/GLB mesh, detect MediaPipe "
            "face landmarks, and map each MediaPipe landmark id to a mesh vertex id."
        )
    )
    parser.add_argument(
        "--mesh",
        default=DEFAULT_MESH,
        help=f"FBX, GLB, or GLTF mesh path. Defaults to {DEFAULT_MESH}.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write face_debug.png, face_debug_landmark.png, a reprojection overlay, and a GLB with red vertex markers.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Render width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Render height in pixels.",
    )
    parser.add_argument(
        "--views",
        default=",".join(DEFAULT_VIEWS),
        help="Comma-separated Blender cardinal views to try. Default: neg_y,pos_y,neg_x,pos_x.",
    )
    parser.add_argument(
        "--yaw-offsets",
        default="0",
        help=(
            "Comma-separated yaw offsets in degrees to try for each render view. "
            "Default: 0."
        ),
    )
    parser.add_argument(
        "--roll-offset",
        type=float,
        default=0.0,
        help=(
            "Camera-local roll in degrees, applied consistently to rendering "
            "and landmark ray-casting. Default: 0."
        ),
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.35,
        help="MediaPipe face detection confidence threshold.",
    )
    parser.add_argument(
        "--refine-landmarks",
        action="store_true",
        help="Enable refined landmarks, including iris landmarks, when available.",
    )
    parser.add_argument(
        "--mediapipe-model",
        default=None,
        help=(
            "Optional MediaPipe Face Landmarker .task model. When using the "
            "newer MediaPipe Tasks API, the script downloads the default model "
            "to ~/.cache/landmark3d if this is not supplied."
        ),
    )
    parser.add_argument(
        "--blender",
        default=None,
        help="Optional path to the Blender executable. Defaults to the first blender on PATH.",
    )
    parser.add_argument(
        "--blender-threads",
        type=int,
        default=None,
        help="Optional Blender --threads value. Use 1 on systems with tight thread limits.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Defaults to the mesh path with .json extension.",
    )
    parser.add_argument(
        "--debug-glb",
        default=None,
        help="Optional debug GLB path. Defaults to <mesh_stem>_debug_vertices.glb next to the mesh.",
    )
    parser.add_argument(
        "--no-debug-glb",
        action="store_true",
        help="Write debug images without exporting the marker-heavy debug GLB.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("raycast", "scored", "depth"),
        default="raycast",
        help=(
            "Vertex selection strategy. 'raycast' is the default and uses the corrected direct ray hit; "
            "'depth' uses MediaPipe z as a relative depth prior; "
            "'scored' enables the experimental face-oval candidate scoring."
        ),
    )
    parser.add_argument(
        "--preserve-materials",
        action="store_true",
        help=(
            "Render using the materials imported with the mesh instead of "
            "replacing them with the neutral landmark-detection material."
        ),
    )
    parser.add_argument(
        "--output-surface-anchors",
        action="store_true",
        help=(
            "Write a versioned payload containing ray-hit triangle vertices and "
            "barycentric weights in addition to the legacy vertex mapping."
        ),
    )
    parser.add_argument(
        "--surface-hits-only", action="store_true",
        help="Use surface intersections, including meshes without UVs; skip projected-vertex fallback.",
    )
    parser.add_argument(
        "--filter-noisy",
        action="store_true",
        help="Keep only landmarks whose 3D vertex mapping is stable across small yaw-perturbed renders.",
    )
    parser.add_argument(
        "--stability-yaws",
        default="-4,-2,2,4",
        help="Comma-separated yaw offsets in degrees used by --filter-noisy.",
    )
    parser.add_argument(
        "--stability-keep-ratio",
        type=float,
        default=0.85,
        help="Fraction of lowest-noise landmarks to keep when --filter-noisy is enabled.",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=None,
        help="Optional absolute 3D stability threshold. When supplied, landmarks must also be below this distance.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    if process.returncode != 0 or "Traceback (most recent call last):" in process.stderr:
        stdout = process.stdout.strip()
        stderr = process.stderr.strip()
        details = "\n".join(part for part in (stdout, stderr) if part)
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{details}")
    return process


def copy_failure_debug_renders(
    rendered_views: list[dict[str, Any]],
    temp_dir: Path,
    output_json: Path,
) -> Path:
    debug_dir = output_json.with_name(f"{output_json.stem}_debug_failed")
    debug_dir.mkdir(parents=True, exist_ok=True)
    for index, rendered in enumerate(rendered_views):
        image_path = Path(rendered["image"])
        if not image_path.exists():
            continue
        yaw_offset = float(rendered.get("yaw_offset", 0.0))
        yaw_label = f"yaw_{yaw_offset:+.2f}".replace("+", "p").replace("-", "m").replace(".", "_")
        destination = debug_dir / f"{index:03d}_{rendered['view']}_{yaw_label}.png"
        shutil.copyfile(image_path, destination)
    views_path = temp_dir / "views.json"
    if views_path.exists():
        shutil.copyfile(views_path, debug_dir / "views.json")
    return debug_dir


def write_blender_helper(temp_dir: Path) -> Path:
    helper_path = temp_dir / "landmark_blender_helper.py"
    helper_path.write_text(BLENDER_HELPER, encoding="utf-8")
    return helper_path


def blender_executable(path: str | None) -> str:
    if path:
        candidate = Path(path).expanduser()
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"Blender executable not found: {candidate}")

    blender = shutil.which("blender")
    if not blender:
        raise FileNotFoundError(
            "Blender was not found on PATH. Install Blender or pass --blender /path/to/blender."
        )
    return blender


def blender_command_prefix(blender: str, threads: int | None) -> list[str]:
    command = [blender, "--background"]
    if threads is not None:
        if threads < 1:
            raise ValueError("--blender-threads must be positive when provided.")
        command.extend(["--threads", str(threads)])
    return command


def solutions_face_mesh_available() -> bool:
    try:
        return hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
    except Exception:
        return False


def task_face_landmarker_available() -> bool:
    try:
        from mediapipe.tasks.python.vision import FaceLandmarker  # noqa: F401

        return True
    except Exception:
        return False


def ensure_face_landmarker_model(model_arg: str | None) -> Path:
    if model_arg:
        model_path = Path(model_arg).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"MediaPipe model not found: {model_path}")
        return model_path

    model_path = Path.home() / ".cache" / "landmark3d" / "face_landmarker.task"
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(".task.tmp")
    try:
        urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, tmp_path)
        tmp_path.replace(model_path)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            "Could not download the MediaPipe Face Landmarker model. "
            "Pass --mediapipe-model /path/to/face_landmarker.task to use a local model."
        ) from exc

    return model_path


def render_candidates(
    blender: str,
    blender_threads: int | None,
    helper: Path,
    mesh_path: Path,
    temp_dir: Path,
    width: int,
    height: int,
    views: str,
    yaw_offsets: str = "0",
    roll_offset: float = 0.0,
    preserve_materials: bool = False,
) -> list[dict[str, Any]]:
    command = [
            *blender_command_prefix(blender, blender_threads),
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
            str(width),
            "--height",
            str(height),
            "--views",
            views,
            f"--yaw-offsets={yaw_offsets}",
            "--roll-offset",
            str(roll_offset),
        ]
    if preserve_materials:
        command.append("--preserve-materials")
    run_command(command)
    with (temp_dir / "views.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)["views"]


def landmark_score(landmarks: list[Any]) -> float:
    xs = [float(landmark.x) for landmark in landmarks]
    ys = [float(landmark.y) for landmark in landmarks]
    inside = [
        1.0
        for x, y in zip(xs, ys)
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
    ]
    inside_ratio = len(inside) / max(len(landmarks), 1)
    bbox_area = max(xs) - min(xs)
    bbox_area *= max(ys) - min(ys)
    centered = 1.0 - min(
        ((sum(xs) / len(xs) - 0.5) ** 2 + (sum(ys) / len(ys) - 0.5) ** 2) ** 0.5,
        1.0,
    )
    return bbox_area * inside_ratio * max(centered, 0.1)


def choose_best_detection_with_solutions(
    rendered_views: list[dict[str, Any]],
    min_detection_confidence: float,
    refine_landmarks: bool,
    return_all: bool = False,
) -> DetectionCandidate | list[DetectionCandidate]:
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
    )

    candidates: list[DetectionCandidate] = []
    try:
        for rendered in rendered_views:
            image_path = Path(rendered["image"])
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                continue
            landmarks = list(results.multi_face_landmarks[0].landmark)
            candidates.append(
                DetectionCandidate(
                    view=str(rendered["view"]),
                    yaw_offset=float(rendered.get("yaw_offset", 0.0)),
                    roll_offset=float(rendered.get("roll_offset", 0.0)),
                    image_path=image_path,
                    landmarks=landmarks,
                    score=landmark_score(landmarks),
                )
            )
    finally:
        face_mesh.close()

    if return_all:
        return candidates

    if not candidates:
        views = ", ".join(str(view["view"]) for view in rendered_views)
        raise RuntimeError(
            "MediaPipe did not detect a face in any rendered view. "
            f"Tried views: {views}. Try a different --views order, a lower "
            "--min-detection-confidence, or a textured/frontal mesh."
        )

    return max(candidates, key=lambda candidate: candidate.score)


def choose_best_detection_with_tasks(
    rendered_views: list[dict[str, Any]],
    min_detection_confidence: float,
    model_path: Path,
    refine_landmarks: bool,
    return_all: bool = False,
) -> DetectionCandidate | list[DetectionCandidate]:
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=min_detection_confidence,
        min_face_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_detection_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = FaceLandmarker.create_from_options(options)

    candidates: list[DetectionCandidate] = []
    try:
        for rendered in rendered_views:
            image_path = Path(rendered["image"])
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = landmarker.detect(mp_image)
            if not results.face_landmarks:
                continue
            landmarks = list(results.face_landmarks[0])
            if not refine_landmarks:
                landmarks = landmarks[:468]
            candidates.append(
                DetectionCandidate(
                    view=str(rendered["view"]),
                    yaw_offset=float(rendered.get("yaw_offset", 0.0)),
                    roll_offset=float(rendered.get("roll_offset", 0.0)),
                    image_path=image_path,
                    landmarks=landmarks,
                    score=landmark_score(landmarks),
                )
            )
    finally:
        landmarker.close()

    if return_all:
        return candidates

    if not candidates:
        views = ", ".join(str(view["view"]) for view in rendered_views)
        raise RuntimeError(
            "MediaPipe did not detect a face in any rendered view. "
            f"Tried views: {views}. Try a different --views order, a lower "
            "--min-detection-confidence, or a textured/frontal mesh."
        )

    return max(candidates, key=lambda candidate: candidate.score)


def choose_best_detection(
    rendered_views: list[dict[str, Any]],
    min_detection_confidence: float,
    refine_landmarks: bool,
    model_path: Path | None,
) -> DetectionCandidate:
    if solutions_face_mesh_available():
        return choose_best_detection_with_solutions(
            rendered_views,
            min_detection_confidence=min_detection_confidence,
            refine_landmarks=refine_landmarks,
        )
    if task_face_landmarker_available():
        if model_path is None:
            raise RuntimeError("A MediaPipe Face Landmarker .task model is required.")
        return choose_best_detection_with_tasks(
            rendered_views,
            min_detection_confidence=min_detection_confidence,
            model_path=model_path,
            refine_landmarks=refine_landmarks,
        )

    raise RuntimeError("No supported MediaPipe face landmark API is available.")


def detect_rendered_views(
    rendered_views: list[dict[str, Any]],
    min_detection_confidence: float,
    refine_landmarks: bool,
    model_path: Path | None,
) -> list[DetectionCandidate]:
    if solutions_face_mesh_available():
        return choose_best_detection_with_solutions(
            rendered_views,
            min_detection_confidence=min_detection_confidence,
            refine_landmarks=refine_landmarks,
            return_all=True,
        )
    if task_face_landmarker_available():
        if model_path is None:
            raise RuntimeError("A MediaPipe Face Landmarker .task model is required.")
        return choose_best_detection_with_tasks(
            rendered_views,
            min_detection_confidence=min_detection_confidence,
            model_path=model_path,
            refine_landmarks=refine_landmarks,
            return_all=True,
        )

    raise RuntimeError("No supported MediaPipe face landmark API is available.")


def write_landmarks_json(
    candidate: DetectionCandidate,
    path: Path,
    width: int,
    height: int,
    allowed_ids: set[int] | None = None,
) -> None:
    payload = {
        "image_width": width,
        "image_height": height,
        "view": candidate.view,
        "yaw_offset": candidate.yaw_offset,
        "roll_offset": candidate.roll_offset,
        "landmarks": [
            {
                "index": index,
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z),
            }
            for index, landmark in enumerate(candidate.landmarks)
            if allowed_ids is None or index in allowed_ids
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_landmark_overlay(candidate: DetectionCandidate, output_path: Path) -> None:
    image = cv2.imread(str(candidate.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read rendered image: {candidate.image_path}")

    height, width = image.shape[:2]
    overlay = image.copy()
    for index, landmark in enumerate(candidate.landmarks):
        x = int(round(float(landmark.x) * width))
        y = int(round(float(landmark.y) * height))
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(overlay, (x, y), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            if index % 25 == 0:
                cv2.putText(
                    overlay,
                    str(index),
                    (x + 3, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.28,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
    cv2.imwrite(str(output_path), overlay)


def write_reprojection_overlay(
    candidate: DetectionCandidate,
    projections: dict[str, dict[str, float] | None],
    output_path: Path,
) -> dict[str, float]:
    image = cv2.imread(str(candidate.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read rendered image: {candidate.image_path}")

    height, width = image.shape[:2]
    overlay = image.copy()
    errors: list[float] = []
    for index, landmark in enumerate(candidate.landmarks):
        source_x = int(round(float(landmark.x) * width))
        source_y = int(round(float(landmark.y) * height))
        projection = projections.get(str(index))
        if projection is None:
            continue

        target_x = int(round(float(projection["x"]) * width))
        target_y = int(round(float(projection["y"]) * height))
        if 0 <= source_x < width and 0 <= source_y < height:
            cv2.circle(overlay, (source_x, source_y), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        if 0 <= target_x < width and 0 <= target_y < height:
            cv2.circle(overlay, (target_x, target_y), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        if (
            0 <= source_x < width
            and 0 <= source_y < height
            and 0 <= target_x < width
            and 0 <= target_y < height
        ):
            cv2.line(overlay, (source_x, source_y), (target_x, target_y), (255, 0, 0), 1, lineType=cv2.LINE_AA)
            errors.append(((source_x - target_x) ** 2 + (source_y - target_y) ** 2) ** 0.5)

    cv2.imwrite(str(output_path), overlay)
    if not errors:
        return {"mean_px": 0.0, "max_px": 0.0}
    return {
        "mean_px": float(sum(errors) / len(errors)),
        "max_px": float(max(errors)),
    }


def raycast_landmarks(
    blender: str,
    blender_threads: int | None,
    helper: Path,
    mesh_path: Path,
    landmarks_json: Path,
    candidate: DetectionCandidate,
    temp_dir: Path,
    width: int,
    height: int,
    debug_glb: Path | None,
    selection_mode: str,
    surface_hits_only: bool = False,
) -> dict[str, Any]:
    raycast_output = temp_dir / "raycast_result.json"
    command = [
        *blender_command_prefix(blender, blender_threads),
        "--python",
        str(helper),
        "--",
        "--mode",
        "raycast",
        "--mesh",
        str(mesh_path),
        "--view",
        candidate.view,
        "--yaw-offset",
        str(candidate.yaw_offset),
        "--roll-offset",
        str(candidate.roll_offset),
        "--width",
        str(width),
        "--height",
        str(height),
        "--landmarks-json",
        str(landmarks_json),
        "--output-json",
        str(raycast_output),
        "--selection-mode",
        selection_mode,
    ]
    if debug_glb is not None:
        command.extend(["--debug-glb", str(debug_glb)])
    if surface_hits_only:
        command.append("--surface-hits-only")

    run_command(command)
    with raycast_output.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def position_distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.sqrt(
        (a["x"] - b["x"]) * (a["x"] - b["x"])
        + (a["y"] - b["y"]) * (a["y"] - b["y"])
        + (a["z"] - b["z"]) * (a["z"] - b["z"])
    )


def mean_position(positions: list[dict[str, float]]) -> dict[str, float]:
    count = len(positions)
    return {
        "x": sum(position["x"] for position in positions) / count,
        "y": sum(position["y"] for position in positions) / count,
        "z": sum(position["z"] for position in positions) / count,
    }


def compute_stability_filter(
    base_result: dict[str, Any],
    perturbation_results: list[dict[str, Any]],
    keep_ratio: float,
    threshold: float | None,
) -> tuple[set[int], dict[str, Any]]:
    base_positions = base_result["vertex_positions"]
    scores: dict[str, dict[str, Any]] = {}

    for mp_index, base_position in base_positions.items():
        positions = []
        if base_position is not None:
            positions.append(base_position)
        for result in perturbation_results:
            position = result["vertex_positions"].get(mp_index)
            if position is not None:
                positions.append(position)

        if len(positions) < 2:
            scores[mp_index] = {
                "kept": False,
                "sample_count": len(positions),
                "mean_distance": None,
                "max_distance": None,
            }
            continue

        center = mean_position(positions)
        distances = [position_distance(position, center) for position in positions]
        scores[mp_index] = {
            "kept": False,
            "sample_count": len(positions),
            "mean_distance": float(sum(distances) / len(distances)),
            "max_distance": float(max(distances)),
        }

    finite_items = [
        (mp_index, score["mean_distance"])
        for mp_index, score in scores.items()
        if score["mean_distance"] is not None
    ]
    finite_items.sort(key=lambda item: item[1])

    keep_ratio = min(max(keep_ratio, 0.0), 1.0)
    keep_count = max(0, math.ceil(len(finite_items) * keep_ratio))
    kept_ids: set[int] = set()
    for mp_index, score in finite_items[:keep_count]:
        if threshold is None or score <= threshold:
            kept_ids.add(int(mp_index))
            scores[mp_index]["kept"] = True

    kept_scores = [scores[str(index)]["mean_distance"] for index in kept_ids]
    rejected_scores = [
        score["mean_distance"]
        for score in scores.values()
        if not score["kept"] and score["mean_distance"] is not None
    ]
    report = {
        "method": "multi_render_vertex_position_stability",
        "keep_ratio": keep_ratio,
        "threshold": threshold,
        "total_landmarks": len(scores),
        "kept_landmarks": len(kept_ids),
        "rejected_landmarks": len(scores) - len(kept_ids),
        "kept_mean_distance_max": max(kept_scores) if kept_scores else None,
        "rejected_mean_distance_min": min(rejected_scores) if rejected_scores else None,
        "landmarks": scores,
    }
    return kept_ids, report


def raycast_detection_candidate(
    blender: str,
    blender_threads: int | None,
    helper: Path,
    mesh_path: Path,
    candidate: DetectionCandidate,
    temp_dir: Path,
    width: int,
    height: int,
    selection_mode: str,
    name: str,
    surface_hits_only: bool = False,
) -> dict[str, Any]:
    landmarks_json = temp_dir / f"{name}_landmarks.json"
    write_landmarks_json(candidate, landmarks_json, width, height)
    return raycast_landmarks(
        blender=blender,
        blender_threads=blender_threads,
        helper=helper,
        mesh_path=mesh_path,
        landmarks_json=landmarks_json,
        candidate=candidate,
        temp_dir=temp_dir,
        width=width,
        height=height,
        debug_glb=None,
        selection_mode=selection_mode,
        surface_hits_only=surface_hits_only,
    )


def main() -> None:
    args = parse_args()
    mesh_path = Path(args.mesh).expanduser().resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    if mesh_path.suffix.lower() not in {".fbx", ".glb", ".gltf"}:
        raise ValueError("Only .fbx, .glb, and .gltf meshes are supported.")

    output_json = Path(args.output).expanduser().resolve() if args.output else mesh_path.with_suffix(".json")
    debug_dir = output_json.parent
    if args.no_debug_glb:
        debug_glb = None
    elif args.debug_glb:
        debug_glb = Path(args.debug_glb).expanduser().resolve()
    else:
        debug_glb = mesh_path.with_name(f"{mesh_path.stem}_debug_vertices.glb")

    blender = blender_executable(args.blender)
    model_path = None
    if not solutions_face_mesh_available():
        model_path = ensure_face_landmarker_model(args.mediapipe_model)

    with tempfile.TemporaryDirectory(prefix="mediapipe_mesh_map_") as temp_name:
        temp_dir = Path(temp_name)
        helper = write_blender_helper(temp_dir)
        rendered_views = render_candidates(
            blender=blender,
            blender_threads=args.blender_threads,
            helper=helper,
            mesh_path=mesh_path,
            temp_dir=temp_dir,
            width=args.width,
            height=args.height,
            views=args.views,
            yaw_offsets=args.yaw_offsets,
            roll_offset=args.roll_offset,
            preserve_materials=args.preserve_materials,
        )
        try:
            candidate = choose_best_detection(
                rendered_views,
                min_detection_confidence=args.min_detection_confidence,
                refine_landmarks=args.refine_landmarks,
                model_path=model_path,
            )
        except Exception:
            if args.debug:
                debug_failure_dir = copy_failure_debug_renders(
                    rendered_views,
                    temp_dir,
                    output_json,
                )
                print(f"Wrote failed detection renders to {debug_failure_dir}", file=sys.stderr)
            raise

        landmarks_json = temp_dir / "mediapipe_landmarks.json"
        write_landmarks_json(candidate, landmarks_json, args.width, args.height)

        if args.debug:
            debug_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(candidate.image_path, debug_dir / "face_debug.png")
            write_landmark_overlay(candidate, debug_dir / "face_debug_landmark.png")

        stability_report: dict[str, Any] | None = None
        if args.filter_noisy:
            base_result = raycast_landmarks(
                blender=blender,
                blender_threads=args.blender_threads,
                helper=helper,
                mesh_path=mesh_path,
                landmarks_json=landmarks_json,
                candidate=candidate,
                temp_dir=temp_dir,
                width=args.width,
                height=args.height,
                debug_glb=None,
                selection_mode=args.selection_mode,
                surface_hits_only=args.surface_hits_only,
            )
            stability_dir = temp_dir / "stability"
            stability_views = render_candidates(
                blender=blender,
                blender_threads=args.blender_threads,
                helper=helper,
                mesh_path=mesh_path,
                temp_dir=stability_dir,
                width=args.width,
                height=args.height,
                views=candidate.view,
                yaw_offsets=args.stability_yaws,
                roll_offset=args.roll_offset,
                preserve_materials=args.preserve_materials,
            )
            stability_candidates = detect_rendered_views(
                stability_views,
                min_detection_confidence=args.min_detection_confidence,
                refine_landmarks=args.refine_landmarks,
                model_path=model_path,
            )
            if not stability_candidates:
                raise RuntimeError("No faces were detected in the stability-filter perturbation renders.")

            perturbation_results = [
                raycast_detection_candidate(
                    blender=blender,
                    blender_threads=args.blender_threads,
                    helper=helper,
                    mesh_path=mesh_path,
                    candidate=stability_candidate,
                    temp_dir=temp_dir,
                    width=args.width,
                    height=args.height,
                    selection_mode=args.selection_mode,
                    name=f"stability_{index}",
                    surface_hits_only=args.surface_hits_only,
                )
                for index, stability_candidate in enumerate(stability_candidates)
            ]
            stable_ids, stability_report = compute_stability_filter(
                base_result,
                perturbation_results,
                keep_ratio=args.stability_keep_ratio,
                threshold=args.stability_threshold,
            )

            filtered_landmarks_json = temp_dir / "mediapipe_landmarks_filtered.json"
            write_landmarks_json(
                candidate,
                filtered_landmarks_json,
                args.width,
                args.height,
                allowed_ids=stable_ids,
            )
            raycast_result = raycast_landmarks(
                blender=blender,
                blender_threads=args.blender_threads,
                helper=helper,
                mesh_path=mesh_path,
                landmarks_json=filtered_landmarks_json,
                candidate=candidate,
                temp_dir=temp_dir,
                width=args.width,
                height=args.height,
                debug_glb=debug_glb if args.debug else None,
                selection_mode=args.selection_mode,
                surface_hits_only=args.surface_hits_only,
            )
        else:
            raycast_result = raycast_landmarks(
                blender=blender,
                blender_threads=args.blender_threads,
                helper=helper,
                mesh_path=mesh_path,
                landmarks_json=landmarks_json,
                candidate=candidate,
                temp_dir=temp_dir,
                width=args.width,
                height=args.height,
                debug_glb=debug_glb if args.debug else None,
                selection_mode=args.selection_mode,
                surface_hits_only=args.surface_hits_only,
            )

        reprojection_stats: dict[str, float] | None = None
        if args.debug:
            reprojection_stats = write_reprojection_overlay(
                candidate,
                raycast_result["vertex_projections"],
                debug_dir / "face_debug_reprojected.png",
            )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_payload: Any
    if args.output_surface_anchors:
        output_payload = {
            "version": 4,
            "mapping": raycast_result["mapping"],
            "surface_anchors": raycast_result["surface_anchors"],
            "surface_anchor_selection": raycast_result.get(
                "surface_anchor_selection", {}
            ),
            "iris_surface_anchor_candidates": raycast_result.get(
                "iris_surface_anchor_candidates", {}
            ),
            "mesh_object_offsets": raycast_result["mesh_object_offsets"],
            "selection_mode": raycast_result["selection_mode"],
            "selected_view": raycast_result["selected_view"],
            "yaw_offset": raycast_result["yaw_offset"],
            "roll_offset": raycast_result["roll_offset"],
        }
    else:
        output_payload = raycast_result["mapping"]
    output_json.write_text(
        json.dumps(output_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    stability_report_path = None
    if stability_report is not None:
        stability_report_path = output_json.with_name(f"{output_json.stem}_stability.json")
        stability_report_path.write_text(json.dumps(stability_report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {output_json}")
    print(f"Selected render view: {candidate.view}")
    if stability_report is not None:
        print(
            "Stability filter kept "
            f"{stability_report['kept_landmarks']}/{stability_report['total_landmarks']} landmarks"
        )
        print(f"Wrote {stability_report_path}")
    if args.debug:
        print(f"Wrote {debug_dir / 'face_debug.png'}")
        print(f"Wrote {debug_dir / 'face_debug_landmark.png'}")
        print(f"Wrote {debug_dir / 'face_debug_reprojected.png'}")
        if debug_glb is not None:
            print(f"Wrote {debug_glb}")
        if reprojection_stats is not None:
            print(
                "Reprojection error: "
                f"mean={reprojection_stats['mean_px']:.2f}px "
                f"max={reprojection_stats['max_px']:.2f}px"
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
