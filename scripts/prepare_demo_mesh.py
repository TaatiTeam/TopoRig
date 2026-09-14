#!/usr/bin/env python3
"""Export a neutral, self-contained GLB using Blender's Python interpreter.

blender --background --python scripts/prepare_demo_mesh.py -- \
    --input samples/head.fbx --output samples/head.glb --face-only
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys


def prepare_mesh(source: Path, output: Path, *, face_only: bool = False,
                 frame: int | None = None) -> None:
    import bpy
    from mathutils import Matrix

    bpy.ops.wm.read_factory_settings(use_empty=True)
    suffix = source.suffix.lower()
    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(source), use_anim=False)
    elif suffix == ".glb":
        bpy.ops.import_scene.gltf(filepath=str(source), merge_vertices=False)
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(source))
    else:
        raise ValueError("Upload a GLB, FBX or OBJ mesh.")

    scene = bpy.context.scene
    meshes = [obj for obj in scene.objects if obj.type == "MESH" and obj.data.polygons]
    if not meshes:
        raise ValueError("The file contains no triangle surface.")
    if face_only:
        candidates = [obj for obj in meshes if not any(
            token in obj.name.lower() for token in ("eye", "teeth", "tongue", "lash", "gum")
        )]
        meshes = [max(candidates or meshes, key=lambda obj: len(obj.data.vertices))]

    if frame is None:
        for obj in scene.objects:
            obj.animation_data_clear()
            if obj.type == "ARMATURE":
                obj.data.pose_position = "REST"
            if obj.type == "MESH" and obj.data.shape_keys:
                obj.data.shape_keys.animation_data_clear()
                for key in obj.data.shape_keys.key_blocks:
                    key.value = 0.0
                obj.active_shape_key_index = 0
                obj.show_only_shape_key = False
    else:
        scene.frame_set(frame)
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()

    # Flatten transforms and evaluated geometry into one mesh. This keeps GLB
    # accessor order identical to the inference loader and landmark mapper.
    baked = []
    for obj in meshes:
        evaluated = obj.evaluated_get(graph)
        data = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True,
                                              depsgraph=graph)
        data.transform(obj.matrix_world)
        # Dataset FBXs use Z up / +Y front; GLB viewers use Y up / +Z front.
        if suffix == ".fbx":
            data.transform(Matrix.Rotation(math.pi, 4, "Z"))
        new = bpy.data.objects.new("Neutral face" if frame is None else "Expression", data)
        scene.collection.objects.link(new)
        baked.append(new)
    for obj in list(scene.objects):
        if obj not in baked:
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in baked:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = baked[0]
    if len(baked) > 1:
        bpy.ops.object.join()
    mesh = bpy.context.view_layer.objects.active
    mesh.name = "Neutral face" if frame is None else "Expression"
    for polygon in mesh.data.polygons:
        polygon.use_smooth = True
    if not mesh.data.materials:
        material = bpy.data.materials.new("Face")
        material.diffuse_color = (0.64, 0.69, 0.76, 1.0)
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = material.diffuse_color
        bsdf.inputs["Roughness"].default_value = 0.72
        mesh.data.materials.append(material)

    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output), export_format="GLB", use_selection=True,
        export_yup=True, export_apply=True, export_animations=False,
        export_skins=False, export_morph=False, export_cameras=False,
        export_lights=False,
    )
    print(f"Exported {output}: {len(mesh.data.vertices)} vertices, "
          f"{len(mesh.data.polygons)} polygons", flush=True)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--face-only", action="store_true")
    parser.add_argument("--frame", type=int, help="Bake this animation frame instead of neutralizing.")
    parser.add_argument("--thumbnail", action="store_true", help="Render a front-view PNG from a prepared GLB.")
    args = parser.parse_args(argv)
    if args.thumbnail:
        render_thumbnail(args.input.resolve(), args.output.resolve())
        return
    prepare_mesh(args.input.resolve(), args.output.resolve(),
                 face_only=args.face_only, frame=args.frame)


def render_thumbnail(source: Path, output: Path) -> None:
    import bpy
    from mathutils import Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    scene = bpy.context.scene
    objects = [obj for obj in scene.objects if obj.type == "MESH"]
    corners = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    lower = Vector(tuple(min(p[axis] for p in corners) for axis in range(3)))
    upper = Vector(tuple(max(p[axis] for p in corners) for axis in range(3)))
    center = (lower + upper) * 0.5
    extent = max(upper - lower)
    camera = bpy.data.objects.new("Front preview", bpy.data.cameras.new("Front preview"))
    scene.collection.objects.link(camera)
    camera.location = center + Vector((0, -3 * extent, 0))
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = max(upper.x - lower.x, upper.z - lower.z) * 1.08
    camera.data.clip_start = max(extent * 0.001, 0.0001)
    camera.data.clip_end = extent * 10
    scene.camera = camera
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = scene.render.resolution_y = 320
    scene.render.resolution_percentage = 100
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "TEXTURE"
    shading.show_shadows = True
    shading.show_cavity = True
    shading.background_type = "VIEWPORT"
    shading.background_color = (0.055, 0.07, 0.09)
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image:
                    material.node_tree.nodes.active = node
                    break
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.file_format = "PNG"
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
