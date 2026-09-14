from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch

bpy = None


PathLike = Union[str, Path]


AU_NAME = {
    0: "browDown_L",
    1: "browDown_R",
    2: "browInnerUp_L",
    3: "browInnerUp_R",
    4: "browOuterUp_L",
    5: "browOuterUp_R",
    6: "cheekPuff_L",
    7: "cheekPuff_R",
    8: "cheekSquint_L",
    9: "cheekSquint_R",
    10: "eyeBlink_L",
    11: "eyeBlink_R",
    12: "eyeLookDown_L",
    13: "eyeLookDown_R",
    14: "eyeLookIn_L",
    15: "eyeLookIn_R",
    16: "eyeLookOut_L",
    17: "eyeLookOut_R",
    18: "eyeLookUp_L",
    19: "eyeLookUp_R",
    20: "eyeSquint_L",
    21: "eyeSquint_R",
    22: "eyeWide_L",
    23: "eyeWide_R",
    24: "jawForward",
    25: "jawLeft",
    26: "jawOpen",
    27: "jawRight",
    28: "mouthClose",
    29: "mouthDimple_L",
    30: "mouthDimple_R",
    31: "mouthFrown_L",
    32: "mouthFrown_R",
    33: "mouthFunnel",
    34: "mouthLeft",
    35: "mouthLowerDown_L",
    36: "mouthLowerDown_R",
    37: "mouthPress_L",
    38: "mouthPress_R",
    39: "mouthPucker",
    40: "mouthRight",
    41: "mouthRollLower",
    42: "mouthRollUpper",
    43: "mouthShrugLower",
    44: "mouthShrugUpper",
    45: "mouthSmile_L",
    46: "mouthSmile_R",
    47: "mouthStretch_L",
    48: "mouthStretch_R",
    49: "mouthUpperUp_L",
    50: "mouthUpperUp_R",
    51: "noseSneer_L",
    52: "noseSneer_R",
}


METAHUMAN_HEAD_MESH_NAME = "head_lod0_ORIGINAL"
METAHUMAN_MESH_NAMES = (
    "head_lod0_ORIGINAL",
    "eyeLeft_ORIGINAL",
    "eyeRight_ORIGINAL",
    "teeth_ORIGINAL",
)

METAHUMAN_BLENDSHAPE_NAMES = {
    "browDown_L": "browDownLeft",
    "browDown_R": "browDownRight",
    "browInnerUp_L": "browInnerUp",
    "browInnerUp_R": "browInnerUp",
    "browOuterUp_L": "browOuterUpLeft",
    "browOuterUp_R": "browOuterUpRight",
    "cheekPuff_L": "cheekPuff",
    "cheekPuff_R": "cheekPuff",
    "cheekSquint_L": "cheekSquintLeft",
    "cheekSquint_R": "cheekSquintRight",
    "eyeBlink_L": "eyeBlinkLeft",
    "eyeBlink_R": "eyeBlinkRight",
    "eyeLookDown_L": "eyeLookDownLeft",
    "eyeLookDown_R": "eyeLookDownRight",
    "eyeLookIn_L": "eyeLookInLeft",
    "eyeLookIn_R": "eyeLookInRight",
    "eyeLookOut_L": "eyeLookOutLeft",
    "eyeLookOut_R": "eyeLookOutRight",
    "eyeLookUp_L": "eyeLookUpLeft",
    "eyeLookUp_R": "eyeLookUpRight",
    "eyeSquint_L": "eyeSquintLeft",
    "eyeSquint_R": "eyeSquintRight",
    "eyeWide_L": "eyeWideLeft",
    "eyeWide_R": "eyeWideRight",
    "jawForward": "jawForward",
    "jawLeft": "jawLeft",
    "jawOpen": "jawOpen",
    "jawRight": "jawRight",
    "mouthClose": "mouthClose",
    "mouthDimple_L": "mouthDimpleLeft",
    "mouthDimple_R": "mouthDimpleRight",
    "mouthFrown_L": "mouthFrownLeft",
    "mouthFrown_R": "mouthFrownRight",
    "mouthFunnel": "mouthFunnel",
    "mouthLeft": "mouthLeft",
    "mouthLowerDown_L": "mouthLowerDownLeft",
    "mouthLowerDown_R": "mouthLowerDownRight",
    "mouthPress_L": "mouthPressLeft",
    "mouthPress_R": "mouthPressRight",
    "mouthPucker": "mouthPucker",
    "mouthRight": "mouthRight",
    "mouthRollLower": "mouthRollLower",
    "mouthRollUpper": "mouthRollUpper",
    "mouthShrugLower": "mouthShrugLower",
    "mouthShrugUpper": "mouthShrugUpper",
    "mouthSmile_L": "mouthSmileLeft",
    "mouthSmile_R": "mouthSmileRight",
    "mouthStretch_L": "mouthStretchLeft",
    "mouthStretch_R": "mouthStretchRight",
    "mouthUpperUp_L": "mouthUpperUpLeft",
    "mouthUpperUp_R": "mouthUpperUpRight",
    "noseSneer_L": "noseSneerLeft",
    "noseSneer_R": "noseSneerRight",
}

CANONICAL_BLENDSHAPE_NAMES = {
    action_unit_name: action_unit_name for action_unit_name in AU_NAME.values()
}

METAHUMAN_SHARED_BLENDSHAPE_SIDES = {
    "browInnerUp_L": ("browInnerUp", "L"),
    "browInnerUp_R": ("browInnerUp", "R"),
    "cheekPuff_L": ("cheekPuff", "L"),
    "cheekPuff_R": ("cheekPuff", "R"),
}
METAHUMAN_SHARED_BLENDSHAPE_TRANSITION_RATIO = 0.06


def _require_bpy():
    global bpy
    if bpy is not None:
        return bpy
    try:
        import bpy as bpy_module
    except ImportError as exc:  # pragma: no cover - depends on local Blender installation.
        raise ImportError(
            "fbx_to_tensor requires Blender's Python module (`bpy`) to import FBX "
            "files. Install bpy or run this code in a Blender Python environment. "
            f"Current Python executable: {sys.executable}. Try "
            "`python3 -m pip install bpy` inside the active environment, or run "
            "`python3 -c \"import bpy; print(bpy.app.version_string)\"` to verify it."
        ) from exc
    bpy = bpy_module
    return bpy


def fbx_to_tensor(
    fbx_path: PathLike,
    mesh_names: Optional[Union[str, Sequence[str]]] = METAHUMAN_MESH_NAMES,
    blendshape_name_map: Mapping[str, str] = METAHUMAN_BLENDSHAPE_NAMES,
    dtype: torch.dtype = torch.float32,
    device: Optional[Union[str, torch.device]] = None,
    missing_blendshape: str = "zeros",
    mesh_name: Optional[str] = None,
    include_metadata: bool = False,
    exclude_mesh_name_patterns: Optional[Sequence[str]] = None,
) -> (
    Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]
    | Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, object]]
):
    """Load a MetaHuman FBX face mesh as torch tensors.

    The returned blendshape dictionary follows ``AU_NAME`` insertion order.
    Each blendshape tensor is a per-vertex delta that can be added to the
    neutral vertices, e.g. ``vertices + blendshapes["jawOpen"]``.
    MetaHuman's shared ``browInnerUp`` and ``cheekPuff`` shape keys are split
    into left/right AU tensors with a smooth centerline cross-fade.

    Args:
        fbx_path: Path to an FBX file.
        mesh_names: Mesh object names to concatenate. Defaults to the sample
            MetaHuman head, eyes, and teeth. A single string is accepted.
        blendshape_name_map: Maps canonical ``AU_NAME`` entries to FBX shape keys.
        dtype: Floating point dtype for vertex and blendshape tensors.
        device: Optional torch device for returned tensors.
        missing_blendshape: ``"zeros"`` fills missing AUs with zero deltas;
            ``"raise"`` raises a ``KeyError`` instead.
        mesh_name: Backward-compatible single-mesh override.
        include_metadata: If True, also return vertex groups showing where
            each source mesh object and inferred section lands in the combined
            vertex tensor.
        exclude_mesh_name_patterns: Optional case-insensitive substrings. Mesh
            objects with names containing any pattern are skipped before
            tensors are built.

    Returns:
        ``(faces, vertices, blendshapes)`` where ``faces`` is ``[F, 3]`` long,
        ``vertices`` is ``[V, 3]``, and each blendshape delta is ``[V, 3]``.
        Vertices from separate FBX meshes are concatenated into one tensor, and
        faces are offset to index that combined vertex tensor.
    """
    if missing_blendshape not in {"zeros", "raise"}:
        raise ValueError('missing_blendshape must be either "zeros" or "raise".')
    _require_bpy()

    fbx_path = Path(fbx_path).expanduser().resolve()
    if not fbx_path.exists():
        raise FileNotFoundError(fbx_path)

    imported_objects = _import_fbx(fbx_path)
    try:
        selected_mesh_names = _normalize_mesh_names(mesh_names, mesh_name)
        mesh_objects = _find_mesh_objects(
            imported_objects,
            mesh_names=selected_mesh_names,
        )
        mesh_objects = _exclude_mesh_objects(mesh_objects, exclude_mesh_name_patterns)
        faces, vertices, blendshapes, metadata = _combine_mesh_tensors(
            mesh_objects=mesh_objects,
            blendshape_name_map=blendshape_name_map,
            dtype=dtype,
            missing_blendshape=missing_blendshape,
        )

        if device is not None:
            torch_device = torch.device(device)
            vertices = vertices.to(torch_device)
            faces = faces.to(torch_device)
            blendshapes = {
                name: delta.to(torch_device) for name, delta in blendshapes.items()
            }
            metadata = _move_metadata_tensors(metadata, torch_device)
        if include_metadata:
            return faces, vertices, blendshapes, metadata
        return faces, vertices, blendshapes
    finally:
        _remove_objects(imported_objects)


def _normalize_mesh_names(
    mesh_names: Optional[Union[str, Sequence[str]]],
    mesh_name: Optional[str],
) -> Optional[Tuple[str, ...]]:
    selected_mesh_names = (mesh_name,) if mesh_name is not None else mesh_names
    if selected_mesh_names is None:
        return None
    if isinstance(selected_mesh_names, str):
        return (selected_mesh_names,)
    return tuple(selected_mesh_names)


def _import_fbx(fbx_path: Path) -> Tuple["bpy.types.Object", ...]:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(fbx_path))
    return tuple(obj for obj in bpy.data.objects if obj not in before)


def _find_mesh_objects(
    imported_objects: Iterable["bpy.types.Object"],
    mesh_names: Optional[Sequence[str]],
) -> Tuple["bpy.types.Object", ...]:
    meshes = tuple(obj for obj in imported_objects if obj.type == "MESH")
    if not meshes:
        raise ValueError("No mesh objects were found in the FBX file.")

    if mesh_names is not None:
        by_name = {_base_blender_name(obj.name): obj for obj in meshes}
        by_name.update({obj.name: obj for obj in meshes})
        selected = []
        seen = set()
        for name in mesh_names:
            obj = by_name.get(name)
            if obj is None or obj.name in seen:
                continue
            selected.append(obj)
            seen.add(obj.name)
        if selected:
            return tuple(selected)

    keyed_meshes = tuple(obj for obj in meshes if obj.data.shape_keys is not None)
    if keyed_meshes:
        return tuple(
            sorted(
                keyed_meshes,
                key=lambda obj: len(obj.data.vertices),
                reverse=True,
            )
        )
    return tuple(
        sorted(
            meshes,
            key=lambda obj: len(obj.data.vertices),
            reverse=True,
        )
    )


def _exclude_mesh_objects(
    mesh_objects: Sequence["bpy.types.Object"],
    patterns: Optional[Sequence[str]],
) -> Tuple["bpy.types.Object", ...]:
    normalized = tuple(str(pattern).strip().lower() for pattern in (patterns or ()))
    normalized = tuple(pattern for pattern in normalized if pattern)
    if not normalized:
        return tuple(mesh_objects)

    kept = tuple(
        mesh_object
        for mesh_object in mesh_objects
        if not _mesh_name_matches_any(mesh_object.name, normalized)
    )
    if not kept:
        raise ValueError(
            "All selected mesh objects were excluded by "
            f"exclude_mesh_name_patterns={tuple(patterns or ())!r}."
        )
    return kept


def _mesh_name_matches_any(name: str, patterns: Sequence[str]) -> bool:
    base_name = _base_blender_name(name).lower()
    full_name = name.lower()
    return any(pattern in base_name or pattern in full_name for pattern in patterns)


def _combine_mesh_tensors(
    mesh_objects: Sequence["bpy.types.Object"],
    blendshape_name_map: Mapping[str, str],
    dtype: torch.dtype,
    missing_blendshape: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, object]]:
    faces_by_mesh = []
    vertices_by_mesh = []
    blendshapes_by_au = {au_name: [] for _, au_name in AU_NAME.items()}
    found_blendshape_by_au = {au_name: False for _, au_name in AU_NAME.items()}
    object_vertex_groups: Dict[str, torch.Tensor] = {}
    section_vertex_groups: Dict[str, list[torch.Tensor]] = {}
    vertex_offset = 0

    for mesh_object in mesh_objects:
        vertices = _neutral_vertices(mesh_object, dtype=dtype)
        faces = _triangulated_faces(mesh_object.data) + vertex_offset
        vertex_ids = torch.arange(
            vertex_offset,
            vertex_offset + vertices.shape[0],
            dtype=torch.long,
        )
        object_vertex_groups[_base_blender_name(mesh_object.name)] = vertex_ids
        for section_name in _section_names_for_mesh(mesh_object.name):
            section_vertex_groups.setdefault(section_name, []).append(vertex_ids)
        key_blocks = _shape_key_blocks(mesh_object)
        for _, au_name in AU_NAME.items():
            fbx_shape_name = blendshape_name_map.get(au_name, au_name)
            found_blendshape_by_au[au_name] |= fbx_shape_name in key_blocks

        blendshapes = _blendshape_deltas(
            mesh_object=mesh_object,
            neutral_vertices=vertices,
            blendshape_name_map=blendshape_name_map,
            dtype=dtype,
            missing_blendshape="zeros",
        )

        vertices_by_mesh.append(vertices)
        faces_by_mesh.append(faces)
        for _, au_name in AU_NAME.items():
            blendshapes_by_au[au_name].append(blendshapes[au_name])
        vertex_offset += vertices.shape[0]

    combined_vertices = torch.cat(vertices_by_mesh, dim=0)
    combined_faces = torch.cat(faces_by_mesh, dim=0)
    combined_blendshapes = {
        au_name: torch.cat(deltas, dim=0)
        for au_name, deltas in blendshapes_by_au.items()
    }
    combined_blendshapes = _split_shared_metahuman_blendshapes(
        vertices=combined_vertices,
        blendshapes=combined_blendshapes,
        blendshape_name_map=blendshape_name_map,
    )
    if missing_blendshape == "raise":
        missing = [
            au_name
            for au_name, found in found_blendshape_by_au.items()
            if not found
        ]
        if missing:
            raise KeyError(f"Missing FBX shape keys for AUs: {missing}")
    metadata = {
        "mesh_object_names": tuple(_base_blender_name(obj.name) for obj in mesh_objects),
        "vertex_groups": {
            "objects": object_vertex_groups,
            "sections": {
                name: torch.cat(groups, dim=0).unique(sorted=True)
                for name, groups in section_vertex_groups.items()
            },
        },
    }
    return combined_faces, combined_vertices, combined_blendshapes, metadata


def _section_names_for_mesh(mesh_name: str) -> Tuple[str, ...]:
    name = _base_blender_name(mesh_name).lower()
    sections = []
    if any(token in name for token in ("eye", "eyeball", "cornea", "lacrimal")):
        sections.append("eyes")
    if any(token in name for token in ("teeth", "tooth")):
        sections.extend(("mouth", "teeth"))
    elif any(token in name for token in ("mouth", "lip", "tongue", "gum")):
        sections.append("mouth")
    if any(token in name for token in ("head", "face")):
        sections.append("head")
    return tuple(dict.fromkeys(sections))


def _move_metadata_tensors(metadata: Dict[str, object], device: torch.device) -> Dict[str, object]:
    def move(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value

    return move(metadata)


def _split_shared_metahuman_blendshapes(
    vertices: torch.Tensor,
    blendshapes: Dict[str, torch.Tensor],
    blendshape_name_map: Mapping[str, str],
) -> Dict[str, torch.Tensor]:
    side_masks: Dict[str, torch.Tensor] = {}

    for au_name, (shared_shape_name, side) in METAHUMAN_SHARED_BLENDSHAPE_SIDES.items():
        if blendshape_name_map.get(au_name, au_name) != shared_shape_name:
            continue
        if side not in side_masks:
            side_masks[side] = _metahuman_side_mask(vertices, side)
        blendshapes[au_name] = blendshapes[au_name] * side_masks[side]

    return blendshapes


def _metahuman_side_mask(vertices: torch.Tensor, side: str) -> torch.Tensor:
    center_x = (vertices[:, 0].amin() + vertices[:, 0].amax()) * 0.5
    transition_half_width = (
        (vertices[:, 0].amax() - vertices[:, 0].amin())
        * METAHUMAN_SHARED_BLENDSHAPE_TRANSITION_RATIO
    )
    transition_half_width = transition_half_width.clamp_min(1.0e-8)
    t = (
        (vertices[:, 0] - center_x + transition_half_width)
        / (transition_half_width * 2.0)
    ).clamp(0.0, 1.0)
    left_mask = t * t * (3.0 - 2.0 * t)

    if side == "L":
        mask = left_mask
    elif side == "R":
        mask = 1.0 - left_mask
    else:
        raise ValueError(f"Unknown MetaHuman side: {side!r}")
    return mask.unsqueeze(-1)


def _base_blender_name(name: str) -> str:
    suffix = name.rpartition(".")[2]
    if len(suffix) == 3 and suffix.isdigit():
        return name[:-4]
    return name


def _triangulated_faces(mesh: "bpy.types.Mesh") -> torch.Tensor:
    mesh.calc_loop_triangles()
    faces = [
        tuple(loop_triangle.vertices)
        for loop_triangle in mesh.loop_triangles
        if len(loop_triangle.vertices) == 3
    ]
    if not faces:
        raise ValueError(f"Mesh {mesh.name!r} does not contain triangle faces.")
    return torch.tensor(faces, dtype=torch.long)


def _neutral_vertices(
    mesh_object: "bpy.types.Object",
    dtype: torch.dtype,
) -> torch.Tensor:
    shape_keys = mesh_object.data.shape_keys
    if shape_keys is not None and "Basis" in shape_keys.key_blocks:
        return _key_block_vertices(mesh_object, shape_keys.key_blocks["Basis"], dtype)

    coords = [0.0] * (len(mesh_object.data.vertices) * 3)
    mesh_object.data.vertices.foreach_get("co", coords)
    vertices = torch.tensor(coords, dtype=dtype).view(-1, 3)
    return _apply_matrix_world(vertices, mesh_object.matrix_world, dtype)


def _blendshape_deltas(
    mesh_object: "bpy.types.Object",
    neutral_vertices: torch.Tensor,
    blendshape_name_map: Mapping[str, str],
    dtype: torch.dtype,
    missing_blendshape: str,
) -> Dict[str, torch.Tensor]:
    key_blocks = _shape_key_blocks(mesh_object)
    zeros = torch.zeros_like(neutral_vertices)
    blendshapes: Dict[str, torch.Tensor] = {}

    for _, au_name in AU_NAME.items():
        fbx_shape_name = blendshape_name_map.get(au_name, au_name)
        if fbx_shape_name not in key_blocks:
            if missing_blendshape == "raise":
                raise KeyError(
                    f"Blendshape {au_name!r} maps to missing FBX shape key "
                    f"{fbx_shape_name!r}."
                )
            blendshapes[au_name] = zeros.clone()
            continue

        target_vertices = _key_block_vertices(
            mesh_object,
            key_blocks[fbx_shape_name],
            dtype,
        )
        blendshapes[au_name] = target_vertices - neutral_vertices

    return blendshapes


def _shape_key_blocks(mesh_object: "bpy.types.Object"):
    shape_keys = mesh_object.data.shape_keys
    if shape_keys is None:
        return {}
    return shape_keys.key_blocks


def _key_block_vertices(
    mesh_object: "bpy.types.Object",
    key_block: "bpy.types.ShapeKey",
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = [0.0] * (len(key_block.data) * 3)
    key_block.data.foreach_get("co", coords)
    vertices = torch.tensor(coords, dtype=dtype).view(-1, 3)
    return _apply_matrix_world(vertices, mesh_object.matrix_world, dtype)


def _apply_matrix_world(
    vertices: torch.Tensor,
    matrix_world: "bpy.types.Matrix",
    dtype: torch.dtype,
) -> torch.Tensor:
    matrix = torch.tensor(
        [[matrix_world[row][col] for col in range(4)] for row in range(4)],
        dtype=dtype,
    )
    linear = matrix[:3, :3]
    translation = matrix[:3, 3]
    return vertices @ linear.T + translation


def _remove_objects(objects: Iterable["bpy.types.Object"]) -> None:
    for obj in objects:
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)


load_fbx_tensors = fbx_to_tensor
