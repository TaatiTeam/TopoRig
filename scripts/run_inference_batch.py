from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import demo
import train
from utils.fbx_to_tensor import AU_NAME


def main() -> None:
    args = parse_args()
    mesh_path = args.mesh.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = demo.load_checkpoint(checkpoint_path)
    config = demo.checkpoint_config(checkpoint, args.config)
    device = demo.resolve_device(args.device, config)
    model = demo.load_model(checkpoint, config, device)

    raw_mesh = demo.load_mesh_for_model(mesh_path)
    input_landmark_json = demo.ensure_mediapipe_landmarks(
        mesh_path=mesh_path,
        cache_dir=args.landmark_cache_dir.expanduser(),
        mapper_path=args.mediapipe_mapper.expanduser(),
        blender=args.blender,
        force=args.refresh_landmarks,
    )
    reference_mesh_path = demo.metahuman_sample_mesh(
        args.metahuman_sample_dir.expanduser()
    )
    reference_landmark_json = demo.ensure_mediapipe_landmarks(
        mesh_path=reference_mesh_path,
        cache_dir=args.landmark_cache_dir.expanduser(),
        mapper_path=args.mediapipe_mapper.expanduser(),
        blender=args.blender,
        force=args.refresh_landmarks,
        prefer_sidecar=True,
    )
    mesh = preprocess_mesh_for_batch(
        raw_mesh=raw_mesh,
        config=config,
        mesh_path=mesh_path,
        input_convention=args.input_convention,
        input_landmarks=demo.load_landmark_mapping(input_landmark_json),
        reference_mesh_path=reference_mesh_path,
        reference_landmarks=demo.load_landmark_mapping(reference_landmark_json),
        min_alignment_landmarks=args.min_alignment_landmarks,
        alignment_mode=args.alignment_mode,
    )

    action_unit_ids = parse_au_ids(args.au_ids)
    prefix = args.prefix or mesh_path.stem
    rows: list[dict[str, str]] = []
    image_paths: list[Path] = []
    write_neutral_mesh(
        mesh=mesh,
        output_dir=output_dir,
        prefix=prefix,
        output_format=args.output_format,
        output_convention=args.output_convention,
    )

    for index, action_unit_id in enumerate(action_unit_ids, start=1):
        action_unit_name = AU_NAME[action_unit_id]
        stem = f"{prefix}_au{action_unit_id:02d}_{action_unit_name}"
        image_path = output_dir / f"{stem}.jpg"
        output_mesh_path = output_dir / f"{stem}.{args.output_format}"

        deformed_vertices, predicted_delta = demo.run_model(
            model=model,
            mesh=mesh,
            action_unit_id=action_unit_id,
            config=config,
            device=device,
        )
        image = demo.render_output_image(
            neutral_vertices=mesh["vertices"],
            deformed_vertices=deformed_vertices,
            normals=mesh.get("normals"),
            faces=mesh["faces"],
            action_unit_id=action_unit_id,
            config=config,
            device=device,
        )
        image.save(image_path, quality=95)

        export_vertices = demo.output_vertices_for_convention(
            deformed_vertices,
            args.output_convention,
        )
        demo.export_mesh(
            vertices=export_vertices,
            faces=mesh["faces"],
            normals=demo.recompute_vertex_normals(export_vertices, mesh["faces"]),
            path=output_mesh_path,
        )

        row = {
            "action_unit_id": str(action_unit_id),
            "action_unit_name": action_unit_name,
            "delta_mean_norm": f"{predicted_delta.norm(dim=-1).mean().item():.9f}",
            "delta_max_norm": f"{predicted_delta.norm(dim=-1).max().item():.9f}",
            "image_path": str(image_path),
            "mesh_path": str(output_mesh_path),
        }
        rows.append(row)
        image_paths.append(image_path)
        print(
            f"[{index:03d}/{len(action_unit_ids):03d}] "
            f"AU{action_unit_id:02d} {action_unit_name} "
            f"mean={row['delta_mean_norm']} max={row['delta_max_norm']}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_path = output_dir / "summary.csv"
    write_summary(summary_path, rows)
    if not args.no_contact_sheet:
        write_contact_sheet(output_dir / "contact_sheet.jpg", image_paths)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Loaded mesh:       {mesh_path}")
    print(f"Wrote summary:     {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a TopoRig checkpoint over multiple action units."
    )
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--au-ids",
        nargs="+",
        default=["all"],
        help="Action-unit ids or names. Use 'all' for every AU_NAME entry.",
    )
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--output-format", choices=("glb", "obj"), default="glb")
    parser.add_argument(
        "--output-convention",
        choices=("metahuman", "model"),
        default="metahuman",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--input-convention",
        choices=(
            "auto",
            "fbx",
            "custom-glb",
            "custom-glb-y-front",
            "custom-glb-neg-y-front",
            "custom-glb-y-up-z-front",
            "custom-glb-y-up-neg-z-front",
        ),
        default="auto",
    )
    parser.add_argument(
        "--alignment-mode",
        choices=("none", "similarity", "scale_translation"),
        default="similarity",
        help=(
            "Landmark alignment mode. Use none for an already canonicalized "
            "mesh, or scale_translation for custom GLBs whose orientation "
            "should be preserved."
        ),
    )
    parser.add_argument(
        "--metahuman-sample-dir",
        type=Path,
        default=demo.DEFAULT_METAHUMAN_SAMPLE_DIR,
    )
    parser.add_argument(
        "--landmark-cache-dir",
        type=Path,
        default=demo.DEFAULT_LANDMARK_CACHE_DIR,
    )
    parser.add_argument(
        "--mediapipe-mapper",
        type=Path,
        default=demo.DEFAULT_MEDIAPIPE_MAPPER,
    )
    parser.add_argument("--blender", default=None)
    parser.add_argument("--refresh-landmarks", action="store_true")
    parser.add_argument("--min-alignment-landmarks", type=int, default=32)
    parser.add_argument("--no-contact-sheet", action="store_true")
    return parser.parse_args()


def preprocess_mesh_for_batch(
    raw_mesh: dict[str, torch.Tensor],
    config: dict,
    mesh_path: Path,
    input_convention: str,
    input_landmarks: dict[int, int],
    reference_mesh_path: Path,
    reference_landmarks: dict[int, int],
    min_alignment_landmarks: int,
    alignment_mode: str,
) -> dict[str, torch.Tensor]:
    processed_mesh = train.preprocess_custom_visualization_mesh(
        raw_mesh,
        config,
        input_suffix=mesh_path.suffix.lower(),
        input_convention=input_convention,
    )
    if alignment_mode == "none":
        processed_mesh["landmarks_3d"] = train.landmark_payload_from_mapping(
            processed_mesh,
            input_landmarks,
        )
        return processed_mesh

    reference_mesh = train.preprocess_custom_visualization_mesh(
        demo.load_mesh_for_model(reference_mesh_path),
        config,
        input_suffix=reference_mesh_path.suffix.lower(),
        input_convention="auto",
    )
    aligned_mesh = train.align_mesh_to_reference_landmarks(
        mesh=processed_mesh,
        input_landmarks=input_landmarks,
        reference_mesh=reference_mesh,
        reference_landmarks=reference_landmarks,
        min_landmarks=min_alignment_landmarks,
        alignment_mode=alignment_mode,
    )
    aligned_mesh["landmarks_3d"] = train.landmark_payload_from_mapping(
        aligned_mesh,
        input_landmarks,
    )
    return aligned_mesh


def write_neutral_mesh(
    mesh: dict[str, torch.Tensor],
    output_dir: Path,
    prefix: str,
    output_format: str,
    output_convention: str,
) -> None:
    neutral_vertices = demo.output_vertices_for_convention(
        mesh["vertices"],
        output_convention,
    )
    demo.export_mesh(
        vertices=neutral_vertices,
        faces=mesh["faces"],
        normals=demo.recompute_vertex_normals(neutral_vertices, mesh["faces"]),
        path=output_dir / f"{prefix}_canonical_neutral.{output_format}",
    )


def parse_au_ids(values: Sequence[str]) -> list[int]:
    if len(values) == 1 and values[0].strip().lower() == "all":
        return sorted(AU_NAME)
    parsed = [demo.parse_au_id(value) for value in values]
    unknown = [action_unit_id for action_unit_id in parsed if action_unit_id not in AU_NAME]
    if unknown:
        raise ValueError(f"Unknown AU ids: {unknown}")
    return parsed


def write_summary(path: Path, rows: Sequence[dict[str, str]]) -> None:
    fieldnames = [
        "action_unit_id",
        "action_unit_name",
        "delta_mean_norm",
        "delta_max_norm",
        "image_path",
        "mesh_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_contact_sheet(path: Path, image_paths: Sequence[Path]) -> None:
    if not image_paths:
        return
    thumbnails: list[Image.Image] = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((420, 220), Image.Resampling.LANCZOS)
        thumbnails.append(image.copy())
        image.close()

    columns = 4
    cell_width = max(image.width for image in thumbnails)
    cell_height = max(image.height for image in thumbnails)
    rows = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    for index, image in enumerate(thumbnails):
        x = (index % columns) * cell_width + (cell_width - image.width) // 2
        y = (index // columns) * cell_height + (cell_height - image.height) // 2
        sheet.paste(image, (x, y))
    sheet.save(path, quality=95)


if __name__ == "__main__":
    main()
