#!/usr/bin/env python3
"""Launch the TopoRig web demo: python gradio_app.py --device cuda."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re

import gradio as gr
import torch

from utils.fbx_to_tensor import AU_NAME
from utils.gradio_runtime import ROOT, RigRuntime


LOGGER = logging.getLogger(__name__)
CSS = """
.gradio-container { width: 100% !important; max-width: 1480px !important; margin: auto; }
#intro { padding: 20px 0 12px; }
#intro h1 { font-size: 38px; letter-spacing: -1.5px; margin-bottom: 8px; }
#intro p { color: var(--body-text-color-subdued); font-size: 16px; }
#controls { padding: 20px; border: 1px solid var(--border-color-primary);
            border-radius: 16px; background: var(--block-background-fill); }
#workspace { display: grid; grid-template-columns: minmax(260px, 310px) minmax(0, 1fr);
             align-items: start; gap: 24px; }
#workspace > .column { min-width: 0 !important; }
#viewers { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
@media (max-width: 900px) { #workspace { grid-template-columns: 1fr; } }
@media (max-width: 580px) { #viewers { grid-template-columns: 1fr; } }
.mesh-view { border-radius: 16px !important; overflow: hidden; }
#sample-faces { border-radius: 16px; margin-bottom: 12px; }
#sample-faces .grid-container { padding-top: 28px !important; }
#sample-faces .gallery-item { height: 160px !important; aspect-ratio: auto !important; }
#sample-faces .thumbnail-item { height: 100% !important; }
@media (max-width: 580px) {
  #sample-faces, #sample-faces .grid-wrap { height: 370px !important; }
  #sample-faces .grid-container { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
}
#status { min-height: 50px; font-size: 14px; }
footer { display: none !important; }
"""


def au_label(name: str) -> str:
    name = name.replace("_L", " left").replace("_R", " right")
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).capitalize()


def create_app(runtime: RigRuntime, samples_dir: Path = ROOT / "samples_demo") -> gr.Blocks:
    samples = {path.stem: path.resolve() for path in sorted(samples_dir.glob("*.glb"))}
    first = next(iter(samples), None)
    initial = None

    def load_samples():
        return [str(runtime.sample_thumbnail(path)) for path in samples.values()]

    def choose_sample(selection, event: gr.SelectData):
        index = event.index
        if not isinstance(index, int) or not 0 <= index < len(samples):
            raise gr.Error("Choose one of the example faces.")
        return (list(samples)[index], *clear_mesh(selection))
    def au_choices(eyes_enabled):
        return [(f"{index:02d} · {au_label(name)}", index) for index, name in AU_NAME.items()
                if eyes_enabled or index not in range(12, 20)]

    def update_eye_controls(enabled, current_au):
        value = 26 if not enabled and current_au in range(12, 20) else current_au
        return gr.Dropdown(choices=au_choices(enabled), value=value)

    def select_mesh(source, sample, upload, selection):
        revision = selection["mesh_revision"]
        selection["path"] = None
        if source == "Sample":
            if sample not in samples:
                return selection, None, None, gr.DownloadButton(visible=False), "Choose a sample face."
            source_path = samples[sample]
        elif upload:
            source_path = Path(upload)
        else:
            return selection, None, None, gr.DownloadButton(visible=False), "Upload a neutral face to begin."
        try:
            path = runtime.prepare_upload(source_path)
        except Exception as exc:
            if revision != selection["mesh_revision"]:
                return (gr.skip(),) * 5
            LOGGER.exception("Unable to prepare mesh")
            raise gr.Error("Could not read this mesh. Use a self-contained GLB, FBX or OBJ face.") from exc
        if revision != selection["mesh_revision"]:
            return (gr.skip(),) * 5
        selection["path"] = str(path)
        return selection, str(path), None, gr.DownloadButton(visible=False), "Ready. Choose an action unit and intensity."

    def generate(selection, action_unit, intensity, mouth_postprocessing, eyes_postprocessing,
                 progress=gr.Progress()):
        path, revision = selection["path"], selection["revision"]
        if not path:
            raise gr.Error("Select a sample or upload a neutral face first.")
        progress(0.05, desc="Preparing face and expression…")
        try:
            output = runtime.predict(Path(path), int(action_unit), float(intensity),
                                     mouth_postprocessing=mouth_postprocessing,
                                     eyes_postprocessing=eyes_postprocessing)
        except (FileNotFoundError, ValueError) as exc:
            if revision != selection["revision"]:
                return (gr.skip(),) * 3
            LOGGER.exception("Inference prerequisites or input are invalid")
            raise gr.Error(str(exc)) from exc
        except Exception as exc:
            if revision != selection["revision"]:
                return (gr.skip(),) * 3
            LOGGER.exception("Expression generation failed")
            if eyes_postprocessing:
                raise gr.Error("Could not fit the eyes for this gaze control. Use a textured face "
                               "with clearly visible eyes, or choose another action unit.") from exc
            raise gr.Error("Could not generate this expression. Use a clearly visible, upright neutral face. "
                           "The server log contains the processing error.") from exc
        if revision != selection["revision"]:
            return (gr.skip(),) * 3
        progress(1, desc="Expression ready")
        description = f"**{au_label(AU_NAME[int(action_unit)])} · {float(intensity):.0%} intensity**"
        enabled = [name for name, value in (("mouth", mouth_postprocessing),
                                            ("eyes", eyes_postprocessing)) if value]
        if enabled:
            description += "\n\nPostprocessing: " + ", ".join(enabled) + "."
        return str(output), gr.DownloadButton(value=str(output), visible=True), description

    def clear_result(selection):
        selection["revision"] += 1
        return None, gr.DownloadButton(visible=False), "Settings changed. Generate to update the result."

    def clear_mesh(selection):
        selection["revision"] += 1
        selection["mesh_revision"] += 1
        selection["path"] = None
        return selection, None, None, gr.DownloadButton(visible=False), "Preparing input…"

    with gr.Blocks(title="TopoRig", analytics_enabled=False, fill_width=True,
                   delete_cache=(3600, 86400)) as app:
        gr.Markdown("# TopoRig\nBring a neutral face to life. Choose a mesh, pick an expression, and explore it in 3D.",
                    elem_id="intro")
        selected = gr.State({"path": initial, "revision": 0, "mesh_revision": 0})
        sample = gr.State(first)
        with gr.Row(elem_id="workspace"):
            with gr.Column(scale=1, min_width=270, elem_id="controls"):
                source = gr.Radio(["Sample", "Upload"], value="Sample" if first else "Upload",
                                  label="Mesh source")
                upload = gr.File(label="Upload a neutral mesh", file_types=[".glb", ".fbx", ".obj"],
                                 type="filepath", visible=not bool(first))
                mouth = gr.Checkbox(False, label="Mouth postprocessing", info=(
                    "Separates joined upper and lower lips and fits teeth and a tongue inside "
                    "the mouth, so mouth-opening expressions reveal an interior."))
                eyes = gr.Checkbox(False, label="Eyes postprocessing", info=(
                    "Replaces the eyeballs with complete spheres, preserving their visible appearance, "
                    "and unlocks eye-gaze controls. Requires textured, clearly visible eyes."))
                action_unit = gr.Dropdown(au_choices(False), value=26, label="Action unit", interactive=True)
                intensity = gr.Slider(0, 1, value=1, step=0.05, label="Intensity",
                                      info="0 = neutral · 1 = full expression")
                run = gr.Button("Generate expression", variant="primary", size="lg")
                download = gr.DownloadButton("Download result (.glb)", visible=False)
                status = gr.Markdown("Preparing input…" if first
                                     else "Upload a neutral face to begin.", elem_id="status")
            with gr.Column(scale=3, min_width=520):
                gallery = gr.Gallery(label="Example faces — click to select", columns=4, rows=1,
                                     height=200, object_fit="contain", allow_preview=False,
                                     selected_index=0 if first else None, interactive=False,
                                     buttons=[], visible=bool(first), elem_id="sample-faces")
                with gr.Row(elem_id="viewers"):
                    input_view = gr.Model3D(initial, label="Neutral input", height=500,
                                            clear_color=(0.055, 0.07, 0.09, 1),
                                            camera_position=(90, 85, None), interactive=False,
                                            elem_classes="mesh-view")
                    output_view = gr.Model3D(label="Generated expression", height=500,
                                             clear_color=(0.055, 0.07, 0.09, 1),
                                             camera_position=(90, 85, None), interactive=False,
                                             elem_classes="mesh-view")
                gr.Markdown("Drag to rotate · Scroll to zoom · Right-drag to pan")

        source.change(lambda value: (gr.Gallery(visible=value == "Sample"),
                                     gr.File(visible=value == "Upload")),
                      source, [gallery, upload], queue=False, api_visibility="private")
        app.load(load_samples, outputs=gallery, concurrency_id="inference", concurrency_limit=1,
                 api_visibility="private")
        app.load(select_mesh, [source, sample, upload, selected],
                 [selected, input_view, output_view, download, status],
                 concurrency_id="inference", concurrency_limit=1, api_visibility="private")
        gallery.select(choose_sample, selected,
                       [sample, selected, input_view, output_view, download, status],
                       queue=False, api_visibility="private").then(
            select_mesh, [source, sample, upload, selected],
            [selected, input_view, output_view, download, status],
            concurrency_id="inference", concurrency_limit=1, api_visibility="private",
        )
        gr.on([source.change, upload.change], clear_mesh, selected,
              [selected, input_view, output_view, download, status], queue=False,
              api_visibility="private").then(
            select_mesh, [source, sample, upload, selected],
            [selected, input_view, output_view, download, status],
            concurrency_id="inference", concurrency_limit=1, api_name="select_mesh",
        )
        # Clear stale results as soon as the user changes an expression or mesh.
        eyes.change(update_eye_controls, [eyes, action_unit], action_unit,
                    queue=False, api_visibility="private")
        gr.on([action_unit.change, intensity.change, mouth.change, eyes.change],
              clear_result, selected, [output_view, download, status], queue=False,
              api_visibility="private")
        run.click(clear_result, selected, [output_view, download, status],
                  queue=False, api_visibility="private").then(
            generate, [selected, action_unit, intensity, mouth, eyes], [output_view, download, status],
            concurrency_id="inference", concurrency_limit=1, api_name="generate",
        )
    return app.queue(max_size=16, default_concurrency_limit=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--blender", help="Path to the Blender executable")
    parser.add_argument("--samples-dir", type=Path, default=ROOT / "samples_demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--threads", type=int, default=4, help="CPU inference threads")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    runtime = RigRuntime(args.checkpoint, args.device, args.blender)
    app = create_app(runtime, args.samples_dir)
    app.launch(server_name=args.host, server_port=args.port, share=False,
               allowed_paths=[str(args.samples_dir.resolve()), str(runtime.cache / "thumbnails"),
                              str(runtime.cache / "uploads_v2"),
                              str(runtime.cache / "results")],
               max_file_size="200mb", theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate",
                                                         font=["system-ui", "sans-serif"]),
               css=CSS, footer_links=[], run_history=False)


if __name__ == "__main__":
    main()
