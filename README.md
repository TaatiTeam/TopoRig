# TopoRig

TopoRig predicts facial blendshape deformations for neutral face meshes.

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/temp)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97-dataset-87CEFA)](https://huggingface.co/datasets/SoroushMehraban/TopoRig_Dataset)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97-model-FFD21E)](https://huggingface.co/SoroushMehraban/TopoRig)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97-space-FFC266)](https://huggingface.co/spaces/SoroushMehraban/TopoRig)

## Install

The commands below target Linux x86_64 with Python 3.11. GPU execution uses
PyTorch 2.7.0 with CUDA 12.8 and requires a compatible NVIDIA driver.
Run all commands from the project root in the same shell.

```bash
git clone https://github.com/SoroushMehraban/TopoRig_Project.git
cd TopoRig_Project
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[demo,hub]'
```

For CPU inference, use `--index-url https://download.pytorch.org/whl/cpu`
in the PyTorch installation command and `--device cpu` in the demo commands.
See [PyTorch installation options](https://pytorch.org/get-started/previous-versions/#v270).

The Python dependencies include Blender's `bpy` module. Automatic landmark
mapping also needs the Blender executable:

```bash
mkdir -p third_party
curl -fL https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz \
  -o third_party/blender-4.2.0-linux-x64.tar.xz
tar -xJf third_party/blender-4.2.0-linux-x64.tar.xz -C third_party
export PATH="$PWD/third_party/blender-4.2.0-linux-x64:$PATH"
blender --background --factory-startup --python-expr 'import bpy; print(bpy.app.version_string)'
python -c 'import bpy, mediapipe as mp; print(bpy.app.version_string); print(mp.solutions.face_mesh.FaceMesh)'
```

## Download checkpoints and demo assets

```bash
export TOPORIG_DATA_ROOT="$PWD/data"
export TOPORIG_METADATA_ROOT="$TOPORIG_DATA_ROOT/metadata"
export TOPORIG_REFERENCE_ROOT="$TOPORIG_DATA_ROOT/reference_assets"
export TOPORIG_CHECKPOINT_ROOT="$PWD/checkpoints"
export TOPORIG_CACHE_ROOT="$PWD/.cache"
export TOPORIG_OUTPUT_ROOT="$PWD/runs"
export TOPORIG_RESULTS_ROOT="$PWD/outputs"
export TOPORIG_DATA_REVISION=1764fa2d5fb783fb5a190a8f95b04bd2224c2729
export TOPORIG_MODEL_REVISION=6ee838a38d45f841b17300d0b9f8b3f59c7eb496

hf download SoroushMehraban/TopoRig stage1.pt stage2.pt \
  --revision "$TOPORIG_MODEL_REVISION" --local-dir "$TOPORIG_CHECKPOINT_ROOT"
hf download SoroushMehraban/TopoRig_Dataset --repo-type dataset \
  --revision "$TOPORIG_DATA_REVISION" --include 'reference_assets/*' \
  --local-dir "$TOPORIG_DATA_ROOT"
```

`stage1.pt` is the mesh-pretraining checkpoint from epoch 20. `stage2.pt` is
the refined checkpoint from epoch 4 and is the default for inference. Both
were selected by validation mesh MAE. Demo assets and checkpoints require about
170 MiB. The complete training dataset is downloaded in the training guide.

## Run the demo

### Web interface

After installation and checkpoint/demo-asset downloads above:

```bash
python -m pip install -e '.[app]'
python gradio_app.py --device cuda
```

Open `http://127.0.0.1:7860`. Click a front-view thumbnail from `samples_demo/` or
upload a GLB, FBX or OBJ. Choose an action unit and intensity, then click
**Generate expression**. Both viewers support rotation, zoom and pan; the
result can be downloaded as GLB. Intensity 0 keeps the neutral mesh and 1 uses
the full expression. Two optional settings are off by default:

- **Mouth postprocessing:** separates joined lips and fits teeth and a tongue inside the mouth.
- **Eyes postprocessing:** replaces the eyeballs, preserves their visible appearance,
  and enables gaze AUs through the project's eye-animation pipeline.

At intensity 0, selected postprocessing still applies to the neutral expression.

GLB uploads should embed their textures; OBJ uploads display geometry only.
Eye-direction controls need clearly visible, textured eyes.
The first expression for a new mesh takes longer while facial landmarks are
prepared. Later expressions reuse the loaded model and cached mesh data.
Use `--device cpu` for CPU inference, `--checkpoint` for different weights,
or `--host 0.0.0.0 --port 7860` to listen on a remote server. Blender is found
on `PATH` or under `third_party/blender-*`; `--blender` overrides it.

### Command line

Generate a jaw-opening deformation for the included sample:

```bash
python demo.py \
  --mesh "$TOPORIG_REFERENCE_ROOT/examples/neutral_meshes/mesh_00006.glb" \
  --checkpoint "$TOPORIG_CHECKPOINT_ROOT/stage2.pt" \
  --au-id jawOpen --device cuda \
  --output-dir "$TOPORIG_RESULTS_ROOT/demo"
```

Outputs include the deformed GLB and preview images. Replace `--mesh` with a
neutral FBX or GLB to use your own head. Automatic alignment uses the downloaded
ICT reference mesh. For OBJ inputs, provide a landmark mapping beside the
mesh with the same stem and a `.json` extension. `--au-id` accepts
an AU name or an integer from 0 to 52; the names are printed by:

```bash
python -c 'from utils.fbx_to_tensor import AU_NAME; print(AU_NAME)'
```

To split a closed mouth and add the supplied teeth/tongue assembly:

```bash
python scripts/prepare_lip_split.py \
  --mesh "$TOPORIG_REFERENCE_ROOT/examples/neutral_meshes/mesh_00006.glb" \
  --checkpoint "$TOPORIG_CHECKPOINT_ROOT/stage2.pt" \
  --oral-asset "$TOPORIG_REFERENCE_ROOT/ict/ict_oral_assembly.npz" \
  --device cuda --output-dir "$TOPORIG_RESULTS_ROOT/lip_split"
```

Use `python demo.py --help` and `python scripts/prepare_lip_split.py --help`
for input conventions, gaze animation, export and landmark-cache options.
For training from scratch or evaluating the published checkpoints, follow
[Training and evaluation](docs/REPRODUCING.md).

## License

The TopoRig code and model checkpoints are available for noncommercial academic
research, education, and teaching under the
[TopoRig Academic Use License 1.0](LICENSE.md).

Commercial use requires separate written permission from the applicable rights
holders. Third-party dependencies and assets remain subject to their respective
licenses.
