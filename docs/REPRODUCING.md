# Training and evaluation

Complete [installation and checkpoint downloads](../README.md) first. Run the
following commands from the project root with the environment variables defined
in the README. Training uses CUDA. The launch commands below use four GPUs and
batch size 4 per GPU; use `--gpus 1` for a single GPU.

## Download the dataset

```bash
python -m dataset.hf_assets --repo-id SoroushMehraban/TopoRig_Dataset \
  --revision "$TOPORIG_DATA_REVISION" --local-dir "$TOPORIG_DATA_ROOT"
```

The complete download is about 113 GiB. Allow about 225 GiB for the download
and extracted source files, plus space for tensor caches, WAFT caches and runs.
Mesh and image files are extracted into `TOPORIG_CACHE_ROOT` as needed.
Use `--components meshes_ict meshes_custom` to download only the data needed
for Stage 1 or checkpoint evaluation.

The dataset includes original split CSVs, exclusions, ICT landmark mappings,
custom mesh landmark sidecars, and `metadata/evaluation_manifest.csv`. Use these
files directly. After exclusions, the training split has 499 ICT and 2,796
custom heads; validation has 63 ICT and 350 custom heads. Stage 2 uses image AU
IDs 8, 9, 10, 11, 20, 21, 22 and 23 from identities without the `v2_` prefix,
with the supplied exclusions and eyelid-cache validation. Its image stream uses
`split: all`; checkpoint selection uses mesh validation. Inspect the complete
settings before launching with:

```bash
python scripts/launch_training.py --config config/stage2.yaml --gpus 4 --print-config
```

Optional integrity check:

```bash
mkdir -p "$TOPORIG_RESULTS_ROOT"
python scripts/validate_dataset_layout.py --data-root "$TOPORIG_DATA_ROOT" \
  --checksums --output "$TOPORIG_RESULTS_ROOT/dataset-validation.json"
```

This check requires all four shard collections.

## Stage 1: mesh pretraining

```bash
python scripts/prepare_training.py --config config/train.yaml --step check
python scripts/prepare_training.py --config config/train.yaml --step all
python scripts/launch_training.py --config config/train.yaml --gpus 4
```

Stage 1 trains for 20 epochs at learning rate 0.0005. Its validation-selected
checkpoint is written to `$TOPORIG_OUTPUT_ROOT/phase1/checkpoints/best.pt`.
Preparation builds train and validation mesh caches before training.

## Stage 2: image and mesh refinement

Stage 2 requires [WAFT](https://github.com/princeton-vl/WAFT/tree/b152ff1cad1af8c185ee7b141997c48ff3334c87)
and two pretrained weight files. Install its inference dependencies in the
same environment. The xformers version below matches PyTorch 2.7.0.

```bash
python -m pip install -e '.[training]'
python -m pip install xformers==0.0.30 \
  --index-url https://download.pytorch.org/whl/cu128
export TOPORIG_WAFT_ROOT="$PWD/third_party/WAFT"
git clone https://github.com/princeton-vl/WAFT.git "$TOPORIG_WAFT_ROOT"
git -C "$TOPORIG_WAFT_ROOT" checkout b152ff1cad1af8c185ee7b141997c48ff3334c87
git -C "$TOPORIG_WAFT_ROOT" submodule update --init --recursive
mkdir -p "$TOPORIG_WAFT_ROOT/ckpts"
gdown 1CxzBQx0iSg6AyIgt6MF0ROlF_cAeZLPC \
  -O "$TOPORIG_WAFT_ROOT/ckpts/waft_a1_adaptation.pth"
hf download depth-anything/Depth-Anything-V2-Small depth_anything_v2_vits.pth \
  --revision 03876f8651c73a60fe4c2c48294e09fcb6838fcf \
  --local-dir "$TOPORIG_WAFT_ROOT/depth-anything-ckpts"
```

WAFT loads `config/a1/tar-c-t.json`. Its source and weight provenance is recorded
in `$TOPORIG_METADATA_ROOT/waft.json`. The first cache-preparation run needs
internet access for additional pretrained backbone weights downloaded by `timm`.

```bash
python scripts/prepare_training.py --config config/stage2.yaml --step check
python scripts/prepare_training.py --config config/stage2.yaml --step all
python scripts/launch_training.py --config config/stage2.yaml --gpus 4 \
  --initial-checkpoint "$TOPORIG_OUTPUT_ROOT/phase1/checkpoints/best.pt"
```

Stage 2 trains for 10 epochs at learning rate 0.0001. Preparation builds mesh,
image, eyelid and WAFT caches in dependency order. To run only refinement, use
`$TOPORIG_CHECKPOINT_ROOT/stage1.pt` as the initial checkpoint. Stage 2's best
checkpoint is written to `$TOPORIG_OUTPUT_ROOT/phase2/checkpoints/best.pt`.

Both stages select `best.pt` using validation mesh MAE. Changing the GPU count
changes the effective batch size. Resume an interrupted stage
by adding `--resume` followed by that stage's `checkpoints/latest.pt` path,
and omit `--initial-checkpoint`. W&B logging is enabled only with `--use-wandb`.
Each run stores its resolved configuration and checkpoints under
`TOPORIG_OUTPUT_ROOT`. Save the code revision and `python -m pip freeze` with
results, alongside the pinned dataset and model revisions from the README.

## Evaluate checkpoints

The supplied benchmark contains 350 ICT and 350 custom heads. Evaluation uses
45 ICT AUs and 41 custom AUs. It reports moving-vertex MAE in millimeters with
a 0.01 mm motion threshold, and errors standardized to a 240 mm head height.
WAFT is unnecessary for this mesh-only evaluation.

```bash
python scripts/evaluate_toporig_checkpoint_physical.py \
  --paired-manifest "$TOPORIG_METADATA_ROOT/evaluation_manifest.csv" --prepare-only

for stage in stage1 stage2; do
  python scripts/evaluate_toporig_checkpoint_physical.py \
    --paired-manifest "$TOPORIG_METADATA_ROOT/evaluation_manifest.csv" \
    --checkpoint "$TOPORIG_CHECKPOINT_ROOT/$stage.pt" \
    --output-dir "$TOPORIG_RESULTS_ROOT/evaluation/$stage" --device cuda
done

python scripts/aggregate_toporig_physical_evaluations.py \
  --result-root "$TOPORIG_RESULTS_ROOT/evaluation" --runs stage1 stage2
```

Replace `--checkpoint` with a trained run's `checkpoints/best.pt` to evaluate
new weights. Each output directory contains `summary.json` and per-head/per-AU
CSV reports. The aggregation command writes `all_checkpoint_summary.csv` and
`all_checkpoint_summary.json` in `$TOPORIG_RESULTS_ROOT/evaluation`.
