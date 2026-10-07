# CS4343: satellite land-cover classification

Runnable PyTorch foundation for EuroSAT sample-efficiency and RGB/13-band experiments, plus an optional Nepal transfer case study. Primary metric: macro-F1 across all ten classes. **The Turing study completed all 63 training/validation runs and 51 final test evaluations on October 7, 2026.** The [shared results and opening instructions](docs/reports/turing/README.md) include the test and validation reports, charts and summary tables. Nepal evaluation remains unfinished. See [verification](docs/verification.md) for setup checks, [experiment plan](docs/experiment-plan.md) for the protocol, and [TODO](TODO.md) for the historical team checklist.

The [supplied final-project report](docs/final-project-report-reference.md) is preserved as a reference. The [October 5 completion audit](docs/project-completion-audit.md) records remaining deliverables, report/code discrepancies, and owners from that report.

Update: the full local dataset is prepared under `data/prepared-ms-v2/`; see [data readiness](docs/data-readiness.md). `EuroSAT_MS/`, data, and all run artifacts are Git-ignored. The earlier `data/prepared/` and `outputs/grid/` use obsolete band metadata/settings and must not be used for new research runs.

The [final study commands](docs/final-study.md) use the frozen 3,000-step protocol and automatically produce results/graphs. Run `bash scripts/run_final_study.sh` for the 63 training/validation jobs, then `bash scripts/run_final_test.sh` for the explicit final test phase. Reports open from `outputs/final-study/report-val/index.html` and `report-test/index.html`. The baseline/grid examples below retain pilot defaults; the final study uses `configs/final-study.yaml`.

## Install

New collaborators can clone the code with:

```sh
git clone https://github.com/doanh280605/CS4343_final_project.git
cd CS4343_final_project
```

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). Run from this repository root:

```sh
uv sync --locked
make check
uv run landcover smoke --output outputs/synthetic-smoke
```

The lockfile installs Python 3.12-compatible dependencies; `.python-version` selects 3.12. Python 3.11–3.13 are allowed. Linux uses the official CPU PyTorch index for lightweight CI; macOS uses PyPI. CUDA experiments require a separately recorded CUDA environment; do not silently change the committed lockfile. `device: auto` chooses CUDA when available, otherwise CPU. Set `device: mps` explicitly on Apple hardware; deterministic operations may be unsupported there.

`uv sync --locked --extra nepal` installs the optional Earth Engine API. No credentials are needed for EuroSAT, local training or tests. Pretrained ResNet-18 downloads official torchvision weights once. Optional cache settings:

```sh
export TORCH_HOME="$PWD/.cache/torch"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"
```

`.env.example` documents optional settings; it is not automatically loaded. If an output already exists, choose a new path/config instead of overwriting it.

## Google Colab and shared experiments

Turing users: [A30 batch setup](docs/final-study.md#turing-a30-batch-jobs) runs through
SLURM and keeps running after SSH disconnects. The supplied scripts use the verified
`ece341x` account and `academic` partition; colleagues need an authorized account.

Open the [Colab notebook](https://colab.research.google.com/github/doanh280605/CS4343_final_project/blob/main/notebooks/landcover_colab.ipynb) and choose a GPU runtime. It installs CUDA PyTorch, caches the verified dataset ZIP in Drive, streams errors/progress, and saves results to Drive. See [study instructions](docs/final-study.md) for storage requirements and interrupted-run behavior.

The notebook requests `landcover-colab.zip`, which is intentionally not committed.
For the existing team study, get the same bundle from the team lead; it includes the
exact source, manifests and fixed splits. On an already prepared checkout, create a
bundle with `.venv/bin/python scripts/build_colab_bundle.py`; it appears under
`outputs/colab-setup/`. A fresh checkout needs the data preparation below first.
Keep the same bundle for an ongoing frozen study. Do not run two notebook sessions
against the same study output folder. Coordinate experiment assignments before
starting another full 63-run study.

## Prepare EuroSAT

The main experiment only requires the multispectral archive: RGB is extracted from the same TIFFs. Download is approximately 2.07 GB, plus extraction space.

```sh
uv run landcover download --kind ms
uv run landcover prepare --ms-root data/raw/ms --output data/prepared-ms-v2
uv run landcover audit --manifest data/prepared-ms-v2/manifest.json --splits data/prepared-ms-v2/splits.json
```

For an already extracted root-level folder, use `--ms-root EuroSAT_MS` instead of downloading again. Do not overwrite an existing preparation. On this Mac, hidden `.pth` flags can prevent editable-package discovery; `export PYTHONPATH="$PWD/src"` from the repository root avoids that local issue.

Optional official JPEG comparison and cross-archive identity verification (approximately 95 MB):

```sh
uv run landcover download --kind rgb
uv run landcover prepare --ms-root data/raw/ms --rgb-root data/raw/rgb --output data/prepared-paired
uv run landcover audit --manifest data/prepared-paired/manifest.json --splits data/prepared-paired/splits.json
```

To use the paired preparation, update both paths in your run configs. Preparation scans nested archive directories automatically. It checks exact class/stem identity between supplied modalities; never silently drops unmatched samples. It hashes each file and persists fixed stratified 60/20/20 splits plus nested 100/10/5/1% training subsets. For all 27,000 samples, split sizes are 16,200/5,400/5,400. Store/share the resulting manifests alongside experiment artifacts; they are ignored by Git because they contain dataset paths. Recreate with the same archive contents and split seed 2026, or copy the full data tree with relative paths intact.

EuroSAT TIFF bands: B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B8A. RGB uses B4/B3/B2. The v2 schema corrects the previous placement of B8A; new MS checkpoints record their band schema, and missing/incompatible MS checkpoint schemas are rejected. TIFF digital numbers are divided by 10,000; JPEG values by 255. Channel normalization is fit on each selected training subset, reused for validation/test and stored in the checkpoint. Flips and 90-degree rotations are training-only.

## Quick real-data baseline

This small JPEG run is a pipeline check, **not a benchmark or the RGB/MS comparison**:

```sh
uv run landcover download --kind rgb
uv run landcover prepare --rgb-root data/raw/rgb --output data/real-smoke --limit-per-class 20
uv run landcover train --config configs/real-smoke.yaml
uv run landcover evaluate --checkpoint outputs/real-smoke/best.pt
```

For a complete compact-CNN baseline after preparing multispectral data:

```sh
uv run landcover train --config configs/baseline.yaml
uv run landcover evaluate --checkpoint outputs/baseline/best.pt --split val
```

Configured baseline: four convolution/BN/ReLU/max-pool blocks (32/64/128/256), global average pooling, dropout 0.5 and ten outputs. Training uses cross-entropy, AdamW and 1,000 optimizer steps with 50 warm-up steps followed by cosine decay. This step budget is a starting configuration, not an established sufficient training duration. Choose it using validation-only pilots before the main grid.

## Experiments and evaluation

```sh
uv run landcover grid --output outputs/grid
# Inspect configs, storage and compute budget before explicitly running:
bash outputs/grid/run.sh
```

Generation produces **48** ResNet-18 configs: pretrained/scratch × RGB/MS × 100/10/5/1% × seeds 42/43/44. It uses `configs/baseline.yaml` unless `--base-config` selects another file. Choose a fresh output directory; the original local `outputs/grid/` is obsolete. A configurable step budget, batch size and evaluation schedule are shared across fractions. Full batches are sampled with replacement. The same persisted subsets are shared across architectures, initializations, modalities and training seeds.

Four compact-CNN ablations are in `configs/ablations/`: augmentation on/off × dropout 0/0.5 at the 10% training fraction. Run each with `landcover train --config ...`; repeat with seeds 43/44 and unique output names for the final analysis (12 ablation runs).

The main configs use a 1e-3 learning rate for scratch models and the pretrained classifier head, and 1e-4 for the pretrained backbone. Both groups share the warm-up/cosine multiplier. Curves record the rates actually used, plus unaugmented selected-training-subset F1/loss and the train/validation F1 gap. Generic `Config` defaults retain constant-rate behavior for older RGB smoke configs. No early stopping is used.

All pretrained layers are fine-tuned. For 13-band ResNet-18, RGB kernels are copied to B4/B3/B2, remaining bands receive the RGB-kernel mean, then all input kernels are multiplied by 3/13. Scratch 13-band kernels use Kaiming initialization. The head is replaced with ten classes. Native 64×64 inputs and train-derived normalization deliberately differ from the ImageNet default 224×224 transform. See [methodological limits](docs/experiment-plan.md).

`best.pt` is selected by validation macro-F1; ties keep the earlier checkpoint. Test data is never read by training. After freezing model selection and the analysis protocol:

```sh
uv run landcover evaluate --checkpoint outputs/baseline/best.pt --split test --allow-test
```

Evaluation uses the saved model and normalization, checks manifest/split hashes, and needs no pretrained-weight download. `--manifest` and `--splits` allow relocation of an unchanged dataset tree. Checkpoints reload for inference; interrupted training resume is not implemented.

## Outputs

| Location | Contents |
| --- | --- |
| `data/raw/` | Verified archives, extracted imagery, download provenance |
| `data/prepared-ms-v2/` | Corrected sample hashes, class/band schema, fixed splits and subsets |
| `outputs/<run>/` | Config, manifest/split snapshots, package/Git/device metadata, normalization, best/last checkpoints |
| `outputs/<run>/learning_curves.*` | CSV/PNG training loss, validation loss/F1/accuracy versus optimizer steps |
| `outputs/<run>/evaluation-{val,test}/` | Predictions CSV with IDs/labels/probabilities, metrics JSON, confusion matrix PNG |
| `outputs/grid/` | 48 YAML configs, index, explicit execution script |

All data, checkpoints, caches and output directories are ignored by Git. The reviewed report export in `docs/reports/turing/` is shared for the presentation; it contains HTML pages, summary tables and linked figures, without checkpoints or per-image prediction files. Keep full experiment artifacts in team storage. Never commit credentials.

## Nepal and collaboration

[The Nepal guide](docs/nepal.md) documents authentication, verified acquisition planning, cloud masking, grid alignment, paired patches, hand labels, transfer evaluation and exploratory changes. `configs/nepal.yaml` intentionally has no event or acquisition dates. No live Earth Engine export has been validated.

Root [AGENTS.md](AGENTS.md) and four `.agents/skills/*/SKILL.md` files define project workflows. [Integration notes](docs/integrations.md) record checked Codex support and authentication decisions. CI runs lint, regression tests and synthetic train/evaluate checks; it never downloads EuroSAT or runs the research grid.

## Sources

Dataset: [official EuroSAT repository](https://github.com/phelber/EuroSAT), [Zenodo record 7711810](https://zenodo.org/records/7711810), [Helber et al., EuroSAT paper](https://arxiv.org/abs/1709.00029). Cite EuroSAT and comply with its MIT dataset license and applicable Copernicus terms. Pretrained architecture/weights: [torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
