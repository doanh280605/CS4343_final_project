# CS4343: satellite land-cover classification

Runnable PyTorch foundation for EuroSAT sample-efficiency and RGB/13-band experiments, plus an optional Nepal transfer case study. Primary metric: macro-F1 across all ten classes. **No full research experiments have been run.** See [verification](docs/verification.md) for setup checks, [experiment plan](docs/experiment-plan.md) for the protocol, and [TODO](TODO.md) for team assignments.

## Install

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

## Prepare EuroSAT

The main experiment only requires the multispectral archive: RGB is extracted from the same TIFFs. Download is approximately 2.07 GB, plus extraction space.

```sh
uv run landcover download --kind ms
uv run landcover prepare --ms-root data/raw/ms --output data/prepared
uv run landcover audit
```

Optional official JPEG comparison and cross-archive identity verification (approximately 95 MB):

```sh
uv run landcover download --kind rgb
uv run landcover prepare --ms-root data/raw/ms --rgb-root data/raw/rgb --output data/prepared-paired
uv run landcover audit --manifest data/prepared-paired/manifest.json --splits data/prepared-paired/splits.json
```

To use the paired preparation, update both paths in your run configs. Preparation scans nested archive directories automatically. It checks exact class/stem identity between supplied modalities; never silently drops unmatched samples. It hashes each file and persists fixed stratified 60/20/20 splits plus nested 100/10/5/1% training subsets. For all 27,000 samples, split sizes are 16,200/5,400/5,400. Store/share the resulting manifests alongside experiment artifacts; they are ignored by Git because they contain dataset paths. Recreate with the same archive contents and split seed 2026, or copy the full data tree with relative paths intact.

Bands: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12. RGB uses B4/B3/B2. TIFF digital numbers are divided by 10,000; JPEG values by 255. Channel normalization is fit on each selected training subset, reused for validation/test and stored in the checkpoint. Flips and 90-degree rotations are training-only.

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

Default baseline: four convolution/BN/ReLU/max-pool blocks (32/64/128/256), global average pooling, dropout 0.3 and ten outputs. Training uses cross-entropy, AdamW and 1,000 optimizer steps. This step budget is a starting configuration, not an established sufficient training duration. Choose it using validation-only pilots before the main grid.

## Experiments and evaluation

```sh
uv run landcover grid --output outputs/grid
# Inspect configs, storage and compute budget before explicitly running:
bash outputs/grid/run.sh
```

Generation produces **48** ResNet-18 configs: pretrained/scratch × RGB/MS × 100/10/5/1% × seeds 42/43/44. A configurable step budget, batch size and evaluation schedule are shared across fractions. Full batches are sampled with replacement. The same persisted subsets are shared across architectures, initializations, modalities and training seeds. `--base-config configs/baseline.yaml` can provide different common settings to grid generation.

Four compact-CNN ablations are in `configs/ablations/`: augmentation on/off × dropout 0/0.3. Run each with `landcover train --config ...`; repeat with seeds 43/44 and unique output names for the final analysis.

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
| `data/prepared/` | Sample hashes, class/band schema, fixed splits and subsets |
| `outputs/<run>/` | Config, manifest/split snapshots, package/Git/device metadata, normalization, best/last checkpoints |
| `outputs/<run>/learning_curves.*` | CSV/PNG training loss, validation loss/F1/accuracy versus optimizer steps |
| `outputs/<run>/evaluation-{val,test}/` | Predictions CSV with IDs/labels/probabilities, metrics JSON, confusion matrix PNG |
| `outputs/grid/` | 48 YAML configs, index, explicit execution script |

All data, checkpoints, caches and output directories are ignored by Git. Share small reviewed result summaries in a report; keep large artifacts in team storage. Never commit credentials.

## Nepal and collaboration

[The Nepal guide](docs/nepal.md) documents authentication, verified acquisition planning, cloud masking, grid alignment, paired patches, hand labels, transfer evaluation and exploratory changes. `configs/nepal.yaml` intentionally has no event or acquisition dates. No live Earth Engine export has been validated.

Root [AGENTS.md](AGENTS.md) and four `.agents/skills/*/SKILL.md` files define project workflows. [Integration notes](docs/integrations.md) record checked Codex support and authentication decisions. CI runs lint, regression tests and synthetic train/evaluate checks; it never downloads EuroSAT or runs the research grid.

## Sources

Dataset: [official EuroSAT repository](https://github.com/phelber/EuroSAT), [Zenodo record 7711810](https://zenodo.org/records/7711810), [Helber et al., EuroSAT paper](https://arxiv.org/abs/1709.00029). Cite EuroSAT and comply with its MIT dataset license and applicable Copernicus terms. Pretrained architecture/weights: [torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
