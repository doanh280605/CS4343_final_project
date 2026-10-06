# Local ResNet pilot commands

Run from the repository root on the current Mac. These are validation-only pilots on the fixed 10% training subset: 1,620 training patches, seed 42, batch 64, 1,000 optimizer steps, validation every 100 steps, cosine decay with 50 warm-up steps, and explicit Apple MPS GPU use. Settings match the completed pretrained RGB pilot. These runs do not evaluate the test set or launch the full experiment grid.

Set up the shell once:

```sh
export PYTHONPATH="$PWD/src"
export TORCH_HOME="$PWD/.cache/torch"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export PYTHONUNBUFFERED=1
```

Run these sequentially to avoid competing for GPU memory:

```sh
uv run --no-sync landcover train --config configs/pilots/resnet18-scratch-rgb.yaml
uv run --no-sync landcover train --config configs/pilots/resnet18-pretrained-ms.yaml
uv run --no-sync landcover train --config configs/pilots/resnet18-scratch-ms.yaml
```

The existing environment is already installed; `--no-sync` preserves its optional packages. Progress prints every 100 steps after validation/training-subset metrics finish. Initial dataset statistics and model loading can take time before the first line. Watch `step`, `val_macro_f1`, `train_macro_f1` and `macro_f1_gap`. Each run saves curves, resolved configuration, normalization, provenance, and best/last checkpoints under its distinct `outputs/pilots/` directory.

After the scratch RGB run:

```sh
open outputs/pilots/resnet18-scratch-rgb-f0.1-s42/learning_curves.png
uv run --no-sync landcover evaluate --checkpoint outputs/pilots/resnet18-scratch-rgb-f0.1-s42/best.pt --split val
open outputs/pilots/resnet18-scratch-rgb-f0.1-s42/evaluation-val/confusion_matrix.png
```

Use the corresponding output directory for either MS run. Validation reload is optional; training already records validation metrics. Do not use `--split test` during pilot selection. Existing output folders are deliberately never overwritten: change `output` in the relevant YAML before repeating a run. Interrupted training cannot resume exactly; a restart also needs a fresh output path. Do not interpret pilot scores as completed three-seed research comparisons.

## Longer scratch pilots

All three original local pilots completed 1,000 steps. Scratch RGB reached best validation macro-F1 0.84908 and scratch MS 0.93379, both at the final step. To check a longer training budget, the following configs were copied from those saved run configs, changing only `steps` to 3,000 and the output directory:

```sh
uv run --no-sync landcover train --config configs/pilots/resnet18-scratch-rgb-3000.yaml
uv run --no-sync landcover train --config configs/pilots/resnet18-scratch-ms-3000.yaml
```

Use the same shell setup above and run one at a time. These are fresh training runs, not checkpoint resumes. Their output directories are `outputs/pilots/resnet18-scratch-rgb-f0.1-s42-3000/` and `outputs/pilots/resnet18-scratch-ms-f0.1-s42-3000/`. Original results are retained.

Compare each run's best validation macro-F1 with its corresponding 1,000-step pilot, and inspect the final several validation points, training F1 and validation loss. Improvement only on training images does not show better generalization. Small single-seed differences need repeated-seed confirmation; do not use test evaluation to decide the budget.

The cosine learning-rate schedule now decays over 3,000 steps, so the first 1,000 updates do not reproduce the earlier trajectory. This compares a longer budget with its associated schedule, not just continuing the previous model. If the longer budget is adopted, use a shared frozen budget across all main comparison conditions; do not report longer-trained scratch models against shorter-trained pretrained models as a controlled final comparison. Full-data pilot coverage remains outstanding.

Both longer scratch pilots are now complete: best validation F1 was 0.85988 for RGB and 0.94487 for MS. Preserve these results. To check whether the candidate 3,000-step budget is suitable when using all 16,200 training images, the next config is:

```sh
uv run --no-sync landcover train --config configs/pilots/resnet18-scratch-rgb-full-3000.yaml
```

This uses the same seed, model, batch size and schedule as the longer RGB pilot, but changes the fraction to 100% and writes a fresh directory: `outputs/pilots/resnet18-scratch-rgb-f1-s42-3000/`. Normalization and training-set evaluation now cover the full training split, so it can take longer than the 10% pilot. The assistant has prepared this config without starting training.

Update: the user completed this full-data pilot. Best validation macro-F1 is 0.94304 at step 2,700, versus 0.85988 for the 10% run. All 3,000 steps completed; do not rerun into the same directory. See `docs/data-readiness.md` for the verified comparison and remaining protocol-freeze work.
