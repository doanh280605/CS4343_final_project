---
name: landcover-train
description: Configure or run reproducible EuroSAT compact-CNN and ResNet-18 experiments with shared step budgets and training subsets.
---

Inputs: audited prepared manifest/splits and a YAML config based on `configs/baseline.yaml`. Read `docs/experiment-plan.md` before changing comparisons.

```sh
uv run landcover train --config configs/baseline.yaml
uv run landcover grid --output outputs/grid
```

Grid generation creates 48 configs and an explicit shell script; it does not start runs. Do not launch the grid during setup/verification. For compact ablations use `configs/ablations/`, repeating seeds with unique output paths when authorized to run the full analysis.

Keep steps, batch size, validation interval and persisted subsets shared across comparisons. Training seeds 42/43/44 do not alter the split. Normalization uses the selected subset only; random flips/rotations are training-only. Main RGB uses MS-derived bands, not the JPEG smoke configuration.

Outputs: resolved YAML, provenance, manifest/split snapshots, normalization, `best.pt`, `last.pt`, CSV/PNG learning curves. Best means maximum validation macro-F1, earliest on ties. Test is never used for training/selection. Check actual steps/examples drawn and sample counts in provenance.

Run `uv run pytest tests/test_training.py -q` after model/engine changes; `uv run landcover smoke --output outputs/smoke-new` checks both input modalities. Real pretrained weights require a network/cache check separately from mocked initialization tests.

Failure modes: missing TIFF input, existing output directory, missing pretrained weights offline, nonfinite loss, unsupported deterministic MPS operations, inappropriate batch size/memory. Select CPU for portable smoke checks; reduce batch size only as an explicitly tracked protocol change across comparisons. Checkpoint reload works; exact interrupted-run resume is not implemented. Never report smoke losses as benchmark evidence.
