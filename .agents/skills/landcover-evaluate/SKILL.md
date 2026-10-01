---
name: landcover-evaluate
description: Evaluate frozen land-cover checkpoints, audit predictions and metrics, or summarize controlled experiment results without test leakage.
---

Inputs: `best.pt` plus its unchanged prepared dataset. Default to validation:

```sh
uv run landcover evaluate --checkpoint outputs/baseline/best.pt --split val
```

After model selection/protocol freeze, final test reporting explicitly uses `--split test --allow-test`. The command intentionally refuses accidental test evaluation. Do not tune from test scores. For moved data, supply `--manifest` and `--splits` with unchanged JSON content and preserved relative data paths.

Outputs: `evaluation-val/` or `evaluation-test/` with per-sample IDs, labels, predictions, all ten probabilities, metrics JSON, confusion PNG. Verify one row per expected ID, probabilities sum to one, confusion totals equal sample count, class mapping is the canonical ten classes, and metrics use fixed-label macro-F1 with zero-division handling. Frozen normalization comes from the checkpoint, never from evaluation imagery.

Run `uv run pytest tests/test_training.py -q` after metric/evaluation changes. For research summaries report all seeds, mean/sample SD, paired condition differences and learning curves. Distinguish training-seed variation from test-population uncertainty. Preserve failed-run records; never fabricate missing experiments.

Failure modes: dataset/split hash mismatch, wrong checkpoint architecture, stale output directory, missing classes or mislabeled domain. Do not bypass a mismatch by editing the checkpoint. Choose a new output path. Synthetic and small-real-data checks validate plumbing, not scientific performance.
