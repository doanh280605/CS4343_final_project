# Repository instructions

## Communication and scope

Respond briefly and directly. No icons or emojis. Preserve existing work. Never present synthetic checks or smoke metrics as research results. Do not run the complete experiment grid during setup or routine verification.

## Architecture

- `src/landcover/data.py`: stable IDs, manifest hashes, splits/subsets, loaders and statistics.
- `download.py`: verified official EuroSAT archives. `models.py`: compact CNN and ResNet-18.
- `engine.py`, `metrics.py`: training, frozen-checkpoint evaluation and artifacts.
- `nepal.py`: optional Earth Engine exports, raster alignment, paired patches and exploratory changes.
- `configs/`: complete run settings and compact-CNN ablations. `tests/`: offline regression checks.
- `docs/experiment-plan.md`, `docs/nepal.md`, `TODO.md`: research protocol and team deliverables.
- `.agents/skills/`: four focused project skills. Read the relevant skill before that workflow.

## Commands

Run from the repository root with Python 3.12 and uv:

```sh
uv sync --locked
make check
uv run landcover smoke --output outputs/synthetic-smoke-new
uv run landcover --help
```

Use a fresh output directory for every run; existing runs are intentionally never overwritten.
`make grid` generates 48 configs without running them. Downloading datasets/pretrained weights needs network access. Tests need neither network nor credentials.

## Coding conventions

Use the `landcover` package and explicit configuration rather than notebooks for shared experiment logic. Keep paths relative to the repository working directory; manifest paths are relative to the manifest location. Use Ruff formatting, 100-column Python, descriptive names, and small functions. Add regression checks for leakage, alignment, metrics, checkpoint semantics or geospatial correctness when changing them. Do not add tests that merely match prose or configuration wording.

## Reproducibility and leakage

- Preserve `uv.lock`; use `uv sync --locked`. Record any GPU environment departure separately.
- Fixed split seed 2026; training seeds 42, 43, 44. Never resplit for a model or seed.
- Persist sample IDs as `Class/filename_stem`; reject duplicate IDs or unequal RGB/MS ID sets.
- Preserve one stratified 60/20/20 split and nested 100/10/5/1% training subsets across all comparisons.
- Compute channel statistics only on the selected training subset. Apply random geometric augmentation only to training data. Evaluation uses frozen statistics and no augmentation.
- Main RGB/MS comparisons use B4/B3/B2 from the same 13-band TIFFs. JPEG RGB is a separate smoke/secondary condition; never mix it into the main grid without recording that confound.
- Select `best.pt` exclusively by validation macro-F1. Test evaluation is a separate explicit command after protocol/model selection is frozen. Do not tune on test or Nepal evaluation labels.
- Equal optimizer steps, batch size and evaluation intervals across fractions; use sampling with replacement. Record actual subset size and effective passes, not just nominal percentage.
- Audit immutable data before experiments; do not mutate raw imagery or prepared manifests in place.

## Tracking and verification

Each run saves config, manifest/split snapshots, normalization, package/hardware/Git provenance, learning curves, `best.pt` and `last.pt`. Evaluation saves sample IDs, labels, predictions, class probabilities, metrics and confusion plots. Record deviations and failed runs; do not cherry-pick seeds. Never imply a checkpoint supports exact interrupted-run resume: these checkpoints support reload/evaluation, but no resume CLI exists yet.

Before committing: `make check`, targeted smoke checks when needed, `git diff --check`, and inspect staged paths for secrets/data/artifacts. CI mirrors lint, tests and the synthetic CLI smoke. Distinguish local from remote CI and synthetic from real-data validation. Push only task-scoped, verified work without force-pushing.

## Nepal constraints

Verify an event date with an authoritative source before filling acquisition dates. L1C provides 13 bands; L2A lacks B10 and cannot silently replace it. Check co-registration visually after grid reprojection. Cloud/no-data pixels must remain unknown. Use a frozen EuroSAT model for transfer evaluation and the same model for both time windows. Class changes are possible land-cover changes, never verified disaster damage.

## Credentials and outputs

Never commit tokens, `.env`, cloud credentials, raw datasets, checkpoints, caches or generated outputs. `.env.example` is documentation and is not automatically loaded. Git and `gh` reuse the user's existing authentication. Earth Engine requires separate user authentication and a registered Cloud project. No dataset or Earth Engine MCP is needed.
