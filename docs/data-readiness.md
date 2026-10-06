# EuroSAT data and training readiness

October 6 update: the [final study](final-study.md) now freezes 3,000 shared optimizer steps and provides commands for 63 training/validation jobs, automatic graphs/tables, and a separate explicit test phase. The automation passed Ruff checks and 29 offline tests, including synthetic orchestration checks for interruptions, seed coverage, paired statistics and the test guard. The real study inventory is prepared; no final-grid training or final test evaluation was launched during setup. Earlier pilot-stage statements below remain historical context.

Updated October 5, 2026. The full local multispectral dataset is prepared and audited. The training/report code discrepancies are addressed; the full experiment grid, final test evaluation and Nepal study remain incomplete.

## Dataset

Raw imagery stays unchanged in `EuroSAT_MS/`. Git ignores that folder, downloaded EuroSAT ZIPs, `data/`, `outputs/` and checkpoints. None of these data/artifact paths are tracked.

| Check | Result |
| --- | --- |
| Samples | 27,000 across all ten classes |
| Integrity | Every TIFF rehashed/read; 13 bands, 64×64 pixels, finite values |
| Duplicate file hashes | 0 |
| Split seed | 2026 |
| Train / validation / test | 16,200 / 5,400 / 5,400 |
| Nested 100 / 10 / 5 / 1% training subsets | 16,200 / 1,620 / 810 / 162 |
| Current preparation | `data/prepared-ms-v2/` |

The original ZIP was not supplied in the repository, so its publisher archive checksum was not verified. Persisted per-file SHA-256 hashes establish the audited local snapshot. One training-only RGB preview per class was inspected; this does not establish full visual quality or geographic independence. RGB comes from the same TIFFs, not a separate JPEG archive; the manifest's `paired: false` means no separate RGB archive was supplied.

The corrected preparation preserves all initial sample memberships and file hashes. Only band metadata/version changed. Never use the retained earlier `data/prepared/` for new runs.

## Band order correction

EuroSAT stores B1–B8, B9, B10, B11, B12, then B8A. The previous schema placed B8A ninth, which would misalign Nepal channels with the trained model. Both training metadata and Nepal export selection now follow the [TorchGeo EuroSAT storage order](https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/eurosat.py). RGB B4/B3/B2 indices are unchanged.

New checkpoints record band/class schemas. Legacy MS checkpoints with missing or incompatible band metadata are rejected. Earlier smoke artifacts, including `outputs/readiness-2026-10-05/pretrained-ms/`, are retained as obsolete pipeline checks, not research evidence.

## Training preparation

- Main configs: AdamW, scratch/head LR 1e-3, pretrained-backbone LR 1e-4, cosine scheduling with 50 warm-up steps in the provisional 1,000-step budget.
- Compact baseline dropout 0.5; ablations use 10% data, augmentation on/off and dropout 0/0.5.
- Unaugmented selected-training-subset F1/loss and the train/validation F1 gap are recorded alongside augmented training loss and actual learning rates.
- Fixed budgets remain in force; there is no early stopping. Checkpoints support reload/evaluation, not exact interrupted-training resume.
- Fresh local configs: 48 in `outputs/readiness-2026-10-05/grid-v2/` and 12 in `outputs/readiness-2026-10-05/ablations-v2/`. Generation did not execute either grid. The older `outputs/grid/` is obsolete.
- Main configs use `device: auto`, which selects CUDA or CPU. Local Apple-GPU pilots explicitly use `device: mps`; choose one recorded research environment before the main runs.

## Local verification

Python 3.12.12 and PyTorch 2.6.0. CUDA is unavailable; the Apple MPS GPU is available outside the execution sandbox. Deterministic ResNet forward/backward passed on MPS. A three-step pretrained 13-band run on the corrected dataset completed on MPS and reloaded successfully for validation. All 5,400 validation prediction IDs/probability sums and confusion totals were checked; normalization used only the selected 162 training patches. These are pipeline checks, not benchmark results. No real test evaluation was run.

Ruff lint/format checks, 26 offline regression tests, and a fresh synthetic RGB/MS train/reload/validation smoke pass. Regression coverage includes scheduler/rate groups, training-metric leakage/RNG isolation, physical spectral channel mapping and rejection of incompatible MS checkpoint schemas. These are local checks; remote CI was not run in this session.

macOS repeatedly marked the editable-install `.pth` file hidden, causing `ModuleNotFoundError` despite correct installation. Set `export PYTHONPATH="$PWD/src"` from the repository root before local commands. Use a repository-local Matplotlib/Torch cache as described in README. The lockfile is unchanged.

## First validation pilot

Completed `outputs/pilot-2026-10-05-resnet18-pretrained-rgb-f0.1-s42-mps/`: pretrained ResNet-18, MS-derived RGB, 10% training data (1,620 patches), seed 42, MPS, batch 64, 1,000 steps, validation every 100 steps. Training including normalization, periodic evaluation and checkpoint/curve writing took 201.42 seconds (3.36 minutes); final standalone validation is outside that timing.

The best checkpoint was step 800: validation macro-F1 0.94708 and accuracy 0.94907. Unaugmented training macro-F1 was 1.00000, a train–validation gap of 0.05292. The run drew 64,000 training examples with replacement, equivalent to 39.51 passes over the selected subset. Validation performance largely flattened late in the run; this does not establish a sufficient common budget for scratch, MS or full-data conditions.

This is one validation-only pilot, not a final test result or a three-seed research comparison. The run records a dirty working tree and includes an exact `source-snapshot/` verified against hashes in `pilot-timing.json`. Do not reuse its scores as the completed RQ1/RQ2 grid. Retain all subsequent pilot decisions and failures. No test checkpoint evaluation has been performed on the real dataset.

## Artifact identities and next work

### Completed scratch duration comparison

Both 3,000-step scratch pilots completed. Saved checkpoints and learning curves agree, and all four scratch runs use the same dataset/split digests, 10% subset, seed 42, batch size and evaluation interval. Only total duration/output changed in their configs; cosine decay stretches with the new duration. No test evaluation was found in these run directories.

| Scratch model | Best validation F1 at 1,000-step budget | Best validation F1 at 3,000-step budget | Difference | Best step in longer run |
| --- | --- | --- | --- | --- |
| RGB | 0.84908 | 0.85988 | +0.01080 | 2,800 |
| Multispectral | 0.93379 | 0.94487 | +0.01108 | 3,000 |

Longer training with the longer schedule improved best validation F1 modestly in both single-seed pilots. Both longer runs reached training F1 1.0; extra practice did not eliminate the unseen-image performance gap. RGB's late validation F1 flattened around 0.858–0.860, while MS improved only slightly across the final several evaluations. Validation loss was higher than in the corresponding shorter run at its best-F1 checkpoint, so the F1 gains do not imply an improvement in every metric. No seed SD or statistical significance is established by these pilots.

Do not keep extending the 10% pilots solely because a best checkpoint happens at the end. A 3,000-step shared budget is a candidate, not yet a frozen protocol. Final comparisons must use the same frozen budget for scratch/pretrained, RGB/MS and all fractions.

### Completed full-data scratch RGB pilot

The user completed `outputs/pilots/resnet18-scratch-rgb-f1-s42-3000/`. All 3,000 steps and 30 validation checkpoints are recorded. Dataset/split digests match the earlier pilots, and normalization used the full 16,200-image training split. The selected checkpoint is step 2,700: validation macro-F1 0.94304, validation accuracy 0.94519, training macro-F1 0.95671, and train–validation F1 gap 0.01367. The final checkpoint's validation macro-F1 is 0.94129; `best.pt` correctly preserves the earlier, better checkpoint.

At the same 3,000-step budget, the 10% scratch RGB pilot achieved validation macro-F1 0.85988 and a gap of 0.14012. The full-data pilot therefore improved best validation macro-F1 by 0.08316 and showed a much smaller training/validation gap. This is a seed-42 validation comparison, not a three-seed or final-test finding. No test evaluation was found in the run directory. The next step is to record the shared-budget/protocol decision before generating and executing the final experiment configs; the original generated 1,000-step grids are not a frozen final protocol.

- Manifest digest: `d6c7b877102ba9f94127b6a2afe8a165fc923d2ac787519fe8ed5121d3513cf8`
- Split digest: `de7461a6e6cf373bedf7c7634430ea6fefce2cfeb8f8a65ab568791b1a1ec8f5`
- Audit: `outputs/readiness-2026-10-05/data-audit-v2.json`
- Corrected GPU check: `outputs/readiness-2026-10-05/pretrained-ms-v2-mps/`
- Shared-manifest bundle: `outputs/readiness-2026-10-05/prepared-manifests-v2.zip` (no raw imagery). Extract into the repository root and keep `EuroSAT_MS/` at the root so relative file paths remain valid. The bundle has not been uploaded to team storage.

Next: inspect validation-only pilots to select a common step budget; freeze settings; share immutable manifests and artifact locations with the team; run the declared comparisons; then perform frozen test analysis. Nepal still needs acquisition, alignment QA and independent labels. Update the final report to describe actual implementation choices and measured findings rather than planned outcomes.
