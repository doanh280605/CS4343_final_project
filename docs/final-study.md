# Frozen EuroSAT study

Protocol decision: October 6, 2026. Use **3,000 optimizer steps for every final run**, batch size 64, validation every 100 steps, 50 warm-up steps and cosine decay. The scratch RGB/MS duration pilots improved modestly over 1,000 steps, and full-data scratch RGB reached best validation F1 0.94304 at step 2,700. This supports a practical shared budget, not a claim that every condition reaches its optimal performance. No test scores informed this decision.

`configs/final-study.yaml` is the authoritative base. Earlier grids and pilot configs remain historical artifacts. All research runs start fresh; pilot results are not silently reused as final-grid seeds.

## Training and validation

For Colab GPU, upload `notebooks/landcover_colab.ipynb` at
<https://colab.research.google.com/> and select **Runtime > Change runtime type > GPU**.
Run its cells in order; upload `outputs/colab-setup/landcover-colab.zip` when requested.
Rebuild that small bundle with `.venv/bin/python scripts/build_colab_bundle.py` if
preparing a new study from updated source. Keep the same bundle throughout a frozen study.

The notebook caches the verified publisher archive at
`MyDrive/CS4343/datasets/EuroSAT_MS.zip`. New runtimes copy this ZIP to local storage
and extract it there; reruns in the same runtime reuse the local download/extraction.
An already-downloaded local ZIP is cached without another publisher download. Copies
are checksum-verified before publishing the destination; interrupted copies retry.
It reuses the bundled manifests/splits and audits image hashes.
Results save directly to `MyDrive/CS4343/outputs/cuda-study`; allow 18–23 GB of free
Drive space including the ZIP, with additional space for interrupted attempts. Drive checkpoint writes
can slow execution. Completed runs persist across runtime resets and are verified/skipped;
unfinished jobs restart in a fresh attempt. Rerun setup after a reset. Do not run two
sessions against the same output directory.

This is a separate CUDA study: the five completed Mac jobs and interrupted sixth job
remain in `outputs/final-study`. They are not merged into CUDA seed summaries. Only
the device/environment changes; all 63 conditions, budgets and fixed data memberships
remain the same. The bundle records source hashes and the original Git revision;
each Colab session records GPU, CUDA and package versions. Because there is no Git
checkout in the bundle, run-level Git provenance is unavailable there; use the bundled
source hashes and originating revision instead.

Do **not** run the normal Linux `uv sync` in this notebook: its locked PyTorch source
is CPU-only. The notebook uses isolated Python 3.12, the non-PyTorch dependencies
exported from `uv.lock`, and the official CUDA 12.4 wheels for PyTorch 2.6.0 and
torchvision 0.21.0. `uv.lock` is unchanged; this is an explicit GPU environment departure.
See [PyTorch installation](https://pytorch.org/get-started/previous-versions/#v260).
GPU allocation and runtime lifetime are not guaranteed; see the
[Colab FAQ](https://research.google.com/colaboratory/faq.html). No Colab runtime estimate
or completed CUDA result is claimed by local notebook syntax/bundle checks.

From the repository root on this Mac:

```sh
bash scripts/run_final_study.sh
```

The wrapper sets source/cache paths, uses the existing environment without resyncing optional packages, prepares `outputs/final-study/` if needed, and runs sequentially on the Apple MPS GPU. To prevent idle sleep, use `caffeinate -i bash scripts/run_final_study.sh`. Keep the computer powered. Expect hours; the earlier single-pilot runtime does not establish runtime for every condition. Allow roughly 15–20 GB for study artifacts, more if attempts fail.

The frozen inventory contains:

- 48 ResNet runs: pretrained/scratch × RGB/MS × 100/10/5/1% × seeds 42/43/44.
- 12 compact ablation runs: augmentation on/off × dropout 0/0.5 × three seeds, at 10% data.
- 3 compact full-data RGB baselines, one per seed.

Progress shows the job number and metrics every 100 steps. Best checkpoints use validation macro-F1 only; ties retain the earlier checkpoint. No early stopping or test evaluation occurs in this command.

When finished:

```sh
open outputs/final-study/report-val/index.html
```

The report contains data-efficiency, per-class and regularization figures, mean/sample-SD tables, paired differences, seed results, and links to each run's learning curves/confusion matrix. Validation summaries require all 63 runs; missing seeds cannot silently disappear.

## Final test phase

After training/validation completes:

```sh
bash scripts/run_final_test.sh
open outputs/final-study/report-test/index.html
```

This evaluates all 48 main checkpoints plus three compact baselines on the fixed test split. Regularization ablations remain validation-only. The command requires every validation job, freezes model selection, and invokes the explicit test guard. Do not tune settings or select seeds from test results.

For optional Nepal transfer, choose the main configuration with highest mean validation macro-F1 across three seeds; exact ties use lexical condition order. Seed 42 is the predeclared representative checkpoint. This avoids best-seed selection. The record is `outputs/final-study/selection.json`; it does not acquire Nepal imagery or complete transfer evaluation.

## Recovery and reproducibility

Running the training command again verifies and skips completed jobs. If interrupted midway through a job:

```sh
bash scripts/run_final_study.sh --retry-incomplete
```

This starts the incomplete job from scratch in a fresh numbered attempt directory. It does not resume optimizer/RNG state. Failed attempts and `events.jsonl` remain available. A completed training job whose evaluation was interrupted can finish evaluation without retraining. Interrupted evaluation outputs are preserved separately. Verified completed evaluations are reused.

`protocol.json` records the job inventory, data digests, source hashes, environment and selection rule. Source/manifest/split snapshots and run provenance are retained. Config/source/package/data-manifest changes stop the runner; do not edit the protocol/digest to bypass checks. A required change needs a documented new study or restoration of the frozen environment.

Regenerate reports without training:

```sh
PYTHONPATH=src uv run --no-sync landcover study-report --split val
PYTHONPATH=src uv run --no-sync landcover study-report --split test
```

Git-ignored results include `seed_results.csv`, `summary.csv`, `paired_differences.csv`, `per_class.csv` and `per_class_summary.csv`. Sample SD measures training-seed variation, not uncertainty over independent geographic test populations. Pretrained/scratch optimization recipes differ and must be disclosed when interpreting transfer-learning differences.

These commands cover EuroSAT experiments and analysis artifacts. Written conclusions, the team contribution record, presentation and optional Nepal study remain separate deliverables.
