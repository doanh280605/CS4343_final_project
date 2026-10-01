# Setup verification

Verified locally on 2026-10-01 with Python 3.12.12, uv 0.9.26, PyTorch 2.6.0, torchvision 0.21.0 and Apple silicon CPU. These checks establish a runnable foundation; they are not research results.

## Checks performed

- Locked environment resolution/installation, including the optional Earth Engine API dependency.
- `make check`: Ruff lint/format checks and **17 passing regression tests**.
- Production-sized split fixture: exact 16,200/5,400/5,400 partition of 27,000 IDs; nested subsets 16,200/1,620/810/162; disjointness, deterministic ordering and leakage rejection.
- Paired synthetic RGB/13-band images: matching IDs, band selection, 3/13×64×64 shapes, finite values, content hashes and training-only statistics.
- Synthetic CLI smoke: 100 paired synthetic samples, two optimizer steps each for RGB and MS compact CNN, checkpoint reload, validation predictions, metrics and plots.
- Training regression: exact step/optimizer budget, best-checkpoint criterion, deterministic CPU replication, protected test evaluation and output-overwrite refusal. Tests touch test splits only on synthetic fixtures.
- Metric correctness against hand-computable examples; prediction probabilities/IDs and confusion totals checked.
- All compact/scratch ResNet input shapes; mocked pretrained band-expansion mapping tested offline.
- Official ResNet-18 weights downloaded and checked through a two-step **real RGB** train/reload/evaluate run, plus a one-step **synthetic MS** pretrained train/reload/evaluate run.
- Official EuroSAT RGB archive downloaded from Zenodo record 7711810; publisher MD5 `f46e308c4d50d4bf32fedad2d3d62f3b` verified. Extracted archive supplies a 200-image fixture (20/class), split into 120 train, 40 validation and 40 untouched test images. Compact CNN ran three training steps and saved/reloaded its best checkpoint before validation evaluation. This fixture uses the first sorted filenames per class and is not statistically representative.
- Nepal offline fixtures: incompatible date rejection, YAML date handling, acquisition plan without network, raster alignment, paired patch nodata exclusion, hand-label inference/evaluation and georeferenced transition map with unknown coverage preserved.
- Grid generation produced 48 unique configs with shared budget and MS-derived RGB; no grid training was started.
- Four project skills passed the installed skill-creator validator. Installed Codex CLI binary and official docs confirmed `.agents/skills` discovery support.

Representative local artifacts (ignored by Git): `outputs/synthetic-smoke/`, `outputs/real-smoke/`, `outputs/pretrained-real-smoke/`, `outputs/pretrained-ms-synthetic/`, and `outputs/grid/`. Real-data confusion matrix rendering was inspected. Unit-test artifacts use temporary directories. Setup runs occurred before the initial project commit; their provenance correctly records an unavailable initial commit and a dirty working tree. Future research runs should start from a recorded commit.

## Limits and remaining work

The full 2.07 GB MS archive was not downloaded, so **real RGB/MS cross-archive alignment is not yet verified**. MS data loading and pretrained 13-band adaptation were checked with synthetic imagery only. Full dataset audit, model-selection pilots, 48-run grid, ablation studies and final test results remain TODOs.

No GPU/MPS run was performed. CI mirrors local checks on Linux; its live status is available in GitHub Actions after push. The CI workflow never needs dataset credentials or pretrained downloads.

Earth Engine authentication, Cloud project selection and live exports remain unconfigured/unverified. No Nepal event, imagery or real labels were acquired. Offline plan/raster tests do not prove Earth Engine access, cloud quality or real co-registration accuracy. Exact training resume, automated subpixel registration and uncertainty calibration are future work.

Local environment issues resolved: uv/network/keyring operations required approved execution outside the managed macOS sandbox; first-use font discovery emitted cache-directory warnings before successfully rendering figures. Cache environment settings are documented in README. No tokens or credentials were written to tracked files.
