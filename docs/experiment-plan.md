# Experiment plan

## Primary questions and fixed protocol

1. How much does ImageNet initialization help as labeled training data decreases?
2. What does adding all 13 Sentinel-2 bands change relative to RGB?
3. How well does a frozen EuroSAT model transfer to hand-labeled Nepal patches?

Use one stratified split with seed 2026: 60% train, 20% validation, 20% test. Persist IDs, class labels and per-file SHA-256 values. Split membership and nested subsets remain fixed across models and seeds. Official dataset class counts yield exactly 16,200/5,400/5,400 samples and training subsets of 16,200/1,620/810/162. For other datasets/smoke fixtures, per-class floor rounding with a minimum of one training example can change nominal fractions; report actual sizes.

Compare ResNet-18 pretrained/scratch, RGB/MS, fractions 1/0.1/0.05/0.01 and training seeds 42/43/44: 48 runs. Seeds vary initialization, sampling and augmentation, not split membership. Derive RGB from the same MS TIFFs; optional supplied JPEG experiments must be labeled separately because JPEG scaling/clipping/compression differs. Compact CNN is a scratch baseline; augmentation × dropout ablations have four cells, repeated over three seeds for final reporting.

## Model fitting and selection

- Use 64×64 inputs. ResNet keeps the standard 7×7 stride-2 stem and max pool; no untracked architecture adjustments across conditions.
- Train all layers with cross-entropy and AdamW; configured scratch/head learning rate 0.001, pretrained-backbone rate 0.0001, weight decay 0.0001, batch size 64. The final study freezes 50 warm-up steps followed by cosine decay over the remaining budget.
- These pretrained/scratch comparisons hold architecture fixed but use different optimization recipes; attribute results to the stated training conditions, not pretraining alone without qualification.
- The final study budget is 3,000 optimizer steps, validation every 100 steps, across every fraction. This October 6 decision follows the completed validation-only duration/full-data pilots; see [frozen study](final-study.md). The generic baseline config and historical grids retain 1,000-step pilot settings. Do not silently increase training duration only for favored conditions.
- Sample with replacement for equal full-batch budgets. This controls optimization effort rather than epochs, so small subsets receive many more repetitions. Save examples drawn and effective passes.
- Fit normalization using selected training examples only. Do not use unused training examples for low-data statistics. Validation/test use those frozen statistics without random transforms.
- Configured runs evaluate the exact selected training subset without augmentation at validation intervals, recording training F1/loss and the F1 generalization gap. Augmented minibatch training loss remains a separate curve. Compact ablations use 10% data, augmentation on/off and dropout 0/0.5 across three seeds; ResNet has no added classifier dropout.
- Fine-tune ImageNet weights in pretrained runs. Thirteen-band initialization copies RGB kernels to B4/B3/B2, fills other channels with the RGB mean, then multiplies all channels by 3/13. This is a deterministic heuristic, not a pretrained multispectral model. Scratch models get random initialization.
- Save best validation macro-F1 checkpoint and last checkpoint. No early stopping; all conditions receive the entire fixed budget. Earliest checkpoint wins ties.
- Freeze hyperparameters, chosen checkpoint procedure and all planned comparisons before test evaluation. `--allow-test` is an explicit workflow guard, not an access-control system. Keep test results out of subsequent tuning.

## Deliverables and analysis

Save per-sample ID, true label, prediction and all ten class probabilities. Report macro-F1 (primary), accuracy, per-class precision/recall/F1/support, confusion matrices, learning curves and compute metadata. Metrics use a fixed ten-class label set and zero for undefined class precision/recall/F1.

For each condition, report all three seed scores, mean and sample standard deviation. Pair pretrained-versus-scratch and RGB-versus-MS differences by training seed on the same test split. Three seeds measure training variability only; they are not three independent test populations. Any bootstrap confidence intervals must preserve paired sample identities and state whether resampling is by patch or spatial group. Plot performance versus actual training subset size and compare classes driving gains/losses. Record unsuccessful runs and protocol changes.

Nepal transfer evaluation uses a frozen chosen EuroSAT checkpoint and independent hand labels. Do not use the same labeled Nepal set for adaptation and reporting. If adaptation is later added, create geographically separated Nepal train/validation/test regions first. Pre/post class-transition maps remain exploratory.

## Assumptions and limitations

Random patch splits are the requested baseline; spatial autocorrelation may inflate apparent generalization. Stable IDs prevent duplicate-ID leakage but do not establish geographic independence or rule out overlapping scenes. Inspect locations and add a separate spatially grouped evaluation if metadata permits.

EuroSAT classes do not cover every Nepal surface. Mixed patches, snow/cloud remnants, terrain shadows, seasonality, sensor preprocessing and acquisition differences may change predictions. Softmax confidence is not calibrated uncertainty or an out-of-distribution detector. Cloud filtering can remove different land-cover mixtures in each period; report retained/excluded coverage.

All bands share the delivered 64×64 grid, but native Sentinel-2 resolutions differ; resampling does not create additional spatial detail. ResNet's native-patch protocol and training-only normalization differ from its ImageNet defaults. Kernel expansion and fixed-step sampling are explicit design choices requiring disclosure.

EuroSAT TIFF storage order is B1–B8, B9, B10, B11, B12, B8A, as documented in [TorchGeo's EuroSAT loader](https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/eurosat.py). The v2 preparation corrects the prior band-name order; shared sample memberships remain identical. Nepal exports select this same order. Legacy MS checkpoints without a verified matching band schema are rejected. See [data readiness](data-readiness.md) for retained obsolete artifacts and current manifest paths.

CPU repeatability is tested. Exact bitwise reproducibility across devices, operating systems and future library versions is not guaranteed. Checkpoints are for reload/evaluation; a fully reproducible resume feature remains future work. No full-grid performance, Nepal accuracy or disaster-damage finding is claimed by setup checks.
