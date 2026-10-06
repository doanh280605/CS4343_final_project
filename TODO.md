# Team TODO

Priorities: P0 before main experiments; P1 core study; P2 optional/extension. Owners remain unassigned. Checked items are implemented and locally verified; unchecked experiments are not setup results.

The [October 5 completion audit](docs/project-completion-audit.md) maps remaining work to the owners in the [supplied report](docs/final-project-report-reference.md) and identifies protocol discrepancies to resolve before experiments. The historical checklist below retains its original assignment fields.

## Setup and integration

- [x] P0 — Deliverable: installable package, locked environment, CLI and Makefile. Dependencies: Python/uv. Owner: Unassigned.
- [x] P0 — Deliverable: root AGENTS.md and four supported `.agents/skills` workflows. Dependencies: inspected Codex 0.159.3 and official docs. Owner: Unassigned.
- [x] P0 — Deliverable: integration/authentication notes, environment example and artifact exclusions. Dependencies: existing Git/gh authentication review. Owner: Unassigned.
- [x] P0 — Deliverable: CI workflow exercising lint, regression tests and synthetic smoke. Dependencies: local checks. Owner: Unassigned.
- [ ] P0 — Deliverable: team installation/compute readiness record, including GPU environment if used. Dependencies: locked environment and chosen compute. Owner: Unassigned.
- [x] P0 — Deliverable: local Apple-GPU readiness check and one validation-only ResNet pilot (1,000 steps, 201.42 seconds). See [data readiness](docs/data-readiness.md); this does not establish readiness on teammates' machines. Owner: Doanh.
- [ ] P2 — Deliverable: authenticated Earth Engine project and verified export access. Dependencies: project registration and user OAuth. Owner: Unassigned.

## Data preparation

- [x] P0 — Deliverable: official-archive downloader with checksum/path checks and stable sample manifests. Dependencies: publisher archive metadata. Owner: Unassigned.
- [x] P0 — Deliverable: stratified 60/20/20 split and nested shared fractions, alignment/shape/hash audits. Dependencies: data module and regression checks. Owner: Unassigned.
- [x] P0 — Deliverable: small real RGB dataset smoke preparation and audit (200 samples). Dependencies: verified official RGB archive. Owner: Unassigned.
- [x] P0 — Deliverable: full 13-band dataset, per-file hash/shape audit and shared MS-derived RGB identity checks. See [data readiness](docs/data-readiness.md); the supplied extracted folder has no archive checksum record. Owner: Doanh.
- [ ] P0 — Deliverable: frozen full manifests/splits shared in team artifact storage. Dependencies: full dataset audit. Owner: Unassigned.
- [ ] P1 — Deliverable: data-quality and spatial-overlap assessment with representative visual checks. Dependencies: full dataset and available location metadata. Owner: Unassigned.

## Models and training

- [x] P0 — Deliverable: compact CNN, scratch/pretrained ResNet-18 and documented 13-band kernel expansion. Dependencies: PyTorch/torchvision. Owner: Unassigned.
- [x] P0 — Deliverable: train-only statistics/augmentation, AdamW, fixed step budgets and validation-selected checkpoints. Dependencies: persisted subsets. Owner: Unassigned.
- [x] P0 — Deliverable: cosine/warm-up schedule, pretrained backbone/head learning rates, unaugmented selected-training F1 and corrected 10% dropout/augmentation configs. Dependencies: report comparison and regression checks. Owner: Doanh.
- [x] P0 — Deliverable: synthetic training/reload/evaluation and small real RGB baseline smoke. Dependencies: local CPU environment. Owner: Unassigned.
- [ ] P0 — Deliverable: validation-only pilot selecting common training duration and optimization settings. Dependencies: full prepared data and compute budget. Owner: Unassigned.
  - First pretrained RGB 10% pilot completed; scratch/MS/full-data pilot coverage and a common-budget decision remain outstanding.
- [ ] P1 — Deliverable: 48 completed ResNet-18 runs with immutable artifacts and failure log. Dependencies: frozen pilot protocol and shared data. Owner: Unassigned.
- [ ] P1 — Deliverable: compact baseline plus augmentation/dropout ablations over three seeds. Dependencies: supplied configs, final protocol and unique run paths. Owner: Unassigned.
- [ ] P2 — Deliverable: exact interrupted-training resume with RNG/sampler recovery if needed. Dependencies: demonstrated long-run operational need. Owner: Unassigned.

## Evaluation and experiments

- [x] P0 — Deliverable: guarded test command, fixed-label metrics, prediction probabilities and plots. Dependencies: frozen-checkpoint loader and metric tests. Owner: Unassigned.
- [x] P0 — Deliverable: generator for 48 unique shared-budget configs and four compact ablation configs. Dependencies: validated configuration schema. Owner: Unassigned.
- [ ] P0 — Deliverable: dated model-selection freeze and test-analysis protocol. Dependencies: validation-only pilots. Owner: Unassigned.
- [x] P0 — Deliverable: frozen 3,000-step study settings, 63-run inventory, sequential runner, complete-seed reports and explicit final-test command. See [final study](docs/final-study.md). Model selection occurs after all validation runs, and final research execution remains outstanding. Owner: Doanh.
- [ ] P1 — Deliverable: final test outputs for all predeclared conditions. Dependencies: completed experiments and protocol freeze. Owner: Unassigned.
- [ ] P1 — Deliverable: seed-level table, mean/SD, paired comparisons and sample-efficiency curves. Dependencies: complete test results and actual subset sizes. Owner: Unassigned.
- [ ] P1 — Deliverable: per-class error analysis and spatial/domain-generalization limitations. Dependencies: predictions, confusion plots and data-quality review. Owner: Unassigned.

## Nepal case study

- [x] P2 — Deliverable: acquisition validator, cloud-mask/export code, alignment/patch/inference/change CLI and guide. Dependencies: maintained Earth Engine/rasterio APIs and offline tests. Owner: Unassigned.
- [ ] P2 — Deliverable: selected event with authoritative date source, AOI and justified windows. Dependencies: imagery coverage check. Owner: Unassigned.
- [ ] P2 — Deliverable: downloaded cloud-masked composites and scene/task provenance. Dependencies: verified event and authenticated Earth Engine. Owner: Unassigned.
- [ ] P2 — Deliverable: co-registration residual/visual QA report and valid paired patches. Dependencies: composites and suitable local CRS. Owner: Unassigned.
- [ ] P2 — Deliverable: independent labels with class rules, labelers, disagreement review and exclusions. Dependencies: paired patches and taxonomy assessment. Owner: Unassigned.
- [ ] P2 — Deliverable: frozen-model Nepal transfer metrics and exploratory change map with coverage/sensitivity analysis. Dependencies: labels, QA and chosen EuroSAT model. Owner: Unassigned.

## Analysis, report, and presentation

- [x] P0 — Deliverable: concise experiment plan, assumptions, limitations and source links. Dependencies: implemented protocol. Owner: Unassigned.
- [ ] P1 — Deliverable: reproducibility appendix with data/commit/config hashes, environment and run inventory. Dependencies: completed experiment artifacts. Owner: Unassigned.
- [ ] P1 — Deliverable: final report answering research questions with uncertainty and failure cases. Dependencies: aggregate results; Nepal section only if completed. Owner: Unassigned.
- [ ] P1 — Deliverable: presentation with methods, learning curves, comparisons and defensible conclusions. Dependencies: reviewed report and figures. Owner: Unassigned.
- [ ] P1 — Deliverable: team peer review of leakage, citations, claims and artifact accessibility. Dependencies: draft report/presentation. Owner: Unassigned.
