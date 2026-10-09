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
- [x] P2 — Deliverable: authenticated Earth Engine project and verified read/download access (`nepal1-511021`, 2026-10-08). Drive export access is untested. Owner: Unassigned.

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
- [x] P1 — Deliverable: 48 completed ResNet-18 runs with immutable artifacts and failure log. Dependencies: frozen pilot protocol and shared data. Owner: Unassigned.
- [x] P1 — Deliverable: compact baseline plus augmentation/dropout ablations over three seeds. Dependencies: supplied configs, final protocol and unique run paths. Owner: Unassigned.
- [ ] P2 — Deliverable: exact interrupted-training resume with RNG/sampler recovery if needed. Dependencies: demonstrated long-run operational need. Owner: Unassigned.

## Evaluation and experiments

- [x] P0 — Deliverable: guarded test command, fixed-label metrics, prediction probabilities and plots. Dependencies: frozen-checkpoint loader and metric tests. Owner: Unassigned.
- [x] P0 — Deliverable: generator for 48 unique shared-budget configs and four compact ablation configs. Dependencies: validated configuration schema. Owner: Unassigned.
- [ ] P0 — Deliverable: dated model-selection freeze and test-analysis protocol. Dependencies: validation-only pilots. Owner: Unassigned.
- [x] P0 — Deliverable: frozen 3,000-step study settings, 63-run inventory, sequential runner, complete-seed reports and explicit final-test command. See [final study](docs/final-study.md). Completed: 63 training/validation runs and 51 final test evaluations; reports are published under docs/reports/turing/. Owner: Doanh.
- [x] P1 — Deliverable: final test outputs for all predeclared conditions (51 evaluations; compact regularization ablations are validation-only). Dependencies: completed experiments and protocol freeze. Owner: Unassigned.
- [x] P1 — Deliverable: seed-level table, mean/SD, paired comparisons and sample-efficiency curves (published Turing reports). Dependencies: complete test results and actual subset sizes. Owner: Unassigned.
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

## Nepal 2026 implementation status (2026-10-08)

- [x] Offline study tooling: checkpoint/schema/normalization/provenance and validation-selection checks; geographic blind sampling (300/100 targets, seed 2026); independent double labels/adjudication/freeze; paired RGB/MS metrics; QA-gated primary-MS sensitivity maps; change-review panels and unknown exclusions. Synthetic regression checks only; see [Nepal protocol](docs/nepal.md).
- [x] Event date verified against ICIMOD; initial/fallback inclusive windows encoded with exclusive Earth Engine ends. This does not establish imagery feasibility.
- [ ] Obtain full-data pretrained seed-42 RGB and MS `best.pt` run folders from a teammate and verify transfer hashes. The account lphung@turing.wpi.edu is unavailable; do not attempt it.
- [x] Source Rasuwagadhi–Timure–Betrawati river centerline and select the study corridor with the user (2026-10-08), using a 2 km buffer. OSM-derived 45.58 km route and source provenance exist locally. Independent detailed geometry audit is not recorded.
- [x] Authenticate own registered Earth Engine project; acquire a real Rasuwagadhi–Timure pilot with provenance: 13 scenes/date, shared 10 m grid, 38 complete pairs and 76 blank label assignments. Google quotas remain; no custom EECU cap.
- [x] Screen the accepted full corridor: initial paired validity 57.57%, fallback 98.78% at 60 m. Select fallback windows for coverage before labels/predictions; submit two 13-band Drive exports.
- [x] Download/verify both full-corridor TIFFs, preserve raw hashes, mask outside-buffer pixels and prepare 341 complete pairs. Seed-2026 blank packet meets 300 pre / 100 post targets with 80 independent second-label assignments.
- [ ] Complete full-corridor human imagery/registration QA; regenerate patches/assignments if alignment changes. No labels have been completed.
- [ ] Inspect ten stable landmarks; correct residual displacement over one pixel and freeze QA for regenerated paired patches.
- [ ] Complete independent labels (>=70% supported dominant cover), double-label 20%, adjudicate and freeze before inference/comparison.
- [ ] Execute frozen transfer evaluation and primary 13-band candidate mapping; review up to 30 changed/30 unchanged patches; retain unsupported surfaces as unknown; publish exploratory results with coverage and limitations.

## Presentation scope decision (2026-10-08)

- [x] User chose to defer human labeling, preserve Option 1 artifacts and present an unvalidated demonstration first. Prepared actual imagery/coverage figures and presentation notes; no model map generated.
- [x] Separate `nepal-demo` tooling preserves frozen-model/patch provenance and unknown 255, outputs sensitivity 0.6/0.7/0.8, and discloses incomplete review. Reviewed research QA gates remain unchanged.
- [ ] Obtain genuine teammate checkpoints, verify them and run primary-MS inference to generate the unvalidated model demonstration. Synthetic smoke checkpoints must not substitute.
- [ ] Later: independent blind labels, adjudication/freeze and imagery/registration/change review before accuracy claims or reviewed research maps. Use labelers unexposed to predictions or a reserved blind set.
