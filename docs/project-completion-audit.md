# Project completion audit

Follow-up: [data readiness](data-readiness.md) records work completed after this initial audit, including full dataset preparation, corrected band order, training fixes and updated checks. The discrepancy table below is the original audit snapshot, not the current implementation status.

Audited October 5, 2026 against the [supplied report](final-project-report-reference.md), repository source/configuration, [verification record](verification.md), and local `data/` and `outputs/` directories. The implementation foundation exists, but the local artifacts contain only smoke runs and generated grid configs. No full research results or Nepal evaluation artifacts were found. Work held only on teammates' machines or external storage is outside this audit.

The supplied report still describes planned work. A final report needs measured results, answers to the research questions, limitations, conclusions, and actual contributions. The original reference is preserved separately rather than silently corrected.

## Completion priorities and owners

Owners below follow the supplied team responsibilities. Checkboxes describe remaining work, not completed milestones.

| Priority | Deliverable | Owner | Completion evidence |
| --- | --- | --- | --- |
| P0 | Reconcile report and executable protocol | Evan, Daksh, Doanh | Decisions below resolved; configs and methods agree; dated protocol freeze |
| P0 | Full multispectral EuroSAT data | Julian, Daksh | Verified archive, 27,000 samples, hashes and data-quality audit; RGB extracted from the same TIFFs |
| P0 | Shared immutable splits and subsets | Julian | 16,200/5,400/5,400 split; nested training sizes 16,200/1,620/810/162; shared artifact location |
| P0 | GPU readiness and validation-only pilot | Evan, Doanh | Hardware/environment record, measured runtime, adequate common step budget, working checkpoint evaluation |
| P1 | Main RQ1/RQ2 experiments | Evan, Daksh | 48 completed runs, three seeds per condition, best/last checkpoints, curves, provenance, failures recorded |
| P1 | Compact baseline and regularization ablation | Evan, Doanh | Baseline repeated across seeds; four ablation cells at 10% × three seeds = 12 ablation runs |
| P1 | Frozen final test evaluation | Doanh, all reviewers | Explicit protocol/model-selection freeze before evaluating every predeclared comparison checkpoint |
| P1 | Aggregate results and interpretation | Doanh, Evan, Daksh | Seed scores, mean/sample SD, paired differences, data-efficiency curves, per-class plots, confusion matrices, failure examples |
| P2 | Nepal acquisition and labels | Jessica, Julian | Verified event/AOI/windows, authenticated export, downloaded imagery, alignment QA, 300–500 independent pre-event labels or an explicitly documented shortfall |
| P2 | Nepal transfer and exploratory changes | Jessica, Doanh | Same frozen model for both dates, transfer metrics, transition map, coverage, sensitivity analysis, independent reference comparison |
| P1 | Final report, presentation, reproducibility package | All | Actual results and conclusions, contribution record, accessible artifacts, reproduction commands and reviewed citations |

P2 follows the report's optional-Nepal fallback. If Nepal cannot be completed, explicitly mark RQ3 unanswered and revise the title/abstract/scope to describe the EuroSAT study. Do not imply the Nepal objective was achieved.

## Report and implementation discrepancies

| Report commitment | Current implementation | Required resolution before research runs |
| --- | --- | --- |
| Cosine learning-rate schedule with warm-up | `engine.py` uses constant-rate AdamW; no scheduler | Implement and record schedule/warm-up settings, or disclose a revised protocol |
| Pretrained backbone LR 1e-4; head LR 1e-3 | One optimizer parameter group at `learning_rate`, default 1e-3 | Add explicit backbone/head groups or revise the claim; document that different optimization settings also affect the comparison |
| Compact dropout p=0.5 | Default and ablation nonzero dropout are 0.3 | Select one value before experiments and synchronize configs/report |
| Dropout before the classifier described generally | Dropout exists in compact CNN only; ResNet head is a linear layer | State exactly which model uses dropout |
| Ablation at 10% training fraction | All four supplied ablation configs use `fraction: 1.0` | Set the planned fraction to 0.1, create all three seed variants and unique output paths |
| Equal optimizer steps plus early stopping | Fixed steps, best-validation checkpoint selection, no early stopping | Keep the fixed-step protocol and correct the report; best-checkpoint selection is not early stopping |
| Train/validation performance gap | Curves save augmented minibatch training loss and validation loss/F1/accuracy; no deterministic training F1 | Add evaluation on the exact selected training subset without augmentation for a comparable F1 gap; existing evaluation CLI supports only val/test |
| TorchGeo multispectral pipeline | Custom verified downloader and rasterio-based loader; TorchGeo is not a dependency | Describe the actual stack; no need to add TorchGeo solely to match prose |
| Matching RGB and multispectral versions | Main configs intentionally derive B4/B3/B2 from the same MS TIFFs | Explicitly describe this control; JPEG runs remain separate smoke/secondary conditions |
| Split indices in repository or possible TorchGeo split | Fixed split seed 2026 and nested subsets are implemented; generated manifests are Git-ignored | Remove the unresolved split alternative; share hashed manifests in artifact storage |
| One pre-event scene and one post-event scene | Earth Engine exporter builds cloud-masked median composites over configured windows | Choose and describe the actual acquisition protocol, with scene IDs and retained coverage |
| One final test evaluation | RQ1/RQ2 require test results across predeclared conditions and seeds | Interpret this as one frozen evaluation phase; testing only the winning model cannot answer the stated comparisons |
| Every run takes minutes | No local GPU benchmark recorded | Measure the pilot; budget training, validation, data loading and artifact storage before committing to the schedule |

Source locations: `src/landcover/{config,engine,models,experiments,nepal}.py`, `configs/ablations/`, `pyproject.toml`, and `docs/experiment-plan.md`. This audit changes documentation only; it does not implement these protocol decisions.

## Nepal evidence and interpretation

The event date is supported: World Weather Attribution identifies a rock–ice avalanche and resulting debris flood on August 26, 2026, with the initiating rock-wall failure involving glacier ice. Cite this more precise mechanism and date casualty estimates instead of presenting them as timeless totals. [World Weather Attribution analysis, September 17, 2026](https://www.worldweatherattribution.org/rapid-warming-in-the-himalaya-exacerbates-geohazard-cascades-beyond-adaptation-limits/).

The Reuters survivor report is also available as a syndicated article and supports the broad casualty claim. Direct retrieval of the supplied Reuters/AP URLs failed during this audit; that does not establish that the citations are false. Review accessible publisher or syndicated versions and verify all citation metadata before submission. [Reuters report republished by AsiaOne](https://www.asiaone.com/asia/nepal-town-flood-survivors-find-little-trace-former-lives).

Remaining Nepal requirements:

- Record an AOI and justified acquisition windows; the checked-in Nepal config remains incomplete. Verify Earth Engine project/authentication and export access.
- Acquire harmonized L1C with all 13 bands, including B10; inspect cloud, shadow and nodata coverage. Visually check residual co-registration after grid alignment.
- Define label rules, mixed-patch handling, independent annotation, disagreement adjudication and excluded out-of-taxonomy surfaces. Preserve class counts and geography. Never use these labels for tuning.
- Use the same frozen checkpoint and EuroSAT training statistics for both periods. Predeclare handling of absent classes: fixed ten-class macro-F1 and present-class scores have different meanings. A raw EuroSAT–Nepal score difference also reflects class composition and annotation differences, not only geographic shift.
- Each 64×64 patch at 10 m spans 640×640 m (40.96 hectares). Its classification is one label for that footprint, not a 10 m damage segmentation. Small corridor changes may be diluted by surrounding land cover.
- EuroSAT has no debris, bare-rock or glacier class. The current classifier cannot directly substantiate the claim that vegetation or settlements were replaced by debris. Keep such surfaces unknown/excluded and restrict interpretation accordingly.
- Low maximum softmax probability is a heuristic, not proof of an unseen class; high confidence can also be wrong. Freeze thresholds independently of Nepal evaluation labels and report coverage/sensitivity.
- Locate an actual reference impact map with date, extent and provenance; availability is not yet established. Define comparison at the patch scale and acknowledge any mismatch between a damage map and land-cover transitions. Without a suitable reference, report qualitative exploration and leave the quantitative alignment question unanswered.

## Final submission contents

- Results tables answering RQ1 and RQ2, with all seeds, paired comparisons and mean/sample SD. Three seeds quantify training variability, not independent test populations.
- A completed Nepal section answering RQ3, or an explicit incomplete-case-study limitation.
- Learning curves, sample-efficiency plot, per-class RGB/MS differences, ablation table, representative confusion matrices and failure cases. Explain whether hypotheses were supported; do not replace observations with expected outcomes.
- Actual methods, limitations and conclusions; artifact links; data/split/config/Git hashes; compute environment, runtime, failed runs and deviations; reproduction instructions and actual team contributions.
- Remove template directions in Team Responsibilities, Timeline and References. Correct “Daskh” to “Daksh” and “All team member.” Confirm whether “OpenToWork” is the intended team name. Check bibliography metadata against publisher records.
- Verify the final course rubric separately: it was not supplied, so this checklist establishes completion against the report rather than full rubric compliance.

## Deadline triage

The supplied deadline is October 8, with presentation October 9. As of this audit, the local repository is behind the report's experiment timeline. First inventory any completed teammate runs. Then resolve protocol discrepancies, prepare the full MS dataset and benchmark one validation-only pilot; Jessica can establish Nepal imagery/label feasibility concurrently. Use measured runtime to determine whether all promised runs fit. If they do not, document a reduced study transparently and confirm its acceptability against course requirements; never substitute smoke scores or omit unsuccessful seeds without disclosure.
