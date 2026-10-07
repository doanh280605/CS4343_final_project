# Turing EuroSAT results

Completed October 7, 2026: 63 training/validation runs and 51 final test evaluations.
All conditions use seeds 42, 43 and 44. Main comparisons use 3,000 optimizer steps,
batch size 64 and validation every 100 steps. Test scores are for final reporting;
the compact-CNN regularization ablations use validation only.

## Open the reports

Download the whole repository using **Code > Download ZIP** on GitHub and extract
it, or update an existing clone with `git pull --ff-only`.

Double-click these files in the downloaded repository:

- `docs/reports/turing/report-test/index.html` — final test results.
- `docs/reports/turing/report-val/index.html` — validation and regularization results.

On a Mac, from the repository folder:

```sh
open docs/reports/turing/report-test/index.html
open docs/reports/turing/report-val/index.html
```

On Windows or Linux, open those files in a browser using the file manager. No Python,
GPU or installation is needed. Keep the `report-test`, `report-val` and `runs` folders
together so all charts, CSV links and individual-run figures work. GitHub's file view
shows HTML source; download the folders to view the rendered reports.

## Contents and verification

Both reports include mean macro-F1, mean accuracy, variation across the three training
seeds, data-efficiency charts, per-class results and CSV tables. The validation report
also includes the compact-CNN regularization chart. Individual-run links open learning
curves and confusion matrices. Error bars are sample standard deviations across seeds,
not confidence intervals for performance on new geographic regions.

These files were copied from `outputs/turing-study` on Turing via the local presentation
export; only CSV line endings were normalized for Git, with every field unchanged.
A local audit recomputed the reported metrics from all
340,200 validation and 275,400 test prediction rows. It checked the fixed sample IDs,
labels, probabilities, three-seed coverage, summary statistics, paired comparisons,
validation-selected checkpoint steps and report links. These are repeated predictions
on the same 5,400 validation or test images, not additional independent samples.

This reviewed report export contains no raw imagery, checkpoints, per-image prediction
files or credentials. Full experiment artifacts remain in the original Turing study;
this folder is not a complete backup for checkpoint/provenance verification. `run` and
`evaluation` columns in the CSVs retain their original artifact paths. Nepal transfer
and disaster-change evaluation have not been completed. Pretrained and scratch runs
use different learning rates, as recorded in the [experiment plan](../../experiment-plan.md).
