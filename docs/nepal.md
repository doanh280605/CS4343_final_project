# Nepal 2026 frozen-transfer case study

Status on 2026-10-08: offline tooling is implemented; Nepal research is not complete.
Real pilot composites have been downloaded and 38 complete paired patches prepared.
There are no inspected landmarks, hand labels, verified local study checkpoints, Nepal
accuracy results or reviewed change maps. EuroSAT training/validation
has completed 63 runs and final testing has completed 51 evaluations; published artifacts are
under [reports/turing](reports/turing/). No Nepal fine-tuning is permitted.

## Event, corridor and feasibility

[ICIMOD's media advisory](https://www.icimod.org/press-release/major-flash-flood-sweeps-through-nepals-rasuwa-district-raising-fears-of-further-downstream-flooding/)
identifies the event as 26 August 2026 and describes the Bhote Koshi–Trishuli river system.
The source was checked on 2026-10-08. Start the feasibility assessment around
Rasuwagadhi–Timure, then extend along the connected river through Syabrubesi toward Betrawati.
The [Nepal Department of Roads Rasuwa district map](https://dor.gov.np/uploads/pages/Documents_1486708436.pdf)
provides geographic context for those settlements and rivers. It is a context source,
not an acquired, georeferenced centerline. Do not substitute a straight line between towns
for the winding river, or derive invented coordinates from settlement names.

`configs/nepal-2026.yaml` encodes the event and protocol but deliberately leaves the actual
geometry, bounding box, geometry source/reviewer and Cloud project empty. Acquisition refuses
this incomplete configuration. Obtain a sourced WGS84 river centerline from an authoritative
GIS dataset or a reviewed georeferenced digitization, retain its source/version/license,
inspect its continuity and endpoints, and record the geometry source and reviewer.
`aoi_geojson` accepts a LineString/MultiLineString or a reviewed Polygon/MultiPolygon;
Earth Engine buffers river lines by 2,000 m on each side. Polygon inputs must already
represent the reviewed buffer. Set the bounding box to enclose the full buffer.
Keep that geometry and its metadata under ignored `data/nepal/`. EPSG:32645 is the initial
projected CRS for this corridor; verify it against the sourced geometry before export.
The geometry is embedded and hashed in the validated plan.

Use harmonized Sentinel-2 **L1C TOA**, with the corrected EuroSAT v2 order:
`B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11 B12 B8A`. L2A omits B10 and is incompatible.
[Catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED).
Reflectance is scaled by 10,000; inference uses the frozen EuroSAT training normalization.
Main RGB uses B4/B3/B2 from these same TIFFs. No JPEG checkpoint preprocessing.

Initial inclusive windows are August 1–25 and August 27–September 20, 2026.
Earth Engine exclusive ends are `2026-08-26` and `2026-09-21`.
If coverage is inadequate, copy the config into `data/nepal/event.yaml` and explicitly use
July 15–August 25 and August 27–October 7 (exclusive ends August 26 and October 8).
Record the reason for changing windows before labeling; do not choose windows to improve scores.
Link [Cloud Score+](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED)
by scene identity, retain `cs_cdf >= 0.6`, and compute median composites on a projected 10 m grid.
Missing quality scores remain masked. Inspect valid coverage, clouds, shadows, snow, saturation
and seasonal differences before accepting either window. Scene existence alone is insufficient.

## Access and acquisition

Earth Engine is authenticated with registered project `nepal1-511021`; initialization
and real imagery reads/downloads succeeded on 2026-10-08. No custom EECU cap is retained.
Google-provided quotas remain in effect; quota snapshots are retained with local provenance.
The Community monthly allowance is 150 EECU-hours; wall-clock time is not EECU usage.
See [Google quota documentation](https://developers.google.com/earth-engine/guides/noncommercial_tiers).
Exact consumed EECU usage has not been measured by this pilot.
The Turing account `lphung@turing.wpi.edu` is unavailable; do not attempt to log in.
Obtain checkpoints from a teammate through agreed artifact storage.
Keep credentials, `.env`, geometry, rasters, checkpoints and generated outputs out of Git.
All commands run from the repository root; always choose fresh output directories.

```sh
uv sync --locked --extra nepal
# Fill ignored data/nepal/event.yaml with reviewed geometry and project first.
uv run landcover nepal-acquire --config data/nepal/event.yaml --output outputs/nepal-plan
# User authenticates their own Earth Engine account when available:
uv run earthengine authenticate
uv run landcover nepal-acquire --config data/nepal/event.yaml --output outputs/nepal-export --submit
uv run earthengine task list
```

The plan command validates offline. `--submit` starts Drive export tasks and records scene IDs,
acquisition timestamps, task IDs and submission status. Submission is neither completion nor
download. After checking completed tasks, download the ordered GeoTIFFs to `data/nepal/`.
Exports use -9999 nodata. Band ordering must be checked against the plan; raster band descriptions,
when present, must match the schema. Missing descriptions cannot prove correct ordering.

## Completed pilot (2026-10-08)

OpenStreetMap settlement nodes [Rasuwagadhi](https://www.openstreetmap.org/node/992955542),
[Timure](https://www.openstreetmap.org/node/2553894641) and
[Betrawati](https://www.openstreetmap.org/node/268864226), together with connected river ways,
provide a reproducible candidate centerline. Attribution: © OpenStreetMap contributors,
ODbL. Raw source responses, versions, query and geometry hashes are retained under ignored
`data/nepal/discovery-20261008/`. `scripts/nepal_source_corridor.py` replays the derivation
from those cached sources into a fresh output directory. The full candidate is 45.58 km;
the user accepted it as the study corridor on 2026-10-08 after reviewing the packet and
discussing its scope. Selection is recorded locally in `corridor-selection.json`; it does
not establish an independent detailed geometry audit or imagery/registration QA.

The Rasuwagadhi–Timure pilot uses a 2 km buffer (25.78 km²), initial windows,
13 scenes per period with matching Cloud Score+, and median masked composites.
The 60 m coverage screen found 100% pre and 85.15% post/paired validity; this is a coarse
screen, not a finding that all retained imagery is cloud-free. Both downloaded 13-band
TIFFs share EPSG:32645, a 10 m grid and dimensions 531 × 680. Physical registration
remains unverified. No Drive export tasks were submitted.

Local artifacts:

- `outputs/nepal-pilot-initial-20261008-v2/`: inventory, quota snapshot and previews.
- `data/nepal/pilot-initial-20261008/`: raw/tagged TIFFs, hashes and paired patches.
- `outputs/nepal-pilot-annotation-20261008/`: 76 blank assignments and RGB panels.

There are 38 eligible pairs: 155,648 valid, 172,032 invalid-window and 33,400 edge pixels
in the rectangular footprint. Exclusions include outside-buffer pixels; these counts are not
AOI-only coverage. Sampling targets remain short by 262 pre and 62 post patches.
Eight assignments per period require independent second labels. No labels have been supplied.
Full-corridor feasibility is the next acquisition step before freezing the final study sample.
Use `nepal-feasibility --full-corridor` for the accepted centerline: the coarse reduction
uses a maximum pixel count derived from its bounding grid. This does not alter Google quotas.

```sh
uv run landcover nepal-feasibility --geometry data/nepal/discovery-20261008/pilot.geojson --source-record data/nepal/discovery-20261008/source-record.json --project nepal1-511021 --coverage --output outputs/nepal-pilot-new
uv run landcover nepal-download-pilot --inventory outputs/nepal-pilot-new/inventory.json --geometry data/nepal/discovery-20261008/pilot.geojson --output data/nepal/pilot-new
```

Requests are sequential and the pilot checks request sizes against the documented
[download limits](https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl).
These request-size guards do not set a project spending or EECU cap.

## Full-corridor acquisition progress (2026-10-08)

The user accepted Rasuwagadhi–Timure–Betrawati as the study corridor. Its 2 km
buffer covers about 186.71 km². Initial coarse paired validity was 57.57%; the agreed
fallback windows improve it to 98.78% (pre 99.31%, post 99.30%). The wider windows
were selected for coverage before any labeling or model predictions; they also introduce
more temporal mixing and seasonal differences. Coverage is a 60 m screen, not completed
10 m eligibility or visual QA. Initial and fallback inventories/previews are retained in
`outputs/nepal-full-initial-20261008/` and `outputs/nepal-full-fallback-20261008/`.

Both 13-band, 10 m exports have now been downloaded and read successfully. Raw files
are preserved under ignored `data/nepal/full-fallback-raw-20261008/` with SHA256 hashes:
pre 248,776,489 bytes and post 252,535,771 bytes. Both have dimensions 2387 × 3768,
EPSG:32645, identical transforms and verified corrected band descriptions. Source task IDs
and scene/time records are in `outputs/nepal-full-fallback-export-20261008/`.

The exports include valid pixels outside the buffer because the composite was unmasked
before export. Before patch sampling, `nepal-mask-corridor` applies the retrieved Earth
Engine buffer polygon in EPSG:32645 using pixel centres (`all_touched=false`), preserves
nodata and writes immutable derivatives. Its source buffer has 10 m maximum geometry error.
Hashes, the polygon and coverage are retained in `masking.json`. No interpretation uses
pixels outside the study corridor.

```sh
uv run landcover nepal-mask-corridor --pre data/nepal/full-fallback-raw-20261008/nepal_pre.tif --post data/nepal/full-fallback-raw-20261008/nepal_post.tif --geometry data/nepal/discovery-20261008/corridor-buffer-32645.json --output data/nepal/full-masked-new
```

Completed local full-corridor artifacts:

- `data/nepal/full-fallback-20261008/`: masked pre/post TIFFs and masking provenance;
  1,867,096 inside-buffer pixels, 1,844,140 valid pairs at pixel level.
- `data/nepal/full-fallback-20261008/patches/`: 341 complete 64×64 paired patches.
  Rectangular footprint: 8,994,216 total pixels, 1,396,736 eligible patch pixels,
  7,393,280 excluded-window pixels and 204,200 edge pixels. Outside-buffer pixels
  are included in exclusions; these are not AOI-only invalid counts.
- `outputs/nepal-full-annotation-20261008/`: seed-2026 sample of 300 pre and 100 matching
  post assignments, no target shortfall; 60 pre and 20 post independent second labels.
  All labels are blank. Draft entry pages with CSV downloads are in
  `outputs/nepal-full-label-viewer-20261008-v2/label-primary.html` and
  `label-second.html`. Both scripts passed JavaScript syntax checks; the browser
  interface has not been interactively verified. Earlier generated entry pages failed
  syntax checks and are not the labeling interface. Raw CSV templates are authoritative.
- `outputs/nepal-full-qa-20261008/`: blank ten-landmark worksheet and false QA template
  tied to the full patch hash. Shared grids establish no physical registration result.

The initial pilot packet is superseded for study sampling. Human imagery/registration
review remains required; if alignment changes, regenerate patches and assignments before
freezing labels. No frozen labels, model predictions, accuracy or change results exist.

## Presentation demonstration: labeling deferred by user

On 2026-10-08 the user chose to postpone human labels for the presentation and return
later to quantitative evaluation. The full imagery, blind assignment, blank CSVs and
false QA templates are preserved. No QA pass or reference labels are fabricated.
`outputs/nepal-presentation-demo-20261008/` contains actual before/after RGB panels,
eligible patch coverage, presentation notes and hashed status. It contains no model
predictions, Nepal accuracy/F1 or damage results because genuine checkpoints are absent.

A separate `nepal-demo` command supports an explicitly **unvalidated model demonstration**
after genuine frozen models are verified and both inference runs are available:

```sh
uv run landcover nepal-models --rgb data/nepal/models/rgb/best.pt --ms data/nepal/models/ms/best.pt --output outputs/nepal-models-new
uv run landcover nepal-infer --checkpoint data/nepal/models/ms/best.pt --patches data/nepal/full-fallback-20261008/patches/patches.json --period pre --output outputs/nepal-demo-ms-pre
uv run landcover nepal-infer --checkpoint data/nepal/models/ms/best.pt --patches data/nepal/full-fallback-20261008/patches/patches.json --period post --output outputs/nepal-demo-ms-post
uv run landcover nepal-demo --pre outputs/nepal-demo-ms-pre/predictions.json --post outputs/nepal-demo-ms-post/predictions.json --patches data/nepal/full-fallback-20261008/patches/patches.json --models outputs/nepal-models-new/models.json --output outputs/nepal-model-demonstration
```

The command requires the same verified primary MS model, matching patch hashes/IDs and
valid probabilities. It preserves unknown 255 and records thresholds 0.6/0.7/0.8. Outputs
are named `unvalidated_model_changes` and carry unvalidated status in metadata, reports
and figure titles. Human imagery/registration/change review and labels remain explicitly
incomplete; no accuracy is calculated. It does not modify QA or relax the reviewed
`nepal-change` / `nepal-sensitivity` workflow. This demonstration supports no event attribution.

To return to Option 1, complete imagery/registration checks and independent human labels,
regenerate patches/assignments if alignment changes, adjudicate and freeze, then use the
original evaluation workflow. After showing model predictions, use labelers who have not
seen them or a separately reserved blind set to protect reference-label independence.

## Verify the frozen models

Request the original run folders for **full-data, pretrained ResNet-18, seed 42**, one RGB
and one 13-band MS: `best.pt`, `learning_curves.csv`, config, normalization, provenance and
manifest/split snapshots. Teammate should provide transfer hashes and artifact origin.
Compare received hashes with that transfer record independently.

```sh
uv run landcover nepal-models --rgb data/nepal/models/rgb/best.pt --ms data/nepal/models/ms/best.pt --output outputs/nepal-models
```

This verifies configuration, explicit band/class schemas, finite positive frozen normalization,
shared data/split/training IDs, training provenance, and the first maximum validation macro-F1
in the original learning curve. `models.json` retains hashes, schemas, selection evidence and
provenance. It establishes internal consistency, not independent authenticity of teammate files.
The 13-band model is the primary change model; both models are frozen for transfer evaluation.

## Align, inspect and pair

```sh
uv run landcover nepal-align --reference data/nepal/pre.tif --source data/nepal/post.tif --output data/nepal/post-aligned.tif
uv run landcover nepal-patches --pre data/nepal/pre.tif --post data/nepal/post-aligned.tif --output data/nepal/patches
```

Reprojection onto the reference grid does not correct or establish subpixel registration.
Inspect at least ten distinct stable landmarks distributed along the corridor. Record location,
independent reference source, measurement method, reviewer, date and residual displacement.
Resolve any displacement greater than one pixel before interpretation; if registration changes,
regenerate patches and repeat QA. Review masked coverage and imagery artifacts as well.
Create `data/nepal/qa.json` only after human inspection, with this schema (no fabricated entries):

```json
{
  "patch_manifest_sha256": "actual hash of patches.json",
  "imagery_review_passed": false,
  "reviewer": "",
  "reviewed_on": "",
  "landmarks": []
}
```

Each completed landmark has `id`, `row`, `col`, `residual_pixels`, `reference_source` and
`notes` describing evidence and method. Change CLI commands reject failed QA, fewer than ten
distinct landmarks, missing evidence, or residuals outside 0–1 pixels.
Paired 64×64 patches require matching CRS, transform, dimensions and thirteen bands,
a north-up 10 m grid in metres, and no masked/nonfinite pixels in either date.
`patches.json` records input hashes and excluded windows, valid/invalid/edge pixel counts
and pixel area. Incomplete edges stay unknown. Coverage describes the raster footprint,
including any excluded area outside the corridor; it is not a river-only area estimate.

## Blind labels and frozen transfer evaluation

```sh
uv run landcover nepal-annotate --patches data/nepal/patches/patches.json --output outputs/nepal-annotation
```

Seed 2026 selects one random patch per ordered geographic stratum, targeting 300 pre-event
patches and 100 matching post-event patches. Small valid pools report shortfalls.
Sampling uses geography and validity only, with no predictions. The packet includes RGB panels,
primary/second label templates, and a hashed assignment. Independently double-label 20% of each
period (round up); primary and second labelers must differ. Annotate before viewing predictions.

Use exact classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture,
PermanentCrop, Residential, River, SeaLake. Require at least 70% dominant supported cover.
Do not infer annual/permanent crops or pasture merely from greenness; use independent reference
context where needed. Road pixels alone do not establish Highway, and settlement pixels do not
establish Residential without the dominant-area rule. Never force debris, bare rock or glaciers
into these classes. Otherwise leave `label` blank and record `mixed`, `unsupported`, `uncertain`
or `cloud` in `exclusion_reason`. Every record needs `labeler` and `confidence`; supported labels
also need `dominant_fraction`. Fill notes with evidence. Copy all primary rows to a final CSV,
adjudicate double-label disagreements, and document every edited record in final notes.

```sh
uv run landcover nepal-freeze-labels --assignment outputs/nepal-annotation/assignment.json --primary outputs/nepal-annotation/labels-primary.csv --second outputs/nepal-annotation/labels-second.csv --final data/nepal/labels-final.csv --output outputs/nepal-labels-frozen
```

Only after freezing labels, run each verified checkpoint on each date (four runs), for example:

```sh
uv run landcover nepal-infer --checkpoint data/nepal/models/ms/best.pt --patches data/nepal/patches/patches.json --period pre --output outputs/nepal-ms-pre
uv run landcover nepal-compare --rgb outputs/nepal-rgb-pre/predictions.json --ms outputs/nepal-ms-pre/predictions.json --frozen outputs/nepal-labels-frozen --models outputs/nepal-models/models.json --output outputs/nepal-comparison-pre
```

Repeat with `post` and fresh paths. `nepal-compare` checks label integrity, matching period/IDs,
patch hashes and verified model hashes, then evaluates identical eligible patches. Report
accuracy, fixed ten-class macro-F1, reference-present-class macro-F1, class support and per-class
metrics, confusion plots, paired predictions/correctness counts, metric differences and exclusions
by reason. Absent reference classes remain in ten-class macro-F1. Geographic dependence and
nonrepresentative coverage limit conclusions; these sample metrics are not regionwide accuracy.
The legacy `nepal-infer --labels` is a basic evaluator, not this frozen-label study protocol.

## Possible change maps and human review

```sh
uv run landcover nepal-sensitivity --pre outputs/nepal-ms-pre/predictions.json --post outputs/nepal-ms-post/predictions.json --patches data/nepal/patches/patches.json --qa data/nepal/qa.json --models outputs/nepal-models/models.json --output outputs/nepal-candidates
```

This requires the same verified 13-band model for both dates, matching prediction provenance,
and successful human imagery QA. It maps threshold 0.7 with sensitivity at 0.6/0.8.
Confidence is uncalibrated; these thresholds are declared analysis settings.
The raster codes are 0=same prediction, 1=changed prediction and 255=unknown.
Coverage, transitions, hashes and sensitivity are recorded. Results are patch-level values
painted over footprints, not pixel-level classification or verified damage.

Seed 2026 selects up to 30 changed and 30 unchanged confident patches and writes paired panels,
`review.csv` and `review-assignment.json`. Fill reviewer, independent reference source, conclusion
(`possible_change`, `unchanged`, `unsupported`, `uncertain`, `cloud`) and evidence notes.
A disagreement with predictions is part of review, not a reason to tune the model.

```sh
uv run landcover nepal-review-exclusions --assignment outputs/nepal-candidates/review-assignment.json --review outputs/nepal-candidates/review.csv --output outputs/nepal-reviewed
uv run landcover nepal-change --pre outputs/nepal-ms-pre/predictions.json --post outputs/nepal-ms-post/predictions.json --patches data/nepal/patches/patches.json --qa data/nepal/qa.json --models outputs/nepal-models/models.json --exclusions outputs/nepal-reviewed/exclusions.json --threshold 0.7 --output outputs/nepal-reviewed-map
```

Reviewed unsupported, uncertain and cloud patches remain 255. Repeat final mapping at 0.6/0.8
with the same frozen exclusions and fresh directories. Review samples cannot establish the
validity of unsampled patches. Explanations must consider seasonal differences, cloud/shadow
residuals, registration error and classification mistakes. Event attribution requires additional
independent evidence. **Possible land-cover changes are never verified disaster damage.**

## Verification and remaining work

Local verification on 2026-10-08: `make check` passed (57 tests), CLI help passed,
a fresh synthetic training/reload smoke passed, and `git diff --check` passed.
The locked Nepal extra is installed locally with Python 3.12.11; no remote CI or
Earth Engine export was run. Offline regression tests use synthetic rasters/labels only. Required checks are `make check`,
`uv run landcover --help`, a fresh synthetic smoke when needed, and `git diff --check`.
Remaining human/access work: teammate checkpoint delivery and independent hash check; sourced
full candidate corridor review and extended scene/coverage feasibility/download; ten-landmark QA; blind labels/adjudication; frozen transfer evaluation;
change review and final map/report. No research completion follows from passing offline tests.
