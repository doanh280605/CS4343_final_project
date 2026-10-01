# Optional Nepal case study

No Nepal event or acquisition dates have been selected. No authenticated Earth Engine acquisition or real Nepal accuracy has been verified. Local raster operations are covered by synthetic geospatial checks.

## 1. Select and verify the event

Copy `configs/nepal.yaml` to an ignored location such as `data/nepal/event.yaml`. Record the event name/date, an authoritative source URL, source-check date, WGS84 AOI bounding box, cloud project, local projected CRS, and pre/post windows. The source must establish the actual event date, not merely the article publication date. Check imagery coverage before committing to an event; use comparable seasons and explain window widths.

The [Sentinel-2 harmonized L1C catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED) starts on 2015-06-27. Consequently an April-2015 pre-event Sentinel-2 comparison cannot be produced by this workflow. Choosing Landsat instead would require a different band/model protocol. This guide does not encode a historical Nepal disaster date as a selected study.

L1C TOA provides all 13 spectral bands including B10, scaled by 10,000. [L2A surface reflectance](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) omits B10; it is not a drop-in 13-band replacement. Default EPSG:32645 must be checked against the selected AOI; western Nepal may require a different UTM zone. Keep the EuroSAT band order recorded in `data.py`.

## 2. Authenticate, plan and acquire

```sh
uv sync --locked --extra nepal
uv run earthengine authenticate
uv run landcover nepal-acquire --config data/nepal/event.yaml --output outputs/nepal-plan
```

The last command only validates/saves a plan. Once the source, dates, AOI and export size are reviewed, submit exports explicitly to your Drive:

```sh
uv run landcover nepal-acquire --config data/nepal/event.yaml --output outputs/nepal-export --submit
uv run earthengine task list
```

The code links [Cloud Score+](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED) by scene identity, masks pixels below `cs_cdf >= 0.6`, then creates a median composite. Save scene IDs, task IDs and configuration. Missing quality scores are masked; inspect coverage rather than assuming a nonempty collection yields useful clear imagery. The threshold is an initial setting requiring visual QA, not a validated Nepal cloud detector.

Download the completed Drive GeoTIFFs into `data/nepal/pre.tif` and `data/nepal/post.tif`. Submission is not completion or local download. Inspect clouds, shadows, snow, saturation, seasonal effects and retained AOI coverage. Exports use -9999 nodata and a common 10 m target resolution; verify band order after download. Exports larger than service limits may need a smaller AOI or tiling.

## 3. Align and make patches

```sh
uv run landcover nepal-align --reference data/nepal/pre.tif --source data/nepal/post.tif --output data/nepal/post-aligned.tif
uv run landcover nepal-patches --pre data/nepal/pre.tif --post data/nepal/post-aligned.tif --output data/nepal/patches
```

Alignment reprojects/resamples the post image onto the reference CRS, transform and dimensions using bilinear interpolation. This establishes grid correspondence; **it does not estimate or correct subpixel displacement**. Inspect stable roads/buildings visually and measure residual displacement before interpreting changes. If needed, perform a separate control-point registration and record the transform and residual error.

Patches are nonoverlapping 64×64 pairs with identical row/column IDs. Incomplete edges and any patch with masked/nonfinite pixels in either period are excluded. This conservative policy may substantially reduce coverage. `patches.json` saves band order, raster geometry, input hashes and paired paths. `labels.csv` is an editable labeling template. Patches represent areas, not dense per-pixel supervised classification.

## 4. Hand-label and evaluate transfer

Use exact EuroSAT class names from README/source. Fill `sample_id,period,label,labeler,confidence,notes` in `labels.csv`; leave out-of-taxonomy/ambiguous patches blank with a reason in notes. Annotate independently of model predictions, define a dominant-cover rule in advance, double-label a subset and adjudicate disagreements. Retain a frozen evaluation set and document geographic coverage, class counts and excluded surfaces. Do not tune the EuroSAT model using these labels.

Select an MS or MS-derived RGB checkpoint; the pipeline refuses JPEG-trained RGB checkpoints because their preprocessing differs. The checkpoint's EuroSAT training statistics remain fixed. For each period:

```sh
uv run landcover nepal-infer --checkpoint outputs/baseline/best.pt --patches data/nepal/patches/patches.json --period pre --labels data/nepal/patches/labels.csv --output outputs/nepal-pre
uv run landcover nepal-infer --checkpoint outputs/baseline/best.pt --patches data/nepal/patches/patches.json --period post --labels data/nepal/patches/labels.csv --output outputs/nepal-post
```

Without `--labels`, inference saves probabilities and IDs only. With labels, it evaluates only labeled patches for that period and writes the standard CSV, metrics and confusion plot. Fixed ten-class macro-F1 can be low when classes are absent; report class support and define any alternative present-class metric before analysis.

## 5. Exploratory changes

```sh
uv run landcover nepal-change --pre outputs/nepal-pre/predictions.json --post outputs/nepal-post/predictions.json --patches data/nepal/patches/patches.json --output outputs/nepal-changes --threshold 0.7
```

The same frozen checkpoint and patch manifest must be used for both dates. Output is a georeferenced GeoTIFF with 0=same prediction, 1=changed prediction, 255=unknown, plus class-transition counts. A patch is eligible only when both predictions exceed the chosen confidence threshold. Confidence is not calibrated; run a documented sensitivity analysis and report excluded coverage. The map assigns a patch-level result over its footprint and must not be presented as pixel-level damage detection.

**These are possible land-cover changes, not verified disaster damage.** Validate candidate changes against independent imagery/reference information and consider seasonal variation, cloud/shadow residuals, registration error and classification mistakes. Event attribution requires additional evidence outside this classifier.
