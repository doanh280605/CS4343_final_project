---
name: nepal-imagery
description: Prepare verified Nepal imagery acquisition, aligned paired patches, hand-label evaluation and exploratory land-cover changes.
---

Read `docs/nepal.md` before this workflow. Inputs: a verified event/source/date, AOI, pre/post windows, Earth Engine project, and a frozen MS or MS-derived RGB EuroSAT checkpoint. Never invent event dates. Keep an unverified config incomplete.

```sh
uv sync --locked --extra nepal
uv run landcover nepal-acquire --config data/nepal/event.yaml --output outputs/nepal-plan
```

This saves a validated plan without exporting. `--submit` uses existing Earth Engine auth to submit Drive exports; report submission separately from completion/download. `earthengine authenticate` may require the user. Use harmonized L1C for 13 bands including B10; L2A is incompatible. Cloud Score+ masks unclear pixels; inspect retained coverage.

After downloading GeoTIFFs, run `nepal-align`, `nepal-patches`, `nepal-infer` for both periods, then `nepal-change`; exact commands and schemas are in the guide. Outputs include alignment raster, paired patch manifest, `labels.csv`, class probabilities/metrics, transition counts and georeferenced change map.

Checks: matching CRS/transform/dimensions/band order, visual residual co-registration, no cloud/nodata contamination, paired IDs, independent hand labels and the same frozen checkpoint for both periods. Grid reprojection does not prove subpixel registration. Keep unknowns as 255 in change maps and report excluded coverage. Run `uv run pytest tests/test_nepal.py -q` for local geospatial invariants.

Failure modes: pre-2015-06-27 Sentinel-2 windows, absent Cloud Score+ coverage, no valid paired patches, unverified event source, missing auth/project, wrong UTM zone, JPEG-trained preprocessing, out-of-taxonomy labels. Stop dependent interpretation until resolved; continue offline processing when possible. Predicted transitions indicate possible land-cover changes, never verified disaster damage.
