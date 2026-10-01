---
name: eurosat-data
description: Download, prepare or audit EuroSAT imagery, sample alignment and fixed experiment splits in this repository.
---

Use the repository-root CLI. Read `src/landcover/data.py` and `docs/experiment-plan.md` when changing split or band semantics.

Inputs: official archive(s) or class-organized RGB JPEG/13-band TIFF roots. Main comparisons require MS TIFFs; RGB comes from B4/B3/B2 of those same files.

```sh
uv run landcover download --kind ms
uv run landcover prepare --ms-root data/raw/ms --output data/prepared
uv run landcover audit
```

For both supplied modalities, add `--rgb-root data/raw/rgb`. Outputs: `manifest.json` with stable class/stem IDs, paths and content hashes; `splits.json` with seed-2026 stratified 60/20/20 IDs and nested 100/10/5/1% train subsets. Preserve them across every model/seed. Use a new output directory if preparation exists.

Checks: exact RGB/MS ID equality, all ten classes, no duplicate IDs or split overlap, expected 3/13×64×64 shapes and finite values. Full dataset: 27,000 samples and split sizes 16,200/5,400/5,400. Fit statistics only on the selected training subset during training. Run `uv run pytest tests/test_data.py -q` after changes.

Failure modes: wrong archive nesting (scan accepts nested roots), mismatched stems, missing classes, corrupt ZIP/checksum, partial extraction, non-13-band TIFFs. Resolve discrepancies rather than silently intersecting pairs or reshuffling splits. `--limit-per-class` is only for clearly labeled smoke fixtures, never the full study. Raw data and prepared artifacts stay out of Git.
