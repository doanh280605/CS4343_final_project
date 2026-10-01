"""Optional Nepal transfer: explicit provenance, aligned valid patches, exploratory changes."""

import csv
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

from landcover.data import BANDS, CLASSES, RGB_INDICES, dump_json, file_hash
from landcover.engine import load_checkpoint
from landcover.metrics import save_predictions


def validate_acquisition(config):
    config = dict(config)
    config["project"] = config.get("project") or os.environ.get("EE_PROJECT")
    required = (
        "event_name",
        "event_date",
        "event_source",
        "source_checked_on",
        "project",
        "bbox",
        "pre_start",
        "pre_end",
        "post_start",
        "post_end",
        "crs",
    )
    if any(not config.get(key) for key in required):
        raise ValueError("fill event/source verification, dates, project, AOI and CRS first")
    if not config["event_source"].startswith("https://"):
        raise ValueError("event_source must identify the verified authoritative HTTPS source")
    dates = {
        key: date.fromisoformat(str(config[key]))
        for key in (
            "event_date",
            "source_checked_on",
            "pre_start",
            "pre_end",
            "post_start",
            "post_end",
        )
    }
    if not (
        dates["pre_start"]
        < dates["pre_end"]
        <= dates["event_date"]
        <= dates["post_start"]
        < dates["post_end"]
    ):
        raise ValueError("pre/post windows must bracket the verified event date")
    if dates["pre_start"] < date(2015, 6, 27):
        raise ValueError(
            "Sentinel-2 TOA coverage starts 2015-06-27; no pre-event 2015 April imagery"
        )
    bbox = config["bbox"]
    if len(bbox) != 4 or not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
        raise ValueError("bbox must be WGS84 west,south,east,north")
    if config.get("scale", 10) != 10 or config.get("collection") != "COPERNICUS/S2_HARMONIZED":
        raise ValueError("13-band workflow requires harmonized L1C TOA at a common 10 m grid")
    if not 0 <= config.get("clear_threshold", 0.6) <= 1:
        raise ValueError("clear_threshold must be in [0,1]")
    for key, value in dates.items():
        config[key] = value.isoformat()
    return config


def acquire(config_path, output, submit=False):
    config = validate_acquisition(yaml.safe_load(Path(config_path).read_text()))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dump_json(output / "acquisition-plan.json", config)
    if not submit:
        return {"status": "validated-plan-only", "submitted": False}
    import ee

    ee.Initialize(project=config["project"] or os.environ.get("EE_PROJECT"))
    aoi = ee.Geometry.Rectangle(config["bbox"])
    tasks = []
    for period in ("pre", "post"):
        collection = (
            ee.ImageCollection(config["collection"])
            .filterBounds(aoi)
            .filterDate(config[f"{period}_start"], config[f"{period}_end"])
        )
        scores = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        collection = collection.linkCollection(scores, ["cs_cdf"])

        def mask(image):
            return image.updateMask(image.select("cs_cdf").gte(config.get("clear_threshold", 0.6)))

        collection = collection.map(mask)
        scenes = collection.aggregate_array("system:index").getInfo()
        if not scenes:
            raise ValueError(f"no scenes for {period}; revise verified window/AOI")
        composite = collection.select(BANDS).median().toFloat().unmask(-9999)
        task = ee.batch.Export.image.toDrive(
            image=composite,
            description=f"nepal_{period}",
            folder=config.get("drive_folder", "CS4343_Nepal"),
            region=aoi,
            scale=10,
            crs=config["crs"],
            maxPixels=1e9,
            fileFormat="GeoTIFF",
            formatOptions={"noData": -9999},
        )
        task.start()
        tasks.append({"period": period, "task_id": task.id, "scenes": scenes})
        dump_json(output / "export-tasks.json", tasks)
    return {"status": "exports-submitted-not-downloaded", "tasks": tasks}


def align(reference, source, output):
    """Resample source to the reference grid; does not estimate subpixel displacement."""
    if Path(output).exists():
        raise FileExistsError(output)
    with rasterio.open(reference) as ref, rasterio.open(source) as src:
        if ref.crs is None or src.crs is None or src.count != 13 or ref.count != 13:
            raise ValueError("both rasters need CRS and all 13 ordered bands")
        profile = ref.profile.copy()
        profile.update(dtype="float32", nodata=-9999, count=13)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **profile) as dst:
            for band in range(1, 14):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    dst_nodata=-9999,
                    resampling=Resampling.bilinear,
                )


def patches(pre, post, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    with rasterio.open(pre) as before, rasterio.open(post) as after:
        if (before.crs, before.transform, before.shape, before.count) != (
            after.crs,
            after.transform,
            after.shape,
            after.count,
        ) or before.count != 13:
            raise ValueError("align both 13-band rasters to exactly the same grid first")
        if before.crs is None:
            raise ValueError("CRS required")
        for row in range(0, before.height - 63, 64):
            for col in range(0, before.width - 63, 64):
                window = Window(col, row, 64, 64)
                pair = [src.read(window=window, masked=True) for src in (before, after)]
                if any(np.ma.getmaskarray(x).any() or not np.isfinite(x.data).all() for x in pair):
                    continue
                sid = f"r{row:06d}_c{col:06d}"
                record = {"sample_id": sid, "row": row, "col": col}
                for period, data in zip(("pre", "post"), pair, strict=True):
                    path = output / period / f"{sid}.tif"
                    path.parent.mkdir(exist_ok=True)
                    profile = before.profile.copy()
                    profile.update(height=64, width=64, transform=before.window_transform(window))
                    with rasterio.open(path, "w", **profile) as dst:
                        dst.write(data)
                    record[period] = f"{period}/{sid}.tif"
                rows.append(record)
        if not rows:
            raise ValueError("no fully valid paired 64x64 patches")
        dump_json(
            output / "patches.json",
            {
                "bands": BANDS,
                "crs": before.crs.to_string(),
                "transform": list(before.transform),
                "height": before.height,
                "width": before.width,
                "pre_sha256": file_hash(pre),
                "post_sha256": file_hash(post),
                "samples": rows,
            },
        )
    with (output / "labels.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "period", "label", "labeler", "confidence", "notes"])
        writer.writerows(
            [r["sample_id"], period, "", "", "", ""] for r in rows for period in ("pre", "post")
        )
    return len(rows)


def infer(checkpoint_path, patch_manifest, period, output, labels_path=None):
    if period not in {"pre", "post"}:
        raise ValueError("period must be pre or post")
    model, checkpoint, config = load_checkpoint(checkpoint_path)
    if config.input_type == "rgb" and config.rgb_source != "ms":
        raise ValueError("Nepal RGB inference requires an MS-derived RGB training checkpoint")
    torch.set_num_threads(config.threads)
    model.eval()
    manifest = json.loads(Path(patch_manifest).read_text())
    if manifest["bands"] != BANDS:
        raise ValueError("band order mismatch")
    mean = torch.tensor(checkpoint["normalization"]["mean"])[:, None, None]
    std = torch.tensor(checkpoint["normalization"]["std"])[:, None, None]
    ids, probs = [], []
    with torch.inference_mode():
        for row in manifest["samples"]:
            with rasterio.open(Path(patch_manifest).parent / row[period]) as src:
                x = src.read(out_dtype="float32") / 10000
            if x.shape != (13, 64, 64) or not np.isfinite(x).all():
                raise ValueError("invalid Nepal patch")
            if config.input_type == "rgb":
                x = x[RGB_INDICES]
            prediction = model(((torch.from_numpy(x) - mean) / std)[None]).softmax(1)[0]
            ids.append(row["sample_id"])
            probs.append(prediction.tolist())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dump_json(
        output / "predictions.json",
        {
            "period": period,
            "sample_ids": ids,
            "probabilities": probs,
            "checkpoint_sha256": file_hash(checkpoint_path),
            "patch_manifest_sha256": file_hash(patch_manifest),
        },
    )
    if labels_path:
        with Path(labels_path).open() as f:
            labels = [r for r in csv.DictReader(f) if r["period"] == period and r["label"]]
        lookup = {}
        for row in labels:
            if (
                row["sample_id"] in lookup
                or row["sample_id"] not in ids
                or row["label"] not in CLASSES
            ):
                raise ValueError("invalid, duplicate or unknown hand label")
            lookup[row["sample_id"]] = CLASSES.index(row["label"])
        selected = [i for i, sid in enumerate(ids) if sid in lookup]
        if not selected:
            raise ValueError("no hand labels for the requested period")
        save_predictions(
            output / "evaluation",
            [ids[i] for i in selected],
            [lookup[ids[i]] for i in selected],
            [probs[i] for i in selected],
            {"domain": "Nepal", "period": period, "hand_labeled_count": len(selected)},
        )
    return len(ids)


def change_map(pre_predictions, post_predictions, patch_manifest, output, threshold=0.7):
    if not 0 <= threshold <= 1:
        raise ValueError("confidence threshold must be in [0,1]")
    pre, post = [json.loads(Path(p).read_text()) for p in (pre_predictions, post_predictions)]
    manifest = json.loads(Path(patch_manifest).read_text())
    ids = [r["sample_id"] for r in manifest["samples"]]
    if (
        pre["period"] != "pre"
        or post["period"] != "post"
        or pre["sample_ids"] != ids
        or post["sample_ids"] != ids
    ):
        raise ValueError("prediction periods/IDs are not aligned")
    if pre["checkpoint_sha256"] != post["checkpoint_sha256"]:
        raise ValueError("use the same frozen model for both dates")
    if any(p["patch_manifest_sha256"] != file_hash(patch_manifest) for p in (pre, post)):
        raise ValueError("patch provenance mismatch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    array = np.full((manifest["height"], manifest["width"]), 255, dtype="uint8")
    transitions = np.zeros((10, 10), dtype=int)
    before, after = np.asarray(pre["probabilities"]), np.asarray(post["probabilities"])
    accepted = 0
    for i, row in enumerate(manifest["samples"]):
        if min(before[i].max(), after[i].max()) < threshold:
            continue
        a, b = before[i].argmax(), after[i].argmax()
        transitions[a, b] += 1
        accepted += 1
        array[row["row"] : row["row"] + 64, row["col"] : row["col"] + 64] = int(a != b)
    with rasterio.open(
        output / "possible_changes.tif",
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="uint8",
        nodata=255,
        crs=manifest["crs"],
        transform=rasterio.Affine(*manifest["transform"][:6]),
    ) as dst:
        dst.write(array, 1)
    dump_json(
        output / "transitions.json",
        {
            "classes": CLASSES,
            "counts": transitions.tolist(),
            "accepted_patches": accepted,
            "excluded_patches": len(ids) - accepted,
            "confidence_threshold": threshold,
            "interpretation": "Possible land-cover changes; NOT verified disaster damage.",
            "map_codes": {"0": "same prediction", "1": "changed prediction", "255": "unknown"},
        },
    )
