"""Bounded pilot inventory and coarse coverage; never a registration or imagery QA pass."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import requests

from landcover.data import BANDS, dump_json, file_hash

MAX_PILOT_AREA_M2 = 50_000_000
MAX_SCENES = 60
COVERAGE_SCALE = 60
MAX_COVERAGE_PIXELS = 30_000
WINDOWS = {
    "initial_pre": ("2026-08-01", "2026-08-26"),
    "initial_post": ("2026-08-27", "2026-09-21"),
    "fallback_pre": ("2026-07-15", "2026-08-26"),
    "fallback_post": ("2026-08-27", "2026-10-08"),
}


def candidate_geometry(path):
    record = json.loads(Path(path).read_text())
    geometry = record["geometry"] if record.get("type") == "Feature" else record
    if geometry.get("type") != "LineString":
        raise ValueError("pilot needs a sourced WGS84 river LineString")
    coordinates = np.asarray(geometry["coordinates"], dtype=float)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or not 2 <= len(coordinates) <= 2000
        or not np.isfinite(coordinates).all()
        or (np.abs(coordinates[:, 0]) > 180).any()
        or (np.abs(coordinates[:, 1]) > 90).any()
    ):
        raise ValueError("invalid or oversized candidate river geometry")
    return geometry


def daily_limit(quotas):
    if quotas.get("nextPageToken"):
        raise ValueError("quota inspection is incomplete")
    limits = [
        int(bucket["effectiveLimit"])
        for metric in quotas.get("metrics", [])
        if metric["metric"] == "earthengine.googleapis.com/daily_eecu_usage_time"
        for limit in metric["consumerQuotaLimits"]
        for bucket in limit["quotaBuckets"]
        if not bucket.get("dimensions")
    ]
    if len(limits) != 1 or limits[0] <= 0:
        raise ValueError("daily EECU quota missing or exhausted; stop raster processing")
    return None if limits[0] >= 9223372036854775807 else limits[0]


def pilot_budget(area_m2, bounds, full_corridor=False):
    points = np.asarray(bounds, dtype=float)
    if not np.isfinite(area_m2) or area_m2 <= 0:
        raise ValueError("invalid corridor area")
    if not full_corridor and area_m2 > MAX_PILOT_AREA_M2:
        raise ValueError("pilot area must be <=50 square kilometres")
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("invalid projected pilot bounds")
    span = np.ptp(points, axis=0)
    pixels = int(np.prod(np.ceil(span / COVERAGE_SCALE) + 1))
    if not full_corridor and pixels > MAX_COVERAGE_PIXELS:
        raise ValueError("pilot bounding grid exceeds 30000 coarse pixels")
    return pixels


def scene_inventory(values):
    ids = values["ids"]
    if not ids or len(ids) > MAX_SCENES or len(ids) != len(set(ids)):
        raise ValueError("empty, duplicate or oversized scene inventory; stop and revise pilot")
    scores = set(values["score_ids"])
    if len(scores) > MAX_SCENES:
        raise ValueError("oversized Cloud Score+ inventory")
    scenes = [
        {
            "id": sid,
            "acquired_at_utc": datetime.fromtimestamp(timestamp / 1000, UTC).isoformat(),
            "tile": tile,
            "scene_cloud_percent": cloud,
            "cloud_score_available": sid in scores,
        }
        for sid, timestamp, tile, cloud in zip(
            ids, values["times"], values["tiles"], values["cloud_percent"], strict=True
        )
    ]
    if not any(s["cloud_score_available"] for s in scenes):
        raise ValueError("no matching Cloud Score+ scenes; stop raster processing")
    return scenes


def scout(
    geometry_path,
    source_record,
    project,
    output,
    coverage=False,
    fallback=False,
    full_corridor=False,
):
    """Run sequential cached-by-output metadata queries, optionally coarse raster coverage."""
    geometry = candidate_geometry(geometry_path)
    if not Path(source_record).is_file():
        raise ValueError("source provenance file is required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    import ee
    from google.auth.transport.requests import AuthorizedSession

    ee.Initialize(project=project)
    ee.data.setDeadline(120_000)
    ee.data.setMaxRetries(1)
    ee.data.setWorkloadTag("nepal-2026-pilot")
    session = AuthorizedSession(ee.data.get_persistent_credentials())
    session.headers["x-goog-user-project"] = project
    response = session.get(
        f"https://serviceusage.googleapis.com/v1beta1/projects/{project}/services/"
        "earthengine.googleapis.com/consumerQuotaMetrics",
        params={"view": "FULL", "pageSize": 100},
        timeout=30,
    )
    response.raise_for_status()
    quotas = response.json()
    cap = daily_limit(quotas)
    dump_json(output / "quota-snapshot.json", quotas)
    pending = [
        op["name"]
        for op in ee.data.listOperations()
        if op.get("metadata", {}).get("state") in {"PENDING", "READY", "RUNNING"}
    ]
    if pending:
        raise ValueError("other project operations are pending; avoid concurrent pilot work")
    projection = ee.Projection("EPSG:32645")
    aoi = ee.Geometry(geometry).buffer(2000, 10)
    projected_bounds = aoi.bounds(10, projection)
    shape = ee.Dictionary(
        {"area_m2": aoi.area(10, projection), "bounds": projected_bounds.coordinates()}
    ).getInfo()
    budget = pilot_budget(shape["area_m2"], shape["bounds"][0], full_corridor)
    collection_name = "COPERNICUS/S2_HARMONIZED"
    score_name = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
    records, composites = {}, {}
    windows = {
        k: v for k, v in WINDOWS.items() if k.startswith("fallback" if fallback else "initial")
    }
    for name, (start, end) in windows.items():
        collection = (
            ee.ImageCollection(collection_name)
            .filterBounds(aoi)
            .filterDate(start, end)
            .sort("system:time_start")
            .limit(MAX_SCENES + 1)
        )
        scores = ee.ImageCollection(score_name).filterBounds(aoi).filterDate(start, end)
        values = ee.Dictionary(
            {
                "ids": collection.aggregate_array("system:index"),
                "times": collection.aggregate_array("system:time_start"),
                "tiles": collection.aggregate_array("MGRS_TILE"),
                "cloud_percent": collection.aggregate_array("CLOUDY_PIXEL_PERCENTAGE"),
                "score_ids": scores.limit(MAX_SCENES + 1).aggregate_array("system:index"),
            }
        ).getInfo()
        records[name] = {
            "start_inclusive": start,
            "end_exclusive": end,
            "scenes": scene_inventory(values),
        }
        if coverage:
            linked = collection.linkCollection(scores, ["cs_cdf"])
            composites[name] = (
                linked.map(
                    lambda image: image.select(BANDS).updateMask(image.select("cs_cdf").gte(0.6))
                )
                .median()
                .clip(aoi)
            )
    report = {
        "project": project,
        "geometry_sha256": file_hash(geometry_path),
        "source_record_sha256": file_hash(source_record),
        "buffer_m": 2000,
        "google_daily_eecu_limit": cap,
        "pilot_area_m2": shape["area_m2"],
        "scope": "full_corridor" if full_corridor else "pilot",
        "coarse_bounding_pixels_upper_bound": budget,
        "windows": records,
        "collection": collection_name,
        "bands": BANDS,
        "status": "candidate AOI; human review and clear-coverage QA pending",
        "exports_submitted": False,
    }
    dump_json(output / "inventory.json", report)
    if not coverage:
        return report
    before, after = [composites[name] for name in windows]
    valid = [
        image.mask().reduce(ee.Reducer.min()).unmask(0).gt(0).rename(name)
        for image, name in zip((before, after), ("pre_valid", "post_valid"), strict=True)
    ]
    summaries = (
        ee.Image.cat(
            valid[0],
            valid[1],
            valid[0].And(valid[1]).rename("paired_valid"),
            ee.Image.constant(1).rename("total"),
        )
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            crs="EPSG:32645",
            scale=COVERAGE_SCALE,
            maxPixels=budget,
            bestEffort=False,
        )
        .getInfo()
    )
    total = summaries.get("total") or 0
    if total <= 0:
        raise ValueError("no coarse pilot pixels")
    report["coarse_coverage"] = {
        "scale_m": COVERAGE_SCALE,
        "counts": summaries,
        "fractions": {k: summaries[k] / total for k in ("pre_valid", "post_valid", "paired_valid")},
        "note": "60m screening; 10m eligibility, visual QA and registration remain unverified",
    }
    # Small previews use a common projected rectangle and fixed display range.
    region = projected_bounds.transform("EPSG:4326", 10)
    for period, image in zip(("pre", "post"), (before, after), strict=True):
        preview = image.select(["B4", "B3", "B2"]).visualize(min=0, max=3000)
        url = preview.getThumbURL({"region": region, "dimensions": 512, "format": "png"})
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        if len(response.content) > 2_000_000:
            raise ValueError("unexpectedly large pilot preview")
        (output / f"{period}-preview.png").write_bytes(response.content)
    dump_json(output / "inventory.json", report)
    return report


def download_grid(bounds):
    """A shared snapped 10m grid that fits Google's synchronous download limits."""
    points = np.asarray(bounds, dtype=float)
    left, bottom = np.floor(points.min(axis=0) / 10) * 10
    right, top = np.ceil(points.max(axis=0) / 10) * 10
    width, height = int((right - left) / 10), int((top - bottom) / 10)
    # Google documents a 32MB request limit and 10000 maximum grid dimension.
    if (
        min(width, height) <= 0
        or max(width, height) > 10000
        or width * height * 13 * 4 > 32_000_000
    ):
        raise ValueError(
            "pilot exceeds Google's synchronous download limit; use a reviewed batch plan"
        )
    return {"width": width, "height": height, "transform": [10, 0, left, 0, -10, top]}


def download_pilot(inventory_path, geometry_path, output):
    """Download the exact inventoried scenes on one grid; preserve raw and nodata-tagged files."""
    import ee
    import rasterio

    report = json.loads(Path(inventory_path).read_text())
    if file_hash(geometry_path) != report["geometry_sha256"]:
        raise ValueError("pilot geometry changed after inventory")
    geometry = candidate_geometry(geometry_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ee.Initialize(project=report["project"])
    ee.data.setDeadline(120_000)
    ee.data.setMaxRetries(1)
    ee.data.setWorkloadTag("nepal-2026-pilot")
    aoi = ee.Geometry(geometry).buffer(2000, 10)
    bounds = aoi.bounds(10, ee.Projection("EPSG:32645")).coordinates().getInfo()[0]
    grid = download_grid(bounds)
    records = {}
    for name, window in report["windows"].items():
        period = name.split("_")[1]
        collection = ee.ImageCollection.fromImages(
            [ee.Image(report["collection"] + "/" + scene["id"]) for scene in window["scenes"]]
        )
        scores = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        linked = collection.linkCollection(scores, ["cs_cdf"])
        image = (
            linked.map(
                lambda image: image.select(BANDS).updateMask(image.select("cs_cdf").gte(0.6))
            )
            .median()
            .clip(aoi)
            .toFloat()
            .unmask(-9999)
        )
        url = image.getDownloadURL(
            {
                "crs": "EPSG:32645",
                "crs_transform": grid["transform"],
                "dimensions": [grid["width"], grid["height"]],
                "format": "GEO_TIFF",
            }
        )
        raw_path = output / f"{period}-raw.tif"
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            size = 0
            with raw_path.open("xb") as stream:
                for chunk in response.iter_content(1_048_576):
                    size += len(chunk)
                    if size > 32_000_000:
                        raise ValueError("download exceeded Google's 32MB request budget")
                    stream.write(chunk)
        path = output / f"{period}.tif"
        with rasterio.open(raw_path) as src:
            expected = rasterio.Affine(*grid["transform"])
            if (
                src.count != 13
                or src.crs.to_epsg() != 32645
                or src.transform != expected
                or src.width != grid["width"]
                or src.height != grid["height"]
            ):
                raise ValueError("downloaded pilot schema/grid mismatch")
            if any(src.descriptions) and list(src.descriptions) != BANDS:
                raise ValueError("downloaded band descriptions disagree with export order")
            profile = src.profile.copy()
            profile.update(nodata=-9999)
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(src.read())
                dst.descriptions = tuple(BANDS)
        records[period] = {
            "path": path.name,
            "raw_sha256": file_hash(raw_path),
            "sha256": file_hash(path),
            "downloaded_bytes": size,
        }
        dump_json(
            output / "download.json",
            {
                "inventory_sha256": file_hash(inventory_path),
                "geometry_sha256": file_hash(geometry_path),
                "grid": grid,
                "bands": BANDS,
                "records": records,
                "status": "pilot downloaded; human imagery/registration QA pending",
                "exports_submitted": False,
            },
        )
    return records
