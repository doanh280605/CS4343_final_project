import json

import numpy as np
import pytest
import rasterio

from landcover.data import BANDS, dump_json, file_hash
from landcover.nepal import align, change_map, patches, validate_acquisition


def test_nepal_requires_verified_event():
    with pytest.raises(ValueError, match="fill"):
        validate_acquisition({})
    config = dict(
        event_name="Synthetic test event",
        event_date="2015-04-25",
        event_source="https://example.org/test-fixture",
        source_checked_on="2026-10-01",
        project="test-project",
        bbox=[85, 27, 86, 28],
        pre_start="2015-04-01",
        pre_end="2015-04-20",
        post_start="2015-05-01",
        post_end="2015-05-10",
        crs="EPSG:32645",
        collection="COPERNICUS/S2_HARMONIZED",
    )
    with pytest.raises(ValueError, match="coverage starts"):
        validate_acquisition(config)


def test_alignment_patches_and_nodata_change_map(tmp_path):
    profile = dict(
        driver="GTiff",
        width=128,
        height=64,
        count=13,
        dtype="float32",
        crs="EPSG:32645",
        transform=rasterio.transform.from_origin(300000, 3100000, 10, 10),
        nodata=-9999,
    )
    for period in ("pre", "post"):
        with rasterio.open(tmp_path / f"{period}.tif", "w", **profile) as dst:
            data = np.full((13, 64, 128), 1000, dtype="float32")
            data[:, :, 64:] = -9999
            dst.write(data)
    aligned = tmp_path / "aligned.tif"
    align(tmp_path / "pre.tif", tmp_path / "post.tif", aligned)
    output = tmp_path / "patches"
    assert patches(tmp_path / "pre.tif", aligned, output) == 1
    path = output / "patches.json"
    manifest = json.loads(path.read_text())
    assert manifest["bands"] == BANDS
    for period, label in (("pre", 0), ("post", 1)):
        probs = [0.0] * 10
        probs[label] = 1.0
        dump_json(
            tmp_path / f"{period}.json",
            {
                "period": period,
                "sample_ids": ["r000000_c000000"],
                "probabilities": [probs],
                "checkpoint_sha256": "fixture",
                "patch_manifest_sha256": file_hash(path),
            },
        )
    change_map(tmp_path / "pre.json", tmp_path / "post.json", path, tmp_path / "changes")
    with rasterio.open(tmp_path / "changes/possible_changes.tif") as src:
        array = src.read(1)
        assert np.all(array[:, :64] == 1)
        assert np.all(array[:, 64:] == 255)
        assert src.nodata == 255
    transition = json.loads((tmp_path / "changes/transitions.json").read_text())
    assert transition["counts"][0][1] == 1
    assert transition["accepted_patches"] == 1


def test_dry_run_acquisition_accepts_yaml_dates(tmp_path):
    import yaml

    from landcover.nepal import acquire

    config = tmp_path / "plan.yaml"
    config.write_text(
        """event_name: Synthetic pipeline test, not a real event
 event_date: 2020-06-01
 event_source: https://example.org/synthetic-fixture
 source_checked_on: 2026-10-01
 project: fake-project
 bbox: [85, 27, 86, 28]
 pre_start: 2020-05-01
 pre_end: 2020-05-20
 post_start: 2020-06-02
 post_end: 2020-06-20
 crs: EPSG:32645
 collection: COPERNICUS/S2_HARMONIZED
""".replace("\n ", "\n")
    )
    assert yaml.safe_load(config.read_text())["event_date"].isoformat() == "2020-06-01"
    result = acquire(config, tmp_path / "plan")
    assert result["submitted"] is False
    assert (
        json.loads((tmp_path / "plan/acquisition-plan.json").read_text())["event_date"]
        == "2020-06-01"
    )


def test_nepal_inference_with_independent_hand_labels(paired, tmp_path):
    import csv
    from dataclasses import asdict

    import torch

    from landcover.config import Config
    from landcover.models import build_model
    from landcover.nepal import infer

    config = Config(input_type="ms", threads=2)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "config": asdict(config),
            "bands": BANDS,
            "model": build_model(config).state_dict(),
            "normalization": {"mean": [0.0] * 13, "std": [1.0] * 13},
        },
        checkpoint,
    )
    patch_dir = tmp_path / "patches"
    source = paired / "ms/Forest/Forest_1.tif"
    patches(source, source, patch_dir)
    labels = patch_dir / "labels.csv"
    with labels.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "period", "label", "labeler", "confidence", "notes"])
        writer.writerow(["r000000_c000000", "pre", "Forest", "fixture", "high", "synthetic"])
    assert infer(checkpoint, patch_dir / "patches.json", "pre", tmp_path / "inference", labels) == 1
    result = json.loads((tmp_path / "inference/evaluation/metrics.json").read_text())
    assert result["provenance"]["hand_labeled_count"] == 1
    assert result["per_class"]["Forest"]["support"] == 1
