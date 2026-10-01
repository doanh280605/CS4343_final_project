import json
import zipfile

import numpy as np
import pytest
import torch

from landcover.data import (
    CLASSES,
    RGB_INDICES,
    SatelliteDataset,
    audit,
    fit_stats,
    load_prepared,
    make_splits,
    prepare,
    validate_splits,
)
from landcover.download import extract_archive


def test_production_split_counts_and_nested_subsets():
    counts = [3000, 3000, 3000, 2500, 2500, 2000, 2500, 3000, 2500, 3000]
    rows = [
        {"id": f"{name}/{i}", "label": label}
        for label, (name, count) in enumerate(zip(CLASSES, counts, strict=True))
        for i in range(count)
    ]
    splits = make_splits(rows)
    assert [len(splits[k]) for k in ("train", "val", "test")] == [16200, 5400, 5400]
    assert [len(splits["subsets"][str(f)]) for f in (1.0, 0.1, 0.05, 0.01)] == [
        16200,
        1620,
        810,
        162,
    ]
    assert splits == make_splits(list(reversed(rows)))
    splits["val"][0] = splits["train"][0]
    with pytest.raises(ValueError, match="disjoint"):
        validate_splits(rows, splits)


def test_alignment_shapes_and_training_stats(paired):
    manifest = paired / "prepared/manifest.json"
    splits_path = paired / "prepared/splits.json"
    _, splits = load_prepared(manifest, splits_path)
    assert audit(manifest, splits_path)["paired"] is True
    ms = SatelliteDataset(manifest, splits["train"], "ms")
    rgb = SatelliteDataset(manifest, splits["train"], "rgb")
    jpg = SatelliteDataset(manifest, splits["train"], "rgb", "jpg")
    assert ms[0][0].shape == (13, 64, 64)
    assert rgb[0][0].shape == jpg[0][0].shape == (3, 64, 64)
    assert torch.equal(rgb.raw(0), ms.raw(0)[RGB_INDICES])
    assert ms[0][2] == rgb[0][2] == jpg[0][2]
    stats = fit_stats(rgb)
    exact = torch.stack([rgb.raw(i) for i in range(len(rgb))]).double()
    assert np.allclose(stats["mean"], exact.mean((0, 2, 3)), atol=1e-8)
    assert np.allclose(stats["std"], exact.std((0, 2, 3), correction=0), atol=1e-8)
    assert stats["sample_count"] == 60
    normalized = SatelliteDataset(manifest, splits["train"], stats=stats)
    assert torch.stack([normalized[i][0] for i in range(len(normalized))]).mean().abs() < 1e-5
    assert normalized.augment is False


def test_stats_never_reads_validation(paired, monkeypatch):
    manifest = paired / "prepared/manifest.json"
    splits = json.loads((paired / "prepared/splits.json").read_text())
    selected = splits["subsets"]["0.1"]
    dataset = SatelliteDataset(manifest, selected)
    seen = []
    original = dataset.raw

    def spy(index):
        seen.append(dataset.rows[index]["id"])
        return original(index)

    monkeypatch.setattr(dataset, "raw", spy)
    fit_stats(dataset)
    assert set(seen) == set(selected)
    assert not set(seen) & set(splits["val"] + splits["test"])


def test_pair_mismatch_and_mutated_manifest(paired, tmp_path):
    partial = tmp_path / "rgb/Forest"
    partial.mkdir(parents=True)
    (partial / "extra.jpg").write_bytes(b"bad")
    with pytest.raises(ValueError, match="mismatch"):
        prepare(tmp_path / "bad", partial.parent, paired / "ms")
    manifest = json.loads((paired / "prepared/manifest.json").read_text())
    manifest["samples"][0]["label"] = 9
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="changed"):
        load_prepared(changed, paired / "prepared/splits.json")


def test_zip_traversal_rejected(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escaped.txt", "unsafe")
    with pytest.raises(ValueError, match="unsafe"):
        extract_archive(archive, tmp_path / "dest")
    assert not (tmp_path / "escaped.txt").exists()
