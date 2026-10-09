"""Synthetic regression checks only; these do not establish Nepal research results."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio

from landcover.data import BANDS, CLASSES, dump_json, file_hash
from landcover.nepal import change_map, patches
from landcover.nepal_study import (
    FIELDS,
    annotation,
    compare,
    freeze_labels,
    review_exclusions,
    validate_change_inputs,
    validate_probabilities,
    verify_models,
    write_csv,
)


@pytest.fixture
def patch_manifest(tmp_path):
    profile = dict(
        driver="GTiff",
        width=129,
        height=65,
        count=13,
        dtype="float32",
        crs="EPSG:32645",
        transform=rasterio.transform.from_origin(300000, 3100000, 10, 10),
        nodata=-9999,
    )
    for period in ("pre", "post"):
        with rasterio.open(tmp_path / f"{period}.tif", "w", **profile) as dst:
            data = np.full((13, 65, 129), 1000, dtype="float32")
            data[:, 0, 64] = -9999
            dst.write(data)
    patches(tmp_path / "pre.tif", tmp_path / "post.tif", tmp_path / "patches")
    return tmp_path / "patches/patches.json"


@pytest.mark.parametrize(
    "values",
    [
        [[0.1] * 9],
        [[0.2] * 10],
        [[float("nan")] * 10],
        [[-1] + [2 / 9] * 9],
    ],
)
def test_invalid_probabilities(values):
    with pytest.raises(ValueError, match="probabilities"):
        validate_probabilities({"sample_ids": ["a"], "probabilities": values})


def test_coverage_and_review_unknown(patch_manifest, tmp_path):
    manifest = json.loads(patch_manifest.read_text())
    coverage = manifest["coverage"]
    assert coverage["valid_pixels"] == coverage["invalid_pixels"] == 4096
    assert coverage["edge_pixels"] == 129 * 65 - 8192
    records = []
    for period in ("pre", "post"):
        path = tmp_path / f"{period}.json"
        dump_json(
            path,
            dict(
                period=period,
                sample_ids=[manifest["samples"][0]["sample_id"]],
                probabilities=[[1.0] + [0.0] * 9],
                checkpoint_sha256="fixture",
                patch_manifest_sha256=file_hash(patch_manifest),
            ),
        )
        records.append(path)
    exclusions = tmp_path / "exclusions.json"
    dump_json(
        exclusions,
        dict(
            patch_manifest_sha256=file_hash(patch_manifest),
            sample_ids=[manifest["samples"][0]["sample_id"]],
        ),
    )
    change_map(*records, patch_manifest, tmp_path / "map", exclusions=exclusions)
    with rasterio.open(tmp_path / "map/possible_changes.tif") as src:
        assert np.all(src.read(1) == 255)
    records[1].write_text(records[1].read_text().replace('"fixture"', '"different"'))
    with pytest.raises(ValueError, match="same frozen"):
        change_map(*records, patch_manifest, tmp_path / "bad")


def label_rows(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row.update(label=CLASSES[0], labeler="first", confidence="high", dominant_fraction="0.7")
    return rows


def test_blind_sampling_freeze_and_comparison(patch_manifest, tmp_path):
    for name in ("a", "b"):
        annotation(patch_manifest, tmp_path / name)
    assert (tmp_path / "a/assignment.json").read_bytes() == (
        tmp_path / "b/assignment.json"
    ).read_bytes()
    plan = json.loads((tmp_path / "a/assignment.json").read_text())
    assert plan["pre_actual"] == plan["post_actual"] == 1
    assert plan["shortfall"] == {"pre": 299, "post": 99}
    a = tmp_path / "a/labels-primary.csv"
    b = tmp_path / "a/labels-second.csv"
    final = tmp_path / "final.csv"
    rows, second = label_rows(a), label_rows(b)
    write_csv(a, FIELDS, rows)
    write_csv(b, FIELDS, second)
    write_csv(final, FIELDS, rows)
    with pytest.raises(ValueError, match="independent"):
        freeze_labels(tmp_path / "a/assignment.json", a, b, final, tmp_path / "bad")
    for row in second:
        row.update(label=CLASSES[1], labeler="second")
    write_csv(b, FIELDS, second)
    with pytest.raises(ValueError, match="adjudication"):
        freeze_labels(tmp_path / "a/assignment.json", a, b, final, tmp_path / "bad2")
    for row in rows:
        row["notes"] = "synthetic adjudication"
    write_csv(final, FIELDS, rows)
    frozen = tmp_path / "frozen"
    freeze_labels(tmp_path / "a/assignment.json", a, b, final, frozen)
    models = tmp_path / "models.json"
    dump_json(models, {name: {"sha256": name} for name in ("rgb", "ms")})
    predictions = []
    for name, index in (("rgb", 1), ("ms", 0)):
        prob = [0.0] * 10
        prob[index] = 1
        path = tmp_path / f"{name}.json"
        dump_json(
            path,
            dict(
                period="pre",
                sample_ids=[plan["keys"][0][0]],
                probabilities=[prob],
                checkpoint_sha256=name,
                patch_manifest_sha256=file_hash(patch_manifest),
            ),
        )
        predictions.append(path)
    compare(*predictions, frozen, models, tmp_path / "metrics")
    result = json.loads((tmp_path / "metrics/summary.json").read_text())
    assert result["ms"]["macro_f1"] == 0.1
    assert result["ms"]["represented_class_macro_f1"] == 1
    assert result["paired_correctness"]["ms_only_correct"] == 1
    (frozen / "labels.csv").write_text("tampered")
    with pytest.raises(ValueError, match="frozen labels changed"):
        compare(*predictions, frozen, models, tmp_path / "bad3")


def test_qa_displacement_and_primary_model_guard(patch_manifest, tmp_path):
    qa, models = tmp_path / "qa.json", tmp_path / "models.json"
    quality = dict(
        patch_manifest_sha256=file_hash(patch_manifest),
        imagery_review_passed=True,
        reviewer="synthetic",
        reviewed_on="2026-10-08",
        landmarks=[
            dict(
                id=str(i),
                row=i,
                col=i,
                residual_pixels=1.1,
                notes="fixture",
                reference_source="fixture",
            )
            for i in range(10)
        ],
    )
    dump_json(qa, quality)
    with pytest.raises(ValueError, match="landmarks"):
        validate_change_inputs("missing", "missing", patch_manifest, qa, models)
    for point in quality["landmarks"]:
        point["residual_pixels"] = 1
    dump_json(qa, quality)
    dump_json(models, {"ms": {"sha256": "ms", "config": {"input_type": "ms"}}})
    prediction = tmp_path / "prediction.json"
    dump_json(
        prediction, dict(sample_ids=["a"], probabilities=[[1] + [0] * 9], checkpoint_sha256="rgb")
    )
    with pytest.raises(ValueError, match="primary 13-band"):
        validate_change_inputs(prediction, prediction, patch_manifest, qa, models)


def test_freeze_review_preserves_unsupported(tmp_path):
    plan, review = tmp_path / "assignment.json", tmp_path / "review.csv"
    dump_json(plan, {"sample_ids": ["a"], "patch_manifest_sha256": "fixture"})
    fields = ["sample_id", "reviewer", "reference_source", "conclusion", "notes"]
    row = dict(
        sample_id="a",
        reviewer="human",
        reference_source="independent fixture",
        conclusion="unsupported",
        notes="synthetic rock surface",
    )
    write_csv(review, fields, [row])
    review_exclusions(plan, review, tmp_path / "reviewed")
    assert json.loads((tmp_path / "reviewed/exclusions.json").read_text())["sample_ids"] == ["a"]
    row["reviewer"] = ""
    write_csv(review, fields, [row])
    with pytest.raises(ValueError, match="reviewer"):
        review_exclusions(plan, review, tmp_path / "bad")


def test_model_verification_requires_validation_selected_checkpoint(monkeypatch, tmp_path):
    import landcover.nepal_study as study

    def load(path):
        name = Path(path).parent.name
        config = dict(
            model="resnet18",
            initialization="pretrained",
            fraction=1.0,
            seed=42,
            input_type=name,
            rgb_source="ms",
        )
        channels = 3 if name == "rgb" else 13
        checkpoint = dict(
            config=config,
            bands=BANDS,
            classes=CLASSES,
            step=2,
            val_macro_f1=0.8,
            normalization=dict(mean=[0] * channels, std=[1] * channels),
            provenance=dict(
                manifest_digest="m", split_digest="s", train_ids_digest="t", git_commit="g"
            ),
        )
        return None, checkpoint, SimpleNamespace(**config)

    monkeypatch.setattr(study, "load_checkpoint", load)
    paths = []
    for name in ("rgb", "ms"):
        folder = tmp_path / name
        folder.mkdir()
        path = folder / "best.pt"
        path.write_text(name)
        (folder / "learning_curves.csv").write_text("step,val_macro_f1\n1,0.7\n2,0.8\n")
        paths.append(path)
    verify_models(*paths, tmp_path / "verified")
    (paths[1].parent / "learning_curves.csv").write_text("step,val_macro_f1\n1,0.9\n2,0.8\n")
    with pytest.raises(ValueError, match="first maximum"):
        verify_models(*paths, tmp_path / "bad")


def test_sensitivity_panels_and_provenance(patch_manifest, tmp_path):
    from landcover.nepal_study import sensitivity

    manifest = json.loads(patch_manifest.read_text())
    sid = manifest["samples"][0]["sample_id"]
    paths = []
    for period, index in (("pre", 0), ("post", 1)):
        path = tmp_path / f"{period}.json"
        probabilities = [0.0] * 10
        probabilities[index] = 0.75
        probabilities[2] = 0.25
        dump_json(
            path,
            dict(
                period=period,
                sample_ids=[sid],
                probabilities=[probabilities],
                checkpoint_sha256="ms",
                patch_manifest_sha256=file_hash(patch_manifest),
            ),
        )
        paths.append(path)
    qa, models = tmp_path / "qa.json", tmp_path / "models.json"
    dump_json(
        qa,
        dict(
            patch_manifest_sha256=file_hash(patch_manifest),
            imagery_review_passed=True,
            reviewer="fixture",
            reviewed_on="2026-10-08",
            landmarks=[
                dict(
                    id=str(i),
                    row=i,
                    col=i,
                    residual_pixels=0.5,
                    notes="synthetic only",
                    reference_source="fixture",
                )
                for i in range(10)
            ],
        ),
    )
    dump_json(models, {"ms": {"sha256": "ms", "config": {"input_type": "ms"}}})
    result = sensitivity(*paths, patch_manifest, qa, models, tmp_path / "sensitivity")
    assert [r["accepted_patches"] for r in result] == [1, 1, 0]
    assert (tmp_path / f"sensitivity/{sid}.png").exists()
    assignment = json.loads((tmp_path / "sensitivity/review-assignment.json").read_text())
    assert assignment["sample_ids"] == [sid]
    assert assignment["qa_sha256"] == file_hash(qa)


def test_unvalidated_demo_preserves_unknown_and_does_not_pass_research_qa(patch_manifest, tmp_path):
    from landcover.nepal_study import demonstration

    manifest = json.loads(patch_manifest.read_text())
    records = []
    for period, label in (("pre", 0), ("post", 1)):
        probs = [0.0] * 10
        probs[label] = 1.0
        path = tmp_path / f"demo-{period}.json"
        dump_json(
            path,
            dict(
                period=period,
                sample_ids=[manifest["samples"][0]["sample_id"]],
                probabilities=[probs],
                checkpoint_sha256="fixture-ms",
                patch_manifest_sha256=file_hash(patch_manifest),
            ),
        )
        records.append(path)
    models = tmp_path / "models.json"
    dump_json(models, {"ms": {"sha256": "fixture-ms", "config": {"input_type": "ms"}}})
    output = tmp_path / "demo"
    demonstration(*records, patch_manifest, models, output)
    report = json.loads((output / "demonstration.json").read_text())
    assert all(
        not r["accuracy_available"] and not r["imagery_review_passed"] for r in report["thresholds"]
    )
    with rasterio.open(output / "threshold-0.7/unvalidated_model_changes.tif") as src:
        array = src.read(1)
        assert (array[:64, :64] == 1).all()
        assert (array[:, 64:] == 255).all()
        assert src.tags()["validation_status"] == "unvalidated_demonstration"
    qa = tmp_path / "qa.json"
    dump_json(qa, {"imagery_review_passed": False})
    with pytest.raises(ValueError, match="imagery QA"):
        validate_change_inputs(*records, patch_manifest, qa, models)
    bad = json.loads(records[1].read_text())
    bad["checkpoint_sha256"] = "other-model"
    dump_json(records[1], bad)
    with pytest.raises(ValueError, match="verified primary"):
        demonstration(*records, patch_manifest, models, tmp_path / "bad-demo")
