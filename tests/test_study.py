import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from landcover.config import Config
from landcover.data import dump_json, load_prepared
from landcover.metrics import score
from landcover.study import jobs, load_study, prepare_study, run_study
from landcover.study import test_study as evaluate_study_test
from landcover.study_report import aggregate, paired_comparisons, report


def base_config(paired):
    return Config(
        manifest=str(paired / "prepared/manifest.json"),
        splits=str(paired / "prepared/splits.json"),
        steps=2,
        eval_every=1,
        batch_size=4,
        threads=2,
        measure_train_metrics=True,
        device="cpu",
    )


def test_study_protocol_and_incomplete_reporting(paired, tmp_path):
    root = tmp_path / "study"
    assert prepare_study(root, base_config(paired)) == 63
    plan = load_study(root)
    assert len({job["name"] for job in plan["jobs"]}) == 63
    assert len([job for job in plan["jobs"] if job["kind"] == "main"]) == 48
    assert len([job for job in plan["jobs"] if job["kind"] == "ablation"]) == 12
    assert {c.fraction for _, kind, c in jobs(base_config(paired)) if kind == "ablation"} == {0.1}
    with pytest.raises(ValueError, match="incomplete"):
        report(root)
    with pytest.raises(ValueError, match="allow-test"):
        evaluate_study_test(root)
    with pytest.raises(ValueError, match="incomplete"):
        evaluate_study_test(root, allow_test=True)
    assert not (root / "test-phase.json").exists()
    path = root / "configs" / f"{plan['jobs'][0]['name']}.yaml"
    replace(Config.load(path), steps=3).save(path)
    with pytest.raises(ValueError, match="Frozen config changed"):
        load_study(root)


def test_study_retries_skip_completed_and_explicit_test_phase(paired, tmp_path, monkeypatch):
    # Synthetic orchestration fixture only; real train/evaluate are covered in test_training.
    import landcover.study as study

    root = tmp_path / "study"
    prepare_study(root, base_config(paired))
    plan = load_study(root)
    calls = {"train": 0, "val": 0, "test": 0}

    def fake_train(config):
        calls["train"] += 1
        path = Path(config.output)
        path.mkdir(parents=True, exist_ok=False)
        config.save(path / "config.yaml")
        if calls["train"] == 1:
            raise RuntimeError("synthetic interruption")
        checkpoint = {
            "config": asdict(config),
            "step": config.steps,
            "val_macro_f1": 1.0,
            "provenance": {key: plan[key] for key in ("manifest_digest", "split_digest")},
        }
        torch.save(checkpoint, path / "best.pt")
        torch.save(checkpoint, path / "last.pt")
        with (path / "learning_curves.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["step", "val_macro_f1", "macro_f1_gap"])
            writer.writeheader()
            writer.writerow({"step": config.steps, "val_macro_f1": 1.0, "macro_f1_gap": 0.0})
        return path / "best.pt"

    def fake_evaluate(checkpoint, split, output, allow_test):
        assert split != "test" or allow_test
        calls[split] += 1
        config = Config.load(checkpoint.parent / "config.yaml")
        manifest, splits = load_prepared(config.manifest, config.splits)
        by_id = {r["id"]: r["label"] for r in manifest["samples"]}
        ids = splits[split]
        labels = [by_id[sid] for sid in ids]
        output.mkdir(exist_ok=False)
        if split == "val" and calls["val"] == 1:
            raise RuntimeError("synthetic evaluation interruption")
        metrics = score(labels, labels)
        metrics["provenance"] = {
            "split": split,
            "training_samples": len(splits["subsets"][str(config.fraction)]),
        }
        dump_json(output / "metrics.json", metrics)
        with (output / "predictions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample_id", *[f"p_{i}" for i in range(10)]])
            for sid, label in zip(ids, labels, strict=True):
                writer.writerow([sid, *[int(i == label) for i in range(10)]])
        (output / "confusion_matrix.png").write_bytes(b"synthetic orchestration placeholder")

    monkeypatch.setattr(study, "train", fake_train)
    monkeypatch.setattr(study, "evaluate", fake_evaluate)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_study(root)
    first = root / "runs" / plan["jobs"][0]["name"]
    with pytest.raises(RuntimeError, match="retry-incomplete"):
        run_study(root)
    with pytest.raises(RuntimeError, match="synthetic evaluation interruption"):
        run_study(root, retry_incomplete=True)
    assert run_study(root).exists()
    assert (first / "attempt-001/config.yaml").exists()
    assert (first / "attempt-002/complete.json").exists()
    assert calls == {"train": 64, "val": 64, "test": 0}
    run_study(root)
    assert calls == {"train": 64, "val": 64, "test": 0}
    assert json.loads((root / "selection.json").read_text())["representative_seed"] == 42
    assert evaluate_study_test(root, allow_test=True).exists()
    assert (
        calls["test"] == 51
    )  # Main conditions and compact baselines; ablations stay validation-only.
    evaluate_study_test(root, allow_test=True)
    assert calls["test"] == 51
    receipt = json.loads((first / "attempt-002/verified-val.json").read_text())
    assert receipt["directory"] == "evaluation-val-retry-1"
    metrics_path = first / "attempt-002" / receipt["directory"] / "metrics.json"
    metrics_path.write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        report(root)


def test_paired_seed_statistics_and_missing_seed_rejection():
    rows = []
    values = {
        ("scratch", "rgb"): [0.5, 0.6, 0.7],
        ("pretrained", "rgb"): [0.6, 0.75, 0.9],
        ("scratch", "ms"): [0.55, 0.7, 0.8],
        ("pretrained", "ms"): [0.65, 0.85, 0.95],
    }
    for (initialization, modality), scores in values.items():
        for seed, value in zip((42, 43, 44), scores, strict=True):
            rows.append(
                {
                    "condition": f"{initialization}-{modality}",
                    "kind": "main",
                    "model": "resnet18",
                    "initialization": initialization,
                    "input_type": modality,
                    "fraction": 0.1,
                    "training_samples": 1620,
                    "augmentation": True,
                    "dropout": 0.5,
                    "seed": seed,
                    "macro_f1": value,
                    "accuracy": value,
                    "validation_gap": 0.1,
                }
            )
    comparisons = paired_comparisons(rows)
    difference = next(r for r in comparisons if r["comparison"] == "pretrained-minus-scratch-rgb")
    assert difference["mean_difference"] == pytest.approx(0.15)
    assert difference["sample_sd"] == pytest.approx(0.05)
    summary = next(r for r in aggregate(rows) if r["condition"] == "scratch-rgb")
    assert summary["macro_f1_mean"] == pytest.approx(0.6)
    assert summary["macro_f1_sd"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="exactly seeds"):
        aggregate(rows[:-1])
