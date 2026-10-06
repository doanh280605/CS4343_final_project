import csv
import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
from torchvision.models import resnet18

from landcover.config import Config
from landcover.engine import build_optimizer, build_scheduler, evaluate, load_checkpoint, train
from landcover.experiments import grid
from landcover.metrics import score
from landcover.models import build_model, expand_rgb_weights


def test_metrics_known_answer():
    result = score(list(range(10)), list(range(10)))
    assert result["accuracy"] == result["macro_f1"] == 1
    result = score([0, 0, 1, 1], [0, 1, 1, 1])
    assert result["accuracy"] == 0.75
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 10)
    assert np.asarray(result["confusion_matrix"]).sum() == 4
    assert result["per_class"]["AnnualCrop"]["recall"] == 0.5


@pytest.mark.parametrize("model", ["compact", "resnet18"])
@pytest.mark.parametrize("input_type,channels", [("rgb", 3), ("ms", 13)])
def test_model_shapes(model, input_type, channels):
    network = build_model(Config(model=model, input_type=input_type))
    network.eval()
    with torch.inference_mode():
        assert network(torch.zeros(2, channels, 64, 64)).shape == (2, 10)


def test_pretrained_band_mapping(monkeypatch):
    source = resnet18(weights=None)
    original = source.conv1.weight.detach().clone()

    def fake_resnet18(weights):
        assert weights is not None
        return source

    monkeypatch.setattr("landcover.models.resnet18", fake_resnet18)
    network = build_model(Config(model="resnet18", initialization="pretrained", input_type="ms"))
    assert torch.equal(network.conv1.weight, expand_rgb_weights(original))
    assert torch.allclose(network.conv1.weight[:, 3], original[:, 0] * 3 / 13)
    assert torch.allclose(network.conv1.weight[:, 10], original.mean(1) * 3 / 13)


def test_training_checkpoint_evaluation_and_budget(paired, tmp_path):
    config = Config(
        manifest=str(paired / "prepared/manifest.json"),
        splits=str(paired / "prepared/splits.json"),
        output=str(tmp_path / "run"),
        steps=3,
        eval_every=2,
        batch_size=4,
        fraction=0.01,
        threads=2,
        device="cpu",
    )
    checkpoint = train(config)
    model, best, _ = load_checkpoint(checkpoint)
    last = torch.load(tmp_path / "run/last.pt", weights_only=True)
    assert last["step"] == 3
    assert {int(state["step"]) for state in last["optimizer"]["state"].values()} == {3}
    assert best["normalization"]["sample_count"] == 10
    with (tmp_path / "run/learning_curves.csv").open() as f:
        curves = list(csv.DictReader(f))
    assert best["val_macro_f1"] == max(float(row["val_macro_f1"]) for row in curves)
    evaluate(checkpoint)
    with (tmp_path / "run/evaluation-val/predictions.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20
    assert len({row["sample_id"] for row in rows}) == 20
    assert all(
        sum(float(v) for k, v in row.items() if k.startswith("p_")) == pytest.approx(1, abs=1e-6)
        for row in rows
    )
    assert (tmp_path / "run/evaluation-val/confusion_matrix.png").stat().st_size > 0
    with pytest.raises(ValueError, match="allow-test"):
        evaluate(checkpoint, "test")
    assert not (tmp_path / "run/evaluation-test").exists()
    evaluate(checkpoint, "test", allow_test=True)
    # Same data, seed and CPU settings reproduce the learned parameters exactly.
    replica = train(replace(config, output=str(tmp_path / "replica")))
    _, repeated, _ = load_checkpoint(replica)
    assert all(torch.equal(v, repeated["model"][k]) for k, v in best["model"].items())
    full = train(replace(config, fraction=1.0, output=str(tmp_path / "full")))
    full_last = torch.load(full.parent / "last.pt", weights_only=True)
    assert full_last["step"] == last["step"]
    assert full_last["provenance"]["examples_drawn"] == last["provenance"]["examples_drawn"]
    with pytest.raises(FileExistsError):
        train(config)


def test_grid_is_48_unique_runs(tmp_path):
    names = grid(tmp_path / "grid")
    assert len(names) == len(set(names)) == 48
    configs = [Config.load(p) for p in (tmp_path / "grid").glob("*.yaml")]
    assert {c.steps for c in configs} == {1000}
    assert {c.rgb_source for c in configs} == {"ms"}
    assert json.loads((tmp_path / "grid/index.json").read_text())["count"] == 48


def test_finetuning_groups_and_warmup_cosine_budget():
    model = resnet18(weights=None)
    config = Config(
        model="resnet18",
        initialization="pretrained",
        pretrained_backbone_learning_rate=1e-4,
        scheduler="cosine",
        warmup_steps=2,
        steps=6,
    ).validate()
    optimizer = build_optimizer(model, config)
    backbone, head = optimizer.param_groups
    head_ids = {id(p) for p in model.fc.parameters()}
    backbone_ids = {id(p) for p in backbone["params"]}
    assert head_ids == {id(p) for p in head["params"]}
    assert not head_ids & backbone_ids
    assert head_ids | backbone_ids == {id(p) for p in model.parameters()}
    scheduler = build_scheduler(optimizer, config)
    used_rates = []
    for _ in range(config.steps):
        used_rates.append([group["lr"] for group in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
    assert np.asarray(used_rates)[:, 1] == pytest.approx(
        np.asarray([0.5, 1, 1, (2 + np.sqrt(2)) / 4, 0.5, (2 - np.sqrt(2)) / 4]) * 1e-3
    )
    assert np.asarray(used_rates)[:, 0] == pytest.approx(np.asarray(used_rates)[:, 1] / 10)
    assert optimizer.param_groups[0]["lr"] == 0
    scratch = build_optimizer(model, replace(config, initialization="scratch"))
    assert len(scratch.param_groups) == 1
    assert scratch.param_groups[0]["lr"] == 1e-3


@pytest.mark.parametrize(
    "overrides",
    [
        {"scheduler": "invalid"},
        {"warmup_steps": 1},
        {"scheduler": "cosine", "warmup_steps": 1000},
        {"pretrained_backbone_learning_rate": 0},
    ],
)
def test_invalid_optimization_settings(overrides):
    with pytest.raises(ValueError):
        Config(**overrides).validate()


@pytest.mark.parametrize(
    "bands",
    [None, ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]],
)
def test_old_ms_checkpoint_cannot_silently_transfer_with_wrong_bands(tmp_path, bands):
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"config": asdict(Config(input_type="ms")), "bands": bands}, checkpoint)
    with pytest.raises(ValueError, match="band schema"):
        load_checkpoint(checkpoint)


def test_training_metrics_use_only_selected_unaugmented_subset(paired, tmp_path, monkeypatch):
    import landcover.engine as engine

    splits = json.loads((paired / "prepared/splits.json").read_text())
    seen = []
    original = engine.predict

    def spy(model, loader, device):
        seen.append((set(row["id"] for row in loader.dataset.rows), loader.dataset.augment))
        return original(model, loader, device)

    monkeypatch.setattr(engine, "predict", spy)
    config = Config(
        manifest=str(paired / "prepared/manifest.json"),
        splits=str(paired / "prepared/splits.json"),
        output=str(tmp_path / "train-metrics"),
        fraction=0.1,
        steps=3,
        eval_every=3,
        batch_size=4,
        threads=2,
        scheduler="cosine",
        warmup_steps=1,
        measure_train_metrics=True,
        device="cpu",
    )
    checkpoint = train(config)
    assert seen == [(set(splits["val"]), False), (set(splits["subsets"]["0.1"]), False)]
    with (checkpoint.parent / "learning_curves.csv").open() as f:
        row = next(csv.DictReader(f))
    assert float(row["macro_f1_gap"]) == pytest.approx(
        float(row["train_macro_f1"]) - float(row["val_macro_f1"])
    )
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["scheduler"]["last_epoch"] == 3
    model, _, _ = load_checkpoint(checkpoint)
    assert model is not None
    # Enabling deterministic training-set evaluation must not perturb training RNG/sampling.
    replica = train(
        replace(config, output=str(tmp_path / "no-train-metrics"), measure_train_metrics=False)
    )
    repeated = torch.load(replica, weights_only=True)
    assert all(torch.equal(value, repeated["model"][key]) for key, value in saved["model"].items())
