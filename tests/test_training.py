import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from torchvision.models import resnet18

from landcover.config import Config
from landcover.engine import evaluate, load_checkpoint, train
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
