"""Frozen study execution: separate validation/training and explicit final test phases."""

import csv
import itertools
import json
import pickle
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import torch

from landcover.config import Config
from landcover.data import FRACTIONS, audit, digest, dump_json, file_hash, load_prepared
from landcover.engine import evaluate, provenance, train

SEEDS = (42, 43, 44)


def source_hashes():
    return {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def jobs(base):
    """48 main comparisons, 12 compact ablations, three full-data compact baselines."""
    result = []
    for init, modality, fraction, seed in itertools.product(
        ("pretrained", "scratch"), ("rgb", "ms"), FRACTIONS, SEEDS
    ):
        name = f"resnet18-{init}-{modality}-f{fraction:g}-s{seed}"
        config = replace(
            base,
            model="resnet18",
            initialization=init,
            input_type=modality,
            fraction=fraction,
            seed=seed,
            rgb_source="ms",
        )
        result.append((name, "main", config))
    for augmentation, dropout, seed in itertools.product((False, True), (0.0, 0.5), SEEDS):
        name = f"compact-aug{int(augmentation)}-drop{dropout:g}-s{seed}"
        config = replace(
            base,
            model="compact",
            initialization="scratch",
            input_type="rgb",
            fraction=0.1,
            seed=seed,
            rgb_source="ms",
            augmentation=augmentation,
            dropout=dropout,
        )
        result.append((name, "ablation", config))
    for seed in SEEDS:
        name = f"compact-baseline-s{seed}"
        config = replace(
            base,
            model="compact",
            initialization="scratch",
            input_type="rgb",
            fraction=1.0,
            seed=seed,
            rgb_source="ms",
            augmentation=True,
            dropout=0.5,
        )
        result.append((name, "baseline", config))
    return result


def prepare_study(output, base):
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"Study already exists: {root}; run it instead of preparing again")
    base.validate()
    if not base.measure_train_metrics:
        raise ValueError("Study reporting requires measure_train_metrics: true")
    manifest, splits = load_prepared(base.manifest, base.splits)
    checks = audit(base.manifest, base.splits)
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    records = []
    for name, kind, config in jobs(base):
        config = replace(config, output=str(root / "runs" / name / "attempt-001"))
        config.validate().save(root / "configs" / f"{name}.yaml")
        records.append({"name": name, "kind": kind, "config": asdict(config)})
    plan = {
        "version": 1,
        "frozen_at": datetime.now(UTC).isoformat(),
        "manifest_digest": digest(manifest),
        "split_digest": digest(splits),
        "source_sha256": source_hashes(),
        "provenance": provenance(),
        "audit": checks,
        "jobs": records,
        "test_kinds": ["main", "baseline"],
        "selection_rule": "highest main-condition mean validation macro-F1; seed 42 representative",
        "interpretation": "Training-seed SD only; fixed test population. Ablations use validation.",
    }
    dump_json(root / "protocol.json", plan)
    dump_json(root / "protocol-digest.json", {"sha256": digest(plan)})
    dump_json(root / "manifest.json", manifest)
    dump_json(root / "splits.json", splits)
    (root / "source").mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        (root / "source" / source.name).write_bytes(source.read_bytes())
    return len(records)


def load_study(root):
    root = Path(root)
    plan = json.loads((root / "protocol.json").read_text())
    if digest(plan) != json.loads((root / "protocol-digest.json").read_text())["sha256"]:
        raise ValueError("Frozen protocol changed; create a separate study for protocol changes")
    if plan["source_sha256"] != source_hashes():
        raise ValueError(
            "Source code changed after study freeze; restore frozen code or use a new study"
        )
    if plan["provenance"]["versions"] != provenance()["versions"]:
        raise ValueError("Package versions changed after study freeze")
    first = Config(**plan["jobs"][0]["config"])
    manifest, splits = load_prepared(first.manifest, first.splits)
    if digest(manifest) != plan["manifest_digest"] or digest(splits) != plan["split_digest"]:
        raise ValueError("Frozen study data changed")
    for job in plan["jobs"]:
        if asdict(Config.load(root / "configs" / f"{job['name']}.yaml")) != job["config"]:
            raise ValueError(f"Frozen config changed: {job['name']}")
    return plan


def log_event(root, **event):
    with (Path(root) / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"time": datetime.now(UTC).isoformat(), **event}) + "\n")


def completed_attempt(root, job):
    parent = Path(root) / "runs" / job["name"]
    completed = [p for p in sorted(parent.glob("attempt-*")) if (p / "complete.json").exists()]
    if len(completed) > 1:
        raise ValueError(f"Multiple completed attempts for {job['name']}; refusing seed selection")
    if not completed:
        return None
    path = completed[0]
    receipt = json.loads((path / "complete.json").read_text())
    expected = {**job["config"], "output": str(path)}
    if receipt["config_digest"] != digest(expected):
        raise ValueError(f"Completion config mismatch: {path}")
    for name, sha256 in receipt["files"].items():
        if file_hash(path / name) != sha256:
            raise ValueError(f"Completed artifact changed: {path / name}")
    return path


def verify_training(path, config, plan):
    saved = Config.load(path / "config.yaml")
    if asdict(saved) != asdict(config):
        raise ValueError(f"Run config mismatch: {path}")
    best = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
    last = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
    for checkpoint in (best, last):
        if checkpoint["config"] != asdict(config):
            raise ValueError("Checkpoint config mismatch")
        for key in ("manifest_digest", "split_digest"):
            if checkpoint["provenance"][key] != plan[key]:
                raise ValueError("Checkpoint data mismatch")
    with (path / "learning_curves.csv").open() as stream:
        curves = list(csv.DictReader(stream))
    if not curves:
        raise ValueError("Missing learning-curve rows")
    if last["step"] != config.steps or int(curves[-1]["step"]) != config.steps:
        raise ValueError("Incomplete optimizer budget")
    selected = max(curves, key=lambda row: float(row["val_macro_f1"]))
    if best["step"] != int(selected["step"]) or best["val_macro_f1"] != float(
        selected["val_macro_f1"]
    ):
        raise ValueError("Checkpoint does not match best validation criterion")


def evaluate_once(path, split, allow_test=False):
    directory = path / f"evaluation-{split}"
    receipt = path / f"verified-{split}.json"
    checkpoint_hash = file_hash(path / "best.pt")
    if receipt.exists():
        record = json.loads(receipt.read_text())
        directory = path / record["directory"]
        if record["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Checkpoint changed after evaluation")
        for name, expected in record["files"].items():
            if file_hash(directory / name) != expected:
                raise ValueError("Evaluation artifact changed")
        return
    if directory.exists():
        # A failed evaluation must remain visible. Use a fresh numbered output directory.
        index = 1
        while (path / f"evaluation-{split}-retry-{index}").exists():
            index += 1
        directory = path / f"evaluation-{split}-retry-{index}"
    evaluate(path / "best.pt", split=split, output=directory, allow_test=allow_test)
    config = Config.load(path / "config.yaml")
    _, splits = load_prepared(config.manifest, config.splits)
    with (directory / "predictions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    ids = [row["sample_id"] for row in rows]
    metrics = json.loads((directory / "metrics.json").read_text())
    if len(ids) != len(set(ids)) or set(ids) != set(splits[split]):
        raise ValueError("Evaluation sample IDs differ from the frozen split")
    if any(
        abs(sum(float(v) for k, v in row.items() if k.startswith("p_")) - 1) > 1e-5 for row in rows
    ):
        raise ValueError("Invalid prediction probability sums")
    if sum(map(sum, metrics["confusion_matrix"])) != len(ids):
        raise ValueError("Confusion matrix sample count mismatch")
    dump_json(
        receipt,
        {
            "checkpoint_sha256": checkpoint_hash,
            "directory": directory.name,
            "files": {p.name: file_hash(p) for p in directory.iterdir() if p.is_file()},
        },
    )


def run_study(root, retry_incomplete=False):
    root = Path(root)
    plan = load_study(root)
    for index, job in enumerate(plan["jobs"], 1):
        if completed_attempt(root, job) is not None:
            print(f"[{index}/{len(plan['jobs'])}] verified complete: {job['name']}", flush=True)
            continue
        parent = root / "runs" / job["name"]
        attempts = sorted(parent.glob("attempt-*"))
        path = attempts[-1] if attempts else parent / "attempt-001"
        config = Config(**{**job["config"], "output": str(path)})
        if path.exists():
            try:
                verify_training(path, config, plan)
            except (
                OSError,
                ValueError,
                KeyError,
                RuntimeError,
                EOFError,
                pickle.UnpicklingError,
            ) as error:
                if not retry_incomplete:
                    raise RuntimeError(
                        f"Incomplete run retained at {path}. Use --retry-incomplete to start it "
                        "from scratch in a fresh attempt directory; exact resume is unavailable."
                    ) from error
                log_event(root, job=job["name"], status="retry-incomplete", previous=str(path))
                path = parent / f"attempt-{len(attempts) + 1:03d}"
                config = replace(config, output=str(path))
        print(f"[{index}/{len(plan['jobs'])}] {job['name']} -> {path}", flush=True)
        try:
            if not path.exists():
                log_event(root, job=job["name"], status="started", output=str(path))
                train(config)
            verify_training(path, config, plan)
            evaluate_once(path, "val")
            files = [
                "config.yaml",
                "best.pt",
                "last.pt",
                "learning_curves.csv",
                "verified-val.json",
            ]
            dump_json(
                path / "complete.json",
                {
                    "config_digest": digest(asdict(config)),
                    "files": {name: file_hash(path / name) for name in files},
                },
            )
            log_event(root, job=job["name"], status="complete", output=str(path))
        except BaseException as error:
            log_event(root, job=job["name"], status="failed", output=str(path), error=str(error))
            raise
    from landcover.study_report import report

    return report(root, "val")


def test_study(root, allow_test=False):
    if not allow_test:
        raise ValueError(
            "Final test evaluation requires --allow-test after protocol/model selection freeze"
        )
    from landcover.study_report import report

    root = Path(root)
    plan = load_study(root)
    report(root, "val")  # Requires every planned validation run and freezes model selection.
    selection = root / "selection.json"
    receipt = root / "test-phase.json"
    expected = {"protocol_digest": digest(plan), "selection_sha256": file_hash(selection)}
    if receipt.exists() and json.loads(receipt.read_text()) != expected:
        raise ValueError("Test-phase protocol/selection changed")
    dump_json(receipt, expected)
    for job in plan["jobs"]:
        if job["kind"] not in plan["test_kinds"]:
            continue
        path = completed_attempt(root, job)
        print(f"Final test: {job['name']}", flush=True)
        evaluate_once(path, "test", allow_test=True)
    return report(root, "test")
