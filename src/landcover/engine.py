"""Step-budgeted training; validation selects checkpoints, test is a separate command."""

import csv
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from landcover.config import Config
from landcover.data import (
    BANDS,
    CLASSES,
    SatelliteDataset,
    digest,
    dump_json,
    fit_stats,
    load_prepared,
)
from landcover.metrics import plot_curves, save_predictions, score
from landcover.models import build_model


def seed_everything(seed, deterministic=True, threads=4):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def device_for(name):
    if name == "auto":
        # CPU is the deterministic portable fallback; MPS must be requested explicitly.
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def provenance():
    def git(*args):
        try:
            result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        except OSError:
            # Compute nodes and source bundles may have no Git executable.
            # Frozen studies independently retain and verify all source-file hashes.
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "git_commit": commit or "unavailable",
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "numpy", "scikit-learn", "rasterio")
        },
    }


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    all_ids, labels, probabilities = [], [], []
    loss_sum = 0.0
    for x, y, ids in loader:
        logits = model(x.to(device))
        loss_sum += nn.functional.cross_entropy(logits, y.to(device), reduction="sum").item()
        all_ids.extend(ids)
        labels.extend(y.tolist())
        probabilities.extend(logits.softmax(1).cpu().tolist())
    if not labels:
        raise ValueError("empty evaluation split")
    return all_ids, labels, np.asarray(probabilities), loss_sum / len(labels)


def build_optimizer(model, config):
    """Keep scratch rates uniform; optionally use a lower pretrained backbone rate."""
    if (
        config.model == "resnet18"
        and config.initialization == "pretrained"
        and config.pretrained_backbone_learning_rate is not None
    ):
        parameters = [
            {
                "params": [p for name, p in model.named_parameters() if not name.startswith("fc.")],
                "lr": config.pretrained_backbone_learning_rate,
                "name": "backbone",
            },
            {"params": model.fc.parameters(), "lr": config.learning_rate, "name": "head"},
        ]
    else:
        parameters = [{"params": model.parameters(), "name": "all"}]
    return torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)


def build_scheduler(optimizer, config):
    """Linear warm-up followed by cosine decay over the fixed optimizer budget."""
    if config.scheduler == "constant":
        return None

    def factor(step):
        if step < config.warmup_steps:
            return (step + 1) / config.warmup_steps
        progress = (step - config.warmup_steps) / (config.steps - config.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train(config):
    config.validate()
    seed_everything(config.seed, config.deterministic, config.threads)
    manifest, splits = load_prepared(config.manifest, config.splits)
    train_ids = splits["subsets"][str(float(config.fraction))]
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=False)
    config.save(output / "config.yaml")
    dump_json(output / "manifest.json", manifest)
    dump_json(output / "splits.json", splits)
    dataset_args = (config.manifest, train_ids, config.input_type, config.rgb_source)
    stats = fit_stats(SatelliteDataset(*dataset_args))
    dump_json(output / "normalization.json", stats)
    train_data = SatelliteDataset(*dataset_args, stats=stats, augment=config.augmentation)
    val_data = SatelliteDataset(
        config.manifest, splits["val"], config.input_type, config.rgb_source, stats=stats
    )
    generator = torch.Generator().manual_seed(config.seed)
    # Sampling with replacement makes every fraction use the exact same full-batch step budget.
    sampler = RandomSampler(
        train_data,
        replacement=True,
        num_samples=config.steps * config.batch_size,
        generator=generator,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config.seed + 1),
    )
    val_loader = DataLoader(val_data, batch_size=config.batch_size, num_workers=config.workers)
    train_eval_loader = None
    if config.measure_train_metrics:
        train_eval_loader = DataLoader(
            SatelliteDataset(*dataset_args, stats=stats, augment=False),
            batch_size=config.batch_size,
            num_workers=config.workers,
            generator=torch.Generator().manual_seed(config.seed + 2),
        )
    device = device_for(config.device)
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    run_info = {
        **provenance(),
        "device": str(device),
        "manifest_digest": digest(manifest),
        "split_digest": digest(splits),
        "train_ids_digest": digest(train_ids),
        "training_samples": len(train_ids),
        "optimizer_steps": config.steps,
        "examples_drawn": config.steps * config.batch_size,
        "effective_passes": config.steps * config.batch_size / len(train_ids),
    }
    dump_json(output / "provenance.json", run_info)
    history, best, interval_loss, interval_count = [], -1.0, 0.0, 0
    for step, (x, y, _) in enumerate(train_loader, 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device))
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        loss.backward()
        learning_rates = {f"lr_{group['name']}": group["lr"] for group in optimizer.param_groups}
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        interval_loss += loss.item()
        interval_count += 1
        if step % config.eval_every == 0 or step == config.steps:
            _, labels, probs, val_loss = predict(model, val_loader, device)
            val = score(labels, probs.argmax(1))
            row = {
                "step": step,
                "train_loss": interval_loss / interval_count,
                "val_loss": val_loss,
                "val_macro_f1": val["macro_f1"],
                "val_accuracy": val["accuracy"],
                **learning_rates,
            }
            if train_eval_loader is not None:
                _, train_labels, train_probs, train_eval_loss = predict(
                    model, train_eval_loader, device
                )
                train_metrics = score(train_labels, train_probs.argmax(1))
                row.update(
                    train_eval_loss=train_eval_loss,
                    train_macro_f1=train_metrics["macro_f1"],
                    train_accuracy=train_metrics["accuracy"],
                    macro_f1_gap=train_metrics["macro_f1"] - val["macro_f1"],
                )
            history.append(row)
            interval_loss, interval_count = 0.0, 0
            with (output / "learning_curves.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "config": asdict(config),
                "bands": manifest["bands"],
                "classes": manifest["classes"],
                "normalization": stats,
                "step": step,
                "val_macro_f1": val["macro_f1"],
                "provenance": run_info,
                "torch_rng_state": torch.get_rng_state(),
            }
            torch.save(checkpoint, output / "last.pt")
            if val["macro_f1"] > best:
                best = val["macro_f1"]
                torch.save(checkpoint, output / "best.pt")
            print(json.dumps(row), flush=True)
    plot_curves(history, output / "learning_curves.png")
    return output / "best.pt"


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = Config(**checkpoint["config"]).validate()
    if config.input_type == "ms" and checkpoint.get("bands") != BANDS:
        raise ValueError(
            "MS checkpoint band schema is missing or incompatible; retrain with v2 data"
        )
    if checkpoint.get("classes", CLASSES) != CLASSES:
        raise ValueError("checkpoint class schema mismatch")
    model = build_model(config, load_pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device), checkpoint, config


def evaluate(
    checkpoint_path,
    split="val",
    output=None,
    allow_test=False,
    manifest_path=None,
    splits_path=None,
):
    if split not in {"val", "test"}:
        raise ValueError("evaluation split must be val or test")
    if split == "test" and not allow_test:
        raise ValueError("test evaluation requires --allow-test after freezing model selection")
    model, checkpoint, config = load_checkpoint(checkpoint_path)
    seed_everything(config.seed, config.deterministic, config.threads)
    manifest_path = manifest_path or config.manifest
    splits_path = splits_path or config.splits
    manifest, splits = load_prepared(manifest_path, splits_path)
    if digest(manifest) != checkpoint["provenance"]["manifest_digest"]:
        raise ValueError("checkpoint/manifest mismatch")
    if digest(splits) != checkpoint["provenance"]["split_digest"]:
        raise ValueError("checkpoint/split mismatch")
    data = SatelliteDataset(
        manifest_path,
        splits[split],
        config.input_type,
        config.rgb_source,
        checkpoint["normalization"],
    )
    device = device_for(config.device)
    ids, labels, probs, loss = predict(
        model.to(device), DataLoader(data, config.batch_size), device
    )
    output = Path(output or Path(checkpoint_path).parent / f"evaluation-{split}")
    return save_predictions(
        output,
        ids,
        labels,
        probs,
        {
            "split": split,
            "loss": loss,
            "checkpoint_step": checkpoint["step"],
            "checkpoint": str(checkpoint_path),
            **checkpoint["provenance"],
        },
    )
