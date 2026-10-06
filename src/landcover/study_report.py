"""Seed-level and paired study summaries; no partial grids disguised as final results."""

import csv
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from landcover.data import CLASSES, dump_json, file_hash
from landcover.study import SEEDS, completed_attempt, load_study


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    result = []
    for condition, group in sorted(groups.items()):
        if sorted(row["seed"] for row in group) != list(SEEDS):
            raise ValueError(f"Need exactly seeds 42/43/44 for {condition}")
        summary = {
            key: group[0][key]
            for key in (
                "condition",
                "kind",
                "model",
                "initialization",
                "input_type",
                "fraction",
                "training_samples",
                "augmentation",
                "dropout",
            )
        }
        summary["seeds"] = len(group)
        for metric in ("macro_f1", "accuracy", "validation_gap"):
            values = [row[metric] for row in group]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_sd"] = statistics.stdev(values)
        result.append(summary)
    return result


def paired_comparisons(rows):
    main = {
        (r["initialization"], r["input_type"], r["fraction"], r["seed"]): r
        for r in rows
        if r["kind"] == "main"
    }
    result = []
    for fraction in sorted({key[2] for key in main}):
        contrasts = []
        for modality in ("rgb", "ms"):
            contrasts.append(
                (
                    f"pretrained-minus-scratch-{modality}",
                    ("pretrained", modality),
                    ("scratch", modality),
                )
            )
        for initialization in ("pretrained", "scratch"):
            contrasts.append(
                (f"ms-minus-rgb-{initialization}", (initialization, "ms"), (initialization, "rgb"))
            )
        for name, positive, negative in contrasts:
            differences = [
                main[(*positive, fraction, seed)]["macro_f1"]
                - main[(*negative, fraction, seed)]["macro_f1"]
                for seed in SEEDS
            ]
            result.append(
                {
                    "comparison": name,
                    "fraction": fraction,
                    **{
                        f"seed_{seed}": difference
                        for seed, difference in zip(SEEDS, differences, strict=True)
                    },
                    "mean_difference": statistics.mean(differences),
                    "sample_sd": statistics.stdev(differences),
                }
            )
    return result


def aggregate_classes(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["class"])].append(row)
    result = []
    for (condition, name), group in sorted(groups.items()):
        if sorted(row["seed"] for row in group) != list(SEEDS):
            raise ValueError("Incomplete per-class seed coverage")
        if len({row["support"] for row in group}) != 1:
            raise ValueError("Class support changed between seeds")
        record = {"condition": condition, "class": name, "support": group[0]["support"]}
        for metric in ("precision", "recall", "f1-score"):
            values = [row[metric] for row in group]
            record[f"{metric}_mean"] = statistics.mean(values)
            record[f"{metric}_sd"] = statistics.stdev(values)
        result.append(record)
    return result


def read_rows(root, plan, split):
    rows, per_class, evidence = [], [], {}
    for job in plan["jobs"]:
        if split == "test" and job["kind"] not in plan["test_kinds"]:
            continue
        path = completed_attempt(root, job)
        if path is None:
            raise ValueError(f"Study incomplete: {job['name']}; finish every planned run first")
        receipt_path = path / f"verified-{split}.json"
        if not receipt_path.exists():
            raise ValueError(f"Missing verified {split} evaluation: {job['name']}")
        receipt = json.loads(receipt_path.read_text())
        if file_hash(path / "best.pt") != receipt["checkpoint_sha256"]:
            raise ValueError("Evaluated checkpoint changed")
        directory = path / receipt["directory"]
        for name, expected in receipt["files"].items():
            if file_hash(directory / name) != expected:
                raise ValueError(f"Evaluation artifact changed: {directory / name}")
        metrics = json.loads((directory / "metrics.json").read_text())
        if metrics["provenance"]["split"] != split:
            raise ValueError("Evaluation split mismatch")
        with (path / "learning_curves.csv").open() as stream:
            selected = max(csv.DictReader(stream), key=lambda r: float(r["val_macro_f1"]))
        config = job["config"]
        row = {
            "condition": job["name"].rsplit("-s", 1)[0],
            "kind": job["kind"],
            **{
                k: config[k]
                for k in (
                    "model",
                    "initialization",
                    "input_type",
                    "fraction",
                    "seed",
                    "augmentation",
                    "dropout",
                )
            },
            "training_samples": metrics["provenance"]["training_samples"],
            "best_step": int(selected["step"]),
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "validation_gap": float(selected["macro_f1_gap"]),
            "run": str(path),
            "evaluation": str(directory),
        }
        rows.append(row)
        for name in CLASSES:
            values = metrics["per_class"][name]
            per_class.append(
                {
                    "condition": row["condition"],
                    "seed": row["seed"],
                    "class": name,
                    **{key: values[key] for key in ("precision", "recall", "f1-score", "support")},
                }
            )
        evidence[job["name"]] = file_hash(receipt_path)
    return rows, per_class, evidence


def plots(directory, summaries, per_class, split):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, modality in zip(axes, ("rgb", "ms"), strict=True):
        for initialization in ("scratch", "pretrained"):
            subset = sorted(
                (
                    r
                    for r in summaries
                    if r["kind"] == "main"
                    and r["input_type"] == modality
                    and r["initialization"] == initialization
                ),
                key=lambda r: r["training_samples"],
            )
            axis.errorbar(
                [r["training_samples"] for r in subset],
                [r["macro_f1_mean"] for r in subset],
                yerr=[r["macro_f1_sd"] for r in subset],
                marker="o",
                capsize=4,
                label=initialization,
            )
        axis.set(xscale="log", xlabel="Training images", title=modality.upper(), ylim=(0, 1.02))
        axis.legend()
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(f"{split.capitalize()} macro-F1 (mean ± sample SD, 3 seeds)")
    fig.tight_layout()
    fig.savefig(directory / "data_efficiency.png", dpi=160)
    plt.close(fig)

    conditions = [r["condition"] for r in summaries if r["kind"] == "main" and r["fraction"] == 1]
    fig, axis = plt.subplots(figsize=(13, 5))
    for index, condition in enumerate(conditions):
        means, deviations = [], []
        for name in CLASSES:
            values = [
                r["f1-score"]
                for r in per_class
                if r["condition"] == condition and r["class"] == name
            ]
            means.append(statistics.mean(values))
            deviations.append(statistics.stdev(values))
        axis.bar(
            [i + (index - 1.5) * 0.2 for i in range(10)],
            means,
            width=0.2,
            yerr=deviations,
            capsize=2,
            label=condition,
        )
    axis.set(
        xticks=range(10),
        xticklabels=CLASSES,
        ylabel=f"{split.capitalize()} per-class F1",
        title="Full training subset: mean ± sample SD across 3 seeds",
        ylim=(0, 1.08),
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "per_class_f1.png", dpi=160)
    plt.close(fig)
    ablations = [r for r in summaries if r["kind"] == "ablation"]
    if ablations:
        fig, axis = plt.subplots(figsize=(9, 4))
        labels = [
            f"Augmentation {'on' if r['augmentation'] else 'off'}\nDropout {r['dropout']}"
            for r in ablations
        ]
        axis.bar(
            labels,
            [r["macro_f1_mean"] for r in ablations],
            yerr=[r["macro_f1_sd"] for r in ablations],
            capsize=4,
        )
        axis.set(
            ylabel="Validation macro-F1",
            title="Compact CNN at 10% data: mean ± sample SD",
            ylim=(0, 1.02),
        )
        fig.tight_layout()
        fig.savefig(directory / "regularization.png", dpi=160)
        plt.close(fig)


def report(root, split="val"):
    if split not in {"val", "test"}:
        raise ValueError("Report split must be val or test")
    root = Path(root)
    plan = load_study(root)
    if split == "test" and not (root / "test-phase.json").exists():
        raise ValueError("No explicit frozen test phase exists")
    rows, per_class, evidence = read_rows(root, plan, split)
    summaries = aggregate(rows)
    if split == "val":
        winner = sorted(
            (r for r in summaries if r["kind"] == "main"),
            key=lambda r: (-r["macro_f1_mean"], r["condition"]),
        )[0]
        representative = next(
            r for r in rows if r["condition"] == winner["condition"] and r["seed"] == 42
        )
        selection = {
            "condition": winner["condition"],
            "mean_validation_macro_f1": winner["macro_f1_mean"],
            "representative_seed": 42,
            "checkpoint": str(Path(representative["run"]) / "best.pt"),
            "validation_evidence": evidence,
        }
        path = root / "selection.json"
        if path.exists() and json.loads(path.read_text()) != selection:
            raise ValueError("Frozen validation selection changed")
        if not path.exists():
            dump_json(path, selection)
    directory = root / f"report-{split}"
    directory.mkdir(exist_ok=True)
    save_csv(directory / "seed_results.csv", rows)
    save_csv(directory / "summary.csv", summaries)
    save_csv(directory / "per_class.csv", per_class)
    save_csv(directory / "per_class_summary.csv", aggregate_classes(per_class))
    save_csv(directory / "paired_differences.csv", paired_comparisons(rows))
    dump_json(
        directory / "summary.json",
        {
            "split": split,
            "conditions": summaries,
            "note": "Sample SD across training seeds, not test-population uncertainty.",
        },
    )
    plots(directory, summaries, per_class, split)
    table = "".join(
        f"<tr><td>{html.escape(r['condition'])}</td><td>{r['macro_f1_mean']:.4f} ± "
        f"{r['macro_f1_sd']:.4f}</td><td>{r['accuracy_mean']:.4f}</td></tr>"
        for r in summaries
    )
    run_links = []
    for row in rows:
        name = html.escape(Path(row["run"]).parent.name)
        relative = f"../runs/{name}/{Path(row['run']).name}"
        evaluation = f"{relative}/{Path(row['evaluation']).name}/confusion_matrix.png"
        run_links.append(
            f"<li>{name}: <a href='{relative}/learning_curves.png'>curves</a> · "
            f"<a href='{evaluation}'>confusion matrix</a></li>"
        )
    ablation_image = (
        '<img src="regularization.png" alt="Regularization ablation">' if split == "val" else ""
    )
    (directory / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f"<title>EuroSAT {split} results</title><style>"
        "body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}"
        "table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;"
        "text-align:left}img{max-width:100%}</style>"
        f"<h1>EuroSAT {'validation' if split == 'val' else 'test'} results</h1>"
        "<p>All listed conditions include seeds 42, 43 and 44. "
        "Error bars show training-seed sample SD. "
        "Validation results support model selection; test results are for final reporting.</p>"
        '<p><a href="summary.csv">Summary CSV</a> · <a href="seed_results.csv">Every seed</a> · '
        '<a href="paired_differences.csv">Paired comparisons</a> · '
        '<a href="per_class_summary.csv">Per-class summary</a> · '
        '<a href="per_class.csv">Per-class seed results</a></p>'
        '<img src="data_efficiency.png" alt="Data efficiency comparison">'
        "<table><tr><th>Condition</th><th>Macro-F1 mean ± SD</th><th>Accuracy mean</th></tr>"
        f'{table}</table><img src="per_class_f1.png" alt="Per-class F1">{ablation_image}'
        f"<h2>Individual runs</h2><ul>{''.join(run_links)}</ul></html>"
    )
    return directory / "index.html"
