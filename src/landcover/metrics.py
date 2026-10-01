"""Fixed-label metrics and noninteractive figures."""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from landcover.data import CLASSES, dump_json


def score(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=list(range(10)), average="macro", zero_division=0)
        ),
        "per_class": classification_report(
            labels,
            predictions,
            labels=list(range(10)),
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(10))).tolist(),
    }


def save_predictions(output, ids, labels, probabilities, metadata=None):
    output.mkdir(parents=True, exist_ok=False)
    probabilities = np.asarray(probabilities)
    predictions = probabilities.argmax(1)
    metrics = score(labels, predictions)
    metrics["provenance"] = metadata or {}
    dump_json(output / "metrics.json", metrics)
    with (output / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "label", "prediction"] + [f"p_{name}" for name in CLASSES])
        writer.writerows(
            [sid, int(y), int(pred), *map(float, prob)]
            for sid, y, pred, prob in zip(ids, labels, predictions, probabilities, strict=True)
        )
    fig, ax = plt.subplots(figsize=(10, 9))
    matrix = np.asarray(metrics["confusion_matrix"])
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(
        xticks=range(10),
        yticks=range(10),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted",
        ylabel="True",
    )
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    for i in range(10):
        for j in range(10):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
            )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return metrics


def plot_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    steps = [row["step"] for row in history]
    for key in ("train_loss", "val_loss"):
        axes[0].plot(steps, [row[key] for row in history], label=key)
    axes[1].plot(steps, [row["val_macro_f1"] for row in history], label="val_macro_f1")
    for ax in axes:
        ax.set_xlabel("Optimizer steps")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
