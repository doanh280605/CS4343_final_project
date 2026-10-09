"""Blind annotation, frozen transfer evaluation and reviewed change-map workflow."""

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from sklearn.metrics import f1_score

from landcover.data import BANDS, CLASSES, RGB_INDICES, dump_json, file_hash
from landcover.engine import load_checkpoint
from landcover.metrics import save_predictions
from landcover.nepal import change_map

SEED = 2026
FIELDS = [
    "sample_id",
    "period",
    "label",
    "labeler",
    "confidence",
    "dominant_fraction",
    "exclusion_reason",
    "notes",
]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_probabilities(record):
    ids = record["sample_ids"]
    values = np.asarray(record["probabilities"], dtype=float)
    if (
        len(set(ids)) != len(ids)
        or values.shape != (len(ids), 10)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.allclose(values.sum(1), 1, atol=1e-5)
    ):
        raise ValueError("invalid IDs or prediction probabilities")
    return values


def verify_models(rgb, ms, output):
    records = {}
    for name, path in (("rgb", rgb), ("ms", ms)):
        _, checkpoint, config = load_checkpoint(path)
        if (
            config.model != "resnet18"
            or config.initialization != "pretrained"
            or config.fraction != 1.0
            or config.seed != 42
            or config.input_type != name
            or config.rgb_source != "ms"
            or checkpoint.get("bands") != BANDS
            or checkpoint.get("classes") != CLASSES
        ):
            raise ValueError("require full-data pretrained seed-42 v2 ResNet checkpoints")
        channels = 3 if name == "rgb" else 13
        stats = checkpoint["normalization"]
        mean, std = np.asarray(stats["mean"]), np.asarray(stats["std"])
        if (
            mean.shape != (channels,)
            or std.shape != (channels,)
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or (std <= 0).any()
        ):
            raise ValueError("invalid frozen normalization")
        if Path(path).name != "best.pt":
            raise ValueError("supply the validation-selected best.pt from each original run")
        provenance = checkpoint.get("provenance", {})
        if any(
            not provenance.get(k)
            for k in ("manifest_digest", "split_digest", "train_ids_digest", "git_commit")
        ):
            raise ValueError("checkpoint training provenance is incomplete")
        history_path = Path(path).parent / "learning_curves.csv"
        with history_path.open() as stream:
            history = list(csv.DictReader(stream))
        best = max(history, key=lambda row: float(row["val_macro_f1"]))
        if int(best["step"]) != checkpoint.get("step") or not np.isclose(
            float(best["val_macro_f1"]), checkpoint.get("val_macro_f1")
        ):
            raise ValueError("best.pt does not match first maximum validation macro-F1")
        records[name] = {
            "path": str(Path(path).resolve()),
            "sha256": file_hash(path),
            "normalization": stats,
            "config": checkpoint["config"],
            "bands": BANDS,
            "classes": CLASSES,
            "provenance": provenance,
            "step": checkpoint["step"],
            "val_macro_f1": checkpoint["val_macro_f1"],
            "history_sha256": file_hash(history_path),
        }
    if any(
        records["rgb"]["provenance"][k] != records["ms"]["provenance"][k]
        for k in ("manifest_digest", "split_digest", "train_ids_digest")
    ):
        raise ValueError("models must share the full-data manifest, split and training IDs")
    if records["rgb"]["sha256"] == records["ms"]["sha256"]:
        raise ValueError("RGB and MS checkpoints must differ")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dump_json(output / "models.json", records)
    return records


def annotation(patch_manifest, output):
    manifest = read_json(patch_manifest)
    rows = sorted(manifest["samples"], key=lambda r: (r["row"], r["col"]))
    ids = [r["sample_id"] for r in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("empty or duplicate patch IDs")
    rng = np.random.default_rng(SEED)
    # One random selection per ordered geographic stratum spans the valid raster.
    count = min(300, len(rows))
    chosen = [rows[int(rng.choice(group))] for group in np.array_split(np.arange(len(rows)), count)]
    post_ids = set(rng.choice([r["sample_id"] for r in chosen], min(100, count), replace=False))
    assignments = [(r, "pre") for r in chosen] + [
        (r, "post") for r in chosen if r["sample_id"] in post_ids
    ]
    double = set()
    for period in ("pre", "post"):
        positions = [i for i, (_, p) in enumerate(assignments) if p == period]
        double.update(rng.choice(positions, int(np.ceil(len(positions) * 0.2)), replace=False))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    template, second = [], []
    for index, (row, period) in enumerate(assignments):
        entry = dict.fromkeys(FIELDS, "")
        entry.update(sample_id=row["sample_id"], period=period)
        template.append(entry)
        if index in double:
            second.append(entry.copy())
        with rasterio.open(Path(patch_manifest).parent / row[period]) as src:
            image = src.read()[RGB_INDICES].transpose(1, 2, 0)
        low, high = np.percentile(image, [2, 98])
        image = np.clip((image - low) / max(high - low, 1), 0, 1)
        plt.imsave(output / f"{row['sample_id']}-{period}.png", image)
    write_csv(output / "labels-primary.csv", FIELDS, template)
    write_csv(output / "labels-second.csv", FIELDS, second)
    dump_json(
        output / "assignment.json",
        {
            "patch_manifest_sha256": file_hash(patch_manifest),
            "seed": SEED,
            "pre_target": 300,
            "post_target": 100,
            "pre_actual": count,
            "post_actual": len(post_ids),
            "shortfall": {"pre": 300 - count, "post": 100 - len(post_ids)},
            "keys": [[r["sample_id"], p] for r, p in assignments],
            "double_keys": [
                [assignments[i][0]["sample_id"], assignments[i][1]] for i in sorted(double)
            ],
        },
    )
    return len(assignments)


def read_labels(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        key = (row["sample_id"], row["period"])
        if key in result or row["period"] not in {"pre", "post"}:
            raise ValueError("duplicate/invalid annotation key")
        if not row.get("labeler") or not row.get("confidence"):
            raise ValueError("every annotation needs labeler and confidence")
        if row.get("label"):
            if (
                row["label"] not in CLASSES
                or row.get("exclusion_reason")
                or not 0.7 <= float(row.get("dominant_fraction") or 0) <= 1
            ):
                raise ValueError("labels require a supported class and dominant fraction >=0.7")
        elif row.get("exclusion_reason") not in {"mixed", "unsupported", "uncertain", "cloud"}:
            raise ValueError("unlabeled patches need an explicit exclusion reason")
        result[key] = row
    return result


def freeze_labels(assignment, primary, second, final, output):
    plan = read_json(assignment)
    a, b, resolved = [read_labels(p) for p in (primary, second, final)]
    expected = {tuple(k) for k in plan["keys"]}
    doubles = {tuple(k) for k in plan["double_keys"]}
    if set(a) != expected or set(resolved) != expected or set(b) != doubles:
        raise ValueError("annotations must cover exactly the predeclared assignments")
    disagreements = []
    for key in sorted(doubles):
        if a[key]["labeler"] == b[key]["labeler"]:
            raise ValueError("double annotation requires independent labelers")
        if (a[key]["label"], a[key]["exclusion_reason"]) != (
            b[key]["label"],
            b[key]["exclusion_reason"],
        ):
            disagreements.append(list(key))
            if not resolved[key].get("notes"):
                raise ValueError("document adjudication of disagreements in final notes")
    for key in expected:
        if resolved[key] != a[key] and not resolved[key].get("notes"):
            raise ValueError("document every adjudicated annotation change")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "labels.csv", FIELDS, [resolved[k] for k in sorted(resolved)])
    dump_json(
        output / "freeze.json",
        {
            "labels_sha256": file_hash(output / "labels.csv"),
            "patch_manifest_sha256": plan["patch_manifest_sha256"],
            "assignment_sha256": file_hash(assignment),
            "primary_sha256": file_hash(primary),
            "second_sha256": file_hash(second),
            "disagreements": disagreements,
            "counts": plan,
            "rule": "Independent labels frozen before evaluation; no Nepal tuning.",
        },
    )


def compare(rgb_predictions, ms_predictions, frozen, models, output):
    frozen = Path(frozen)
    freeze = read_json(frozen / "freeze.json")
    if file_hash(frozen / "labels.csv") != freeze["labels_sha256"]:
        raise ValueError("frozen labels changed")
    labels = read_labels(frozen / "labels.csv")
    records = [read_json(p) for p in (rgb_predictions, ms_predictions)]
    model_records = read_json(models)
    probabilities = [validate_probabilities(r) for r in records]
    rgb, ms = records
    if (
        rgb["sample_ids"] != ms["sample_ids"]
        or rgb["period"] != ms["period"]
        or rgb["period"] not in {"pre", "post"}
        or any(r["patch_manifest_sha256"] != freeze["patch_manifest_sha256"] for r in records)
        or any(
            r["checkpoint_sha256"] != model_records[name]["sha256"]
            for name, r in zip(("rgb", "ms"), records, strict=True)
        )
    ):
        raise ValueError("paired model, period or patch provenance mismatch")
    period = rgb["period"]
    eligible = {sid: row for (sid, p), row in labels.items() if p == period and row["label"]}
    if not eligible or not set(eligible).issubset(rgb["sample_ids"]):
        raise ValueError("missing independently labeled evaluation patches")
    selected = [i for i, sid in enumerate(rgb["sample_ids"]) if sid in eligible]
    ids = [rgb["sample_ids"][i] for i in selected]
    y = [CLASSES.index(eligible[sid]["label"]) for sid in ids]
    represented = sorted(set(y))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summaries, predictions = {}, []
    for name, values in zip(("rgb", "ms"), probabilities, strict=True):
        values = values[selected]
        pred = values.argmax(1)
        predictions.append(pred)
        metric = save_predictions(
            output / name,
            ids,
            y,
            values,
            {
                "domain": "Nepal",
                "period": period,
                "freeze_sha256": file_hash(frozen / "freeze.json"),
                "checkpoint_sha256": model_records[name]["sha256"],
            },
        )
        metric["represented_class_macro_f1"] = float(
            f1_score(y, pred, labels=represented, average="macro", zero_division=0)
        )
        metric["represented_classes"] = [CLASSES[i] for i in represented]
        dump_json(output / name / "metrics.json", metric)
        summaries[name] = metric
    write_csv(
        output / "paired.csv",
        ["sample_id", "label", "rgb", "ms"],
        [
            dict(sample_id=sid, label=CLASSES[label], rgb=CLASSES[int(a)], ms=CLASSES[int(b)])
            for sid, label, a, b in zip(ids, y, *predictions, strict=True)
        ],
    )
    summaries["coverage"] = {
        "eligible": len(ids),
        "assigned": sum(p == period for _, p in labels),
        "excluded": sum(p == period and not r["label"] for (_, p), r in labels.items()),
    }
    summaries["coverage"]["exclusion_reasons"] = dict(
        Counter(
            r["exclusion_reason"] for (_, p), r in labels.items() if p == period and not r["label"]
        )
    )
    correct = [pred == np.asarray(y) for pred in predictions]
    summaries["paired_correctness"] = {
        "both_correct": int((correct[0] & correct[1]).sum()),
        "rgb_only_correct": int((correct[0] & ~correct[1]).sum()),
        "ms_only_correct": int((~correct[0] & correct[1]).sum()),
        "both_incorrect": int((~correct[0] & ~correct[1]).sum()),
    }
    summaries["ms_minus_rgb"] = {
        key: summaries["ms"][key] - summaries["rgb"][key]
        for key in ("accuracy", "macro_f1", "represented_class_macro_f1")
    }
    dump_json(output / "summary.json", summaries)
    return summaries["coverage"]


def validate_change_inputs(pre, post, patches, qa, models):
    quality = read_json(qa)
    if (
        quality.get("patch_manifest_sha256") != file_hash(patches)
        or quality.get("imagery_review_passed") is not True
        or not quality.get("reviewer")
        or not quality.get("reviewed_on")
    ):
        raise ValueError("complete imagery QA for the current patch manifest first")
    manifest = read_json(patches)
    landmarks = quality.get("landmarks", [])
    if (
        len(landmarks) < 10
        or len({p["id"] for p in landmarks}) != len(landmarks)
        or len({(p.get("row"), p.get("col")) for p in landmarks}) != len(landmarks)
        or any(
            not isinstance(p.get("row"), (int, float))
            or not isinstance(p.get("col"), (int, float))
            or not 0 <= p["row"] < manifest["height"]
            or not 0 <= p["col"] < manifest["width"]
            or not p.get("notes")
            or not p.get("reference_source")
            or not np.isfinite(p["residual_pixels"])
            or not 0 <= p["residual_pixels"] <= 1
            for p in landmarks
        )
    ):
        raise ValueError("need ten distinct inspected landmarks with residual <=1 pixel")
    validate_frozen_change_model(pre, post, models)


def validate_frozen_change_model(pre, post, models):
    verified = read_json(models)["ms"]
    if verified["config"]["input_type"] != "ms":
        raise ValueError("change study requires the verified primary 13-band model")
    for path in (pre, post):
        record = read_json(path)
        validate_probabilities(record)
        if record["checkpoint_sha256"] != verified["sha256"]:
            raise ValueError("change predictions must use the verified primary 13-band model")


def checked_change(pre, post, patches, qa, models, output, threshold=0.7, exclusions=None):
    validate_change_inputs(pre, post, patches, qa, models)
    change_map(pre, post, patches, output, threshold, exclusions)
    dump_json(
        Path(output) / "review-provenance.json",
        {
            "qa_sha256": file_hash(qa),
            "models_sha256": file_hash(models),
        },
    )


def sensitivity(pre, post, patches, qa, models, output, exclusions=None):
    validate_change_inputs(pre, post, patches, qa, models)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = []
    for threshold in (0.6, 0.7, 0.8):
        target = output / f"threshold-{threshold}"
        change_map(pre, post, patches, target, threshold, exclusions)
        summary.append(read_json(target / "transitions.json"))
    dump_json(output / "sensitivity.json", summary)
    manifest = read_json(patches)
    before, after = [validate_probabilities(read_json(p)) for p in (pre, post)]
    excluded = set(read_json(exclusions)["sample_ids"]) if exclusions else set()
    rng = np.random.default_rng(SEED)
    reviews = []
    for changed in (True, False):
        candidates = [
            i
            for i, r in enumerate(manifest["samples"])
            if r["sample_id"] not in excluded
            and min(before[i].max(), after[i].max()) >= 0.7
            and bool(before[i].argmax() != after[i].argmax()) == changed
        ]
        for i in rng.choice(candidates, min(30, len(candidates)), replace=False):
            row = manifest["samples"][i]
            reviews.append(
                dict(
                    sample_id=row["sample_id"],
                    candidate_change=changed,
                    reviewer="",
                    reference_source="",
                    conclusion="",
                    notes="",
                )
            )
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            for ax, period in zip(axes, ("pre", "post"), strict=True):
                with rasterio.open(Path(patches).parent / row[period]) as src:
                    image = src.read()[RGB_INDICES].transpose(1, 2, 0)
                ax.imshow(np.clip(image / 3000, 0, 1))
                ax.set_title(period)
                ax.axis("off")
            fig.savefig(output / f"{row['sample_id']}.png", dpi=120)
            plt.close(fig)
    write_csv(
        output / "review.csv",
        ["sample_id", "candidate_change", "reviewer", "reference_source", "conclusion", "notes"],
        reviews,
    )
    dump_json(
        output / "review-assignment.json",
        {
            "patch_manifest_sha256": file_hash(patches),
            "seed": SEED,
            "sample_ids": [r["sample_id"] for r in reviews],
            "pre_predictions_sha256": file_hash(pre),
            "post_predictions_sha256": file_hash(post),
            "qa_sha256": file_hash(qa),
            "models_sha256": file_hash(models),
            "status": "candidate maps; human review pending",
        },
    )
    return summary


def review_exclusions(assignment, review_csv, output):
    """Freeze completed visual review and preserve unsupported/uncertain patches as unknown."""
    plan = read_json(assignment)
    with Path(review_csv).open() as stream:
        rows = list(csv.DictReader(stream))
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(plan["sample_ids"]):
        raise ValueError("review must cover exactly the assigned IDs")
    allowed = {"possible_change", "unchanged", "unsupported", "uncertain", "cloud"}
    for row in rows:
        if (
            not row["reviewer"]
            or not row["reference_source"]
            or row["conclusion"] not in allowed
            or not row["notes"]
        ):
            raise ValueError("complete reviewer, independent reference, conclusion and notes")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dump_json(
        output / "exclusions.json",
        {
            "patch_manifest_sha256": plan["patch_manifest_sha256"],
            "sample_ids": [
                r["sample_id"]
                for r in rows
                if r["conclusion"] in {"unsupported", "uncertain", "cloud"}
            ],
            "review_assignment_sha256": file_hash(assignment),
            "review_sha256": file_hash(review_csv),
            "reviewed_count": len(rows),
        },
    )


def demonstration(pre, post, patches, models, output):
    """An explicitly unvalidated model demonstration, separate from reviewed research maps."""
    validate_frozen_change_model(pre, post, models)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = []
    warning = (
        "UNVALIDATED MODEL DEMONSTRATION. Human imagery, registration and change review "
        "are incomplete. No Nepal accuracy or F1 is established. Predicted differences "
        "may reflect misregistration, seasonal differences or classification errors; "
        "they do not establish flood damage."
    )
    for threshold in (0.6, 0.7, 0.8):
        target = output / f"threshold-{threshold}"
        change_map(pre, post, patches, target, threshold)
        raster = target / "possible_changes.tif"
        raster.rename(target / "unvalidated_model_changes.tif")
        with rasterio.open(target / "unvalidated_model_changes.tif", "r+") as dst:
            dst.update_tags(validation_status="unvalidated_demonstration", description=warning)
            array = dst.read(1)
        record = read_json(target / "transitions.json")
        record.update(
            validation_status="unvalidated_demonstration",
            interpretation=warning,
            imagery_review_passed=False,
            registration_review_passed=False,
            human_change_review_completed=False,
            reference_labels_frozen=False,
            accuracy_available=False,
            models_sha256=file_hash(models),
        )
        dump_json(target / "transitions.json", record)
        summary.append(record)
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        fig, ax = plt.subplots(figsize=(7, 8))
        display = np.where(array == 255, 2, array)
        ax.imshow(
            display,
            cmap=ListedColormap(["#306e9f", "#dc7c32", "#e5e5e5"]),
            vmin=0,
            vmax=2,
            interpolation="nearest",
        )
        ax.set_title(
            f"Unvalidated model differences — threshold {threshold}\n"
            "Not a verified flood-damage map"
        )
        ax.axis("off")
        ax.legend(
            handles=[
                Patch(color=color, label=label)
                for color, label in [
                    ("#306e9f", "same prediction"),
                    ("#dc7c32", "different prediction"),
                    ("#e5e5e5", "unknown / excluded"),
                ]
            ],
            loc="lower left",
        )
        fig.savefig(target / "unvalidated_model_changes.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    dump_json(
        output / "demonstration.json",
        {
            "status": "unvalidated_demonstration",
            "disclosure": warning,
            "thresholds": summary,
            "patch_manifest_sha256": file_hash(patches),
            "models_sha256": file_hash(models),
        },
    )
    return {"status": "unvalidated_demonstration", "thresholds": [0.6, 0.7, 0.8]}
