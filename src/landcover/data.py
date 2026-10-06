"""Stable sample identities, immutable splits, aligned modalities and train-only statistics."""

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
# EuroSAT TIFF storage order (TorchGeo EuroSAT.all_band_names), not numeric S2 order.
BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12", "B8A"]
RGB_INDICES = [3, 2, 1]
FRACTIONS = [1.0, 0.1, 0.05, 0.01]


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scan(root, suffixes):
    found = {}
    if root is None:
        return found
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in suffixes or path.parent.name not in CLASSES:
            continue
        sample_id = f"{path.parent.name}/{path.stem}"
        if sample_id in found:
            raise ValueError(f"duplicate sample ID: {sample_id}")
        found[sample_id] = path.resolve()
    if not found:
        raise ValueError(f"no class-organized samples found under {root}")
    return found


def prepare(output, rgb_root=None, ms_root=None, seed=2026, limit_per_class=None):
    """Never silently intersect modalities: missing pairs invalidate comparisons."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"use a new preparation directory: {output}")
    rgb, ms = scan(rgb_root, {".jpg", ".jpeg", ".png"}), scan(ms_root, {".tif", ".tiff"})
    if not rgb and not ms:
        raise ValueError("provide RGB and/or multispectral root")
    if rgb and ms and rgb.keys() != ms.keys():
        raise ValueError(
            f"RGB/MS IDs mismatch: RGB-only={len(rgb.keys() - ms.keys())}, "
            f"MS-only={len(ms.keys() - rgb.keys())}"
        )
    ids = sorted(rgb or ms)
    if limit_per_class is not None:
        if limit_per_class < 5:
            raise ValueError("limit_per_class must be at least 5")
        ids = [
            sid
            for name in CLASSES
            for sid in sorted(x for x in (rgb or ms) if x.startswith(name + "/"))[:limit_per_class]
        ]
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in ids:
        row = {"id": sid, "label": CLASSES.index(sid.split("/")[0])}
        for modality, paths in (("rgb", rgb), ("ms", ms)):
            row[modality] = os.path.relpath(paths[sid], output.resolve()) if paths else None
            row[f"{modality}_sha256"] = file_hash(paths[sid]) if paths else None
        rows.append(row)
    manifest = {
        "version": 2,
        "classes": CLASSES,
        "bands": BANDS,
        "samples": rows,
        "paired": bool(rgb and ms),
        "limited": limit_per_class is not None,
    }
    splits = make_splits(rows, seed)
    splits["manifest_digest"] = digest(manifest)
    dump_json(output / "manifest.json", manifest)
    dump_json(output / "splits.json", splits)
    return manifest, splits


def make_splits(rows, seed=2026):
    result = {"seed": seed, "train": [], "val": [], "test": [], "subsets": {}}
    for label in range(len(CLASSES)):
        ids = sorted(row["id"] for row in rows if row["label"] == label)
        if len(ids) < 5:
            raise ValueError("each class needs at least 5 samples for 60/20/20 splits")
        rng = np.random.default_rng(seed + label)
        ids = list(rng.permutation(ids))
        n_train, n_val = int(0.6 * len(ids)), int(0.2 * len(ids))
        result["train"] += ids[:n_train]
        result["val"] += ids[n_train : n_train + n_val]
        result["test"] += ids[n_train + n_val :]
    # The same class-wise permutation provides nested subsets across every model and run seed.
    for fraction in FRACTIONS:
        subset = []
        for label in range(len(CLASSES)):
            class_ids = [sid for sid in result["train"] if sid.startswith(CLASSES[label] + "/")]
            subset += class_ids[: max(1, math.floor(len(class_ids) * fraction))]
        result["subsets"][str(fraction)] = sorted(subset)
    for split in ("train", "val", "test"):
        result[split].sort()
    validate_splits(rows, result)
    return result


def validate_splits(rows, splits):
    ids = {r["id"] for r in rows}
    if len(ids) != len(rows):
        raise ValueError("duplicate manifest IDs")
    groups = [set(splits[k]) for k in ("train", "val", "test")]
    if any(len(splits[k]) != len(set(splits[k])) for k in ("train", "val", "test")):
        raise ValueError("duplicate split IDs")
    if set.union(*groups) != ids or any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("splits must be disjoint and cover manifest exactly")
    for fraction in FRACTIONS:
        subset = splits["subsets"][str(fraction)]
        if len(subset) != len(set(subset)) or not set(subset) <= groups[0]:
            raise ValueError("invalid training subset")
        subset_ids = set(subset)
        labels = {r["label"] for r in rows if r["id"] in subset_ids}
        if labels != set(range(10)):
            raise ValueError("training subset must contain all classes")
    if set(splits["subsets"]["1.0"]) != groups[0]:
        raise ValueError("100% subset differs from training split")
    for large, small in zip(FRACTIONS, FRACTIONS[1:], strict=False):
        if not set(splits["subsets"][str(small)]) <= set(splits["subsets"][str(large)]):
            raise ValueError("subsets must be nested")


def load_prepared(manifest_path, splits_path):
    manifest = json.loads(Path(manifest_path).read_text())
    splits = json.loads(Path(splits_path).read_text())
    if manifest["classes"] != CLASSES or manifest["bands"] != BANDS:
        raise ValueError("class/band schema mismatch")
    if digest(manifest) != splits["manifest_digest"]:
        raise ValueError("manifest changed after split generation")
    validate_splits(manifest["samples"], splits)
    return manifest, splits


class SatelliteDataset(Dataset):
    def __init__(
        self, manifest_path, ids, input_type="rgb", rgb_source="ms", stats=None, augment=False
    ):
        self.base = Path(manifest_path).resolve().parent
        manifest = json.loads(Path(manifest_path).read_text())
        by_id = {row["id"]: row for row in manifest["samples"]}
        self.rows = [by_id[sid] for sid in ids]
        self.input_type, self.rgb_source = input_type, rgb_source
        self.stats, self.augment = stats, augment

    def __len__(self):
        return len(self.rows)

    def raw(self, index):
        row = self.rows[index]
        if self.input_type == "ms" or self.rgb_source == "ms":
            if row["ms"] is None:
                raise ValueError("MS input missing; use rgb_source: jpg for JPEG-only data")
            with rasterio.open(self.base / row["ms"]) as src:
                if src.count != 13:
                    raise ValueError(f"expected 13 bands: {row['id']}")
                data = src.read(out_dtype="float32") / 10000.0
            if self.input_type == "rgb":
                data = data[RGB_INDICES]
        else:
            if row["rgb"] is None:
                raise ValueError("JPEG input missing")
            with Image.open(self.base / row["rgb"]) as img:
                data = np.asarray(img.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255
        if data.shape[1:] != (64, 64) or not np.isfinite(data).all():
            raise ValueError(f"expected finite 64x64 patch: {row['id']}, got {data.shape}")
        return torch.from_numpy(data.copy())

    def __getitem__(self, index):
        x = self.raw(index)
        if self.augment:
            if torch.rand(()) < 0.5:
                x = x.flip(-1)
            if torch.rand(()) < 0.5:
                x = x.flip(-2)
            x = torch.rot90(x, int(torch.randint(4, ()).item()), (-2, -1))
        if self.stats:
            mean = torch.tensor(self.stats["mean"])[:, None, None]
            std = torch.tensor(self.stats["std"])[:, None, None]
            x = (x - mean) / std
        return x, self.rows[index]["label"], self.rows[index]["id"]


def fit_stats(dataset):
    """Fit on the selected training subset only, before augmentation."""
    total, squared, pixels = None, None, 0
    for i in range(len(dataset)):
        x = dataset.raw(i).double().flatten(1)
        total = x.sum(1) if total is None else total + x.sum(1)
        squared = x.square().sum(1) if squared is None else squared + x.square().sum(1)
        pixels += x.shape[1]
    mean = total / pixels
    std = (squared / pixels - mean.square()).clamp_min(1e-12).sqrt()
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "sample_count": len(dataset),
        "sample_ids_digest": digest([row["id"] for row in dataset.rows]),
    }


def audit(manifest_path, splits_path):
    manifest, splits = load_prepared(manifest_path, splits_path)
    for modality, source in (("rgb", "jpg"), ("ms", "ms")):
        if all(row[modality] for row in manifest["samples"]):
            dataset = SatelliteDataset(
                manifest_path, [r["id"] for r in manifest["samples"]], modality, source
            )
            for i, row in enumerate(dataset.rows):
                if file_hash(dataset.base / row[modality]) != row[f"{modality}_sha256"]:
                    raise ValueError(f"sample content changed: {row['id']}")
                dataset.raw(i)
    return {
        "samples": len(manifest["samples"]),
        "paired": manifest["paired"],
        "splits": {k: len(splits[k]) for k in ("train", "val", "test")},
    }


def synthetic(root, per_class=10, seed=0):
    """Paired synthetic fixtures; never scientific evidence."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    for label, name in enumerate(CLASSES):
        for i in range(per_class):
            data = rng.integers(100, 8000, (13, 64, 64), dtype=np.uint16)
            data[label % 13] = 500 + label * 500
            for folder in ("rgb", "ms"):
                (root / folder / name).mkdir(parents=True, exist_ok=True)
            stem = f"{name}_{i + 1}"
            with rasterio.open(
                root / "ms" / name / f"{stem}.tif",
                "w",
                driver="GTiff",
                height=64,
                width=64,
                count=13,
                dtype="uint16",
                crs="EPSG:32645",
                transform=rasterio.transform.from_origin(300000, 3100000, 10, 10),
            ) as ds:
                ds.write(data)
            rgb = (data[RGB_INDICES].clip(0, 10000) / 10000 * 255).astype("uint8")
            Image.fromarray(rgb.transpose(1, 2, 0)).save(root / "rgb" / name / f"{stem}.jpg")
    return prepare(root / "prepared", root / "rgb", root / "ms")
