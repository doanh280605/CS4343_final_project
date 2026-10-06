"""Import unchanged team manifests and prepare a separate Turing CUDA study."""

import argparse
import hashlib
import json
import zipfile
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from landcover.config import Config
from landcover.download import download
from landcover.study import load_study, prepare_study


def import_manifests(bundle, destination=Path("data/prepared-ms-v2")):
    """Read only the two explicit files; refuse corrupt or different existing manifests."""
    with zipfile.ZipFile(bundle) as archive:
        metadata = json.loads(archive.read("bundle-provenance.json"))
        contents = {}
        for filename in ("manifest.json", "splits.json"):
            member = f"data/prepared-ms-v2/{filename}"
            contents[filename] = archive.read(member)
            if hashlib.sha256(contents[filename]).hexdigest() != metadata["files"][member]:
                raise ValueError(f"Bundle checksum mismatch: {member}")
    destination = Path(destination)
    for filename, content in contents.items():
        target = destination / filename
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"Existing manifest differs; not overwriting {target}")
    destination.mkdir(parents=True, exist_ok=True)
    for filename, content in contents.items():
        target = destination / filename
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(content)
    return metadata


def preserve_unfinished_setup(root):
    """Move only an unfrozen configs-only setup aside; never move training artifacts."""
    root = Path(root)
    if not root.exists():
        return None
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Expected a real study directory: {root}")
    if (root / "protocol.json").exists():
        return None  # Existing frozen studies must pass load_study, not be reset.
    if any(path.name != "configs" for path in root.iterdir()):
        raise ValueError(f"Unfrozen study contains more than setup configs; inspect {root}")
    configs = root / "configs"
    if configs.is_symlink() or (configs.exists() and not configs.is_dir()):
        raise ValueError(f"Unexpected setup config path: {configs}")
    if configs.exists() and any(
        p.is_symlink() or not p.is_file() or p.suffix != ".yaml" for p in configs.iterdir()
    ):
        raise ValueError(f"Unexpected files in setup configs; inspect {configs}")
    backup = root.with_name(f"{root.name}-unfinished-setup-{uuid4().hex}")
    root.rename(backup)
    print(f"Preserved unfinished setup at {backup}", flush=True)
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    metadata = import_manifests(args.bundle)
    print("Team manifests imported unchanged; downloading/verifying imagery.", flush=True)
    raw = download("ms", root=args.raw_root)
    candidates = [p.parent for p in raw.rglob("AnnualCrop") if p.is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one EuroSAT root, found {candidates}")
    target = Path("EuroSAT_MS")
    if target.exists() or target.is_symlink():
        if target.resolve() != candidates[0].resolve():
            raise ValueError("Existing EuroSAT_MS points to different imagery")
    else:
        target.symlink_to(candidates[0].resolve(), target_is_directory=True)

    # Cache official weights before reserving a GPU. This does not train a model.
    from torchvision.models import ResNet18_Weights

    ResNet18_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True)
    root = Path("outputs/turing-study")
    preserve_unfinished_setup(root)
    if not root.exists():
        print(
            "Auditing all imagery and freezing the study; this can take several minutes.",
            flush=True,
        )
        base = replace(Config.load("configs/final-study.yaml"), device="cuda", output=str(root))
        prepare_study(root, base)
    plan = load_study(root)
    if any(job["config"]["device"] != "cuda" for job in plan["jobs"]):
        raise ValueError("Expected a CUDA study")
    provenance = root / "data-bundle-provenance.json"
    if not provenance.exists():
        provenance.write_text(json.dumps(metadata, indent=2))
    print("Frozen CUDA study ready: 63 runs; no training launched.", flush=True)


if __name__ == "__main__":
    main()
