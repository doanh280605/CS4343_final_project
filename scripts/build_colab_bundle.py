"""Package current source and fixed manifests, excluding imagery and checkpoints."""

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "src/landcover").glob("*.py"))
    files += [
        root / name
        for name in (
            "pyproject.toml",
            "uv.lock",
            "configs/final-study.yaml",
            "configs/colab-requirements.txt",
            "data/prepared-ms-v2/manifest.json",
            "data/prepared-ms-v2/splits.json",
        )
    ]
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Expected regular source/data-manifest file: {path}")
    destination = root / "outputs/colab-setup"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "landcover-colab.zip"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    metadata = {
        "git_commit": commit,
        "note": "Includes current working-tree source, including uncommitted changes.",
        "files": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "environment_change": "New CUDA study; Mac study is preserved separately, not merged.",
    }
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(root))
        bundle.writestr("bundle-provenance.json", json.dumps(metadata, indent=2))
    print(f"{archive} ({archive.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
