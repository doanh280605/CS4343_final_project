"""Official Zenodo archives, publisher checksums, bounded extraction paths."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import requests

from landcover.data import dump_json

ARCHIVES = {
    "rgb": ("EuroSAT_RGB.zip", "f46e308c4d50d4bf32fedad2d3d62f3b"),
    "ms": ("EuroSAT_MS.zip", "091174add3c8e680a49244acf185b9f0"),
}


def extract_archive(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for item in zf.infolist():
            target = (destination / item.filename).resolve()
            if (
                not target.is_relative_to(destination)
                or (item.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError("unsafe archive member")
        zf.extractall(destination)


def download(kind, root="data/raw", archive=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name, expected_md5 = ARCHIVES[kind]
    url = f"https://zenodo.org/api/records/7711810/files/{name}/content"
    cached = Path(archive) if archive else root / name
    if not cached.exists():
        partial = cached.with_suffix(".partial")
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with partial.open("wb") as f:
                shutil.copyfileobj(response.raw, f)
        partial.rename(cached)
    with cached.open("rb") as f:
        actual = hashlib.file_digest(f, "md5").hexdigest()
    if actual != expected_md5:
        raise ValueError(f"archive checksum mismatch: {cached}; remove and re-download")
    output = root / kind
    marker = root / f"{kind}-download.json"
    if marker.exists():
        metadata = json.loads(marker.read_text())
        if metadata["md5"] != actual or not output.is_dir():
            raise ValueError("inconsistent extraction metadata")
        return output
    if output.exists():
        raise FileExistsError(f"incomplete extraction exists: {output}; inspect before retrying")
    extract_archive(cached, output)
    dump_json(
        marker,
        {
            "source": url,
            "md5": actual,
            "record": "7711810",
            "kind": kind,
            "license": "MIT (EuroSAT); Copernicus terms apply",
        },
    )
    return output
