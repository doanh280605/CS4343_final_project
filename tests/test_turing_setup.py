import hashlib
import json
import runpy
import zipfile
from pathlib import Path

import pytest

import_manifests = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/turing_prepare.py")
)["import_manifests"]


def write_bundle(path, contents, corrupt=False):
    hashes = {
        f"data/prepared-ms-v2/{name}": hashlib.sha256(content).hexdigest()
        for name, content in contents.items()
    }
    if corrupt:
        hashes["data/prepared-ms-v2/splits.json"] = "incorrect"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("bundle-provenance.json", json.dumps({"files": hashes}))
        for name, content in contents.items():
            archive.writestr(f"data/prepared-ms-v2/{name}", content)
        archive.writestr("../../unexpected.txt", "must not extract other members")


def test_team_manifest_import_preserves_bytes_and_rejects_replacement(tmp_path):
    bundle = tmp_path / "bundle.zip"
    contents = {"manifest.json": b'{"fixture": "manifest"}', "splits.json": b'{"seed": 2026}'}
    write_bundle(bundle, contents)
    destination = tmp_path / "prepared"
    import_manifests(bundle, destination)
    import_manifests(bundle, destination)
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == contents
    assert not (tmp_path / "unexpected.txt").exists()
    other = {**contents, "splits.json": b'{"seed": 42}'}
    write_bundle(bundle, other)
    with pytest.raises(ValueError, match="not overwriting"):
        import_manifests(bundle, destination)
    assert (destination / "splits.json").read_bytes() == contents["splits.json"]


def test_team_manifest_hash_failure_does_not_publish_partial_preparation(tmp_path):
    bundle = tmp_path / "bundle.zip"
    write_bundle(bundle, {"manifest.json": b"{}", "splits.json": b"{}"}, corrupt=True)
    destination = tmp_path / "prepared"
    with pytest.raises(ValueError, match="checksum mismatch"):
        import_manifests(bundle, destination)
    assert not destination.exists()
