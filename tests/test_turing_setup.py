import hashlib
import json
import runpy
import zipfile
from pathlib import Path

import pytest

setup_functions = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/turing_prepare.py")
)
import_manifests = setup_functions["import_manifests"]
preserve_unfinished_setup = setup_functions["preserve_unfinished_setup"]


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


def test_unfinished_setup_preserved_without_losing_configs(tmp_path):
    root = tmp_path / "study"
    (root / "configs").mkdir(parents=True)
    (root / "configs/run.yaml").write_bytes(b"original config bytes")
    backup = preserve_unfinished_setup(root)
    assert not root.exists()
    assert (backup / "configs/run.yaml").read_bytes() == b"original config bytes"
    assert preserve_unfinished_setup(root) is None
    root.mkdir()
    (root / "protocol.json").write_text("{}")
    assert preserve_unfinished_setup(root) is None
    assert (root / "protocol.json").exists()


@pytest.mark.parametrize("artifact", ["runs/job/best.pt", "protocol-digest.json"])
def test_incomplete_study_with_non_setup_artifacts_is_not_moved(tmp_path, artifact):
    root = tmp_path / "study"
    path = root / artifact
    path.parent.mkdir(parents=True)
    path.write_bytes(b"must preserve in place")
    with pytest.raises(ValueError, match="more than setup configs"):
        preserve_unfinished_setup(root)
    assert path.read_bytes() == b"must preserve in place"


def test_retry_failed_preparation_on_node_without_git(paired, tmp_path, monkeypatch):
    from landcover import engine, study
    from landcover.config import Config

    root = tmp_path / "study"
    config = Config(
        manifest=str(paired / "prepared/manifest.json"),
        splits=str(paired / "prepared/splits.json"),
        measure_train_metrics=True,
        device="cuda",
    )

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    # Reproduce the original failure after configs were written, before a protocol existed.
    monkeypatch.setattr(study, "provenance", no_git)
    with pytest.raises(FileNotFoundError):
        study.prepare_study(root, config)
    backup = preserve_unfinished_setup(root)
    assert len(list((backup / "configs").glob("*.yaml"))) == 63

    monkeypatch.setattr(study, "provenance", engine.provenance)
    monkeypatch.setattr(engine.subprocess, "run", no_git)
    assert study.prepare_study(root, config) == 63
    plan = study.load_study(root)
    assert plan["provenance"]["git_commit"] == "unavailable"
    assert plan["source_sha256"] == study.source_hashes()
