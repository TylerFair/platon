"""Small provenance checks; no PLATON/JAX imports or physics calculations."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from freeze_sources import OBSERVATIONS, freeze_snapshot, verify_snapshot


@pytest.fixture
def source_tree(tmp_path):
    platon = tmp_path / "platon_source"
    nautilus = tmp_path / "nautilus_source"
    for root, package in ((platon, "platon"), (nautilus, "nautilus")):
        (root / package).mkdir(parents=True)
        (root / package / "__init__.py").write_text("version = 1\n")
    (platon / "platon/_nautilus_performance.py").write_text("validated_fixture = True\n")
    (platon / "platon/data").mkdir()
    (platon / "platon/data/opacity.bin").write_bytes(b"external data")
    (platon / "platon/md5sum").write_text("fixture-checksum")
    driver = tmp_path / "driver.py"
    driver.write_text("DATA = None\nARCHIVE = None\n")
    data = tmp_path / "data"
    data.mkdir()
    for name in OBSERVATIONS:
        (data / name).write_text("{}\n" if name.endswith("json") else "value\n1\n")
    archive = tmp_path / "archive.csv"
    archive.write_text("pl_name,st_rad,pl_massj\nWASP-94 A b,1.62,0.456\nOther,1,1\n")
    yield {"platon_source": platon, "nautilus_source": nautilus,
           "driver": driver, "data": data, "archive": archive}
    # Snapshots are deliberately read-only; restore fixture permissions for cleanup.
    for path in tmp_path.rglob("*"):
        if not path.is_symlink():
            path.chmod(0o755 if path.is_dir() else 0o644)


def test_freeze_verify_and_cli_refuses_overwrite(tmp_path, source_tree):
    snapshot = tmp_path / "snapshot"
    manifest = freeze_snapshot(snapshot, **source_tree)
    assert verify_snapshot(snapshot) == manifest
    assert manifest["archive_selected_rows"] == 1
    assert "Other" not in (snapshot / "archive.csv").read_text()
    assert (snapshot / "platon/platon/md5sum").read_text() == "fixture-checksum"
    assert (snapshot / "platon/platon/data").is_symlink()
    assert (snapshot / "driver.py").stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        freeze_snapshot(snapshot, **source_tree)
    script = Path(__file__).with_name("freeze_sources.py")
    result = subprocess.run([sys.executable, str(script), str(snapshot), "--verify"],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["snapshot_id"] == manifest["snapshot_id"]


def test_source_and_external_data_mutation_detected(tmp_path, source_tree):
    snapshot = tmp_path / "snapshot"
    freeze_snapshot(snapshot, **source_tree)
    opacity = source_tree["platon_source"] / "platon/data/opacity.bin"
    opacity.write_bytes(b"changed external data")
    with pytest.raises(ValueError, match="External opacity"):
        verify_snapshot(snapshot)


def test_copied_code_mutation_detected(tmp_path, source_tree):
    snapshot = tmp_path / "snapshot"
    freeze_snapshot(snapshot, **source_tree)
    driver = snapshot / "driver.py"
    driver.chmod(0o644)
    driver.write_text("changed = True\n")
    with pytest.raises(ValueError, match="Snapshot file changed"):
        verify_snapshot(snapshot)


def test_partial_snapshot_is_removed_on_missing_target(tmp_path, source_tree):
    source_tree["archive"].write_text("pl_name,st_rad,pl_massj\nOther,1,1\n")
    snapshot = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="no exact WASP"):
        freeze_snapshot(snapshot, **source_tree)
    assert not snapshot.exists()
