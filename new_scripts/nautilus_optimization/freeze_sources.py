"""Freeze the code and observations for matched WASP-94 Nautilus experiments.

Only the large opacity directory remains external. Its link destination and
per-file stat metadata are checked on verification; those metadata are not
cryptographic content hashes. Source and observation files are SHA256 checked.
The output directory must be new and becomes read-only after creation.
"""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PLATON = HERE / "best" if (HERE / "best/platon").is_dir() else HERE.parents[1]
DEFAULT_DRIVER = Path("/scratch/midway3/tfairnington/time_shift/wasp94_eq_guillot/run_retrieval.py")
DEFAULT_NAUTILUS = Path("/scratch/midway3/tfairnington/speedup/nautilus")
DEFAULT_DATA = Path("/scratch/midway3/tfairnington/time_shift/wasp94_delta_preview")
DEFAULT_ARCHIVE = Path("/project/ekempton/tfairnington/JWST/03_trexolists_extended_with_nasa_ps_filled.csv")
OBSERVATIONS = ("soss_o1.csv", "g395h_nrs1.csv", "g395h_nrs2.csv", "whitelight.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path):
    return {"sha256": sha256(path), "size": Path(path).stat().st_size}


def _stat_record(path):
    value = Path(path).stat()
    # st_dev identifies a local mount, not shared-file identity across nodes.
    return {"size": value.st_size, "mtime_ns": value.st_mtime_ns,
            "inode": value.st_ino}


def _external_data_record(data_root):
    root = Path(data_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Opacity data is not a directory: {root}")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() and path.is_dir():
            raise ValueError(f"Nested directory symlink requires explicit freezing: {path}")
        if path.is_file():
            files[str(path.relative_to(root))] = _stat_record(path)
    return {"target": str(root), "directory_stat": _stat_record(root),
            "files": files, "content_hashed": False}


def _git_record(root):
    def command(*args):
        return subprocess.run(["git", "-C", str(root), *args],
                              check=True, capture_output=True).stdout
    try:
        revision = command("rev-parse", "HEAD").decode().strip()
        status = command("status", "--porcelain=v1").decode()
        diff = command("diff", "--binary", "HEAD", "--")
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"path": str(root), "available": False}
    return {"path": str(root), "available": True, "revision": revision,
            "status": status, "dirty": bool(status),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest()}


def _copy_stable(source, destination):
    before = _file_record(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if _file_record(source) != before or _file_record(destination) != before:
        raise RuntimeError(f"Source changed during snapshot: {source}")


def _source_files(package):
    return sorted(path for path in package.rglob("*.py")
                  if "__pycache__" not in path.parts
                  and "data" not in path.relative_to(package).parts)


def _identity(manifest):
    value = {"files": manifest["files"], "external_data": manifest["external_data"]}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_snapshot(snapshot):
    """Return the manifest after verifying code/data hashes and opacity metadata."""
    snapshot = Path(snapshot).resolve(strict=True)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported snapshot manifest schema")
    if manifest.get("snapshot_id") != _identity(manifest):
        raise ValueError("Snapshot manifest identity is inconsistent")
    for relative, expected in manifest["files"].items():
        name = Path(relative)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"Unsafe manifest filename: {relative}")
        path = snapshot / name
        if path.is_symlink() or not path.is_file() or _file_record(path) != expected:
            raise ValueError(f"Snapshot file changed or missing: {relative}")
    expected_names = set(manifest["files"])
    actual_names = {str(path.relative_to(snapshot)) for path in snapshot.rglob("*")
                    if path.is_file() and not path.is_symlink()
                    and "__pycache__" not in path.parts
                    and path.name != "manifest.json"}
    if actual_names != expected_names:
        raise ValueError(f"Snapshot contains unexpected files: {sorted(actual_names - expected_names)}")
    links = {str(path.relative_to(snapshot)) for path in snapshot.rglob("*") if path.is_symlink()}
    if links != {"platon/platon/data"}:
        raise ValueError(f"Unexpected snapshot symlinks: {sorted(links)}")
    data_link = snapshot / "platon/platon/data"
    if not data_link.is_symlink():
        raise ValueError("Expected an explicit external opacity data symlink")
    observed = _external_data_record(data_link)
    if observed != manifest["external_data"]:
        raise ValueError("External opacity data destination or file metadata changed")
    return manifest


def freeze_snapshot(output, *, platon_source=DEFAULT_PLATON,
                    nautilus_source=DEFAULT_NAUTILUS, driver=DEFAULT_DRIVER,
                    data=DEFAULT_DATA, archive=DEFAULT_ARCHIVE):
    """Create a new immutable snapshot, including validated performance hooks."""
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Snapshot output already exists: {output}")
    output = output.resolve()
    platon_source, nautilus_source = Path(platon_source).resolve(), Path(nautilus_source).resolve()
    driver, data, archive = Path(driver).resolve(), Path(data).resolve(), Path(archive).resolve()
    performance = platon_source / "platon/_nautilus_performance.py"
    if not performance.is_file():
        raise ValueError(f"Integrate and validate performance hooks before freezing: {performance}")
    packages = {"platon": platon_source, "nautilus": nautilus_source}
    for package, root in packages.items():
        if not (root / package / "__init__.py").is_file():
            raise ValueError(f"Missing {package} package: {root}")
    external = _external_data_record(platon_source / "platon/data")
    md5sum = platon_source / "platon/md5sum"
    if not md5sum.is_file():
        md5sum = Path(external["target"]).parent / "md5sum"
    if not md5sum.is_file():
        raise ValueError("PLATON needs its package/md5sum alongside the data link")
    git = {package: _git_record(root) for package, root in packages.items()}
    originals = [driver, archive, md5sum, *(data / name for name in OBSERVATIONS)]
    source_records = {str(path): _file_record(path) for path in originals}
    package_lists = {package: _source_files(root / package) for package, root in packages.items()}
    package_records = {str(path): _file_record(path)
                       for paths in package_lists.values() for path in paths}
    if any(output == root or root in output.parents for root in (platon_source / "platon", nautilus_source / "nautilus", Path(external["target"]))):
        raise ValueError("Snapshot output must be outside source packages and opacity data")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Exclusive reservation: never reuse or overwrite an experiment.
    try:
        for package, root in packages.items():
            for path in package_lists[package]:
                _copy_stable(path, output / package / path.relative_to(root))
        _copy_stable(md5sum, output / "platon/platon/md5sum")
        (output / "platon/platon/data").symlink_to(external["target"], target_is_directory=True)
        _copy_stable(driver, output / "driver.py")
        for name in OBSERVATIONS:
            _copy_stable(data / name, output / "data" / name)
        with archive.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = [row for row in reader if row.get("pl_name") == "WASP-94 A b"]
        if not rows:
            raise ValueError("Archive has no exact WASP-94 A b records")
        with (output / "archive.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        for package, root in packages.items():
            if _source_files(root / package) != package_lists[package] or _git_record(root) != git[package]:
                raise RuntimeError(f"{package} source changed while snapshotting")
        if any(_file_record(path) != record for path, record in {**source_records, **package_records}.items()):
            raise RuntimeError("An input source changed while snapshotting")
        if _external_data_record(external["target"]) != external:
            raise RuntimeError("Opacity data changed while snapshotting")
        files = {str(path.relative_to(output)): _file_record(path)
                 for path in sorted(output.rglob("*")) if path.is_file() and not path.is_symlink()}
        manifest = {
            "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "files": files, "git": git, "source_inputs": source_records,
            "external_data": external, "archive_selected_rows": len(rows),
            "target": "WASP-94 A b; latest supplied driver, unchanged priors",
            "notes": ["External opacity arrays are linked and stat checked, not copied or content hashed.",
                      "Source file hashes identify effective code, including uncommitted changes.",
                      "Run outputs and checkpoints belong outside this read-only snapshot."],
        }
        manifest["snapshot_id"] = _identity(manifest)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        verify_snapshot(output)
    except Exception:
        shutil.rmtree(output)
        raise
    for path in output.rglob("*"):
        if not path.is_symlink() and path.is_file():
            path.chmod(0o444)
    for directory in sorted((p for p in output.rglob("*") if p.is_dir() and not p.is_symlink()),
                            key=lambda p: len(p.parts), reverse=True):
        directory.chmod(0o555)
    output.chmod(0o555)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true", help="Verify an existing snapshot only")
    parser.add_argument("--platon-source", type=Path, default=DEFAULT_PLATON)
    parser.add_argument("--nautilus-source", type=Path, default=DEFAULT_NAUTILUS)
    parser.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    if args.verify:
        manifest = verify_snapshot(args.output)
    else:
        manifest = freeze_snapshot(args.output, platon_source=args.platon_source,
                                   nautilus_source=args.nautilus_source, driver=args.driver,
                                   data=args.data, archive=args.archive)
    print(json.dumps({"snapshot": str(args.output.resolve()),
                      "snapshot_id": manifest["snapshot_id"],
                      "files": len(manifest["files"]), "verified": True}))


if __name__ == "__main__":
    main()
