"""Download and install the atmospheric data archive."""

import hashlib
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat
import tempfile
from urllib.request import urlopen
import zipfile

from platon import __data_url__, __md5sum__


def get_data_if_needed():
    basedir = Path(__file__).resolve().parent
    if not (basedir / "data").is_dir():
        get_data(basedir)
    marker = basedir / "md5sum"
    curr_md5sum = marker.read_text().strip() if marker.is_file() else None
    if __md5sum__ != curr_md5sum:
        print("Warning: data files are out of date. To update, remove the PLATON "
              "data directory ({}) and PLATON will automatically download the "
              "latest data files on the next run.".format(basedir / "data"))


def _validate_archive_members(archive, folder="data"):
    """Only install regular files and directories beneath the expected folder."""
    for member in archive.infolist():
        name = member.filename
        path = PurePosixPath(name)
        mode = member.external_attr >> 16
        if (not path.parts or path.is_absolute() or ".." in path.parts
                or "\\" in name or PureWindowsPath(name).drive
                or path.parts[0] != folder
                or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR))):
            raise RuntimeError(f"Unsafe path in downloaded data archive: {name!r}")


def get_data(target_dir):
    """Download, verify and atomically install atmospheric data."""
    print("Data URL", __data_url__)
    _download_and_install(__data_url__, target_dir, "data", __md5sum__,
                          algorithm="md5", marker="md5sum")


def _download_and_install(url, target_dir, folder, expected_checksum, *,
                          algorithm="sha256", marker=None, required_files=()):
    """Stream a verified ZIP into staging, then replace the installed folder."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".platon-download-", dir=target_dir) as temp:
        temporary = Path(temp)
        archive_path = temporary / f"{folder}.zip"
        checksum = hashlib.new(algorithm)
        downloaded = 0
        with urlopen(url) as response, archive_path.open("wb") as output:
            try:
                content_length = int(response.headers.get("Content-Length"))
            except (TypeError, ValueError):
                content_length = 0
            print(f"Downloading {folder}.zip" + (
                ": {:.0f} MB".format(content_length / 2**20)
                if content_length > 0 else " (size unknown)"))
            while True:
                block = response.read(2**20)
                if not block:
                    break
                output.write(block)
                checksum.update(block)
                downloaded += len(block)
                status = "{:.0f} MB".format(downloaded / 2**20)
                if content_length > 0:
                    status += " [{}%]".format(int(100 * downloaded / content_length))
                print(status, end="\r")

        digest = checksum.hexdigest()
        if digest != expected_checksum:
            raise RuntimeError(f"Downloaded {folder}.zip is corrupt (checksum mismatch). Please try again.")
        staging = temporary / "staging"
        staging.mkdir()
        print("\nExtracting...")
        try:
            with zipfile.ZipFile(archive_path) as archive:
                _validate_archive_members(archive, folder)
                archive.extractall(staging)
        except zipfile.BadZipFile as error:
            raise RuntimeError("Downloaded data archive is not a valid ZIP file") from error
        staged_data = staging / folder
        if not staged_data.is_dir():
            raise RuntimeError(f"Downloaded archive does not contain a {folder} directory")
        if any(not (staged_data / name).is_file() for name in required_files):
            raise RuntimeError("Downloaded stellar archive is missing required grid files")

        staged_marker = temporary / "checksum"
        if marker is not None:
            staged_marker.write_text(digest)
        destination = target_dir / folder
        backup = temporary / "previous_data"
        had_data = destination.exists() or destination.is_symlink()
        if had_data:
            os.replace(destination, backup)
        installed = False
        try:
            os.replace(staged_data, destination)
            installed = True
            if marker is not None:
                os.replace(staged_marker, target_dir / marker)
        except OSError:
            if installed:
                shutil.rmtree(destination)
            if had_data:
                os.replace(backup, destination)
            raise
    print("Extraction finished!")
