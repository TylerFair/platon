from urllib.request import urlopen

from platon import __data_url__, __md5sum__

import zipfile
import os
import hashlib
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath

# NewEra stellar spectra are downloaded separately, into data/stellar_data
STELLAR_FOLDER = "stellar_data"


def has_opacity_data(data_dir):
    """Whether data_dir holds the opacity archive, not just stellar grids or
    the hidden staging folder of an unfinished download."""
    data_dir = Path(data_dir)
    return data_dir.is_dir() and any(
        path.name != STELLAR_FOLDER and not path.name.startswith(".")
        for path in data_dir.iterdir())

def get_data_if_needed():
    basedir = Path(__file__).resolve().parent
    if not has_opacity_data(basedir / "data"):
        get_data(basedir)
        
    with open(str(basedir / "md5sum")) as f:
        curr_md5sum = f.read().strip()

    if __md5sum__ != curr_md5sum:
        print("Warning: data files are out of date.  To update, remove the PLATON data directory ({}) and PLATON will automatically download the latest data files on the next run.  Its stellar_data folder can be kept (move it back in afterwards) to avoid downloading the stellar spectra again.".format(basedir / "data"))
        

def get_data(target_dir):
    """Download and verify the archive before installing its data directory.

    Staging is on the destination filesystem so the final directory rename
    is atomic. Failed downloads or extraction leave no partial installation.
    """
    MB_TO_BYTES = 2**20
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / "data"
    if has_opacity_data(destination):
        raise FileExistsError("Data directory already exists: {}".format(destination))
    print("Data URL", __data_url__)

    with tempfile.TemporaryDirectory(prefix=".platon-download-",
                                     dir=target_dir) as staging_dir:
        staging = Path(staging_dir)
        filename = staging / "data.zip"
        checksum = hashlib.md5()
        # urlopen's default HTTPS context verifies certificates and hostnames.
        with urlopen(__data_url__) as response, filename.open("wb") as output:
            length = response.getheader("Content-Length")
            file_size = int(length) if length is not None else None
            bytes_downloaded = 0
            while True:
                block = response.read(2**20)
                if not block:
                    break
                output.write(block)
                checksum.update(block)
                bytes_downloaded += len(block)
                status = "{:.0f} MB".format(bytes_downloaded / MB_TO_BYTES)
                if file_size:
                    status += "  [{}%]".format(int(100 * bytes_downloaded / file_size))
                print(status, end="\r")

        curr_md5sum = checksum.hexdigest()
        if curr_md5sum != __md5sum__:
            raise RuntimeError(
                "Downloaded data file is corrupt (wrong md5sum). Please try again.")

        print("\nExtracting...")
        with zipfile.ZipFile(filename) as archive:
            # The archive may only populate data/, never package source files
            # or paths outside the staging directory.
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts or \
                   "\\" in member.filename or not path.parts or path.parts[0] != "data":
                    raise ValueError("Invalid data archive path: {}".format(member.filename))
            archive.extractall(staging)

        if not (staging / "data").is_dir():
            raise ValueError("Downloaded archive does not contain a data directory")
        checksum_path = staging / "md5sum"
        checksum_path.write_text(curr_md5sum)
        # Install the checksum first: once data/ becomes visible, its checksum
        # is already present. A failed rename leaves data/ absent and retryable.
        os.replace(checksum_path, target_dir / "md5sum")
        # data/ may already hold stellar grids downloaded on their own; carry
        # them into the new directory
        carried = []
        if destination.exists():
            for path in destination.iterdir():
                if not (staging / "data" / path.name).exists():
                    os.replace(path, staging / "data" / path.name)
                    carried.append(path.name)
            try:
                os.rmdir(destination)
                os.replace(staging / "data", destination)
            except OSError:
                destination.mkdir(exist_ok=True)
                for name in carried:
                    os.replace(staging / "data" / name, destination / name)
                raise
        else:
            os.replace(staging / "data", destination)
    print("Extraction finished!")


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


def _download_and_install(url, target_dir, folder, expected_checksum, *,
                          required_files=()):
    """Stream a SHA-256-verified ZIP into staging, then replace the installed
    folder target_dir/folder (used for the NewEra stellar grid)."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".platon-download-", dir=target_dir) as temp:
        temporary = Path(temp)
        archive_path = temporary / f"{folder}.zip"
        checksum = hashlib.sha256()
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

        destination = target_dir / folder
        backup = temporary / "previous_data"
        had_data = destination.exists() or destination.is_symlink()
        if had_data:
            os.replace(destination, backup)
        installed = False
        try:
            os.replace(staged_data, destination)
            installed = True
        except OSError:
            if installed:
                shutil.rmtree(destination)
            if had_data:
                os.replace(backup, destination)
            raise
    print("Extraction finished!")
