"""Downloader regressions using tiny in-memory responses, never the network."""

import hashlib
import io
from pathlib import Path
import stat
import zipfile

import pytest

from platon import _get_data as downloader


def archive_bytes(entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, contents in entries.items():
            archive.writestr(name, contents)
    return stream.getvalue()


class Response(io.BytesIO):
    def __init__(self, payload, content_length=None, interrupt=False):
        super().__init__(payload)
        self.headers = {"Content-Length": content_length}
        self.interrupt = interrupt
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        if self.interrupt and len(self.read_sizes) > 1:
            raise OSError("Connection interrupted")
        return super().read(size)


def mock_download(monkeypatch, payload, **kwargs):
    response = Response(payload, **kwargs)
    calls = []

    def open_url(url, **options):
        calls.append((url, options))
        return response

    monkeypatch.setattr(downloader, "urlopen", open_url)
    monkeypatch.setattr(downloader, "__md5sum__", hashlib.md5(payload).hexdigest())
    return response, calls


def assert_no_partial_downloads(target):
    assert not list(target.glob(".platon-download-*"))
    assert not (target / "data.zip").exists()


def previous_install(target):
    (target / "data").mkdir(parents=True)
    (target / "data" / "old.npy").write_bytes(b"original data")
    (target / "md5sum").write_text("original checksum")


def assert_previous_install(target):
    assert (target / "data" / "old.npy").read_bytes() == b"original data"
    assert (target / "md5sum").read_text() == "original checksum"
    assert_no_partial_downloads(target)


def test_verified_streaming_download_installs_in_target_only(tmp_path, monkeypatch):
    payload = archive_bytes({"data/large.npy": b"x" * (2 * 2**20 + 31),
                             "data/abundances/example.npy": b"small"})
    response, calls = mock_download(monkeypatch, payload, content_length=str(len(payload)))
    target = tmp_path / "new" / "package"
    cwd = tmp_path / "working"
    cwd.mkdir()
    (cwd / "data.zip").write_bytes(b"keep this unrelated file")
    monkeypatch.chdir(cwd)
    package = tmp_path / "installed-package"
    package.mkdir()
    (package / "md5sum").write_text("keep this installed marker")
    monkeypatch.setattr(downloader, "__file__", str(package / "_get_data.py"))

    downloader.get_data(target)

    assert (target / "data" / "large.npy").stat().st_size == 2 * 2**20 + 31
    assert (target / "data" / "abundances" / "example.npy").read_bytes() == b"small"
    assert (target / "md5sum").read_text() == hashlib.md5(payload).hexdigest()
    assert (package / "md5sum").read_text() == "keep this installed marker"
    assert (cwd / "data.zip").read_bytes() == b"keep this unrelated file"
    assert response.closed
    assert len(response.read_sizes) > 2
    assert all(0 < size <= 2**20 for size in response.read_sizes)
    assert calls == [(downloader.__data_url__, {})]  # urllib's verified TLS defaults
    assert_no_partial_downloads(target)


@pytest.mark.parametrize("content_length", [None, "0", "invalid", "-1"])
def test_content_length_is_optional(tmp_path, monkeypatch, content_length):
    payload = archive_bytes({"data/pressures.npy": b"test"})
    mock_download(monkeypatch, payload, content_length=content_length)
    downloader.get_data(tmp_path)
    assert (tmp_path / "data" / "pressures.npy").read_bytes() == b"test"
    assert_no_partial_downloads(tmp_path)


def test_checksum_checked_before_opening_zip_or_changing_old_data(tmp_path, monkeypatch):
    previous_install(tmp_path)
    payload = archive_bytes({"data/new.npy": b"new data"})
    response, _ = mock_download(monkeypatch, payload)
    monkeypatch.setattr(downloader, "__md5sum__", "0" * 32)
    monkeypatch.setattr(downloader.zipfile, "ZipFile", lambda *args: pytest.fail("ZIP opened before checksum"))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        downloader.get_data(tmp_path)
    assert response.closed
    assert_previous_install(tmp_path)


def test_connection_failure_closes_response_and_preserves_old_data(tmp_path, monkeypatch):
    previous_install(tmp_path)
    payload = archive_bytes({"data/new.npy": b"x" * 2**20})
    response, _ = mock_download(monkeypatch, payload, interrupt=True)
    with pytest.raises(OSError, match="interrupted"):
        downloader.get_data(tmp_path)
    assert response.closed
    assert_previous_install(tmp_path)


@pytest.mark.parametrize("payload", [b"not a ZIP archive", archive_bytes({})])
def test_malformed_archives_leave_no_installation(tmp_path, monkeypatch, payload):
    mock_download(monkeypatch, payload)
    with pytest.raises(RuntimeError, match="ZIP|data directory"):
        downloader.get_data(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["../outside.txt", "/outside.txt", "data/../../outside.txt",
                                 "data\\..\\outside.txt", "C:/outside.txt", "platon/__init__.py"])
def test_unsafe_archive_paths_are_rejected_before_extraction(tmp_path, monkeypatch, name):
    previous_install(tmp_path)
    payload = archive_bytes({"data/valid.npy": b"valid", name: b"unsafe"})
    mock_download(monkeypatch, payload)
    with pytest.raises(RuntimeError, match="Unsafe path"):
        downloader.get_data(tmp_path)
    assert_previous_install(tmp_path)
    assert not (tmp_path / "data" / "valid.npy").exists()


def test_archive_symlinks_are_rejected(tmp_path, monkeypatch):
    info = zipfile.ZipInfo("data/link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    payload = archive_bytes({info: "../../outside"})
    mock_download(monkeypatch, payload)
    with pytest.raises(RuntimeError, match="Unsafe path"):
        downloader.get_data(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_zip_crc_error_preserves_previous_installation(tmp_path, monkeypatch):
    previous_install(tmp_path)
    payload = archive_bytes({"data/new.npy": b"distinct-payload"})
    payload = payload.replace(b"distinct-payload", b"corrupt!-payload", 1)
    mock_download(monkeypatch, payload)
    with pytest.raises(RuntimeError, match="valid ZIP"):
        downloader.get_data(tmp_path)
    assert_previous_install(tmp_path)


@pytest.mark.parametrize("with_previous_installation", [True, False])
def test_marker_failure_rolls_back_data_replacement(tmp_path, monkeypatch, with_previous_installation):
    if with_previous_installation:
        previous_install(tmp_path)
    payload = archive_bytes({"data/new.npy": b"new data"})
    mock_download(monkeypatch, payload)
    real_replace = downloader.os.replace

    def replace(source, destination):
        if Path(destination) == tmp_path / "md5sum":
            raise PermissionError("Cannot replace marker")
        real_replace(source, destination)

    monkeypatch.setattr(downloader.os, "replace", replace)
    with pytest.raises(PermissionError, match="marker"):
        downloader.get_data(tmp_path)
    if with_previous_installation:
        assert_previous_install(tmp_path)
    else:
        assert list(tmp_path.iterdir()) == []
    assert not (tmp_path / "data" / "new.npy").exists()


def test_successful_update_replaces_old_data_and_marker(tmp_path, monkeypatch):
    previous_install(tmp_path)
    payload = archive_bytes({"data/new.npy": b"new data"})
    mock_download(monkeypatch, payload)
    downloader.get_data(tmp_path)
    assert (tmp_path / "data" / "new.npy").read_bytes() == b"new data"
    assert not (tmp_path / "data" / "old.npy").exists()
    assert (tmp_path / "md5sum").read_text() == hashlib.md5(payload).hexdigest()
    assert_no_partial_downloads(tmp_path)


@pytest.mark.parametrize("marker", [None, "outdated"])
def test_missing_or_stale_marker_warns_without_redownload(tmp_path, monkeypatch, capsys, marker):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "pressures.npy").write_bytes(b"installed")
    if marker is not None:
        (tmp_path / "md5sum").write_text(marker)
    monkeypatch.setattr(downloader, "__file__", str(tmp_path / "_get_data.py"))
    monkeypatch.setattr(downloader, "get_data", lambda *args: pytest.fail("Unexpected redownload"))
    downloader.get_data_if_needed()
    assert "out of date" in capsys.readouterr().out


def test_installed_current_data_needs_no_download(tmp_path, monkeypatch, capsys):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "pressures.npy").write_bytes(b"installed")
    (tmp_path / "md5sum").write_text(downloader.__md5sum__ + "\n")
    monkeypatch.setattr(downloader, "__file__", str(tmp_path / "_get_data.py"))
    monkeypatch.setattr(downloader, "get_data", lambda *args: pytest.fail("Unexpected redownload"))
    downloader.get_data_if_needed()
    assert capsys.readouterr().out == ""


def test_absent_data_is_installed_once(tmp_path, monkeypatch, capsys):
    payload = archive_bytes({"data/pressures.npy": b"test"})
    _, calls = mock_download(monkeypatch, payload)
    monkeypatch.setattr(downloader, "__file__", str(tmp_path / "_get_data.py"))
    downloader.get_data_if_needed()
    downloader.get_data_if_needed()
    assert len(calls) == 1
    assert (tmp_path / "data" / "pressures.npy").read_bytes() == b"test"
    assert "out of date" not in capsys.readouterr().out


def test_stellar_grid_alone_does_not_count_as_opacity_data(tmp_path, monkeypatch):
    # NewEra may be downloaded into data/ before the opacity archive
    (tmp_path / "data" / "stellar_data").mkdir(parents=True)
    (tmp_path / "data" / "stellar_data" / "newera_jwst.npz").write_bytes(b"grid")
    payload = archive_bytes({"data/pressures.npy": b"test"})
    _, calls = mock_download(monkeypatch, payload)
    monkeypatch.setattr(downloader, "__file__", str(tmp_path / "_get_data.py"))
    downloader.get_data_if_needed()
    assert len(calls) == 1
    assert (tmp_path / "data" / "pressures.npy").read_bytes() == b"test"


def test_opacity_reinstall_keeps_downloaded_stellar_grid(tmp_path, monkeypatch):
    previous_install(tmp_path)
    (tmp_path / "data" / "stellar_data").mkdir()
    (tmp_path / "data" / "stellar_data" / "newera_jwst.npz").write_bytes(b"grid")
    payload = archive_bytes({"data/new.npy": b"new data"})
    mock_download(monkeypatch, payload)
    downloader.get_data(tmp_path)
    assert (tmp_path / "data" / "new.npy").read_bytes() == b"new data"
    assert not (tmp_path / "data" / "old.npy").exists()
    assert (tmp_path / "data" / "stellar_data" / "newera_jwst.npz").read_bytes() == b"grid"
    assert_no_partial_downloads(tmp_path)
