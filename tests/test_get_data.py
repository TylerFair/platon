import unittest
import os
import tempfile
import hashlib
import io
from pathlib import Path
from unittest import mock
import zipfile

from platon._get_data import get_data


class _Response(io.BytesIO):
    def getheader(self, name):
        # Some servers omit Content-Length.
        return None


class TestGetData(unittest.TestCase):
    def archive(self, name="data/example.txt"):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(name, "test data")
        return buffer.getvalue()

    def test_verified_download_installs_data_and_checksum(self):
        contents = self.archive()
        checksum = hashlib.md5(contents).hexdigest()
        with tempfile.TemporaryDirectory() as target, \
             mock.patch("platon._get_data.__md5sum__", checksum), \
             mock.patch("platon._get_data.urlopen",
                        return_value=_Response(contents)) as download:
            get_data(target)
            self.assertEqual((Path(target) / "data/example.txt").read_text(),
                             "test data")
            self.assertEqual((Path(target) / "md5sum").read_text(), checksum)
            self.assertEqual(sorted(p.name for p in Path(target).iterdir()),
                             ["data", "md5sum"])
            # No insecure SSL context is supplied.
            self.assertEqual(download.call_args.kwargs, {})

    def test_bad_checksum_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as target, \
             mock.patch("platon._get_data.urlopen",
                        return_value=_Response(self.archive())), \
             mock.patch("platon._get_data.__md5sum__", "wrong checksum"), \
             mock.patch("platon._get_data.zipfile.ZipFile.extractall") as extract:
            with self.assertRaisesRegex(RuntimeError, "wrong md5sum"):
                get_data(target)
            extract.assert_not_called()
            self.assertEqual(list(Path(target).iterdir()), [])

    def test_extraction_failure_leaves_no_partial_installation(self):
        contents = self.archive()

        def fail_extraction(archive, target):
            (Path(target) / "data").mkdir()
            (Path(target) / "data/partial").write_text("incomplete")
            raise OSError("extraction failed")

        with tempfile.TemporaryDirectory() as target, \
             mock.patch("platon._get_data.urlopen",
                        return_value=_Response(contents)), \
             mock.patch("platon._get_data.__md5sum__",
                        hashlib.md5(contents).hexdigest()), \
             mock.patch("platon._get_data.zipfile.ZipFile.extractall",
                        autospec=True, side_effect=fail_extraction):
            with self.assertRaisesRegex(OSError, "extraction failed"):
                get_data(target)
            self.assertEqual(list(Path(target).iterdir()), [])

    def test_archive_cannot_overwrite_package_files(self):
        for name in ["__init__.py", "data/../__init__.py", "/data/file"]:
            contents = self.archive(name)
            with self.subTest(name=name), tempfile.TemporaryDirectory() as target, \
                 mock.patch("platon._get_data.urlopen",
                            return_value=_Response(contents)), \
                 mock.patch("platon._get_data.__md5sum__",
                            hashlib.md5(contents).hexdigest()):
                with self.assertRaisesRegex(ValueError, "Invalid data archive path"):
                    get_data(target)
                self.assertEqual(list(Path(target).iterdir()), [])

    def test_existing_data_is_preserved(self):
        with tempfile.TemporaryDirectory() as target:
            data = Path(target) / "data"
            data.mkdir()
            (data / "existing").write_text("keep")
            with mock.patch("platon._get_data.urlopen") as download:
                with self.assertRaises(FileExistsError):
                    get_data(target)
                download.assert_not_called()
            self.assertEqual((data / "existing").read_text(), "keep")

    @unittest.skip("Too long")
    def test_get_data(self):
        target_dir = tempfile.mkdtemp()
        get_data(target_dir)

        self.assertFalse(os.path.isfile(os.path.join(target_dir, "data.zip")))
        self.assertTrue(os.path.isdir(os.path.join(target_dir, "data")))

        expectedFiles = ["collisional_absorption.pkl", "pressures.npy", "species_info", "temperatures.npy", "wavelengths.npy", "stellar_spectra.pkl"]
        expectedDirs = ["Absorption", "abundances"]

        for f in expectedFiles:
            filename = os.path.join(target_dir, "data", f)
            self.assertTrue(os.path.isfile(filename))

        for d in expectedDirs:
            dirname = os.path.join(target_dir, "data", d)
            self.assertTrue(os.path.isdir(dirname))

if __name__ == '__main__':
    unittest.main()        


# The NewEra grid lives in data/stellar_data and may be downloaded first
class TestStellarGridAlongsideOpacities(unittest.TestCase):
    def install(self, target):
        contents = TestGetData.archive(self)
        with mock.patch("platon._get_data.__md5sum__", hashlib.md5(contents).hexdigest()), \
             mock.patch("platon._get_data.urlopen", return_value=_Response(contents)):
            get_data(target)

    def test_stellar_grid_alone_is_not_opacity_data(self):
        from platon._get_data import has_opacity_data
        with tempfile.TemporaryDirectory() as target:
            data = Path(target) / "data"
            (data / "stellar_data").mkdir(parents=True)
            (data / ".platon-download-unfinished").mkdir()
            self.assertFalse(has_opacity_data(data))
            (data / "pressures.npy").write_text("installed")
            self.assertTrue(has_opacity_data(data))

    def test_opacities_install_next_to_an_existing_stellar_grid(self):
        with tempfile.TemporaryDirectory() as target:
            stellar = Path(target) / "data/stellar_data"
            stellar.mkdir(parents=True)
            (stellar / "newera_jwst.npz").write_text("grid")
            self.install(target)
            self.assertEqual((Path(target) / "data/example.txt").read_text(), "test data")
            self.assertEqual((stellar / "newera_jwst.npz").read_text(), "grid")

    def test_get_data_if_needed_downloads_when_only_stellar_data_exists(self):
        from platon import _get_data
        with tempfile.TemporaryDirectory() as target:
            (Path(target) / "data/stellar_data").mkdir(parents=True)
            with mock.patch.object(_get_data, "__file__", str(Path(target) / "_get_data.py")), \
                 mock.patch.object(_get_data, "get_data", side_effect=self.install) as download:
                _get_data.get_data_if_needed()
            download.assert_called_once()
            self.assertTrue((Path(target) / "data/example.txt").is_file())
