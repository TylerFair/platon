"""Offline release installation and first-use regressions."""
import hashlib
import json
import pickle
import subprocess
import sys
import zipfile

import numpy as np
import pytest

import platon
from platon import _stellar_grid as grids
from platon.stellar_grid import download_stellar_grid, generate_stellar_grid
from tests._support import tiny_profile


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    shape = (2, 1, 10)
    shards = [f'newera_jwst_feh_{i:02d}.npz' for i in range(10)]
    np.savez(source / 'newera_jwst.npz', temperatures=[3000., 5000.],
             loggs=[4.5], fehs=np.linspace(-4., .5, 10),
             metadata=json.dumps({'missing_model_policy': 'interpolate_isolated_temperature'}),
             wavelengths_m=[.6e-6, 2.e-6, 28.5e-6], shard_files=shards,
             log_offset=np.full(shape, np.log(1.e12), np.float32),
             log_scale=np.full(shape, .001, np.float32), valid=np.ones(shape, bool))
    for i, name in enumerate(shards):
        np.savez(source / name, encoded_spectra=np.full((2, 1, 3), i, np.uint16))
    (source / 'README.md').write_text('attribution')
    (source / 'newera_jwst.json').write_text('{}')
    (source / 'validation.json').write_text('{}')
    archive = tmp_path / 'release.zip'
    with zipfile.ZipFile(archive, 'w') as output:
        for path in source.iterdir():
            output.write(path, f'stellar_data/{path.name}')
    package = tmp_path / 'package'
    package.mkdir()
    monkeypatch.setattr(grids, '__file__', str(package / '_stellar_grid.py'))
    monkeypatch.setattr(platon, '__stellar_grid_url__', archive.as_uri())
    monkeypatch.setattr(platon, '__stellar_grid_sha256__', hashlib.sha256(archive.read_bytes()).hexdigest())
    grids._load_grid_cached.cache_clear()
    yield package, archive
    grids._load_grid_cached.cache_clear()


def test_first_load_downloads_verified_shards_once(bundle, capsys):
    package, archive = bundle
    assert not grids.resolve_stellar_grid().exists()
    grid = grids.load_stellar_grid()
    assert grid.spectra.resident_nbytes == 0
    assert grid.metadata['missing_model_policy'] == 'interpolate_isolated_temperature'
    np.testing.assert_allclose(grid.interpolate(4000., 4.5, 0.), 1.e12 * np.exp(.008), rtol=2e-6)
    assert len(list((package / 'data/stellar_data').glob('*.npz'))) == 11
    output = capsys.readouterr().out
    assert 'Downloading NewEra stellar spectra' in output
    assert '[100%]' in output
    archive.unlink()  # Reuse must work with no download source.
    assert download_stellar_grid() == package / 'data/stellar_data/newera_jwst.npz'
    assert grids.load_stellar_grid() is grid
    assert capsys.readouterr().out == ''
    assert not list(package.rglob('.platon-download-*'))


def test_grid_in_the_old_location_is_moved_not_downloaded(bundle, capsys):
    package, archive = bundle
    legacy = package / 'stellar_data'
    with zipfile.ZipFile(archive) as source:
        source.extractall(package)
    archive.unlink()  # moving must not need the download source
    path = download_stellar_grid()
    assert path == package / 'data/stellar_data/newera_jwst.npz' and path.is_file()
    assert not legacy.exists()
    assert 'Moved the NewEra stellar spectra' in capsys.readouterr().out


def test_first_tls_call_installs_bundle(tiny_calculator, bundle, monkeypatch):
    from platon import _atmosphere_solver as solver
    from platon.constants import R_sun, M_jup, R_jup
    monkeypatch.setattr(solver, 'load_stellar_grid', grids.load_stellar_grid)
    monkeypatch.setattr(solver, 'resolve_stellar_grid', grids.resolve_stellar_grid)
    args = (tiny_profile(), R_sun, M_jup, R_jup)
    tiny_calculator.compute_depths(*args)
    no_star_device = tiny_calculator.atm.device_data()
    assert tiny_calculator.atm._stellar_grid is None
    assert not grids.resolve_stellar_grid().exists()
    _, depths, _ = tiny_calculator.compute_depths(
        *args,
        T_star=4000., T_het=3200., f_het=.1)
    assert np.all(np.isfinite(depths))
    assert grids.resolve_stellar_grid().is_file()
    assert tiny_calculator.atm.stellar_grid.spectra.resident_nbytes > 0
    assert tiny_calculator.atm.device_data() is not no_star_device


def test_generate_default_source_downloads(bundle, tmp_path):
    cache = generate_stellar_grid(tmp_path / 'star.npz', temperatures=[4000.])
    assert grids.resolve_stellar_grid().is_file()
    np.testing.assert_allclose(grids.load_stellar_grid(cache).interpolate(4000.),
                               grids.load_stellar_grid().interpolate(4000.), rtol=2e-6)


def test_local_and_phoenix_sources_do_not_download_newera(bundle, tmp_path, monkeypatch):
    package, archive = bundle
    data = dict(temperatures=[3000., 5000.], wavelengths_m=[1.e-6, 2.e-6],
                spectra=np.full((2, 2), 1.e12))
    local = tmp_path / 'local.npz'
    np.savez(local, **data)
    (package / 'data').mkdir()
    (package / 'data/stellar_spectra.pkl').write_bytes(pickle.dumps(data))
    monkeypatch.setattr('platon.stellar_grid.download_stellar_grid',
                        lambda *args: pytest.fail('Downloaded default for explicit source'))
    for source in (local, 'phoenix'):
        grid = grids.load_stellar_grid(source)
        np.testing.assert_allclose(grid.interpolate(4000.), 1.e12)
        assert grids.load_stellar_grid(source) is grid
    assert not grids.resolve_stellar_grid().exists()


def test_imports_do_not_open_or_download_stellar_data():
    script = """
import urllib.request
import numpy as np
def unexpected_download(*args, **kwargs):
    raise AssertionError('Download during import')
urllib.request.urlopen = unexpected_download
original_load = np.load
def check_load(path, *args, **kwargs):
    assert 'stellar_data' not in str(path), 'Stellar file opened during import'
    return original_load(path, *args, **kwargs)
np.load = check_load
import platon
import platon.stellar_grid
from platon.transit_depth_calculator import TransitDepthCalculator
from platon.eclipse_depth_calculator import EclipseDepthCalculator
from platon.combined_retriever import CombinedRetriever
"""
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True)


@pytest.mark.parametrize('installed', [False, True])
def test_bad_checksum_preserves_existing_files_and_cleans_staging(bundle, monkeypatch, installed):
    package, archive = bundle
    if installed:
        path = download_stellar_grid()
        original = path.read_bytes()
    monkeypatch.setattr(platon, '__stellar_grid_sha256__', '0' * 64)
    with pytest.raises(RuntimeError, match='checksum mismatch'):
        download_stellar_grid(force=True)
    assert not list(package.rglob('.platon-download-*'))
    if installed:
        assert path.read_bytes() == original
    else:
        assert not (package / 'data/stellar_data').exists()


def test_failed_network_explains_offline_alternatives(bundle):
    package, archive = bundle
    archive.unlink()
    with pytest.raises(RuntimeError, match=r'download_stellar_grid\(\).*stellar_grid='):
        grids.load_stellar_grid()
    assert not (package / 'data/stellar_data').exists()
    assert not list(package.rglob('.platon-download-*'))


@pytest.mark.parametrize('bad_name', ['../outside', 'stellar_data/../../outside'])
def test_unsafe_archive_rejected_without_install(bundle, monkeypatch, bad_name):
    package, archive = bundle
    with zipfile.ZipFile(archive, 'a') as output:
        output.writestr(bad_name, b'bad')
    monkeypatch.setattr(platon, '__stellar_grid_sha256__', hashlib.sha256(archive.read_bytes()).hexdigest())
    with pytest.raises(RuntimeError, match='Unsafe path'):
        download_stellar_grid()
    assert not (package / 'data/stellar_data').exists()
    assert not list(package.rglob('.platon-download-*'))


def test_missing_shard_is_reinstalled_and_force_refreshes(bundle):
    package, archive = bundle
    path = download_stellar_grid()
    (path.parent / 'newera_jwst_feh_09.npz').unlink()
    assert download_stellar_grid() == path
    assert (path.parent / 'newera_jwst_feh_09.npz').is_file()
    path.write_bytes(b'old manifest')
    assert download_stellar_grid(force=True) == path
    assert path.read_bytes() != b'old manifest'


def test_constructors_and_no_star_or_blackbody_models_are_lazy(tiny_calculator, bundle, monkeypatch):
    from platon import _atmosphere_solver as solver
    from platon import eclipse_depth_calculator as eclipse
    from platon.combined_retriever import CombinedRetriever
    from platon.transit_depth_calculator import TransitDepthCalculator
    from platon.constants import R_sun, M_jup, R_jup
    def unexpected_load(*args):
        pytest.fail('Opened stellar grid without needing NewEra spectra')
    monkeypatch.setattr(solver, 'load_stellar_grid', unexpected_load)
    monkeypatch.setattr(eclipse.pd, 'read_csv', lambda *args: eclipse.pd.DataFrame())
    monkeypatch.setattr(eclipse.ascii, 'read', lambda *args, **kwargs: {})
    transit = TransitDepthCalculator()
    thermal = eclipse.EclipseDepthCalculator()
    CombinedRetriever()
    assert transit.atm._stellar_grid is None
    assert thermal.atm._stellar_grid is None
    profile = tiny_profile()
    args = (profile, R_sun, M_jup, R_jup)
    _, clean, _ = transit.compute_depths(*args)
    assert np.all(np.isfinite(clean))
    assert transit.atm._stellar_grid is None
    _, bb, _ = transit.compute_depths(*args, T_star=4000., stellar_blackbody=True)
    assert np.all(np.isfinite(bb))
    _, ones = transit.atm.get_stellar_spectrum(None, None, None)
    np.testing.assert_array_equal(ones, 1.)
    assert thermal.atm._stellar_grid is None
    assert not grids.resolve_stellar_grid().exists()


@pytest.mark.parametrize('policy', [None, 'reject', 'interpolate_isolated_temperature'])
def test_missing_model_policy_is_metadata_driven_at_any_path(bundle, tmp_path, policy):
    package, _ = bundle
    manifest = download_stellar_grid()
    data = dict(temperatures=[3000., 4000., 5000.], loggs=[4.5], fehs=[0.],
                wavelengths_m=[1e-6, 2e-6],
                spectra=np.array([1., 0., 3.])[:, None, None, None] * np.ones((3, 1, 1, 2)),
                valid=np.array([True, False, True])[:, None, None],
                metadata=json.dumps({} if policy is None else {'missing_model_policy': policy}))
    np.savez(manifest, **data)
    copied = tmp_path / 'copied.npz'
    copied.write_bytes(manifest.read_bytes())
    grids._load_grid_cached.cache_clear()
    for source in ('newera', copied):
        grid = grids.load_stellar_grid(source)
        if policy == 'interpolate_isolated_temperature':
            np.testing.assert_allclose(grid.interpolate(4000., 4.5, 0.), [2., 2.])
        else:
            from platon.errors import AtmosphereError
            with pytest.raises(AtmosphereError, match='missing'):
                grid.interpolate(4000., 4.5, 0.)


def test_public_module_does_not_reexport_private_loaders():
    from platon import stellar_grid
    assert not hasattr(stellar_grid, 'load_stellar_grid')
    assert not hasattr(stellar_grid, 'load_native_stellar_grid')
