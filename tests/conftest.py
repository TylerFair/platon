"""Shared fixtures and offline data requirements."""
import pytest
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from platon import _forward_model as fm
from tests._support import synthetic_grid


def pytest_configure(config):
    config.addinivalue_line('markers', 'real_stellar_grid: needs the NewEra release bundle')


def pytest_collection_modifyitems(items):
    data = Path(__file__).resolve().parents[1] / 'platon/data'
    if data.is_dir():
        return
    full_data_modules = {
        'test_mie_absorption.py', 'test_transit_depth_calculator.py',
        'test_eclipse_depth_calculator.py', 'test_retrieve.py',
        'test_eclipse_retrieval.py',
    }
    for item in items:
        if (item.path.name in full_data_modules
                or 'TestTwoSectorForwardModel' in item.nodeid
                or item.name in {'test_set_opacity', 'test_stellar_contamination_example_completes'}
                or (item.path.name == 'test_mie.py' and item.name in {
                    'test_complex_refractive_index', 'test_real_refractive_index'})):
            item.add_marker(pytest.mark.skip(reason='Full 11 GB opacity archive is not installed'))


@pytest.fixture(autouse=True)
def offline_downloads(monkeypatch):
    from platon import _get_data
    open_url = _get_data.urlopen

    def local_only(url, *args, **kwargs):
        if not str(url).startswith('file:'):
            pytest.fail('Tests must use local files or mocked downloads')
        return open_url(url, *args, **kwargs)

    monkeypatch.setattr(_get_data, 'urlopen', local_only)


@pytest.fixture(scope='session')
def real_stellar_grid():
    path = Path(__file__).resolve().parents[1] / 'platon/stellar_data/newera_jwst.npz'
    required = [path] + [path.with_name(f'newera_jwst_feh_{i:02d}.npz') for i in range(10)]
    if not all(file.is_file() for file in required):
        pytest.fail('Install the NewEra release bundle before running these offline tests')
    return path


@pytest.fixture(autouse=True)
def require_real_stellar_grid(request):
    if request.node.get_closest_marker('real_stellar_grid'):
        request.getfixturevalue('real_stellar_grid')


@pytest.fixture
def tiny_calculator(monkeypatch, tmp_path):
    """Real calculator/JIT core with small, analytic atmospheric inputs."""
    from platon import _atmosphere_solver as module
    from platon.transit_depth_calculator import TransitDepthCalculator
    waves = np.geomspace(.8e-6, 4.8e-6, 128)
    raw = dict(key=('tiny_test',), lambda_full=waves, low_res_lambdas=waves,
               P_grid=np.array([1e-4, 1e8]), T_grid=np.array([300., 2500.]),
               master_names=['H2', 'He', 'H2O'], master_index={'H2': 0, 'He': 1, 'H2O': 2},
               masses=np.array([2., 4., 18.], np.float32), pol_sqr=np.array([.64, .04, 2.], np.float32),
               opac_names=['H2O'], opac_master_idx=np.array([2], np.int32),
               ln_xsec_stack=np.full((1, 2, 2, len(waves)), np.log(1e-27), np.float32),
               ln_cia_stack=np.empty((0, 2, len(waves)), np.float32),
               cia_idx1=np.array([], np.int32), cia_idx2=np.array([], np.int32),
               ln_hminus_k=np.full((2, len(waves)), fm.LN_MIN_XSEC, np.float32),
               exp3_x=np.array([1e-6, 1e3], np.float32), exp3_y=np.array([.5, 0.], np.float32),
               bterm_x=np.array([1e-6, 1e3], np.float32), bterm_y=np.array([1., 0.], np.float32))
    abund = np.empty((2, 2, 3, 2, 2), np.float32)
    abund[:] = np.log10(np.array([.85, .149, .001]))[None, None, :, None, None]
    getter = SimpleNamespace(log_abundances=abund, included_species=['H2', 'He', 'H2O'],
                             logZs=np.array([-1., 1.]), CO_ratios=np.array([.3, .9]), min_temperature=300.)
    monkeypatch.setattr(module, 'get_data_if_needed', lambda: None)
    monkeypatch.setattr(module, '_load_raw', lambda *args: raw)
    monkeypatch.setattr(module, 'AbundanceGetter', lambda *args: getter)
    monkeypatch.setattr(module, 'load_dict_from_pickle', lambda *args: {})
    monkeypatch.setattr(module, 'load_numpy', lambda *args: np.geomspace(1e-9, 1e-3, 16))
    monkeypatch.setattr(module, 'load_stellar_grid', lambda *args: synthetic_grid())
    manifest = tmp_path / 'synthetic.npz'
    manifest.touch()
    monkeypatch.setattr(module, 'resolve_stellar_grid', lambda *args: manifest)
    monkeypatch.setattr(module, '_DEVICE_CACHE', {})
    monkeypatch.setattr(module, '_LOG_ABUND_CACHE', {})
    return TransitDepthCalculator()
