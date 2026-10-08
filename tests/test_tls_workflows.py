"""TLS workflow checks with bundled stars and independent reference spectra.

The static fixture was independently evaluated using MSG on native source bins.
Ordinary tests do not import MSG/h5py or read the original HDF5 grid. These tests
verify stellar interpolation, forward models, and retrieval workflows.
"""
import json
import hashlib
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from platon._stellar_grid import load_stellar_grid
from platon.constants import R_jup, R_sun
from platon.stellar_grid import generate_stellar_grid
from platon.transit_depth_calculator import TransitDepthCalculator
from tests._support import (MP, RP, RS, STELLAR_PARAMS, TRACE_GASES,
                            TRACE_LOG_VMRS, tiny_profile)


FIXTURE = Path(__file__).parent / 'fixtures' / 'cool_star_tls_native.json'


@pytest.fixture
def stellar_cache(tmp_path):
    return generate_stellar_grid(
        tmp_path / 'stellar.npz', logg=STELLAR_PARAMS['logg_star'],
        feh=STELLAR_PARAMS['feh_star'], temperatures=np.arange(2300., 5001., 100.))


def _platon_log_integral(flux, source_edges, target):
    """Integrate independent native piecewise-constant F_lambda dln(lambda)."""
    cumulative = np.r_[0., np.cumsum(flux * np.diff(np.log(source_edges)))]
    index = np.clip(np.searchsorted(source_edges, target, side='right') - 1,
                    0, len(flux) - 1)
    return cumulative[index] + flux[index] * np.log(target / source_edges[index])


@pytest.mark.parametrize('resolving_power,tolerance_ppm', [(100, 1.), (300, 3.)])
@pytest.mark.real_stellar_grid
def test_default_cool_star_tls_matches_native_reference(resolving_power, tolerance_ppm):
    reference = json.loads(FIXTURE.read_text())
    params = STELLAR_PARAMS  # reference['parameters'], with current names
    scenario = next(row for row in reference['scenarios']
                    if row['resolving_power'] == resolving_power)
    bins = np.asarray(scenario['wavelength_bins_m'])
    # Use the independent source's native wavelength edges, rather than the
    # bundle's edges, to avoid a test tied to its packing implementation.
    native_edges = np.linspace(.6e-6, 28.5e-6, 139501)
    native_centers = .5 * (native_edges[:-1] + native_edges[1:])
    with patch.dict('sys.modules', {'pymsg': None, 'h5py': None}):
        grid = load_stellar_grid()
        phot, spot = [np.interp(native_centers, grid.wavelengths_m,
                               grid.interpolate(t, params['logg_star'], params['feh_star']))
                      for t in (params['T_star'], params['T_het'])]
    f = params['f_het']
    mixed = (1 - f) * phot + f * spot
    phot_integrals = np.diff(_platon_log_integral(phot, native_edges, bins), axis=1)[:, 0]
    mixed_integrals = np.diff(_platon_log_integral(mixed, native_edges, bins), axis=1)[:, 0]
    # Express the tolerances as ppm of a flat 1% planetary transit. They are
    # absolute depth differences, not ppm of the stellar correction factor.
    compact_depth = .01 * phot_integrals / mixed_integrals
    native_depth = .01 * np.asarray(scenario['tls_factors'])
    np.testing.assert_allclose(compact_depth, native_depth, rtol=0,
                               atol=tolerance_ppm * 1e-6)


@pytest.mark.real_stellar_grid
def test_stellar_cache_and_tls_without_msg_hdf5_or_atmospheric_download(tmp_path):
    with patch.dict('sys.modules', {'pymsg': None, 'h5py': None}), \
            patch('platon._get_data.get_data_if_needed',
                  side_effect=AssertionError('Stellar cache downloaded opacity data')):
        cache = generate_stellar_grid(
            tmp_path / 'stellar.npz', logg=STELLAR_PARAMS['logg_star'],
            feh=STELLAR_PARAMS['feh_star'], temperatures=np.arange(2300., 5001., 100.))
        grid = load_stellar_grid(cache)
        waves = grid.wavelengths_m
        phot, spot = [grid.interpolate(t, STELLAR_PARAMS['logg_star'],
                                       STELLAR_PARAMS['feh_star'])
                      for t in (STELLAR_PARAMS['T_star'], STELLAR_PARAMS['T_het'])]
        f = STELLAR_PARAMS['f_het']
        correction = phot / ((1 - f) * phot + f * spot)
        np.savetxt(tmp_path / 'stellar_tls.csv', np.column_stack((waves, correction)),
                   delimiter=',', header='wavelength_m,tls_factor')
    assert len(waves) > 19000
    assert cache.stat().st_size < 8 * 2**20
    assert np.all(np.isfinite(correction))
    assert correction.max() - correction.min() > .01
    assert (tmp_path / 'stellar_tls.csv').exists()
    grid = load_stellar_grid(cache)
    assert grid.loggs.tolist() == [5.5]
    assert grid.fehs.tolist() == [.5]
    assert grid.temperatures[0] == 2300.
    assert grid.temperatures[-1] == 5000.


@pytest.mark.real_stellar_grid
def test_entire_heterogeneity_prior_has_stellar_support():
    grid = load_stellar_grid()
    # Heterogeneous regions may be hotter or colder than the photosphere. Test
    # endpoints, each source node, and each intervening cell at the star's
    # round gravity/metallicity so no hidden prior truncation slips in.
    temperatures = np.arange(2300., 4501., 50.)
    for temp in temperatures:
        grid.validate(temp, 5.5, .5)


def test_legacy_stellar_pickle_supports_the_given_stellar_properties(tmp_path):
    reference = json.loads(FIXTURE.read_text())
    path = tmp_path / 'legacy_star.pkl'
    # Keep the original 2D layout and omitted axes, with real data rather
    # than assuming a fictitious solar-metallicity/logg4.5 anchor.
    path.write_bytes(pickle.dumps(reference['legacy_pickle_subset']))
    grid = load_stellar_grid(path)
    np.testing.assert_allclose(grid.interpolate(3000., 5.5, .5),
                               reference['legacy_photosphere_flux'], rtol=1e-7)
    np.testing.assert_allclose(grid.interpolate(2550., 5.5, .5),
                               reference['legacy_spot_flux'], rtol=1e-7)


@pytest.fixture
def tls_calculator(tiny_calculator, monkeypatch):
    """Real public constructor/JIT; replace only external atmospheric inputs."""
    from platon import _atmosphere_solver as module
    from platon.transit_depth_calculator import TransitDepthCalculator
    monkeypatch.setattr(module, 'load_stellar_grid', load_stellar_grid)
    return TransitDepthCalculator()


@pytest.mark.real_stellar_grid
def test_transit_tls_jit_and_host_agree_with_default_bundle(tls_calculator):
    calc = tls_calculator
    profile = tiny_profile(390.)
    args = (profile, RS, MP, RP)
    _, clear, _ = calc.compute_depths(
        *args, **dict(STELLAR_PARAMS, f_het=0.), stellar_grid_only=True)
    _, tls, info = calc.compute_depths(
        *args, **STELLAR_PARAMS, stellar_grid_only=True, full_output=True)
    _, correction = calc.atm.get_stellar_spectrum(
        **STELLAR_PARAMS, stellar_grid_only=True)
    np.testing.assert_allclose(tls, clear * correction, rtol=3e-6)
    np.testing.assert_allclose(info['unbinned_correction_factors'], correction, rtol=3e-6)
    assert np.all(np.isfinite(tls))
    assert 'stellar_spectra' not in calc.atm.device_data()._fields
    # This example has a large transit; this catches mixing up Earth/Jupiter
    # units or converting depth fractions into percent inside the model.
    geometric_depth = (.80 * R_jup / (.25 * R_sun))**2
    assert np.median(clear) > .8 * geometric_depth
    assert np.median(clear) < 1.3 * geometric_depth


@pytest.mark.real_stellar_grid
def test_forward_default_and_generated_cache_agree(tls_calculator, stellar_cache, tmp_path):
    # Keep only observation bins containing samples from the tiny opacity grid.
    edges = np.geomspace(.6e-6, 5.3e-6, 160)
    bins = np.column_stack((edges[:-1], edges[1:]))
    waves = tls_calculator.atm.orig_lambda_grid
    keep = [lo >= waves.min() and hi <= waves.max() and
            np.any((waves > lo) & (waves < hi)) for lo, hi in bins]
    bins = bins[np.asarray(keep)]
    profile = tiny_profile(390.)

    def spectra(calc):
        calc.change_wavelength_bins(bins)
        wavelengths, depths, _ = calc.compute_depths(
            profile, RS, MP, RP, **STELLAR_PARAMS, stellar_grid_only=True)
        _, clean, _ = calc.compute_depths(
            profile, RS, MP, RP, **dict(STELLAR_PARAMS, f_het=0.),
            stellar_grid_only=True)
        return wavelengths, depths, clean

    wavelengths, depths, clean = spectra(tls_calculator)
    assert np.all(np.isfinite(depths))
    assert np.all(np.isfinite(clean))
    assert np.all(depths > clean)
    np.savetxt(tmp_path / 'transmission_tls.csv',
               np.column_stack((wavelengths, depths, clean)), delimiter=',',
               header='wavelength_m,tls_depth,clean_depth')
    assert (tmp_path / 'transmission_tls.csv').exists()
    _, cached, cached_clean = spectra(TransitDepthCalculator(stellar_grid=stellar_cache))
    np.testing.assert_allclose(cached, depths, rtol=3e-6)
    np.testing.assert_allclose(cached_clean, clean, rtol=3e-6)


@pytest.mark.real_stellar_grid
def test_tls_priors_relative_offsets_and_short_dynesty(tls_calculator, tmp_path, monkeypatch):
    from platon.combined_retriever import CombinedRetriever
    # Two instruments of two bins each; the second carries a fitted offset.
    bins = np.array([[.8, .9], [.9, 1.], [3., 3.1], [3.1, 3.2]]) * 1e-6
    depths = np.array([.15, .1498, .15, .1498])
    errors = np.array([300e-6, 400e-6, 300e-6, 400e-6])
    retriever = CombinedRetriever()
    fit = retriever.get_default_fit_info(
        RS, MP, RP, T=390., **STELLAR_PARAMS, stellar_grid_only=True,
        transit_offsets={'offset_nirspec': (2, 4)})
    fit.add_gaussian_fit_param('T_star', 50.)
    fit.add_uniform_fit_param('T_het', 2300., 4500.)
    fit.add_uniform_fit_param('f_het', 0., .3)
    fit.add_uniform_fit_param('offset_nirspec', -5e-4, 5e-4)
    assert fit.all_params['T_het'].best_guess == 2550.
    assert fit.all_params['stellar_grid'].best_guess == 'newera'
    assert fit.all_params['stellar_grid_only'].best_guess is True
    assert 'offset_niriss' not in fit.all_params
    assert {'T_star', 'T_het', 'f_het'}.issubset(fit.fit_param_names)
    # Run the real sampler and public calculator on synthetic atmospheric data.
    # Keep the driver's parameter-estimate output in the test directory.
    monkeypatch.chdir(tmp_path)
    result = retriever.run_dynesty(
        bins, depths, errors, None, None, None, fit,
        nlive=12, maxcall=20, num_final_samples=2,
        rstate=np.random.default_rng(123))
    assert result.retrieval_type == 'dynesty'
    assert result.samples.shape[1] == len(fit.fit_param_names)
    assert np.all(np.isfinite(result.best_fit_transit_depths))
    assert np.all(np.isfinite(result.logl))
    np.testing.assert_allclose(result.weights.sum(), 1.)
    path = tmp_path / 'retrieval_result.pkl'
    path.write_bytes(pickle.dumps(result))
    assert path.exists()
    np.testing.assert_allclose(pickle.loads(path.read_bytes()).best_fit_params,
                               result.best_fit_params)


@pytest.mark.real_stellar_grid
def test_stellar_contamination_example_completes(monkeypatch, tmp_path):
    import runpy
    import matplotlib
    example = Path(__file__).resolve().parents[1] / 'examples' / 'stellar_contamination_example.py'
    monkeypatch.setenv('MPLBACKEND', 'Agg')
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    monkeypatch.chdir(tmp_path)
    try:
        namespace = runpy.run_path(str(example), run_name='__main__')
        for name in ('clean', 'spotted', 'mixed'):
            assert np.all(np.isfinite(namespace[name]))
    finally:
        plt.close('all')


@pytest.fixture
def real_opacity_calculator(monkeypatch, tmp_path):
    """Real eight-gas opacity subset; no large archive or source repo required."""
    from platon import _atmosphere_solver as module
    from platon.transit_depth_calculator import TransitDepthCalculator
    path = Path(__file__).parent / 'fixtures' / 'eight_gas_guillot_opacity.npz'
    provenance = json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance['fixture_sha256']
    with np.load(path, allow_pickle=False) as fixture:
        raw = json.loads(str(fixture['metadata']))
        raw.update({name: fixture[name] for name in fixture.files
                    if name != 'metadata' and not name.startswith('legacy_')})
        raw['key'] = tuple(raw['key'])
        assert raw['key'] == ('eight-gas-opacity-fixture',)
        legacy = dict(temperatures=fixture['legacy_temperatures'],
                      wavelengths_m=fixture['legacy_wavelengths_m'],
                      spectra=fixture['legacy_spectra'])
    stellar_path = tmp_path / 'legacy_stellar.npz'
    np.savez_compressed(stellar_path, **legacy)
    # VMR mode never consumes equilibrium abundances. Supply only the small
    # unused grid required by constructor metadata; all actual opacities,
    # masses, polarizabilities, source axes, and reference spectra are real.
    eq = np.empty((2, 2, 2, len(raw['T_grid']), len(raw['P_grid'])), np.float32)
    eq[:] = np.log10(np.array([.86, .14]))[None, None, :, None, None]
    getter = SimpleNamespace(log_abundances=eq, included_species=['H2', 'He'],
                             logZs=np.array([-2., 3.]), CO_ratios=np.array([.001, 2.]),
                             min_temperature=100.)
    monkeypatch.setattr(module, 'get_data_if_needed', lambda: None)
    monkeypatch.setattr(module, '_load_raw', lambda *args: raw)
    monkeypatch.setattr(module, 'AbundanceGetter', lambda *args: getter)
    monkeypatch.setattr(module, 'load_dict_from_pickle', lambda *args: {})
    monkeypatch.setattr(module, 'load_numpy', lambda *args: np.geomspace(1e-9, 1e-3, 16))
    monkeypatch.setattr(module, '_DEVICE_CACHE', {})
    monkeypatch.setattr(module, '_LOG_ABUND_CACHE', {})
    return TransitDepthCalculator(include_opacities=list(TRACE_GASES[:-1]),
                                  stellar_grid=stellar_path)


def _guillot_with_T_irr(T_irr, log_gamma, log_k_th, T_int):
    """Guillot profile with beta=1 and an orbit giving the requested T_irr."""
    from platon.TP_profile import Profile
    T_star = 5000.
    a = RS / 2 * (T_star / T_irr)**2
    return Profile.guillot(T_star, RS, a, MP, RP, 1., log_k_th, log_gamma, T_int)


def test_real_eight_gas_guillot_tls_matches_independent_fp32_reference(real_opacity_calculator):
    from platon.TP_profile import Profile
    reference = json.loads((FIXTURE.parent / 'eight_gas_guillot_opacity.json').read_text())
    calc = real_opacity_calculator
    calc.change_wavelength_bins(np.asarray(reference['wavelength_bins_m']))
    # Reference uses log10(cm^2/g); public uses log10(m^2/kg).
    profile = _guillot_with_T_irr(500., -.75, -.75 - 1, 200.)
    traces = 10**TRACE_LOG_VMRS
    vmrs = np.r_[traces, 1 - traces.sum()]
    _, depths, info = calc.compute_depths(
        profile, RS, MP, RP, logZ=None, CO_ratio=None,
        gases=list(TRACE_GASES), vmrs=vmrs, **STELLAR_PARAMS,
        stellar_grid_only=True, validate_T_grid=False, full_output=True)
    # Independently computed FP32 JAX reference using per-species opacity
    # interpolation, rather than values generated by the tested calculator.
    np.testing.assert_allclose(depths, reference['depths'], rtol=0, atol=.2e-6)
    np.testing.assert_allclose(profile.temperatures, reference['T_profile'], rtol=0, atol=.001)
    assert profile.temperatures.max() > calc.atm.max_temperature
    assert 'CS2' in info['atm_abundances']
    assert np.all(np.isfinite(depths))


@pytest.mark.real_stellar_grid
def test_real_eight_gas_guillot_works_with_default_newera(real_opacity_calculator):
    from platon.TP_profile import Profile
    reference = json.loads((FIXTURE.parent / 'eight_gas_guillot_opacity.json').read_text())
    calc = real_opacity_calculator
    calc.atm.stellar_grid = load_stellar_grid()
    calc.atm._stellar_key = ('bundled-newera-real-opacity',)
    calc.change_wavelength_bins(np.asarray(reference['wavelength_bins_m']))
    profile = _guillot_with_T_irr(500., -.75, -1.75, 200.)
    traces = 10**TRACE_LOG_VMRS
    _, depths, info = calc.compute_depths(
        profile, RS, MP, RP, logZ=None, CO_ratio=None,
        gases=list(TRACE_GASES), vmrs=np.r_[traces, 1 - traces.sum()],
        **STELLAR_PARAMS, stellar_grid_only=True,
        validate_T_grid=False, full_output=True)
    assert np.all(np.isfinite(depths))
    assert np.all((depths > .1) & (depths < .3))
    assert np.all(info['unbinned_correction_factors'] > 1)
    assert 'stellar_spectra' not in calc.atm.device_data()._fields
