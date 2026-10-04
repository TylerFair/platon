"""Stellar-grid tests need no atmospheric data or network access."""
import json
import pickle
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from misc.build_newera_grid import conservative_rebin, encode_log_flux
from platon import _forward_model as fm
from platon._forward_prep import _pack_scalars, prepare_forward_inputs
from platon._stellar_grid import (grid_from_dict, load_stellar_grid,
                                 stellar_components, wavelength_brackets,
                                 ShardedSpectra)
from platon._atmosphere_solver import AtmosphereSolver
from platon.errors import AtmosphereError
from tests._support import synthetic_grid, make_native_h5, tiny_profile



def stellar_device(grid, lambdas):
    idx, frac = wavelength_brackets(lambdas, grid.wavelengths_m)
    dd = fm.DeviceData(**{name: None for name in fm.DeviceData._fields})
    return dd._replace(
        lambda_grid=jnp.asarray(lambdas), orig_lambda_grid=jnp.asarray(lambdas),
        stellar_lambdas=jnp.asarray(grid.wavelengths_m),
        stellar_wave_idx=jnp.asarray(idx), stellar_wave_frac=jnp.asarray(frac),
        orig_stellar_wave_idx=jnp.asarray(idx), orig_stellar_wave_frac=jnp.asarray(frac))


def config(**kwargs):
    values = dict(n_layers=2, abund_mode='vmr', gas_master_idx=(), ch4_idx=-1,
                  el_idx=-1, h_idx=-1, add_gas=False, add_hminus=False,
                  add_scattering=False, add_collisional=False, sort_layers=False,
                  use_mie=False, has_t_star=True, stellar_in_grid=True,
                  has_spots=True, spot_in_grid=True, has_faculae=True, fac_in_grid=True)
    values.update(kwargs)
    return fm.ForwardConfig(**values)


def host_atmosphere(grid, lambdas):
    atm = AtmosphereSolver.__new__(AtmosphereSolver)
    atm.stellar_grid = grid
    atm.lambda_grid = np.asarray(lambdas)
    atm.orig_lambda_grid = np.asarray(lambdas)
    return atm


def test_trilinear_interpolation_on_nonuniform_axes():
    grid = synthetic_grid()
    flux = grid.interpolate(4200, 4.3, -.2)
    expected = 4200e6 + 4.3e9 - .2e9 + grid.wavelengths_m * 1e15
    np.testing.assert_allclose(flux, expected, rtol=2e-7)
    for idx in np.ndindex(2, 2, 2):
        np.testing.assert_array_equal(grid.interpolate(grid.temperatures[idx[0]],
                                                     grid.loggs[idx[1]], grid.fehs[idx[2]]),
                                      grid.spectra[idx])


@pytest.mark.parametrize('point', [(2000, 4.5, 0), (4000, 3.5, 0), (4000, 4.5, 1),
                                  (np.nan, 4.5, 0), (4000, np.inf, 0)])
def test_bounds_reject_extrapolation(point):
    with pytest.raises(AtmosphereError):
        synthetic_grid().interpolate(*point)


def test_missing_nodes_only_rejected_when_their_weight_is_nonzero():
    grid = synthetic_grid()
    valid = np.ones((2, 2, 2), bool)
    valid[1, 1, 1] = False
    grid = grid_from_dict(dict(temperatures=grid.temperatures, loggs=grid.loggs,
                              fehs=grid.fehs, wavelengths_m=grid.wavelengths_m,
                              spectra=grid.spectra, valid=valid))
    grid.interpolate(3000, 4., -.5)  # unused neighboring void is allowed
    with pytest.raises(AtmosphereError, match='missing') as exc:
        grid.interpolate(4000, 4.5, 0)
    assert '(Teff, logg, [Fe/H]) = (4000 K, 4.5, 0)' in str(exc.value)
    assert 'nodes: (5000 K, 5, 0.5)' in str(exc.value)


@pytest.mark.parametrize('component, params', [
    ('T_star', dict(T_star=2200., T_spot=None, spot_cov_frac=0.)),
    ('T_spot', dict(T_star=4000., T_spot=2200., spot_cov_frac=.1)),
    ('T_fac', dict(T_star=4000., T_spot=None, spot_cov_frac=0.,
                   T_fac=2200., fac_cov_frac=.1)),
])
def test_blackbody_fallback_warns_once_and_strict_errors_name_component(
        monkeypatch, component, params):
    monkeypatch.setattr('platon._stellar_grid._blackbody_warnings', set())
    grid = synthetic_grid()
    expected = f'{component} = 2200 K is outside the stellar grid (3000-5000 K)'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        stellar_components(grid, **params, blackbody=True)
        with pytest.raises(AtmosphereError) as exc:
            stellar_components(grid, **params, grid_only=True)
        assert str(exc.value) == expected
        assert not caught
        for temperature in (2200., 2100., 2200.):
            stellar_components(grid, **dict(params, **{component: temperature}))
    assert len(caught) == 1
    assert caught[0].category is UserWarning
    message = str(caught[0].message)
    assert expected in message
    assert 'using a blackbody spectrum' in message
    assert 'stellar_grid_only=True raises an error instead' in message


def test_missing_model_error_lists_all_required_nodes():
    grid = synthetic_grid()
    valid = np.ones((2, 2, 2), bool)
    valid[0, 1, 1] = valid[1, 1, 1] = False
    grid = grid_from_dict(dict(temperatures=grid.temperatures, loggs=grid.loggs,
                              fehs=grid.fehs, wavelengths_m=grid.wavelengths_m,
                              spectra=grid.spectra, valid=valid))
    with pytest.raises(AtmosphereError) as exc:
        grid.validate(4200., 4.3, -.2)
    message = str(exc.value)
    assert '(Teff, logg, [Fe/H]) = (4200 K, 4.3, -0.2)' in message
    assert 'nodes: (3000 K, 5, 0.5), (5000 K, 5, 0.5)' in message


@pytest.mark.parametrize('bad', [np.array([3000., 3000.]), np.array([5000., 3000.]),
                               np.array([np.nan, 5000.])])
def test_malformed_axes_rejected(bad):
    with pytest.raises(ValueError):
        grid_from_dict(dict(temperatures=bad, wavelengths_m=[1e-6, 2e-6],
                            spectra=np.ones((2, 2))))


def test_constant_axis_and_reference_pickle_conventions(tmp_path):
    data = dict(temperatures=[3000., 5000.], loggs=[4., 5.],
                wavelengths_m=[1e-6, 2e-6, 3e-6], spectra=np.arange(12.).reshape(2, 2, 3) + 1)
    path = tmp_path / 'custom.pkl'
    path.write_bytes(pickle.dumps(data))
    grid = load_stellar_grid(path)
    # Legacy grids use (logg, T, wavelength), including ambiguous equal-size axes.
    np.testing.assert_array_equal(grid.interpolate(5000., 4., 0.), data['spectra'][0, 1])
    assert load_stellar_grid(path) is grid
    single = grid_from_dict(dict(temperatures=[4000.], wavelengths_m=[1e-6, 2e-6], spectra=[[1., 2.]]))
    np.testing.assert_array_equal(single.interpolate(4000.), [1, 2])


def test_flux_conserving_rebin_preserves_lines_and_partial_bins():
    source = np.array([0., 1., 2., 3., 4.])
    values = np.array([1., 100., 2., 3.])
    edges = np.array([.5, 1.5, 3.5])
    expected = [50.5, 26.75]
    np.testing.assert_allclose(conservative_rebin(values, source, edges), expected)
    whole = conservative_rebin(values, source, [0., 2., 4.])
    assert whole @ np.array([2., 2.]) == values.sum()
    with pytest.raises(ValueError):
        conservative_rebin(values, source, [-1., 2.])


def test_uint16_encoding_precision_and_dynamic_range():
    flux = np.geomspace(1e-5, 1e14, 1000)
    q, offset, scale = encode_log_flux(flux)
    assert q.dtype == np.uint16
    np.testing.assert_allclose(np.exp(offset + q.astype(float) * scale), flux, rtol=4e-4)
    q, offset, scale = encode_log_flux(np.full(5, 1e12))
    np.testing.assert_allclose(np.exp(offset + q.astype(float) * scale), 1e12, rtol=2e-6)
    with pytest.raises(ValueError):
        encode_log_flux(np.array([0., 1.]))


@pytest.mark.parametrize('encoded', [False, True])
def test_numpy_jit_agree_for_spots_faculae_and_wavelength_tails(encoded):
    grid = synthetic_grid(encoded)
    waves = np.array([.4, .7, .9, 1.3, 4., 5., 8.]) * 1e-6
    atm = host_atmosphere(grid, waves)
    expected, corr = atm.get_stellar_spectrum(4200., 3200., .1, T_fac=4800.,
                                            fac_cov_frac=.05, logg_phot=4.3,
                                            logg_spot=4.1, logg_fac=4.8, feh=-.2)
    sc = jnp.asarray(_pack_scalars(t_star=4200., t_spot=3200., t_fac=4800.,
                                  spot_frac=.1, fac_frac=.05, logg_phot=4.3,
                                  logg_spot=4.1, logg_fac=4.8, feh=-.2), dtype=jnp.float32)
    dd = stellar_device(grid, waves)
    run = jax.jit(fm._stellar_spectrum, static_argnums=(0,))
    rows = np.array([grid.interpolate(t, g, -.2) for t, g in
                     ((4200., 4.3), (3200., 4.1), (4800., 4.8))], np.float32)
    spectrum, correction = run(config(), dd, sc, stellar_fluxes=rows)
    np.testing.assert_allclose(spectrum, expected, rtol=4e-6)
    np.testing.assert_allclose(correction, corr, rtol=4e-6)


def test_components_validate_active_temperatures_and_fractions(monkeypatch):
    grid = synthetic_grid()
    # An unused spot does not require a valid temperature or logg.
    result = stellar_components(grid, 4000., -100., 0., logg_phot=4.5)
    assert result[-1] == (True, False, False)
    for spot, fac in ((-.1, 0), (.8, .3), (np.nan, 0)):
        with pytest.raises(AtmosphereError):
            stellar_components(grid, 4000., 3500., spot, 4500., fac)
    with pytest.raises(AtmosphereError):
        stellar_components(grid, 4000., -100., .1)
    with pytest.raises(AtmosphereError):
        stellar_components(grid, None, 3500., .1)
    with pytest.raises(AtmosphereError):
        stellar_components(grid, 7000., None, 0., grid_only=True)
    monkeypatch.setattr('platon._stellar_grid._blackbody_warnings', set())
    with pytest.warns(UserWarning, match='outside the stellar grid'):
        assert stellar_components(grid, 7000., None, 0.)[-1] == (False, False, False)


def test_no_tls_and_blackbody_paths():
    grid = synthetic_grid()
    atm = host_atmosphere(grid, np.array([1., 2.]) * 1e-6)
    spectrum, corr = atm.get_stellar_spectrum(None, None, None)
    np.testing.assert_array_equal(spectrum, [1, 1])
    np.testing.assert_array_equal(corr, [1, 1])
    spectrum, corr = atm.get_stellar_spectrum(4000., None, 0.)
    np.testing.assert_array_equal(corr, [1, 1])
    bb, corr = atm.get_stellar_spectrum(4000., 3500., .2, blackbody=True)
    phot = np.pi * fm.planck_np(atm.lambda_grid, 4000.)
    spot = np.pi * fm.planck_np(atm.lambda_grid, 3500.)
    np.testing.assert_allclose(bb, .8 * phot + .2 * spot)
    np.testing.assert_allclose(corr, phot / bb)


def test_custom_grid_reaches_calculator_constructor():
    with patch('platon.transit_depth_calculator.AtmosphereSolver') as cls:
        from platon.transit_depth_calculator import TransitDepthCalculator
        TransitDepthCalculator(stellar_grid='custom.npz')
    assert cls.call_args.kwargs['stellar_grid'] == 'custom.npz'


def test_likelihood_passes_stellar_parameters_and_rejects_missing_models():
    from platon.combined_retriever import CombinedRetriever
    from platon.constants import R_sun, M_jup, R_jup
    fit = CombinedRetriever.get_default_fit_info(R_sun, M_jup, R_jup, T=1000.,
        T_star=4200., T_spot=3200., spot_cov_frac=.1, T_fac=4800., fac_cov_frac=.05,
        logg_phot=4.3, logg_spot=4.1, logg_fac=4.8, feh=-.2, stellar_grid_only=True)
    captured = {}
    class Calculator:
        def compute_depths(self, *args, **kwargs):
            captured.update(kwargs)
            return np.array([1e-6]), np.array([.01]), {}
    retriever = CombinedRetriever()
    value = retriever._ln_like([], Calculator(), None, fit, np.array([.01]),
                              np.array([1e-4]), None, None)
    assert np.isfinite(value)
    for name in ('T_fac', 'fac_cov_frac', 'logg_phot', 'logg_spot', 'logg_fac', 'feh', 'stellar_grid_only'):
        assert captured[name] == fit._get(name)
    class InvalidCalculator:
        def compute_depths(self, *args, **kwargs):
            raise AtmosphereError('missing NewEra model')
    assert retriever._ln_like([], InvalidCalculator(), None, fit, np.array([.01]),
                             np.array([1e-4]), None, None) == -np.inf


def test_shards_load_lazily_and_evict_without_breaking_pickling(tmp_path):
    shape = (2, 2, 3, 4)
    paths = []
    for z in range(3):
        path = tmp_path / f'{z}.npz'
        np.savez_compressed(path, encoded_spectra=np.full((2, 2, 4), z, np.uint16))
        paths.append(path)
    spectra = ShardedSpectra(paths, shape)
    assert spectra.resident_nbytes == 0
    for z in range(3):
        np.testing.assert_array_equal(spectra[0, 1, z], np.full(4, z))
    assert len(spectra._cache) == 2
    assert spectra.resident_nbytes == 64
    restored = pickle.loads(pickle.dumps(spectra))
    assert restored.resident_nbytes == 0
    np.testing.assert_array_equal(restored[1, 0, 0], np.zeros(4))


@pytest.mark.real_stellar_grid
def test_bundled_grid_axes_nodes_and_provenance():
    grid = load_stellar_grid()
    assert (grid.temperatures[0], grid.temperatures[-1]) == (2300, 12000)
    assert (grid.loggs[0], grid.loggs[-1]) == (0, 6)
    assert (grid.fehs[0], grid.fehs[-1]) == (-4, .5)
    assert grid.valid.sum() == 8339
    assert grid.metadata['license'] == 'CC-BY-4.0'
    assert grid.metadata['resolving_power'] >= 5000
    assert np.all(grid.interpolate(5000., 4.5, 0.) > 0)
    assert grid.spectra.resident_nbytes < 80 * 1024**2


def test_generate_subset_is_standalone_without_pymsg(tmp_path):
    from platon.stellar_grid import generate_stellar_grid
    grid = synthetic_grid()
    # A custom canonical 4D input avoids any dependency on bundled files.
    source = tmp_path / 'source.npz'
    np.savez(source, temperatures=grid.temperatures, loggs=grid.loggs, fehs=grid.fehs,
             wavelengths_m=grid.wavelengths_m, spectra=grid.spectra)
    with patch.dict('sys.modules', {'pymsg': None}):
        cache = generate_stellar_grid(tmp_path / 'star.npz', source=source,
                                      temperatures=[3200., 4600.], loggs=[4.1, 4.7], fehs=[-.2, .3])
        subset = load_stellar_grid(cache)
    np.testing.assert_allclose(subset.interpolate(4000., 4.5, .1),
                               grid.interpolate(4000., 4.5, .1), rtol=2e-7)
    assert subset.spectra.shape == (2, 2, 2, 4)
    assert subset.metadata['generated_from'] == source.name
    assert str(tmp_path) not in json.dumps(subset.metadata)
    with pytest.raises(ValueError, match='downsample'):
        generate_stellar_grid(tmp_path / 'bad.npz', source=source, downsample=10)
    with pytest.raises(AtmosphereError):
        generate_stellar_grid(tmp_path / 'bad.npz', source=source, temperatures=[6000.])
    assert not (tmp_path / 'bad.npz').exists()



def test_native_generation_and_block_averaging_need_no_pymsg(tmp_path):
    from platon.stellar_grid import generate_stellar_grid
    from platon._stellar_grid import load_native_stellar_grid
    source = tmp_path / 'native.h5'
    make_native_h5(source)
    with patch.dict('sys.modules', {'pymsg': None}):
        for factor in (1, 2, 3, 10, 20):
            native = load_native_stellar_grid(source, factor)
            original = np.arange(1., 41.) * 1.5 * np.pi * 1e7
            expected = [np.mean(original[i:i + factor]) for i in range(0, 40, factor)]
            np.testing.assert_allclose(native.interpolate(4000.), expected, rtol=2e-7)
            assert len(native.wavelengths_m) == int(np.ceil(40 / factor))
        cache = generate_stellar_grid(tmp_path / 'star_native.npz', source=source,
                                      temperatures=[3000., 4000., 5000.], downsample=2)
        subset = load_stellar_grid(cache)
    assert subset.spectra.shape == (3, 1, 1, 20)
    assert subset.metadata['generated_from'] == source.name
    assert str(tmp_path) not in json.dumps(subset.metadata)
    np.testing.assert_allclose(subset.interpolate(4000.),
                               np.arange(1.5, 40., 2.) * 1.5 * np.pi * 1e7, rtol=2e-7)




def test_transit_core_applies_tls_and_keeps_stellar_cube_off_device(tiny_calculator):
    from platon.constants import R_sun, R_jup, M_jup
    calc = tiny_calculator
    profile = tiny_profile()
    args = (profile, R_sun, M_jup, R_jup)
    params = dict(T_star=4200., logg_phot=4.3, feh=-.2)
    _, clean, _ = calc.compute_depths(*args, **params)
    _, tls, info = calc.compute_depths(*args, **params, T_spot=3200., spot_cov_frac=.1,
                                      T_fac=4800., fac_cov_frac=.05, full_output=True)
    _, expected = calc.atm.get_stellar_spectrum(4200., 3200., .1, T_fac=4800.,
                                               fac_cov_frac=.05, logg_phot=4.3, feh=-.2)
    np.testing.assert_allclose(tls, clean * expected, rtol=2e-6)
    np.testing.assert_allclose(info['unbinned_correction_factors'], expected, rtol=2e-6)
    assert 'stellar_spectra' not in calc.atm.device_data()._fields
    assert np.all(np.isfinite(tls))


def test_binning_reset_and_partial_clouds_keep_stellar_correction(tiny_calculator):
    from platon.constants import R_sun, R_jup, M_jup
    calc = tiny_calculator
    args = (tiny_profile(), R_sun, M_jup, R_jup)
    params = dict(T_star=4200., T_spot=3200., spot_cov_frac=.1,
                  logg_phot=4.3, feh=-.2, cloudtop_pressure=1e4)
    bins = np.array([[.9, 1.3], [2., 3.]]) * 1e-6
    calc.change_wavelength_bins(bins)
    _, cloudy, _ = calc.compute_depths(*args, **params, cloud_fraction=1.)
    _, clear, _ = calc.compute_depths(*args, **params, cloud_fraction=0.)
    _, mixed, _ = calc.compute_depths(*args, **params, cloud_fraction=.4)
    np.testing.assert_allclose(mixed, .4 * cloudy + .6 * clear, rtol=2e-6)
    calc.change_wavelength_bins(None)
    assert len(calc.compute_depths(*args, **params)[1]) == 128
    calc.change_wavelength_bins(bins)
    np.testing.assert_allclose(calc.compute_depths(*args, **params)[1], cloudy, rtol=2e-6)


def test_eclipse_core_accepts_interpolated_components(tiny_calculator):
    from platon.eclipse_depth_calculator import EclipseDepthCalculator
    from platon.constants import R_sun, R_jup, M_jup
    calc = EclipseDepthCalculator.__new__(EclipseDepthCalculator)
    calc.atm = tiny_calculator.atm
    profile = tiny_profile()
    params = dict(logg_phot=4.3, feh=-.2)
    args = (profile, R_sun, M_jup, R_jup, 4200.)
    _, clean, _ = calc.compute_depths(*args, **params)
    _, tls, _ = calc.compute_depths(*args, **params, T_spot=3200., spot_cov_frac=.1)
    _, correction = calc.atm.get_stellar_spectrum(4200., 3200., .1, logg_phot=4.3, feh=-.2)
    np.testing.assert_allclose(tls, clean * correction, rtol=3e-6)
    assert np.all(np.isfinite(tls))


def test_binned_tls_matches_explicit_weighted_spectrum(tiny_calculator):
    from platon.constants import R_sun, R_jup, M_jup
    calc = tiny_calculator
    args = (tiny_profile(), R_sun, M_jup, R_jup)
    params = dict(T_star=4200., T_spot=3200., spot_cov_frac=.1,
                  T_fac=4800., fac_cov_frac=.05, logg_phot=4.3, feh=-.2)
    waves, unbinned, info = calc.compute_depths(*args, **params, full_output=True)
    bins = np.array([[.9, 1.3], [2., 3.]]) * 1e-6
    expected = []
    for lo, hi in bins:
        keep = (waves > lo) & (waves < hi)
        expected.append(np.average(unbinned[keep], weights=info['unbinned_stellar_spectrum'][keep]))
    calc.change_wavelength_bins(bins)
    _, binned, _ = calc.compute_depths(*args, **params)
    np.testing.assert_allclose(binned, expected, rtol=2e-6)


def test_two_sector_recursion_forwards_stellar_components(tiny_calculator):
    from platon.constants import R_sun, R_jup, M_jup
    from platon.terminator import TerminatorSector, TwoSectorTerminator
    calc = tiny_calculator
    cold, hot = tiny_profile(), tiny_profile()
    cold.set_isothermal(800.)
    hot.set_isothermal(1400.)
    model = TwoSectorTerminator(TerminatorSector(cold), TerminatorSector(hot), .3)
    params = dict(T_star=4200., T_spot=3200., spot_cov_frac=.1,
                  T_fac=4800., fac_cov_frac=.05, logg_phot=4.3,
                  logg_spot=4.1, logg_fac=4.8, feh=-.2)
    args = (R_sun, M_jup, R_jup)
    _, a, _ = calc.compute_depths(cold, *args, **params)
    _, b, _ = calc.compute_depths(hot, *args, **params)
    _, mix, _ = calc.compute_depths(model, *args, **params)
    np.testing.assert_allclose(mix, .3 * a + .7 * b, rtol=2e-6)


def test_offline_native_bundle_is_usable_without_hdf5_or_pymsg(tmp_path):
    from misc.build_newera_grid import build
    source = tmp_path / 'source.h5'
    make_native_h5(source)
    dest = tmp_path / 'native.npz'
    metadata = build(source, dest, native_downsample=1)
    assert metadata['native_downsample'] == 1
    assert metadata['resolving_power'] is None
    with patch.dict('sys.modules', {'h5py': None, 'pymsg': None}):
        grid = load_stellar_grid(dest)
        flux = grid.interpolate(4000.)
    np.testing.assert_allclose(flux, np.arange(1., 41.) * 1.5 * np.pi * 1e7, rtol=4e-5)
    assert len(grid.wavelengths_m) == 40
