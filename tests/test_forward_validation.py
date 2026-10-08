"""Regression coverage for host validation and finite-temperature clamping."""
import numpy as np
import pytest

from platon.constants import R_sun, M_jup, R_jup, k_B
from platon.errors import AtmosphereError
from platon.fit_info import FitInfo
from platon._params import _GaussianParam, _UniformParam
from platon._stellar_grid import load_stellar_grid, stellar_components
from platon.stellar_grid import generate_stellar_grid
from tests._support import tiny_profile, synthetic_grid


def run_transit(calc, profile=None, **kwargs):
    return calc.compute_depths(tiny_profile() if profile is None else profile,
                               R_sun, M_jup, R_jup, **kwargs)


@pytest.mark.parametrize('fraction', [np.nan, np.inf, -np.inf])
def test_nonfinite_cloud_fraction_rejected(tiny_calculator, fraction):
    with pytest.raises(ValueError, match='cloud_fraction'):
        run_transit(tiny_calculator, cloud_fraction=fraction)


@pytest.mark.parametrize('index', [0, 1, 2])
@pytest.mark.parametrize('value', [0., -1., np.nan, np.inf])
def test_nonphysical_mass_and_radii_rejected(tiny_calculator, index, value):
    args = [R_sun, M_jup, R_jup]
    args[index] = value
    with pytest.raises(AtmosphereError, match='finite and positive'):
        tiny_calculator.compute_depths(tiny_profile(), *args)


@pytest.mark.parametrize('pressures, temperatures', [
    ([1., 10.], [1000.]), ([1.], [1000.]), ([], []),
    ([[1., 10.]], [[1000., 1000.]]),
    ([-1., 10.], [1000., 1000.]), ([0., 10.], [1000., 1000.]),
    ([1., np.inf], [1000., 1000.]), ([1., np.nan], [1000., 1000.]),
    ([10., 1.], [1000., 1000.]),
])
def test_malformed_profiles_rejected_before_jit(tiny_calculator, pressures, temperatures):
    profile = tiny_profile()
    profile.pressures = np.asarray(pressures)
    profile.temperatures = np.asarray(temperatures)
    with pytest.raises(ValueError, match='P_profile'):
        run_transit(tiny_calculator, profile)


@pytest.mark.parametrize('temperature', [np.nan, np.inf, 0., -100.])
@pytest.mark.parametrize('validate_T_grid', [True, False])
def test_nonphysical_temperatures_always_rejected(tiny_calculator, temperature, validate_T_grid):
    profile = tiny_profile()
    profile.temperatures[-1] = temperature
    with pytest.raises(AtmosphereError, match='finite and positive'):
        run_transit(tiny_calculator, profile, validate_T_grid=validate_T_grid)


@pytest.mark.parametrize('kwargs', [dict(logZ=np.nan), dict(CO_ratio=np.nan),
                                    dict(cloudtop_pressure=np.nan), dict(cloudtop_pressure=-np.inf)])
def test_nonfinite_chemistry_and_cloud_pressure_rejected(tiny_calculator, kwargs):
    with pytest.raises(ValueError):
        run_transit(tiny_calculator, **kwargs)


@pytest.mark.parametrize('gases, vmrs', [
    (['H2', 'He'], [.9]), ([], []), (['H2', 'H2'], [.5, .5]),
    (['H2', 'He'], [.9, -.1]), (['H2', 'He'], [np.nan, .1]),
    (['H2', 'He'], [np.inf, .1]), (['H2', 'He'], [0., 0.]),
    (['H2', 'He'], [1.1, .1]), (['H2', 'He'], [[.9, .1]]),
])
def test_invalid_vmr_compositions_rejected(tiny_calculator, gases, vmrs):
    with pytest.raises(ValueError):
        run_transit(tiny_calculator, logZ=None, CO_ratio=None, gases=gases, vmrs=vmrs)


@pytest.mark.parametrize('abundance', [-.1, np.nan, np.inf, 1.1])
def test_invalid_custom_abundances_are_not_silently_floored(tiny_calculator, abundance):
    with pytest.raises(ValueError, match='finite fractions'):
        run_transit(tiny_calculator, logZ=None, CO_ratio=None,
                    custom_abundances={'H2': np.full(12, abundance)})


def test_zero_vmr_species_is_supported(tiny_calculator):
    _, depths, _ = run_transit(tiny_calculator, logZ=None, CO_ratio=None,
                              gases=['H2', 'He'], vmrs=[1., 0.])
    assert np.all(np.isfinite(depths))


def test_temperature_clamping_preserves_physical_profile_and_density(tiny_calculator):
    profile = tiny_profile()
    profile.temperatures = np.geomspace(1000., 4000., 12)
    with pytest.raises(AtmosphereError, match='Invalid temperatures'):
        run_transit(tiny_calculator, profile)
    # Give the gas an opacity that changes by 100x over its T grid. Above the
    # grid, cross sections must clamp while gas density still uses actual T.
    tiny_calculator.atm.raw['ln_xsec_stack'][0, 1, :, :] = np.log(1e-25)
    _, depths, info = run_transit(tiny_calculator, profile, validate_T_grid=False,
                                 add_scattering=False, add_collisional_absorption=False,
                                 full_output=True)
    np.testing.assert_allclose(info['T_profile'], profile.temperatures)
    weight = np.clip((1 / profile.temperatures - 1 / 300.) /
                     (1 / 2500. - 1 / 300.), 0., 1.)
    cross_section = np.exp((1 - weight) * np.log(1e-27) + weight * np.log(1e-25))
    expected = cross_section * .001 * profile.pressures / (k_B * profile.temperatures)
    np.testing.assert_allclose(info['absorption_coeff_atm'],
                               np.broadcast_to(expected[:, None], (12, 128)), rtol=1e-5)
    assert np.all(np.isfinite(depths))


def test_eclipse_temperature_validation_flag(tiny_calculator):
    from platon.eclipse_depth_calculator import EclipseDepthCalculator
    calc = EclipseDepthCalculator.__new__(EclipseDepthCalculator)
    calc.atm = tiny_calculator.atm
    profile = tiny_profile()
    profile.temperatures[-1] = 4000.
    args = (profile, R_sun, M_jup, R_jup, 4200.)
    with pytest.raises(AtmosphereError):
        calc.compute_depths(*args)
    _, depths, info = calc.compute_depths(*args, validate_T_grid=False, full_output=True)
    assert info['T_profile'][-1] == 4000.
    assert np.all(np.isfinite(depths))


def test_two_sector_temperature_validation_flag(tiny_calculator):
    from platon.terminator import TerminatorSector, TwoSectorTerminator
    from platon.TP_profile import Profile
    cold, hot = Profile.isothermal(1000.), Profile.isothermal(1000.)
    hot.temperatures[-1] = 4000.
    model = TwoSectorTerminator(TerminatorSector(cold), TerminatorSector(hot), .3)
    _, cold_depths, _ = run_transit(tiny_calculator, cold, validate_T_grid=False)
    _, hot_depths, _ = run_transit(tiny_calculator, hot, validate_T_grid=False)
    _, depths, _ = run_transit(tiny_calculator, model, validate_T_grid=False)
    np.testing.assert_allclose(depths, .3 * cold_depths + .7 * hot_depths, rtol=2e-6)


def test_blackbody_and_strict_grid_are_mutually_exclusive():
    with pytest.raises(ValueError, match='cannot both'):
        stellar_components(synthetic_grid(), 4000., None, 0., blackbody=True, grid_only=True)


def test_default_stellar_subset_preserves_grid_voids(tmp_path):
    grid = synthetic_grid()
    valid = np.ones((2, 2, 2), bool)
    valid[1, 1, 1] = False
    source = tmp_path / 'source.npz'
    np.savez(source, temperatures=grid.temperatures, loggs=grid.loggs,
             fehs=grid.fehs, wavelengths_m=grid.wavelengths_m,
             spectra=grid.spectra, valid=valid)
    generated = generate_stellar_grid(tmp_path / 'subset.npz', source=source,
                                      logg=4.5, feh=0.)
    subset = load_stellar_grid(generated)
    np.testing.assert_array_equal(subset.valid[:, 0, 0], [True, False])
    np.testing.assert_allclose(subset.interpolate(3000.), grid.interpolate(3000.))
    with pytest.raises(AtmosphereError, match='missing'):
        subset.interpolate(4000.)
    with pytest.raises(AtmosphereError, match='missing'):
        generate_stellar_grid(tmp_path / 'explicit.npz', source=source,
                              temperatures=[3000., 5000.], logg=4.5, feh=0.)
    assert not (tmp_path / 'explicit.npz').exists()


def test_gaussian_log_prior_retains_finite_tail():
    param = _GaussianParam(10., 2., None, None)
    expected = -.5 * 40**2 - np.log(2. * np.sqrt(2 * np.pi))
    assert param.ln_prior(90.) == pytest.approx(expected)
    for value in [np.nan, np.inf, -np.inf]:
        assert not param.within_limits(value)
        assert param.ln_prior(value) == -np.inf


@pytest.mark.parametrize('std', [0., -1., np.nan, np.inf])
def test_invalid_gaussian_prior_does_not_corrupt_fitinfo(std):
    info = FitInfo({'T': 1000.})
    with pytest.raises(ValueError):
        info.add_gaussian_fit_param('T', std)
    assert info.fit_param_names == []
    info.add_gaussian_fit_param('T', 100.)
    assert info.fit_param_names == ['T']


@pytest.mark.parametrize('low, high', [(1., 1.), (2., 1.), (np.nan, 1.), (0., np.nan)])
def test_invalid_uniform_prior_does_not_corrupt_fitinfo(low, high):
    info = FitInfo({'T': 1000.})
    with pytest.raises(ValueError):
        info.add_uniform_fit_param('T', low, high)
    assert info.fit_param_names == []
    info.add_uniform_fit_param('T', 500., 1500.)
    assert info.fit_param_names == ['T']


def test_unbounded_uniform_prior_remains_usable_for_emcee():
    param = _UniformParam(1., 0., np.inf, .5, 1.5)
    assert param.within_limits(1e10)
    with pytest.raises(ValueError, match='infinity'):
        param.from_unit_interval(.5)


def test_ordered_prior_requires_two_distinct_parameters_without_mutating():
    info = FitInfo({'T': 1000.})
    with pytest.raises(ValueError, match='different names'):
        info.add_ordered_uniform_fit_params('T', 'T', 500., 1500.)
    assert info.fit_param_names == []


def isolated_gap_grid(valid=None, policy='interpolate_isolated_temperature'):
    from platon._stellar_grid import grid_from_dict
    if valid is None:
        valid = np.ones((3, 2, 2), bool)
        valid[1, 1, 1] = False
    flux = np.broadcast_to(np.array([1., 100., 7.])[:, None, None, None], (3, 2, 2, 2)).copy()
    return grid_from_dict(dict(temperatures=[2400., 2500., 2700.], loggs=[4., 5.],
                               fehs=[-.5, .5], wavelengths_m=[1e-6, 2e-6],
                               spectra=flux, valid=valid,
                               metadata={'missing_model_policy': policy}))


def test_isolated_gap_repairs_whole_interpolated_row_in_linear_flux():
    grid = isolated_gap_grid()
    before = grid.valid.copy()
    # Nonuniform T spacing: 2500 is 1/3 of the way from 2400 to 2700.
    # Keeping the three valid corners of the missing row would give >75,
    # whereas replacement of the whole row gives exactly 3.
    np.testing.assert_allclose(grid.interpolate(2500., 4.5, 0.), [3., 3.])
    np.testing.assert_allclose(grid.interpolate(2600., 4.5, 0.), [5., 5.])
    # A missing neighboring model with zero weight still requires no repair.
    np.testing.assert_allclose(grid.interpolate(2500., 4., -.5), [100., 100.])
    np.testing.assert_array_equal(grid.valid, before)
    with pytest.raises(AtmosphereError, match='missing'):
        isolated_gap_grid(policy='reject').interpolate(2500., 4.5, 0.)


@pytest.mark.parametrize('rows', [[0], [2], [0, 1], [1, 2]])
def test_missing_endpoints_and_adjacent_temperature_rows_still_rejected(rows):
    valid = np.ones((3, 2, 2), bool)
    valid[rows, 1, 1] = False
    grid = isolated_gap_grid(valid)
    with pytest.raises(AtmosphereError, match='missing'):
        grid.interpolate(grid.temperatures[rows[0]], 4.5, 0.)
    with pytest.raises(AtmosphereError, match='outside'):
        grid.interpolate(2500., 5.1, 0.)


def test_public_generation_can_select_strict_or_repair_policy(tmp_path):
    grid = isolated_gap_grid(policy='reject')
    source = tmp_path / 'strict.npz'
    np.savez(source, temperatures=grid.temperatures, loggs=grid.loggs,
             fehs=grid.fehs, wavelengths_m=grid.wavelengths_m,
             spectra=grid.spectra, valid=grid.valid)
    repaired = generate_stellar_grid(tmp_path / 'repaired.npz', source=source,
                                     logg=4.5, feh=0.,
                                     missing_model_policy='interpolate_isolated_temperature')
    loaded = load_stellar_grid(repaired)
    np.testing.assert_allclose(loaded.interpolate(2500.), [3., 3.])
    assert loaded.valid.all()
    assert loaded.metadata['missing_model_policy'] == 'interpolate_isolated_temperature'
    assert loaded.metadata['generated_repaired_models'] == 1
    strict = generate_stellar_grid(tmp_path / 'still_strict.npz', source=source,
                                   missing_model_policy='reject')
    with pytest.raises(AtmosphereError, match='missing'):
        load_stellar_grid(strict).interpolate(2500.)
    with pytest.raises(ValueError, match='missing_model_policy'):
        generate_stellar_grid(tmp_path / 'bad.npz', source=source, missing_model_policy='invent')


def test_native_reader_repairs_isolated_missing_temperature_row(tmp_path):
    from tests._support import make_native_h5
    from platon._stellar_grid import load_native_stellar_grid
    h5py = pytest.importorskip('h5py')
    path = tmp_path / 'native.h5'
    make_native_h5(path)
    with h5py.File(path, 'r+') as f:
        del f['vgrid/axes[1]/x']
        f['vgrid/axes[1]/x'] = [3000., 4000., 5000.]
        f['vgrid/v_lin_seq'][...] = [1, 3]
    grid = load_native_stellar_grid(path)
    np.testing.assert_array_equal(grid.valid[:, 0, 0], [True, False, True])
    np.testing.assert_allclose(grid.interpolate(4000.),
                               .5 * (grid.interpolate(3000.) + grid.interpolate(5000.)), rtol=2e-7)


@pytest.mark.real_stellar_grid
def test_bundled_isolated_temperature_gap_uses_neighbor_row_average():
    grid = load_stellar_grid('newera')
    # This round gravity/metallicity pair has a real isolated source void.
    ti = np.flatnonzero(grid.temperatures == 2500.)[0]
    gi = np.flatnonzero(grid.loggs == 5.5)[0]
    zi = np.flatnonzero(grid.fehs == .5)[0]
    assert not grid.valid[ti, gi, zi]
    assert grid.valid[ti - 1, gi, zi] and grid.valid[ti + 1, gi, zi]
    expected = .5 * (grid.interpolate(2400., 5.5, .5) +
                     grid.interpolate(2600., 5.5, .5))
    np.testing.assert_allclose(grid.interpolate(2500., 5.5, .5), expected, rtol=2e-15)


@pytest.mark.parametrize('name, value', [
    (name, value) for name in ('Rs', 'Mp', 'Rp')
    for value in (0., -1., np.nan, np.inf)] + [
    ('error_excess', value) for value in (-1., np.nan, np.inf)])
def test_likelihood_rejects_invalid_physical_scalars_before_forward(name, value):
    from unittest.mock import Mock
    from platon.combined_retriever import CombinedRetriever
    fit = CombinedRetriever.get_default_fit_info(R_sun, M_jup, R_jup, T=1000.)
    fit.all_params[name].best_guess = value
    calculator = Mock()
    calculator.compute_depths.side_effect = AssertionError('Invalid trial reached forward model')
    result = CombinedRetriever()._ln_like([], calculator, None, fit,
                                          np.array([.01]), np.array([1e-4]), None, None)
    assert result == -np.inf
    calculator.compute_depths.assert_not_called()


def test_legacy_temperature_only_pickle_accepts_literal_stellar_parameters(tmp_path):
    import pickle
    source = tmp_path / 'legacy_star.pkl'
    source.write_bytes(pickle.dumps(dict(temperatures=[2400., 2600., 3100.],
                                         wavelengths_m=[1e-6, 2e-6],
                                         spectra=[[1., 2.], [3., 4.], [5., 6.]])))
    grid = load_stellar_grid(source)
    assert grid.metadata['ignored_parameter_axes'] == ['logg', 'feh']
    np.testing.assert_allclose(grid.interpolate(2550., 5.5, .5), [2.5, 3.5])
    with pytest.raises(AtmosphereError):
        grid.interpolate(2550., np.nan, .5)
    generated = generate_stellar_grid(tmp_path / 'anchored.npz', source=source,
                                      logg=5.5, feh=.5)
    anchored = load_stellar_grid(generated)
    assert 'ignored_parameter_axes' not in anchored.metadata
    np.testing.assert_allclose(anchored.interpolate(2550., 5.5, .5), [2.5, 3.5])
    with pytest.raises(AtmosphereError, match='outside'):
        anchored.interpolate(2550., 4.5, 0.)


def test_legacy_gravity_pickle_ignores_only_missing_metallicity(tmp_path):
    import pickle
    source = tmp_path / 'legacy_gravity.pkl'
    source.write_bytes(pickle.dumps(dict(temperatures=[2400., 2600.], loggs=[4., 5.],
                                         wavelengths_m=[1e-6, 2e-6],
                                         spectra=np.ones((2, 2, 2)))))
    grid = load_stellar_grid(source)
    assert grid.metadata['ignored_parameter_axes'] == ['feh']
    np.testing.assert_allclose(grid.interpolate(2500., 4.5, .5), [1., 1.])
    with pytest.raises(AtmosphereError, match='outside'):
        grid.interpolate(2500., 5.5, .5)


def test_canonical_singleton_axes_remain_strict():
    from platon._stellar_grid import grid_from_dict
    grid = grid_from_dict(dict(temperatures=[2400., 2600.], wavelengths_m=[1e-6, 2e-6],
                               spectra=np.ones((2, 1, 1, 2))))
    with pytest.raises(AtmosphereError, match='outside'):
        grid.interpolate(2550., 5.5, .5)


def test_retriever_prevalidation_skips_temperature_for_profiles_without_T(tiny_calculator):
    # Parametric and Guillot retrievals have no isothermal T; their computed
    # profiles are validated by each forward model instead.
    tiny_calculator._validate_params(None, None, None, np.inf)
    with pytest.raises(AtmosphereError):
        tiny_calculator._validate_params(np.nan, None, None, np.inf)
