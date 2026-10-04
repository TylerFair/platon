"""Offline regressions for caches, diagnostics, profiles and plotting helpers."""
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
from scipy.special import expn, logsumexp

from platon._mie_cache import MieCache
from platon._atmosphere_solver import AtmosphereSolver
from platon.TP_profile import Profile
from platon.terminator import TerminatorSector, TwoSectorTerminator
from platon.psis import psislw, psisloo, sumlogs
from platon.constants import k_B


def test_mie_cache_uses_both_vertices_of_requested_material():
    cache = MieCache()
    cache.add(1.5, [1., 1.04], [2., 4.])
    cache.add(2., [1.02], [100.])
    np.testing.assert_allclose(cache.get_from_cache(1.5, [1., 1.02, 1.04]), [2., 3., 4.])
    assert np.isnan(cache.get_from_cache(1.5, [.99, 1.05])).all()
    sparse = MieCache()
    sparse.add(1.5, [1., 100.], [2., 4.])
    assert np.isnan(sparse.get_from_cache(1.5, [99.])[0])


def test_mie_cache_accepts_list_and_reuses_single_exact_entry():
    cache = MieCache()
    with patch('platon._mie_cache._mie_multi_x.get_Qext', return_value=np.array([2.])) as calculate:
        np.testing.assert_array_equal(cache.get_and_update(1.5, [1.]), [2.])
        np.testing.assert_array_equal(cache.get_and_update(1.5, [1.]), [2.])
    calculate.assert_called_once()


def test_mie_cache_retains_exact_size_limit():
    cache = MieCache()
    cache.add(1.5, np.arange(1., 6.), np.arange(5.), size_limit=3)
    assert len(cache.all_xs) == len(cache.all_ms) == len(cache.all_Qexts) == 3


def test_effective_mie_cache_limit_and_replacement():
    atm = AtmosphereSolver.__new__(AtmosphereSolver)
    atm._mie_eff_xsec_cache = {}
    for index in range(65):
        atm._cache_mie_eff_xsec(index, [float(index)])
    assert len(atm._mie_eff_xsec_cache) == 64
    assert 0 not in atm._mie_eff_xsec_cache
    atm._cache_mie_eff_xsec(64, [999.])
    assert len(atm._mie_eff_xsec_cache) == 64
    assert atm._mie_eff_xsec_cache[64][0] == 999.


def test_visualizer_layer_background_extends_to_canvas_edge():
    from platon.visualizer import Visualizer
    info = {'radii': np.array([3., 2., 1.]),
            'unbinned_wavelengths': np.array([1., 2., 3.]),
            'tau_los': np.full((3, 2), np.log(2.))}
    image, scale = Visualizer(30).draw(info, [[.5, 1.5], [1.5, 2.5], [2.5, 3.5]],
                                      method='layers', max_dist=4., blur_std=0.)
    assert scale == pytest.approx(.1)
    np.testing.assert_allclose(image[:20], .5)
    np.testing.assert_allclose(image[20:], 1.)


def test_opacity_profile_uses_energy_flux_and_matches_own_pressure_grid():
    pressure, temperature = np.array([1., 2.]), np.array([1000., 1000.])
    density = pressure / (k_B * temperature)
    # Choose optical-depth increment 1, with thermal cross section 2.
    dr = 1. / (2. * density.mean())
    info = dict(unbinned_wavelengths=np.array([1., 2., 5., 8.]) * 1e-6,
                stellar_spectrum=np.array([1., 3., 1., 1.]),
                planet_spectrum=np.ones(4),
                absorption_coeff_atm=density[:, None] * np.array([1., 9., 2., 2.]),
                radii=np.array([2. * dr, dr]), P_profile=pressure, T_profile=temperature)
    profile = Profile()
    profile.pressures = np.array([.5, 1., 1.5, 2., 4.])
    profile.set_from_opacity(1000., info, visible_cutoff=3e-6, T_int=100.)
    gamma = 7. / 2.  # Visible energy-weighted sigma=(1*1+9*3)/4=7.
    expected4 = .75 * 100.**4 * (2/3 + 1.) + .75 * 1000.**4 * (
        2/3 + 2/(3*gamma) * (1 + (gamma/2 - 1)*np.exp(-gamma)) +
        2*gamma/3 * .5 * expn(2, gamma))
    assert profile.temperatures.shape == profile.pressures.shape
    np.testing.assert_allclose(profile.temperatures, expected4**.25)


def test_rebuilt_terminator_retains_independent_pressure_grids():
    cold, hot = Profile(), Profile()
    cold.pressures = np.geomspace(1e-4, 1e7, 7)
    hot.pressures = np.geomspace(1e-3, 1e8, 9)
    cold.set_isothermal(900.)
    hot.set_isothermal(1100.)
    model = TwoSectorTerminator(TerminatorSector(cold), TerminatorSector(hot))
    rebuilt = model.from_params(model.retrieval_defaults(), 1e27, 7e7)
    for original, new in [(cold, rebuilt.cold.profile), (hot, rebuilt.hot.profile)]:
        np.testing.assert_array_equal(new.pressures, original.pressures)
        assert new.temperatures.shape == original.pressures.shape
        assert not np.shares_memory(new.pressures, original.pressures)


@pytest.mark.parametrize('kwargs', [dict(cloudtop_pressure=np.nan),
                                    dict(scattering_factor=np.nan),
                                    dict(scattering_factor=np.inf),
                                    dict(scattering_slope=np.nan)])
def test_nonfinite_terminator_inputs_rejected(kwargs):
    profile = Profile()
    profile.set_isothermal(1000.)
    with pytest.raises(ValueError):
        TerminatorSector(profile, **kwargs)


def test_species_table_ignores_blank_lines_and_indented_comments(tmp_path):
    from platon._species_data_reader import read_species_data
    table = tmp_path / 'species.dat'
    table.write_text('  # Header\n\nH2 2.0 0.8\n   \nHe 4.0 0.0\n')
    (tmp_path / 'absorb_coeffs_H2.npy').touch()
    files, masses, polarizabilities = read_species_data(tmp_path, table, 'xsec', ['H2'])
    assert list(files) == ['H2']
    assert masses == {'H2': 2., 'He': 4.}
    assert polarizabilities == {'H2': .8}


@pytest.mark.parametrize('values', [(np.nan, .5, 1000.), (0., np.nan, 1000.),
                                   (0., .5, np.nan), (0., .5, np.inf)])
def test_abundance_bounds_reject_nonfinite_inputs(values):
    from platon.abundance_getter import AbundanceGetter
    getter = AbundanceGetter.__new__(AbundanceGetter)
    getter.min_temperature = 300.
    getter.logZs = np.array([-1., 1.])
    getter.CO_ratios = np.array([.3, .9])
    assert not getter.is_in_bounds(*values)


def test_psis_integer_weights_are_normalized_without_mutation():
    weights = np.array([0, -1, -2, -3])
    original = weights.copy()
    actual, _ = psislw(weights, overwrite_lw=True)
    np.testing.assert_allclose(actual, weights - logsumexp(weights))
    np.testing.assert_array_equal(weights, original)
    likelihood = np.full((10, 3), -2.)
    loo, pointwise, _ = psisloo(likelihood)
    np.testing.assert_allclose(pointwise, [-2., -2., -2.])
    assert loo == pytest.approx(-6.)


def test_sumlogs_handles_zero_weights_infinite_weights_and_output_buffer():
    values = np.array([[-np.inf, 1000.], [-np.inf, 1001.]])
    output = np.empty(2)
    assert sumlogs(values, axis=0, out=output) is output
    np.testing.assert_allclose(output, [-np.inf, logsumexp([1000., 1001.])])
    assert sumlogs(np.array([0., np.inf])) == np.inf


@pytest.mark.parametrize('reff', [0., -1., np.nan, np.inf])
def test_psis_rejects_invalid_efficiency(reff):
    with pytest.raises(ValueError, match='Reff'):
        psislw(np.array([0., -1.]), Reff=reff)


@pytest.mark.parametrize('weights', [[-np.inf, -np.inf], [np.nan, 0.], [np.inf, 0.]])
def test_psis_rejects_unnormalizable_weights(weights):
    with pytest.raises(ValueError, match='log-weight'):
        psislw(np.array(weights))


@pytest.mark.parametrize('backend', ['dynesty', 'nautilus', 'pymultinest', 'emcee'])
def test_corner_plot_accepts_explicit_label_and_range_options(backend):
    from platon.plotter import Plotter
    from platon.retrieval_result import RetrievalResult
    samples = np.ones((10, 2))
    result = RetrievalResult(dict(samples=samples, equal_samples=samples, flatchain=samples,
                                   weights=np.full(10, .1)), backend, [1., 1.],
                              fit_info=SimpleNamespace(fit_param_names=['a', 'b'], all_params={}))
    with patch('platon.plotter.corner.corner') as corner:
        Plotter().plot_retrieval_corner(result, labels=['A', 'B'], range=[.8, .9], show_titles=False)
    assert corner.call_args.kwargs['labels'] == ['A', 'B']
    assert corner.call_args.kwargs['range'] == [.8, .9]
    assert corner.call_args.kwargs['show_titles'] is False


@pytest.mark.parametrize('count', [0, 1])
def test_reconstruction_with_fewer_than_two_samples_marks_loo_unavailable(count):
    from platon.combined_retriever import CombinedRetriever
    retriever = CombinedRetriever()
    retriever.params_to_lnlike = {(1.,): np.array([-1., -2.])}
    result = SimpleNamespace()
    info = dict(unbinned_depths=np.array([.01, .01]),
                unbinned_correction_factors=np.ones(2),
                full_TP_profile=np.array([[1., 2.], [1000., 1000.]]))
    with patch.object(retriever, '_ln_like', return_value=(None, info, None, None)) as forward:
        retriever._collect_random_samples(result, np.array([[1.]]), count,
                                          None, None, None, np.array([.01, .01]),
                                          np.array([1e-4, 1e-4]), None, None, ())
    assert forward.call_count == count
    assert np.isnan(result.loo_total)
    assert np.isnan(result.loos).all() and result.loos.shape == (2,)
    assert np.isinf(result.loo_ks).all()


def test_two_sector_tp_plot_keeps_each_sectors_pressure_coordinates():
    import matplotlib.pyplot as plt
    from platon.plotter import Plotter
    from platon.retrieval_result import RetrievalResult
    from platon.combined_retriever import CombinedRetriever
    cold, hot = Profile(), Profile()
    cold.pressures, hot.pressures = np.geomspace(1e-4, 1e7, 7), np.geomspace(1e-3, 1e8, 9)
    cold.set_isothermal(900.)
    hot.set_isothermal(1100.)
    model = TwoSectorTerminator(TerminatorSector(cold), TerminatorSector(hot))
    fit = CombinedRetriever.get_default_fit_info(7e8, 1e27, 7e7, transit_terminator=model)
    fit.add_uniform_fit_param('cold_fraction', 0., 1.)
    result = RetrievalResult(dict(flatchain=np.array([[.3], [.5], [.7]])),
                              'emcee', [.5], fit_info=fit)
    try:
        Plotter().plot_retrieval_TP_profiles(result, which='transit',
                                             num_samples=3, plot_samples=True)
        lines = {line.get_label(): line for line in plt.gca().lines}
        np.testing.assert_array_equal(lines['cold best fit'].get_ydata(), cold.pressures / 1e5)
        np.testing.assert_array_equal(lines['hot best fit'].get_ydata(), hot.pressures / 1e5)
    finally:
        plt.close('all')


@pytest.mark.parametrize('hot_pressures', [[1., 10., 100., 1000.], [2., 20., 200.]])
def test_stored_two_sector_profile_uses_union_of_pressure_coordinates(hot_pressures):
    from platon.combined_retriever import CombinedRetriever
    cold, hot = Profile(), Profile()
    cold.pressures, hot.pressures = np.array([1., 10., 100.]), np.array(hot_pressures)
    cold.set_isothermal(900.)
    hot.set_isothermal(1100.)
    cold.temperatures += 100 * np.log10(cold.pressures)
    hot.temperatures += 200 * np.log10(hot.pressures)
    model = TwoSectorTerminator(TerminatorSector(cold), TerminatorSector(hot))
    stored = CombinedRetriever._profile_to_array(model)
    np.testing.assert_array_equal(stored[0], np.union1d(cold.pressures, hot.pressures))
    for row, profile in [(1, cold), (2, hot)]:
        indices = np.searchsorted(stored[0], profile.pressures)
        np.testing.assert_allclose(stored[row, indices], profile.temperatures)
        expected = np.interp(np.log(stored[0]), np.log(profile.pressures), profile.temperatures)
        np.testing.assert_allclose(stored[row], expected)


def test_pointwise_likelihood_cache_is_bounded_and_evicted_posterior_is_recomputed():
    from unittest.mock import Mock
    from platon.combined_retriever import CombinedRetriever
    retriever = CombinedRetriever()
    retriever.params_to_lnlike = {}
    fit = CombinedRetriever.get_default_fit_info(7e8, 1e27, 7e7, T=1000.)
    fit.add_uniform_fit_param('T', 900., 1200.)
    observed, model, errors = np.array([.01, .02]), np.array([.011, .019]), np.array([.001, .002])
    info = dict(unbinned_depths=model, unbinned_correction_factors=np.ones(2))
    calculator = Mock()
    calculator.compute_depths.return_value = (np.array([1e-6, 2e-6]), model, info)
    _, likelihood = retriever._sampler_functions(
        fit, calculator, None, observed, errors, None, None, (), print_evaluations=False)
    expected = -.5 * ((model - observed)**2 / errors**2 + np.log(2 * np.pi * errors**2))
    limit = retriever._POINTWISE_CACHE_MAX_ENTRIES
    trials = np.linspace(1000., 1100., limit + 2)[:, None]
    for trial in trials:
        assert likelihood(trial) == pytest.approx(expected.sum())
    assert len(retriever.params_to_lnlike) == limit
    assert tuple(trials[0]) not in retriever.params_to_lnlike
    assert tuple(trials[1]) not in retriever.params_to_lnlike
    previous_keys = list(retriever.params_to_lnlike)
    likelihood(trials[-1])
    assert list(retriever.params_to_lnlike) == previous_keys

    result = SimpleNamespace()
    before = calculator.compute_depths.call_count
    retriever._collect_random_samples(
        result, trials[[0, -1]], 2, calculator, None, fit,
        observed, errors, None, None, ())
    # Two full model reconstructions plus one pointwise cache miss.
    assert calculator.compute_depths.call_count - before == 3
    np.testing.assert_allclose(result.pointwise_lnlikes, [expected, expected])
    np.testing.assert_allclose(result.loos, expected)
    assert len(retriever.params_to_lnlike) == limit
    assert tuple(trials[0]) in retriever.params_to_lnlike
