"""Stellar heterogeneities: legacy names and different values per visit."""
import warnings
from unittest import mock

import numpy as np
import pytest

from platon.combined_retriever import CombinedRetriever
from platon.constants import R_sun, R_jup, M_jup
from platon.errors import AtmosphereError
from platon.fit_info import FitInfo
from platon.TP_profile import Profile
from platon.terminator import TerminatorSector, TwoSectorTerminator
from tests._support import tiny_profile


BINS = np.array([[.9, 1.1], [1.1, 1.3], [1.5, 2.], [2., 2.5], [2.5, 3.], [3., 4.]]) * 1e-6
ARGS = (R_sun, M_jup, R_jup)
STAR = dict(T_star=4200., logg_star=4.3, feh_star=-.2)
VISIT_A = dict(T_het=3200., f_het=.1, T_het2=4800., f_het2=.05)
VISIT_B = dict(T_het=3600., f_het=.2, T_het2=4600., f_het2=0.)


def per_visit(split):
    """Arrays giving bins [:split] VISIT_A's heterogeneities and the rest VISIT_B's."""
    return {name: np.r_[np.full(split, VISIT_A[name]),
                        np.full(len(BINS) - split, VISIT_B[name])]
            for name in VISIT_A}


@pytest.mark.parametrize('cloud_fraction', [1., .4])
def test_per_bin_heterogeneities_match_separate_single_visit_calls(tiny_calculator, cloud_fraction):
    calc = tiny_calculator
    calc.change_wavelength_bins(BINS)
    args = (tiny_profile(),) + ARGS
    options = dict(STAR, cloudtop_pressure=1e4, cloud_fraction=cloud_fraction)
    _, a, _ = calc.compute_depths(*args, **options, **VISIT_A)
    _, b, _ = calc.compute_depths(*args, **options, **VISIT_B)
    _, mixed, _ = calc.compute_depths(*args, **options, **per_visit(2))
    np.testing.assert_allclose(mixed, np.r_[a[:2], b[2:]], rtol=2e-6)
    assert not np.allclose(a, b)
    _, mixed_full, info = calc.compute_depths(*args, **options, **per_visit(2),
                                              full_output=True)
    np.testing.assert_allclose(mixed_full, mixed, rtol=2e-6)
    # Unbinned diagnostics use, at each wavelength, the visit of its bin
    _, correction_a = calc.atm.get_stellar_spectrum(**STAR, **VISIT_A)
    _, correction_b = calc.atm.get_stellar_spectrum(**STAR, **VISIT_B)
    in_a = info['unbinned_wavelengths'] < BINS[1, 1]
    np.testing.assert_allclose(info['unbinned_correction_factors'],
                               np.where(in_a, correction_a, correction_b), rtol=3e-6)


def test_scalar_and_array_values_mix_and_constant_arrays_are_one_visit(tiny_calculator):
    calc = tiny_calculator
    calc.change_wavelength_bins(BINS)
    args = (tiny_profile(),) + ARGS
    _, scalar, _ = calc.compute_depths(*args, **STAR, **VISIT_A)
    constant = {name: np.full(len(BINS), value) for name, value in VISIT_A.items()}
    _, same, _ = calc.compute_depths(*args, **STAR, **constant)
    np.testing.assert_allclose(same, scalar, rtol=1e-7)
    # Only f_het varies; the other three stay scalars
    f_het = np.r_[np.full(3, .1), np.full(3, .3)]
    _, mixed, _ = calc.compute_depths(*args, **STAR, **dict(VISIT_A, f_het=f_het))
    _, high, _ = calc.compute_depths(*args, **STAR, **dict(VISIT_A, f_het=.3))
    np.testing.assert_allclose(mixed, np.r_[scalar[:3], high[3:]], rtol=2e-6)


def test_three_interleaved_visits(tiny_calculator):
    calc = tiny_calculator
    calc.change_wavelength_bins(BINS)
    args = (tiny_profile(),) + ARGS
    fractions = [.05, .2, .05, .4, .2, .4]
    _, mixed, _ = calc.compute_depths(*args, **STAR, T_het=3500., f_het=np.array(fractions))
    for value in set(fractions):
        _, single, _ = calc.compute_depths(*args, **STAR, T_het=3500., f_het=value)
        rows = np.array(fractions) == value
        np.testing.assert_allclose(mixed[rows], single[rows], rtol=2e-6)


def test_two_sector_terminator_forwards_per_bin_heterogeneities(tiny_calculator):
    calc = tiny_calculator
    calc.change_wavelength_bins(BINS)
    model = TwoSectorTerminator(TerminatorSector(Profile.isothermal(800.)),
                                TerminatorSector(Profile.isothermal(1400.)), .3)
    _, cold, _ = calc.compute_depths(model.cold.profile, *ARGS, **STAR, **per_visit(3))
    _, hot, _ = calc.compute_depths(model.hot.profile, *ARGS, **STAR, **per_visit(3))
    _, mixed, _ = calc.compute_depths(model, *ARGS, **STAR, **per_visit(3))
    np.testing.assert_allclose(mixed, .3 * cold + .7 * hot, rtol=2e-6)


def test_per_bin_values_need_bins_and_one_value_per_bin(tiny_calculator):
    calc = tiny_calculator
    args = (tiny_profile(),) + ARGS
    with pytest.raises(ValueError, match='change_wavelength_bins'):
        calc.compute_depths(*args, **STAR, **per_visit(2))
    calc.change_wavelength_bins(BINS)
    with pytest.raises(ValueError, match='one value per wavelength bin'):
        calc.compute_depths(*args, **STAR, T_het=3500., f_het=np.full(len(BINS) + 1, .1))
    with pytest.raises(AtmosphereError, match='T_star is required'):
        calc.compute_depths(*args, f_het=np.r_[np.full(3, .1), np.full(3, .2)])


def test_eclipse_depths_reject_per_bin_heterogeneities(tiny_calculator):
    from platon.eclipse_depth_calculator import EclipseDepthCalculator
    calc = EclipseDepthCalculator.__new__(EclipseDepthCalculator)
    calc.atm = tiny_calculator.atm
    calc.atm.change_wavelength_bins(BINS)
    with pytest.raises(ValueError, match='transit depths'):
        calc.compute_depths(tiny_profile(), *ARGS, 4200., **per_visit(2))


def test_legacy_spot_names_are_aliases(tiny_calculator):
    calc = tiny_calculator
    args = (tiny_profile(),) + ARGS
    _, new, _ = calc.compute_depths(*args, T_star=4200., T_het=3200., f_het=.1)
    _, old, _ = calc.compute_depths(*args, T_star=4200., T_spot=3200., spot_cov_frac=.1)
    np.testing.assert_array_equal(new, old)
    with pytest.raises(ValueError, match='not both'):
        calc.compute_depths(*args, T_star=4200., T_het=3200., T_spot=3200., f_het=.1)
    fit_info = CombinedRetriever.get_default_fit_info(
        R_sun, M_jup, R_jup, T=1000, T_star=4200., T_spot=3200., spot_cov_frac=.1)
    assert fit_info.all_params['T_het'].best_guess == 3200.
    assert fit_info.all_params['f_het'].best_guess == .1
    assert 'T_spot' not in fit_info.all_params
    with pytest.warns(DeprecationWarning, match='renamed f_het'):
        fit_info.add_uniform_fit_param('spot_cov_frac', 0, .5)
    assert fit_info.fit_param_names == ['f_het']


def test_unknown_fit_parameters_suggest_close_names():
    fit_info = FitInfo({'T_het': 3000., 'f_het': .1, 'offset_NRS1': 0.})
    with pytest.raises(KeyError, match='did you mean offset_NRS1'):
        fit_info.add_uniform_fit_param('offset_nrs1', -1e-4, 1e-4)
    with pytest.raises(KeyError, match='Unknown parameter banana'):
        fit_info.add_gaussian_fit_param('banana', 1.)
    fit_info.all_params['visit1.f_het'] = fit_info.all_params['f_het'].__class__(None)
    with pytest.raises(ValueError, match='best_guess first'):
        fit_info.add_gaussian_fit_param('visit1.f_het', .05)


def visit_fit_info(**kwargs):
    return CombinedRetriever.get_default_fit_info(
        R_sun, M_jup, R_jup, T=1000, T_star=4200., T_het=3500., f_het=.1,
        transit_visits={'soss': (0, 2), 'g395h': (2, 6)}, **kwargs)


def test_transit_visits_create_inheriting_parameters():
    fit_info = visit_fit_info()
    for visit in ('soss', 'g395h'):
        for name in ('T_het', 'f_het', 'T_het2', 'f_het2'):
            assert fit_info.all_params[f'{visit}.{name}'].best_guess is None
    params = fit_info._interpret_param_array([])
    # Nothing overridden: plain scalars, so the calculator keeps one visit
    assert CombinedRetriever._visit_het_kwargs(params, 6) == dict(
        T_het=3500., f_het=.1, T_het2=None, f_het2=None)
    fit_info.add_uniform_fit_param('g395h.f_het', 0, .5)
    params = fit_info._interpret_param_array([.3])
    het = CombinedRetriever._visit_het_kwargs(params, 6)
    np.testing.assert_array_equal(het['f_het'], [.1, .1, .3, .3, .3, .3])
    np.testing.assert_array_equal(het['T_het'], np.full(6, 3500.))
    # An unset second heterogeneity has no contrast and no area
    np.testing.assert_array_equal(het['T_het2'], np.full(6, 4200.))
    np.testing.assert_array_equal(het['f_het2'], np.zeros(6))


def test_transit_visits_reach_the_calculator_but_not_eclipses():
    fit_info = visit_fit_info()
    fit_info.add_uniform_fit_param('soss.T_het', 2500., 4000.)
    fit_info.add_uniform_fit_param('g395h.f_het', 0, .5)
    transit, eclipse = mock.Mock(), mock.Mock()
    transit.compute_depths.return_value = (None, np.full(6, .01), None)
    eclipse.compute_depths.return_value = (None, np.full(2, 1e-3), None)
    retriever = CombinedRetriever()
    retriever._ln_like([3000., .25], transit, eclipse, fit_info,
                       np.full(6, .01), np.full(6, 1e-4),
                       np.full(2, 1e-3), np.full(2, 1e-4))
    kwargs = transit.compute_depths.call_args.kwargs
    np.testing.assert_array_equal(kwargs['T_het'], [3000., 3000., 3500., 3500., 3500., 3500.])
    np.testing.assert_array_equal(kwargs['f_het'], [.1, .1, .25, .25, .25, .25])
    eclipse_kwargs = eclipse.compute_depths.call_args.kwargs
    assert eclipse_kwargs['T_het'] == 3500. and eclipse_kwargs['f_het'] == .1


@pytest.mark.parametrize('visits, match', [
    ({'a': (0, 3), 'b': (2, 5)}, 'overlap'),
    ({'a': (3, 3)}, 'start < end'),
    ({'a.b': (0, 3)}, 'without dots'),
    ({' a': (0, 3)}, 'without dots'),
])
def test_invalid_transit_visits(visits, match):
    with pytest.raises(ValueError, match=match):
        CombinedRetriever.get_default_fit_info(
            R_sun, M_jup, R_jup, T=1000, T_star=4200., transit_visits=visits)


def test_transit_visits_require_a_photosphere():
    with pytest.raises(ValueError, match='T_star must be set'):
        CombinedRetriever.get_default_fit_info(
            R_sun, M_jup, R_jup, T=1000, transit_visits={'a': (0, 3)})


def test_too_many_distinct_per_bin_values_are_refused(tiny_calculator):
    from platon._forward_prep import MAX_VISITS
    edges = np.geomspace(.9e-6, 4.5e-6, MAX_VISITS + 2)
    tiny_calculator.change_wavelength_bins(np.column_stack([edges[:-1], edges[1:]]))
    with pytest.raises(ValueError, match="at most {}".format(MAX_VISITS)):
        tiny_calculator.compute_depths(tiny_profile(), *ARGS, **STAR, T_het=3500.,
                                       f_het=np.linspace(0, .2, MAX_VISITS + 1))


def test_per_visit_fit_survives_validation_and_walker_setup(tiny_calculator):
    tiny_calculator.change_wavelength_bins(BINS)
    fit_info = visit_fit_info()
    fit_info.add_uniform_fit_param('soss.f_het', 0, .3)
    assert fit_info.all_params['soss.f_het'].best_guess == pytest.approx(.15)
    CombinedRetriever()._validate_params(fit_info, tiny_calculator)
    walkers = fit_info._generate_rand_param_arrays(4)
    assert np.all(np.isfinite(walkers)) and walkers[0, 0] == pytest.approx(.15)
