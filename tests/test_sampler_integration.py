"""Exercise the real retrieval drivers without opacities or native samplers."""

from contextlib import ExitStack
import sys
from types import SimpleNamespace
from unittest import mock

import dynesty.results
import numpy as np
import pytest

from platon.combined_retriever import CombinedRetriever


def _sampler_stubs(fit_info):
    """Evaluate three fixed trial points through each backend's callback API."""
    target = [0.6 if name == "offset_nirspec" else 0.3
              for name in fit_info.fit_param_names]
    alternative = [0.7 if name == "offset_nirspec" else 0.4
                   for name in fit_info.fit_param_names]
    cubes = np.array([[0.5] * len(target), target, alternative])
    state = SimpleNamespace()

    def evaluate(prior, log_likelihood):
        state.samples = np.array([prior(cube) for cube in cubes])
        state.logl = np.array([log_likelihood(point) for point in state.samples])

    class Emcee:
        def __init__(self, nwalkers, ndim, log_probability, args):
            evaluate(fit_info._from_unit_interval_array,
                     lambda point: log_probability(point, *args))
            self.flatchain = state.samples
            self.flatlnprobability = state.logl
            self.chain = state.samples[:, None, :]
            self.lnprobability = state.logl[:, None]
            self.acceptance_fraction = np.full(3, 0.5)

        def sample(self, initial_positions, iterations):
            yield self.chain

    class Dynesty:
        def __init__(self, likelihood, prior, ndim, **kwargs):
            evaluate(prior, likelihood)
            self.results = dynesty.results.Results(dict(
                samples=state.samples, samples_u=cubes,
                samples_id=np.arange(3), logl=state.logl, nlive=3,
                logwt=np.log([0.25, 0.5, 0.25]), logz=np.array([-12.5])))

        def run_nested(self, **kwargs):
            pass

    class Nautilus:
        def __init__(self, prior, likelihood, **kwargs):
            evaluate(prior, likelihood)
            self.log_z, self.n_eff = -12.5, 3

        def run(self, **kwargs):
            return True

        def posterior(self):
            return state.samples, np.log([0.25, 0.5, 0.25]), state.logl

    def solve(LogLikelihood, Prior, n_dims, **kwargs):
        evaluate(Prior, LogLikelihood)
        return {"logz": np.array([-12.5])}

    class Analyzer:
        def __init__(self, **kwargs):
            pass

        def get_data(self):
            return np.column_stack(([0.25, 0.5, 0.25], -2 * state.logl, state.samples))

        def get_equal_weighted_posterior(self):
            return np.column_stack((state.samples, state.logl))

    return state, Emcee, Dynesty, Nautilus, SimpleNamespace(solve=solve, Analyzer=Analyzer)


@pytest.mark.parametrize("backend", ["emcee", "dynesty", "multinest", "nautilus"])
@pytest.mark.parametrize("spectra", ["transit", "eclipse", "joint"])
def test_sampler_drivers_preserve_tls_offsets_and_opacity_settings(backend, spectra):
    transit_bins = np.array([[1e-6, 1.1e-6], [1e-6, 1.1e-6], [2e-6, 2.1e-6]])
    eclipse_bins = np.array([[4e-6, 4.1e-6], [5e-6, 5.1e-6]])
    transit_depths = np.array([0.01, 0.0101, 0.0101])
    eclipse_depths = np.array([0.001, 0.0008])
    fit_transit = spectra in {"transit", "joint"}
    fit_eclipse = spectra in {"eclipse", "joint"}
    tls_options = dict(T_het2=6500, f_het2=0.05, logg_star=4.2,
                       logg_het=4.0, logg_het2=4.6, feh_star=-0.3,
                       stellar_grid_only=True, stellar_blackbody=False,
                       validate_T_grid=False)
    fit_info = CombinedRetriever.get_default_fit_info(
        7e8, 1.9e27, 7e7, T=1000, T_star=5500, T_het=4500,
        f_het=0.02, stellar_grid="legacy",
        transit_offsets={"offset_nirspec": (1, 3)} if fit_transit else None,
        eclipse_offsets={"offset_miri": (1, 2)} if fit_eclipse else None,
        **tls_options)
    if fit_transit:
        fit_info.add_uniform_fit_param("offset_nirspec", -5e-4, 5e-4)
    if fit_eclipse:
        fit_info.add_uniform_fit_param("offset_miri", -5e-4, 5e-4)
    state, emcee_stub, dynesty_stub, nautilus_stub, multinest_stub = _sampler_stubs(fit_info)
    calculators = []

    class Calculator:
        def __init__(self, spectrum, **options):
            self.spectrum, self.options = spectrum, options
            self.calls, self.validation_calls = [], []
            calculators.append(self)

        def change_wavelength_bins(self, bins):
            self.bins = bins

        def _validate_params(self, *args, **kwargs):
            self.validation_calls.append(kwargs)

        def compute_depths(self, *args, **kwargs):
            self.calls.append(kwargs)
            baseline = 0.01 if self.spectrum == "transit" else 0.001
            # Omitting opacity settings during posterior reconstruction must
            # produce a visibly different spectrum, rather than pass silently.
            if "H2O" not in kwargs["zero_opacities"]:
                baseline += 1e-4
            depths = np.full(len(self.bins), baseline)
            info = ({"unbinned_depths": depths.copy(),
                     "unbinned_correction_factors": np.ones(len(depths))}
                    if self.spectrum == "transit" else {"unbinned_eclipse_depths": depths.copy()})
            return self.bins.mean(axis=1), depths, info

    transit_args = (transit_bins, transit_depths, np.full(3, 1e-5))
    eclipse_args = (eclipse_bins, eclipse_depths, np.full(2, 1e-5))
    if spectra == "transit":
        eclipse_args = (None, None, None)
    elif spectra == "eclipse":
        transit_args = (None, None, None)
    retriever = CombinedRetriever()
    with ExitStack() as stack:
        stack.enter_context(mock.patch("platon.combined_retriever.TransitDepthCalculator",
                                       side_effect=lambda **opts: Calculator("transit", **opts)))
        stack.enter_context(mock.patch("platon.combined_retriever.EclipseDepthCalculator",
                                       side_effect=lambda **opts: Calculator("eclipse", **opts)))
        stack.enter_context(mock.patch("platon.combined_retriever.emcee.EnsembleSampler", emcee_stub))
        stack.enter_context(mock.patch("platon.combined_retriever.NestedSampler", dynesty_stub))
        stack.enter_context(mock.patch.dict(sys.modules, {
            "nautilus": SimpleNamespace(Sampler=nautilus_stub), "pymultinest": multinest_stub}))
        stack.enter_context(mock.patch("platon.combined_retriever.write_param_estimates_file"))
        loo = stack.enter_context(mock.patch("platon.combined_retriever.psisloo",
                                            return_value=(0, np.zeros(1), np.zeros(1))))
        # Preserve every evaluated trial in the posterior for independent
        # likelihood checks, without making the test depend on random draws.
        stack.enter_context(mock.patch("platon.combined_retriever.dynesty.utils.resample_equal",
                                       side_effect=lambda samples, weights: samples.copy()))
        stack.enter_context(mock.patch("platon.combined_retriever.np.random.shuffle"))
        result = getattr(retriever, "run_" + backend)(
            *transit_args, *eclipse_args, fit_info,
            zero_opacities=["H2O"], num_final_samples=3)

    np.testing.assert_allclose(result.best_fit_params, state.samples[1])
    expected_pointwise = []
    for sample in state.samples:
        values = dict(zip(fit_info.fit_param_names, sample))
        residuals = []
        if transit_args[0] is not None:
            model = np.array([0.01, 0.01, 0.01])
            model[1:] += values["offset_nirspec"]
            residuals.extend(model - transit_depths)
        if eclipse_args[0] is not None:
            model = np.array([0.001, 0.001])
            model[1] += values["offset_miri"]
            residuals.extend(model - eclipse_depths)
        expected_pointwise.append(-0.5 * (np.square(np.array(residuals) / 1e-5)
                                         + np.log(2 * np.pi * 1e-10)))
    np.testing.assert_allclose(result.pointwise_lnlikes, expected_pointwise)
    np.testing.assert_allclose(loo.call_args.args[0], expected_pointwise)
    if transit_args[0] is not None:
        np.testing.assert_allclose(result.best_fit_transit_depths, transit_depths)
        np.testing.assert_allclose(result.random_transit_depths, np.full((3, 3), 0.01))
    if eclipse_args[0] is not None:
        np.testing.assert_allclose(result.best_fit_eclipse_depths, eclipse_depths)
        np.testing.assert_allclose(result.random_eclipse_depths, np.full((3, 2), 0.001))
    for calculator in calculators:
        assert calculator.options["stellar_grid"] == "legacy"
        for call in calculator.calls:
            assert call["zero_opacities"] == ["H2O"]
            assert {name: call[name] for name in tls_options} == tls_options
        if calculator.spectrum == "transit":
            assert calculator.validation_calls
            assert all(call["validate_T_grid"] is False for call in calculator.validation_calls)


def test_posterior_reconstruction_recovers_pointwise_cache_misses():
    """Native samplers can return posterior points absent from the callback cache."""
    retriever = CombinedRetriever()
    retriever.params_to_lnlike = {}
    result = SimpleNamespace()
    pointwise = np.array([-2.0, -3.0])
    transit_info = {"unbinned_depths": np.array([0.01, 0.02]),
                    "unbinned_correction_factors": np.array([1.1, 1.2]),
                    "full_TP_profile": np.array([[1, 2], [1000, 1000]])}

    def likelihood(params, *args, **kwargs):
        assert kwargs["zero_opacities"] == ["H2O"]
        if kwargs.get("ret_best_fit"):
            return None, transit_info, None, None
        return pointwise

    with mock.patch.object(retriever, "_ln_like", side_effect=likelihood), \
            mock.patch("platon.combined_retriever.psisloo", return_value=(0, 0, 0)):
        retriever._collect_random_samples(
            result, [[0.1]], 1, None, None, None, None, None, None, None, ["H2O"])
    np.testing.assert_array_equal(result.pointwise_lnlikes, [pointwise])
    np.testing.assert_allclose(result.random_transit_depths, [[0.011, 0.024]])


def test_old_fit_info_uses_stellar_and_temperature_validation_defaults():
    fit_info = CombinedRetriever.get_default_fit_info(7e8, 1.9e27, 7e7, T=1000)
    fit_info.add_uniform_fit_param("T", 800, 1200)
    new_options = {"T_het2": None, "f_het2": None, "logg_star": 4.5,
                   "logg_het": None, "logg_het2": None, "feh_star": 0.0,
                   "stellar_grid_only": False, "stellar_blackbody": False,
                   "validate_T_grid": True}
    for name in ["stellar_grid", *new_options]:
        del fit_info.all_params[name]
    calculator = mock.Mock()
    calculator.compute_depths.return_value = (np.array([1e-6]), np.array([0.01]), {})
    retriever = CombinedRetriever()
    bins = np.array([[1e-6, 1.1e-6]])
    with mock.patch("platon.combined_retriever.TransitDepthCalculator", return_value=calculator) as factory:
        transit, eclipse = retriever._make_calculators(fit_info, bins, None, True, "xsec")
    assert transit is calculator and eclipse is None
    assert factory.call_args.kwargs["stellar_grid"] == "newera"
    assert all(call.kwargs["validate_T_grid"] is True
               for call in calculator._validate_params.call_args_list)
    retriever._ln_like([1000], calculator, None, fit_info,
                       np.array([0.01]), np.array([1e-5]), None, None)
    kwargs = calculator.compute_depths.call_args.kwargs
    assert {name: kwargs[name] for name in new_options} == new_options


def test_out_of_prior_trials_do_not_invoke_forward_models():
    fit_info = CombinedRetriever.get_default_fit_info(
        7e8, 1.9e27, 7e7, T=1000, transit_offsets={"offset_visit": (0, 1)})
    fit_info.add_uniform_fit_param("offset_visit", -5e-4, 5e-4)
    calculator = mock.Mock()
    calculator.compute_depths.side_effect = AssertionError("Invalid trial reached calculator")
    retriever = CombinedRetriever()
    arguments = (calculator, None, fit_info, np.array([0.01]), np.array([1e-5]), None, None)
    assert retriever._ln_prob([6e-4], *arguments) == -np.inf
    _, nested_likelihood = retriever._sampler_functions(
        fit_info, calculator, None, np.array([0.01]), np.array([1e-5]),
        None, None, (), print_evaluations=False)
    assert nested_likelihood([6e-4]) == -np.inf
    calculator.compute_depths.assert_not_called()
