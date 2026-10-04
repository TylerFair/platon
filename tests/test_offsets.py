import unittest

import numpy as np

from platon._offsets import apply_offsets as _apply_offsets, normalize_offset_map
from platon.fit_info import FitInfo


def apply_offsets(depths, params, spectrum):
    return _apply_offsets(depths, FitInfo(params)._interpret_param_array([]), spectrum)


class TestOffsets(unittest.TestCase):
    def test_ppm_supports_large_and_fractional_values_in_every_window_api(self):
        for value, depth_change in [(500, 0.0005), (50000, 0.05),
                                    (-50000, -0.05), (0.5, 0.0000005)]:
            for spectrum in ("transit", "eclipse"):
                configurations = [
                    {f"offset_{spectrum}": value, "offset_start": 1, "offset_end": 2},
                    {f"{spectrum}_offset_windows": {"offset_visit": (1, 2)},
                     "offset_visit": value},
                ]
                for params in configurations:
                    with self.subTest(value=value, spectrum=spectrum, params=params):
                        shifted = apply_offsets(np.full(3, 0.01), params, spectrum)
                        np.testing.assert_allclose(shifted, [0.01, 0.01 + depth_change, 0.01])

    def test_disjoint_shared_and_additive_offsets(self):
        original = np.array([0.01] * 6)
        params = {
            "transit_offset_windows": {
                "offset_shared": [(0, 2), (4, 6)],
                "offset_relative": (1, 5)},
            "offset_shared": 100,
            "offset_relative": -20,
        }
        shifted = apply_offsets(original, params, "transit")
        np.testing.assert_allclose(
            shifted - original, np.array([100, 80, -20, -20, 80, 100]) * 1e-6)
        np.testing.assert_array_equal(original, [0.01] * 6)
        np.testing.assert_allclose(apply_offsets(original, params, "transit"), shifted)

    def test_legacy_slice_and_explicit_override(self):
        params = {"offset_start": 1, "offset_end": 100000,
                  "offset_transit": 100, "offset_eclipse": -200}
        np.testing.assert_allclose(apply_offsets(np.zeros(3), params, "transit"), [0, 1e-4, 1e-4])
        np.testing.assert_allclose(apply_offsets(np.zeros(3), params, "eclipse"), [0, -2e-4, -2e-4])
        params["transit_offset_windows"] = {"offset_transit": (0, 1)}
        np.testing.assert_allclose(apply_offsets(np.zeros(3), params, "transit"), [1e-4, 0, 0])

    def test_adjacent_bounds(self):
        params = {"transit_offset_windows": {"shift": [(2, 3), (0, 2), (4, 5)]}, "shift": 20}
        np.testing.assert_allclose(apply_offsets(np.zeros(5), params, "transit"),
                                   [20e-6, 20e-6, 20e-6, 0, 20e-6])

    def test_reject_malformed_or_double_counted_windows(self):
        for bounds in [(-1, 2), (1, 1), (2, 1), (0.5, 2), (False, 2),
                       [], np.array(3), [np.array(3)], [(0, 3), (2, 4)], {"start": 0}, {"start": 0, "end": 1}, np.array([0, 1])]:
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                normalize_offset_map({"offset_visit": bounds})
        for name in ["", "offset_start", "offset_end", "transit_offset_windows"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                normalize_offset_map({name: (0, 1)})

    def test_reject_missing_values_out_of_range_and_nonfinite(self):
        params = {"transit_offset_windows": {"offset_visit": (0, 2)}}
        with self.assertRaisesRegex(ValueError, "Missing value"):
            _apply_offsets(np.zeros(2), {"transit_offset_windows": {"offset_visit": ((0, 2),)}}, "transit")
        for value in [np.nan, np.inf, "text", 1j, [1], True]:
            params["offset_visit"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite real scalar"):
                apply_offsets(np.zeros(2), params, "transit")
        params["offset_visit"] = 0
        with self.assertRaisesRegex(ValueError, "spectrum length"):
            apply_offsets(np.zeros(1), params, "transit")

    def test_fit_info_registers_zero_guesses_and_priors(self):
        info = FitInfo({"transit_offset_windows": {"offset_a": (0, 2)},
                        "offset_a": 50})
        info.add_offset("offset_b", [(2, 3), (4, 5)])
        info.add_offset("offset_day", (0, 2), spectrum="eclipse")
        info.add_uniform_fit_param("offset_a", -1000, 1000)
        info.add_gaussian_fit_param("offset_b", 200)
        params = info._interpret_param_array([100, -100])
        self.assertEqual(params["offset_a"], 100)
        self.assertEqual(params["offset_b"], -100)
        self.assertEqual(params["offset_day"], 0)
        self.assertEqual(params["transit_offset_windows"]["offset_b"], ((2, 3), (4, 5)))
        np.testing.assert_allclose(info._from_unit_interval_array([0.5, 0.5]), [0, 0])
        self.assertTrue(np.isfinite(info._ln_prior([100, -100])))
        self.assertEqual(info._ln_prior([2000, 0]), -np.inf)
        with self.assertRaisesRegex(ValueError, "Already"):
            info.add_offset("offset_a", (0, 2))

    def test_cross_spectrum_registration_preserves_value_and_prior(self):
        info = FitInfo({})
        info.add_offset("offset_shared", (0, 2), value=50)
        info.add_gaussian_fit_param("offset_shared", 10)
        prior = info.all_params["offset_shared"]
        info.add_offset("offset_shared", (2, 4), spectrum="eclipse")
        self.assertIs(info.all_params["offset_shared"], prior)
        self.assertEqual(info.fit_param_names, ["offset_shared"])
        # The Gaussian's mean and width, and transformed sampler values, are ppm.
        np.testing.assert_allclose(info._from_unit_interval_array([0.5]), [50])
        self.assertAlmostEqual(info._ln_prior([60]) - info._ln_prior([50]), -0.5)
        params = info._interpret_param_array([70])
        np.testing.assert_allclose(apply_offsets(np.zeros(4), params, "transit"), [70e-6, 70e-6, 0, 0])
        np.testing.assert_allclose(apply_offsets(np.zeros(4), params, "eclipse"), [0, 0, 70e-6, 70e-6])

    def test_reject_conflicting_shared_value_without_mutation(self):
        info = FitInfo({"offset_shared": 50})
        with self.assertRaisesRegex(ValueError, "existing value"):
            info.add_offset("offset_shared", (0, 2), value=0)
        self.assertEqual(info._get("offset_shared"), 50)
        self.assertNotIn("transit_offset_windows", info.all_params)

    def test_validate_offset_guesses_at_registration(self):
        for value in [np.nan, np.inf, True, "bad"]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite real scalar"):
                FitInfo({"transit_offset_windows": {"offset_a": (0, 1)}, "offset_a": value})
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            apply_offsets(np.zeros((2, 1)), {}, "transit")


class TestOffsetLikelihood(unittest.TestCase):
    def test_fixed_legacy_offsets_use_ppm_in_likelihood_and_best_fit(self):
        from unittest import mock
        from platon.combined_retriever import CombinedRetriever

        fit_info = CombinedRetriever.get_default_fit_info(
            7e8, 1.9e27, 7e7, T=1000, offset_transit=500, offset_eclipse=-500)
        transit, eclipse = mock.Mock(), mock.Mock()
        transit.compute_depths.return_value = (np.array([1e-6]), np.array([0.01]), {})
        eclipse.compute_depths.return_value = (np.array([5e-6]), np.array([0.001]), {})
        retriever = CombinedRetriever()
        retriever.params_to_lnlike = {}
        errors = np.array([1e-5])
        args = ([], transit, eclipse, fit_info,
                np.array([0.0105]), errors, np.array([0.0005]), errors)
        pointwise = retriever._ln_like(*args, lnlike_per_point=True)
        np.testing.assert_allclose(pointwise, -0.5 * np.log(2 * np.pi * errors[0]**2))
        best_transit, _, best_eclipse, _ = retriever._ln_like(*args, ret_best_fit=True)
        np.testing.assert_allclose(best_transit, [0.0105])
        np.testing.assert_allclose(best_eclipse, [0.0005])

    def test_pretty_print_keeps_offset_values_in_ppm(self):
        from platon.combined_retriever import CombinedRetriever

        info = FitInfo({"offset_visit": 0, "offset_start": 0})
        info.add_offset("offset_visit", (0, 1))
        info.add_uniform_fit_param("offset_visit", -50000, 50000)
        info.add_uniform_fit_param("offset_start", 0, 10)
        retriever = CombinedRetriever()
        retriever.last_lnprob = 0
        for value, expected in [(500, "500.00"), (50000, "5.00e+04"), (0.5, "0.50")]:
            retriever.last_params = [value, 2]
            line = retriever.pretty_print(info)
            self.assertIn(f"offset_visit={expected} ppm", line)
            self.assertIn("offset_start=2.00 \t", line)

    def test_transit_and_eclipse_likelihood_and_best_fit(self):
        from platon.combined_retriever import CombinedRetriever

        class Calculator:
            def __init__(self, depths):
                self.depths = np.array(depths)

            def compute_depths(self, *args, **kwargs):
                # Reuse this array so accidental in-place modification is seen.
                return np.arange(len(self.depths)), self.depths, {}

        fit_info = CombinedRetriever.get_default_fit_info(
            7e8, 1.9e27, 7e7, T=1000,
            transit_offset_windows={"offset_instrument": [(0, 1), (2, 3)]},
            eclipse_offset_windows={"offset_day": (1, 3)}, offset_day=-100)
        fit_info.add_uniform_fit_param("offset_instrument", -1000, 1000)
        fit_info.add_gaussian_fit_param("offset_day", 1000)
        transit = Calculator([0.01, 0.01, 0.01])
        eclipse = Calculator([0.001, 0.001, 0.001])
        transit_obs = np.array([0.0102, 0.01, 0.0102])
        eclipse_obs = np.array([0.001, 0.0009, 0.0009])
        errors = np.full(3, 1e-5)
        retriever = CombinedRetriever()
        retriever.params_to_lnlike = {}
        args = ([200, -100], transit, eclipse, fit_info,
                transit_obs, errors, eclipse_obs, errors)
        pointwise = retriever._ln_like(*args, lnlike_per_point=True)
        np.testing.assert_allclose(pointwise, -0.5 * np.log(2 * np.pi * errors[0]**2))
        best_transit, _, best_eclipse, _ = retriever._ln_like(*args, ret_best_fit=True)
        np.testing.assert_allclose(best_transit, transit_obs)
        np.testing.assert_allclose(best_eclipse, eclipse_obs)
        np.testing.assert_array_equal(transit.depths, [0.01] * 3)
        np.testing.assert_array_equal(eclipse.depths, [0.001] * 3)


if __name__ == "__main__":
    unittest.main()


def test_registered_offset_labels_ignore_name_syntax():
    from platon.combined_retriever import CombinedRetriever
    from platon._offsets import offset_labels
    info = FitInfo({'offset_unregistered': 7, 'offset_transit': 500})
    info.add_offset('calibration_start', (0, 1), value=500)
    info.add_uniform_fit_param('calibration_start', -1000, 1000)
    retriever = CombinedRetriever()
    retriever.last_lnprob = 0
    retriever.last_params = [500]
    assert 'calibration_start=500.00 ppm' in retriever.pretty_print(info)
    names = ['calibration_start', 'offset_unregistered', 'offset_transit']
    divisors, labels = retriever._get_divisors_labels([500, 7, 500], names, info)
    np.testing.assert_array_equal(divisors, [1, 1, 1])
    assert labels == ['calibration_start (ppm)', 'offset_unregistered', 'offset_transit (ppm)']
    assert offset_labels(names, info) == labels


def test_corner_plot_preserves_ppm_samples_and_registered_labels():
    from unittest.mock import patch
    from platon.plotter import Plotter
    from platon.retrieval_result import RetrievalResult
    fit = FitInfo({'offset_transit': 500, 'offset_unregistered': 7})
    fit.add_offset('calibration_end', (0, 1), value=500)
    for name in ('offset_transit', 'offset_unregistered', 'calibration_end'):
        fit.add_uniform_fit_param(name, -1000, 1000)
    samples = np.array([[500., 7., 500.], [600., 8., 600.]])
    result = RetrievalResult({'flatchain': samples}, 'emcee', samples[0], fit_info=fit)
    with patch('platon.plotter.corner.corner') as corner:
        Plotter().plot_retrieval_corner(result)
    np.testing.assert_array_equal(corner.call_args.args[0], samples)
    assert corner.call_args.kwargs['labels'] == [
        'offset_transit (ppm)', 'offset_unregistered', 'calibration_end (ppm)']


def test_offset_application_uses_windows_already_normalized_at_registration(monkeypatch):
    from platon import _offsets
    fit = FitInfo({})
    fit.add_offset('shift', [(2, 3), (0, 1)], value=500)
    params = fit._interpret_param_array([])
    def unexpected_normalization(*args):
        raise AssertionError('Repeated normalization')

    monkeypatch.setattr(_offsets, 'normalize_offset_map', unexpected_normalization)
    np.testing.assert_allclose(_apply_offsets(np.zeros(3), params, 'transit'), [5e-4, 0., 5e-4])
