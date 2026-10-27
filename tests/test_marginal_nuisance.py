import unittest

import numpy as np
import scipy.special
import scipy.stats

from platon.combined_retriever import CombinedRetriever
from platon.transit_depth_calculator import TransitDepthCalculator
from platon.constants import R_sun, M_jup, R_jup


def trapezoid_ln_weights(x):
    weights = np.full(len(x), x[1] - x[0])
    weights[[0, -1]] /= 2
    return np.log(weights)


def grid_ln_like(residuals, errors, s_lim=None, offset_prior=None, n=1201):
    """ln of the Gaussian likelihood of one group integrated over a uniform
    error excess s in s_lim (added in quadrature) and an offset with
    offset_prior (("uniform", lo, hi) or ("gaussian", mean, std)), by brute
    force on a grid.  residuals are model - data, with the offset added to the
    model."""
    s = np.linspace(*s_lim, n) if s_lim is not None else np.zeros(1)
    if offset_prior is None:
        o, ln_prior_o = np.zeros(1), np.zeros(1)
    else:
        mean = -np.sum(residuals / errors**2) / np.sum(errors**-2)
        std = np.sum(1 / (errors**2 + np.max(np.abs(s))**2))**-0.5
        if offset_prior[0] == "uniform":
            lo, hi = offset_prior[1:]
            o = np.linspace(max(lo, mean - 12 * std), min(hi, mean + 12 * std), n)
            ln_prior_o = np.full(n, -np.log(hi - lo))
        else:
            o = mean + std * np.linspace(-12, 12, n)
            ln_prior_o = scipy.stats.norm.logpdf(o, *offset_prior[1:])
    ln_l = np.empty(len(s))
    for i, excess in enumerate(s):
        sigma = np.sqrt(errors**2 + excess**2)
        terms = np.sum(-0.5 * ((residuals[None, :] + o[:, None]) / sigma)**2
                       - np.log(np.sqrt(2 * np.pi) * sigma), axis=1) + ln_prior_o
        ln_l[i] = scipy.special.logsumexp(terms + trapezoid_ln_weights(o)) \
            if len(o) > 1 else terms[0]
    if len(s) > 1:
        return scipy.special.logsumexp(
            ln_l + trapezoid_ln_weights(s) - np.log(s_lim[1] - s_lim[0]))
    return ln_l[0]


class TestMarginalNuisance(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.n = 40
        self.errors = rng.uniform(80e-6, 150e-6, self.n)
        self.noise = rng.normal(size=self.n)
        self.shift_errors = rng.uniform(5, 15, self.n)
        self.shift_noise = rng.normal(size=self.n)

    def fit_info(self, **kwargs):
        return CombinedRetriever.get_default_fit_info(
            Rs=R_sun, Mp=M_jup, Rp=R_jup, T=1200, **kwargs)

    def test_excess_exact(self):
        # A well-placed excess, one piled against the prior's floor (errors
        # overestimated) and one against its ceiling (much more scatter)
        fit_info = self.fit_info()
        fit_info.add_uniform_fit_param("error_excess", 0, 3e-4)
        reduced = CombinedRetriever._marginalize(fit_info)
        self.assertEqual(reduced.marginal_params, ["error_excess"])
        params_dict = reduced._interpret_param_array([])
        np.random.seed(3)
        for extra, scale in ((1.5e-4, 1), (0, 0.5), (2e-3, 1)):
            residuals = scale * self.errors * self.noise + \
                extra * np.random.default_rng(7).normal(size=self.n)
            ln_like, groups = CombinedRetriever._nuisance_ln_like(
                reduced, params_dict, {"transit": (residuals, self.errors)})
            expected = grid_ln_like(residuals, self.errors, (0, 3e-4), n=20001)
            self.assertAlmostEqual(ln_like, expected, delta=5e-4)
            draws = CombinedRetriever._draw_nuisance(reduced, groups, 5000)["error_excess"]
            self.assertTrue(np.all((draws > 0) & (draws < 3e-4)))
            # draws follow the brute-force conditional of s
            s = np.linspace(0, 3e-4, 20001)
            sigma = np.sqrt(self.errors[None, :]**2 + s[:, None]**2)
            ln_p = np.sum(-0.5 * (residuals / sigma)**2 - np.log(sigma), axis=1)
            p = np.exp(ln_p - ln_p.max())
            cdf = np.cumsum(p) / p.sum()
            self.assertAlmostEqual(np.median(draws), s[np.searchsorted(cdf, 0.5)],
                                   delta=0.03 * 3e-4)

    def test_matches_brute_force(self):
        n, half = self.n, self.n // 2
        # depths: extra scatter of 1e-4 everywhere, and an offset of -9.8e-4
        # near its prior edge for the second half; time shifts: a shift of 30 s
        # with a Gaussian prior on its offset
        residuals = np.sqrt(self.errors**2 + 1e-4**2) * self.noise
        residuals[half:] -= 9.8e-4
        shift_residuals = self.shift_errors * self.shift_noise - 30
        fit_info = self.fit_info(
            transit_offsets={"off_b": (half, n)},
            time_shift_offsets={"t0": (0, n)},
            time_shift_errors=self.shift_errors)
        fit_info.add_uniform_fit_param("T", 800, 1800)
        fit_info.add_uniform_fit_param("off_b", -1e-3, 1e-3)
        fit_info.add_uniform_fit_param("error_excess", 0, 4e-4)
        fit_info.add_gaussian_fit_param("t0", 20)
        reduced = CombinedRetriever._marginalize(fit_info)
        self.assertEqual(reduced.fit_param_names, ["T"])
        self.assertEqual(sorted(reduced.marginal_params), ["error_excess", "off_b", "t0"])
        blocks = {"transit": (residuals, self.errors),
                  "time_shift": (shift_residuals, self.shift_errors)}
        params_dict = reduced._interpret_param_array([1200])
        ln_like, groups = CombinedRetriever._nuisance_ln_like(reduced, params_dict, blocks)

        # The excess couples both halves of the depths: brute force over (s, o)
        s = np.linspace(0, 4e-4, 2001)
        o = np.linspace(-1e-3, 1e-3, 8001)
        sigma = np.sqrt(self.errors[None, :]**2 + s[:, None]**2)
        part_a = np.sum(-0.5 * (residuals[:half] / sigma[:, :half])**2
                        - np.log(np.sqrt(2 * np.pi) * sigma[:, :half]), axis=1)
        part_b = np.array([scipy.special.logsumexp(np.sum(
            -0.5 * ((residuals[None, half:] + o[:, None]) / row[None, half:])**2
            - np.log(np.sqrt(2 * np.pi) * row[None, half:]), axis=1)
            + trapezoid_ln_weights(o)) for row in sigma]) - np.log(2e-3)
        depths = scipy.special.logsumexp(part_a + part_b + trapezoid_ln_weights(s)) - np.log(4e-4)
        shifts = grid_ln_like(shift_residuals, self.shift_errors, None, ("gaussian", 0, 20))
        self.assertAlmostEqual(ln_like, depths + shifts, delta=2e-3)

        # The conditional draws match the brute-force posterior moments
        np.random.seed(0)
        draws = CombinedRetriever._draw_nuisance(reduced, groups, 20000)
        ln_post = part_a + part_b
        p = np.exp(ln_post - ln_post.max())
        mean_s = np.sum(s * p) / np.sum(p)
        std_s = np.sqrt(np.sum((s - mean_s)**2 * p) / np.sum(p))
        self.assertAlmostEqual(np.mean(draws["error_excess"]), mean_s, delta=0.05 * std_s)
        self.assertAlmostEqual(np.std(draws["error_excess"]), std_s, delta=0.05 * std_s)
        self.assertLess(np.max(draws["off_b"]), 1e-3)
        self.assertAlmostEqual(np.median(draws["off_b"]), 9.8e-4, delta=8e-5)
        precision = np.sum(self.shift_errors**-2) + 20**-2
        t0_mean = -np.sum(shift_residuals / self.shift_errors**2) / precision
        self.assertAlmostEqual(np.mean(draws["t0"]), t0_mean, delta=0.05)
        self.assertAlmostEqual(np.std(draws["t0"]), precision**-0.5, delta=0.02)

        mode = CombinedRetriever._draw_nuisance(reduced, groups, 1, mode=True)
        self.assertLess(mode["off_b"][0], 1e-3)
        self.assertGreater(mode["error_excess"][0], 0)

    def test_what_is_integrated(self):
        # Only offsets and a uniform-prior error_excess are integrated out
        fit_info = self.fit_info(transit_offsets={"off": (10, self.n)})
        fit_info.add_uniform_fit_param("T", 800, 1800)
        fit_info.add_uniform_fit_param("off", -1e-3, 1e-3)
        fit_info.add_gaussian_fit_param("error_excess", 1e-4)
        reduced = CombinedRetriever._marginalize(fit_info)
        self.assertEqual(reduced.fit_param_names, ["T", "error_excess"])
        self.assertEqual(reduced.marginal_params, ["off"])
        # Names may not clash between offsets and other parameters
        with self.assertRaises(ValueError):
            self.fit_info(transit_offsets={"Rp": (0, 5)})
        # Errors are never inflated by a factor
        for name in ("log_error_inflation", "transit_error_inflations"):
            self.assertNotIn(name, self.fit_info().all_params)

    def test_ln_like_end_to_end(self):
        min_wavelength, max_wavelength, depths, errors = np.loadtxt(
            "tests/testing_data/hd209458b_transit_depths", unpack=True)
        bins = np.array([min_wavelength, max_wavelength]).T
        n = len(depths)
        half = n // 2
        calc = TransitDepthCalculator()
        calc.change_wavelength_bins(bins)
        retriever = CombinedRetriever()

        fit_info = CombinedRetriever.get_default_fit_info(
            Rs=1.19 * R_sun, Mp=0.73 * M_jup, Rp=1.4 * R_jup, T=1200,
            transit_offsets={"off_b": (half, n)})
        fit_info.add_uniform_fit_param("T", 800, 1800)
        fit_info.add_uniform_fit_param("error_excess", 0, 1e-4)
        fit_info.add_uniform_fit_param("off_b", -1e-3, 1e-3)
        args = (calc, None)
        model = retriever._ln_like([1200, 1e-9, 0], *args, fit_info, depths,
                                   errors, None, None, ret_best_fit=True)[0].copy()
        noise = np.random.default_rng(4).normal(size=n)
        data = model + np.sqrt(errors**2 + 4e-5**2) * noise
        data[half:] -= 1.5e-4

        reduced = CombinedRetriever._marginalize(fit_info)
        ln_like = retriever._ln_like([1200], *args, reduced, data, errors, None, None)
        residuals = model - data
        s = np.linspace(0, 1e-4, 801)
        o = np.linspace(-1e-3, 1e-3, 4001)
        sigma = np.sqrt(errors[None, :]**2 + s[:, None]**2)
        part_a = np.sum(-0.5 * (residuals[:half] / sigma[:, :half])**2
                        - np.log(np.sqrt(2 * np.pi) * sigma[:, :half]), axis=1)
        part_b = np.array([scipy.special.logsumexp(np.sum(
            -0.5 * ((residuals[None, half:] + o[:, None]) / row[None, half:])**2
            - np.log(np.sqrt(2 * np.pi) * row[None, half:]), axis=1)
            + trapezoid_ln_weights(o)) for row in sigma]) - np.log(2e-3)
        expected = scipy.special.logsumexp(part_a + part_b + trapezoid_ln_weights(s)) - np.log(1e-4)
        self.assertAlmostEqual(ln_like, expected, delta=2e-3)
        # Per-point output is not defined once nuisances are integrated out
        self.assertEqual(retriever._ln_like(
            [1200], *args, reduced, data, errors, None, None,
            lnlike_per_point=True).shape, (1,))

        # Full rows: sampled parameters kept, integrated ones drawn
        like_args = (calc, None, data, errors, None, None, [])
        np.random.seed(2)
        rows = retriever._with_nuisance(
            [[1200], [1300], [1200]], reduced, fit_info, like_args)
        self.assertEqual(rows.shape, (3, 3))
        np.testing.assert_array_equal(rows[:, 0], [1200, 1300, 1200])
        # repeated rows get independent draws near the injected offset
        off = fit_info.fit_param_names.index("off_b")
        self.assertTrue(np.all(np.abs(rows[[0, 2], off] + 1.5e-4) < 1e-4))
        self.assertNotEqual(rows[0, off], rows[2, off])
        excess = fit_info.fit_param_names.index("error_excess")
        self.assertTrue(np.all((rows[:, excess] > 0) & (rows[:, excess] < 1e-4)))
        best, ln_prob = retriever._best_with_nuisance(
            [1200], reduced, fit_info, like_args)
        self.assertTrue(np.isfinite(ln_prob))
        self.assertAlmostEqual(ln_prob, retriever._ln_like(
            best, *args, fit_info, data, errors, None, None))

        # A sampled error_excess adds in quadrature
        retriever.params_to_lnlike = {}
        per_point = retriever._ln_like(
            [1200, 3e-5, 0], *args, fit_info, data, errors, None, None,
            lnlike_per_point=True)
        scaled = np.sqrt(errors**2 + 3e-5**2)
        np.testing.assert_allclose(per_point, -0.5 * (
            residuals**2 / scaled**2 + np.log(2 * np.pi * scaled**2)), rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
