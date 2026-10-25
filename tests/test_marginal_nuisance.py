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


def grid_ln_like(residuals, errors, x_lim=None, offset_prior=None, n=1201):
    """ln of the Gaussian likelihood of one group integrated over a uniform
    log10 inflation x in x_lim and an offset with offset_prior (("uniform",
    lo, hi) or ("gaussian", mean, std)), by brute force on a grid.  residuals
    are model - data, with the offset added to the model."""
    x = np.linspace(*x_lim, n) if x_lim is not None else np.zeros(1)
    if offset_prior is None:
        o, ln_prior_o = np.zeros(1), np.zeros(1)
    else:
        mean = -np.sum(residuals / errors**2) / np.sum(errors**-2)
        std = np.sum(errors**-2)**-0.5 * 10**np.max(x)
        if offset_prior[0] == "uniform":
            lo, hi = offset_prior[1:]
            o = np.linspace(max(lo, mean - 12 * std), min(hi, mean + 12 * std), n)
            ln_prior_o = np.full(n, -np.log(hi - lo))
        else:
            o = mean + std * np.linspace(-12, 12, n)
            ln_prior_o = scipy.stats.norm.logpdf(o, *offset_prior[1:])
    ln_l = np.empty(len(x))
    for i, b in enumerate(10**x):
        sigma = b * errors
        terms = np.sum(-0.5 * ((residuals[None, :] + o[:, None]) / sigma)**2
                       - np.log(np.sqrt(2 * np.pi) * sigma), axis=1) + ln_prior_o
        ln_l[i] = scipy.special.logsumexp(terms + trapezoid_ln_weights(o)) \
            if len(o) > 1 else terms[0]
    if len(x) > 1:
        return scipy.special.logsumexp(
            ln_l + trapezoid_ln_weights(x) - np.log(x_lim[1] - x_lim[0]))
    return ln_l[0]


def gamma_ln_like(residuals, errors, x_lim):
    """Exact ln likelihood of one group integrated over a uniform log10
    inflation x in x_lim, with no offset: in u = 10**(-2x) the integral is an
    incomplete gamma function."""
    n, chi2 = len(residuals), np.sum((residuals / errors)**2)
    a, c = n / 2, chi2 / 2
    z_low, z_high = c * 10**(-2 * x_lim[1]), c * 10**(-2 * x_lim[0])
    if scipy.special.gammainc(a, z_low) > 0.5:
        ln_diff = np.log(scipy.special.gammaincc(a, z_low) - scipy.special.gammaincc(a, z_high))
    else:
        ln_diff = np.log(scipy.special.gammainc(a, z_high) - scipy.special.gammainc(a, z_low))
    return -0.5 * n * np.log(2 * np.pi) - np.sum(np.log(errors)) \
        - np.log(2 * np.log(10)) + scipy.special.gammaln(a) - a * np.log(c) + ln_diff \
        - np.log(x_lim[1] - x_lim[0])


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

    def test_inflation_exact(self):
        # A well-placed inflation, and ones piled against the prior's ceiling
        # and floor, where the likelihood changes fastest
        fit_info = self.fit_info(transit_error_inflations={"infl": (0, self.n)})
        fit_info.add_uniform_fit_param("infl", -0.3, 0.7)
        reduced = CombinedRetriever._marginalize(fit_info)
        params_dict = reduced._interpret_param_array([])
        np.random.seed(3)
        for scale, x_draws in ((1.3, None), (30, 0.7), (0.2, -0.3)):
            residuals = scale * self.errors * self.noise
            ln_like, groups = CombinedRetriever._nuisance_ln_like(
                reduced, params_dict, {"transit": (residuals, self.errors)})
            self.assertAlmostEqual(
                ln_like, gamma_ln_like(residuals, self.errors, (-0.3, 0.7)), delta=5e-4)
            draws = CombinedRetriever._draw_nuisance(reduced, groups, 5000)["infl"]
            self.assertTrue(np.all((draws > -0.3) & (draws < 0.7)))
            if x_draws is not None:
                self.assertAlmostEqual(np.median(draws), x_draws, delta=0.01)

    def test_matches_brute_force(self):
        n, half = self.n, self.n // 2
        # instrument a: inflated 1.5x; b: offset 9e-4 near its prior edge,
        # errors overestimated 2x so its inflation sits on the prior floor
        residuals = self.errors * self.noise
        residuals[:half] *= 1.5
        residuals[half:] = residuals[half:] / 2 - 9.8e-4
        shift_residuals = self.shift_errors * self.shift_noise - 30
        fit_info = self.fit_info(
            transit_offsets={"off_b": (half, n)},
            transit_error_inflations={"infl_a": (0, half), "infl_b": (half, n)},
            time_shift_offsets={"t0": (0, n)},
            time_shift_errors=self.shift_errors)
        fit_info.add_uniform_fit_param("T", 800, 1800)
        fit_info.add_uniform_fit_param("off_b", -1e-3, 1e-3)
        fit_info.add_uniform_fit_param("infl_a", -0.3, 0.7)
        fit_info.add_uniform_fit_param("infl_b", -0.2, 0.7)
        fit_info.add_gaussian_fit_param("t0", 20)
        reduced = CombinedRetriever._marginalize(fit_info)
        self.assertEqual(reduced.fit_param_names, ["T"])
        self.assertEqual(sorted(reduced.marginal_params),
                         ["infl_a", "infl_b", "off_b", "t0"])
        blocks = {"transit": (residuals, self.errors),
                  "time_shift": (shift_residuals, self.shift_errors)}
        params_dict = reduced._interpret_param_array([1200])
        ln_like, groups = CombinedRetriever._nuisance_ln_like(
            reduced, params_dict, blocks)

        expected = grid_ln_like(residuals[:half], self.errors[:half], (-0.3, 0.7)) + \
            grid_ln_like(residuals[half:], self.errors[half:], (-0.2, 0.7),
                         ("uniform", -1e-3, 1e-3)) + \
            grid_ln_like(shift_residuals, self.shift_errors, None,
                         ("gaussian", 0, 20))
        self.assertAlmostEqual(ln_like, expected, delta=2e-3)

        # The conditional draws match the brute-force posterior moments
        np.random.seed(0)
        draws = CombinedRetriever._draw_nuisance(reduced, groups, 20000)
        x = np.linspace(-0.3, 0.7, 4001)
        sigma = 10**x[:, None] * self.errors[None, :half]
        ln_p = np.sum(-0.5 * (residuals[:half] / sigma)**2 - np.log(sigma), axis=1)
        p = np.exp(ln_p - ln_p.max())
        mean_x = np.sum(x * p) / np.sum(p)
        std_x = np.sqrt(np.sum((x - mean_x)**2 * p) / np.sum(p))
        self.assertAlmostEqual(np.mean(draws["infl_a"]), mean_x, delta=0.05 * std_x)
        self.assertAlmostEqual(np.std(draws["infl_a"]), std_x, delta=0.05 * std_x)
        self.assertLess(np.max(draws["off_b"]), 1e-3)
        self.assertLess(np.median(draws["infl_b"]), -0.15)
        precision = np.sum(self.shift_errors**-2) + 20**-2
        t0_mean = -np.sum(shift_residuals / self.shift_errors**2) / precision
        self.assertAlmostEqual(np.mean(draws["t0"]), t0_mean, delta=0.05)
        self.assertAlmostEqual(np.std(draws["t0"]), precision**-0.5, delta=0.02)

        mode = CombinedRetriever._draw_nuisance(reduced, groups, 1, mode=True)
        self.assertLess(mode["off_b"][0], 1e-3)
        self.assertGreater(mode["infl_b"][0], -0.2)

    def test_global_inflation_and_errors(self):
        residuals = 1.3 * self.errors * self.noise + 2e-4
        fit_info = self.fit_info(transit_offsets={"off": (10, self.n)})
        fit_info.add_uniform_fit_param("T", 800, 1800)
        fit_info.add_uniform_fit_param("off", -1e-3, 1e-3)
        fit_info.add_uniform_fit_param("log_error_inflation", -0.3, 0.7)
        reduced = CombinedRetriever._marginalize(fit_info)
        params_dict = reduced._interpret_param_array([1200])
        ln_like, _ = CombinedRetriever._nuisance_ln_like(
            reduced, params_dict, {"transit": (residuals, self.errors)})

        # One inflation couples both parts: brute force over (x, o) jointly
        x = np.linspace(-0.3, 0.7, 1501)
        o = np.linspace(-1e-3, 1e-3, 8001)
        b = 10**x[:, None]
        sigma = self.errors[None, :] * b
        part1 = np.sum(-0.5 * (residuals[:10] / sigma[:, :10])**2
                       - np.log(np.sqrt(2 * np.pi) * sigma[:, :10]), axis=1)
        W = np.sum(self.errors[10:]**-2)
        R = -np.sum(residuals[10:] / self.errors[10:]**2)
        S = np.sum((residuals[10:] / self.errors[10:])**2)
        chi2 = (S - 2 * o[None, :] * R + o[None, :]**2 * W) / b**2
        part2 = -0.5 * chi2 - np.sum(np.log(np.sqrt(2 * np.pi) * sigma[:, 10:]),
                                     axis=1)[:, None]
        part2 = scipy.special.logsumexp(part2 + trapezoid_ln_weights(o), axis=1) - np.log(2e-3)
        expected = scipy.special.logsumexp(part1 + part2 + trapezoid_ln_weights(x))
        self.assertAlmostEqual(ln_like, expected, delta=1e-3)

        # Global and per-range inflations cannot both be integrated out
        fit_info = self.fit_info(transit_error_inflations={"infl_a": (0, 10)})
        fit_info.add_uniform_fit_param("log_error_inflation", -0.3, 0.7)
        fit_info.add_uniform_fit_param("infl_a", -0.3, 0.7)
        with self.assertRaises(ValueError):
            CombinedRetriever._marginalize(fit_info)

        # An offset spanning two inflations cannot be integrated out
        fit_info = self.fit_info(
            transit_offsets={"off": (5, 30)},
            transit_error_inflations={"infl_a": (0, 10), "infl_b": (10, self.n)})
        for name in ("infl_a", "infl_b"):
            fit_info.add_uniform_fit_param(name, -0.3, 0.7)
        fit_info.add_uniform_fit_param("off", -1e-3, 1e-3)
        reduced = CombinedRetriever._marginalize(fit_info)
        with self.assertRaises(ValueError):
            CombinedRetriever._nuisance_ln_like(
                reduced, reduced._interpret_param_array([]),
                {"transit": (residuals, self.errors)})

        # Names may not clash between offsets and inflations
        with self.assertRaises(ValueError):
            self.fit_info(transit_offsets={"a": (0, 5)},
                          transit_error_inflations={"a": (5, 10)})

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
            transit_offsets={"off_b": (half, n)},
            transit_error_inflations={"infl_a": (0, half), "infl_b": (half, n)})
        fit_info.add_uniform_fit_param("T", 800, 1800)
        fit_info.add_uniform_fit_param("off_b", -1e-3, 1e-3)
        fit_info.add_uniform_fit_param("infl_a", -0.3, 0.7)
        fit_info.add_uniform_fit_param("infl_b", -0.3, 0.7)
        args = (calc, None)
        model = retriever._ln_like([1200, 0, 0, 0], *args, fit_info, depths,
                                   errors, None, None, ret_best_fit=True)[0].copy()
        noise = np.random.default_rng(4).normal(size=n)
        data = model + errors * noise * np.where(np.arange(n) < half, 1.3, 0.8)
        data[half:] -= 1.5e-4

        reduced = CombinedRetriever._marginalize(fit_info)
        ln_like = retriever._ln_like([1200], *args, reduced, data, errors, None, None)
        residuals = model - data
        expected = grid_ln_like(residuals[:half], errors[:half], (-0.3, 0.7)) + \
            grid_ln_like(residuals[half:], errors[half:], (-0.3, 0.7),
                         ("uniform", -1e-3, 1e-3))
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
        self.assertEqual(rows.shape, (3, 4))
        np.testing.assert_array_equal(rows[:, 0], [1200, 1300, 1200])
        # repeated rows get independent draws near the injected offset
        self.assertTrue(np.all(np.abs(rows[[0, 2], 1] + 1.5e-4) < 1e-4))
        self.assertNotEqual(rows[0, 1], rows[2, 1])
        best, ln_prob = retriever._best_with_nuisance(
            [1200], reduced, fit_info, like_args)
        self.assertTrue(np.isfinite(ln_prob))
        self.assertAlmostEqual(ln_prob, retriever._ln_like(
            best, *args, fit_info, data, errors, None, None))

        # A sampled inflation multiplies the errors
        retriever.params_to_lnlike = {}
        per_point = retriever._ln_like(
            [1200, 0, 0.1, 0.2], *args, fit_info, data, errors, None, None,
            lnlike_per_point=True)
        scaled = errors * np.where(np.arange(n) < half, 10**0.1, 10**0.2)
        np.testing.assert_allclose(per_point, -0.5 * (
            residuals**2 / scaled**2 + np.log(2 * np.pi * scaled**2)), rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
