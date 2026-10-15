import unittest
from unittest import mock

import numpy as np
from scipy.special import logsumexp, ndtr, ndtri

from platon.smc_sampler import GaussianModel, _StiffMap, _resampling_folds, run_smc


class TestSMC(unittest.TestCase):
    def test_crossfit_validation_and_families(self):
        flat = lambda u: np.zeros(len(u))
        for folds in (0, 17, 1.5):
            with self.assertRaisesRegex(ValueError, 'folds'):
                run_smc(flat, 2, n_particles=16, folds=folds)
        with self.assertRaisesRegex(ValueError, 'complement'):
            run_smc(flat, 6, n_particles=16)
        z = np.array([[0.], [0.], [1.], [2.]])
        parents = np.array([0, 1, 1, 2, 3, 3])
        groups = _resampling_folds(z, parents, 2, np.random.default_rng(3))
        self.assertTrue(any(set([0, 1, 2]).issubset(g) for g in groups))
        self.assertTrue(any(set([4, 5]).issubset(g) for g in groups))

    def test_crossfit_exact_start_stationarity(self):
        # Constant likelihood leaves the initial iid normal ensemble exact.
        # Use the validated 60/12 budget: shortening chains weakens the
        # fitted-on-self contraction signal. Ensemble SE is about .0025.
        means = {1: [], 2: []}
        for seed in range(24):
            for folds in means:
                result = run_smc(lambda u: np.zeros(len(u)), 27,
                                 n_particles=512, inner_steps=60, outer_steps=12,
                                 adaptive_steps=(folds == 1), folds=folds, seed=seed)
                means[folds].append(np.mean(ndtri(result['samples'])**2))
                self.assertEqual(result['history'][0]['n_parents'], 512)
                self.assertEqual(result['history'][0]['n_unique'], 512)
        self.assertAlmostEqual(np.mean(means[2]), 1, delta=.012)
        self.assertLess(np.mean(means[1]), np.mean(means[2]) - .006)

    def test_crossfit_freezes_objects_and_caches_predictions(self):
        from platon import smc_sampler as smc
        seen = []
        def predict(u):
            seen.extend(map(tuple, u))
            return ndtri(u), np.ones_like(u)
        full = GaussianModel(predict, [0., 0.])
        cheap = GaussianModel(lambda u: (1.03 * ndtri(u) + .02, np.ones_like(u)), [0., 0.])
        with mock.patch.object(smc, '_Reference', wraps=smc._Reference) as ref, \
             mock.patch.object(smc, '_Correction', wraps=smc._Correction) as correction:
            result = run_smc(full, 2, n_particles=128, cheap_loglike=cheap,
                             inner_steps=2, outer_steps=3, seed=14)
        self.assertEqual(ref.call_count, 2 * len(result['history']))
        self.assertEqual(correction.call_count, ref.call_count)
        self.assertEqual(len(seen), result['n_evals'])
        for h in result['history']:
            self.assertEqual(h['n_outer'], 3)
            self.assertEqual(h['n_inner'], 2)

    def test_gaussian_evidence(self):
        mu, sd = np.array([0.7, -0.4]), np.array([0.5, 0.8])

        def likelihood(u):
            return -0.5 * np.sum(((ndtri(u) - mu) / sd)**2, axis=1)

        result = run_smc(likelihood, 2, n_particles=2048, inner_steps=10,
                         outer_steps=3, seed=20)
        exact = -0.5 * np.sum(np.log1p(1 / sd**2) + mu**2 / (1 + sd**2))
        self.assertLess(abs(result['logz'] - exact), max(0.08, 4 * result['logz_err']))
        np.testing.assert_allclose(ndtri(result['samples']).mean(0),
                                   mu / (1 + sd**2), atol=0.05)
        self.assertAlmostEqual(result['weights'].sum(), 1)

    def test_bimodal_mass(self):
        def likelihood(u):
            z = ndtri(u)[:, 0]
            terms = np.array([np.log(0.3) - np.log(0.5) - 0.5 * ((z + 2) / 0.5)**2,
                              np.log(0.7) - np.log(0.5) - 0.5 * ((z - 2) / 0.5)**2])
            return logsumexp(terms, axis=0) + 0.5 * z**2

        result = run_smc(likelihood, 1, n_particles=4096, inner_steps=15,
                         outer_steps=4, seed=4)
        self.assertLess(abs(result['logz']), 0.10)
        self.assertAlmostEqual(np.mean(result['samples'][:, 0] < 0.5), 0.3, delta=0.05)

    def test_invalid_prior_region(self):
        def likelihood(u):
            return np.where(u[:, 0] < 0.2, 2.0, -np.inf)

        result = run_smc(likelihood, 1, n_particles=2048, inner_steps=5,
                         outer_steps=2, seed=7)
        self.assertTrue(np.all(result['samples'][:, 0] < 0.2))
        self.assertAlmostEqual(result['logz'], 2 + np.log(0.2), delta=0.08)
        self.assertAlmostEqual(result['logz'] - result['log_valid_fraction'], 2)
        with self.assertRaisesRegex(RuntimeError, 'valid particles'):
            run_smc(lambda u: np.full(len(u), -np.inf), 1, max_prior_draws=256)

    def test_delayed_acceptance_biased_cheap(self):
        def full(u):
            return -0.5 * ((ndtri(u)[:, 0] - 0.6) / 0.7)**2

        def cheap(u):
            z = ndtri(u)[:, 0]
            return full(u) + 1.5 * np.sin(3 * z) + 0.6 * z

        result = run_smc(full, 1, cheap_loglike=cheap, n_particles=4096,
                         inner_steps=10, outer_steps=8, seed=17)
        z = ndtri(result['samples'])[:, 0]
        self.assertAlmostEqual(z.mean(), 0.6 / 1.49, delta=0.05)
        self.assertAlmostEqual(z.var(), 0.49 / 1.49, delta=0.05)
        exact = -0.5 * (np.log1p(1 / 0.49) + 0.6**2 / 1.49)
        self.assertAlmostEqual(result['logz'], exact, delta=0.08)
        self.assertTrue(any(h['outer_accept'] < 0.9 for h in result['history']))
        self.assertGreater(result['n_cheap_evals'], result['n_evals'])

    def test_cheap_support_fallback(self):
        result = run_smc(lambda u: np.zeros(len(u)), 1,
                         cheap_loglike=lambda u: np.full(len(u), -np.inf),
                         n_particles=256, inner_steps=2, outer_steps=2, seed=8)
        self.assertEqual(result['logz'], 0)
        self.assertAlmostEqual(result['samples'].mean(), 0.5, delta=0.07)

    def test_adaptive_lengths_avoid_excess_da_work(self):
        flat = lambda u: np.zeros(len(u))
        options = dict(cheap_loglike=flat, n_particles=512,
                       inner_steps=25, outer_steps=8, seed=30, folds=1)
        fixed = run_smc(flat, 2, adaptive_steps=False, **options)
        adaptive = run_smc(flat, 2, adaptive_steps=True, **options)
        self.assertEqual(adaptive['logz'], 0)
        self.assertLess(adaptive['n_evals'], 0.6 * fixed['n_evals'])
        self.assertLess(adaptive['n_cheap_evals'], 0.25 * fixed['n_cheap_evals'])
        np.testing.assert_allclose(adaptive['samples'].mean(0), [0.5, 0.5], atol=0.04)

    def test_matched_support_screening_preserves_prior_volume(self):
        def full(u):
            self.assertTrue(np.all(u[:, 0] < 0.2))
            return np.full(len(u), 2.0)

        cheap = lambda u: np.where(u[:, 0] < 0.2, 0.0, -np.inf)
        result = run_smc(full, 1, cheap_loglike=cheap, same_support=True,
                         n_particles=2048, inner_steps=5, outer_steps=2, seed=7)
        self.assertAlmostEqual(result['logz'], 2 + np.log(0.2), delta=0.08)
        self.assertTrue(np.all(result['samples'][:, 0] < 0.2))

    def test_stiff_map_roundtrip_and_jacobian(self):
        rng = np.random.default_rng(4)
        z = rng.standard_normal((100, 2))

        def predict(u):
            p = ndtri(u)
            return (p[:, :1] - p[:, 1:]**2), np.full((len(u), 1), 0.01)

        model = GaussianModel(predict, [0])
        mapping = _StiffMap(z, 0, 0.5, lambda z: model(ndtr(z)),
                            lambda z: predict(ndtr(z)), model.data)
        x, jac = mapping.forward(z)
        back, jac_back = mapping.inverse(x)
        np.testing.assert_allclose(back, z, atol=1e-10)
        np.testing.assert_allclose(jac, jac_back)
        delta = 1e-5
        dx = np.eye(2) * delta
        point = x[:1]
        base = mapping.inverse(point)[0][0]
        matrix = np.column_stack([(mapping.inverse(point + step)[0][0] - base) / delta
                                  for step in dx])
        self.assertAlmostEqual(np.linalg.slogdet(matrix)[1], jac[0], places=5)

    def test_stiff_curved_target_evidence(self):
        from scipy.integrate import quad

        sd = 0.05

        def predict(u):
            z = ndtri(u)
            return z[:, :1] - z[:, 1:]**2, np.full((len(u), 1), sd**2)

        model = GaussianModel(predict, [0.0])
        normalizer = quad(lambda z: np.exp(-z**2 / 2 - z**4 / (2 * (1 + sd**2))) /
                          np.sqrt(2 * np.pi), -8, 8)[0] * sd / np.sqrt(1 + sd**2)
        # GaussianModel includes the observation-density normalizer.
        exact = np.log(normalizer) - np.log(np.sqrt(2 * np.pi) * sd)
        result = run_smc(model, 2, cheap_loglike=model, stiff_index=0,
                         n_particles=2048, inner_steps=10, outer_steps=4, seed=9)
        self.assertAlmostEqual(result['logz'], exact, delta=0.12)
        z = ndtri(result['samples'])
        self.assertLess(np.std(z[:, 0] - z[:, 1]**2), 1.15 * sd)

    def test_wavelength_cap_and_reset(self):
        from platon._atmosphere_solver import AtmosphereSolver

        atm = AtmosphereSolver.__new__(AtmosphereSolver)
        atm.raw = {"key": ("xsec", (), 1)}
        atm.orig_lambda_grid = np.arange(101, dtype=float)
        atm.lambda_grid = atm.orig_lambda_grid.copy()
        atm.wavelength_bins = None
        bins = [[0, 50], [50, 100]]
        atm.change_wavelength_bins(bins, max_points_per_bin=4)
        np.testing.assert_array_equal(atm.lambda_grid, [1, 17, 33, 49, 51, 67, 83, 99])
        self.assertEqual(atm._bin_info["bin_idx"].shape, (2, 4))
        atm.change_wavelength_bins(bins)
        self.assertEqual(atm.N_lambda, 98)
        atm.change_wavelength_bins(None)
        np.testing.assert_array_equal(atm.lambda_grid, atm.orig_lambda_grid)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            atm.change_wavelength_bins(bins, max_points_per_bin=0)

    def test_retriever_result(self):
        from platon.combined_retriever import CombinedRetriever
        from tests.test_nautilus import _FitInfo

        retriever = CombinedRetriever()
        fi = _FitInfo()
        fi._stellar_grid_options = lambda: {}

        def likelihood(params, *args, ret_model=False, ret_best_fit=False,
                       lnlike_per_point=False, **kwargs):
            if ret_best_fit:
                return None, None, None, None
            if ret_model:
                return np.array([params[1]]), np.ones(1)
            return np.array([-0.5 * params[1]**2])

        with mock.patch.object(retriever, '_ln_like', side_effect=likelihood), \
             mock.patch('platon.combined_retriever.write_param_estimates_file'), \
             mock.patch('platon.combined_retriever.psisloo', return_value=(0, [], [])):
            result = retriever.run_smc(None, np.array([0.0]), None, None, None, None,
                                       fi, n_particles=64, inner_steps=2, outer_steps=2,
                                       num_final_samples=2, seed=5)
        self.assertEqual(result.retrieval_type, 'smc')
        self.assertEqual(result.samples.shape, (64, 2))
        self.assertEqual(result.logl.shape, result.weights.shape)
        self.assertEqual(len(result.equal_samples), 64)
        self.assertAlmostEqual(result.weights.sum(), 1)
        self.assertTrue(np.all(result.samples[:, 0] >= 800))
        self.assertAlmostEqual(result.final_logz, result.logz[-1])
        from platon.plotter import Plotter
        import matplotlib.pyplot as plt
        Plotter.plot_retrieval_corner(result)
        Plotter.plot_retrieval_TP_profiles(result, num_samples=2)
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
