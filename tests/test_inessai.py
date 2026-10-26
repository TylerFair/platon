import contextlib
import os
import signal
import tempfile
import unittest
from unittest import mock

import numpy as np
import scipy.stats

from platon.combined_retriever import CombinedRetriever
from platon.constants import M_jup, R_jup, R_sun

try:
    import torch
    from nessai.samplers import ImportanceNestedSampler
    from platon import _inessai, _inessai_fast
except ImportError:  # optional dependencies
    _inessai = None

SIGMA_T, SIGMA_R = 50., .02 * R_jup


def gaussian_fit():
    fit = CombinedRetriever.get_default_fit_info(Rs=R_sun, Mp=M_jup, Rp=R_jup, T=1200)
    fit.add_uniform_fit_param("T", 800, 1600)
    fit.add_uniform_fit_param("Rp", .9 * R_jup, 1.1 * R_jup)
    return fit


def gaussian_ln_like(params, *args, ret_best_fit=False, lnlike_per_point=False, **kwargs):
    """A normalized Gaussian likelihood: ln Z = -ln(prior volume)."""
    if ret_best_fit:
        return None, None, None, None
    terms = np.array([scipy.stats.norm.logpdf(params[0], 1200, SIGMA_T),
                      scipy.stats.norm.logpdf(params[1], R_jup, SIGMA_R)])
    return terms if lnlike_per_point else terms.sum()


@unittest.skipIf(_inessai is None, "nessai 0.15.2 and torch are not installed")
class TestInessai(unittest.TestCase):
    TRUTH = -np.log(800 * .2 * R_jup)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        previous = os.getcwd()
        os.chdir(self.directory.name)
        self.addCleanup(os.chdir, previous)

    def run_inessai(self, output, ln_like=gaussian_ln_like, **kwargs):
        retriever = CombinedRetriever()
        settings = dict(output=output, n_live=100, target_ess=0, min_final_draws=8192,
                        max_final_draws=8192, seed=3, num_final_samples=2, verbose=False,
                        checkpoint_on_iteration=True, checkpoint_interval=1)
        settings.update(kwargs)
        with mock.patch.object(retriever, "_ln_like", side_effect=ln_like), \
             mock.patch("platon.combined_retriever.write_param_estimates_file"), \
             mock.patch("platon.combined_retriever.psisloo",
                        return_value=(0, np.zeros(2), np.zeros(2))):
            return retriever.run_inessai(None, None, None, None, None, None,
                                         gaussian_fit(), **settings)

    def test_evidence_and_result_fields(self):
        result = self.run_inessai("run")
        self.reference = result
        self.assertEqual(result.retrieval_type, "inessai")
        self.assertLess(abs(result.final_logz - self.TRUTH), 5 * result.logz_err + .01)
        self.assertLess(result.pareto_k, .7)
        self.assertEqual(result.n_draws, 8192)
        self.assertGreater(result.n_calls, 8192)
        self.assertEqual(len(result.draws), len(result.weights))
        self.assertAlmostEqual(result.weights.sum(), 1)
        self.assertEqual(result.samples.shape, (int(result.ess) + 1, 2))
        mean = result.weights @ result.draws
        self.assertLess(abs(mean[0] - 1200), 5 * SIGMA_T / np.sqrt(result.ess))
        # A finished run is read back from its output directory
        again = self.run_inessai("run")
        np.testing.assert_array_equal(again.logw, result.logw)
        # A different setting in the same directory is refused
        with self.assertRaisesRegex(ValueError, "different settings"):
            self.run_inessai("run", n_live=200)

    def test_resume_reproduces_uninterrupted_run(self):
        reference = self.run_inessai("reference")
        # Stop at the first nested-sampling checkpoint
        with self.assertRaises(_inessai.Checkpointed) as stopped:
            self.run_inessai("split_ns", max_seconds=0)
        self.assertEqual(stopped.exception.code, 75)
        resumed = self.run_inessai("split_ns")
        # Stop by signal during the final importance sample
        calls = [0]
        n_nested = reference.n_calls - reference.n_draws

        def interrupting(*args, **kwargs):
            calls[0] += 1
            if calls[0] == n_nested + 3000:
                os.kill(os.getpid(), signal.SIGUSR1)
            return gaussian_ln_like(*args, **kwargs)
        with self.assertRaises(_inessai.Checkpointed):
            self.run_inessai("split_is", ln_like=interrupting)
        self.assertTrue(os.path.exists("split_is/is_state.pkl"))
        finished = self.run_inessai("split_is")
        for result in (resumed, finished):
            np.testing.assert_array_equal(result.logw, reference.logw)
            np.testing.assert_array_equal(result.draws, reference.draws)
            self.assertEqual(result.final_logz, reference.final_logz)

    def test_overhead_removal_is_exact_on_cpu(self):
        fast = self.run_inessai("fast")
        with mock.patch.object(_inessai_fast, "installed", contextlib.nullcontext):
            stock = self.run_inessai("stock")
        # The same run: in-place mixture updates round the last bit of
        # log Q differently from stock's logsumexp over every flow
        np.testing.assert_array_equal(fast.draws, stock.draws)
        self.assertEqual(fast.n_calls, stock.n_calls)
        np.testing.assert_allclose(fast.logw, stock.logw, rtol=0, atol=1e-12)

    def test_jax_flow_densities_match_torch(self):
        from nessai.flowmodel.importance import ImportanceFlowModel
        torch.manual_seed(1)
        model = ImportanceFlowModel(flow_config=dict(_inessai.flow_config(), n_inputs=2),
                                    training_config=_inessai.training_config(),
                                    output=self.directory.name)
        model.initialise()
        for _ in range(3):
            model.add_new_flow(reset=True)
        x = np.random.default_rng(2).normal(size=(500, 2))
        stock = model.log_prob_all(x)
        jax_flows = _inessai_fast.JaxFlows(model)
        jax_flows.sync()
        np.testing.assert_allclose(jax_flows.evaluate(x, np.arange(model.n_models)), stock,
                                   rtol=0, atol=1e-4)


@unittest.skipIf(_inessai is None, "nessai 0.15.2 and torch are not installed")
class TestInessaiParts(unittest.TestCase):
    def test_cube_transform_matches_fit_info(self):
        fit = gaussian_fit()
        fit.add_gaussian_fit_param("Mp", .1 * M_jup)
        u = np.random.default_rng(5).random((200, 3))
        expected = np.array([fit._from_unit_interval_array(row) for row in u])
        np.testing.assert_array_equal(_inessai.cube_transform(fit)(u), expected)
        x = expected
        np.testing.assert_allclose(_inessai.ln_prior(fit, x),
                                   [fit._ln_prior(row) for row in x], rtol=0, atol=1e-12)

    def test_tail_base_distribution(self):
        x = torch.tensor(np.random.default_rng(6).normal(scale=2, size=(50, 3)))
        for independent in (True, False):
            base = _inessai.NormalStudentT((3,), fraction=.1, df=5., independent=independent)
            normal = scipy.stats.multivariate_normal(np.zeros(3)).logpdf(x.numpy())
            if independent:
                student = scipy.stats.t(5).logpdf(x.numpy()).sum(axis=1)
            else:
                student = scipy.stats.multivariate_t(np.zeros(3), np.eye(3), df=5).logpdf(x.numpy())
            expected = np.logaddexp(normal + np.log(.9), student + np.log(.1))
            np.testing.assert_allclose(base.log_prob(x).numpy(), expected, rtol=1e-10)
            self.assertEqual(tuple(base.sample(7).shape), (7, 3))
        with self.assertRaises(ValueError):
            _inessai.NormalStudentT((3,), fraction=0)

    def make_sampler(self):
        sampler = _inessai.FiniteThresholdImportanceNestedSampler.__new__(
            _inessai.FiniteThresholdImportanceNestedSampler)
        sampler.plot = False
        sampler.min_remove = 1
        sampler.min_samples = 25
        sampler.draw_constant = True
        sampler.max_samples = None
        return sampler, np.zeros(200, dtype=[("logL", float), ("logW", float)])

    def test_threshold_guard(self):
        sampler, samples = self.make_sampler()
        samples["logL"][:120] = -np.inf
        samples["logL"][120:] = np.arange(80) - 40.
        before = samples.copy()
        self.assertTrue(np.isneginf(ImportanceNestedSampler.determine_log_likelihood_threshold(
            sampler, samples)))
        self.assertEqual(sampler.determine_log_likelihood_threshold(samples), -40.)
        np.testing.assert_array_equal(samples, before)
        # A finite threshold is unchanged
        samples["logL"] = np.arange(200) - 200.
        self.assertEqual(ImportanceNestedSampler.determine_log_likelihood_threshold(
            sampler, samples), sampler.determine_log_likelihood_threshold(samples))
        samples["logL"] = -np.inf
        with self.assertRaisesRegex(RuntimeError, "No finite likelihood"):
            sampler.determine_log_likelihood_threshold(samples)


if __name__ == "__main__":
    unittest.main()
