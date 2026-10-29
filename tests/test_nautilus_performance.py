"""Numerical, lifecycle and trajectory gates for streaming shell moments."""

import inspect
import shutil

import numpy as np
import pytest
from scipy.special import logsumexp

nautilus = pytest.importorskip("nautilus")
from platon._nautilus_performance import _Moments, install_streaming_shell_statistics


def make_sampler(path=None, seed=17):
    kwargs = {}
    if "backend" in inspect.signature(nautilus.Sampler).parameters:
        kwargs["backend"] = "numpy"
    return nautilus.Sampler(
        lambda x: x,
        lambda x: -np.sum(((x - 0.4) / 0.08) ** 2, axis=-1) / 2,
        n_dim=3, n_live=80, n_networks=0, n_batch=32,
        vectorized=True, seed=seed, filepath=path, **kwargs)


def one_shell(values, discard=False):
    sampler = make_sampler()
    sampler.run(n_like_max=32)
    values = np.asarray(values, dtype=np.float64)
    sampler.points = [np.zeros((len(values), 3))]
    sampler.log_l = [values.copy()]
    sampler.shell_n_sample[:] = len(values)
    sampler.shell_n_sample_exp = np.array([0])
    sampler.shell_end_exp = np.array([0])
    sampler.explored = True
    sampler.discard_exploration = discard
    return sampler


def append_values(sampler, values):
    values = np.asarray(values, dtype=np.float64)
    sampler.sample_shell = lambda shell: (np.zeros((len(values), 3)), len(values))

    def evaluate(points):
        sampler.n_like += len(points)
        return values.copy(), None

    sampler.evaluate_likelihood = evaluate
    sampler.add_samples(0)


def summaries(sampler):
    return np.array([sampler.log_z, sampler.n_eff,
                     sampler.shell_log_v[0], sampler.shell_n[0]])


@pytest.mark.parametrize("discard", [False, True])
@pytest.mark.parametrize("offset", [-1e5, 0, 1200])
def test_extreme_weights_match_full_reductions(discard, offset):
    rng = np.random.default_rng(94)
    values = np.full(10, -np.inf)
    reference = one_shell(values, discard)
    streaming = one_shell(values, discard)
    controller = install_streaming_shell_statistics(streaming, refresh_interval=19)
    chunks = [np.full(5, -np.inf), rng.normal(offset, 1, 512),
              np.array([offset + 700]), np.full(512, offset - 1000)]
    chunks += [rng.normal(offset + 699, 10, 128) for _ in range(100)]
    for chunk in chunks:
        append_values(reference, chunk)
        append_values(streaming, chunk)
        np.testing.assert_allclose(summaries(streaming), summaries(reference),
                                   rtol=2e-10, atol=5e-10)
    assert controller.diagnostics["streaming_updates"] > 90
    controller.refresh()
    np.testing.assert_array_equal(summaries(streaming), summaries(reference))


def test_explicit_update_discard_and_array_replacement_invalidate_cache():
    values = np.arange(100, dtype=float)
    sampler = one_shell(values)
    controller = install_streaming_shell_statistics(sampler)
    append_values(sampler, [100, 101])
    sampler.log_l[0][0] = 1000
    sampler.update_shell_info(0)
    append_values(sampler, [102, 103])
    before = summaries(sampler)
    controller.refresh()
    np.testing.assert_allclose(summaries(sampler), before, rtol=1e-12)
    sampler.shell_end_exp[:] = 50
    sampler.shell_n_sample_exp[:] = 50
    sampler.discard_exploration = True
    append_values(sampler, [104])
    before = summaries(sampler)
    controller.refresh()
    np.testing.assert_allclose(summaries(sampler), before, rtol=1e-12)
    assert sampler.shell_n[0] == 55
    sampler.log_l[0] = np.zeros(105)
    append_values(sampler, [0])
    assert sampler.n_eff == pytest.approx(56)


@pytest.mark.parametrize("discard", [False, True])
@pytest.mark.parametrize("seed", [14, 71, 940])
def test_seeded_trajectory_and_final_statistics(discard, seed):
    reference = make_sampler(seed=seed)
    streaming = make_sampler(seed=seed)
    controller = install_streaming_shell_statistics(streaming, refresh_interval=11)
    kwargs = {"n_eff": 2000, "discard_exploration": discard, "n_like_max": 20000}
    assert reference.run(**kwargs)
    assert streaming.run(**kwargs)
    assert streaming.n_like == reference.n_like
    # These seeds do not have numerical allocation ties.  Proposal arrays and
    # the final full-reduction statistics can consequently agree exactly.
    for left, right in zip(streaming.posterior(), reference.posterior()):
        np.testing.assert_array_equal(left, right)
    assert streaming.log_z == reference.log_z
    assert streaming.n_eff == reference.n_eff
    assert streaming.rng.bit_generator.state == reference.rng.bit_generator.state
    assert controller.diagnostics["streaming_updates"] > 0
    controller.close()
    assert not hasattr(streaming, "_platon_streaming_statistics")
    assert "run" not in vars(streaming)
    controller.close()


@pytest.mark.parametrize("discard", [False, True])
def test_checkpoint_resume_rebuilds_moments(tmp_path, discard):
    path = tmp_path / "streaming.h5"
    sampler = make_sampler(path=path)
    controller = install_streaming_shell_statistics(sampler)
    assert sampler.run(n_eff=0, discard_exploration=discard)
    assert not sampler.run(n_eff=2000, discard_exploration=discard,
                           n_like_max=sampler.n_like + 256)
    resumed_path = tmp_path / "resumed.h5"
    shutil.copyfile(path, resumed_path)
    resumed = make_sampler(path=resumed_path)
    resumed_controller = install_streaming_shell_statistics(resumed)
    for current in (sampler, resumed):
        assert current.run(n_eff=2000, discard_exploration=discard, n_like_max=20000)
    for left, right in zip(sampler.posterior(), resumed.posterior()):
        np.testing.assert_array_equal(left, right)
    assert sampler.log_z == resumed.log_z
    assert sampler.n_eff == resumed.n_eff
    assert controller.diagnostics["streaming_updates"] > 0
    assert resumed_controller.diagnostics["streaming_updates"] > 0


def test_invalid_settings_and_duplicate_installation():
    sampler = make_sampler()
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            install_streaming_shell_statistics(sampler, refresh_interval=value)
    controller = install_streaming_shell_statistics(sampler)
    with pytest.raises(ValueError, match="already installed"):
        install_streaming_shell_statistics(sampler)
    controller.close()


def test_infinite_target_does_not_refresh_after_every_batch():
    sampler = one_shell(np.zeros(100))
    controller = install_streaming_shell_statistics(sampler)
    assert not sampler.run(n_eff=np.inf, n_like_max=sampler.n_like + 256)
    assert controller.diagnostics["streaming_updates"] == 8
    assert controller.diagnostics["exact_refreshes"] == 2


def test_compensated_moments_across_thousands_of_rare_tail_batches():
    values = np.random.default_rng(9420).normal(1200, 4, size=4096 * 64)
    values[::1973] += 25
    moments = _Moments(np.empty(0), 0)
    for stop in range(64, len(values) + 1, 64):
        moments.append(values[stop - 64:stop])
        if stop % (256 * 64) == 0:
            actual = [moments.maximum + np.log(moments.total),
                      2 * moments.maximum + np.log(moments.squared)]
            expected = [logsumexp(values[:stop]), logsumexp(2 * values[:stop])]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_run_wrapped_as_kwargs_only():
    """Hooks (e.g. nautilus_accel's stall stop) may replace run with run(**kwargs)."""
    class Wrapped(nautilus.Sampler):
        def run(self, **kwargs):
            return super().run(**kwargs)

    def build(cls):
        kwargs = {"backend": "numpy"} if "backend" in inspect.signature(nautilus.Sampler).parameters else {}
        return cls(lambda x: x, lambda x: -np.sum(((x - 0.4) / 0.08) ** 2, axis=-1) / 2,
                   n_dim=3, n_live=80, n_networks=0, n_batch=32, vectorized=True, seed=5, **kwargs)

    reference, streaming = build(nautilus.Sampler), build(Wrapped)
    controller = install_streaming_shell_statistics(streaming)
    run = {"n_eff": 2000, "discard_exploration": True, "n_like_max": 20000}
    assert reference.run(**run) and streaming.run(**run)
    assert streaming.log_z == reference.log_z and streaming.n_like == reference.n_like
    controller.close()
