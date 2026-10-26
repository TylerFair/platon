"""Importance nested sampling with normalizing flows, for CombinedRetriever.run_inessai.

The sampler is nessai's importance nested sampler (i-nessai) in the unit cube
of the sampled parameters.  Its successive flows form a mixture that covers
the posterior; once it converges, the mixture is frozen and a fresh,
independent importance sample from it gives the evidence, its error, the
posterior weights and a Pareto-k reliability diagnostic.  Three changes to
stock i-nessai 0.15.2, none of which changes its algorithm:

- an entropy threshold of -inf (possible when many prior draws have zero
  likelihood) is moved to the least finite likelihood, so that the run
  advances; rejected points keep their weights in the evidence;
- the flows' base distribution has a 10% independent Student-t component,
  which keeps their tails heavy enough for importance sampling;
- the per-level bookkeeping is O(N) instead of O(N x levels), and flow
  densities are evaluated in JAX on a GPU (_inessai_fast).

Requires nessai 0.15.2 and PyTorch: pip install "platon[inessai]".
"""
import json
import logging
import math
import os
import pickle
import signal
import threading
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
import scipy.stats
import torch
from glasflow.nflows.distributions import Distribution
import nessai
from nessai import config as nessai_config
import nessai.flowsampler as flowsampler
from nessai.model import Model
from nessai.samplers import ImportanceNestedSampler
from nessai.utils.io import safe_file_dump

from ._params import _GaussianParam, _UniformParam
from .psis import psislw

NESSAI_VERSION = "0.15.2"
if nessai.__version__ != NESSAI_VERSION:
    raise ImportError("run_inessai patches nessai {} internals; found nessai {}".format(
        NESSAI_VERSION, nessai.__version__))

logger = logging.getLogger(__name__)
PLANNING_DRAWS = 8192
BATCH_DRAWS = 2048
PRIOR_MASS = 0.01


class Checkpointed(SystemExit):
    """Raised after a complete checkpoint, on SIGTERM/SIGUSR1 or after
    max_seconds.  Uncaught, the process exits with status 75; call
    run_inessai again with the same output directory to resume."""

    def __init__(self, output):
        super().__init__(75)
        self.output = output

    def __str__(self):
        return "i-nessai checkpointed in {}; run again to resume".format(self.output)


class _Stop(Exception):
    pass


class NormalStudentT(Distribution):
    """Normal / Student-t mixture base distribution for the flows.

    fraction of the mass is a product of independent Student-t distributions
    with df degrees of freedom (or, with independent=False, a multivariate
    Student-t); the rest is a standard normal.
    """

    def __init__(self, shape, fraction=.1, df=5., scale=1., independent=False):
        super().__init__()
        if not 0 < fraction < 1 or df <= 2 or scale <= 0:
            raise ValueError("Need 0 < fraction < 1, df > 2 and scale > 0")
        self._shape = torch.Size(shape)
        self.dim = math.prod(shape)
        self.fraction = float(fraction)
        self.df = float(df)
        self.scale = float(scale)
        self.independent = bool(independent)
        normal = -self.dim * math.log(scale * math.sqrt(2 * math.pi))
        student = (math.lgamma((df + self.dim) / 2) - math.lgamma(df / 2)
                   - self.dim * math.log(scale * math.sqrt(df * math.pi)))
        if independent:
            student = self.dim * (math.lgamma((df + 1) / 2) - math.lgamma(df / 2)
                                  - math.log(scale * math.sqrt(df * math.pi)))
        self.register_buffer("_normal_constant", torch.tensor(normal, dtype=torch.float64),
                             persistent=False)
        self.register_buffer("_student_constant", torch.tensor(student, dtype=torch.float64),
                             persistent=False)

    def _log_prob(self, inputs, context):
        if context is not None or inputs.shape[1:] != self._shape:
            raise ValueError("Expected unconditioned inputs with the configured shape")
        squared = ((inputs / self.scale)**2).reshape(len(inputs), -1)
        radius2 = squared.sum(dim=1)
        normal = self._normal_constant - .5 * radius2
        student = self._student_constant - (self.df + self.dim) / 2 * torch.log1p(radius2 / self.df)
        if self.independent:
            student = self._student_constant - \
                (self.df + 1) / 2 * torch.log1p(squared / self.df).sum(dim=1)
        return torch.logaddexp(normal + math.log1p(-self.fraction),
                               student + math.log(self.fraction))

    def _sample(self, num_samples, context):
        if context is not None:
            raise ValueError("Conditional sampling is not supported")
        device = self._normal_constant.device
        samples = torch.randn((num_samples, *self._shape), device=device) * self.scale
        tail = torch.rand(num_samples, device=device) < self.fraction
        count = int(tail.sum())
        if count:
            df = torch.tensor(self.df, device=device)
            shape = (count, *self._shape) if self.independent else (count,)
            radial = torch.distributions.Chi2(df).sample(shape) / self.df
            if not self.independent:
                # One radial scale per row: a multivariate Student-t
                radial = radial.reshape((-1,) + (1,) * len(self._shape))
            samples[tail] /= radial.sqrt()
        return samples

    def _mean(self, context):
        if context is not None:
            raise ValueError("Conditional means are not supported")
        return torch.zeros(self._shape, device=self._normal_constant.device)


class FiniteThresholdImportanceNestedSampler(ImportanceNestedSampler):
    """nessai can choose -inf as an entropy threshold when many points have
    zero likelihood.  No point is then discarded, and the default live/total
    evidence stopping statistic is zero.  Move only that threshold to the
    least finite likelihood; rejected points remain in the importance sample
    history with their original weights."""

    def determine_log_likelihood_threshold(self, samples, method="entropy", **kwargs):
        threshold = super().determine_log_likelihood_threshold(samples, method=method, **kwargs)
        if np.isneginf(threshold):
            finite = samples["logL"][np.isfinite(samples["logL"])]
            if not finite.size:
                raise RuntimeError("No finite likelihood points for an i-nessai level")
            threshold = finite.min()
            logger.info("Advancing -inf threshold to least finite likelihood %s; "
                        "zero-weight points remain in the evidence history", threshold)
        return threshold


def flow_config():
    return dict(ftype="realnvp", n_blocks=6, n_neurons=64, n_layers=2,
                batch_norm_between_layers=False, linear_transform="lu",
                distribution=NormalStudentT,
                distribution_kwargs=dict(fraction=.1, df=5., independent=True))


def training_config():
    return dict(max_epochs=300, patience=10, batch_size=512, lr=1e-3, device_tag="cpu")


class CubeModel(Model):
    """The unit cube of the sampled parameters, with a vectorized likelihood."""

    def __init__(self, ndim, log_likelihood):
        self.names = ["x{}".format(i) for i in range(ndim)]
        self.bounds = {name: [0., 1.] for name in self.names}
        self.vectorised_likelihood = True
        self.vectorised_prior = True
        self.vectorised_prior_unit_hypercube = True
        self.cube_log_likelihood = log_likelihood

    def array(self, x):
        return np.column_stack([np.atleast_1d(x[name]) for name in self.names])

    def log_prior(self, x):
        u = self.array(x)
        return np.where(np.all((u > 0) & (u < 1), axis=1), 0., -np.inf)

    def log_likelihood(self, x):
        return self.cube_log_likelihood(self.array(x))

    def to_unit_hypercube(self, x):
        return x.copy()

    def from_unit_hypercube(self, x):
        return x.copy()


def cube_transform(fit_info):
    """Vectorized fit_info._from_unit_interval_array, with the same operations."""
    names = list(fit_info.fit_param_names)
    params = [fit_info.all_params[name] for name in names]
    for param in params:
        if isinstance(param, _UniformParam):
            if not (np.isfinite(param.low_lim) and np.isfinite(param.high_lim)):
                raise ValueError("Limit cannot be infinity")
        elif not isinstance(param, _GaussianParam):
            raise ValueError("run_inessai supports uniform and Gaussian priors")
    pairs = [(names.index(cold), names.index(hot))
             for cold, hot in getattr(fit_info, "ordered_pairs", [])]

    def transform(u):
        u = np.asarray(u, dtype=float)
        x = np.empty_like(u)
        for i, param in enumerate(params):
            if isinstance(param, _GaussianParam):
                x[:, i] = scipy.stats.norm.ppf(u[:, i], param.best_guess, param.std)
            else:
                x[:, i] = param.low_lim + (param.high_lim - param.low_lim) * u[:, i]
        for cold, hot in pairs:
            x[:, cold], x[:, hot] = np.minimum(x[:, cold], x[:, hot]), \
                np.maximum(x[:, cold], x[:, hot])
        return x
    return transform


def ln_prior(fit_info, x):
    """fit_info._ln_prior for rows inside the prior limits, vectorized."""
    result = np.zeros(len(x))
    for i, name in enumerate(fit_info.fit_param_names):
        param = fit_info.all_params[name]
        if isinstance(param, _GaussianParam):
            result += np.log(scipy.stats.norm.pdf(x[:, i], param.best_guess, param.std))
    return result


def importance_stats(log_weights, n_draws):
    """Evidence, its error, ESS and Pareto k of an importance sample of
    n_draws draws.  log_weights holds every evaluated draw, -inf included:
    the PSIS tail length depends on their number."""
    log_weights = np.asarray(log_weights, dtype=float)
    if not np.isfinite(log_weights).any():
        raise RuntimeError("No final draw has a finite likelihood")
    total = logsumexp(log_weights)
    weights = np.exp(log_weights - total)
    ess = 1 / np.sum(weights**2)
    smooth, k = psislw(np.array(log_weights))
    return dict(logz=float(total - np.log(n_draws)),
                logz_err=float(np.sqrt(max(0., (n_draws / ess - 1) / (n_draws - 1)))),
                ess=float(ess), pareto_k=float(k), psis_ess=float(1 / np.sum(np.exp(2 * smooth))),
                max_weight=float(weights.max()), n_draws=int(n_draws))


def _save_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=float) + "\n")
    temporary.replace(path)


def _independent_draws(ns, model, output, prefix, weights, n_total, should_stop):
    """Fixed-size importance sample from the frozen mixture `weights`, in
    append-only batch files, resuming the proposal and torch RNGs exactly."""
    path = output / (prefix + "_state.pkl")
    batches = output / (prefix + "_batches")
    batches.mkdir(exist_ok=True)
    if path.exists():
        with path.open("rb") as f:
            state = pickle.load(f)
        ns.proposal.rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        weights = state["weights"]
    else:
        state = dict(n=0, batches=[], weights=weights)
    while state["n"] < n_total:
        n = min(BATCH_DRAWS, n_total - state["n"])
        samples, logq, _ = ns.proposal.draw_from_flows(n, weights=weights)
        active = logq[:, weights > 0]
        if np.any(np.isnan(active) | np.isposinf(active)):
            raise RuntimeError("Nonfinite active component density in frozen mixture")
        u = model.array(samples)
        logl = model.cube_log_likelihood(u)
        # Draws the flows reject numerically keep zero weight in the denominator
        name = "batch_{:09d}.npz".format(state["n"])
        temporary = batches / (name + ".tmp")
        with temporary.open("wb") as f:
            np.savez(f, u=u, logl=logl, lw=logl - logsumexp(logq, b=weights, axis=1))
        temporary.replace(batches / name)
        state["batches"].append(name)
        state["n"] += n
        state["rng"] = ns.proposal.rng.bit_generator.state
        state["torch_rng"] = torch.get_rng_state()
        safe_file_dump(state, str(path), pickle, save_existing=False)
        if should_stop():
            raise _Stop
    data = []
    for name in state["batches"]:
        with np.load(batches / name) as batch:
            data.append({key: batch[key] for key in ("u", "lw", "logl")})
    values = {key: np.concatenate([d[key] for d in data]) for key in ("u", "lw", "logl")}
    return state["n"], values


def sample(cube_log_likelihood, ndim, output, n_live, seed, target_ess,
           min_final_draws, max_final_draws, max_seconds=None, verbose=True,
           nessai_kwargs=None):
    """Run or resume i-nessai and its final importance sample in output.

    Returns dict(u, logl, logw, n_draws, n_calls, training_logz) for every
    final draw (numerically rejected draws have logw -inf); raises
    Checkpointed after a checkpoint.
    """
    from . import _inessai_fast
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configuration = dict(n_live=n_live, seed=seed, target_ess=target_ess,
                         min_final_draws=min_final_draws, max_final_draws=max_final_draws,
                         ndim=ndim, nessai=NESSAI_VERSION,
                         nessai_kwargs=repr(sorted((nessai_kwargs or {}).items())))
    config_path = output / "configuration.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved != json.loads(json.dumps(configuration, default=float)):
            raise ValueError("{} holds a run with different settings: {}".format(output, saved))
    else:
        _save_json(config_path, configuration)
    done = output / "final_draws.npz"
    if done.exists():
        with np.load(done) as d:
            return {key: d[key] for key in d.files}

    start = time.perf_counter()
    deadline = np.inf if max_seconds is None else start + max_seconds
    stop = threading.Event()

    def should_stop():
        return stop.is_set() or time.perf_counter() >= deadline

    previous = {}
    if threading.current_thread() is threading.main_thread():
        for number in (signal.SIGTERM, signal.SIGUSR1):
            previous[number] = signal.signal(number, lambda *_: stop.set())
    if verbose:
        from nessai.utils.logging import configure_logger
        configure_logger(log_level="INFO")
    eps = nessai_config.general.eps
    model = CubeModel(ndim, cube_log_likelihood)
    try:
        torch.manual_seed(seed)

        def checkpoint(ns):
            ns._platon_torch_state = torch.get_rng_state()
            optimiser = getattr(ns.proposal.flow, "_optimiser", None)
            ns._platon_optimiser_state = optimiser.state_dict() if optimiser is not None else None
            safe_file_dump(ns, ns.resume_file, pickle, save_existing=True)
            live = ns.live_points_unit
            _save_json(output / "progress.json", dict(
                iteration=ns.iteration, logz=float(ns.log_evidence),
                finalised=bool(ns.finalised), n_calls=int(getattr(ns.samples_unit, "size", 0)),
                live_logl_max=float(np.max(live["logL"])) if live is not None and live.size else None))
            if should_stop():
                raise _Stop

        options = dict(nlive=n_live, seed=seed, plot=False, checkpoint_interval=60,
                       checkpoint_callback=checkpoint, flow_config=flow_config(),
                       training_config=training_config(), reparameterisation="logit",
                       threshold_method="entropy", threshold_kwargs={"q": .5},
                       min_samples=min(500, n_live // 2), tolerance=0., reset_flow=8,
                       draw_iid_live=False, save_log_q=True)
        for key, value in (nessai_kwargs or {}).items():
            if isinstance(value, dict) and isinstance(options.get(key), dict):
                options[key] = {**options[key], **value}
            else:
                options[key] = value
        with _inessai_fast.installed():
            original = flowsampler.ImportanceNestedSampler
            flowsampler.ImportanceNestedSampler = FiniteThresholdImportanceNestedSampler
            try:
                sampler = flowsampler.FlowSampler(
                    model, output=str(output), importance_nested_sampler=True, resume=True,
                    signal_handling=False, pytorch_threads=1, likelihood_chunksize=256,
                    eps=0., **options)
            finally:
                flowsampler.ImportanceNestedSampler = original
            ns = sampler.ns
            ns.save_log_q = True
            ns.training_samples.save_log_q = True
            if not ns.resumed:
                # nessai 0.15.2 leaves the proposal and training-split RNGs unseeded
                ns.proposal.rng = np.random.default_rng(seed + 30000000)
                ns.proposal.initialise()
                ns.proposal.flow.rng = np.random.default_rng(seed + 20000000)
            if hasattr(ns, "_platon_torch_state"):
                torch.set_rng_state(ns._platon_torch_state)
            optimiser = getattr(ns.proposal.flow, "_optimiser", None)
            if getattr(ns, "_platon_optimiser_state", None) is not None and optimiser is not None:
                optimiser.load_state_dict(ns._platon_optimiser_state)

            ns.nested_sampling_loop()
            if not ns.finalised:
                raise _Stop
            n_ns = int(ns.samples_unit.size)
            # Freeze the successive-flow mixture, keeping the prior with at
            # least 1% of its mass, and draw fresh points from it
            weights = np.asarray(ns.importance["posterior"], dtype=float)
            weights = (1 - PRIOR_MASS) * weights / weights.sum()
            weights[0] += PRIOR_MASS
            n_final = min_final_draws
            n_planning = 0
            if target_ess:
                n_planning, planning = _independent_draws(
                    ns, model, output, "planning", weights, PLANNING_DRAWS, should_stop)
                plan_path = output / "final_plan.json"
                if not plan_path.exists():
                    stats = importance_stats(planning["lw"], n_planning) \
                        if np.isfinite(planning["lw"]).any() else dict(ess=0.)
                    efficiency = max(stats["ess"] / n_planning, 1e-12)
                    required = int(np.ceil(1.25 * target_ess / efficiency))
                    _save_json(plan_path, dict(
                        n_final=min(max_final_draws, max(min_final_draws, required)),
                        planning=stats, note="planning draws are not in the final estimate"))
                n_final = json.loads(plan_path.read_text())["n_final"]
            n_draws, draws = _independent_draws(ns, model, output, "is", weights, n_final,
                                                should_stop)
        result = dict(u=draws["u"], logl=draws["logl"], logw=draws["lw"], n_draws=n_draws,
                      n_calls=n_ns + n_planning + n_draws, training_logz=float(ns.log_evidence))
        np.savez(done, **result)
        return result
    except _Stop:
        raise Checkpointed(output) from None
    finally:
        nessai_config.general.eps = eps
        for number, handler in previous.items():
            signal.signal(number, handler)
