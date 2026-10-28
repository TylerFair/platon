"""Repeat the frozen WASP-94b target with controlled Nautilus settings.

Run on an allocated compute node. Sources and inputs must first be captured
with freeze_sources.py. A 5,000/20,000-live comparison keeps n_update=5,000,
n_like_new_bound=50,000, n_eff=5,000 and all physical inputs fixed by default.
Sampling checkpoints retain matching unit-prior points and log likelihoods;
no equal-weight array is paired with unrelated importance weights.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process owns this output directory") from error
        yield


def _load_driver(snapshot):
    for name in ("platon", "nautilus"):
        if name in sys.modules:
            raise RuntimeError("Run in a fresh Python process before importing " + name)
    sys.path[:0] = [str(snapshot / "platon"), str(snapshot / "nautilus")]
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("wasp94_frozen_driver", snapshot / "driver.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.DATA = snapshot / "data"
    driver.ARCHIVE = snapshot / "archive.csv"
    return driver


class GracefulStop(Exception):
    """Raised only after the current sampler batch has been saved."""


def main(args):
    # Verify before any model import. The helper checks every captured source
    # and observation file, and explicitly describes external opacity identity.
    from freeze_sources import verify_snapshot

    started = time.perf_counter()
    snapshot, output = args.snapshot.resolve(), args.output.resolve()
    manifest = verify_snapshot(snapshot)
    identity = {
        "snapshot_sha256": file_hash(snapshot / "manifest.json"),
        "runner_sha256": file_hash(__file__),
        "n_live": args.n_live,
        "n_update": args.n_update,
        "n_like_new_bound": args.n_like_new_bound,
        "n_eff": args.n_eff,
        "n_networks": args.n_networks,
        "explore_batch": args.explore_batch,
        "sample_batch": args.sample_batch,
        "enlarge_per_dim": args.enlarge_per_dim,
        "split_threshold": args.split_threshold,
        "f_live": args.f_live,
        "n_shell": args.n_shell,
        "seed": args.seed,
        "backend": args.backend,
        "likelihood": args.likelihood,
        "likelihood_batch_size": args.likelihood_batch_size,
        "streaming_statistics": args.streaming_statistics,
        "marginalize_nuisance": True,
        "opacity_downsample": 1,
    }
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    identity_path = output / "identity.json"
    checkpoint = output / "checkpoint.hdf5"
    summary_path = output / "summary.json"
    if identity_path.exists():
        previous_identity = json.loads(identity_path.read_text())
        if previous_identity["identity"] != identity:
            raise ValueError("Resume refused: frozen target, runner or sampler settings changed")
        if not args.resume:
            raise ValueError("Output already exists; use --resume or a new output directory")
    elif checkpoint.exists():
        raise ValueError("Checkpoint has no identity.json; refusing an unverified resume")
    else:
        atomic_json(identity_path, {"identity": identity, "identity_hash": identity_hash,
                                   "snapshot": str(snapshot)})

    driver = _load_driver(snapshot)
    import h5py
    import jax
    import nautilus
    import numpy as np
    from scipy.special import logsumexp
    from threadpoolctl import threadpool_limits
    from platon.combined_retriever import CombinedRetriever

    if args.backend == "jax" and jax.default_backend() != "gpu":
        raise RuntimeError("--backend jax requires an allocated GPU; choose numpy for a CPU check")
    for package in (nautilus, sys.modules["platon"]):
        if not Path(package.__file__).resolve().is_relative_to(snapshot):
            raise RuntimeError("Package imported outside the verified source snapshot")
    previous = {}
    if checkpoint.exists():
        with h5py.File(checkpoint, "r") as stream:
            if stream.attrs.get("benchmark_identity") != identity_hash:
                raise ValueError("Checkpoint identity does not match this benchmark")
            if "benchmark_ledger" not in stream.attrs:
                raise ValueError("Checkpoint has no durable timing/call ledger")
            previous = json.loads(stream.attrs["benchmark_ledger"])

    data, bins, ranges = driver.load_data()
    full_fit = driver.build_fit(data, ranges)
    fit = CombinedRetriever._marginalize(full_fit)
    if len(full_fit.fit_param_names) != 22 or len(fit.fit_param_names) != 18:
        raise ValueError("Expected the frozen 22-parameter target with four marginalized offsets")
    for name in ("sector1.scatt_slope", "sector2.scatt_slope"):
        parameter = fit.all_params[name]
        if (parameter.low_lim, parameter.high_lim) != (-2, 15):
            raise ValueError("This experiment requires the current scattering slope prior [-2, 15]")
    calculator = driver.use_opacities(1)(**fit._stellar_grid_options())
    calculator.change_wavelength_bins(bins)
    retriever = CombinedRetriever()
    retriever._validate_params(full_fit, calculator)
    timings = {"likelihood_seconds": 0.0, "prior_seconds": 0.0,
               "likelihood_calls": 0, "prior_rows": 0, "checkpoint_seconds": 0.0,
               "checkpoint_writes": 0}

    def prior(cube):
        before = time.perf_counter()
        if args.likelihood == "batch":
            value = fit._from_unit_interval_batch(cube)
        else:
            value = fit._from_unit_interval_array(cube)
        timings["prior_seconds"] += time.perf_counter() - before
        timings["prior_rows"] += len(cube) if args.likelihood == "batch" else 1
        return value

    def scalar_likelihood(parameters):
        before = time.perf_counter()
        value = float(retriever._ln_like(parameters, calculator, None, fit,
                                        data["depth"], data["depth_err"], None, None))
        timings["likelihood_seconds"] += time.perf_counter() - before
        timings["likelihood_calls"] += 1
        return value

    guess = np.asarray([fit._get(name) for name in fit.fit_param_names])
    initial_log_l = [scalar_likelihood(guess) for _ in range(2)]
    if not np.all(np.isfinite(initial_log_l)) or initial_log_l[0] != initial_log_l[1]:
        raise ValueError("Repeated full-resolution starting likelihoods are invalid or differ")
    parity = None
    likelihood = scalar_likelihood
    if args.likelihood == "batch":
        from platon._batched_likelihood import BatchedLikelihood
        batched = BatchedLikelihood(retriever, calculator, None, fit,
                                    data["depth"], data["depth_err"], None, None,
                                    batch_size=args.likelihood_batch_size)

        def likelihood(parameters):
            before = time.perf_counter()
            value = np.asarray(batched(parameters))
            timings["likelihood_seconds"] += time.perf_counter() - before
            timings["likelihood_calls"] += len(parameters)
            return value

        # Probe both valid and invalid prior rows at native wavelength resolution.
        # This costs 34 physical evaluations (including two scalar warmups),
        # reported separately from Nautilus's n_like counter.
        cubes = np.random.default_rng(940028).random((15, len(fit.fit_param_names)))
        physical = np.vstack([guess, fit._from_unit_interval_batch(cubes)])
        reference = np.asarray([scalar_likelihood(row) for row in physical])
        candidate = likelihood(physical)
        same_support = np.array_equal(np.isfinite(reference), np.isfinite(candidate))
        same_values = np.array_equal(reference, candidate)
        finite = np.isfinite(reference) & np.isfinite(candidate)
        parity = {"points": len(physical), "same_support": bool(same_support),
                  "bitwise_equal": bool(same_values),
                  "finite_points": int(finite.sum()),
                  "maximum_absolute_difference": float(np.max(np.abs(reference[finite] - candidate[finite])))
                  if finite.any() else None}
        if not same_values:
            raise ValueError("Scalar/batched full-resolution likelihood parity failed: " + json.dumps(parity))
    warmup = dict(timings)
    for name in timings:
        timings[name] = 0 if isinstance(timings[name], int) else 0.0
    setup = {
        "identity_hash": identity_hash, "target_dimension": len(fit.fit_param_names),
        "full_dimension": len(full_fit.fit_param_names), "parameter_names": fit.fit_param_names,
        "priors": {name: [float(fit.all_params[name].low_lim), float(fit.all_params[name].high_lim)]
                   for name in fit.fit_param_names},
        "data_rows": len(data), "ranges": ranges, "warmup": warmup,
        "initial_log_likelihood": initial_log_l[-1], "jax_backend": jax.default_backend(),
        "scalar_batch_parity": parity,
        "devices": [str(device) + ": " + device.device_kind for device in jax.devices()],
        "nautilus_source": nautilus.__file__, "platon_source": sys.modules["platon"].__file__,
        "external_data": manifest.get("external_data"), "job_id": os.getenv("SLURM_JOB_ID"),
        "node": os.uname().nodename,
    }
    atomic_json(output / "setup.json", setup)
    if args.check_only:
        print(json.dumps({"check_only": True, **setup}), flush=True)
        return

    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    class CheckpointSampler(nautilus.Sampler):
        def __init__(self, *positional, **keywords):
            self._last_checkpoint_time = time.perf_counter()
            self._last_checkpoint_calls = 0
            super().__init__(*positional, **keywords)
            self._last_checkpoint_calls = self.n_like

        def checkpoint(self, force=False, final=False):
            due = (self.n_like - self._last_checkpoint_calls >= args.checkpoint_calls or
                   time.perf_counter() - self._last_checkpoint_time >= args.checkpoint_seconds)
            if not (force or due or stop_requested):
                return
            before = time.perf_counter()
            temporary = checkpoint.with_suffix(".tmp.hdf5")
            super().write(str(temporary), overwrite=True)
            checkpoint_work = time.perf_counter() - before
            timings["checkpoint_seconds"] += checkpoint_work
            timings["checkpoint_writes"] += 1
            ledger = current_ledger(final=final)
            with h5py.File(temporary, "a") as stream:
                stream.attrs["benchmark_identity"] = identity_hash
                stream.attrs["benchmark_ledger"] = json.dumps(ledger)
            temporary.replace(checkpoint)
            timings["checkpoint_seconds"] += time.perf_counter() - before - checkpoint_work
            self._last_checkpoint_time = time.perf_counter()
            self._last_checkpoint_calls = self.n_like

        def write(self, filepath, overwrite=False):
            self.checkpoint()

        def write_shell_update(self, filepath, shell):
            self.checkpoint()
            if stop_requested:
                raise GracefulStop()

    constructor = dict(n_dim=len(fit.fit_param_names), n_live=args.n_live,
                       n_update=args.n_update, n_like_new_bound=args.n_like_new_bound,
                       n_networks=args.n_networks, n_batch=args.explore_batch,
                       enlarge_per_dim=args.enlarge_per_dim,
                       split_threshold=args.split_threshold, seed=args.seed,
                       vectorized=args.likelihood == "batch", filepath=str(checkpoint), resume=args.resume)
    if args.backend != "stock":
        constructor["backend"] = args.backend
    sampler = CheckpointSampler(prior, likelihood, **constructor)
    controller = None
    if args.streaming_statistics:
        from platon._nautilus_performance import install_streaming_shell_statistics
        controller = install_streaming_shell_statistics(sampler)
    calls_before = int(sampler.n_like)
    exploration_calls = previous.get("exploration_calls")
    elapsed_before = float(previous.get("sampling_seconds", 0.0))
    previous_timings = previous.get("timings", {})
    sampling_started = time.perf_counter()
    exploration_seconds = float(previous.get("exploration_seconds", 0.0))
    phase = {"exploring": False, "started": None}
    previous_warmup = previous.get("warmup_totals", {})
    warmup_totals = {key: previous_warmup.get(key, 0) + value for key, value in warmup.items()}
    interrupted_segments = int(previous.get("interrupted_segments", 0))
    if previous and not previous.get("clean_segment_end", False):
        interrupted_segments += 1

    def current_ledger(final=False):
        active_exploration = (time.perf_counter() - phase["started"]
                              if phase["exploring"] and phase["started"] is not None else 0.0)
        return {
            "calls": int(sampler.n_like),
            "sampling_seconds": elapsed_before + time.perf_counter() - sampling_started,
            "exploration_seconds": exploration_seconds + active_exploration,
            "exploration_calls": (int(sampler.n_like) if sampler.explored and exploration_calls is None
                                  else exploration_calls),
            "timings": {key: previous_timings.get(key, 0) + value for key, value in timings.items()},
            "warmup_totals": warmup_totals,
            "native_calls_including_warmup": int(sampler.n_like) + warmup_totals["likelihood_calls"],
            "clean_segment_end": final,
            "interrupted_segments": interrupted_segments,
            "lost_work_after_checkpoint_unknown": interrupted_segments > 0,
            "accounting_note": "Counters are durable through this checkpoint; work after the last checkpoint of a hard-killed segment cannot be recovered.",
        }

    def state():
        log_z = sampler.log_z
        row = {"calls": int(sampler.n_like), "explored": bool(sampler.explored),
               "n_eff": float(sampler.n_eff), "log_z": None if log_z is None else float(log_z),
               "bounds": len(sampler.bounds), "seconds_this_segment": time.perf_counter() - sampling_started}
        if row["log_z"] is not None and not np.isfinite(row["log_z"]):
            row["log_z"] = None
        return row

    success = False
    stopped_by_signal = False
    clean_end = False
    try:
        with threadpool_limits(limits=args.blas_threads):
            while sampler.n_like < args.max_calls and time.perf_counter() - sampling_started < args.seconds:
                if stop_requested:
                    stopped_by_signal = True
                    break
                exploring = not sampler.explored
                sampler.n_batch = args.explore_batch if exploring else args.sample_batch
                before = time.perf_counter()
                phase.update(exploring=exploring, started=before)
                remaining = args.seconds - (before - sampling_started)
                success = sampler.run(f_live=args.f_live, n_shell=0 if exploring else args.n_shell,
                                      n_eff=0 if exploring else args.n_eff,
                                      discard_exploration=True, n_like_max=args.max_calls,
                                      timeout=min(args.progress_seconds, max(0, remaining)), verbose=False)
                if exploring:
                    exploration_seconds += time.perf_counter() - before
                    if sampler.explored:
                        exploration_calls = int(sampler.n_like)
                    # Exploration's n_eff=0 success is not posterior convergence.
                    success = False
                phase["started"] = None
                row = state()
                print(json.dumps(row), flush=True)
                with (output / "progress.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                if success:
                    break
            clean_end = True
    except GracefulStop:
        stopped_by_signal = True
        clean_end = True
    finally:
        if phase["exploring"] and phase["started"] is not None:
            exploration_seconds += time.perf_counter() - phase["started"]
        phase["started"] = None
        # A failed likelihood can leave an unfinished batch in memory. Keep
        # the last completed checkpoint intact rather than saving that state.
        if clean_end:
            if controller is not None:
                controller.refresh()
            sampler.checkpoint(force=True, final=True)

    elapsed = time.perf_counter() - sampling_started
    success = (sampler.explored and np.all(sampler.shell_n >= args.n_shell)
               and sampler.n_eff >= args.n_eff)
    total_timings = {key: previous_timings.get(key, 0) + value for key, value in timings.items()}
    log_z = sampler.log_z
    shell_mass = np.zeros(len(sampler.shell_n))
    finite = np.isfinite(sampler.shell_log_l + sampler.shell_log_v)
    if finite.any():
        contributions = sampler.shell_log_l[finite] + sampler.shell_log_v[finite]
        shell_mass[finite] = np.exp(contributions - logsumexp(contributions))
    inverse_ess = np.divide(shell_mass**2, sampler.shell_n_eff,
                            out=np.zeros_like(shell_mass), where=sampler.shell_n_eff > 0)
    total_squared = inverse_ess.sum()
    summary = {
        **state(), **current_ledger(final=True), "identity_hash": identity_hash, "success": bool(success),
        "stopped_by_signal": stopped_by_signal, "calls_before_segment": calls_before,
        "calls_this_segment": int(sampler.n_like) - calls_before,
        "exploration_calls": exploration_calls,
        "estimator_calls": int(sampler.shell_n.sum()) if sampler.explored else 0,
        "sampling_seconds": elapsed_before + elapsed,
        "exploration_seconds": exploration_seconds,
        "seconds_including_setup_this_segment": time.perf_counter() - started,
        "timings": total_timings,
        "sampler_overhead_seconds": elapsed_before + elapsed - total_timings["likelihood_seconds"],
        "timing_note": "Sampling wall includes prior, bound fitting/proposals, checkpoint I/O and reporting. Warmup is separate.",
        "checkpoint": str(checkpoint), "settings": identity,
        "streaming_statistics": None if controller is None else controller.diagnostics,
        "shells": [{"index": index, "calls": int(count), "mass": float(shell_mass[index]),
                    "n_eff": float(sampler.shell_n_eff[index]),
                    "inverse_ess_fraction": float(inverse_ess[index] / total_squared) if total_squared else 0.0}
                   for index, count in enumerate(sampler.shell_n)],
        "posterior_note": "Use checkpoint points/log_l with shell weights; posterior arrays have not been resampled.",
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary), flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--snapshot", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--n-live", type=int, default=5000)
    result.add_argument("--n-update", type=int, default=5000)
    result.add_argument("--n-like-new-bound", type=int, default=50000)
    result.add_argument("--n-eff", type=int, default=5000)
    result.add_argument("--n-networks", type=int, default=16)
    result.add_argument("--explore-batch", type=int, default=64)
    result.add_argument("--sample-batch", type=int, default=512)
    result.add_argument("--enlarge-per-dim", type=float, default=1.1)
    result.add_argument("--split-threshold", type=float, default=100)
    result.add_argument("--f-live", type=float, default=.01)
    result.add_argument("--n-shell", type=int, default=1)
    result.add_argument("--seed", type=int, default=94)
    result.add_argument("--backend", choices=("jax", "numpy", "stock"), default="jax")
    result.add_argument("--likelihood", choices=("batch", "scalar"), default="batch")
    result.add_argument("--likelihood-batch-size", type=int, default=64)
    result.add_argument("--streaming-statistics", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--seconds", type=float, default=1500)
    result.add_argument("--max-calls", type=int, default=12000000)
    result.add_argument("--checkpoint-seconds", type=float, default=120)
    result.add_argument("--checkpoint-calls", type=int, default=100000)
    result.add_argument("--progress-seconds", type=float, default=60)
    result.add_argument("--blas-threads", type=int, default=1)
    result.add_argument("--check-only", action="store_true")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    for option in ("n_live", "n_update", "n_like_new_bound", "n_eff", "n_networks",
                   "explore_batch", "sample_batch", "n_shell", "max_calls", "blas_threads",
                   "likelihood_batch_size"):
        if getattr(arguments, option) < 1:
            raise SystemExit(option + " must be positive")
    for option in ("seconds", "checkpoint_seconds", "checkpoint_calls", "progress_seconds"):
        if getattr(arguments, option) <= 0:
            raise SystemExit(option + " must be positive")
    with output_lock(arguments.output.resolve()):
        main(arguments)
