"""Replay real WASP-94 shell likelihoods without geometry or PLATON calls.

Run against the pinned ``bd80c0a`` Nautilus fork to include its spare-capacity
append improvement.  The checkpoint is opened read-only, and only likelihood
arrays are read (under 100 MB for the default largest slice).  Point arrays
have zero columns to isolate append/statistics costs from physical geometry.
"""

import argparse
import inspect
import json
import platform
from pathlib import Path
from time import perf_counter

import h5py
import nautilus
import numpy as np
from threadpoolctl import threadpool_limits

from platon._nautilus_performance import install_streaming_shell_statistics


def make_sampler(values):
    kwargs = {}
    if "backend" in inspect.signature(nautilus.Sampler).parameters:
        kwargs["backend"] = "numpy"
    sampler = nautilus.Sampler(
        lambda x: x, lambda x: np.zeros(len(x)), n_dim=2,
        n_networks=0, vectorized=True, n_batch=512, seed=94, **kwargs)
    sampler.run(n_like_max=512)
    sampler.explored = True
    sampler.points = [np.empty((len(values), 0))]
    sampler.log_l = [values.copy()]
    sampler.shell_n_sample[:] = len(values)
    sampler.shell_n_sample_exp = np.array([0])
    sampler.shell_end_exp = np.array([0])
    sampler.discard_exploration = False
    return sampler


def main(args):
    rows = []
    for size in args.sizes:
        with h5py.File(args.checkpoint, "r") as stream:
            dataset = stream[f"sampler/log_l_{args.shell}"]
            required = size + args.batch * (args.repetitions + 1)
            if required > len(dataset):
                raise ValueError(f"shell has only {len(dataset)} rows; need {required}")
            values = dataset[:required]
        results = []
        for mode in ("full_reduction", "streaming"):
            sampler = make_sampler(values[:size])
            controller = (install_streaming_shell_statistics(sampler)
                          if mode == "streaming" else None)
            times = []
            for iteration in range(args.repetitions + 1):
                offset = size + iteration * args.batch
                tail = values[offset:offset + args.batch]
                sampler.sample_shell = lambda shell, tail=tail: (
                    np.empty((len(tail), 0)), len(tail))

                def evaluate(points, sampler=sampler, tail=tail):
                    sampler.n_like += len(points)
                    return tail, None

                sampler.evaluate_likelihood = evaluate
                started = perf_counter()
                sampler.add_samples(0)
                times.append(perf_counter() - started)
            estimate = (float(sampler.log_z), float(sampler.n_eff))
            started = perf_counter()
            if controller:
                controller.refresh()
            else:
                sampler.update_shell_info(0)
            final_refresh_seconds = perf_counter() - started
            exact = (float(sampler.log_z), float(sampler.n_eff))
            row = {
                "mode": mode, "existing_rows": size, "batch": args.batch,
                "first_append_seconds": times[0],
                "warm_append_seconds": times[1:],
                "warm_median_seconds": float(np.median(times[1:])),
                "final_full_refresh_seconds": final_refresh_seconds,
                "estimated_log_z": estimate[0], "estimated_n_eff": estimate[1],
                "exact_log_z": exact[0], "exact_n_eff": exact[1],
                "logz_roundoff": estimate[0] - exact[0],
                "neff_relative_roundoff": (estimate[1] - exact[1]) / exact[1],
                "diagnostics": controller.diagnostics if controller else None}
            rows.append(row)
            results.append(row)
            print(json.dumps(row), flush=True)
            if controller:
                controller.close()
            del sampler
        assert results[0]["exact_log_z"] == results[1]["exact_log_z"]
        assert results[0]["exact_n_eff"] == results[1]["exact_n_eff"]
        print(json.dumps({
            "existing_rows": size,
            "measured_warm_speedup": (results[0]["warm_median_seconds"] /
                                      results[1]["warm_median_seconds"])}))
    output = {
        "checkpoint": str(Path(args.checkpoint).resolve()), "shell": args.shell,
        "nautilus_source": nautilus.__file__, "node": platform.node(), "rows": rows,
        "scope": "likelihood-array append and statistics only; no geometry, "
                 "checkpoint writes, or physical likelihood evaluations"}
    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shell", type=int, default=39)
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[100_000, 1_000_000, 4_000_000, 12_000_000])
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.repetitions < 1 or args.batch < 1 or min(args.sizes) < 1:
        parser.error("sizes, batch and repetitions must be positive")
    with threadpool_limits(limits=1):
        main(args)
