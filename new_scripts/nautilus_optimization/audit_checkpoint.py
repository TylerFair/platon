"""Read-only, bounded-memory accounting of a Nautilus HDF5 checkpoint.

The default reads attributes and dataset shapes only. ``--verify-likelihoods``
also streams the log likelihoods (never the much larger point arrays) and
independently checks the recorded evidence and effective sample size. Do not
audit a checkpoint while another process is writing it.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
from scipy.special import logsumexp


def file_record(path, *, digest=False):
    path = Path(path).resolve()
    stat = path.stat()
    record = {"path": str(path), "size_bytes": stat.st_size,
              "mtime_ns": stat.st_mtime_ns}
    if digest:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        record["sha256"] = hasher.hexdigest()
    return record


def _native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode()
    return value


def _json_safe(value):
    """Emit strict JSON; unsupported estimates are null, never NaN/Infinity."""
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(val) for val in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _stream_likelihood_stats(dataset, start, chunk_rows):
    total, squared, finite, maximum = -np.inf, -np.inf, 0, -np.inf
    for offset in range(start, len(dataset), chunk_rows):
        values = dataset[offset:min(offset + chunk_rows, len(dataset))]
        if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
            raise ValueError(f"Invalid likelihood in {dataset.name}")
        finite += int(np.count_nonzero(np.isfinite(values)))
        total = np.logaddexp(total, logsumexp(values))
        squared = np.logaddexp(squared, logsumexp(2 * values))
        maximum = max(maximum, float(np.max(values)))
    return float(total), float(squared), finite, maximum


def audit_checkpoint(path, *, verify_likelihoods=False, chunk_rows=65536):
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    before = file_record(path)
    with h5py.File(path, "r") as checkpoint:
        sampler = checkpoint["sampler"]
        attrs = dict(sampler.attrs)
        counts = np.asarray(attrs["shell_n"], dtype=np.int64)
        volumes = np.asarray(attrs["shell_log_v"], dtype=float)
        means = np.asarray(attrs["shell_log_l"], dtype=float)
        shell_ess = np.asarray(attrs["shell_n_eff"], dtype=float)
        explored = bool(attrs["explored"])
        discard = bool(attrs.get("_discard_exploration", False))
        exp_end = np.asarray(attrs["shell_end_exp"], dtype=np.int64)
        z = float(logsumexp(volumes + means))
        masses = np.exp(volumes + means - z)
        inverse_ess = np.divide(masses ** 2, shell_ess,
                                out=np.zeros_like(masses), where=shell_ess > 0)
        n_eff = float(1 / inverse_ess.sum()) if inverse_ess.sum() > 0 else 0.0
        rows, streamed_z, streamed_inverse_terms = [], [], []
        for shell, count in enumerate(counts):
            dataset = sampler[f"log_l_{shell}"]
            start = int(exp_end[shell]) if discard else 0
            row = {
                "shell": shell,
                "stored_calls": len(dataset),
                "exploration_membership": int(exp_end[shell]),
                "estimator_calls": int(count),
                "dataset_estimator_rows": len(dataset) - start,
                "log_volume": float(volumes[shell]),
                "log_likelihood_threshold": float(attrs["shell_log_l_min"][shell]),
                "log_evidence": float(volumes[shell] + means[shell]),
                "posterior_mass": float(masses[shell]),
                "effective_samples": float(shell_ess[shell]),
                "inverse_ess_fraction": float(inverse_ess[shell] * n_eff),
                "likelihood_call_fraction": float(count / attrs["n_like"]),
                "bound_draws": int(attrs["shell_n_sample"][shell]),
            }
            if verify_likelihoods:
                if count != len(dataset) - start:
                    raise ValueError(f"Shell {shell}: estimator count disagrees with rows")
                total, squared, finite, maximum = _stream_likelihood_stats(
                    dataset, start, chunk_rows)
                shell_z = total + volumes[shell] - np.log(count) if count else -np.inf
                ess = np.exp(2 * total - squared) if finite else 0.0
                row["streamed"] = {
                    "finite_calls": finite, "max_log_likelihood": maximum,
                    "log_evidence": float(shell_z), "effective_samples": float(ess),
                    "log_evidence_delta": float(shell_z - row["log_evidence"]),
                    "relative_ess_delta": float(ess / shell_ess[shell] - 1)
                    if shell_ess[shell] else None,
                }
                streamed_z.append(shell_z)
                streamed_inverse_terms.append(
                    squared + 2 * (volumes[shell] - np.log(count)) if count else -np.inf)
            rows.append(row)

        stored_calls = sum(row["stored_calls"] for row in rows)
        transfer_count = len(sampler["shell_t"]) if "shell_t" in sampler else 0
        unused = 0
        if "shell_t" in sampler:
            for offset in range(0, transfer_count, chunk_rows):
                unused += int(np.count_nonzero(sampler["shell_t"][offset:offset + chunk_rows] >= 0))
        report = {
            "checkpoint": before,
            "checkpoint_content_hashed": False,
            "read_mode": "streamed_likelihoods" if verify_likelihoods else "metadata_and_shapes",
            "settings": {key: _native(value) for key, value in attrs.items()
                         if np.ndim(value) == 0 and not key.startswith("rng_")},
            "summary": {
                "likelihood_calls": int(attrs["n_like"]),
                "stored_calls": stored_calls,
                "unaccounted_calls": int(attrs["n_like"]) - stored_calls - unused,
                "exploration_membership": sum(row["exploration_membership"] for row in rows),
                "estimator_calls": int(counts.sum()),
                "transfer_candidates": transfer_count,
                "unused_transfers": unused,
                "shells": len(counts),
                "log_evidence": z, "effective_samples": n_eff,
                "calls_per_effective_sample": float(attrs["n_like"] / n_eff) if n_eff else None,
                "explored": explored,
            },
            "shells": rows,
            "caveats": [
                "ESS and evidence do not by themselves establish posterior accuracy.",
                "Exploration memberships are final shell assignments, not chronological evaluations.",
                "The checkpoint is identified by path, size, and mtime, not a full-file hash.",
            ],
        }
        if verify_likelihoods:
            empirical_z = float(logsumexp(streamed_z))
            empirical_ess = float(np.exp(2 * empirical_z - logsumexp(streamed_inverse_terms)))
            report["streamed_summary"] = {
                "log_evidence": empirical_z, "effective_samples": empirical_ess,
                "log_evidence_delta": empirical_z - z,
                "relative_ess_delta": empirical_ess / n_eff - 1 if n_eff else None,
            }
    if file_record(path) != before:
        raise RuntimeError("Checkpoint changed while it was being read; audit a stable copy")
    return _json_safe(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--setup", type=Path, help="Saved setup.json, including historical priors")
    parser.add_argument("--source", type=Path, action="append", default=[],
                        help="Small provenance file to hash; repeat for scripts/data/logs")
    parser.add_argument("--verify-likelihoods", action="store_true")
    parser.add_argument("--chunk-rows", type=int, default=65536)
    parser.add_argument("--output", type=Path, help="Create a new JSON file; existing paths are refused")
    args = parser.parse_args()
    report = audit_checkpoint(args.checkpoint, verify_likelihoods=args.verify_likelihoods,
                              chunk_rows=args.chunk_rows)
    if args.setup:
        report["setup_file"] = file_record(args.setup, digest=True)
        report["setup"] = json.loads(args.setup.read_text())
    report["source_files"] = [file_record(path, digest=True) for path in args.source]
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
