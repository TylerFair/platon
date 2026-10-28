# Nautilus integration and experiments

The maintained PLATON branch is `nautilus_best`. It starts from `time_shift`
0176720 and preserves the supplied uncommitted time-shift reporting updates.
The original `jax_1d` and `time_shift` worktrees remain unchanged. The matching
Nautilus dependency branch is also `nautilus_best`, initially at bd80c0a, which
already includes the validated GPU backend, spare-capacity appends, and
incremental checkpoint writes. No global Python environment is replaced.

## Accepted improvements

- Batched prior transforms preserve scalar formulas and ordered-pair handling
  while avoiding one Python callback per row. Custom priors keep their scalar
  transformation.
- Streaming shell moments replace repeated scans of old likelihoods with
  compensated sums over each appended batch. Exploration and explicit edits
  retain full reductions; resume initializes from the arrays, periodic checks
  refresh sums, and stopping/final output use full reductions. A floating-point
  allocation tie can change the subsequent trajectory. This does not change
  the physical likelihood, prior, bounds, or ESS target.

The statistics microbenchmark on real shell-39 likelihoods measured 295.19 ms
versus 0.078 ms at 12 million existing rows for a warm 512-row append. It
excludes physical likelihoods, geometric proposals, and checkpoint writes;
the full refresh costs about 0.44 seconds and is still required periodically.
This is not an end-to-end retrieval speedup measurement.

## Controlled live-point experiment

Freeze current source, observations and the optimized Nautilus dependency
with `freeze_sources.py`. Both runs use the same full-resolution, 18-dimensional
model with four offsets marginalized, ESS 5,000, 16 networks, `f_live=0.01`,
and batches 64 during exploration / 512 during sampling. Task 0 uses 5,000
live points and task 1 uses 20,000. Both keep `n_update=5000` and
`n_like_new_bound=50000` to isolate the live population. Optional task 2 tests
20,000 live points with these update intervals scaled to 20,000/200,000.

Record exploration and sampling calls separately, every native parity/warmup
evaluation, wall time, likelihood time, checkpoint time, ESS, evidence, and
shell weight concentration. Checkpoint identities reject changes to the
target or sampler settings. Time/call caps produce incomplete results, never
convergence claims. Start with one paired seed; follow promising candidates
with independent seeds and comparisons of evidence, parameter quantiles,
and shallow-quench/lower-metallicity region masses.

The historical 35,559,744-call run used a different target: 22 sampled
parameters, scattering slopes up to 20, and older temperature handling.
It is a useful production stress case, but its call count alone cannot
establish a speedup for the current model. See `BENCHMARK.md` and the
independently reconstructed `baseline_wasp94.json`.

## Supplied research leads

[Nautilus issue 70](https://github.com/johannesulf/nautilus/issues/70) describes
slow importance sampling after exploration. The maintainer confirms that
bounds are then fixed and recommends restarting with more live points when
coverage is poor. Two-stage runs with larger sampling batches are supported.
This directly motivates the 20,000-live experiment.

[Neural Transport Nested Sampling](https://arxiv.org/abs/2609.29413) uses a
learned flow-matching velocity as a local proposal drift, with a
Metropolis–Hastings correction. It needs neither target gradients nor flow
density evaluations. Its fixed-kernel invariant distribution is protected,
but finite mutation-chain mixing still needs numerical validation, and the
paper reports substantial training cost. This is a research candidate,
not a drop-in replacement for Nautilus's uniform shell draws and volume
accounting. A possible bridge is exploration-only rejuvenation to learn
bounds, followed by fresh independent shell sampling with all training
draws discarded. Such changes remain separate until they pass inference
and wall-time checks on this target.
