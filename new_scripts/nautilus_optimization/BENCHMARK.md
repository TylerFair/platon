# WASP-94Ab Nautilus production benchmark

The historical benchmark is the completed two-sector Guillot/equilibrium run in
`/scratch/midway3/tfairnington/time_shift/wasp94_eq_guillot/results`.
Its checkpoint contains **35,559,744 likelihood evaluations**, with
**log evidence 1142.9780363272064** and **ESS 5000.129215695**. This is a
historical performance reference, not a noiseless posterior truth.

## Historical target and settings

| Item | Saved value |
|---|---|
| Sampled dimensions | 22, including four linear offsets |
| Live points / bound update | 5000 / 5000 |
| Networks | 16 |
| Enlargement per dimension / split threshold | 1.1 / 100 |
| Exploration / sampling batch | 64 / 512 |
| Exploration samples in final estimator | Discarded |
| Seed | 94 |
| Opacity downsampling | 1 (native grid) |
| Data bins | 336: SOSS order 1 (114), NRS1 (69), NRS2 (153) |
| Exploration / sampling calls | 1,287,680 / 34,272,064 |
| Sectors | Equal fractions; independently parameterized, temperature labelled |

The original driver installed GPU prediction, union-bound, MVEE, split,
training and exploration-stall hooks from its local `nautilus_accel` package.
It wrote full atomic checkpoints every 100,000 calls; it was already avoiding
stock Nautilus's every-batch whole-shell checkpoint rewrites. A comparison
against completely unthrottled stock I/O overstates savings for this run.

The old bounds contain rare high-weight points in large outer shells. Shells
39 and 42 consume 63.65% of all calls, hold 9.20% of posterior probability,
and contribute 66.08% of inverse ESS. The earlier investigation found
geometric exclusion from later bounds, not merely slow likelihood calls.
Removing those shells or stopping before their heavy weights are discovered
changes inference. Larger ESS can drop sharply as new heavy points appear.

## Provenance and changes since the run

- Checkpoint: `results/nautilus_checkpoint.hdf5` (6.3 GiB); open read-only.
- Configuration and original priors: `results/setup.json`.
- Success/ESS record: `results/run_summary.json`, written by later postprocessing.
- Original launch/source: `run.sbatch` and `run_retrieval.py` beside `results`.
- Data: `/scratch/midway3/tfairnington/time_shift/wasp94_delta_preview/`:
  `soss_o1.csv`, `g395h_nrs1.csv`, `g395h_nrs2.csv`, `whitelight.json`.
- Stellar/planet properties:
  `/project/ekempton/tfairnington/JWST/03_trexolists_extended_with_nasa_ps_filled.csv`.
- Python: `/home/tfairnington/miniconda3/envs/platon_env/bin/python`.
- Original PLATON import path:
  `/project/ekempton/tfairnington/platon_parity/platon-jax_1d-time_shift`.
- Original job logs: `logs/wasp94_eq_guillot_61172476.{out,err}` and
  `logs/wasp94_eq_guillot_61242792.{out,err}` beside the driver.

The currently available driver and worktree have changed since those jobs:

1. On October 27 the driver changed both scattering-slope priors from
   `[-2, 20]` to `[-2, 15]`. The saved `setup.json` retains the historical bounds.
2. The completed run edge-clamped out-of-grid temperatures throughout the
   atmosphere. The newer physical rule rejects out-of-grid values above the
   protected 10-bar/quench region and permits deeper clamping. Prior research
   retained 97.2865% of historical posterior mass under that rule and obtained
   log evidence 1142.9505268. That is a reweighted old run, not a fresh repeat.
3. Newer PLATON analytically marginalizes the four linear offsets, reducing
   sampled dimension to 18. With matched bounds and physics this preserves
   the intended marginal target, but changes its sampling geometry.
4. The `time_shift` worktree itself contains uncommitted changes. A branch
   name or commit hash alone does not identify its effective likelihood.

Therefore invoking today's original driver does not reproduce the old target.
The new experiment must freeze source, data and saved priors, and record which
temperature rule and nuisance integration are used. Compare baseline and
optimized samplers on the **same frozen current target**. Keep the old 35.56M
budget as a separately labelled historical reference.

Avoid combining `posterior_samples.npz` sample coordinates with checkpoint
weights: the historical export contains equal-weight coordinates while stored
Nautilus weights describe unequal-weight coordinates. Use the HDF5 checkpoint
for weighted reference analysis. Do not deserialize the giant result pickle
or load all point arrays on a login node.

## Recorded time and hardware

Slurm accounting queried on October 28 reported:

| Job | State | Allocated elapsed | GPU in stdout | CPU / RAM |
|---|---|---:|---|---|
| 61172476 | COMPLETED | 35:40:20 | Tesla V100 PCIe 16 GB | 4 / 40 GB |
| 61242792 | OUT_OF_MEMORY | 12:10:39 | Quadro RTX 6000 | 4 / 40 GB |

The second job's stderr confirms an OOM kill; its saved checkpoint and later
postprocessing still report the completed ESS. Their combined allocated
elapsed time is **47:50:59**, including non-sampling work and two different
GPUs. It is not a measured pure likelihood or sampler-overhead time. Sampling
progress logs report approximately 0.56--1.02 million draws/hour. Separate
postprocessing produced the final figures and success record.

## Recompute the audit

`audit_checkpoint.py` defaults to attributes and dataset shapes. Its optional
streaming check independently recomputes the evidence and ESS from saved log
likelihoods in bounded chunks; it never reads point coordinates. The output
records hashes of small inputs, but does not pretend a path/size/mtime tuple
is a cryptographic hash of the 6.3-GiB checkpoint. Existing output files are
refused, and checkpoint mode is always `r`.

```bash
PY=/home/tfairnington/miniconda3/envs/platon_env/bin/python
SRC=/scratch/midway3/tfairnington/time_shift/wasp94_eq_guillot
OPENBLAS_NUM_THREADS=1 "$PY" new_scripts/nautilus_optimization/audit_checkpoint.py \
  "$SRC/results/nautilus_checkpoint.hdf5" \
  --setup "$SRC/results/setup.json" \
  --verify-likelihoods --output /tmp/wasp94-audit-new.json
```

`baseline_wasp94.json` is the audit generated for this investigation, with
data and current-driver hashes. Current-driver hashes prove what was inspected
on October 28, not that this exact source produced the October 22 checkpoint.

## Repetition and acceptance protocol

`freeze_sources.py` creates a new read-only snapshot for the matched trials.
It copies Python sources, the current driver, observations, the matching
archive rows and PLATON's required `md5sum` file. It records SHA256 hashes,
Git revision/dirty state and the metadata of every linked opacity file. Its
`verify_snapshot()` function verifies that identity before a launch or resume.
The large external opacity arrays are not content hashed or copied; the
manifest explicitly records this limitation. The selected PLATON source must
already contain the validated performance helper before freezing.

```bash
PY=/home/tfairnington/miniconda3/envs/platon_env/bin/python
"$PY" new_scripts/nautilus_optimization/freeze_sources.py /path/to/new/snapshot \
  --platon-source new_scripts/nautilus_optimization/best
"$PY" new_scripts/nautilus_optimization/freeze_sources.py /path/to/new/snapshot --verify
```

Freeze each compared target before launching a new seed. The requested next
comparison is **5000 versus 20000 live points**, both on the same latest target,
with native opacity resolution and target ESS 5000. Freeze identical prior
bounds and physical likelihood code for both; use today's `[-2, 15]` scattering
slopes in both if the latest target is chosen. Explicitly record whether
`n_update` stays at 5000 or tracks `n_live`, since that also changes the bound
construction. Keep all other settings matched in the first comparison. Record all
likelihood calls, including pilot/training/failed/correction calls, and separate
cheap approximate evaluations if they are used. Give every seed its own output
directory and never resume the historical checkpoint with changed model code.

Timing records must separate initialization/JIT, physical likelihood time,
proposal/bound work, shell-statistic updates, checkpoint I/O and posterior
export, with hardware and library/source fingerprints. Report end-to-end wall
time and sampling-only time, because overlap and asynchronous GPU dispatch can
otherwise make component sums misleading.

An exploratory cap or high nominal ESS is not a completed validation. Before
promoting a call-reducing method, compare multiple independent completed seeds
against a matched baseline for evidence, weighted marginals and the shallow
quench region, and account for uncertainty in the heavy-tailed reference.
Accuracy thresholds should be set before looking at candidate outcomes.
Overhead-only changes should additionally pass paired trajectory/output tests,
including checkpoint/resume behavior and their numerical tolerance.

## Existing investigations to reuse

- `/scratch/midway3/tfairnington/speedup/runs/nautilus/REPORT.md`: shell
  accounting, exact-output append/checkpoint improvements, CPU/GPU profiles and
  analytic/proxy trials. Its measured-operation time savings are projections,
  not an accelerated physical production repeat.
- `/scratch/midway3/tfairnington/speedup/runs/calls/REPORT.md` and
  `EXPERIMENT_LOG.md`: physical tail diagnosis, transports, profiling,
  stratification and approximate-likelihood correction experiments. The report
  explicitly states that no replacement recipe passed all required gates.
- `/scratch/midway3/tfairnington/speedup/runs/calls/scripts/reference.py`:
  reconstruction of the changed-temperature reference from old samples.
- `/scratch/midway3/tfairnington/speedup/runs/calls/scripts/common.py`:
  loads original observations and fit, marginalizes offsets and builds a native
  calculator. Its source driver is mutable, so freeze it before reuse.
