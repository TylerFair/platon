# Streaming Nautilus shell statistics

The opt-in adapter in `platon/_nautilus_performance.py` removes repeated
full-shell likelihood reductions during sampling. Existing proposal geometry,
likelihood functions, exploration and requested ESS remain unchanged.

```python
from platon._nautilus_performance import install_streaming_shell_statistics

sampler = Sampler(...)  # stock 1.0.6 or the pinned bd80c0a GPU fork
statistics = install_streaming_shell_statistics(sampler)
try:
    success = sampler.run(n_eff=5000, discard_exploration=True)
    print(statistics.diagnostics)
finally:
    statistics.close()
```

The sampler keeps compensated, maximum-scaled sums of likelihood weights and
squared weights. A sampling update reduces just the appended batch. Full
reductions rebuild the cache on resume, after explicit `update_shell_info`
calls, after discard-state changes, every 2,048 updates per shell, near an ESS
stopping boundary, and before `run` returns. If a final exact refresh reverses
a convergence decision, sampling continues within the original call/time
limits. An infinite ESS target remains suitable for capped pilot runs.

Floating-point reduction order changes. Allocation decisions at numerical
ties can therefore differ; bitwise identity is not a general guarantee.
Direct changes to existing likelihood rows require `statistics.refresh()` or
`sampler.update_shell_info(index)`. The adapter is instance-local; install it
again after reading an ordinary Nautilus HDF5 checkpoint.

## Measured scope

`streaming_statistics_benchmark.json` records an October 28 replay of the
original WASP-94b shell-39 likelihood values. Each step appends 512 values.
Point arrays intentionally have zero columns to isolate likelihood-array
append/statistics work, and the shell uses a unit-cube volume. Its displayed
log evidence is consequently a test summary, not the physical retrieval's
evidence. No physical likelihood or geometry calls are timed.

| Existing rows | Full reduction median | Streaming median |
| ---: | ---: | ---: |
| 100,000 | 0.988 ms | 0.076 ms |
| 1,000,000 | 18.638 ms | 0.080 ms |
| 4,000,000 | 99.283 ms | 0.077 ms |
| 12,000,000 | 295.187 ms | 0.078 ms |

These are five warm appends after the first spare-capacity allocation, using
the existing bd80c0a fork. At 12M rows that initial append cost 22.7 ms and a
full final cache/statistics refresh cost 436 ms. Periodic buffer reallocations
and refreshes still cost time. The final exact summaries matched baseline
bit for bit at every tested size. This is a measured component improvement,
not an end-to-end retrieval speedup or a reduction in likelihood calls.

The regression file `tests/test_nautilus_performance.py` covers extreme and
all-zero weights, large log-likelihood offsets, 4,096 rare-tail updates,
discard/replacement invalidation, six complete seeded trajectories, two
checkpoint resumes, installation cleanup, and capped infinite-target runs.
The 17 initial tests passed against bd80c0a; the additional long accumulation
test passed separately. Stock 1.0.6 passed the initial 16 plus the two added
tests separately. The long accumulation test matches full log sums to
absolute tolerance 1e-12; complete seeded trajectories in these tests match
the baseline exactly, including points, RNG state and final summaries.

The fork's already validated append/checkpoint/GPU changes are a separate
requirement for its full speed benefit. Stock Nautilus still copies growing
point and likelihood arrays even with this statistics adapter installed.
