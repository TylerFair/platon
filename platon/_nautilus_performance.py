"""Opt-in streaming statistics for append-only Nautilus sampling shells.

Nautilus normally reduces every likelihood in a shell after each new batch.
Large shells make this quadratic in the number of samples.  This adapter
reduces only the appended batch, retaining compensated, maximum-scaled sums
of weights and squared weights.  It does not change bounds, proposals,
likelihoods, exploration, or the requested effective sample size.

Floating-point reduction order changes, so statistical values agree to
roundoff rather than necessarily bit for bit.  Full reductions initialize a
resumed sampler, periodically refresh each shell, check close stopping
decisions, and finalize every run.  Install this separately in each process;
it is instance-local and does not alter the installed Nautilus package.
"""

import inspect
from dataclasses import dataclass
from functools import wraps
from time import perf_counter

import numpy as np


@dataclass
class _Moments:
    """Compensated sums of exp(log_l - maximum) and its square."""

    array: np.ndarray
    start: int
    maximum: float = -np.inf
    total: float = 0.0
    total_correction: float = 0.0
    squared: float = 0.0
    squared_correction: float = 0.0
    updates: int = 0

    @staticmethod
    def _add(total, correction, value):
        adjusted = value - correction
        result = total + adjusted
        return result, (result - total) - adjusted

    def append(self, values):
        if len(values) == 0:
            return
        maximum = max(self.maximum, float(np.max(values)))
        if maximum == -np.inf:
            return
        with np.errstate(over="ignore", under="ignore"):
            scale = np.exp(self.maximum - maximum)
            weights = np.exp(values - maximum)
        self.total *= scale
        self.total_correction *= scale
        self.squared *= scale * scale
        self.squared_correction *= scale * scale
        self.total, self.total_correction = self._add(
            self.total, self.total_correction,
            float(np.sum(weights, dtype=np.float64)))
        self.squared, self.squared_correction = self._add(
            self.squared, self.squared_correction,
            float(np.sum(weights * weights, dtype=np.float64)))
        self.maximum = maximum


class StreamingShellStatistics:
    """Controller returned by :func:`install_streaming_shell_statistics`.

    Call ``refresh()`` after changing old likelihood rows in place, or call
    Nautilus's ``update_shell_info(index)`` for that shell.  Directly changing
    old rows without either notification violates Nautilus's append-only
    sampling invariant.  ``close()`` refreshes the statistics and restores the
    original instance methods.  HDF5 checkpoints remain ordinary Nautilus
    checkpoints; reinstall the adapter after constructing a resumed sampler.
    """

    _method_names = ("add_samples", "update_shell_info", "run")

    def __init__(self, sampler, refresh_interval=2048):
        if (isinstance(refresh_interval, (bool, np.bool_)) or
                not isinstance(refresh_interval, (int, np.integer)) or
                refresh_interval < 1):
            raise ValueError("refresh_interval must be a positive integer")
        if hasattr(sampler, "_platon_streaming_statistics"):
            raise ValueError("streaming shell statistics are already installed")
        for name in self._method_names:
            if not callable(getattr(sampler, name, None)):
                raise TypeError("sampler must expose Nautilus's " + name)
        self.sampler = sampler
        self.refresh_interval = int(refresh_interval)
        self._original = {
            name: getattr(sampler, name) for name in self._method_names}
        self._instance_methods = {
            name: vars(sampler)[name] for name in self._method_names
            if name in vars(sampler)}
        self._cache = {}
        self._pending_append = None
        self._target = None
        self._closed = False
        self.diagnostics = {
            "full_updates": 0, "streaming_updates": 0, "rows_reduced": 0,
            "exact_refreshes": 0, "stopping_corrections": 0,
            "max_logz_refresh_delta": 0.0,
            "max_neff_refresh_relative_delta": 0.0}
        for name in self._method_names:
            method = getattr(self, "_" + name)
            # Preserve signatures for callers inspecting the bound method.
            def wrapper(*args, _method=method, **kwargs):
                return _method(*args, **kwargs)
            setattr(sampler, name, wraps(self._original[name])(wrapper))
        sampler._platon_streaming_statistics = self
        # A checkpoint can contain statistics from an earlier streaming run.
        # Rebuild from the likelihood arrays, never trust serialized caches.
        if sampler.explored:
            self.refresh()

    def _start(self, index):
        if self.sampler.explored and self.sampler._discard_exploration:
            return int(self.sampler.shell_end_exp[index])
        return 0

    def _rebuild(self, index):
        self._original["update_shell_info"](index)
        self.diagnostics["full_updates"] += 1
        array = self.sampler.log_l[index]
        start = self._start(index)
        values = array[start:]
        self.diagnostics["rows_reduced"] += len(values)
        self._cache.pop(index, None)
        if (not self.sampler.explored or array.dtype != np.float64 or
                np.any(np.isnan(values)) or np.any(np.isposinf(values))):
            return
        moments = _Moments(array, start)
        moments.append(values)
        self._cache[index] = moments

    def _update_shell_info(self, index):
        index = int(index) % len(self.sampler.log_l)
        before = self._pending_append
        moments = self._cache.get(index)
        array = self.sampler.log_l[index]
        start = self._start(index)
        if (not self.sampler.explored or before is None or
                before[0] != index or moments is None or
                moments.array is not before[1] or moments.start != start or
                len(array) < len(moments.array) or
                moments.updates >= self.refresh_interval or
                array.dtype != np.float64):
            self._rebuild(index)
            return
        tail = array[len(moments.array):]
        if np.any(np.isnan(tail)) or np.any(np.isposinf(tail)):
            self._rebuild(index)
            return
        moments.append(tail)
        moments.array = array
        moments.updates += 1
        self.diagnostics["streaming_updates"] += 1
        self.diagnostics["rows_reduced"] += len(tail)
        sampler = self.sampler
        size = len(array) - start
        sampler.shell_n[index] = size
        if size == 0:
            sampler.shell_log_v[index] = -np.inf
            sampler.shell_log_l[index] = np.nan
            sampler.shell_n_eff[index] = 0
            return
        proposed = sampler.shell_n_sample[index]
        if sampler._discard_exploration:
            proposed -= sampler.shell_n_sample_exp[index]
        sampler.shell_log_v[index] = (
            sampler.bounds[index].log_v + np.log(size / proposed))
        if moments.maximum == -np.inf:
            sampler.shell_log_l[index] = -np.inf
            sampler.shell_n_eff[index] = size
        else:
            log_sum = moments.maximum + np.log(moments.total)
            log_sum_squared = 2 * moments.maximum + np.log(moments.squared)
            sampler.shell_log_l[index] = log_sum - np.log(size)
            # Keep Nautilus's final formula, including its cancellation order.
            sampler.shell_n_eff[index] = np.exp(
                2 * log_sum - log_sum_squared)

    def _add_samples(self, shell, *args, **kwargs):
        if self._pending_append is not None:
            raise RuntimeError("concurrent updates to one sampler are unsupported")
        index = int(shell) % len(self.sampler.log_l)
        self._pending_append = (index, self.sampler.log_l[index])
        try:
            result = self._original["add_samples"](shell, *args, **kwargs)
        finally:
            self._pending_append = None
        # Check the requested stopping threshold with full reductions if tiny
        # summation errors could decide which side of it the sampler is on.
        if (self.sampler.explored and self._target is not None and
                np.isfinite(self._target) and
                abs(self.sampler.n_eff - self._target) <=
                1e-9 * max(1.0, abs(self._target))):
            self.refresh()
        return result

    def refresh(self):
        """Recompute all shell summaries exactly with Nautilus's reductions."""
        sampler = self.sampler
        previous = (sampler.log_z, sampler.n_eff)
        self._cache.clear()
        for index in range(len(sampler.log_l)):
            self._rebuild(index)
        self.diagnostics["exact_refreshes"] += 1
        current = (sampler.log_z, sampler.n_eff)
        if (all(value is not None for value in previous + current) and
                np.all(np.isfinite(previous + current))):
            self.diagnostics["max_logz_refresh_delta"] = max(
                self.diagnostics["max_logz_refresh_delta"],
                abs(previous[0] - current[0]))
            self.diagnostics["max_neff_refresh_relative_delta"] = max(
                self.diagnostics["max_neff_refresh_relative_delta"],
                abs(previous[1] - current[1]) / max(1.0, abs(current[1])))

    def _run(self, *args, **kwargs):
        # Hooks may wrap run as run(**kwargs); read the arguments from the first
        # run in the class hierarchy that declares them.
        for cls in type(self.sampler).__mro__:
            run = cls.__dict__.get("run")
            if run is not None and "n_eff" in inspect.signature(run).parameters:
                declared = inspect.signature(run).bind(self.sampler, *args, **kwargs)
                declared.apply_defaults()
                break
        else:
            raise TypeError("No Sampler.run declares n_eff")
        arguments = {k: v for k, v in declared.arguments.items() if k != "self"}
        target = declared.arguments["n_eff"]
        n_shell = declared.arguments["n_shell"]
        timeout = declared.arguments["timeout"]
        started = perf_counter()
        previous_target = self._target
        self._target = target
        try:
            while True:
                reported_success = self._original["run"](**arguments)
                self.refresh()
                success = (self.sampler.explored and
                           np.all(self.sampler.shell_n >= n_shell) and
                           self.sampler.n_eff >= target)
                if success or not reported_success:
                    return bool(success)
                remaining = timeout - (perf_counter() - started)
                if (remaining <= 0 or
                        self.sampler.n_like >= arguments["n_like_max"]):
                    return False
                self.diagnostics["stopping_corrections"] += 1
                arguments["timeout"] = remaining
        finally:
            self._target = previous_target

    def close(self):
        """Restore original methods after an exact final refresh."""
        if self._closed:
            return
        self.refresh()
        for name in self._method_names:
            if name in self._instance_methods:
                setattr(self.sampler, name, self._instance_methods[name])
            else:
                delattr(self.sampler, name)
        del self.sampler._platon_streaming_statistics
        self._cache.clear()
        self._closed = True


def install_streaming_shell_statistics(sampler, refresh_interval=2048):
    """Accelerate sampling statistics on one existing Nautilus sampler.

    Compatible with stock Nautilus 1.0.6 and the ``bd80c0a`` GPU/append/IO
    fork.  The fork additionally removes full-array copies on each append;
    this adapter only removes repeated statistical scans.  Sampling shell
    allocation can differ at floating-point ties, so validate inference
    quality across seeds before adopting a new production configuration.
    """
    return StreamingShellStatistics(sampler, refresh_interval=refresh_interval)
