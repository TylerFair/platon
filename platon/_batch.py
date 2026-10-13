"""Drive the same calculator steps either individually or in populations.

Calculators yield host preparation and device calls.  The scalar driver calls
these directly; the population driver coalesces compatible calls without
changing validation, visit/sector blending, offsets, or likelihood arithmetic.
"""
from functools import wraps

import jax
import numpy as np

_STEPS = {}


class Call:
    def __init__(self, function, *args, **kwargs):
        self.function, self.args, self.kwargs = function, args, kwargs

    def run(self):
        return self.function(*self.args, **self.kwargs)


def scalar_steps(function):
    @wraps(function)
    def scalar(*args, **kwargs):
        steps = function(*args, **kwargs)
        value, failed = None, False
        while True:
            try:
                call = steps.throw(value) if failed else steps.send(value)
            except StopIteration as done:
                return done.value
            try:
                value, failed = call.run(), False
            except Exception as error:
                value, failed = error, True
    _STEPS[scalar] = function
    return scalar


def steps_of(method):
    """The shared steps behind a bound scalar_steps method, or None for
    user-defined overrides (including decorated ones)."""
    return _STEPS.get(getattr(method, "__func__", None))


def calculator_steps(method, *args, **kwargs):
    """Delegate to shared steps, preserving user-defined scalar overrides."""
    steps = steps_of(method)
    if steps is not None:
        return (yield from steps(method.__self__, *args, **kwargs))
    return method(*args, **kwargs)


class BatchEvaluator:
    """Fixed-size padded lax.map calls, with a cache per static config.

    A map (rather than vmap over the opacity tensor) bounds device workspace
    independently of population size and preserves the scalar FP32 core.
    """
    def __init__(self, chunk_size=64):
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.chunk_size = chunk_size
        self._cores = {}

    def _device(self, calls):
        groups = {}
        for i, call in enumerate(calls):
            cfg, data, inputs = call.args
            shapes = tuple(None if x is None else (np.shape(x), np.asarray(x).dtype.str)
                           for x in inputs)
            key = (call.function, cfg, id(data), shapes)
            groups.setdefault(key, []).append(i)
        results = [None] * len(calls)
        pending = []
        for (core, cfg, _, shapes), indices in groups.items():
            cache_key = (core, cfg, shapes)
            if cache_key not in self._cores:
                self._cores[cache_key] = jax.jit(
                    lambda data, inputs, core=core, cfg=cfg: jax.lax.map(
                        lambda row: core(cfg, data, row), inputs))
            fn = self._cores[cache_key]
            for start in range(0, len(indices), self.chunk_size):
                part = indices[start:start + self.chunk_size]
                padded = part + [part[0]] * (self.chunk_size - len(part))
                inputs = jax.tree_util.tree_map(
                    lambda *xs: np.stack(xs),
                    *(calls[i].args[2] for i in padded))
                pending.append((part, fn(calls[part[0]].args[1], inputs)))
        for indices, output in pending:
            output = jax.tree_util.tree_map(np.asarray, output)
            for row, i in enumerate(indices):
                results[i] = jax.tree_util.tree_map(lambda x: x[row], output)
        return results

    def evaluate(self, generators):
        from . import _forward_model as fm
        from ._forward_prep import prepare_forward_inputs, prepare_forward_inputs_batch
        from .TP_profile import Profile
        from .terminator import TwoSectorTerminator

        cores = (fm.transit_depths_core, fm.transit_depths_dual_core,
                 fm.eclipse_depths_core)
        results = [None] * len(generators)
        active = {i: (None, False) for i in range(len(generators))}
        while active:
            calls, indices = [], []
            for i, (value, failed) in active.items():
                try:
                    call = generators[i].throw(value) if failed else generators[i].send(value)
                except StopIteration as done:
                    results[i] = done.value
                    continue
                calls.append(call)
                indices.append(i)
            active = {}
            groups = {}
            for j, call in enumerate(calls):
                if call.function in cores:
                    key = 'device'
                elif call.function is prepare_forward_inputs:
                    key = ('prep', id(call.args[0]))
                elif call.function == Profile.from_params_dict:
                    key = ('profile', call.args[0], call.kwargs.get('suffix', ''))
                elif getattr(call.function, '__func__', None) is TwoSectorTerminator.from_params and \
                        type(call.function.__self__).sectors_from_params is \
                        TwoSectorTerminator.sectors_from_params:
                    key = ('terminator', id(call.function.__self__))
                else:
                    key = ('scalar', j)
                groups.setdefault(key, []).append(j)
            for key, members in groups.items():
                subset = [calls[j] for j in members]
                if key == 'device':
                    values = self._device(subset)
                elif key[0] == 'prep':
                    values = prepare_forward_inputs_batch(subset)
                elif key[0] == 'profile':
                    values = Profile.from_params_dict_batch(
                        key[1], [c.args[1] for c in subset], suffix=key[2])
                elif key[0] == 'terminator':
                    values = subset[0].function.__self__.from_params_batch(
                        [c.args[0] for c in subset])
                else:
                    values = []
                    for call in subset:
                        try:
                            values.append(call.run())
                        except Exception as error:
                            values.append(error)
                for j, value in zip(members, values):
                    active[indices[j]] = (value, isinstance(value, Exception))
        return results
