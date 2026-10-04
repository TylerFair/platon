"""Validate offsets in ppm and apply them to dimensionless model depths."""

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np


def validate_offset_value(value):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)) or not np.isfinite(value):
        raise ValueError("Offset value must be a finite real scalar")


def normalize_windows(windows):
    """Return disjoint half-open index windows for one offset parameter."""
    if not isinstance(windows, (tuple, list)) or not windows:
        raise ValueError("Offset windows must be (start, end) or a list of pairs")
    if len(windows) == 2 and all(isinstance(x, Integral) for x in windows):
        windows = [windows]
    result = []
    for bounds in windows:
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError("Each offset window must contain start and end")
        start, end = bounds
        if any(not isinstance(x, Integral) or isinstance(x, (bool, np.bool_)) for x in bounds):
            raise ValueError("Offset window indexes must be integers")
        if start < 0 or end <= start:
            raise ValueError("Offset windows require 0 <= start < end")
        result.append((int(start), int(end)))
    result.sort()
    if any(right[0] < left[1] for left, right in zip(result, result[1:])):
        raise ValueError("Windows for the same offset must not overlap")
    return tuple(result)


def normalize_offset_map(windows):
    if windows is None:
        return {}
    if not isinstance(windows, Mapping):
        raise ValueError("Offset windows must be a mapping from parameter names to windows")
    result = {}
    for name, bounds in windows.items():
        if not isinstance(name, str) or not name or name in {
                "offset_start", "offset_end", "transit_offset_windows", "eclipse_offset_windows"}:
            raise ValueError("Offset parameter names must be nonempty and cannot be window controls")
        result[name] = normalize_windows(bounds)
    return result


def apply_offsets(depths, params, spectrum):
    """Apply registered ppm offsets to a copy of dimensionless depths."""
    if spectrum not in {"transit", "eclipse"}:
        raise ValueError("spectrum must be transit or eclipse")
    result = np.array(depths, dtype=float, copy=True)
    if result.ndim != 1:
        raise ValueError("Binned depths must be a one-dimensional array")
    windows = params.get(f"{spectrum}_offset_windows") or {}
    for name, bounds in windows.items():
        if name not in params:
            raise ValueError(f"Missing value for offset parameter {name}")
        value = params[name]
        validate_offset_value(value)
        for start, end in bounds:
            if end > len(result):
                raise ValueError(f"Window for {name} exceeds the {spectrum} spectrum length")
            result[start:end] += value * 1e-6
    legacy_name = f"offset_{spectrum}"
    if legacy_name not in windows:
        value = params.get(legacy_name, 0)
        validate_offset_value(value)
        result[params.get("offset_start", 0):params.get("offset_end", len(result))] += value * 1e-6
    return result


def offset_parameter_names(fit_info):
    """Registered depth offsets, including the two legacy parameters."""
    names = {"offset_transit", "offset_eclipse"}
    for spectrum in ("transit", "eclipse"):
        param = fit_info.all_params.get(f"{spectrum}_offset_windows")
        if param is not None:
            names.update(param.best_guess or {})
    return names


def offset_labels(names, fit_info):
    """Label registered offsets in ppm without changing their values."""
    offsets = offset_parameter_names(fit_info)
    return [f"{name} (ppm)" if name in offsets else name for name in names]
