"""Flux-conserving integration of piecewise-constant stellar spectra."""
import numpy as np


def conservative_rebin(values, source_edges, target_edges):
    values = np.asarray(values, dtype=np.float64)
    source_edges = np.asarray(source_edges, dtype=np.float64)
    target_edges = np.asarray(target_edges, dtype=np.float64)
    if (values.ndim != 1 or target_edges.ndim != 1 or len(target_edges) < 2 or
            source_edges.shape != (values.size + 1,) or
            np.any(~np.isfinite(source_edges)) or np.any(~np.isfinite(target_edges)) or
            np.any(np.diff(source_edges) <= 0) or np.any(np.diff(target_edges) <= 0) or
            target_edges[0] < source_edges[0] or target_edges[-1] > source_edges[-1]):
        raise ValueError('Invalid rebinning edges')
    integral = np.r_[0., np.cumsum(values * np.diff(source_edges))]
    return np.diff(np.interp(target_edges, source_edges, integral)) / np.diff(target_edges)
