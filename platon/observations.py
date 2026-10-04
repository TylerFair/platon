"""Load one CSV per observation dataset without losing its offset identity."""

import csv
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ._offsets import validate_offset_value


@dataclass
class SpectrumData:
    """Binned observations, preserving input dataset and row order.

    Parameters
    ----------
    wavelength_bins : ndarray, shape (N, 2)
        Lower and upper wavelength edges in metres.
    depths : ndarray, shape (N,)
        Dimensionless transit or eclipse depths.
    errors : ndarray, shape (N,)
        Positive one-sigma depth uncertainties, dimensionless.
    dataset_slices : dict
        Dataset names mapped to half-open (start, end) row indexes.
    offset_groups : dict
        Dataset names mapped to group names; groups share an offset in ppm.
    """

    wavelength_bins: np.ndarray
    depths: np.ndarray
    errors: np.ndarray
    dataset_slices: dict
    offset_groups: dict

    def offset_windows(self, reference=None):
        """Return windows for relative offsets shared by observation groups.

        Parameters
        ----------
        reference : str, optional
            Group fixed at zero offset; defaults to the first group.

        Returns
        -------
        dict
            Offset parameter names mapped to lists of half-open row windows.
        """
        groups = {}
        for dataset, bounds in self.dataset_slices.items():
            groups.setdefault(self.offset_groups[dataset], []).append(bounds)
        if reference is None:
            reference = next(iter(groups))
        if reference not in groups:
            raise ValueError(f"Unknown reference offset group {reference!r}")
        return {f"offset_{group}": bounds for group, bounds in groups.items()
                if group != reference}

    def add_offsets(self, fit_info, *, reference=None, spectrum="transit",
                    prior_half_width=500):
        """Register relative group offsets and uniform priors on fit_info.

        Parameters
        ----------
        fit_info : FitInfo
            Fit configuration to update.
        reference : str, optional
            Group fixed at zero offset; defaults to the first group.
        spectrum : {'transit', 'eclipse'}, optional
            Spectrum receiving the offsets (default 'transit').
        prior_half_width : float, optional
            Symmetric uniform prior half width in ppm (default 500).

        Returns
        -------
        FitInfo
            The updated fit configuration.
        """
        if spectrum not in {"transit", "eclipse"}:
            raise ValueError("spectrum must be transit or eclipse")
        validate_offset_value(prior_half_width)
        if prior_half_width <= 0:
            raise ValueError("prior_half_width must be positive")
        windows = self.offset_windows(reference)
        # Check all collisions before registering parameters.
        if any(name in fit_info.all_params for name in windows):
            raise ValueError("Offset parameters already exist; configure their windows and priors with FitInfo directly")
        for name, bounds in windows.items():
            fit_info.add_offset(name, bounds, spectrum=spectrum)
            fit_info.add_uniform_fit_param(name, -prior_half_width, prior_half_width)
        return fit_info


_COLUMN_DEFAULTS = {
    "wavelength": ("wavelength",),
    "depth": ("depth",),
    "error": ("error", "depth_error", "depth_err"),
    "wavelength_low": ("wavelength_low", "wavelength_min"),
    "wavelength_high": ("wavelength_high", "wavelength_max"),
    "wavelength_err": ("wavelength_err",),
    "bin_width": ("bin_width",),
}


def _validate_columns(columns):
    if not isinstance(columns, Mapping):
        raise ValueError("columns must map column roles to CSV headers")
    unknown = set(columns) - set(_COLUMN_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown column roles: {sorted(unknown)}")
    if any(not isinstance(header, str) or not header for header in columns.values()):
        raise ValueError("Column headers must be nonempty strings")


def _check_units(bins, depths, wavelength_unit, depth_unit, path):
    median_depth = np.median(np.abs(depths))
    likely_depth_unit = None
    if depth_unit == 'fraction' and median_depth > 1:
        likely_depth_unit = 'ppm' if median_depth > 100 else 'percent'
    elif depth_unit == 'ppm' and np.all(np.abs(depths) < 1):
        likely_depth_unit = 'fraction'
    if likely_depth_unit:
        raise ValueError(f"Depths in {path} look inconsistent with depth_unit={depth_unit!r}; "
                         f"try depth_unit={likely_depth_unit!r}")
    median_wave = np.median(bins)
    likely_wave_unit = None
    if wavelength_unit == 'um' and np.max(bins) > 1000:
        likely_wave_unit = 'nm'
    elif wavelength_unit == 'm' and median_wave > 1e-3:
        likely_wave_unit = 'nm' if median_wave > 100 else 'um'
    elif wavelength_unit == 'um' and 0 < median_wave < 1e-4:
        likely_wave_unit = 'm'
    elif wavelength_unit == 'nm' and 0 < median_wave < .01:
        likely_wave_unit = 'm' if median_wave < 1e-4 else 'um'
    if likely_wave_unit:
        raise ValueError(f"Wavelengths in {path} look inconsistent with wavelength_unit={wavelength_unit!r}; "
                         f"try wavelength_unit={likely_wave_unit!r}")


def _read_csv(path, columns):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise ValueError(f"No observation rows in {path}")
    resolved = {}
    for role, candidates in _COLUMN_DEFAULTS.items():
        if role in columns:
            if columns[role] not in names:
                raise ValueError(f"Missing column {columns[role]!r} in {path}")
            resolved[role] = columns[role]
        else:
            resolved[role] = next((name for name in candidates if name in names), None)
    for role in ("depth", "error"):
        if resolved[role] is None:
            raise ValueError(f"No recognized {role} column in {path}; supply columns mapping")
    data = {}
    for role, column in resolved.items():
        if column is not None:
            try:
                data[role] = np.array([float(row[column]) for row in rows])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Non-numeric values in column {column!r} of {path}") from error
    if ("wavelength_low" in data) != ("wavelength_high" in data):
        raise ValueError(f"Both wavelength_low and wavelength_high are required in {path}")
    if "wavelength_low" in data:
        bins = np.column_stack([data["wavelength_low"], data["wavelength_high"]])
    else:
        if "wavelength" not in data:
            raise ValueError(f"Provide wavelength centers or both bin edge columns in {path}")
        centers = data["wavelength"]
        if "wavelength_err" in data:
            half_widths = data["wavelength_err"]
        elif "bin_width" in data:
            half_widths = data["bin_width"] / 2
        else:
            if len(centers) < 2 or np.any(np.diff(centers) <= 0):
                raise ValueError(f"Inferring bin edges requires at least two increasing wavelengths in {path}")
            midpoints = (centers[:-1] + centers[1:]) / 2
            edges = np.concatenate([
                [centers[0] - (centers[1] - centers[0]) / 2],
                midpoints,
                [centers[-1] + (centers[-1] - centers[-2]) / 2]])
            bins = np.column_stack([edges[:-1], edges[1:]])
            return bins, data["depth"], data["error"]
        bins = np.column_stack([centers - half_widths, centers + half_widths])
    return bins, data["depth"], data["error"]


def load_spectrum_csvs(files, *, wavelength_unit="um", depth_unit="fraction",
                       columns=None, offset_groups=None, dataset_options=None):
    """Load CSV observations without sorting or dropping rows.

    Provide wavelength_low/wavelength_high edges, or wavelength centers
    with wavelength_err half widths or bin_width full widths. Centers alone
    use midpoint edges and must increase within each file.

    Parameters
    ----------
    files : mapping
        Ordered dataset names mapped to CSV paths.
    wavelength_unit : {'m', 'um', 'nm'}, optional
        Units of wavelengths and widths (default 'um').
    depth_unit : {'fraction', 'ppm', 'percent'}, optional
        Units of depths and errors (default 'fraction'). Obvious unit
        mismatches raise ValueError with a suggested setting.
    columns : mapping, optional
        Column roles mapped to custom CSV headers. Defaults recognize depth,
        error (also depth_error/depth_err), and the wavelength roles above.
    offset_groups : mapping, optional
        Dataset names mapped to shared offset groups; defaults to each
        dataset's name. The first group is the default reference.
    dataset_options : mapping, optional
        Dataset names mapped to wavelength_unit, depth_unit, or columns
        overrides. Column overrides merge with the common mapping.

    Returns
    -------
    SpectrumData
        Wavelength bins in metres, with dimensionless depths and errors.
    """
    if not isinstance(files, Mapping) or not files:
        raise ValueError("files must be a nonempty mapping of dataset names to paths")
    wave_scales = {"m": 1.0, "um": 1e-6, "nm": 1e-9}
    depth_scales = {"fraction": 1.0, "ppm": 1e-6, "percent": 1e-2}
    if offset_groups is not None and not isinstance(offset_groups, Mapping):
        raise ValueError("offset_groups must map dataset names to group names")
    groups = {} if offset_groups is None else dict(offset_groups)
    if set(groups) - set(files):
        raise ValueError("offset_groups contains unknown dataset names")
    if dataset_options is not None and not isinstance(dataset_options, Mapping):
        raise ValueError("dataset_options must map dataset names to options")
    options = {} if dataset_options is None else dict(dataset_options)
    if set(options) - set(files):
        raise ValueError("dataset_options contains unknown dataset names")
    common_columns = {} if columns is None else columns
    _validate_columns(common_columns)
    if wavelength_unit not in wave_scales or depth_unit not in depth_scales:
        raise ValueError("Unsupported wavelength or depth unit")
    configurations = {}
    for dataset in files:
        group = groups.setdefault(dataset, dataset)
        if not isinstance(dataset, str) or not dataset or not isinstance(group, str) or not group:
            raise ValueError("Dataset and offset group names must be nonempty strings")
        override = options.get(dataset, {})
        if not isinstance(override, Mapping) or set(override) - {"wavelength_unit", "depth_unit", "columns"}:
            raise ValueError(f"Invalid dataset_options for {dataset!r}")
        dataset_columns = dict(common_columns)
        if "columns" in override:
            _validate_columns(override["columns"])
            dataset_columns.update(override["columns"])
        wave_unit = override.get("wavelength_unit", wavelength_unit)
        dep_unit = override.get("depth_unit", depth_unit)
        if wave_unit not in wave_scales or dep_unit not in depth_scales:
            raise ValueError(f"Unsupported wavelength or depth unit for {dataset!r}")
        configurations[dataset] = (wave_unit, dep_unit, dataset_columns)
    all_bins, all_depths, all_errors = [], [], []
    dataset_slices = {}
    start = 0
    for dataset, path in files.items():
        wave_unit, dep_unit, dataset_columns = configurations[dataset]
        bins, depths, errors = _read_csv(path, dataset_columns)
        if not np.all(np.isfinite(bins)) or np.any(bins[:, 0] <= 0) or np.any(bins[:, 1] <= bins[:, 0]):
            raise ValueError(f"Wavelength bins in {path} must be finite, positive, and have increasing edges")
        if not np.all(np.isfinite(depths)) or not np.all(np.isfinite(errors)) or np.any(errors <= 0):
            raise ValueError(f"Depths must be finite and errors finite and positive in {path}")
        _check_units(bins, depths, wave_unit, dep_unit, path)
        bins *= wave_scales[wave_unit]
        depths *= depth_scales[dep_unit]
        errors *= depth_scales[dep_unit]
        dataset_slices[dataset] = (start, start + len(depths))
        start += len(depths)
        all_bins.append(bins)
        all_depths.append(depths)
        all_errors.append(errors)
    return SpectrumData(np.concatenate(all_bins), np.concatenate(all_depths),
                        np.concatenate(all_errors), dataset_slices, groups)
