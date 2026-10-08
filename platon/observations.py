"""Load transit or eclipse spectra from tables, one per instrument or visit.

:func:`load_spectra` reads CSV, ECSV, whitespace-separated, and IPAC tables
(including the Eureka!, exoTEDRF, and NASA Exoplanet Archive formats),
converts every dataset to metres and dimensionless depths, and returns the
row ranges that :func:`.CombinedRetriever.get_default_fit_info` needs for
per-dataset offsets (``transit_offsets``) and per-visit stellar
heterogeneities (``transit_visits``).  It refuses to guess when a table is
ambiguous, and names the option that resolves the ambiguity.
"""
from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import numpy as np

WAVELENGTH_UNITS = {"m": 1.0, "um": 1e-6, "nm": 1e-9, "angstrom": 1e-10}
DEPTH_UNITS = {"fraction": 1.0, "ppm": 1e-6, "percent": 1e-2, "rprs": None}

# Spellings of units in column headers and astropy unit strings
_UNIT_WORDS = {
    "um": "um", "µm": "um", "micron": "um", "microns": "um",
    "micrometer": "um", "micrometers": "um", "nm": "nm", "m": "m",
    "angstrom": "angstrom", "angstroms": "angstrom", "aa": "angstrom",
    "å": "angstrom", "ppm": "ppm", "%": "percent", "percent": "percent",
    "pct": "percent", "fraction": "fraction",
}

# Recognised column names (lower case, units removed) for each role
_ROLES = {
    "wavelength": {"wavelength", "wave", "wav", "lambda", "lam", "wl",
                   "wavelength_center", "wave_center", "central_wavelength",
                   "centralwavelng", "wavelength_mid", "wave_mid"},
    "wavelength_low": {"wavelength_low", "wave_low", "wavelength_min",
                       "wave_min", "lambda_low", "lambda_min", "wl_low",
                       "wl_min", "wavelength_lower", "wave_lower", "bin_low",
                       "wavelength_start", "wave_start", "wave_lo"},
    "wavelength_high": {"wavelength_high", "wave_high", "wavelength_max",
                        "wave_max", "lambda_high", "lambda_max", "wl_high",
                        "wl_max", "wavelength_upper", "wave_upper", "bin_high",
                        "wavelength_end", "wave_end", "wave_hi"},
    # A wavelength "error" is half the bin width (as written by exoTEDRF)
    "half_width": {"wave_err", "wavelength_err", "wave_error",
                   "wavelength_error", "half_width", "halfwidth",
                   "bin_half_width", "wave_half_width"},
    # Ambiguous: Eureka!'s bin_width is a half width, other tables' are full
    "width": {"bin_width", "width", "bandwidth", "wave_width",
              "wavelength_width", "delta_wavelength", "delta_wave", "dwave",
              "dlambda", "bin_size"},
    "depth": {"depth", "transit_depth", "eclipse_depth", "occultation_depth",
              "dppm", "rp^2", "rprs^2", "(rp/rs)^2", "rp2", "rprs2",
              "pl_trandep", "fp/fs", "fpfs", "fp_fs", "fp"},
    "radius_ratio": {"rprs", "rp/rs", "rp_rs", "rp/r*", "rp", "radius_ratio",
                     "pl_ratror", "ror"},
    "error": {"error", "err", "depth_err", "depth_error", "dppm_err", "sigma",
              "uncertainty", "unc", "depth_unc", "depth_uncertainty",
              "rprs_err", "rp_err", "fp_err"},
    "error_high": {"error_high", "err_high", "error_plus", "err_plus",
                   "error_upper", "err_upper", "upper_error", "errorpos",
                   "errpos", "depth_err_high", "pl_trandeperr1",
                   "pl_ratrorerr1"},
    "error_low": {"error_low", "err_low", "error_minus", "err_minus",
                  "error_lower", "err_lower", "lower_error", "errorneg",
                  "errneg", "depth_err_low", "pl_trandeperr2",
                  "pl_ratrorerr2"},
}
_ROLE_KEYS = tuple(_ROLES) + ("order",)
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_+\-]+$")


@dataclass(frozen=True)
class SpectrumData:
    """Spectra from several datasets, concatenated in the order given.

    Attributes
    ----------
    bins : ndarray, shape (N, 2)
        Wavelength bin edges in metres; pass as the bins of a retrieval.
    depths, errors : ndarray, shape (N,)
        Dimensionless depths and their one-sigma errors.
    dataset : ndarray of str, shape (N,)
        The dataset each row came from.
    ranges : dict
        Each dataset's (start, end) rows.
    offsets : dict
        Offset parameter names mapped to (start, end) rows, for
        get_default_fit_info's transit_offsets or eclipse_offsets.
    visits : dict
        Visit names mapped to (start, end) rows, for transit_visits; empty
        unless visits were given.
    files : dict
        Each dataset's file.
    """
    bins: np.ndarray
    depths: np.ndarray
    errors: np.ndarray
    dataset: np.ndarray
    ranges: dict
    offsets: dict
    visits: dict
    files: dict

    @property
    def wavelengths(self):
        """Bin centres in metres."""
        return self.bins.mean(axis=1)

    def __len__(self):
        return len(self.depths)

    def __str__(self):
        offset_of = {}
        for name, (start, end) in self.offsets.items():
            for dataset, (lo, hi) in self.ranges.items():
                if start <= lo and hi <= end:
                    offset_of[dataset] = name
        visit_of = {}
        for name, (start, end) in self.visits.items():
            for dataset, (lo, hi) in self.ranges.items():
                if start <= lo and hi <= end:
                    visit_of[dataset] = name
        lines = ["{:<14} {:>5} {:>17} {:>12} {:>11}  {:<18} {}".format(
            "dataset", "rows", "wavelength (um)", "depth (ppm)",
            "error (ppm)", "offset", "visit")]
        for dataset, (start, end) in self.ranges.items():
            rows = slice(start, end)
            lines.append("{:<14} {:>5} {:>8.3f}-{:<8.3f} {:>12.1f} {:>11.1f}  {:<18} {}".format(
                dataset, end - start, 1e6 * self.bins[rows].min(),
                1e6 * self.bins[rows].max(), 1e6 * np.median(self.depths[rows]),
                1e6 * np.median(self.errors[rows]),
                offset_of.get(dataset, "(reference)"),
                visit_of.get(dataset, "")))
        return "\n".join(lines)


def load_spectra(datasets, *, wavelength_unit=None, depth_unit=None,
                 width=None, columns=None, reference=None, offsets=None,
                 visits=None):
    """Load and check spectra from one table per dataset.

    Each table needs depths, their errors, and either bin edges or bin
    centres.  Rows keep their order, and datasets are concatenated in the
    order given, so each dataset occupies one contiguous range of rows.

    Parameters
    ----------
    datasets : dict or list
        Dataset names (e.g. "NIRISS", "NRS1") mapped to table files, or a
        list of files named by their stems.  A value may also be a dict with
        a "file" key and any of wavelength_unit, depth_unit, width, and
        columns, overriding the keyword arguments below for that dataset.
        Names may contain letters, digits, "_", "-", and "+".
    wavelength_unit : {"um", "nm", "m", "angstrom"}, optional
        Unit of the wavelength columns.  By default it is read from the
        table (e.g. "wavelength (um)", or a units row), else microns.
    depth_unit : {"fraction", "ppm", "percent", "rprs"}, optional
        Unit of the depth and error columns; "rprs" means the planet-star
        radius ratio, which is squared (errors are propagated).  By default
        it is read from the table (e.g. "depth_ppm", "dppm", "%"), else
        "fraction".  Radius-ratio columns (e.g. "rprs", "rp_value") are
        recognised by name.
    width : {"half", "full"}, optional
        Whether an ambiguous width column such as "bin_width" holds half or
        full bin widths.  By default this is inferred by comparing widths
        with the spacing of the bin centres, and an error is raised if that
        is inconclusive.  (Eureka!'s bin_width is a half width.)
    columns : dict, optional
        Column names (or 0-based positions, for tables without a header)
        for any of the roles "wavelength", "wavelength_low",
        "wavelength_high", "half_width", "width", "depth", "radius_ratio",
        "error", "error_low", "error_high", and "order", overriding the
        names recognised automatically.
    reference : str, optional
        The dataset whose depths define the absolute scale; every other
        dataset gets an offset parameter named "offset_<dataset>".
        Defaults to the first dataset.
    offsets : dict, optional
        Instead of one offset per non-reference dataset, offset parameter
        names mapped to the dataset (or list of adjacent datasets) each
        applies to, e.g. {"offset_G395H": ["NRS1", "NRS2"]}.  Datasets not
        listed have no offset.  Use {} for no offsets at all.
    visits : dict, optional
        Visit names mapped to the dataset (or list of adjacent datasets)
        observed in each visit, e.g. {"soss": "NIRISS", "g395h": ["NRS1",
        "NRS2"]}, for stellar heterogeneities that differ between visits.
        Every dataset must belong to one visit.

    Returns
    -------
    SpectrumData
        Print it for a summary table.  Pass data.bins, data.depths and
        data.errors to a retrieval, data.offsets as transit_offsets (or
        eclipse_offsets), and data.visits as transit_visits.

    Examples
    --------
    >>> data = load_spectra({"NIRISS": "soss.csv", "NRS1": "nrs1.csv",
    ...                      "NRS2": "nrs2.csv"}, depth_unit="ppm")
    >>> print(data)
    >>> fit_info = retriever.get_default_fit_info(
    ...     Rs, Mp, Rp, T=1000, transit_offsets=data.offsets)
    >>> for name in data.offsets:
    ...     fit_info.add_uniform_fit_param(name, -500e-6, 500e-6)
    """
    defaults = dict(wavelength_unit=wavelength_unit, depth_unit=depth_unit,
                    width=width, columns=columns)
    specs = _dataset_specs(datasets, defaults)

    bins, depths, errors, names, ranges = [], [], [], [], {}
    start = 0
    for name, spec in specs.items():
        b, d, e = _load_dataset(name, spec)
        bins.append(b)
        depths.append(d)
        errors.append(e)
        names.extend([name] * len(d))
        ranges[name] = (start, start + len(d))
        start += len(d)
    _check_duplicates(specs, bins, depths)

    if offsets is None:
        if reference is None:
            reference = next(iter(specs))
        if reference not in specs:
            raise ValueError("reference {!r} is not one of the datasets: {}".format(
                reference, ", ".join(specs)))
        offsets = {"offset_" + name: [name] for name in specs
                   if name != reference}
    elif reference is not None:
        raise ValueError("Give either reference or offsets, not both; with "
                         "offsets, datasets not listed have no offset")
    offset_ranges = _group_ranges(offsets, ranges, "offset")
    if offset_ranges and sum(len(_as_list(members))
                             for members in offsets.values()) == len(ranges):
        raise ValueError(
            "Every dataset has an offset, which is degenerate with the "
            "planet radius; leave one dataset (the reference) without one")
    visit_ranges = _group_ranges(visits or {}, ranges, "visit")
    if visits:
        assigned = [d for members in visits.values() for d in _as_list(members)]
        missing = [name for name in specs if name not in assigned]
        if missing:
            raise ValueError("Datasets {} are not in any visit; every dataset "
                             "needs one".format(", ".join(missing)))
        for visit in visits:
            if "." in visit or not _NAME_PATTERN.match(visit):
                raise ValueError("Visit name {!r} may only contain letters, "
                                 "digits, '_', '-', and '+'".format(visit))

    return SpectrumData(np.concatenate(bins), np.concatenate(depths),
                        np.concatenate(errors), np.array(names),
                        ranges, offset_ranges, visit_ranges,
                        {name: spec["file"] for name, spec in specs.items()})


def _as_list(members):
    return [members] if isinstance(members, str) else list(members)


def _dataset_specs(datasets, defaults):
    if isinstance(datasets, (str, Path)):
        datasets = [datasets]
    if not isinstance(datasets, dict):
        files = list(datasets)
        datasets = {Path(f).stem: f for f in files}
        if len(datasets) != len(files):
            raise ValueError("Files share a name; pass a dict naming each dataset")
    if not datasets:
        raise ValueError("No datasets given")
    specs = {}
    for name, value in datasets.items():
        if not isinstance(name, str) or not _NAME_PATTERN.match(name):
            raise ValueError(
                "Dataset name {!r} may only contain letters, digits, '_', "
                "'-', and '+', since it becomes part of parameter names such "
                "as offset_<dataset>".format(name))
        spec = dict(defaults)
        if isinstance(value, dict):
            unknown = set(value) - set(defaults) - {"file"}
            if unknown or "file" not in value:
                raise ValueError(
                    "Options for {} must include 'file' and may include {}; "
                    "got {}".format(name, ", ".join(defaults), sorted(value)))
            spec.update(value)
        else:
            spec["file"] = value
        spec["file"] = Path(spec["file"]).expanduser()
        if not spec["file"].is_file():
            raise FileNotFoundError("{}: no such file {}".format(name, spec["file"]))
        for option, allowed in (("wavelength_unit", WAVELENGTH_UNITS),
                                ("depth_unit", DEPTH_UNITS),
                                ("width", ("half", "full"))):
            if spec[option] is not None and spec[option] not in allowed:
                raise ValueError("{}: {}={!r} is not one of {}".format(
                    name, option, spec[option], ", ".join(allowed)))
        if spec["columns"] is not None:
            unknown = set(spec["columns"]) - set(_ROLE_KEYS)
            if unknown:
                raise ValueError("{}: unknown column roles {}; roles are {}".format(
                    name, sorted(unknown), ", ".join(_ROLE_KEYS)))
        specs[name] = spec
    return specs


def _header_unit(header):
    """Unit written in a column header, e.g. "depth (ppm)", "wave_um"."""
    text = header.strip().lower()
    match = re.search(r"[\(\[]\s*([^\)\]]+?)\s*[\)\]]$", text)
    if match:
        return _UNIT_WORDS.get(match.group(1))
    if text.startswith("dppm"):
        return "ppm"
    for word, unit in _UNIT_WORDS.items():
        if len(word) > 1 and (text.endswith("_" + word) or
                              text.endswith(" " + word)):
            return unit
    return None


def _base_name(header):
    """Header in lower case without units, e.g. "Depth (ppm)" -> "depth"."""
    text = re.sub(r"\s*[\(\[][^\)\]]*[\)\]]$", "", header.strip().lower())
    unit = _header_unit(header)
    if unit is not None and not text.startswith("dppm"):
        for word, value in _UNIT_WORDS.items():
            if value == unit and re.search(r"[_ ]" + re.escape(word) + "$", text):
                text = text[:-len(word) - 1]
                break
    return text.replace(" ", "_")


def _find_columns(name, table, columns):
    """Map each role to a column name, from explicit choices or by name."""
    headerless = all(re.fullmatch(r"col\d+", c) for c in table.colnames)
    found = {}
    for role, column in (columns or {}).items():
        if isinstance(column, (int, np.integer)):
            if not 0 <= column < len(table.colnames):
                raise ValueError("{}: column position {} for {} is outside the "
                                 "table's {} columns".format(
                                     name, column, role, len(table.colnames)))
            column = table.colnames[column]
        elif column not in table.colnames:
            raise ValueError("{}: no column {!r} (for {}); columns are {}".format(
                name, column, role, ", ".join(table.colnames)))
        found[role] = column
    if headerless and not columns:
        raise ValueError(
            "{}: the table has no header, so its columns must be named, e.g. "
            "columns={{'wavelength_low': 0, 'wavelength_high': 1, 'depth': 2, "
            "'error': 3}} (the order of PLATON's example files)".format(name))

    candidates = {role: [] for role in _ROLE_KEYS}
    for column in table.colnames:
        if column in found.values():
            continue
        base = _base_name(column)
        # Eureka! writes <quantity>_value/_errorneg/_errorpos, e.g. rp^2_value
        eureka = re.fullmatch(r"(.+)_(value|errorneg|errorpos)", base)
        if eureka:
            quantity, kind = eureka.groups()
            if kind == "value":
                role = "radius_ratio" if quantity in ("rp", "rprs") else "depth"
            else:
                role = "error_low" if kind == "errorneg" else "error_high"
            candidates[role].append(column)
            continue
        if base == "order":
            candidates["order"].append(column)
        for role, names in _ROLES.items():
            if base in names:
                candidates[role].append(column)
    for role, matches in candidates.items():
        if role in found or not matches:
            continue
        if len(matches) > 1 and role.startswith("error"):
            # Resolved once the depth column is known (see _pick_errors)
            found[role] = matches
            continue
        if len(matches) > 1:
            raise ValueError(
                "{}: columns {} could all be the {}; choose one with "
                "columns={{'{}': ...}}".format(name, matches, role, role))
        found[role] = matches[0]
    return found


def _pick_errors(name, roles, value_column):
    """Among several candidate error columns, keep the one named after the
    value column (e.g. PL_TRANDEPERR1 for PL_TRANDEP, not PL_RATRORERR1)."""
    stem = _base_name(value_column).replace("_value", "")
    for role in ("error", "error_low", "error_high"):
        matches = roles.get(role)
        if not isinstance(matches, list):
            continue
        related = [m for m in matches if _base_name(m).startswith(stem)]
        if len(related) != 1:
            raise ValueError(
                "{}: columns {} could all be the {}; choose one with "
                "columns={{'{}': ...}}".format(name, matches, role, role))
        roles[role] = related[0]


def _column(table, column):
    values = table[column]
    if hasattr(values, "mask") and np.any(values.mask):
        values = values.filled(np.nan)
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("Column {!r} is not numeric".format(column))


def _table_unit(table, column):
    unit = table[column].unit
    if unit is None:
        return _header_unit(column)
    return _UNIT_WORDS.get(str(unit).strip().lower())


def _resolve_unit(name, kind, given, found, fallback):
    if given is not None and found is not None and given != found and \
            not (kind == "depth" and given == "rprs"):
        raise ValueError(
            "{}: {}_unit={!r}, but the table says its values are in {!r}".format(
                name, kind, given, found))
    return given or found or fallback


def _load_dataset(name, spec):
    from astropy.io import ascii
    try:
        table = ascii.read(str(spec["file"]), guess=True)
    except Exception as error:
        raise ValueError("{}: could not read {} as a table ({})".format(
            name, spec["file"], error)) from error
    if len(table) == 0:
        raise ValueError("{}: {} has no rows".format(name, spec["file"]))
    roles = _find_columns(name, table, spec["columns"])

    # Depths, possibly as radius ratios
    depth_column = roles.get("depth")
    ratio = roles.get("radius_ratio")
    cross_check = None
    if depth_column is not None and ratio is not None:
        # Prefer depths; a radius ratio alongside them (as in NASA Exoplanet
        # Archive tables) only checks the depth unit
        if np.all(np.isnan(_column(table, depth_column))):
            depth_column = None
        else:
            cross_check = _column(table, ratio)**2
            ratio = None
    if depth_column is None and ratio is None:
        raise ValueError("{}: no depth column among {}; name it with "
                         "columns={{'depth': ...}}".format(name, ", ".join(table.colnames)))
    if depth_column is not None:
        values = _column(table, depth_column)
        unit = _resolve_unit(name, "depth", spec["depth_unit"],
                             _table_unit(table, depth_column), "fraction")
    else:
        values = _column(table, ratio)
        if spec["depth_unit"] not in (None, "rprs"):
            raise ValueError("{}: {} holds radius ratios, but depth_unit={!r}".format(
                name, ratio, spec["depth_unit"]))
        unit = "rprs"
    _pick_errors(name, roles, depth_column or ratio)
    errors = _errors(name, table, roles)
    for role in ("error", "error_low", "error_high"):
        error_unit = _table_unit(table, roles[role]) if role in roles else None
        if error_unit is not None and unit != "rprs" and error_unit != unit:
            raise ValueError("{}: depths are in {} but {} is in {}".format(
                name, unit, roles[role], error_unit))
    _check_depth_unit(name, values, unit)
    if unit == "rprs":
        errors = 2 * np.abs(values) * errors
        values = values**2
    else:
        values = values * DEPTH_UNITS[unit]
        errors = errors * DEPTH_UNITS[unit]
    if cross_check is not None and np.any(np.isfinite(cross_check)):
        mismatch = np.nanmedian(np.abs(cross_check / values - 1))
        if mismatch > 0.02:
            raise ValueError(
                "{}: depths in {} disagree with the squared radius ratios in {} "
                "by {:.0%}; check depth_unit".format(
                    name, unit, roles["radius_ratio"], mismatch))

    bins = _bins(name, table, roles, spec)
    good = np.isfinite(values) & np.isfinite(errors) & np.all(np.isfinite(bins), axis=1)
    if not np.all(good):
        raise ValueError("{}: rows {} have missing or non-finite values".format(
            name, np.flatnonzero(~good)[:10].tolist()))
    if np.any(errors <= 0):
        raise ValueError("{}: rows {} have errors <= 0".format(
            name, np.flatnonzero(errors <= 0)[:10].tolist()))
    return bins, values, errors


def _errors(name, table, roles):
    if "error" in roles:
        return np.abs(_column(table, roles["error"]))
    if "error_low" in roles and "error_high" in roles:
        # Gaussian likelihood: average asymmetric errors (some tables store
        # the lower error as a negative number)
        return 0.5 * (np.abs(_column(table, roles["error_low"])) +
                      np.abs(_column(table, roles["error_high"])))
    if "error_low" in roles or "error_high" in roles:
        raise ValueError("{}: found only one of the lower/upper error columns; "
                         "name both with columns={{'error_low': ..., "
                         "'error_high': ...}}, or one symmetric 'error'".format(name))
    raise ValueError("{}: no error column among {}; name it with "
                     "columns={{'error': ...}}".format(name, ", ".join(table.colnames)))


def _check_depth_unit(name, values, unit):
    median = np.nanmedian(np.abs(values))
    likely = None
    if unit == "fraction" and median > 1:
        likely = "ppm" if median > 100 else "percent"
    elif unit == "percent" and median > 100:
        likely = "ppm"
    elif unit == "ppm" and median < 1:
        likely = "fraction"
    elif unit == "rprs" and median > 1:
        likely = "ppm" if median > 100 else "percent"
    if likely is not None:
        raise ValueError("{}: a median depth of {:g} does not look like {}; "
                         "did you mean depth_unit={!r}?".format(name, median, unit, likely))


def _bins(name, table, roles, spec):
    """Bin edges in metres, from edges, centres and widths, or centres."""
    if ("wavelength_low" in roles) != ("wavelength_high" in roles):
        raise ValueError("{}: found only one bin-edge column; name both with "
                         "columns={{'wavelength_low': ..., 'wavelength_high': ...}}".format(name))
    first = roles.get("wavelength_low", roles.get("wavelength"))
    if first is None:
        raise ValueError("{}: no wavelength column among {}; name it with "
                         "columns={{'wavelength': ...}}".format(name, ", ".join(table.colnames)))
    unit = _resolve_unit(name, "wavelength", spec["wavelength_unit"],
                         _table_unit(table, first), "um")
    order = _column(table, roles["order"]) if "order" in roles else \
        np.zeros(len(table))

    if "wavelength_low" in roles:
        low = _column(table, roles["wavelength_low"])
        high = _column(table, roles["wavelength_high"])
        if np.any(high <= low):
            raise ValueError("{}: rows {} have upper bin edges <= lower edges".format(
                name, np.flatnonzero(high <= low)[:10].tolist()))
        bins = np.column_stack([low, high])
    else:
        centers = _column(table, roles["wavelength"])
        if "half_width" in roles and "width" in roles:
            raise ValueError("{}: both {} and {} could give bin widths; choose one "
                             "with columns=".format(name, roles["half_width"], roles["width"]))
        if "half_width" in roles or "width" in roles:
            column = roles.get("half_width", roles.get("width"))
            widths = np.abs(_column(table, column))
            declared = "half" if "half_width" in roles else spec["width"]
            kind = _width_kind(name, column, centers, widths, order, declared)
            half = widths if kind == "half" else widths / 2
            bins = np.column_stack([centers - half, centers + half])
        else:
            bins = _edges_from_centers(name, centers, order)
    _check_wavelength_unit(name, bins, unit)
    return bins * WAVELENGTH_UNITS[unit]


def _local_spacing(centers, order):
    """Distance from each bin centre to its neighbours (mean of both sides),
    within groups of equal order; NaN where a group has one bin."""
    spacing = np.full(len(centers), np.nan)
    for value in np.unique(order):
        rows = np.flatnonzero(order == value)
        if len(rows) < 2:
            continue
        rows = rows[np.argsort(centers[rows])]
        gaps = np.diff(centers[rows])
        left = np.r_[gaps[0], gaps]
        right = np.r_[gaps, gaps[-1]]
        spacing[rows] = 0.5 * (left + right)
    return spacing


def _width_kind(name, column, centers, widths, order, declared):
    """Whether widths are "half" or "full" bin widths, checked against the
    spacing of the bin centres (contiguous bins have full width = spacing)."""
    spacing = _local_spacing(centers, order)
    usable = np.isfinite(spacing) & (spacing > 0)
    ratio = np.median(widths[usable] / spacing[usable]) if np.sum(usable) >= 3 \
        else np.nan
    inferred = "full" if 0.75 <= ratio <= 1.33 else \
        "half" if 0.375 <= ratio <= 0.667 else None
    if declared is None:
        if inferred is None:
            raise ValueError(
                "{}: cannot tell whether {} holds half or full bin widths "
                "(median width / bin spacing = {:.2f}); pass width='half' or "
                "width='full'".format(name, column, ratio))
        return inferred
    if inferred is not None and inferred != declared:
        warnings.warn(
            "{}: {} was taken as {} bin widths, but compared with the bin "
            "spacing they look like {} widths (median ratio {:.2f}); check the "
            "width option".format(name, column, declared, inferred, ratio),
            UserWarning, stacklevel=4)
    return declared


def _edges_from_centers(name, centers, order):
    """Bin edges halfway between neighbouring centres, per order."""
    bins = np.empty((len(centers), 2))
    for value in np.unique(order):
        rows = np.flatnonzero(order == value)
        if len(rows) < 2:
            raise ValueError("{}: inferring bin edges from centres needs at least "
                             "two bins; give widths or edges".format(name))
        rows = rows[np.argsort(centers[rows])]
        c = centers[rows]
        gaps = np.diff(c)
        if np.any(gaps <= 0):
            raise ValueError("{}: repeated wavelengths; give bin widths or edges".format(name))
        # A gap much wider than its neighbours (e.g. between detectors) would
        # otherwise become one enormous bin
        neighbours = np.minimum(np.r_[gaps[1:], np.inf], np.r_[np.inf, gaps[:-1]])
        if len(gaps) > 1 and np.any(gaps > 3 * neighbours):
            where = np.flatnonzero(gaps > 3 * neighbours)[0]
            raise ValueError(
                "{}: the gap after {:g} is much wider than the bins next to it, "
                "so bin edges cannot be inferred from the centres; give bin "
                "widths or edges".format(name, c[where]))
        mid = 0.5 * (c[1:] + c[:-1])
        edges = np.r_[c[0] - gaps[0] / 2, mid, c[-1] + gaps[-1] / 2]
        bins[rows, 0] = edges[:-1]
        bins[rows, 1] = edges[1:]
    return bins


def _check_wavelength_unit(name, bins, unit):
    if np.any(bins <= 0):
        raise ValueError("{}: wavelengths must be positive".format(name))
    microns = np.median(bins) * WAVELENGTH_UNITS[unit] / 1e-6
    if not 0.1 <= microns <= 100:
        likely = [u for u, scale in WAVELENGTH_UNITS.items()
                  if 0.1 <= np.median(bins) * scale / 1e-6 <= 100]
        raise ValueError(
            "{}: in {} the median wavelength would be {:g} microns{}".format(
                name, unit, microns, "; did you mean {}?".format(" or ".join(
                    "wavelength_unit={!r}".format(u) for u in likely))
                if likely else ""))


def _check_duplicates(specs, bins, depths):
    names = list(specs)
    resolved = [spec["file"].resolve() for spec in specs.values()]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if resolved[i] == resolved[j]:
                raise ValueError("Datasets {} and {} are the same file {}".format(
                    names[i], names[j], resolved[i]))
            if len(depths[i]) == len(depths[j]) and \
                    np.array_equal(depths[i], depths[j]) and \
                    np.array_equal(bins[i], bins[j]):
                raise ValueError("Datasets {} and {} contain identical data".format(
                    names[i], names[j]))


def _group_ranges(groups, ranges, kind):
    """(start, end) rows of each group of adjacent datasets."""
    if not isinstance(groups, dict):
        raise ValueError("{}s must be a dict of names to datasets".format(kind))
    result, used = {}, {}
    order = list(ranges)
    for group, members in groups.items():
        members = _as_list(members)
        if not members:
            raise ValueError("{} {} has no datasets".format(kind, group))
        for member in members:
            if member not in ranges:
                raise ValueError("{} {}: {!r} is not one of the datasets: {}".format(
                    kind, group, member, ", ".join(ranges)))
            if member in used:
                raise ValueError("Dataset {} is in both {}s {} and {}".format(
                    member, kind, used[member], group))
            used[member] = group
        positions = sorted(order.index(m) for m in members)
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError(
                "Datasets in {} {} must be listed next to each other in "
                "datasets, so that their rows are contiguous".format(kind, group))
        result[group] = (ranges[order[positions[0]]][0],
                         ranges[order[positions[-1]]][1])
    return result
