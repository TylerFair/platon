Combining instruments: loading data and fitting offsets
*******************************************************

Spectra from different instruments, detectors, or visits are often offset
from each other in depth, because of systematics in the light-curve fits or
because the star changed between visits.  PLATON can fit one offset for each
dataset relative to a reference dataset, which sets the absolute depth scale.

Loading the data
================

Keep one table per dataset and load them together with
:func:`~platon.observations.load_spectra`.  With the synthetic NIRISS SOSS and
NIRSpec G395H spectra in ``examples/data/``::

  from platon.observations import load_spectra

  data = load_spectra({"NIRISS": "examples/data/niriss_soss_example.csv",
                       "G395H": "examples/data/nirspec_g395h_example.csv"},
                      depth_unit="ppm")
  print(data)

The names (``NIRISS``, ``G395H``) are yours to choose; they become parameter
names.  Printing ``data`` shows a summary that is worth a glance before every
retrieval::

  dataset         rows   wavelength (um)  depth (ppm) error (ppm)  offset             visit
  NIRISS           119    0.850-2.800         44791.0       150.0  (reference)
  G395H             55    2.870-5.180         45026.7       100.0  offset_G395H

``data.bins``, ``data.depths``, and ``data.errors`` hold all datasets in the
order given (bins in metres, depths and errors as fractions), so each dataset
is one contiguous range of rows.  ``data.offsets`` maps an offset parameter
for every dataset except the first (the reference) to its rows, in the form
expected by ``get_default_fit_info``.

Table formats
-------------

``load_spectra`` reads comma-separated, whitespace-separated, ECSV, and IPAC
tables, including the spectra written by
`Eureka! <https://github.com/kevin218/Eureka>`_ (Stage 6 tables),
`exoTEDRF <https://github.com/radicamc/exoTEDRF>`_ (Stage 4 spectra), and the
`NASA Exoplanet Archive <https://exoplanetarchive.ipac.caltech.edu/docs/atmospheres/atmospheres_columns.html>`_
atmospheric spectroscopy tables.  It recognises common column names, for
example:

=================  ============================================================
Role               Recognised names (not case sensitive)
=================  ============================================================
bin edges          ``wavelength_low``/``wavelength_high``, ``wave_low``/``wave_high``,
                   ``wavelength_min``/``wavelength_max``, ...
bin centres        ``wavelength``, ``wave``, ``lambda``, ``CENTRALWAVELNG``, ...
half widths        ``wave_err``, ``wavelength_err``, ``half_width``
widths             ``bin_width``, ``width``, ``BANDWIDTH``, ... (see below)
depths             ``depth``, ``transit_depth``, ``eclipse_depth``, ``dppm``,
                   ``rp^2_value``, ``PL_TRANDEP``, ``fpfs``, ...
radius ratios      ``rprs``, ``rp/rs``, ``rp_value``, ``PL_RATROR``; squared,
                   with errors propagated
errors             ``error``, ``err``, ``depth_err``, ``dppm_err``, ``sigma``, ...;
                   or lower and upper errors (``errorneg``/``errorpos``,
                   ``PL_TRANDEPERR2``/``PL_TRANDEPERR1``, ...), which are
                   averaged
=================  ============================================================

If a column has another name, or the table has no header, say which column is
which, by name or by 0-based position::

  data = load_spectra({"HST": "wfc3.txt"},
                      columns={"wavelength_low": 0, "wavelength_high": 1,
                               "depth": 2, "error": 3})

Units are read from the table where it states them (``depth (ppm)``,
``wavelength_um``, ``dppm``, or a units row), and otherwise default to
microns and fractions.  Set them explicitly with ``wavelength_unit`` (``"um"``,
``"nm"``, ``"m"``, ``"angstrom"``) and ``depth_unit`` (``"fraction"``,
``"ppm"``, ``"percent"``, or ``"rprs"`` for a radius ratio).  Options can
differ between datasets::

  data = load_spectra({"NIRSpec": "nirspec.csv",
                       "WFC3": {"file": "wfc3.csv", "depth_unit": "percent"}},
                      depth_unit="ppm")

What ``load_spectra`` checks
----------------------------

Rather than guess, ``load_spectra`` raises an error that names the problem and
the option that fixes it.  It checks that:

* depths and wavelengths are plausible for their units (e.g. a median depth of
  21000 is not a fraction: did you mean ``depth_unit="ppm"``?), and that units
  stated in the table agree with the ones you give;
* depths agree with radius ratios when a table has both, as in Exoplanet
  Archive tables;
* every row has finite values and positive errors, and bin edges increase;
* a column such as ``bin_width`` is used correctly.  Some tables store half
  the bin width under this name (Eureka!) and others the full width, so PLATON
  compares the widths with the spacing of the bin centres.  If that is
  inconclusive, pass ``width="half"`` or ``width="full"``;
* bin edges are only inferred from centres when the bins are evenly spread,
  not across a gap between detectors;
* no two datasets are the same file or contain identical data;
* each column role matches at most one column.

Fitting the offsets
===================

Pass ``data.offsets`` to ``get_default_fit_info`` as ``transit_offsets`` (or
``eclipse_offsets`` for eclipse data), and give each offset a prior::

  fit_info = retriever.get_default_fit_info(
      Rs, Mp, Rp, T, transit_offsets=data.offsets)
  for name in data.offsets:
      fit_info.add_uniform_fit_param(name, -500e-6, 500e-6)

  result = retriever.run_dynesty(
      data.bins, data.depths, data.errors, None, None, None, fit_info)

Offsets are in units of depth, like the depths themselves: 500 ppm is
``500e-6``.  A positive offset means the observed depths of that dataset are
decreased before being compared with the model (equivalently, the model
depths are increased).

Before sampling, every retrieval checks the offsets against the data it is
given, and raises an error if an offset range runs past the end of the data,
if the bins, depths, and errors have different lengths, or if an offset (or
``error_excess``) prior is so wide that it was probably given in ppm.  It
warns if every data point has a freely fitted offset, which is degenerate with
the planet radius.

Choosing the reference and sharing offsets
------------------------------------------

The first dataset is the reference unless you choose another::

  data = load_spectra(files, reference="NRS1")

To give several datasets one offset, list them under one name; datasets that
are not listed have no offset.  For example, to fit one offset for both G395H
detectors relative to NIRISS::

  data = load_spectra(files, offsets={"offset_G395H": ["NRS1", "NRS2"]})

Datasets sharing an offset must be listed next to each other in ``files``, so
that their rows are contiguous.  ``offsets={}`` disables offsets.

Without ``load_spectra``
------------------------

The offsets are plain dictionaries of row ranges, so they can also be written
by hand for arrays you assembled yourself::

  fit_info = retriever.get_default_fit_info(
      Rs, Mp, Rp, T,
      transit_offsets={"offset_nrs1": (101, 122), "offset_nrs2": (122, 155)})

Visits
======

The same tables can be grouped into visits, for stellar heterogeneities that
differ between visits (see :doc:`stellar_contamination`)::

  data = load_spectra(files, visits={"soss": "NIRISS",
                                     "g395h": ["NRS1", "NRS2"]})
  fit_info = retriever.get_default_fit_info(
      Rs, Mp, Rp, T, T_star=3400, T_het=3000, f_het=0.05,
      transit_offsets=data.offsets, transit_visits=data.visits)

Every dataset must belong to one visit.

A complete example, retrieving a heterogeneity and the G395H offset from the
synthetic NIRISS SOSS and NIRSpec G395H spectra in ``examples/data/``, is in
``examples/retrieve_multi_instrument.py``.
