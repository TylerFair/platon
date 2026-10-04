Combining instruments: offsets
******************************

Spectra from different instruments, detectors, or visits are often offset from
each other by a constant in transit depth, because of differences in
systematics treatment, limb darkening, or orbital parameters.  PLATON can fit an
additive offset for each dataset.

.. note::

   Offsets are in **ppm**: a value of 500 adds 500 ppm (5e-4) to the model
   depths.  This also applies to ``offset_transit`` and ``offset_eclipse``,
   which were dimensionless in earlier versions of PLATON.  If you are reusing
   an older script, multiply their values and priors by 1e6.

Loading one CSV per instrument
==============================

The easiest way to combine datasets is to keep one CSV file per instrument
(or detector, or visit)::

  wavelength_low,wavelength_high,depth,error
  0.8500,0.8586,14823.1,150.0
  0.8586,0.8673,14790.4,150.0
  ...

Load them with :func:`~platon.observations.load_spectrum_csvs`, then let PLATON
add one offset per dataset::

  from platon.combined_retriever import CombinedRetriever
  from platon.observations import load_spectrum_csvs

  data = load_spectrum_csvs({"niriss": "niriss_soss.csv",
                             "g395h": "nirspec_g395h.csv"},
                            wavelength_unit="um", depth_unit="ppm")

  retriever = CombinedRetriever()
  fit_info = retriever.get_default_fit_info(Rs, Mp, Rp, T)
  fit_info.add_uniform_fit_param("Rp", 0.9 * Rp, 1.1 * Rp)
  fit_info.add_uniform_fit_param("T", 300, 1500)
  data.add_offsets(fit_info)

  result = retriever.run_dynesty(data.wavelength_bins, data.depths, data.errors,
                                 None, None, None, fit_info)

The first dataset (here ``niriss``) is the reference and stays fixed.  Every
other dataset gets a parameter named ``offset_<name>`` (here ``offset_g395h``)
with a uniform prior of +/-500 ppm.  To change the prior width or the reference
dataset, call ``add_offsets`` with ``prior_half_width`` or ``reference``
instead::

  data.add_offsets(fit_info, reference="g395h", prior_half_width=1000)

Each offset can only be registered once, so ``add_offsets`` and the
alternatives below should each be used on a new ``fit_info``.

Fixing one dataset avoids a degeneracy between a common offset and the planet
radius.  The retrieved offsets, like their priors, are in ppm.  A complete
example is in ``examples/retrieve_multi_instrument.py``, with synthetic data in
``examples/data/``.

``data.wavelength_bins`` are in metres and ``data.depths`` and ``data.errors``
are dimensionless, as everywhere else in PLATON.  Rows keep the order of the
files, and the files may overlap in wavelength.

CSV format
----------

Use ``wavelength_low`` and ``wavelength_high`` for the bin edges, or give bin
centres as ``wavelength`` together with either ``wavelength_err`` (half the bin
width) or ``bin_width`` (the full bin width).  With only ``wavelength``, the
edges are placed halfway between neighbouring centres.  Depths and their
uncertainties go in ``depth`` and ``error``.

``wavelength_unit`` can be ``"um"`` (default), ``"nm"``, or ``"m"``, and
``depth_unit`` can be ``"fraction"`` (default), ``"ppm"``, or ``"percent"``.  If
your headers are named differently, map them with ``columns``::

  data = load_spectrum_csvs({"visit1": "visit1.csv"},
                            columns={"wavelength": "wave", "depth": "rp2",
                                     "error": "rp2_err"})

Files that differ in units or headers can be combined with
``dataset_options``, which overrides the common settings for individual
files::

  data = load_spectrum_csvs(
      {"jwst": "jwst.csv", "hst": "hst.csv"},
      depth_unit="ppm",
      dataset_options={"hst": {"depth_unit": "percent",
                               "columns": {"wavelength": "wave_center",
                                           "bin_width": "wave_width"}}})

Sharing offsets and choosing priors
-----------------------------------

Datasets that should share one offset, such as the two orders of NIRISS SOSS,
can be grouped with ``offset_groups``.  Each group gets one offset, named
after the group::

  data = load_spectrum_csvs({"soss_order1": "order1.csv",
                             "soss_order2": "order2.csv",
                             "g395h": "g395h.csv"},
                            wavelength_unit="um", depth_unit="ppm",
                            offset_groups={"soss_order1": "soss",
                                           "soss_order2": "soss"})
  data.add_offsets(fit_info)  # fits offset_g395h relative to soss

For other priors, register the offsets yourself instead of calling
``add_offsets``.  ``offset_windows`` returns the rows covered by each offset::

  for name, rows in data.offset_windows().items():
      fit_info.add_offset(name, rows)
      fit_info.add_gaussian_fit_param(name, 100)  # 100 ppm prior

For eclipse spectra, use ``data.add_offsets(fit_info, spectrum="eclipse")`` and
pass the data as the eclipse arguments of the retriever.

Offsets without CSV files
=========================

If your data are already in arrays, register each offset with the rows it
applies to.  Row ranges are half-open, so ``(20, 40)`` means rows 20 to 39::

  fit_info.add_offset("offset_nirspec", (20, 40))
  fit_info.add_offset("offset_miri", (40, 55))
  fit_info.add_uniform_fit_param("offset_nirspec", -500, 500)
  fit_info.add_gaussian_fit_param("offset_miri", 200)

One offset can cover several separate ranges, such as ``[(0, 10), (30, 40)]``,
and offsets add where their ranges overlap, so a shared instrument offset can be
combined with a per-visit one.  Pass ``spectrum="eclipse"`` to ``add_offset``
for eclipse depths.

The original single offset, ``offset_transit``, still applies to the rows from
``offset_start`` to ``offset_end``.
