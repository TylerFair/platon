Stellar contamination (unocculted heterogeneities)
**************************************************

Starspots and faculae that the planet does not cross make the disk-averaged
stellar spectrum differ from the spectrum of the chord the planet actually
transits.  This *transit light source effect* (Rackham, Apai & Giampapa 2018)
imprints stellar features on the transmission spectrum, and it is often the
largest systematic for planets around M and K dwarfs.  PLATON models it with
the PHOENIX NewEra stellar spectra (Hauschildt et al. 2025), interpolated in
effective temperature, surface gravity, and metallicity.

PLATON describes the star as a photosphere plus one or two *heterogeneities*.
A heterogeneity is any region of the unocculted disk with its own temperature:
it may be cooler than the photosphere (spots) or hotter (faculae), and PLATON
does not assume which.

Computing a contaminated spectrum
=================================

Pass the stellar parameters to ``compute_depths``::

  from platon.transit_depth_calculator import TransitDepthCalculator

  calculator = TransitDepthCalculator()
  wavelengths, depths, info = calculator.compute_depths(
      p, Rs, Mp, Rp,
      T_star=3400, logg_star=4.9, feh_star=0.0,  # photosphere
      T_het=3000, f_het=0.05,                    # a heterogeneity
      T_het2=3600, f_het2=0.02)                  # a second one (optional)

The first time stellar parameters are used, PLATON downloads the NewEra spectra
(about 300 MB) into its data directory (``platon/data/stellar_data``), next to
the opacity data.  This happens only once.
``examples/stellar_contamination_example.py`` plots a clean and a contaminated
spectrum side by side.

PLATON assumes the planet transits the photosphere, so each transit depth is
multiplied by

.. math::

   \frac{F_{\rm phot}}{(1 - f_{\rm het} - f_{\rm het2})\,F_{\rm phot}
   + f_{\rm het}\,F_{\rm het} + f_{\rm het2}\,F_{\rm het2}}

where *f* are the covering fractions and *F* the component spectra.  Cool
unocculted heterogeneities therefore make the planet look larger in the blue,
and hot ones have the opposite effect.

The stellar parameters are:

===================  ===========================================================
Parameter            Meaning
===================  ===========================================================
``T_star``           Photosphere effective temperature (K)
``T_het``            Temperature of the first heterogeneity (K)
``f_het``            Fraction of the stellar disk it covers (unocculted)
``T_het2``           Temperature of the second heterogeneity (K)
``f_het2``           Fraction of the stellar disk it covers (unocculted)
``logg_star``        Photosphere surface gravity, log10(g / cm s\ :sup:`-2`).
                     Default 4.5
``logg_het``         Gravity of the first heterogeneity.  Defaults to
                     ``logg_star``
``logg_het2``        Gravity of the second heterogeneity.  Defaults to
                     ``logg_star``
``feh_star``         Stellar metallicity [Fe/H] in dex, shared by all
                     components.  Default 0.  This is independent of the
                     planet's ``logZ``
===================  ===========================================================

The two covering fractions must sum to at most 1.  An unset heterogeneity
temperature equals the photosphere's, and an unset fraction is zero.  Without
``T_star``, no stellar spectrum is used.  The same parameters work in
:class:`.EclipseDepthCalculator`, where they set the stellar spectrum.
``T_spot`` and ``spot_cov_frac``, the names used by earlier versions of
PLATON, are still accepted for ``T_het`` and ``f_het``.

Retrieving stellar contamination
================================

All of the parameters above can be fit.  Give them initial values in
``get_default_fit_info`` and add priors as usual::

  fit_info = retriever.get_default_fit_info(
      Rs, Mp, Rp, T, logZ=0, CO_ratio=0.53,
      T_star=3400, logg_star=4.9, feh_star=0.0,
      T_het=3000, f_het=0.05,
      stellar_grid_only=True)

  fit_info.add_gaussian_fit_param("T_star", 100)
  fit_info.add_uniform_fit_param("T_het", 2300, 4400)
  fit_info.add_uniform_fit_param("f_het", 0, 0.3)

With one heterogeneity, a temperature prior on both sides of ``T_star`` lets
the data decide between spots and faculae.  With two, the labels are
interchangeable: if both have the same priors, the posterior has two
mirror-image modes.  Give them priors that do not overlap (e.g. ``T_het``
below ``T_star`` and ``T_het2`` above it), and choose fraction priors that
cannot sum to more than 1.  Stellar surface gravity and metallicity can be fit
as well, but are usually better constrained by a Gaussian prior from the
literature than by the transmission spectrum.

``stellar_grid_only=True`` makes temperatures outside the grid raise an error,
which the retriever treats as zero likelihood.  Without it, temperatures outside
2300--12000 K fall back to a blackbody (PLATON warns the first time this
happens), so a wide prior could switch between stellar models partway through
a retrieval.  We recommend ``stellar_grid_only=True`` for retrievals.

Different heterogeneities in different visits
=============================================

Spots and faculae evolve, and the star rotates, so transits observed on
different dates generally see different unocculted heterogeneities.  The
photosphere does not change.  PLATON lets ``T_het``, ``f_het``, ``T_het2``,
and ``f_het2`` differ between visits, while ``T_star``, ``logg_star``, and
``feh_star`` are always shared.

Give each dataset a visit when loading the data (see :doc:`instrument_offsets`),
and pass the visits to ``get_default_fit_info``::

  from platon.observations import load_spectra

  data = load_spectra({"NIRISS": "soss.csv", "NRS1": "nrs1.csv",
                       "NRS2": "nrs2.csv"},
                      visits={"soss": "NIRISS", "g395h": ["NRS1", "NRS2"]})
  fit_info = retriever.get_default_fit_info(
      Rs, Mp, Rp, T, T_star=3400, T_het=3000, f_het=0.05,
      transit_offsets=data.offsets, transit_visits=data.visits,
      stellar_grid_only=True)

This creates the parameters ``soss.T_het``, ``soss.f_het``, ``g395h.T_het``,
``g395h.f_het`` (and the same for ``T_het2`` and ``f_het2``).  They default
to None, which means "use the shared value", so you choose what varies by
choosing what to fit.  For example, a spot temperature shared by both visits
with a separate covering fraction for each::

  fit_info.add_uniform_fit_param("T_het", 2300, 3300)
  fit_info.add_uniform_fit_param("soss.f_het", 0, 0.3)
  fit_info.add_uniform_fit_param("g395h.f_het", 0, 0.3)

To let the temperature differ too, fit ``soss.T_het`` and ``g395h.T_het``
instead of ``T_het``.  PLATON raises an error before sampling if a fitted
parameter cannot affect the spectrum, for example a fitted ``f_het`` when
every visit fits its own ``<visit>.f_het``.  Eclipse depths always use the
shared values.

The atmosphere is computed once per likelihood evaluation; only the stellar
correction and the binning are repeated for each visit, so extra visits cost
little.  For forward models, ``compute_depths`` accepts the same flexibility
directly: once wavelength bins are set, ``T_het``, ``f_het``, ``T_het2``, and
``f_het2`` may be arrays with one value per bin.

``examples/retrieve_multi_instrument.py`` retrieves a heterogeneity together
with an instrument offset.

The NewEra grid
===============

PLATON uses the solar-alpha NewEra JWST grid: effective temperatures of
2300--12000 K, log g of 0--6, [Fe/H] from -4 to +0.5, and wavelengths of
0.6--28.5 um.  The spectra are averaged into flux-conserving bins with a
resolving power of about 5000, and each component's spectrum is linearly
interpolated in temperature, gravity, and metallicity.  Outside 0.6--28.5 um,
each component is extended with a blackbody tail scaled to match the grid at
its edge.  Gravities and metallicities outside the grid raise an error.

A few models are missing from the NewEra grid.  If a single temperature is
missing at the requested gravity and metallicity, PLATON interpolates between
the neighbouring temperatures.  Larger gaps raise an error, which retrievals
treat as zero likelihood.

Compared with native-resolution NewEra spectra, the binned grid reproduces the
contamination of a 1% transit (10% spots, 5% faculae) to better than 1 ppm in
R=100 bins and better than 5 ppm in R=300 bins.  At R=1000, errors reach tens of
ppm for the coolest stars.  If you fit high-resolution data of a very
heterogeneous star, check your results against native-resolution spectra (see
below).

**Computers without internet access.**  If your compute nodes are offline, run
this once on a machine that can reach the internet, such as a login node::

  from platon.stellar_grid import download_stellar_grid
  download_stellar_grid()

**Results from earlier versions.**  Earlier versions of PLATON used a
temperature-only stellar grid.  Because NewEra is now the default, spectra
computed with ``T_star`` can differ slightly from earlier results, even without
spots, because the stellar spectrum weights the binned depths.  To reproduce
earlier results, use ``TransitDepthCalculator(stellar_grid="phoenix")``.  The
same argument is accepted by :class:`.EclipseDepthCalculator` and by
``get_default_fit_info``.

Custom and higher-resolution grids
==================================

``stellar_grid`` also accepts the path of your own grid: an ``.npz`` file
containing ``temperatures``, ``loggs``, ``fehs``, ``wavelengths_m`` (in m), and
``spectra`` with shape ``(temperatures, loggs, fehs, wavelengths)`` in
W m\ :sup:`-2` m\ :sup:`-1`.  :func:`~platon.stellar_grid.generate_stellar_grid`
writes such a file from the NewEra grid, for example to keep a small
single-star grid alongside your results::

  from platon.stellar_grid import generate_stellar_grid

  path = generate_stellar_grid("my_star.npz", logg=4.9, feh=0.0)
  calculator = TransitDepthCalculator(stellar_grid=path)

A grid made with fixed ``logg`` and ``feh`` only supports those values, so pass
the same ``logg_star`` and ``feh_star`` to the calculator.  To keep gravity and
metallicity free, give arrays instead: ``loggs=[4.5, 5.0]`` and
``fehs=[-0.5, 0.0, 0.5]``.

For native resolution, download the NewEra JWST grid in HDF5 format from the
`MSG grids page <https://user.astro.wisc.edu/~townsend/static.php?ref=msg-grids>`_,
install ``h5py`` (``pip install ".[stellar-build]"``), and use it as the
source::

  path = generate_stellar_grid("my_star_native.npz", source="sg-NewEra-JWST.h5",
                               logg=4.9, feh=0.0)

Citing
======

For the transit light source effect, please cite Rackham, Apai & Giampapa
2018, ApJ 853, 122
(`doi:10.3847/1538-4357/aaa08c <https://doi.org/10.3847/1538-4357/aaa08c>`_).

If you use the NewEra spectra, please cite Hauschildt et al. 2025, A&A 698, A47
(`doi:10.1051/0004-6361/202554171 <https://doi.org/10.1051/0004-6361/202554171>`_).
The spectra are distributed under the
`CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_ license
(`doi:10.25592/uhhfdm.17935 <https://doi.org/10.25592/uhhfdm.17935>`_); the
version downloaded by PLATON has been binned to R=5000 and re-encoded, as
described in ``platon/data/stellar_data/README.md``.
