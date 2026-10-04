Stellar contamination (spots and faculae)
*****************************************

Starspots and faculae that the planet does not cross make the disk-averaged
stellar spectrum differ from the spectrum of the chord the planet actually
transits.  This *transit light source effect* imprints stellar features on the
transmission spectrum, and it is often the largest systematic for planets
around M and K dwarfs.  PLATON models it with the PHOENIX NewEra stellar
spectra (Hauschildt et al. 2025), interpolated in effective temperature, surface
gravity, and metallicity.

Computing a contaminated spectrum
=================================

Pass the stellar parameters to ``compute_depths``::

  from platon.transit_depth_calculator import TransitDepthCalculator

  calculator = TransitDepthCalculator()
  wavelengths, depths, info = calculator.compute_depths(
      p, Rs, Mp, Rp,
      T_star=3400, logg_phot=4.9, feh=0.0,  # photosphere
      T_spot=3000, spot_cov_frac=0.05,      # unocculted spots
      T_fac=3600, fac_cov_frac=0.02)        # unocculted faculae (optional)

The first time stellar parameters are used, PLATON downloads the NewEra spectra
(about 300 MB) into its installation directory, next to the opacity data.  This
happens only once.  ``examples/stellar_contamination_example.py`` plots a clean
and a contaminated spectrum side by side.

PLATON assumes the planet transits the unspotted photosphere, so each transit
depth is multiplied by

.. math::

   \frac{F_{\rm phot}}{(1 - f_{\rm spot} - f_{\rm fac})\,F_{\rm phot}
   + f_{\rm spot}\,F_{\rm spot} + f_{\rm fac}\,F_{\rm fac}}

where *f* are the covering fractions and *F* the component spectra.  Cool
unocculted spots therefore make the planet look larger in the blue, and
faculae have the opposite effect.

The stellar parameters are:

===================  ===========================================================
Parameter            Meaning
===================  ===========================================================
``T_star``           Photosphere effective temperature (K)
``T_spot``           Spot temperature (K)
``spot_cov_frac``    Fraction of the stellar disk covered by unocculted spots
``T_fac``            Facula temperature (K)
``fac_cov_frac``     Fraction of the stellar disk covered by unocculted faculae
``logg_phot``        Photosphere surface gravity, log10(g / cm s\ :sup:`-2`).
                     Default 4.5
``logg_spot``        Spot surface gravity.  Defaults to ``logg_phot``
``logg_fac``         Facula surface gravity.  Defaults to ``logg_phot``
``feh``              Stellar metallicity [Fe/H] in dex, shared by all
                     components.  Default 0.  This is independent of the
                     planet's ``logZ``
===================  ===========================================================

Spot and facula temperatures may be hotter or cooler than the photosphere, and
the two covering fractions must sum to at most 1.  Without ``T_star``, no
stellar spectrum is used.  The same parameters work in
:class:`.EclipseDepthCalculator`, where they set the stellar spectrum.

Retrieving stellar contamination
================================

All of the parameters above can be fit.  Give them initial values in
``get_default_fit_info`` and add priors as usual::

  fit_info = retriever.get_default_fit_info(
      Rs, Mp, Rp, T, logZ=0, CO_ratio=0.53,
      T_star=3400, logg_phot=4.9, feh=0.0,
      T_spot=3000, spot_cov_frac=0.05,
      stellar_grid_only=True)

  fit_info.add_gaussian_fit_param("T_star", 100)
  fit_info.add_uniform_fit_param("T_spot", 2300, 3400)
  fit_info.add_uniform_fit_param("spot_cov_frac", 0, 0.3)

If you fit both spots and faculae, choose priors for which the two covering
fractions cannot sum to more than 1.  Stellar surface gravity and metallicity
can be fit as well, but are usually better constrained by a Gaussian prior from
the literature than by the transmission spectrum.

``stellar_grid_only=True`` makes temperatures outside the grid raise an error,
which the retriever treats as zero likelihood.  Without it, temperatures outside
2300--12000 K fall back to a blackbody (PLATON warns the first time this
happens), so a wide prior could switch between stellar models partway through
a retrieval.  We recommend ``stellar_grid_only=True`` for retrievals.

``examples/retrieve_multi_instrument.py`` retrieves spot parameters together
with an instrument offset (see :doc:`instrument_offsets`).

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
the same ``logg_phot`` and ``feh`` to the calculator.  To keep gravity and
metallicity free, give arrays instead: ``loggs=[4.5, 5.0]`` and
``fehs=[-0.5, 0.0, 0.5]``.

For native resolution, download the NewEra JWST grid in HDF5 format from the
`MSG grids page <https://user.astro.wisc.edu/~townsend/static.php?ref=msg-grids>`_,
install ``h5py`` (``pip install ".[stellar-build]"``), and use it as the
source::

  path = generate_stellar_grid("my_star_native.npz", source="sg-NewEra-JWST.h5",
                               logg=4.9, feh=0.0)

Citing NewEra
=============

If you use the NewEra spectra, please cite Hauschildt et al. 2025, A&A 698, A47
(`doi:10.1051/0004-6361/202554171 <https://doi.org/10.1051/0004-6361/202554171>`_).
The spectra are distributed under the
`CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_ license
(`doi:10.25592/uhhfdm.17935 <https://doi.org/10.25592/uhhfdm.17935>`_); the
version downloaded by PLATON has been binned to R=5000 and re-encoded, as
described in ``platon/stellar_data/README.md``.
