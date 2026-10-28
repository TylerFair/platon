PyMultiNest and Nautilus
************************

PLATON supports several samplers through the same
:class:`.CombinedRetriever` interface.  PyMultiNest and Nautilus are both
optional: importing PLATON does not require either package.

Installing a sampler
====================

PyMultiNest wraps the compiled MultiNest library.  A conda installation is
usually the simplest::

  conda install -c conda-forge mpi4py pymultinest

Nautilus can be installed with PLATON's optional extra::

  pip install ".[nautilus]"

For an existing PLATON installation, the sampler package can also be installed
directly::

  pip install nautilus-sampler

The distribution is named ``nautilus-sampler`` even though it is imported in
Python as ``nautilus``.

Running the same retrieval
==========================

Once ``fit_info`` and the observations have been prepared, the two calls are
nearly identical::

  data = (
      transit_bins, transit_depths, transit_errors,
      None, None, None, fit_info)

  multinest_result = retriever.run_multinest(
      *data,
      nlive=2000)

  nautilus_result = retriever.run_nautilus(*data)

For PyMultiNest, ``maxiter`` sets the iteration limit (``0`` means
unlimited). ``maxcall`` is not supported and raises ``ValueError``;
use ``maxiter`` when a bounded run is needed.

The Nautilus call above uses PLATON's defaults of 5,000 live points, 10,000
effective posterior samples, and 16 neural networks.  Written out explicitly,
it is::

  nautilus_result = retriever.run_nautilus(
      *data,
      n_live=5000,
      n_eff=10000,
      n_networks=16)

The Nautilus documentation recommends at least 10,000 effective samples
(a log-evidence error of about 0.01) and more networks to reduce the number
of likelihood calls.  The live points set how well separate solutions are
found: with 1,000, a HAT-P-18 b retrieval whose posterior has two solutions
missed the narrower, higher-likelihood one, which 5,000 found.  When the
batched likelihood applies, Nautilus evaluates 256 points per batch unless
``n_batch`` is given.

PyMultiNest retains its historical default of 250 live points.  The comparison
uses 2,000 explicitly so that the live-point settings match.

Choosing between them
=====================

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * -
     - PyMultiNest
     - Nautilus
   * - Installation
     - Python wrapper and compiled MultiNest library
     - Optional ``nautilus-sampler`` package
   * - Live points
     - ``nlive``
     - ``n_live``
   * - Posterior target
     - Controlled mainly by the nested-sampling settings
     - ``n_eff`` sets the effective sample target
   * - PLATON posterior
     - Equal-weight samples from PyMultiNest
     - Equal-weight samples resampled from Nautilus's weighted posterior
   * - Extra options
     - ``multinest_kwargs``
     - Additional keywords go to ``nautilus.Sampler``

Nautilus defaults to ``discard_exploration=True``.  This follows the
`Nautilus documentation
<https://nautilus-sampler.readthedocs.io/en/stable/api_full.html>`_, which
notes that discarding exploration points is required for fully unbiased
posterior and evidence estimates.

Both methods return a :class:`.RetrievalResult`, so the normal plotting calls
work for either result::

  from platon.plotter import Plotter

  Plotter.plot_retrieval_transit_spectrum(nautilus_result)
  Plotter.plot_retrieval_corner(nautilus_result)

For Nautilus, ``result.samples`` holds equal-weight rows resampled from its
weighted posterior (``n_eff`` of them when nuisance parameters are integrated
out, each with those parameters drawn from their posterior), followed by the
best fit.  ``result.weights``, ``result.logw`` and ``result.logl`` are
Nautilus's own normalized posterior weights and log-likelihoods.

Nautilus also provides reliable Bayesian evidence estimates and more honest
posterior widths.  In a PLATON benchmark, `Savel et al. (2026)
<https://arxiv.org/abs/2607.18409>`_ found that MultiNest underestimated
posterior widths by about 10% and that its standard evidence was less accurate
and precise than Nautilus under the tested settings.

Offsets and error excess
========================

Instrument offsets and an excess error are nuisance parameters: every
sampler would otherwise spend dimensions on them.  Each offset shifts its
range of data by a constant, so with Gaussian errors its integral over the
prior is closed form.  ``error_excess`` adds in quadrature to every transit
and eclipse depth error, ``sqrt(error**2 + error_excess**2)``; errors are only
ever inflated this way, never by a factor.  With a uniform prior its integral
is one-dimensional, and PLATON evaluates it on a fine grid over the prior,
accurate to well below 0.01 in ln(likelihood), with the offsets integrated in
closed form at each grid value.  So all four retrievers integrate every fitted
offset, and ``error_excess`` if fitted with a uniform prior, out of the
likelihood by default.  The evidence and the posterior of every other
parameter are unchanged; the sampler just has fewer dimensions.  Set
``marginalize_nuisance=False`` to sample them instead::

  fit_info = retriever.get_default_fit_info(
      ...,
      transit_offsets={"offset_nrs1": (n_soss, n_soss + n_nrs1)})
  fit_info.add_uniform_fit_param("offset_nrs1", -1e-3, 1e-3)
  fit_info.add_uniform_fit_param("error_excess", 0, 1e-3)

Time shifts have no excess error.

The returned samples still include the integrated-out parameters: after
sampling, each one is drawn from its exact posterior given the other
parameters of its sample, so corner plots, posterior spectra and the
leave-one-out scores are those of the full model.  The best fit sets them to
their most probable values given its other parameters.  For Nautilus, the
equal-weight samples are then ``n_eff`` rows of the resampled posterior, since
each row needs one more model evaluation.
