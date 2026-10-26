PyMultiNest, Nautilus and i-nessai
**********************************

PLATON supports several samplers through the same
:class:`.CombinedRetriever` interface.  PyMultiNest, Nautilus and i-nessai
are all optional: importing PLATON does not require any of them.

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

The Nautilus call above uses PLATON's defaults of 2,000 live points, 10,000
effective posterior samples, and 16 neural networks.  Written out explicitly,
it is::

  nautilus_result = retriever.run_nautilus(
      *data,
      n_live=2000,
      n_eff=10000,
      n_networks=16)

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
     - Original samples with normalized weights
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

Nautilus keeps its weighted posterior in ``result.samples`` and
``result.weights``.  PLATON only resamples it when equal-weight draws are
needed for posterior spectra or temperature profiles.

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

Importance nested sampling with flows (i-nessai)
================================================

``run_inessai`` runs nessai's importance nested sampler, which fits a new
normalizing flow to the likelihood-constrained prior at every level, and
then draws an independent importance sample from the frozen mixture of all
its flows.  For degenerate two-limb retrievals it needs far fewer likelihood
calls than Nautilus.  For WASP-94Ab (18 sampled parameters), it agreed with a
5,000-live-point Nautilus run to 0.18 in ln(evidence) and to 0.2 posterior
standard deviations in the physical parameters, with 11 times fewer calls.
Install it with::

  pip install ".[inessai]"

which pins nessai 0.15.2 (PLATON speeds up its internals for that version)
and installs PyTorch.  The call takes the same data as the others::

  result = retriever.run_inessai(*data, output="inessai_output")

The defaults are 5,000 live points and RealNVP flows (6 blocks of width 64,
LU linear layers, a fresh flow every 8 levels, up to 300 epochs) whose base
distribution has a 10% independent Student-t component.  The final
importance sample has 1 to 3 million draws, sized from 8,192 planning draws
for an effective sample size of ``target_ess=5000``.  Two-sector Guillot
transit retrievals evaluate likelihoods in batches on the GPU, and on a GPU
the flow densities are also evaluated there; neither changes the result.
Extra keyword arguments override nessai's options.

The run checkpoints in ``output``.  On SIGTERM or SIGUSR1, or after
``max_seconds``, it stops at its next checkpoint and raises
``platon._inessai.Checkpointed``, which exits the program with status 75
unless caught; calling ``run_inessai`` again with the same arguments resumes
it, so a batch job can simply be resubmitted.  A run that has finished is
read back from ``output``.

The result holds the evidence ``logz`` and its error ``logz_err``, the
effective sample size ``ess``, the weighted final draws (``draws``,
``weights``, ``logw``, ``logl``, ``logp``), ``n_calls``, and, in
``samples``, ``ess`` equal-weight rows (at most ``target_ess``) followed by
the best fit.  Check ``pareto_k``, the Pareto shape of the importance
weights' tail: above 0.7 a few draws carry much of the weight, so the
evidence error is unreliable and the posterior's rare regions may be
under-sampled, even when ``logz`` and the posterior bulk are accurate.
``psis_ess`` is the effective sample size after Pareto smoothing.
