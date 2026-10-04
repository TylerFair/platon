Install
*******
We highly recommend installing PLATON in a conda environment, which typically comes with optimized BLAS libraries.  Once in the conda environment::
  
  git clone https://github.com/ideasrule/platon.git
  cd platon/
  pip install -e .

Doing "pip install -e ." instead of "pip install ." installs PLATON in-place instead of copying the source files somewhere else.  This way, you can modify the source code and have the changes reflected instantly.

PLATON is built on JAX and performs all heavy computation in single precision (FP32).  It uses the GPU by default, and falls back on the CPU if it can't find one.  If you have a Nvidia GPU, we highly recommend that you install the CUDA-enabled JAX, which will speed up PLATON many-fold::

  pip install -U "jax[cuda12]"

JAX will automatically detect the GPU and use it.

PLATON supports both dynesty and pymultinest for nested sampling.  dynesty is installed by default.  To install pymultinest::
  
  conda install -c conda-forge mpi4py pymultinest

Nautilus is also available as an optional sampler.  It is not installed with
PLATON's base dependencies.  To add it::

  pip install ".[nautilus]"

If PLATON is already installed, use ``pip install nautilus-sampler``.

After installing PLATON, run one of the examples so that the data files are automatically downloaded::

  cd examples/
  python transit_depth_example.py

The NewEra stellar spectra used for stellar contamination (about 300 MB) are
downloaded separately, the first time you pass stellar parameters such as
``T_star`` (see :doc:`stellar_contamination`).  If your compute nodes have no
internet access, download both sets of data once on a machine that does, such
as a login node::

  python -c "from platon.transit_depth_calculator import TransitDepthCalculator; TransitDepthCalculator()"
  python -c "from platon.stellar_grid import download_stellar_grid; download_stellar_grid()"

You can also run unit tests to make sure everything works::

  pip install pytest
  python -m pytest tests

The default data files (in platon/data) have a wavelength resolution of R=20k.
If you want higher resolution, you can download higher-resolution opacities from the `DACE opacity database <https://dace.unige.ch/opacityDatabase>`_ and interpolate to PLATON's temperature and pressure grid.
