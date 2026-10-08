import pickle
from pathlib import Path

import numpy as np
from dynesty.utils import quantile

from platon.combined_retriever import CombinedRetriever
from platon.observations import load_spectra
from platon.constants import R_sun, R_jup, M_jup
from platon.plotter import Plotter

# Synthetic NIRISS SOSS and NIRSpec G395H spectra of a made-up warm Saturn
# around an M dwarf; see data/make_example_data.py for the true parameters.
# Each CSV has columns wavelength_low, wavelength_high, depth, error, with
# wavelengths in microns and depths in ppm.
data_dir = Path(__file__).resolve().parent / "data"
data = load_spectra({"NIRISS": data_dir / "niriss_soss_example.csv",
                     "G395H": data_dir / "nirspec_g395h_example.csv"},
                    depth_unit="ppm")
print(data)

# NIRISS is the reference, so data.offsets is {"offset_G395H": (rows)}.
# T_het and f_het describe an unocculted heterogeneity (here, spots);
# logg_star is log10(g) in cgs and feh_star is the stellar [Fe/H]
retriever = CombinedRetriever()
fit_info = retriever.get_default_fit_info(
    Rs=0.40 * R_sun, Mp=0.30 * M_jup, Rp=0.80 * R_jup, T=700,
    logZ=0, CO_ratio=0.53,
    T_star=3400, logg_star=4.9, feh_star=0.0, T_het=3000, f_het=0.05,
    transit_offsets=data.offsets,
    stellar_grid_only=True)

fit_info.add_uniform_fit_param("Rp", 0.75 * R_jup, 0.85 * R_jup)
fit_info.add_uniform_fit_param("T", 500, 900)
fit_info.add_uniform_fit_param("logZ", -1, 1)
fit_info.add_gaussian_fit_param("T_star", 100)
fit_info.add_uniform_fit_param("T_het", 2600, 3200)
fit_info.add_uniform_fit_param("f_het", 0, 0.3)

# Offsets are in units of depth: this prior is +/-500 ppm
for name in data.offsets:
    fit_info.add_uniform_fit_param(name, -500e-6, 500e-6)

# Use Nested Sampling to do the fitting.  This takes about 75 minutes on a
# 16-core CPU, and much less on a GPU
result = retriever.run_dynesty(
    data.bins, data.depths, data.errors,
    None, None, None,
    fit_info,
    nlive=100, sample="rwalk", walks=20, rstate=np.random.default_rng(123))
with open("multi_instrument_result.pkl", "wb") as f:
    pickle.dump(result, f)

for name in ["offset_G395H", "T_het", "f_het"]:
    values = result.samples[:, fit_info.fit_param_names.index(name)]
    low, median, high = quantile(values, [0.16, 0.5, 0.84], weights=result.weights)
    print(f"{name}: {median:.4g} +{high - median:.3g} -{median - low:.3g}")

Plotter.plot_retrieval_transit_spectrum(result, prefix="multi_instrument")
Plotter.plot_retrieval_corner(result, filename="multi_instrument_corner.png")
