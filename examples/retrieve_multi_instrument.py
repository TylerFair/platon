import pickle
from pathlib import Path

import numpy as np
from dynesty.utils import quantile

from platon.combined_retriever import CombinedRetriever
from platon.observations import load_spectrum_csvs
from platon.constants import R_sun, R_jup, M_jup
from platon.plotter import Plotter

# Synthetic NIRISS SOSS and NIRSpec G395H spectra of a made-up warm Saturn
# around an M dwarf; see data/make_example_data.py for the true parameters.
# Each CSV has columns wavelength_low, wavelength_high, depth, error.
data_dir = Path(__file__).resolve().parent / "data"
data = load_spectrum_csvs(
    {"niriss": data_dir / "niriss_soss_example.csv",
     "g395h": data_dir / "nirspec_g395h_example.csv"},
    wavelength_unit="um", depth_unit="ppm")

#create a Retriever object and set best guess parameters (SI units).
#logg_phot is log10(g) in cgs and feh is the stellar [Fe/H]
retriever = CombinedRetriever()
fit_info = retriever.get_default_fit_info(
    Rs=0.40*R_sun, Mp=0.30*M_jup, Rp=0.80*R_jup, T=700,
    logZ=0, CO_ratio=0.53,
    T_star=3400, logg_phot=4.9, feh=0.0, T_spot=3000, spot_cov_frac=0.05,
    stellar_grid_only=True)

fit_info.add_uniform_fit_param("Rp", 0.75*R_jup, 0.85*R_jup)
fit_info.add_uniform_fit_param("T", 500, 900)
fit_info.add_uniform_fit_param("logZ", -1, 1)
fit_info.add_gaussian_fit_param("T_star", 100)
fit_info.add_uniform_fit_param("T_spot", 2600, 3200)
fit_info.add_uniform_fit_param("spot_cov_frac", 0, 0.3)

#Fit an offset for G395H relative to NIRISS, with a uniform prior of
#+/-500 ppm.  The offset is named offset_g395h and is in ppm
data.add_offsets(fit_info)

#Use Nested Sampling to do the fitting.  This takes about 75 minutes on a
#16-core CPU, and much less on a GPU
result = retriever.run_dynesty(
    data.wavelength_bins, data.depths, data.errors,
    None, None, None,
    fit_info,
    nlive=100, sample="rwalk", walks=20, rstate=np.random.default_rng(123))
with open("multi_instrument_result.pkl", "wb") as f:
    pickle.dump(result, f)

for name in ["offset_g395h", "T_spot", "spot_cov_frac"]:
    values = result.samples[:, fit_info.fit_param_names.index(name)]
    low, median, high = quantile(values, [0.16, 0.5, 0.84], weights=result.weights)
    print(f"{name}: {median:.4g} +{high - median:.3g} -{median - low:.3g}")

#Plot the spectrum and save it to multi_instrument_transit.png
plotter = Plotter()
plotter.plot_retrieval_transit_spectrum(result, prefix="multi_instrument")
plotter.plot_retrieval_corner(result, filename="multi_instrument_corner.png")
