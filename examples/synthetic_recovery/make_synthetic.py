"""Synthetic TOI-6894 b spectra: the retrieval's model at truth.json, on the
real data's bins, plus Gaussian noise with variance error^2 + error_excess^2.

    python make_synthetic.py   # writes F01_s0.csv ... F01_s3.csv
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "toi6894_smc"))
from retrieve_toi6894_smc import Calculator, get_fit_info, load  # noqa: E402
from platon.combined_retriever import CombinedRetriever  # noqa: E402

bins, depths, errors = load("../toi6894_smc/TOI-6894_qiao_power2_0p8_5p2.csv")
fit_info = get_fit_info()
truth = np.array(json.load(open("truth.json"))["values"])
calc = Calculator(**fit_info._stellar_grid_options())
calc.change_wavelength_bins(bins)
model = CombinedRetriever()._ln_like(truth, calc, None, fit_info, depths, errors,
                                     None, None, ret_best_fit=True)[0]
sigma = np.sqrt(errors**2 + truth[fit_info.fit_param_names.index("error_excess")]**2)
for seed in range(4):
    noisy = model + sigma * np.random.default_rng(seed).standard_normal(len(model))
    np.savetxt(f"F01_s{seed}.csv", np.column_stack([bins.mean(1) * 1e6, np.diff(bins)[:, 0] / 2 * 1e6,
                                                     noisy, errors]),
               delimiter=",", header="wavelength,wavelength_err,depth,depth_err", comments="", fmt="%.10e")
