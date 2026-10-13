"""Synthetic TOI-6894 b-like spectrum from a retrieval config, at a known truth.

    python make_synthetic.py --config ../toi6894_smc/TOI-6894_F01_2tp_guillot_free_tls_noise.yaml \
        --truth truth_TOI-6894_F01.json --seed 0 --out F01_s0

Uses the config's data file for the wavelength bins and errors, evaluates the
config's model at the truth (parameter names as in the fit; Rp in R_jup, Mp in
M_jup, error_excess in ppm, T_irr in K for Guillot betas), adds Gaussian noise
with variance error^2 + error_excess^2, and writes <out>.csv, <out>.yaml (the
config pointed at the synthetic data) and <out>_truth.json (the truth in fit
units).
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "toi6894_smc"))
import run_free_retrieval as rfr  # noqa: E402
from platon.combined_retriever import CombinedRetriever  # noqa: E402
from platon.constants import AU, M_jup, R_jup, R_sun  # noqa: E402


def truth_vector(cfg, fit_info, truth):
    star, planet = cfg["star"], cfg["planet"]
    T_ref = star["T_star"] * np.sqrt(star["Rs_Rsun"] * R_sun / (2 * planet["a_AU"] * AU))
    values = {}
    for name in fit_info.fit_param_names:
        if name.endswith("beta"):
            values[name] = truth[name.replace("beta", "T_irr")] / T_ref
            continue
        v = truth[name]
        if name == "Rp":
            v *= R_jup
        elif name == "Mp":
            v *= M_jup
        elif name == "error_excess":
            v *= 1e-6
        values[name] = v
    return np.array([values[n] for n in fit_info.fit_param_names])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    cfg = yaml.safe_load(open(config_path))
    truth = json.load(open(args.truth))
    bins, depths, errors, visits = rfr.load_spectrum(cfg, os.path.dirname(config_path))
    fit_info = rfr.build_fit_info(cfg, visits)
    x = truth_vector(cfg, fit_info, truth)
    if not fit_info._within_limits(x):
        raise ValueError("truth outside the priors")

    retriever = CombinedRetriever()
    Calculator = rfr.use_opacities(cfg["chemistry"]["gases"])
    calc = Calculator(method=cfg.get("rad_method", "xsec"), **fit_info._stellar_grid_options())
    calc.change_wavelength_bins(bins)
    out = retriever._ln_like(x, calc, None, fit_info, depths, errors, None, None,
                             ret_best_fit=True)
    if not isinstance(out, tuple):
        raise ValueError("the truth has a non-finite likelihood (e.g. T outside the grid)")
    model = out[0]
    rng = np.random.default_rng(args.seed)
    sigma = np.sqrt(errors**2 + (truth.get("error_excess", 0.0) * 1e-6)**2)
    synthetic = model + sigma * rng.standard_normal(len(model))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(args.out)
    mid, half = bins.mean(1) * 1e6, 0.5 * (bins[:, 1] - bins[:, 0]) * 1e6
    np.savetxt(args.out + ".csv", np.column_stack([mid, half, synthetic, errors]),
               delimiter=",", header="wavelength,wavelength_err,depth,depth_err",
               comments="", fmt="%.10e")
    new_cfg = copy.deepcopy(cfg)
    new_cfg.update(tag=name, data_file=name + ".csv", output_dir=name + "_results")
    with open(args.out + ".yaml", "w") as f:
        yaml.safe_dump(new_cfg, f, sort_keys=False)
    lnl_truth = retriever._ln_like(x, calc, None, fit_info, synthetic, errors, None, None)
    with open(args.out + "_truth.json", "w") as f:
        json.dump(dict(names=list(fit_info.fit_param_names), values=list(map(float, x)),
                       truth=truth, lnl_truth=float(lnl_truth), seed=args.seed,
                       chi2_truth=float(np.sum(((synthetic - model) / sigma)**2))), f, indent=1)
    print(f"wrote {args.out}.csv/.yaml/_truth.json; lnL(truth)={lnl_truth:.2f}, {len(model)} points")


if __name__ == "__main__":
    main()
