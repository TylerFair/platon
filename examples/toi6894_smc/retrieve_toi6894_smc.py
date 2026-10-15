"""Two-sector Guillot free retrieval of TOI-6894 b with tempered SMC.

27 free parameters: Rp, Mp, per-sector Guillot beta/log_gamma, cloud deck
and haze, shared log_k_th and T_int, sector1 fraction, excess noise, one
stellar heterogeneity, T_star and 8 gases.  MultiNest needed ~1.8e8
likelihood calls for this retrieval. SMC defaults to sequential two-fold
cross-fitting with fixed 60 inner / 12 outer moves. For a biased legacy
comparison, pass folds=1, adaptive_steps=True.

    python retrieve_toi6894_smc.py [spectrum.csv]
"""
import pickle
import sys

import numpy as np

import platon.combined_retriever as cr
from platon.constants import AU, M_jup, R_jup, R_sun
from platon.plotter import Plotter
from platon.terminator import TerminatorSector, TwoSectorTerminator
from platon.TP_profile import Profile
from platon.transit_depth_calculator import TransitDepthCalculator

GASES = ["H2O", "NH3", "CH4", "CS2", "H2S", "HCN", "CO2", "CO"]
Rs, Mp, Rp, a, T_star = 0.2276 * R_sun, 0.168 * M_jup, 0.855 * R_jup, 0.02603 * AU, 3007.0
T_ref = T_star * np.sqrt(Rs / (2 * a))  # T_irr = beta * T_ref


class Calculator(TransitDepthCalculator):
    # PLATON's default opacity list lacks CS2
    def __init__(self, *args, include_opacities=None, **kwargs):
        super().__init__(*args, include_opacities=GASES, **kwargs)


cr.TransitDepthCalculator = Calculator


def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    bins = 1e-6 * np.column_stack([d["wavelength"] - d["wavelength_err"],
                                   d["wavelength"] + d["wavelength_err"]])
    return bins, d["depth"], d["depth_err"]


def get_fit_info():
    profile = Profile.guillot(T_star, Rs, a, Mp, Rp, beta=390 / T_ref,
                              log_k_th=-1.75, log_gamma=-0.75, T_int=30)
    sector = TerminatorSector(profile, cloudtop_pressure=1e8)
    fit_info = cr.CombinedRetriever.get_default_fit_info(
        Rs=Rs, Mp=Mp, Rp=Rp, T=None, logZ=None, CO_ratio=None, fit_vmr=True,
        log_cloudtop_P=8, T_star=T_star, T_het=T_star - 300, het_cov_frac=0,
        stellar_grid="newera", logg_star=5.039, feh_star=0.142,
        transit_terminator=TwoSectorTerminator(sector, sector, cold_fraction=0.5))
    fit_info.add_uniform_fit_param("Rp", 0.85 * Rp, 1.15 * Rp)
    fit_info.add_gaussian_fit_param("Mp", 0.022 * M_jup)
    for s in ["sector1.", "sector2."]:
        fit_info.add_uniform_fit_param(s + "beta", 300 / T_ref, 3000 / T_ref)
        fit_info.add_uniform_fit_param(s + "log_gamma", -2, 0.5)
    fit_info.add_uniform_fit_param("log_k_th", -3, -0.5)
    fit_info.add_uniform_fit_param("T_int", 30, 630)
    for s in ["sector1.", "sector2."]:
        fit_info.add_uniform_fit_param(s + "log_cloudtop_P", -1, 8)
        fit_info.add_uniform_fit_param(s + "log_scatt_factor", -4, 8)
        fit_info.add_uniform_fit_param(s + "scatt_slope", -2, 15)
    fit_info.add_uniform_fit_param("sector1.fraction", 0, 1)
    fit_info.add_uniform_fit_param("error_excess", 0, 1e-3)
    fit_info.add_uniform_fit_param("T_het", 2300, 4507)
    fit_info.add_uniform_fit_param("het_cov_frac", 0, 0.2)
    fit_info.add_gaussian_fit_param("T_star", 58)
    fit_info.add_gases_vmr(GASES + ["H2-He"], 1e-12, 1e-1)
    return fit_info


if __name__ == "__main__":
    bins, depths, errors = load(sys.argv[1] if len(sys.argv) > 1 else "TOI-6894_qiao_power2_0p8_5p2.csv")
    result = cr.CombinedRetriever().run_smc(bins, depths, errors, None, None, None, get_fit_info(),
                                            n_particles=512, cheap_downsample=8, seed=0, log=print)
    print(f"logZ = {result.final_logz:.2f}; {result.n_evals} full, {result.n_cheap_evals} cheap evaluations")
    with open("retrieval_result.pkl", "wb") as f:
        pickle.dump(result, f)
    Plotter.plot_retrieval_transit_spectrum(result, prefix="best_fit")
    Plotter.plot_retrieval_corner(result, filename="corner.png")
    Plotter.plot_retrieval_TP_profiles(result, prefix="TP")
