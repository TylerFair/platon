"""Config-driven free-chemistry transit retrieval with PLATON.

Usage:
    python run_free_retrieval.py --config TOI-6894_F01_2tp_guillot_free_tls_noise.yaml
    python run_free_retrieval.py --config ... --check   # build + 1 likelihood call, no sampling

The sampler (smc, multinest or nautilus) is chosen in the config's sampler
section, so the same retrieval can be run with each for comparison.

The data file is a CSV with columns wavelength and wavelength_err
(half-width), in um, and depth and depth_err, as fractions, plus an
optional visit column for joint multi-visit fits (see coadd_visits.py).  The terminator
is isothermal or Guillot, either 1D (optionally with a patchy cloud_fraction) or
two sectors (terminator.two_sector) with independent, identical priors per sector.
"""
import argparse
import os
import pickle
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import yaml

from platon.combined_retriever import CombinedRetriever
from platon.constants import AU, R_sun, R_jup, M_jup, M_earth
from platon.plotter import Plotter
from platon.terminator import TerminatorSector, TwoSectorTerminator
from platon.TP_profile import Profile


def load_spectrum(cfg, config_dir):
    path = cfg["data_file"]
    if not os.path.isabs(path):
        path = os.path.join(config_dir, path)
    d = np.genfromtxt(path, delimiter=",", names=True)
    wave, half = d["wavelength"], d["wavelength_err"]
    bins = 1e-6 * np.column_stack([wave - half, wave + half])
    # An optional integer "visit" column (rows grouped by visit) becomes
    # PLATON transit_visits, so stellar contamination can differ per visit
    visits = None
    if "visit" in d.dtype.names:
        v = d["visit"].astype(int)
        if np.any(np.diff(v) < 0):
            raise ValueError("rows must be grouped by visit")
        visits = {f"visit{i}": (int(np.argmax(v == i)), int(len(v) - np.argmax(v[::-1] == i)))
                  for i in np.unique(v)}
    print(f"{len(wave)} bins, {wave.min() - half.max():.2f}-{wave.max() + half.max():.2f} um, "
          f"visits {visits}, from {path}")
    return bins, d["depth"], d["depth_err"], visits


def planet_mass(planet):
    """Mp and its prior sigma in kg, from Mp_Mearth or Mp_Mjup (sigma optional)."""
    unit, key = (M_jup, "Mjup") if "Mp_Mjup" in planet else (M_earth, "Mearth")
    sigma = planet.get(f"Mp_sigma_{key}")
    return planet[f"Mp_{key}"] * unit, None if sigma is None else sigma * unit


def build_terminator(cfg, Rs, Mp, Rp):
    """Template TwoSectorTerminator (cold and hot sectors identical at the
    starting guess, so the cold one is not hotter), or None for a 1D
    terminator.  Guillot sectors share T_star, Rs, a, Mp, Rp, log_k_th, T_int."""
    if not cfg.get("terminator", {}).get("two_sector", False):
        return None
    prof = cfg.get("profile", {"type": "isothermal"})
    if prof["type"] == "guillot":
        profile = Profile.guillot(cfg["star"]["T_star"], Rs, cfg["planet"]["a_AU"] * AU, Mp, Rp,
                                  **guillot_guess(cfg))
    else:
        profile = Profile.isothermal(cfg["planet"]["T_guess"])
    sector = TerminatorSector(profile, cloudtop_pressure=10.0**cfg["clouds"]["log_cloudtop_P"][1])
    return TwoSectorTerminator(sector, sector, cold_fraction=0.5)


def beta_range(cfg):
    """Guillot T_irr prior in K, as a prior on beta = T_irr / (T_star sqrt(Rs / 2a))
    at the nominal T_star (T_irr scales with T_star if T_star is fitted)."""
    star, planet = cfg["star"], cfg["planet"]
    T_ref = star["T_star"] * np.sqrt(star["Rs_Rsun"] * R_sun / (2 * planet["a_AU"] * AU))
    return [T / T_ref for T in cfg["profile"]["T_irr_range"]]


def guillot_guess(cfg):
    """Starting Guillot parameters: T_irr = T_guess and the lowest T_int, so
    that the deep profile stays within PLATON's temperature grid (PLATON
    rejects the fit if the starting profile does not)."""
    prof, star, planet = cfg["profile"], cfg["star"], cfg["planet"]
    T_ref = star["T_star"] * np.sqrt(star["Rs_Rsun"] * R_sun / (2 * planet["a_AU"] * AU))
    return dict(beta=planet["T_guess"] / T_ref, log_k_th=np.mean(prof["log_k_th"]),
                log_gamma=np.mean(prof["log_gamma"]),
                T_int=prof["T_int"][0] if "T_int" in prof else 100)


def build_fit_info(cfg, visits=None):
    star, planet, tls = cfg["star"], cfg["planet"], cfg.get("tls", {})
    model = tls.get("model", "none")   # none | one_het | two_het
    # Covering fractions per visit (spots evolve between transits); the
    # heterogeneity temperatures stay shared
    per_visit = tls.get("per_visit_cov_frac", False) and visits
    prof = cfg.get("profile", {"type": "isothermal"})
    guillot = prof["type"] == "guillot"
    Rs, Rp = star["Rs_Rsun"] * R_sun, planet["Rp_RJ"] * R_jup
    Mp, Mp_sigma = planet_mass(planet)
    terminator = build_terminator(cfg, Rs, Mp, Rp)

    kw = {}
    if model != "none":
        kw.update(T_het=star["T_star"] - 300, het_cov_frac=0.0,
                  stellar_grid=tls.get("stellar_grid", "newera"),
                  logg_star=star["logg"], feh_star=star["feh"],
                  transit_visits=visits if per_visit else None)
        if model == "two_het":
            kw.update(T_het2=star["T_star"] + 300, het2_cov_frac=0.0)
    if model != "none" or guillot:
        kw["T_star"] = star["T_star"]
    if terminator is not None:
        kw["transit_terminator"] = terminator
    elif guillot:
        kw.update(transit_profile_type="guillot", a=planet["a_AU"] * AU, **guillot_guess(cfg))

    cl = cfg["clouds"]
    fit_info = CombinedRetriever.get_default_fit_info(
        Rs=Rs, Mp=Mp, Rp=Rp, T=None if guillot else planet["T_guess"],
        logZ=None, CO_ratio=None, fit_vmr=True,
        log_cloudtop_P=cl["log_cloudtop_P"][1], log_scatt_factor=0, scatt_slope=4,
        cloud_fraction=1, error_excess=0, **kw)

    fit_info.add_uniform_fit_param("Rp", *[f * Rp for f in planet["Rp_frac_range"]])
    if Mp_sigma is not None:
        fit_info.add_gaussian_fit_param("Mp", Mp_sigma)

    # Temperature profile and clouds: once for a 1D terminator, or with
    # independent, identical priors for sector1 and sector2 of a two-sector one
    sectors = [""] if terminator is None else ["sector1.", "sector2."]
    if guillot:
        for s in sectors:
            fit_info.add_uniform_fit_param(s + "beta", *beta_range(cfg))
            fit_info.add_uniform_fit_param(s + "log_gamma", *prof["log_gamma"])
        fit_info.add_uniform_fit_param("log_k_th", *prof["log_k_th"])
        if "T_int" in prof:
            fit_info.add_uniform_fit_param("T_int", *prof["T_int"])
    else:
        for s in sectors:
            fit_info.add_uniform_fit_param(s + "T", *planet["T_range"])
    for s in sectors:
        fit_info.add_uniform_fit_param(s + "log_cloudtop_P", *cl["log_cloudtop_P"])
        fit_info.add_uniform_fit_param(s + "log_scatt_factor", *cl["log_scatt_factor"])
        fit_info.add_uniform_fit_param(s + "scatt_slope", *cl["scatt_slope"])
    if terminator is not None and "fraction_range" in cfg["terminator"]:
        fit_info.add_uniform_fit_param("sector1.fraction", *cfg["terminator"]["fraction_range"])
    if "cloud_fraction" in cl:
        fit_info.add_uniform_fit_param("cloud_fraction", *cl["cloud_fraction"])
    fit_info.add_uniform_fit_param("error_excess", *[1e-6 * x for x in cfg["error_excess_ppm"]])

    if model != "none":
        hets = [("T_het", "het_cov_frac")]
        if model == "two_het":
            hets.append(("T_het2", "het2_cov_frac"))
        for T_name, f_name in hets:
            fit_info.add_uniform_fit_param(T_name, *tls[T_name + "_range"])
            for name in ([f"{v}.{f_name}" for v in visits] if per_visit else [f_name]):
                fit_info.add_uniform_fit_param(name, *tls["cov_frac_range"])
    if "T_star_sigma" in star:
        fit_info.add_gaussian_fit_param("T_star", star["T_star_sigma"])

    chem = cfg["chemistry"]
    fit_info.add_gases_vmr(list(chem["gases"]) + [chem.get("filler", "H2-He")],
                           10.0**chem["log_vmr_range"][0], 10.0**chem["log_vmr_range"][1])
    return fit_info


def use_opacities(gases):
    """Load opacities for exactly these gases.  CombinedRetriever builds its
    TransitDepthCalculator internally with PLATON's default include_opacities
    (which lacks e.g. CS2), so point it at a subclass with our list."""
    import platon.combined_retriever as cr
    from platon.transit_depth_calculator import TransitDepthCalculator

    class Calculator(TransitDepthCalculator):
        def __init__(self, *args, include_opacities=None, **kwargs):
            super().__init__(*args, include_opacities=list(gases), **kwargs)

    cr.TransitDepthCalculator = Calculator
    print("opacities:", list(gases))
    return Calculator


def print_free_params(fit_info):
    names = fit_info.fit_param_names
    print(f"{len(names)} free parameters:")
    for name in names:
        p = fit_info.all_params[name]
        if name == "Mp":
            prior = f"N({p.best_guess / M_earth:.4g}, {p.std / M_earth:.4g}) M_earth"
        elif hasattr(p, "std"):
            prior = f"N({p.best_guess:.4g}, {p.std:.4g})"
        else:
            lo, hi = p.low_lim, p.high_lim
            if name == "Rp":
                prior = f"U({lo / R_jup:.4g}, {hi / R_jup:.4g}) R_J"
            elif name == "error_excess":
                prior = f"U({lo * 1e6:.4g}, {hi * 1e6:.4g}) ppm"
            else:
                prior = f"U({lo:.4g}, {hi:.4g})"
        print(f"  {name:18s} {prior}")
    fixed = {k: fit_info.all_params[k].best_guess for k in ("Rs", "Mp", "T_star", "T_het", "het_cov_frac",
                                                           "T_het2", "het2_cov_frac", "cloud_fraction",
                                                           "stellar_grid", "logg_star", "feh_star")
             if k in fit_info.all_params and k not in names
             and not any(n.endswith("." + k) for n in names)}
    fixed["Rs"] /= R_sun
    if "Mp" in fixed:
        fixed["Mp"] /= M_earth
    print("  fixed:", ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in fixed.items()))
    print("  filler gas:", fit_info.gases[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--check", action="store_true",
                        help="build everything and evaluate one likelihood, then exit")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    config_dir = os.path.dirname(config_path)

    bins, depths, errors, visits = load_spectrum(cfg, config_dir)
    fit_info = build_fit_info(cfg, visits)
    print_free_params(fit_info)
    Calculator = use_opacities(cfg["chemistry"]["gases"])

    out_dir = cfg["output_dir"]
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(config_dir, out_dir)
    os.makedirs(os.path.join(out_dir, "chains"), exist_ok=True)
    shutil.copy(config_path, os.path.join(out_dir, os.path.basename(config_path)))
    # PLATON writes BestFit.txt to the cwd, and MultiNest needs a short
    # (< 100 character) basename, so work inside the output directory.
    os.chdir(out_dir)
    tag = cfg["tag"]
    retriever = CombinedRetriever()
    sampler = cfg.get("sampler", {})
    common = dict(rad_method=cfg.get("rad_method", "xsec"),
                  num_final_samples=sampler.get("num_final_samples", 100))

    if args.check:
        calc = Calculator(method=common["rad_method"], **fit_info._stellar_grid_options())
        print("loaded opacities:", sorted(calc.atm.raw["opac_names"]) if "opac_names" in calc.atm.raw else "?")
        calc.change_wavelength_bins(bins)
        retriever._validate_params(fit_info, calc)
        centre = fit_info._from_unit_interval_array(np.full(len(fit_info.fit_param_names), 0.5))
        for i in range(3):
            t0 = time.time()
            lnl = retriever._ln_like(centre, calc, None, fit_info, depths, errors, None, None)
            print(f"lnL at prior centre = {lnl:.3f} ({time.time() - t0:.3f} s)")
        cube = np.random.default_rng(0).random((5, len(centre)))
        for u in cube:
            print("lnL at prior draw =", retriever._ln_like(
                fit_info._from_unit_interval_array(u), calc, None, fit_info, depths, errors, None, None))
        return

    name = sampler.get("name", "multinest")
    n_free = len(fit_info.fit_param_names)
    n_live = sampler["n_live_per_param"] * n_free if "n_live_per_param" in sampler \
        else sampler.get("n_live", 1000)
    t0 = time.time()
    if name == "multinest":
        mn_kwargs = dict(outputfiles_basename="chains/mn_", resume=True,
                         sampling_efficiency=sampler.get("sampling_efficiency", 0.8),
                         evidence_tolerance=sampler.get("evidence_tolerance", 0.5),
                         multimodal=sampler.get("multimodal", False))
        result = retriever.run_multinest(bins, depths, errors, None, None, None, fit_info,
                                         nlive=n_live,
                                         multinest_kwargs=mn_kwargs, **common)
    elif name == "nautilus":
        result = retriever.run_nautilus(bins, depths, errors, None, None, None, fit_info,
                                        n_live=n_live,
                                        n_eff=sampler.get("n_eff", 10000),
                                        filepath="chains/nautilus_checkpoint.hdf5", resume=True,
                                        **common)
    elif name == "smc":
        # Options other than name go to CombinedRetriever.run_smc (and from
        # there to platon.smc_sampler.run_smc)
        smc_kwargs = {k: v for k, v in sampler.items()
                      if k not in ("name", "num_final_samples")}
        result = retriever.run_smc(bins, depths, errors, None, None, None, fit_info,
                                   log=print, **smc_kwargs, **common)
        print(f"{result.n_evals} full and {result.n_cheap_evals} cheap model evaluations")
    else:
        raise ValueError(f"unknown sampler {name}")
    print(f"Sampling finished in {(time.time() - t0) / 3600:.2f} h; logZ = {result.logz}")

    with open(f"retrieval_result_{tag}.pkl", "wb") as f:
        pickle.dump(result, f)
    Plotter.plot_retrieval_transit_spectrum(result, prefix=f"best_fit_{tag}")
    Plotter.plot_retrieval_corner(result, filename=f"corner_{tag}.png")
    if cfg.get("profile", {}).get("type") == "guillot" or cfg.get("terminator", {}).get("two_sector"):
        Plotter.plot_retrieval_TP_profiles(result, prefix=f"TP_{tag}")
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
