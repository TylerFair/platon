"""Vectorized host preparation for transit likelihoods: two-sector terminators
or a single (1-D) terminator, with Guillot, isothermal or Madhusudhan & Seager
parametric profiles.

CombinedRetriever's samplers use this batch wherever it applies; it returns
the scalar likelihood's values exactly.  It deliberately checks its
supported model family:
equilibrium abundances, free vertically constant VMRs with a filler gas, or
free VMRs on top of equilibrium, shared or per sector; fitted (partial) or
fixed full cloud cover per sector, or a 1-D patchy cloud_fraction; at most
one unocculted stellar heterogeneity.  Unsupported physics/settings raise at
construction, and rows the batch cannot reproduce exactly are evaluated by
the scalar likelihood, instead of silently using an approximate forward or
changing the scalar likelihood.
"""
import math
from types import SimpleNamespace

import numpy as np
from scipy.special import expn
from scipy.stats import norm

from . import _forward_model as fm
from ._batched_forward_model import make_forward
from ._forward_prep import prepare_forward_inputs
from .TP_profile import Profile, _default_pressures
from .constants import G
from ._params import _GaussianParam
from .terminator import TwoSectorTerminator, limb_time_shifts
from .transit_depth_calculator import DEEP_CLAMP_PRESSURE

SECTORS = ("sector1.", "sector2.")
# A 1-D terminator is one lane; its resolved values are stored under this prefix
LANE = "lane0."
GUILLOT_SHARED = ("T_star", "Rs", "a", "Mp", "Rp", "log_k_th", "T_int")
CLOUD_PARAMS = ("log_cloudtop_P", "log_scatt_factor", "scatt_slope", "cloud_fraction")
# Per-sector profile parameters of each supported two-sector profile type
PROFILE_PARAMS = {"guillot": ("beta", "log_gamma"), "isothermal": ("T",),
                  "parametric": ("T0", "log_P1", "alpha1", "alpha2", "log_P3", "T3")}
# Rejection reasons: the scalar path's AtmosphereError messages where it raises
INVALID_PROFILE = "Invalid T/P profile"
INVALID_TEMPERATURES = "Invalid temperatures in T/P profile"
BELOW_CLOUDS = "Entire atmosphere is below the clouds"
UNBOUND = "Atmosphere unbound: height > hill radius"


def _scalar_power(base, exponent):
    """base**exponent elementwise with NumPy-scalar semantics (C pow).

    The scalar likelihood raises NumPy scalars to powers, while the array
    ufunc may use SIMD kernels that differ from it in the last bit, so the
    batch loops over scalars wherever the scalar path computes on scalars."""
    base, exponent = np.broadcast_arrays(np.asarray(base, dtype=float),
                                         np.asarray(exponent, dtype=float))
    return np.array([b ** e for b, e in zip(base.ravel(), exponent.ravel())],
                    dtype=float).reshape(base.shape)


class BatchedLikelihood:
    # Two-sector by default; __init__ sets these from the fit
    one_d = False
    prefixes = SECTORS
    profile_type = "guillot"

    def __init__(self, retriever, transit_calc, eclipse_calc, fit_info,
                 transit_depths, transit_errors, eclipse_depths, eclipse_errors,
                 zero_opacities=(), batch_size=64):
        if transit_calc is None or eclipse_calc is not None or eclipse_depths is not None:
            raise ValueError("Batched likelihood requires transit-only retrievals")
        self.retriever = retriever
        self.calc = transit_calc
        self.fit = fit_info
        self.zero_opacities = zero_opacities
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.depths = np.asarray(transit_depths)
        self.errors = np.asarray(transit_errors)
        self.defaults = {n: p.best_guess for n, p in fit_info.all_params.items()}
        p = self.defaults
        names = set(fit_info.fit_param_names)
        offsets = set().union(*[set(p.get(kind + "_offsets") or {})
                                for kind in ("transit", "time_shift")])
        self.vectorized_finish = set(getattr(fit_info, "marginal_params", ())) <= offsets
        terminator = p.get("transit_terminator")
        self.one_d = terminator is None
        if self.one_d:
            self.profile_type = p.get("transit_profile_type", "isothermal")
            self.prefixes = (LANE,)
        elif isinstance(terminator, TwoSectorTerminator):
            self.profile_type = terminator.profile_type
            self.prefixes = SECTORS
        else:
            self.profile_type = None
        if self.profile_type not in PROFILE_PARAMS:
            raise ValueError("Batched likelihood requires Guillot, isothermal or parametric "
                             "transit profiles")
        # Free VMRs either replace equilibrium chemistry entirely (the last gas
        # is the filler) or, with logZ and CO_ratio set, replace the listed
        # gases' equilibrium abundances; either may be shared or per sector
        on_eq = p["logZ"] is not None and p["CO_ratio"] is not None
        self.free_vmr = bool(p["fit_vmr"]) and p["logZ"] is None and p["CO_ratio"] is None
        if p["fit_clr"] or (p["fit_vmr"] and not (self.free_vmr or on_eq)) or \
           (not p["fit_vmr"] and not on_eq):
            raise ValueError("Batched likelihood requires equilibrium abundances, free VMRs "
                             "with a filler gas, or free VMRs on top of equilibrium")
        self.gases = list(fit_info.gases) if p["fit_vmr"] else None
        # The fitted gases: all of them on top of equilibrium, all but the filler otherwise
        self.n_free = 0 if self.gases is None else len(self.gases) - self.free_vmr
        self.per_sector_gases = self.gases is not None and not self.one_d and any(
            s + "log_" + gas in fit_info.all_params for s in SECTORS for gas in self.gases)
        if p.get("n") is not None:
            raise ValueError("Batched likelihood does not support Mie scattering")
        if not self.one_d and p["cloud_fraction"] != 1:
            raise ValueError("A two-sector terminator takes per-sector cloud fractions")
        # Fixed full cloud cover uses the scalar path's single-terminator core,
        # otherwise its fused cloudy/clear core (the 1-D patchy cloud_fraction,
        # or the sectors' own)
        if self.one_d:
            self.dual = "cloud_fraction" in names or p["cloud_fraction"] != 1
        else:
            self.dual = any(s + "cloud_fraction" in names or p[s + "cloud_fraction"] != 1
                            for s in SECTORS)
        if p.get("transit_visits") or {"T_het2", "het2_cov_frac"} & names or \
           np.ndim(p["T_het"]) or np.ndim(p["het_cov_frac"]) or \
           (p.get("het2_cov_frac") is not None and np.any(np.asarray(p["het2_cov_frac"]) != 0)):
            raise ValueError("Batched likelihood supports one heterogeneity without "
                             "per-visit values")
        supported = {"Rp", "Mp", "T_star", "T_het", "het_cov_frac", "log_P_quench",
                     "error_excess"}
        if self.profile_type == "guillot":
            supported.update(("log_k_th", "T_int"))
        if not self.free_vmr:
            supported.update(("logZ", "CO_ratio"))
        if self.gases is not None:
            for prefix in SECTORS if self.per_sector_gases else ("",):
                supported.update(prefix + "log_" + gas for gas in self.gases[:self.n_free])
        if self.one_d:
            # Profile.from_params_dict lets "_transit" names override them
            profile_names = PROFILE_PARAMS[self.profile_type] + \
                (("log_k_th", "T_int") if self.profile_type == "guillot" else ())
            supported.update(profile_names)
            supported.update(name + "_transit" for name in profile_names)
            supported.update(CLOUD_PARAMS)
            supported.update((p.get("transit_offsets") or {}).keys())
        else:
            supported.add("sector1.fraction")
            for sector in SECTORS:
                supported.update(sector + n for n in PROFILE_PARAMS[self.profile_type] + CLOUD_PARAMS)
            for kind in ("transit", "time_shift"):
                supported.update((p.get(kind + "_offsets") or {}).keys())
        unsupported = names - supported
        if unsupported:
            raise ValueError("Unsupported batched parameters: {}".format(sorted(unsupported)))
        for key in ("transit_error_excesses", "time_shift_slopes"):
            if any(p[name] != 0 for name in (p.get(key) or {})
                   if name not in getattr(fit_info, "marginal_params", ())):
                raise ValueError("Batched likelihood supports integrated {} only".format(key))
        self.pressures = _default_pressures()
        if self.one_d:
            # The scalar path's own 1-D profile and clouds
            first = SimpleNamespace(
                profile=Profile.from_params_dict(self.profile_type, p, suffix="_transit"),
                cloudtop_pressure=10.0**p["log_cloudtop_P"],
                scattering_factor=10.0**p["log_scatt_factor"],
                scattering_slope=p["scatt_slope"])
        else:
            first = terminator.sectors_from_params(p)[0]
        het_cov_frac = p["het_cov_frac"]
        if "het_cov_frac" in names and not het_cov_frac:
            # Any covering fraction inside the prior: the template only needs
            # the heterogeneity branch that sampled rows use
            prior = fit_info.all_params["het_cov_frac"]
            het_cov_frac = 0.5 * (prior.low_lim + prior.high_lim)
        vmrs = None
        if self.gases is not None:
            prefix = "sector1." if self.per_sector_gases else ""
            vmrs = [10.0**p[prefix + "log_" + gas] for gas in self.gases[:self.n_free]]
            if self.free_vmr:
                vmrs.append(1 - np.sum(vmrs))
        cfg, pin, _ = prepare_forward_inputs(
            transit_calc.atm, star_radius=p["Rs"], planet_mass=p["Mp"],
            planet_radius=p["Rp"], P_profile=self.pressures,
            T_profile=first.profile.temperatures, logZ=p["logZ"], CO_ratio=p["CO_ratio"],
            CH4_mult=10.0**p["log_CH4_mult"], gases=self.gases, vmrs=vmrs,
            add_gas_absorption=True, add_H_minus_absorption=False, add_scattering=True,
            scattering_factor=first.scattering_factor, scattering_slope=first.scattering_slope,
            scattering_ref_wavelength=1e-6, add_collisional_absorption=True,
            cloudtop_pressure=first.cloudtop_pressure, custom_abundances=None,
            T_star=p["T_star"], T_het=p["T_het"], het_cov_frac=het_cov_frac,
            T_het2=p.get("T_het2"), het2_cov_frac=p.get("het2_cov_frac"), ri=None,
            frac_scale_height=p["frac_scale_height"], number_density=10.0**p["log_number_density"],
            part_size=10.0**p["log_part_size"], part_size_std=0.5,
            P_quench=10.0**p["log_P_quench"], zero_opacities=zero_opacities,
            stellar_blackbody=False, bot_pressure=first.cloudtop_pressure)
        self.cfg = cfg
        self.template = pin.packed
        self.stellar_temps = transit_calc.atm.raw["stellar_temps"]
        self.cloudtop_range = (float(transit_calc.atm.P_grid.min()),
                               float(transit_calc.atm.P_grid.max()))
        self.data = transit_calc.atm.device_data()
        self.forward = make_forward(cfg, dual=self.dual, with_vmrs=self.gases is not None)
        self.last_depths = None
        self.last_reasons = None

    def _values(self, rows):
        values = dict(self.defaults)
        values.update(zip(self.fit.fit_param_names, rows.T))
        for name in getattr(self.fit, "marginal_params", ()):
            values[name] = 0.0
        if self.one_d:
            def resolve(name, default=None):
                # Profile.from_params_dict's "_transit" override
                value = values.get(name + "_transit")
                value = values.get(name, default) if value is None else value
                return default if value is None else value
            for name in PROFILE_PARAMS[self.profile_type]:
                values[LANE + name] = resolve(name)
            if self.profile_type == "guillot":
                for name in GUILLOT_SHARED:
                    values["profile." + name] = resolve(name, 100 if name == "T_int" else None)
            for name in CLOUD_PARAMS:
                values[LANE + name] = values[name]
        return values

    def _temperatures(self, values, size):
        """Unclamped temperatures (size, lanes, layers), exactly as the scalar
        profiles compute them; lanes are the two sectors, or one for 1-D."""
        if self.profile_type == "isothermal":
            # Profile.isothermal: np.ones(layers) * T
            T = np.stack([np.broadcast_to(np.asarray(values[s + "T"], dtype=float), (size,))
                          for s in self.prefixes], axis=1)
            return np.ones(len(self.pressures)) * T[..., None]
        if self.profile_type == "parametric":
            # Profile.parametric mixes scalar and array maths; call it per row
            # so that every operation is the scalar path's own
            names = PROFILE_PARAMS["parametric"]
            temperatures = np.empty((size, len(self.prefixes), len(self.pressures)))
            with np.errstate(all="ignore"):
                for i in range(size):
                    for k, s in enumerate(self.prefixes):
                        params = {n: values[s + n][i] if np.ndim(values[s + n]) else values[s + n]
                                  for n in names}
                        temperatures[i, k] = Profile.from_params_dict(
                            "parametric", params).temperatures
            return temperatures
        return self._guillot_temperatures(values, size)

    def _guillot_temperatures(self, values, size):
        """Unclamped Guillot temperatures (size, lanes, layers), as Profile.guillot
        computes them: its scalar powers use C pow, its array ones the ufunc."""
        def shared(name):
            return values["profile." + name] if self.one_d else values[name]

        def column(name):
            return np.broadcast_to(np.asarray(shared(name), dtype=float), (size,))

        def sectors(name):
            return np.stack([np.broadcast_to(np.asarray(values[s + name], dtype=float), (size,))
                             for s in self.prefixes], axis=1)

        p = self.pressures
        g = G * column("Mp") / _scalar_power(column("Rp"), 2)
        t_eq = sectors("beta") * np.sqrt(shared("Rs") / (2 * shared("a"))) * \
            column("T_star")[:, None]
        taus = _scalar_power(10.0, column("log_k_th"))[:, None, None] * p / g[:, None, None]
        gamma = _scalar_power(10.0, sectors("log_gamma"))[..., None]
        with np.errstate(all="ignore"):
            e1 = 3.0 / 4 * _scalar_power(t_eq, 4)[..., None] * (
                2.0 / 3 + 2.0 / 3 / gamma *
                (1 + (gamma * taus / 2 - 1) * np.exp(-gamma * taus)) +
                2.0 * gamma / 3 * (1 - taus**2 / 2) * expn(2, gamma * taus))
            t4 = 3.0 / 4 * _scalar_power(column("T_int"), 4)[:, None, None] * \
                (2.0 / 3 + taus) + e1
            return t4**0.25

    def _swap(self, temperatures):
        if self.one_d:
            return np.zeros(len(temperatures), dtype=bool)
        probed = (self.pressures >= 10) & (self.pressures <= 1e5)
        means = temperatures[..., probed].mean(axis=2)
        return means[:, 0] > means[:, 1]

    def sector_swap(self, rows):
        """Whether sector1 is the hotter sector (so labelled hot), per row."""
        rows = np.asarray(rows, dtype=float)
        return self._swap(self._temperatures(self._values(rows), len(rows)))

    def prepare(self, rows):
        """Float64 numpy profiles/validation, followed by one float32 pack.

        Returns packed inputs, float32 VMRs (or None), rows to evaluate on
        the device, rows for the scalar likelihood, values, cloud fractions,
        sector1 fractions and per-row rejection reasons."""
        rows = np.asarray(rows, dtype=float)
        if rows.ndim != 2 or rows.shape[1] != len(self.fit.fit_param_names):
            raise ValueError("Expected (batch, n_parameters) input")
        size = len(rows)
        values = self._values(rows)

        def column(name):
            return np.broadcast_to(np.asarray(values[name], dtype=float), (size,))

        def sectors(name):
            return np.stack([column(prefix + name) for prefix in self.prefixes], axis=1)

        reasons = np.full(size, "", dtype=object)

        def reject(mask, reason):
            reasons[mask & (reasons == "")] = reason

        # Checks in the scalar path's order, so reasons match its first one
        within_limits = np.array([self.fit._within_limits(row) for row in rows], dtype=bool)
        reject(~within_limits, "limits")
        fraction = np.ones(size) if self.one_d else column("sector1.fraction")
        clouds = sectors("cloud_fraction")
        reject(~((0 <= fraction) & (fraction <= 1) &
                 np.all((0 <= clouds) & (clouds <= 1), axis=1)), "fractions")
        # Rows the batch does not reproduce exactly go to the scalar likelihood.
        # Boundary cloud fractions take different scalar cores. Evaluate those
        # uncommon rows with the reference rather than changing their rounding.
        fallback = np.zeros(size, dtype=bool) if self.one_d else \
            (fraction <= 0) | (fraction >= 1)
        if self.dual:
            fallback |= ~np.all((clouds > 0) & (clouds < 1), axis=1)
        vmrs = None
        if self.gases is not None:
            # (size, 1 or 2 lanes, gases): shared by a row's sectors, or per sector
            lanes_vmrs = []
            for prefix in SECTORS if self.per_sector_gases else ("",):
                lane = np.empty((size, len(self.gases)))
                lane[:, :self.n_free] = _scalar_power(10.0, np.stack(
                    [column(prefix + "log_" + gas) for gas in self.gases[:self.n_free]], axis=1))
                if self.free_vmr:
                    lane[:, -1] = [1 - np.sum(row) for row in lane[:, :-1]]
                    reject(lane[:, -1] < 0, "filler")
                lanes_vmrs.append(lane)
            vmrs = np.stack(lanes_vmrs, axis=1)
        reject((column("Rs") <= 0) | (column("Mp") <= 0), "mass")

        p = self.pressures
        temperatures = self._temperatures(values, size)
        reject(np.any(np.isnan(temperatures), axis=(1, 2)), INVALID_PROFILE)
        values["_batch_swap"] = self._swap(temperatures)
        if not self.one_d:
            # With all of the area in one sector the scalar path skips the other
            cold_fraction = np.where(values["_batch_swap"], 1 - fraction, fraction)
            fallback |= (cold_fraction <= 0) | (cold_fraction >= 1)
        # Same rule as TransitDepthCalculator._clamp_unprobed: out-of-grid
        # temperatures deeper than 10 bar and the quench point are clamped
        quench = _scalar_power(10.0, column("log_P_quench"))
        deep = p > np.maximum(DEEP_CLAMP_PRESSURE, quench)[:, None]
        first = np.searchsorted(p, quench, side="right")
        inside = first < len(p)
        deep[np.flatnonzero(inside), first[inside]] = False
        positive = np.all(np.isfinite(temperatures) & (temperatures > 0), axis=2)
        with np.errstate(invalid="ignore"):
            temperatures = np.where(deep[:, None], np.clip(
                temperatures, self.calc.atm.min_temperature,
                self.calc.atm.max_temperature), temperatures)
            in_grid = positive & \
                (temperatures.min(axis=2) >= self.calc.atm.min_temperature) & \
                (temperatures.max(axis=2) <= self.calc.atm.max_temperature)
            sort_layers = temperatures.max(axis=2) > temperatures.min(axis=2)
        cloudtop = _scalar_power(10.0, sectors("log_cloudtop_P"))
        # Outside the pressure grid the scalar path raises ValueError
        low, high = self.cloudtop_range
        fallback |= np.any(~np.isinf(cloudtop) & ((cloudtop <= low) | (cloudtop > high)),
                           axis=1)
        above = p < cloudtop[..., None]
        visible = np.any(above, axis=2)
        # The scalar path checks the cold sector, then the hot one
        rows_index = np.arange(size)
        lanes = len(self.prefixes)
        for k in range(lanes):
            sector = np.where(values["_batch_swap"], lanes - 1 - k, k)
            reject(~in_grid[rows_index, sector], INVALID_TEMPERATURES)
            reject(~visible[rows_index, sector], BELOW_CLOUDS)

        # Static branches of the forward configuration must match the template's
        het_cov_frac = column("het_cov_frac") if values["het_cov_frac"] is not None \
            else np.zeros(size)
        t_star = column("T_star") if values["T_star"] is not None else None
        if t_star is not None:
            star32 = t_star.astype(np.float32)
            fallback |= ((self.stellar_temps[0] <= star32) &
                         (star32 <= self.stellar_temps[-1])) != self.cfg.stellar_in_grid
        fallback |= (het_cov_frac != 0.0) != self.cfg.has_het
        fallback |= np.any(sort_layers != self.cfg.sort_layers, axis=1)
        fallback &= np.all(np.isfinite(rows), axis=1) & within_limits
        reasons[fallback] = "scalar"
        valid = reasons == ""

        packed = np.broadcast_to(self.template, (size, len(self.prefixes),
                                                 len(self.template))).copy()
        for name, index in (("Rp", fm.SC_RP), ("Mp", fm.SC_MP)):
            packed[:, :, index] = column(name)[:, None]
        if not self.free_vmr:
            packed[:, :, fm.SC_LOGZ] = column("logZ")[:, None]
            packed[:, :, fm.SC_CO] = column("CO_ratio")[:, None]
        if t_star is not None:
            packed[:, :, fm.SC_T_STAR] = t_star[:, None]
            packed[:, :, fm.SC_T_STAR_HYDRO] = t_star[:, None]
            # prepare_forward_inputs: an unset heterogeneity has T_star
            for name, index in (("T_het", fm.SC_T_HET), ("T_het2", fm.SC_T_HET2)):
                if values.get(name) is None:
                    packed[:, :, index] = t_star[:, None]
        if values["T_het"] is not None:
            packed[:, :, fm.SC_T_HET] = column("T_het")[:, None]
        packed[:, :, fm.SC_HET_FRAC] = het_cov_frac[:, None]
        packed[:, :, fm.SC_P_QUENCH] = quench[:, None]
        packed[:, :, fm.SC_LOG10_P_QUENCH] = np.array(
            [math.log10(max(q, 1e-99)) for q in quench])[:, None]
        packed[:, :, fm.SC_SCAT_FACTOR] = _scalar_power(10.0, sectors("log_scatt_factor"))
        packed[:, :, fm.SC_SCAT_SLOPE] = sectors("scatt_slope")
        packed[:, :, fm.SC_CLOUDTOP] = cloudtop
        packed[:, :, fm.SC_N_SCALARS + fm.IX_FLOOR] = above.sum(axis=2) - 1
        offset = fm.SC_N_SCALARS + fm.IX_N_INTS
        packed[:, :, offset:offset + len(p)] = temperatures
        packed[:, :, offset + 2 * len(p):offset + 3 * len(p) - 1] = above[..., 1:]
        if vmrs is not None:
            vmrs = np.asarray(vmrs, dtype=np.float32)
        return packed, vmrs, valid, fallback, values, clouds, fraction, reasons

    def _limbs(self, raw, clouds):
        """Float64 binned depths per lane from device rows (n, lanes, outputs)."""
        nbin = len(self.depths)
        if not self.dual:
            return raw[:, :, :nbin].astype(float)
        parts = raw[:, :, :2 * nbin].reshape(len(raw), raw.shape[1], 2, nbin).astype(float)
        cov = clouds[:, :, None]
        return cov * parts[:, :, 0] + (1 - cov) * parts[:, :, 1]

    def finish(self, rows, raw, indices, values, clouds, fraction):
        """Keep mixing, time shifts and the final likelihood in float64."""
        if not self.vectorized_finish:
            return self._finish_scalar(rows, raw, indices, values, clouds, fraction)
        result = np.full(len(rows), -np.inf)
        nbin = len(self.depths)
        self.last_depths = np.full((len(rows), len(self.prefixes), nbin), np.nan)
        good = np.all(raw[:, :, -2] == 0, axis=1)
        indices = indices[good]
        if not len(indices):
            return result
        limbs = self._limbs(raw[good], clouds[indices])
        self.last_depths[indices] = limbs
        result[indices] = self._finish_from_limbs(
            limbs, indices, values, fraction[indices])
        return result

    def _finish_from_limbs(self, limbs, indices, values, fraction):
        """Vectorize offset-only nuisance integration over rows, in float64.

        Cumulative sums preserve _nuisance_ln_like's sequential bincount
        accumulation. An integrated error_excess uses the general scalar
        implementation instead, including its quadrature.
        """
        size, _, nbin = limbs.shape
        marginal = getattr(self.fit, "marginal_params", ())

        def column(name, default=0):
            value = np.asarray(values.get(name, default))
            return np.full(size, value) if value.ndim == 0 else value[indices]

        def apply_offsets(model, kind):
            for name, (start, end) in (values.get(kind + "_offsets") or {}).items():
                model[:, start:end] += column(name)[:, None]

        if self.one_d:
            calculated = limbs[:, 0].copy()
        else:
            swap = values["_batch_swap"][indices, None]
            cold = np.where(swap, limbs[:, 1], limbs[:, 0])
            hot = np.where(swap, limbs[:, 0], limbs[:, 1])
            fraction = np.where(swap[:, 0], 1 - fraction, fraction)[:, None]
            calculated = fraction * cold + (1 - fraction) * hot
        apply_offsets(calculated, "transit")
        excess = _scalar_power(column("error_excess"), 2)
        errors = np.sqrt(self.errors**2 + excess[:, None])
        blocks = {"transit": (calculated - self.depths, errors)}
        if values.get("time_shifts") is not None:
            coefficients = values.get("time_shift_coefficients")
            shifts = limb_time_shifts(limbs[:, 0], limbs[:, 1], values["Rs"],
                values["a"], values["period"], values["b"]) if coefficients is None else \
                coefficients * (np.sqrt(limbs[:, 1]) - np.sqrt(limbs[:, 0]))
            apply_offsets(shifts, "time_shift")
            errors = np.broadcast_to(values["time_shift_errors"], shifts.shape)
            blocks["time_shift"] = shifts - values["time_shifts"], errors
        if not marginal:
            return np.concatenate([-0.5 * (r**2 / e**2 + np.log(2 * np.pi * e**2))
                                   for r, e in blocks.values()], axis=1).sum(axis=1)

        chi2, ln_errors, counts = np.zeros((3, size))
        offsets = {}
        for kind, (residuals, errors) in blocks.items():
            chi2 += np.cumsum((residuals / errors)**2, axis=1)[:, -1]
            ln_errors += np.cumsum(np.log(errors), axis=1)[:, -1]
            counts += residuals.shape[1]
            for name, (start, end) in (values.get(kind + "_offsets") or {}).items():
                if name in marginal:
                    W, R = offsets.get(name, (0, 0))
                    weights = errors[:, start:end]**-2
                    offsets[name] = (W + weights.sum(axis=1),
                                    R - (weights * residuals[:, start:end]).sum(axis=1))
        terms = -0.5 * chi2 - ln_errors - 0.5 * counts * np.log(2 * np.pi)
        for name, (W, R) in offsets.items():
            if np.any(W == 0):
                raise ValueError("Offset {} covers no data".format(name))
            prior = self.fit.all_params[name]
            mean, std = R / W, 1 / np.sqrt(W)
            terms = terms + 0.5 * R * mean + np.log(np.sqrt(2 * np.pi) * std)
            if isinstance(prior, _GaussianParam):
                terms += norm.logpdf(mean, prior.best_guess, np.hypot(std, prior.std))
            else:
                terms += self.retriever._log_ndtr_diff(
                    (prior.high_lim - mean) / std, (prior.low_lim - mean) / std) \
                    - np.log(prior.high_lim - prior.low_lim)
        return terms

    def _finish_scalar(self, rows, raw, indices, values, clouds, fraction):
        result = np.full(len(rows), -np.inf)
        nbin = len(self.depths)
        limb_depths = np.full((len(rows), len(self.prefixes), nbin), np.nan)
        marginal_rows = []
        for j, i in enumerate(indices):
            if np.any(raw[j, :, -2] != 0):
                continue
            limbs = self._limbs(raw[j:j + 1], clouds[i:i + 1])[0]
            limb_depths[i] = limbs
            if self.one_d:
                calculated = limbs[0].copy()
            elif values["_batch_swap"][i]:
                cold_fraction = 1 - fraction[i]
                calculated = cold_fraction * limbs[1] + (1 - cold_fraction) * limbs[0]
            else:
                calculated = fraction[i] * limbs[0] + (1 - fraction[i]) * limbs[1]
            p = self.fit._interpret_param_array(rows[i])
            for name in getattr(self.fit, "marginal_params", ()):
                p[name] = 0
            self.retriever._apply_offsets(calculated, p, "transit")
            errors = np.sqrt(self.errors**2 + p["error_excess"]**2)
            blocks = {"transit": (calculated - self.depths, errors)}
            if p.get("time_shifts") is not None:
                coefficients = p.get("time_shift_coefficients")
                shifts = limb_time_shifts(limbs[0], limbs[1], p["Rs"], p["a"], p["period"], p["b"]) \
                    if coefficients is None else coefficients * (np.sqrt(limbs[1]) - np.sqrt(limbs[0]))
                self.retriever._apply_offsets(shifts, p, "time_shift")
                errors = p["time_shift_errors"]
                blocks["time_shift"] = shifts - p["time_shifts"], errors
            if getattr(self.fit, "marginal_params", ()):
                marginal_rows.append((i, p, blocks))
            else:
                terms = np.concatenate([-0.5 * (r**2 / e**2 + np.log(2 * np.pi * e**2))
                                        for r, e in blocks.values()])
                result[i] = terms.sum()
        if marginal_rows:
            # All rows' nuisance integrals at once, equal to _nuisance_ln_like row by row
            rows_i, params, blocks_rows = zip(*marginal_rows)
            result[list(rows_i)] = self.retriever._nuisance_ln_like_rows(self.fit, params, blocks_rows)
        self.last_depths = limb_depths
        return result

    def __call__(self, rows):
        rows = np.asarray(rows, dtype=float)
        if rows.ndim != 2:
            raise ValueError("Expected (batch, n_parameters) input")
        lanes = len(self.prefixes)
        if len(rows) == 0:
            self.last_depths = np.empty((0, lanes, len(self.depths)))
            self.last_reasons = np.empty(0, dtype=object)
            return np.empty(0)
        if len(rows) > self.batch_size:
            results, depths, reasons = [], [], []
            for start in range(0, len(rows), self.batch_size):
                results.append(self(rows[start:start + self.batch_size]))
                depths.append(self.last_depths)
                reasons.append(self.last_reasons)
            self.last_depths = np.concatenate(depths)
            self.last_reasons = np.concatenate(reasons)
            return np.concatenate(results)
        packed, vmrs, valid, fallback, values, clouds, fraction, reasons = self.prepare(rows)
        indices = np.flatnonzero(valid)
        raw = np.empty((0, lanes, len(self.template)))
        if len(indices):
            # Cache power-of-two shapes: rejected priors should not force a
            # full GPU batch of duplicate valid rows. There are only log2(B)
            # shapes, rather than a separate compilation for every count.
            capacity = min(self.batch_size, 1 << (len(indices) - 1).bit_length())

            def pad(x):
                return np.concatenate([x, np.repeat(x[-1:], capacity - len(x), axis=0)])
            inputs = [pad(packed[indices]).reshape(-1, packed.shape[-1])]
            if vmrs is not None:
                # One abundance row per lane: a row's sectors share it, or have their own
                lane_vmrs = pad(vmrs[indices])
                inputs.append(np.repeat(lane_vmrs[:, 0], lanes, axis=0) if lane_vmrs.shape[1] == 1
                              else lane_vmrs.reshape(-1, lane_vmrs.shape[-1]))
            raw = np.asarray(self.forward(self.data, *inputs))
            raw = raw.reshape(capacity, lanes, -1)[:len(indices)]
            reasons[indices[np.any(raw[:, :, -2] != 0, axis=1)]] = UNBOUND
        result = self.finish(rows, raw, indices, values, clouds, fraction)
        for i in np.flatnonzero(fallback):
            result[i] = self.retriever._ln_like(
                rows[i], self.calc, None, self.fit, self.depths, self.errors, None, None,
                zero_opacities=self.zero_opacities)
        self.last_reasons = reasons
        return result
