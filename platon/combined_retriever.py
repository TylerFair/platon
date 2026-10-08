import numpy as np
import emcee
from dynesty import NestedSampler
import dynesty.utils
import copy
import warnings

from .psis import psisloo
from .transit_depth_calculator import TransitDepthCalculator
from .eclipse_depth_calculator import EclipseDepthCalculator
from .fit_info import FitInfo
from ._stellar_grid import resolve_legacy_het

from .constants import METRES_TO_UM, M_jup, R_jup, R_earth, M_earth, R_sun
from ._params import _UniformParam
from .errors import AtmosphereError
from ._output_writer import write_param_estimates_file
from .TP_profile import Profile
from .terminator import (TwoSectorTerminator, SECTOR_LABELS,
                         SECTOR_FRACTION, label_by_temperature)
from .retrieval_result import RetrievalResult
from .custom_dynesty_result import CustomDynestyResult

# Stellar heterogeneity parameters that can differ between transit visits
HET_PARAMS = ("T_het", "f_het", "T_het2", "f_het2")


class CombinedRetriever:
    _POINTWISE_CACHE_MAX_ENTRIES = 1024

    def pretty_print(self, fit_info):
        if not hasattr(self, "last_lnprob"):
            return
        
        line = "ln_prob={:.2e}\t".format(self.last_lnprob)
        for i, name in enumerate(fit_info.fit_param_names):            
            value = self.last_params[i]
            unit = ""
            if name == "Rs":
                value /= R_sun
                unit = "R_sun"
            if name == "Mp":
                value /= M_jup
                unit = "M_jup"
            if name == "Rp":
                value /= R_jup
                unit = "R_jup"
            if name == "T" or name.endswith(".T"):
                unit = "K"

            if name == "T" or name.endswith(".T"):
                format_str = "{:4.0f}"                
            elif abs(value) < 1e4: format_str = "{:.2f}"
            else: format_str = "{:.2e}"

            if name == "error_excess" or \
               name in self._offset_names(fit_info):
                unit = "ppm"
                value *= 1e6
            
            format_str = "{}=" + format_str + " " + unit + "\t"
            line += format_str.format(name, value)
            
        return line
    
    def _validate_params(self, fit_info, calculator):
        # This assumes that the valid parameter space is rectangular, so that
        # the bounds for each parameter can be treated separately. Unfortunately
        # there is no good way to validate Gaussian parameters, which have
        # infinite range.
        fit_info = copy.deepcopy(fit_info)
        validation_param = fit_info.all_params.get("validate_T_grid")
        validate_T_grid = True if validation_param is None else validation_param.best_guess
        terminator_param = fit_info.all_params.get("transit_terminator")
        terminator = None if terminator_param is None else \
            terminator_param.best_guess

        if terminator is not None:
            cloud_fraction = fit_info.all_params["cloud_fraction"].best_guess
            if cloud_fraction != 1 or \
               "cloud_fraction" in fit_info.fit_param_names:
                raise ValueError(
                    "cloud_fraction must be fixed at 1 for a "
                    "TwoSectorTerminator")
        
        if fit_info.all_params["log_k"].best_guess is None:
            # Not using Mie scattering
            if fit_info.all_params["log_number_density"].best_guess != -np.inf:
                raise ValueError("log number density must be -inf if not using Mie scattering")            
        else:
            if fit_info.all_params["log_scatt_factor"].best_guess != 0:
                raise ValueError("log scattering factor must be 0 if using Mie scattering")           
            
        
        for name in fit_info.fit_param_names:
            this_param = fit_info.all_params[name]
            if not isinstance(this_param, _UniformParam):
                continue

            if this_param.best_guess < this_param.low_lim \
               or this_param.best_guess > this_param.high_lim:
                raise ValueError(
                    "Value {} for {} not between low and high limits {}-{}".format(
                        this_param.best_guess, name, this_param.low_lim, this_param.high_lim))
            if this_param.low_lim >= this_param.high_lim:
                raise ValueError(
                    "low_lim for {} is higher than high_lim".format(name))

            if terminator is not None:
                continue

            for lim in [this_param.low_lim, this_param.high_lim]:
                this_param.best_guess = lim
                calculator._validate_params(
                    fit_info._get("T"),
                    fit_info._get("logZ"),
                    fit_info._get("CO_ratio"),
                    10**fit_info._get("log_cloudtop_P"),
                    validate_T_grid=validate_T_grid)

        if terminator is not None:
            best = [fit_info.all_params[name].best_guess
                    for name in fit_info.fit_param_names]
            params = fit_info._interpret_param_array(best)
            sectors = terminator.sectors_from_params(params)
            for sector in sectors:
                calculator._validate_params(
                    sector.profile.temperatures, params["logZ"],
                    params["CO_ratio"], sector.cloudtop_pressure,
                    validate_T_grid=validate_T_grid)
            for name in fit_info.fit_param_names:
                param = fit_info.all_params[name]
                if not isinstance(param, _UniformParam):
                    continue
                if name not in (
                        "logZ", "CO_ratio", "sector1.log_cloudtop_P",
                        "sector2.log_cloudtop_P"):
                    continue
                for limit in (param.low_lim, param.high_lim):
                    for label, sector in zip(SECTOR_LABELS, sectors):
                        logZ = limit if name == "logZ" else params["logZ"]
                        ratio = limit if name == "CO_ratio" else \
                            params["CO_ratio"]
                        cloudtop = 10**limit if \
                            name == f"{label}.log_cloudtop_P" else \
                            sector.cloudtop_pressure
                        calculator._validate_params(
                            sector.profile.temperatures, logZ, ratio, cloudtop,
                            validate_T_grid=validate_T_grid)

    @staticmethod
    def _offset_names(fit_info):
        """Names of the per-instrument offset parameters in fit_info."""
        names = set()
        for key in ("transit_offsets", "eclipse_offsets"):
            param = fit_info.all_params.get(key)
            if param is not None and param.best_guess is not None:
                names.update(param.best_guess.keys())
        return names

    @staticmethod
    def _visit_het_kwargs(params_dict, n_points):
        """T_het, f_het, T_het2 and f_het2 for the transit calculator.  They
        are scalars unless a visit in transit_visits overrides one (e.g.
        visit1.f_het), in which case each becomes an array with one value
        per data point."""
        values = {name: params_dict.get(name) for name in HET_PARAMS}
        visits = params_dict.get("transit_visits") or {}
        overrides = [(start, end, name, params_dict[f"{visit}.{name}"])
                     for visit, (start, end) in visits.items()
                     for name in HET_PARAMS
                     if params_dict.get(f"{visit}.{name}") is not None]
        if not overrides:
            return values
        # An unset temperature means no contrast; an unset fraction, none
        defaults = dict(T_het=params_dict["T_star"], f_het=0.,
                        T_het2=params_dict["T_star"], f_het2=0.)
        arrays = {name: np.full(n_points, defaults[name] if value is None
                                else value, dtype=np.float64)
                  for name, value in values.items()}
        for start, end, name, value in overrides:
            arrays[name][start:end] = value
        return arrays

    @staticmethod
    def _apply_offsets(depths, params_dict, kind):
        """Adds each named offset in params_dict[kind + "_offsets"] (kind is
        "transit" or "eclipse") to its index range of the calculated depths,
        in place."""
        offsets = params_dict.get(kind + "_offsets")
        if offsets is not None:
            for name, (start, end) in offsets.items():
                depths[start:end] += params_dict[name]

    @staticmethod
    def convert_clr_to_vmr(clrs):
        clr_bkg = -np.sum(clrs)
        clrs_with_bkg = np.append(clrs, clr_bkg)
        geometric_mean = 1 / np.sum(np.exp(clrs_with_bkg))
        vmrs_with_bkg = np.exp(clrs_with_bkg + np.log(geometric_mean))
        assert(np.around(np.sum(vmrs_with_bkg), decimals=5) == 1)
        return vmrs_with_bkg
    
    def _ln_like(self, params, transit_calc, eclipse_calc, fit_info, measured_transit_depths,
                 measured_transit_errors, measured_eclipse_depths,
                 measured_eclipse_errors, ret_best_fit=False,
                 lnlike_per_point=False,
                 zero_opacities=[]):

        if not fit_info._within_limits(params):
            return -np.inf

        params_dict = fit_info._interpret_param_array(params)
        
        Rp = params_dict["Rp"]
        T = params_dict["T"]
        logZ = params_dict["logZ"]
        CO_ratio = params_dict["CO_ratio"]
        scatt_factor = 10.0**params_dict["log_scatt_factor"]
        scatt_slope = params_dict["scatt_slope"]
        cloudtop_P = 10.0**params_dict["log_cloudtop_P"]
        error_excess = params_dict["error_excess"]
        Rs = params_dict["Rs"]
        Mp = params_dict["Mp"]
        T_star = params_dict["T_star"]
        stellar_kwargs = {name: params_dict.get(name, default) for name, default in (
            ('logg_star', 4.5), ('logg_het', None), ('logg_het2', None),
            ('feh_star', 0.), ('stellar_grid_only', False),
            ('stellar_blackbody', False), ('validate_T_grid', True))}
        het_kwargs = {name: params_dict.get(name) for name in HET_PARAMS}
        frac_scale_height = params_dict["frac_scale_height"]
        number_density = 10.0**params_dict["log_number_density"]
        part_size = 10.**params_dict["log_part_size"]
        P_quench = 10.** params_dict["log_P_quench"]
        CH4_mult = 10.**params_dict["log_CH4_mult"]
        cloud_fraction = params_dict.get("cloud_fraction", 1)
        transit_terminator = params_dict.get("transit_terminator")

        if cloud_fraction < 0 or cloud_fraction > 1:
            return -np.inf
        if transit_terminator is not None:
            # The sectors are unordered; from_params labels the colder one
            if cloud_fraction != 1 or \
               not 0 <= params_dict[SECTOR_FRACTION] <= 1:
                return -np.inf

        if params_dict["fit_vmr"]:
            assert(logZ is None and CO_ratio is None)
            gases = fit_info.gases
            vmrs = [10.**params_dict[f'log_{gas}'] for gas in gases[:-1]]
            vmrs.append(1 - np.sum(vmrs))
            if vmrs[-1] < 0: return -np.inf
        elif params_dict["fit_clr"]:
            assert(logZ is None and CO_ratio is None)
            gases = fit_info.gases
            clrs = [params_dict[f'clr_{gas}'] for gas in gases[:-1]]
            vmrs = self.convert_clr_to_vmr(clrs)
            if np.min(vmrs) < fit_info.clr_low_lim: return -np.inf
        else:
            vmrs = None
            gases = None
        if "n" in params_dict and params_dict["n"] is not None and "log_k" in params_dict:
            ri = params_dict["n"] - 1j * 10**params_dict["log_k"]
        else:
            ri = None
            
        if any(not np.isfinite(value) or value <= 0 for value in (Rs, Mp, Rp)) \
           or not np.isfinite(error_excess) or error_excess < 0:
            return -np.inf

        ln_likelihood = np.array([])
        calculated_transit_depths = None
        transit_info_dict = None
        calculated_eclipse_depths = None
        eclipse_info_dict = None
        
        try:
            if measured_transit_depths is not None:
                if transit_terminator is None:
                    transit_profile_type = params_dict.get(
                        "transit_profile_type", "isothermal")
                    if transit_profile_type == "isothermal" and \
                       params_dict.get("T_transit") is None and T is None:
                        raise ValueError(
                            "Must fit for T if using transit depths")
                    transit_profile = Profile.from_params_dict(
                        transit_profile_type, params_dict, suffix="_transit")
                    transit_profiles = (transit_profile,)
                else:
                    transit_profile = transit_terminator.from_params(
                        params_dict)
                    transit_profiles = (
                        transit_profile.cold.profile,
                        transit_profile.hot.profile)

                if any(np.any(np.isnan(p.temperatures))
                       for p in transit_profiles):
                    raise AtmosphereError("Invalid T/P profile")

                transit_wavelengths, calculated_transit_depths, transit_info_dict = transit_calc.compute_depths(
                    transit_profile, Rs, Mp, Rp, logZ, CO_ratio, CH4_mult, gases, vmrs,
                    custom_abundances=None,
                    scattering_factor=scatt_factor, scattering_slope=scatt_slope,
                    cloudtop_pressure=cloudtop_P,
                    cloud_fraction=cloud_fraction, T_star=T_star,
                    **self._visit_het_kwargs(
                        params_dict, len(measured_transit_depths)),
                    **stellar_kwargs,
                    frac_scale_height=frac_scale_height, number_density=number_density,
                    part_size=part_size, ri=ri, P_quench=P_quench, full_output=ret_best_fit, zero_opacities=zero_opacities)

                self._apply_offsets(calculated_transit_depths, params_dict, "transit")
                residuals = calculated_transit_depths - measured_transit_depths
                scaled_errors = np.sqrt(measured_transit_errors**2 + error_excess**2)
                ln_likelihood = np.append(ln_likelihood, -0.5 * (residuals**2 / scaled_errors**2 + np.log(2 * np.pi * scaled_errors**2)))
                
            if measured_eclipse_depths is not None:
                if params_dict["profile_type"] == "isothermal" and T is None:
                    raise ValueError(
                        "Must fit for T when profile_type is isothermal")

                t_p_profile = Profile.from_params_dict(
                    params_dict["profile_type"], params_dict)

                if np.any(np.isnan(t_p_profile.temperatures)):
                    raise AtmosphereError("Invalid T/P profile")

                eclipse_wavelengths, calculated_eclipse_depths, eclipse_info_dict = eclipse_calc.compute_depths(
                    t_p_profile, Rs, Mp, Rp, T_star, logZ, CO_ratio, CH4_mult, gases, vmrs,
                    custom_abundances=None,
                    scattering_factor=scatt_factor, scattering_slope=scatt_slope,
                    cloudtop_pressure=cloudtop_P,
                    **het_kwargs, **stellar_kwargs,
                    frac_scale_height=frac_scale_height, number_density=number_density,
                    part_size = part_size, ri=ri, P_quench=P_quench, full_output=ret_best_fit, zero_opacities=zero_opacities)
                self._apply_offsets(calculated_eclipse_depths, params_dict, "eclipse")
                residuals = calculated_eclipse_depths - measured_eclipse_depths
                scaled_errors = np.sqrt(measured_eclipse_errors**2 + error_excess**2)
                ln_likelihood = np.append(ln_likelihood, -0.5 * (residuals**2 / scaled_errors**2 + np.log(2 * np.pi * scaled_errors**2)))

        except AtmosphereError as e:
            return -np.inf
        
        self.last_params = params
        self.last_lnprob = fit_info._ln_prior(params) + ln_likelihood.sum()
        
        if ret_best_fit:
            # Attach the full (untruncated) T/P profiles so that downstream
            # code (RetrievalResult.random_TP_profiles, Plotter) has them
            if transit_info_dict is not None:
                transit_info_dict["full_TP_profile"] = \
                    self._profile_to_array(transit_profile)
            if eclipse_info_dict is not None:
                eclipse_info_dict["full_TP_profile"] = \
                    self._profile_to_array(t_p_profile)
            return calculated_transit_depths, transit_info_dict, calculated_eclipse_depths, eclipse_info_dict

        if lnlike_per_point:
            key = tuple(params)
            if (key not in self.params_to_lnlike and
                    len(self.params_to_lnlike) >= self._POINTWISE_CACHE_MAX_ENTRIES):
                # Most sampler trials never reach posterior reconstruction.
                # Bound retained pointwise arrays; evicted posterior draws
                # are recomputed by _collect_random_samples when needed.
                self.params_to_lnlike.pop(next(iter(self.params_to_lnlike)))
            self.params_to_lnlike[key] = ln_likelihood
            return ln_likelihood

        return ln_likelihood.sum()


    @staticmethod
    def _profile_to_array(profile):
        """Converts a Profile to a (2, N) array [P, T], or a TwoSectorTerminator
        to a (3, N) array [P, T_cold, T_hot].  Pressures are in Pa,
        temperatures in K."""
        if isinstance(profile, TwoSectorTerminator):
            cold = profile.cold.profile
            hot = profile.hot.profile
            return np.array([
                np.asarray(cold.pressures, dtype=np.float64),
                np.asarray(cold.temperatures, dtype=np.float64),
                np.asarray(hot.temperatures, dtype=np.float64)])
        return np.array([
            np.asarray(profile.pressures, dtype=np.float64),
            np.asarray(profile.temperatures, dtype=np.float64)])

    @staticmethod
    def _init_random_samples(retrieval_result):
        """Creates the empty lists that _record_random_sample fills."""
        retrieval_result.random_transit_depths = []
        retrieval_result.random_eclipse_depths = []
        retrieval_result.random_transit_TP_profiles = []
        retrieval_result.random_eclipse_TP_profiles = []
        # random_TP_profiles holds the dayside (eclipse) profiles when
        # eclipse data are fit, and the terminator (transit) profiles
        # otherwise.  It is kept for backwards compatibility.
        retrieval_result.random_TP_profiles = []
        retrieval_result.pointwise_lnlikes = []

    @staticmethod
    def _record_random_sample(retrieval_result, transit_info, eclipse_info,
                              pointwise_lnlike):
        """Appends the spectra and T/P profiles of one posterior sample to
        the random_* lists of retrieval_result.

        Transit T/P profiles are (2, N) arrays [P, T], or (3, N) arrays
        [P, T_cold, T_hot] for 1.5-D terminator retrievals."""
        if transit_info is not None:
            retrieval_result.random_transit_depths.append(
                transit_info["unbinned_depths"] *
                transit_info["unbinned_correction_factors"])
            retrieval_result.random_transit_TP_profiles.append(
                transit_info["full_TP_profile"])
        if eclipse_info is not None:
            retrieval_result.random_eclipse_depths.append(
                eclipse_info["unbinned_eclipse_depths"])
            retrieval_result.random_eclipse_TP_profiles.append(
                eclipse_info["full_TP_profile"])
        if eclipse_info is not None:
            retrieval_result.random_TP_profiles.append(
                eclipse_info["full_TP_profile"])
        elif transit_info is not None:
            retrieval_result.random_TP_profiles.append(
                transit_info["full_TP_profile"])
        retrieval_result.pointwise_lnlikes.append(pointwise_lnlike)

    @staticmethod
    def _check_data(fit_info, transit_bins, transit_depths, transit_errors,
                    eclipse_bins, eclipse_depths, eclipse_errors):
        """Catch inconsistent data, offsets, visits, and priors before any
        sampling starts."""
        params = fit_info.all_params
        fitted = set(fit_info.fit_param_names)

        def value(name):
            param = params.get(name)
            return None if param is None else param.best_guess

        counts = {}
        for kind, arrays in (
                ("transit", (transit_bins, transit_depths, transit_errors)),
                ("eclipse", (eclipse_bins, eclipse_depths, eclipse_errors))):
            given = [a is not None for a in arrays]
            if any(given) and not all(given):
                raise ValueError(
                    "Pass all of {0}_bins, {0}_depths and {0}_errors, or none "
                    "of them".format(kind))
            if not any(given):
                counts[kind] = 0
                continue
            lengths = {len(arrays[0]), len(arrays[1]), len(arrays[2])}
            if len(lengths) != 1:
                raise ValueError(
                    "{0}_bins, {0}_depths and {0}_errors have different lengths "
                    "({1}, {2}, {3})".format(kind, *map(len, arrays)))
            counts[kind] = len(arrays[1])

        groups = [("transit_offsets", counts["transit"]),
                  ("eclipse_offsets", counts["eclipse"]),
                  ("transit_visits", counts["transit"])]
        for key, n_points in groups:
            for name, (start, end) in (value(key) or {}).items():
                if n_points == 0:
                    raise ValueError("{} has {}, but there are no {} depths".format(
                        key, name, key.split("_")[0]))
                if end > n_points:
                    raise ValueError(
                        "{} {} covers rows {}-{}, but there are only {} {} "
                        "depths".format(key, name, start, end, n_points,
                                        key.split("_")[0]))

        # Offsets and error_excess are in units of depth, not ppm
        for name in fitted & (CombinedRetriever._offset_names(fit_info) |
                              {"error_excess"}):
            param = params[name]
            scale = max(abs(param.low_lim), abs(param.high_lim)) \
                if isinstance(param, _UniformParam) else param.std
            if scale >= 0.1:
                raise ValueError(
                    "The prior on {} reaches {:g}, but offsets and error_excess "
                    "are in units of depth, not ppm (100 ppm = 1e-4)".format(
                        name, scale))
        for key, n_points in groups[:2]:
            offsets = value(key) or {}
            free = [r for name, r in offsets.items() if name in fitted and
                    isinstance(params[name], _UniformParam)]
            covered = sum(end - start for start, end in free)
            if free and covered == n_points and "Rp" in fitted:
                warnings.warn(
                    "Every {} depth has a freely fitted offset, which is "
                    "degenerate with Rp; leave one dataset without an offset "
                    "or give the offsets Gaussian priors".format(
                        key.split("_")[0]), UserWarning, stacklevel=3)

        visits = value("transit_visits") or {}
        visit_rows = sum(end - start for start, end in visits.values())
        for name in HET_PARAMS:
            per_visit = [v for v in visits if "{}.{}".format(v, name) in fitted]
            if name in fitted and per_visit and len(per_visit) == len(visits) \
               and visit_rows == counts["transit"]:
                raise ValueError(
                    "{0} is fitted, but every visit fits its own <visit>.{0}, "
                    "so {0} has no effect".format(name))
        for T_name, f_name in (("T_het", "f_het"), ("T_het2", "f_het2")):
            T_set = value(T_name) is not None or T_name in fitted
            for owner in [None] + list(visits):
                prefix = "" if owner is None else owner + "."
                if prefix + f_name not in fitted:
                    continue
                visit_Ts = [v for v in visits
                            if "{}.{}".format(v, T_name) in fitted or
                            value("{}.{}".format(v, T_name)) is not None]
                if owner is None and visits and len(visit_Ts) == len(visits):
                    continue
                if not (T_set or prefix + T_name in fitted or
                        value(prefix + T_name) is not None):
                    raise ValueError(
                        "{0}{1} is fitted, but {2} is not set, so the "
                        "heterogeneity has the photosphere's temperature and "
                        "no effect; set {2} or fit it".format(prefix, f_name, T_name))

    def _make_calculators(self, fit_info, transit_bins, eclipse_bins,
                          include_condensation, rad_method):
        """Build the same forward models for every inference backend."""
        grid_param = fit_info.all_params.get("stellar_grid")
        stellar_grid = "newera" if grid_param is None else grid_param.best_guess
        options = dict(include_condensation=include_condensation,
                       method=rad_method, stellar_grid=stellar_grid)
        transit_calc = eclipse_calc = None
        if transit_bins is not None:
            transit_calc = TransitDepthCalculator(**options)
            transit_calc.change_wavelength_bins(transit_bins)
            self._validate_params(fit_info, transit_calc)
        if eclipse_bins is not None:
            eclipse_calc = EclipseDepthCalculator(**options)
            eclipse_calc.change_wavelength_bins(eclipse_bins)
        return transit_calc, eclipse_calc

    @staticmethod
    def _sum_pointwise_lnlikes(values):
        if np.isscalar(values):
            assert values == -np.inf
            return -np.inf
        return values.sum()

    def _sampler_functions(self, fit_info, transit_calc, eclipse_calc,
                           transit_depths, transit_errors,
                           eclipse_depths, eclipse_errors, zero_opacities,
                           print_evaluations=True):
        """Adapt the shared prior and likelihood to nested samplers."""
        def log_likelihood(params):
            values = self._ln_like(
                params, transit_calc, eclipse_calc, fit_info,
                transit_depths, transit_errors, eclipse_depths, eclipse_errors,
                zero_opacities=zero_opacities, lnlike_per_point=True)
            ln_like = self._sum_pointwise_lnlikes(values)
            if print_evaluations and np.random.randint(100) == 0:
                print("\nEvaluated params: {}".format(self.pretty_print(fit_info)))
            return ln_like

        return fit_info._from_unit_interval_array, log_likelihood

    def _collect_random_samples(self, retrieval_result, equal_samples,
                                num_final_samples, transit_calc, eclipse_calc,
                                fit_info, transit_depths, transit_errors,
                                eclipse_depths, eclipse_errors, zero_opacities):
        """Recompute posterior spectra with the settings used for inference."""
        self._init_random_samples(retrieval_result)
        likelihood_args = (transit_calc, eclipse_calc, fit_info,
                           transit_depths, transit_errors,
                           eclipse_depths, eclipse_errors)
        for params in equal_samples[:num_final_samples]:
            best = self._ln_like(
                params, *likelihood_args, zero_opacities=zero_opacities,
                ret_best_fit=True)
            if np.isscalar(best):
                assert best == -np.inf
                continue
            pointwise = self.params_to_lnlike.get(tuple(params))
            if pointwise is None:
                pointwise = self._ln_like(
                    params, *likelihood_args, zero_opacities=zero_opacities,
                    lnlike_per_point=True)
            self._record_random_sample(
                retrieval_result, best[1], best[3], pointwise)
        if len(retrieval_result.pointwise_lnlikes) < 2:
            # PSIS requires multiple posterior draws. Keep an otherwise valid
            # retrieval usable when reconstruction was disabled or only one
            # posterior draw survived, and mark diagnostics unavailable.
            n_points = sum(len(depths) for depths in (transit_depths, eclipse_depths)
                           if depths is not None)
            retrieval_result.loo_total = np.nan
            retrieval_result.loos = np.full(n_points, np.nan)
            retrieval_result.loo_ks = np.full(n_points, np.inf)
            return
        retrieval_result.loo_total, retrieval_result.loos, \
            retrieval_result.loo_ks = psisloo(
                np.array(retrieval_result.pointwise_lnlikes))

    def _ln_prob(self, params, transit_calc, eclipse_calc, fit_info, measured_transit_depths,
                 measured_transit_errors, measured_eclipse_depths,
                 measured_eclipse_errors, zero_opacities=[]):
        
        lnlike_per_point = self._ln_like(params, transit_calc, eclipse_calc, fit_info, measured_transit_depths,
                                measured_transit_errors, measured_eclipse_depths,
                                measured_eclipse_errors, zero_opacities=zero_opacities, lnlike_per_point=True)
        
        return fit_info._ln_prior(params) + self._sum_pointwise_lnlikes(lnlike_per_point)


    def run_emcee(self, transit_bins, transit_depths, transit_errors,
                  eclipse_bins, eclipse_depths, eclipse_errors,
                  fit_info, nwalkers=50,
                  nsteps=1000, include_condensation=True,
                  rad_method="xsec",
                  num_final_samples=100, zero_opacities=[]):
        '''Runs affine-invariant MCMC to retrieve atmospheric parameters.

        Parameters
        ----------
        transit_bins : array_like, shape (N,2)
            Wavelength bins, where wavelength_bins[i][0] is the start
            wavelength and wavelength_bins[i][1] is the end wavelength for
            bin i.
        transit_depths : array_like, length N
            Measured transit depths for the specified wavelength bins
        transit_errors : array_like, length N
            Errors on the aforementioned transit depths
        eclipse_bins : array_like, shape (N,2)
            Wavelength bins, where wavelength_bins[i][0] is the start
            wavelength and wavelength_bins[i][1] is the end wavelength for
            bin i.
        eclipse_depths : array_like, length N
            Measured eclipse depths for the specified wavelength bins
        eclipse_errors : array_like, length N
            Errors on the aforementioned eclipse depths
        fit_info : :class:`.FitInfo` object
            Tells the method what parameters to
            freely vary, and in what range those parameters can vary. Also
            sets default values for the fixed parameters.
        nwalkers : int, optional
            Number of walkers to use
        nsteps : int, optional
            Number of steps that the walkers should walk for
        include_condensation : bool, optional
            When determining atmospheric abundances, whether to include
            condensation.
        rad_method : string, optional
            "xsec" for opacity sampling (correlated-k is no longer supported)
        zero_opacities : list of strings
            List of molecules to zero opacities for

        Returns
        -------
        result : RetrievalResult object
        '''
        self.params_to_lnlike = {}
        initial_positions = fit_info._generate_rand_param_arrays(nwalkers)
        self._check_data(fit_info, transit_bins, transit_depths,
                         transit_errors, eclipse_bins, eclipse_depths,
                         eclipse_errors)
        transit_calc, eclipse_calc = self._make_calculators(
            fit_info, transit_bins, eclipse_bins,
            include_condensation, rad_method)

        sampler = emcee.EnsembleSampler(
            nwalkers, fit_info._get_num_fit_params(), self._ln_prob,
            args=(transit_calc, eclipse_calc, fit_info, transit_depths, transit_errors,
                                 eclipse_depths, eclipse_errors, zero_opacities))

        for i, result in enumerate(sampler.sample(
                initial_positions, iterations=nsteps)):
            if (i + 1) % 10 == 0:
                print("Step {}: {}".format(i + 1, self.pretty_print(fit_info)))

        best_params_arr = sampler.flatchain[np.argmax(
            sampler.flatlnprobability)]
        
        divisors, new_labels = self._write_estimates(
            fit_info, sampler.flatchain, best_params_arr,
            np.max(sampler.flatlnprobability))

        best_fit_transit_depths, best_fit_transit_info, best_fit_eclipse_depths, best_fit_eclipse_info = self._ln_like(
            best_params_arr, transit_calc, eclipse_calc, fit_info,
            transit_depths, transit_errors,
            eclipse_depths, eclipse_errors, zero_opacities=zero_opacities, ret_best_fit=True)
        retrieval_result = RetrievalResult(
            {"best_fit_params": best_params_arr,
                "acceptance_fraction": sampler.acceptance_fraction,
             "chain": sampler.chain,
             "flatchain": sampler.flatchain,
             "lnprobability": sampler.lnprobability,
             "flatlnprobability": sampler.flatlnprobability},             
            "emcee", best_params_arr,
            transit_bins, transit_depths, transit_errors,
            eclipse_bins, eclipse_depths, eclipse_errors,
            best_fit_transit_depths, best_fit_transit_info,
            best_fit_eclipse_depths, best_fit_eclipse_info,
            fit_info, divisors, new_labels)
        equal_samples = np.copy(sampler.flatchain)
        np.random.shuffle(equal_samples)
        self._collect_random_samples(
            retrieval_result, equal_samples, num_final_samples,
            transit_calc, eclipse_calc, fit_info,
            transit_depths, transit_errors, eclipse_depths, eclipse_errors,
            zero_opacities)
        return retrieval_result

    def _write_estimates(self, fit_info, samples, best_params, best_lnprob):
        """Write BestFit.txt, with two-sector parameters labelled cold and
        hot by temperature.  Returns the divisors and labels of its columns."""
        names, samples = label_by_temperature(fit_info, samples)
        _, best = label_by_temperature(fit_info, [best_params])
        divisors, labels = self._get_divisors_labels(
            np.median(samples, axis=0), names)
        write_param_estimates_file(
            samples / divisors, best[0] / divisors, best_lnprob, labels)
        return divisors, labels

    def _get_divisors_labels(self, medians, labels):
        divisors = np.ones(len(labels))
        new_labels = np.copy(labels)
        
        for i, l in enumerate(labels):            
            if l == "Rs":
                divisors[i] = R_sun
                new_labels[i] = "R_star/R_sun"
            if l == "Rp":
                if medians[i] > 0.5 * R_jup:
                    divisors[i] = R_jup
                    new_labels[i] = "R_p/R_j"
                else:
                    divisors[i] = R_earth
                    new_labels[i] = "R_p/R_e"
            if l == "Mp":
                if medians[i] > 0.1 * M_jup:
                    divisors[i] = M_jup
                    new_labels[i] = "M_p/M_j"
                else:
                    divisors[i] = M_earth
                    new_labels[i] = "M_p/M_e"
                    
        return divisors, new_labels
    
    def run_dynesty(self, transit_bins, transit_depths, transit_errors,
                      eclipse_bins, eclipse_depths, eclipse_errors,
                      fit_info,
                      include_condensation=True, rad_method="xsec",
                      maxiter=None, maxcall=None, nlive=250,
                      num_final_samples=100, zero_opacities=[],
                      **dynesty_kwargs):
        '''Runs nested sampling to retrieve atmospheric parameters.

        Parameters
        ----------
        transit_bins : array_like, shape (N,2)
            Wavelength bins, where wavelength_bins[i][0] is the start
            wavelength and wavelength_bins[i][1] is the end wavelength for
            bin i.
        transit_depths : array_like, length N
            Measured transit depths for the specified wavelength bins
        transit_errors : array_like, length N
            Errors on the aforementioned transit depths
        eclipse_bins : array_like, shape (N,2)
            Wavelength bins, where wavelength_bins[i][0] is the start
            wavelength and wavelength_bins[i][1] is the end wavelength for
            bin i.
        eclipse_depths : array_like, length N
            Measured eclipse depths for the specified wavelength bins
        eclipse_errors : array_like, length N
            Errors on the aforementioned eclipse depths
        fit_info : :class:`.FitInfo` object
            Tells us what parameters to
            freely vary, and in what range those parameters can vary. Also
            sets default values for the fixed parameters.
        include_condensation : bool, optional
            When determining atmospheric abundances, whether to include
            condensation.
        rad_method : string, optional
            "xsec" for opacity sampling (correlated-k is no longer supported)       
        nlive : int
            Number of live points to use for nested sampling
        zero_opacities : list of strings                                                                                                                                                                   
            List of molecules to zero opacities for
        **dynesty_kwargs : keyword arguments to pass to dynesty's NestedSampler

        Returns
        -------
        result : RetrievalResult object
        '''        
        self.params_to_lnlike = {}
        self._check_data(fit_info, transit_bins, transit_depths,
                         transit_errors, eclipse_bins, eclipse_depths,
                         eclipse_errors)
        transit_calc, eclipse_calc = self._make_calculators(
            fit_info, transit_bins, eclipse_bins,
            include_condensation, rad_method)

        transform_prior, dynesty_ln_like = self._sampler_functions(
            fit_info, transit_calc, eclipse_calc,
            transit_depths, transit_errors, eclipse_depths, eclipse_errors,
            zero_opacities)

        num_dim = fit_info._get_num_fit_params()
        sampler = NestedSampler(dynesty_ln_like, transform_prior, num_dim, bound='multi', nlive=nlive, **dynesty_kwargs)
        sampler.run_nested(maxiter=maxiter, maxcall=maxcall)
        result = CustomDynestyResult(sampler.results)
        result.logp = result.logl + np.array([fit_info._ln_prior(params) for params in result.samples])
        best_params_arr = result.samples[np.argmax(result.logp)]

        normalized_weights = np.exp(result.logwt - np.max(result.logwt))
        normalized_weights /= np.sum(normalized_weights)
        result.weights = normalized_weights                                
        equal_samples = dynesty.utils.resample_equal(result.samples, result.weights)
        np.random.shuffle(equal_samples)

        divisors, new_labels = self._write_estimates(
            fit_info, equal_samples, best_params_arr, np.max(result.logp))
        
        best_fit_transit_depths, best_fit_transit_info, best_fit_eclipse_depths, best_fit_eclipse_info = self._ln_like(
            best_params_arr, transit_calc, eclipse_calc, fit_info,
            transit_depths, transit_errors,
            eclipse_depths, eclipse_errors, zero_opacities=zero_opacities, ret_best_fit=True)

        retrieval_result = RetrievalResult(
            result, "dynesty", best_params_arr,
            transit_bins, transit_depths, transit_errors,
            eclipse_bins, eclipse_depths, eclipse_errors,
            best_fit_transit_depths, best_fit_transit_info,
            best_fit_eclipse_depths, best_fit_eclipse_info,
            fit_info, divisors, new_labels)

        self._collect_random_samples(
            retrieval_result, equal_samples, num_final_samples,
            transit_calc, eclipse_calc, fit_info,
            transit_depths, transit_errors, eclipse_depths, eclipse_errors,
            zero_opacities)
        return retrieval_result


    def run_multinest(self, transit_bins, transit_depths, transit_errors,
                      eclipse_bins, eclipse_depths, eclipse_errors,
                      fit_info,
                      include_condensation=True, rad_method="xsec",
                      maxiter=None, maxcall=None, nlive=250,
                      num_final_samples=100, zero_opacities=[],
                      multinest_kwargs={}):
        """multinest_kwargs are forwarded to pymultinest.solve/run (e.g.
        sampling_efficiency, const_efficiency_mode, evidence_tolerance,
        multimodal, outputfiles_basename)."""
        import pymultinest
        
        self.params_to_lnlike = {}
        self._check_data(fit_info, transit_bins, transit_depths,
                         transit_errors, eclipse_bins, eclipse_depths,
                         eclipse_errors)
        transit_calc, eclipse_calc = self._make_calculators(
            fit_info, transit_bins, eclipse_bins,
            include_condensation, rad_method)

        transform_prior, multinest_ln_like = self._sampler_functions(
            fit_info, transit_calc, eclipse_calc,
            transit_depths, transit_errors, eclipse_depths, eclipse_errors,
            zero_opacities)

        num_dim = fit_info._get_num_fit_params()
        solve_kwargs = dict(
            verbose=True, resume=False, n_live_points=nlive,
            outputfiles_basename="multinest_" + str(np.random.randint(1000)))
        solve_kwargs.update(multinest_kwargs)
        basename = solve_kwargs["outputfiles_basename"]
        result = pymultinest.solve(LogLikelihood=multinest_ln_like, Prior=transform_prior,
                                   n_dims=num_dim, **solve_kwargs)
        a = pymultinest.Analyzer(outputfiles_basename=basename, n_params=num_dim)
        data = a.get_data()
        result["samples"] = data[:,2:]
        result["logp"] = np.log(data[:,0])
        result["logl"] = -0.5 * data[:,1]
        best_params_arr = result["samples"][np.argmax(result["logp"])]
        
        equal_samples = a.get_equal_weighted_posterior()[:,:-1]
        np.random.shuffle(equal_samples)
        result["equal_samples"] = equal_samples
        
        divisors, new_labels = self._write_estimates(
            fit_info, equal_samples, best_params_arr, np.max(result["logp"]))

        best_fit_transit_depths, best_fit_transit_info, best_fit_eclipse_depths, best_fit_eclipse_info = self._ln_like(
            best_params_arr,
            transit_calc, eclipse_calc, fit_info,
            transit_depths, transit_errors,
            eclipse_depths, eclipse_errors, zero_opacities=zero_opacities, ret_best_fit=True)

        retrieval_result = RetrievalResult(
            result, "pymultinest", best_params_arr,
            transit_bins, transit_depths, transit_errors,
            eclipse_bins, eclipse_depths, eclipse_errors,
            best_fit_transit_depths, best_fit_transit_info,
            best_fit_eclipse_depths, best_fit_eclipse_info,
            fit_info, divisors, new_labels)

        self._collect_random_samples(
            retrieval_result, equal_samples, num_final_samples,
            transit_calc, eclipse_calc, fit_info,
            transit_depths, transit_errors, eclipse_depths, eclipse_errors,
            zero_opacities)
        return retrieval_result

    def run_nautilus(self, transit_bins, transit_depths, transit_errors,
                     eclipse_bins, eclipse_depths, eclipse_errors,
                     fit_info, include_condensation=True, rad_method="xsec",
                     n_live=2000, n_eff=10000, n_networks=16,
                     discard_exploration=True, verbose=True,
                     num_final_samples=100, zero_opacities=(),
                     **nautilus_kwargs):
        """Run optional Nautilus nested sampling.

        Install support with ``pip install "platon[nautilus]"`` or
        ``pip install nautilus-sampler``. Extra keyword arguments are passed
        directly to :class:`nautilus.Sampler`.
        """
        try:
            from nautilus import Sampler
        except ImportError as error:
            raise ImportError(
                'run_nautilus requires the optional "nautilus-sampler" '
                'package. Install it with pip install "platon[nautilus]" '
                'or pip install nautilus-sampler.') from error

        self.params_to_lnlike = {}
        self._check_data(fit_info, transit_bins, transit_depths,
                         transit_errors, eclipse_bins, eclipse_depths,
                         eclipse_errors)
        transit_calc, eclipse_calc = self._make_calculators(
            fit_info, transit_bins, eclipse_bins,
            include_condensation, rad_method)

        transform_prior, nautilus_ln_like = self._sampler_functions(
            fit_info, transit_calc, eclipse_calc,
            transit_depths, transit_errors, eclipse_depths, eclipse_errors,
            zero_opacities, print_evaluations=False)

        sampler = Sampler(
            transform_prior, nautilus_ln_like,
            n_dim=fit_info._get_num_fit_params(),
            n_live=n_live, n_networks=n_networks, **nautilus_kwargs)
        success = sampler.run(
            n_eff=n_eff, discard_exploration=discard_exploration,
            verbose=verbose)

        samples, log_weights, logl = sampler.posterior()
        samples = np.asarray(samples)
        log_weights = np.asarray(log_weights)
        logl = np.asarray(logl)
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= np.sum(weights)
        logp = logl + np.array(
            [fit_info._ln_prior(params) for params in samples])
        best_params_arr = samples[np.argmax(logp)]

        equal_samples = dynesty.utils.resample_equal(samples, weights)
        np.random.shuffle(equal_samples)
        divisors, new_labels = self._write_estimates(
            fit_info, equal_samples, best_params_arr, np.max(logp))

        best = self._ln_like(
            best_params_arr, transit_calc, eclipse_calc, fit_info,
            transit_depths, transit_errors, eclipse_depths, eclipse_errors,
            zero_opacities=zero_opacities, ret_best_fit=True)
        result = {
            "samples": samples,
            "weights": weights,
            "logw": log_weights,
            "logl": logl,
            "logp": logp,
            "logz": np.atleast_1d(sampler.log_z),
            "n_eff": sampler.n_eff,
            "success": success,
        }
        retrieval_result = RetrievalResult(
            result, "nautilus", best_params_arr,
            transit_bins, transit_depths, transit_errors,
            eclipse_bins, eclipse_depths, eclipse_errors,
            best[0], best[1], best[2], best[3],
            fit_info, divisors, new_labels)

        self._collect_random_samples(
            retrieval_result, equal_samples, num_final_samples,
            transit_calc, eclipse_calc, fit_info,
            transit_depths, transit_errors, eclipse_depths, eclipse_errors,
            zero_opacities)
        return retrieval_result
        

    @staticmethod
    def get_default_fit_info(Rs, Mp, Rp, T=None, logZ=0, CO_ratio=0.53, log_CH4_mult=0,
                             free_retrieval=False,
                             log_cloudtop_P=np.inf, cloud_fraction=1,
                             log_scatt_factor=0,
                             scatt_slope=4, error_excess=0, T_star=None,
                             T_het=None, f_het=None,
                             frac_scale_height=1,
                             log_number_density=-np.inf, log_part_size=-6,
                             n=None, log_k=-np.inf,
                             log_P_quench=-99,
                             transit_offsets=None, eclipse_offsets=None,
                             fit_vmr=False, fit_clr=False,
                             profile_type = 'isothermal',
                             transit_profile_type = 'isothermal',
                             transit_terminator=None,
                             stellar_grid='newera', stellar_blackbody=False,
                             stellar_grid_only=False, T_het2=None, f_het2=None,
                             logg_star=4.5, logg_het=None, logg_het2=None,
                             feh_star=0., transit_visits=None,
                             validate_T_grid=True, T_spot=None,
                             spot_cov_frac=None,
                             **profile_kwargs):
        '''Get a :class:`.FitInfo` object filled with best guess values.  A few
        parameters are required, but others can be set to default values if you
        do not want to specify them.  Physical parameters use SI, except stellar logg
        (log10 cgs) and [Fe/H] (dex).  For
        information on the parameters not described below, see the documentation
        for :func:`~platon.transit_depth_calculator.TransitDepthCalculator.compute_depths` and :func:`~platon.eclipse_depth_calculator.EclipseDepthCalculator.compute_depths`

        Parameters
        ----------
        cloud_fraction : float
            Fraction of the terminator covered by clouds, between 0 and 1.
            Only affects transit depths; see
            :func:`~platon.transit_depth_calculator.TransitDepthCalculator.compute_depths`
        validate_T_grid : bool
            Require atmospheric temperatures within the opacity grid. Set
            False for retrievals intentionally clamping opacity lookup to
            the grid boundaries.
        stellar_grid : str or pathlib.Path
            'newera' (default), 'phoenix', or a custom stellar-grid file, used by every
            sampler for transit and eclipse calculations.
        T_star : float, optional
            Photosphere temperature (K).  Required for stellar contamination.
        T_het, f_het : float, optional
            Temperature (K) and covering fraction of an unocculted stellar
            heterogeneity, which may be cooler or hotter than the
            photosphere.  T_spot and spot_cov_frac are accepted as older
            names.
        T_het2, f_het2 : float, optional
            A second heterogeneity; f_het + f_het2 must be at most 1.
        logg_star, logg_het, logg_het2, feh_star : float
            Stellar log10 gravity (cgs) and shared [Fe/H] (dex).  The
            heterogeneities inherit logg_star; defaults are 4.5 and solar
            metallicity.  These can be given priors like other parameters.
        transit_visits : dict, optional
            Visits whose stellar heterogeneity may differ, as a dict mapping
            each visit name to the (start, end) indices of its transit data,
            e.g. {"visit1": (0, 120), "visit2": (120, 176)} (see
            :func:`~platon.observations.load_spectra`).  Each visit gets the
            parameters "<visit>.T_het", "<visit>.f_het", "<visit>.T_het2" and
            "<visit>.f_het2", which default to None, meaning "use T_het,
            f_het, ...".  Fit whichever should vary between visits, e.g.
            fit_info.add_uniform_fit_param("visit1.f_het", 0, 0.3); the
            photosphere (T_star, logg_star, feh_star) is always shared.
        stellar_grid_only : bool, optional
            Reject out-of-grid stellar temperatures (default False).
            Boundary-normalized Planck wavelength tails remain enabled.
        stellar_blackbody : bool, optional
            Use blackbody spectra for all stellar components (default False).
        n : float
            Real component of the refractive index of haze particles. Set to
            None to disable Mie scattering
        log_k : float
            log10 of the imaginary component of the refractive index of haze
            particles.  Set to -np.inf for k=0
        error_excess : float
            Extra error, in units of transit/eclipse depth, added in
            quadrature to every measured error: the likelihood uses
            sqrt(error**2 + error_excess**2).  Fit for it (e.g. with a
            uniform prior from 0 to 1e-4) to account for underestimated
            errors or scatter the model cannot explain.
        transit_offsets : dict, optional
            Per-instrument offsets for transit data, as a dict mapping each
            offset parameter name to the (start, end) indices of the data it
            applies to, e.g. {"offset_niriss": (0, 1010),
            "offset_nrs1": (1010, 2397)}.  Each name becomes a parameter
            with a default value of 0, which can be fit for like any other
            (e.g. fit_info.add_uniform_fit_param("offset_niriss", -2e-4, 2e-4)).
            A positive offset means the observed depths are decreased
            before comparing to the model.  Each range must satisfy
            0 <= start < end, and ranges may not overlap.  Leave one
            instrument without an offset to serve as the reference.
        eclipse_offsets : dict, optional
            Same as above, but for eclipse depths.  A name may appear in
            both transit_offsets and eclipse_offsets to share one offset.
        profile_type : string
            "isothermal", "parametric" (Madhusudhan & Seager 2009),
            "radiative_solution" (Line et al 2013), or "guillot" (Guillot
            2010) T/P profile parameterizations.  This profile applies to the dayside, and is
            used for eclipse depths.
        transit_profile_type : string
            Same options as profile_type.  This profile applies to the
            terminator, and is used for transit depths.  Its parameters are
            the profile_kwargs suffixed with "_transit" (e.g. T0_transit,
            T3_transit); any parameter without a "_transit" version falls
            back to the unsuffixed (dayside) value.  For "isothermal", the
            temperature is T_transit, falling back to T.
        transit_terminator : TwoSectorTerminator, optional
            A cold and hot terminator template for a 1.5-D transit retrieval.
            Its sector values are added to the returned FitInfo under neutral
            names, sector1.<name> (from the cold template) and
            sector2.<name> (from the hot one), plus sector1.fraction.  Fit
            both sectors with the same independent priors, e.g.
            add_uniform_fit_param("sector1.T", 300, 3000) and likewise for
            sector2.T; the results name each sample's colder sector "cold"
            (see :func:`~platon.terminator.label_by_temperature`).  For
            Guillot sectors, the shared T_star, Rs, a, Mp, Rp, log_k_th, and
            T_int are taken from the template; any of these also passed here
            must agree with it.
        profile_kwargs : kwargs
            T/P profile arguments.  For "isothermal": T (K).  For "parametric":
            T0, log_P1, alpha1, alpha2, log_P3, T3. Pressures are
            log10(Pa). For "radiative_solution":
            T_star, Rs, a, Mp, Rp, beta, log_k_th, log_gamma, log_gamma2,
            alpha, and T_int (optional).  We recommend that T_star, Rs, a, and
            Mp be fixed, and that T_int be omitted (which sets it to 100 K).
            For "guillot": the same as "radiative_solution", without
            log_gamma2 and alpha.
            

        Returns
        -------
        fit_info : :class:`.FitInfo` object
            This object is used to indicate which parameters to fit for, which
            to fix, and what values all parameters should take.'''
        all_variables = locals().copy()
        del all_variables["profile_kwargs"]
        all_variables.update(profile_kwargs)
        all_variables["T_het"], all_variables["f_het"] = resolve_legacy_het(
            T_het, f_het, all_variables.pop("T_spot"),
            all_variables.pop("spot_cov_frac"))
        if transit_terminator is not None:
            if not isinstance(transit_terminator, TwoSectorTerminator):
                raise TypeError(
                    "transit_terminator must be a TwoSectorTerminator")
            for name, value in \
                    transit_terminator.retrieval_defaults().items():
                current = all_variables.get(name)
                if name != "transit_terminator" and current is not None \
                   and not np.isclose(current, value):
                    raise ValueError(
                        "{}={} conflicts with the transit_terminator's "
                        "{}={}".format(name, current, name, value))
                all_variables[name] = value

        offset_names = set()
        for kind, offsets in (("transit", transit_offsets),
                              ("eclipse", eclipse_offsets)):
            if offsets is None:
                continue
            for name, index_range in offsets.items():
                if name in all_variables and name not in offset_names:
                    raise ValueError(
                        "Offset name {} conflicts with an existing "
                        "parameter".format(name))
                if len(index_range) != 2 or \
                   not 0 <= index_range[0] < index_range[1]:
                    raise ValueError(
                        "Range for offset {} must be (start, end) with "
                        "0 <= start < end".format(name))
                all_variables[name] = 0
                offset_names.add(name)

            sorted_ranges = sorted(offsets.items(), key=lambda kv: kv[1][0])
            for (name1, range1), (name2, range2) in zip(
                    sorted_ranges, sorted_ranges[1:]):
                if range1[1] > range2[0]:
                    raise ValueError(
                        "{} offsets {} {} and {} {} overlap".format(
                            kind, name1, tuple(range1), name2, tuple(range2)))
        
        for visit, index_range in (transit_visits or {}).items():
            if not isinstance(visit, str) or not visit or "." in visit \
               or visit != visit.strip():
                raise ValueError(
                    "Visit name {!r} must be a nonempty string without dots "
                    "or surrounding spaces".format(visit))
            if len(index_range) != 2 or \
               not 0 <= index_range[0] < index_range[1]:
                raise ValueError(
                    "Range for visit {} must be (start, end) with "
                    "0 <= start < end".format(visit))
            for name in HET_PARAMS:
                if "{}.{}".format(visit, name) in all_variables:
                    raise ValueError(
                        "Visit name {} conflicts with existing parameter "
                        "{}.{}".format(visit, visit, name))
                all_variables["{}.{}".format(visit, name)] = None
        if transit_visits:
            sorted_ranges = sorted(transit_visits.items(),
                                   key=lambda kv: kv[1][0])
            for (name1, range1), (name2, range2) in zip(
                    sorted_ranges, sorted_ranges[1:]):
                if range1[1] > range2[0]:
                    raise ValueError(
                        "Visits {} {} and {} {} overlap".format(
                            name1, tuple(range1), name2, tuple(range2)))
            if T_star is None:
                raise ValueError(
                    "transit_visits describes stellar heterogeneities, so "
                    "T_star must be set")

        fit_info = FitInfo(all_variables)
        return fit_info
