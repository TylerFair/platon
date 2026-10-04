import numpy as np
import emcee
from dynesty import NestedSampler
import dynesty.utils
import copy
import sys

from .psis import psisloo
from .transit_depth_calculator import TransitDepthCalculator
from .eclipse_depth_calculator import EclipseDepthCalculator
from .fit_info import FitInfo
from ._offsets import apply_offsets, offset_parameter_names, offset_labels

from .constants import METRES_TO_UM, M_jup, R_jup, R_earth, M_earth, R_sun
from ._params import _UniformParam
from .errors import AtmosphereError
from ._output_writer import write_param_estimates_file
from .TP_profile import Profile
from .terminator import TwoSectorTerminator
from .retrieval_result import RetrievalResult
from .custom_dynesty_result import CustomDynestyResult

class CombinedRetriever:
    _POINTWISE_CACHE_MAX_ENTRIES = 1024

    def pretty_print(self, fit_info):
        if not hasattr(self, "last_lnprob"):
            return
        
        offsets = offset_parameter_names(fit_info)
        line = "ln_prob={:.2e}\t".format(self.last_lnprob)
        for i, name in enumerate(fit_info.fit_param_names):            
            value = self.last_params[i]
            unit = "ppm" if name in offsets else ""
            is_temperature = name not in offsets and (
                name == "T" or name.endswith(".T") or name.endswith(".T_irr"))
            if name not in offsets:
                if name == "Rs":
                    value /= R_sun
                    unit = "R_sun"
                elif name == "Mp":
                    value /= M_jup
                    unit = "M_jup"
                elif name == "Rp":
                    value /= R_jup
                    unit = "R_jup"
                elif is_temperature:
                    unit = "K"
            if is_temperature:
                format_str = "{:4.0f}"
            elif abs(value) < 1e4:
                format_str = "{:.2f}"
            else:
                format_str = "{:.2e}"

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
            rebuilt = terminator.from_params(params, params["Mp"], params["Rp"])
            for sector in (rebuilt.cold, rebuilt.hot):
                calculator._validate_params(
                    sector.profile.temperatures, params["logZ"],
                    params["CO_ratio"], sector.cloudtop_pressure,
                    validate_T_grid=validate_T_grid)
            for name in fit_info.fit_param_names:
                param = fit_info.all_params[name]
                if not isinstance(param, _UniformParam):
                    continue
                if name not in (
                        "logZ", "CO_ratio", "cold.log_cloudtop_P",
                        "hot.log_cloudtop_P"):
                    continue
                for limit in (param.low_lim, param.high_lim):
                    for label, sector in (
                            ("cold", rebuilt.cold), ("hot", rebuilt.hot)):
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
        error_multiple = params_dict["error_multiple"]
        Rs = params_dict["Rs"]
        Mp = params_dict["Mp"]
        T_star = params_dict["T_star"]
        T_spot = params_dict["T_spot"]
        spot_cov_frac = params_dict["spot_cov_frac"]
        forward_kwargs = {name: params_dict.get(name, default) for name, default in (
            ('T_fac', None), ('fac_cov_frac', None), ('logg_phot', 4.5),
            ('logg_spot', None), ('logg_fac', None), ('feh', 0.),
            ('stellar_grid_only', False), ('stellar_blackbody', False),
            ('validate_T_grid', True))}
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
            if cloud_fraction != 1:
                return -np.inf
            order_name = transit_terminator.order_parameter
            if params_dict[f"cold.{order_name}"] > \
               params_dict[f"hot.{order_name}"]:
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
            
        if any(not np.isfinite(value) or value <= 0
               for value in (Rs, Mp, Rp, error_multiple)):
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
                    transit_profile = Profile()
                    transit_profile.set_from_params_dict(
                        transit_profile_type, params_dict, suffix="_transit")
                    transit_profiles = (transit_profile,)
                else:
                    transit_profile = transit_terminator.from_params(
                        params_dict, Mp, Rp)
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
                    T_spot=T_spot, spot_cov_frac=spot_cov_frac, **forward_kwargs,
                    frac_scale_height=frac_scale_height, number_density=number_density,
                    part_size=part_size, ri=ri, P_quench=P_quench, full_output=ret_best_fit, zero_opacities=zero_opacities)

                calculated_transit_depths = apply_offsets(
                    calculated_transit_depths, params_dict, "transit")
                residuals = calculated_transit_depths - measured_transit_depths
                scaled_errors = error_multiple * measured_transit_errors
                ln_likelihood = np.append(ln_likelihood, -0.5 * (residuals**2 / scaled_errors**2 + np.log(2 * np.pi * scaled_errors**2)))
                
            if measured_eclipse_depths is not None:
                if params_dict["profile_type"] == "isothermal" and T is None:
                    raise ValueError(
                        "Must fit for T when profile_type is isothermal")

                t_p_profile = Profile()
                t_p_profile.set_from_params_dict(
                    params_dict["profile_type"], params_dict)

                if np.any(np.isnan(t_p_profile.temperatures)):
                    raise AtmosphereError("Invalid T/P profile")

                eclipse_wavelengths, calculated_eclipse_depths, eclipse_info_dict = eclipse_calc.compute_depths(
                    t_p_profile, Rs, Mp, Rp, T_star, logZ, CO_ratio, CH4_mult, gases, vmrs,
                    custom_abundances=None,
                    scattering_factor=scatt_factor, scattering_slope=scatt_slope,
                    cloudtop_pressure=cloudtop_P,
                    T_spot=T_spot, spot_cov_frac=spot_cov_frac, **forward_kwargs,
                    frac_scale_height=frac_scale_height, number_density=number_density,
                    part_size = part_size, ri=ri, P_quench=P_quench, full_output=ret_best_fit, zero_opacities=zero_opacities)
                calculated_eclipse_depths = apply_offsets(
                    calculated_eclipse_depths, params_dict, "eclipse")
                residuals = calculated_eclipse_depths - measured_eclipse_depths
                scaled_errors = error_multiple * measured_eclipse_errors
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
        temperatures in K. Distinct sector grids use their sorted union,
        interpolating temperatures in log pressure and holding edge values
        outside each sector's sampled range."""
        if isinstance(profile, TwoSectorTerminator):
            cold = profile.cold.profile
            hot = profile.hot.profile
            pressures = np.union1d(cold.pressures, hot.pressures)
            return np.array([
                pressures,
                np.interp(np.log(pressures), np.log(cold.pressures), cold.temperatures),
                np.interp(np.log(pressures), np.log(hot.pressures), hot.temperatures)])
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
        
        divisors, new_labels = self._get_divisors_labels(
            np.median(sampler.flatchain, axis=0),
            fit_info.fit_param_names, fit_info)
        
        write_param_estimates_file(
            sampler.flatchain / divisors,
            best_params_arr / divisors,
            np.max(sampler.flatlnprobability),
            new_labels)

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

    def _get_divisors_labels(self, medians, labels, fit_info):
        divisors = np.ones(len(labels))
        new_labels = offset_labels(labels, fit_info)
        offsets = offset_parameter_names(fit_info)
        
        for i, l in enumerate(labels):
            if l in offsets:
                continue
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

        divisors, new_labels = self._get_divisors_labels(
            np.median(equal_samples, axis=0),
            fit_info.fit_param_names, fit_info)
        
        write_param_estimates_file(
            equal_samples / divisors,
            best_params_arr / divisors,
            np.max(result.logp),
            new_labels)
        
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
                      multinest_kwargs={},
                      **dynesty_kwargs):
        """multinest_kwargs are forwarded to pymultinest.solve/run (e.g.
        sampling_efficiency, const_efficiency_mode, evidence_tolerance,
        multimodal, outputfiles_basename)."""
        import pymultinest
        
        self.params_to_lnlike = {}
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
        
        divisors, new_labels = self._get_divisors_labels(
            np.median(equal_samples, axis=0),
            fit_info.fit_param_names, fit_info)
        
        write_param_estimates_file(
            equal_samples / divisors,
            best_params_arr / divisors,
            np.max(result["logp"]),
            new_labels)

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
        divisors, new_labels = self._get_divisors_labels(
            np.median(equal_samples, axis=0), fit_info.fit_param_names, fit_info)
        write_param_estimates_file(
            equal_samples / divisors, best_params_arr / divisors,
            np.max(logp), new_labels)

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
                             scatt_slope=4, error_multiple=1, T_star=None,
                             T_spot=None, spot_cov_frac=None,
                             frac_scale_height=1,
                             log_number_density=-np.inf, log_part_size=-6,
                             n=None, log_k=-np.inf,
                             log_P_quench=-99,
                             offset_transit=0, offset_eclipse=0, offset_start=0, offset_end=sys.maxsize,
                             fit_vmr=False, fit_clr=False,
                             profile_type = 'isothermal',
                             transit_profile_type = 'isothermal',
                             transit_terminator=None,
                             transit_offset_windows=None,
                             eclipse_offset_windows=None,
                             stellar_grid='newera', stellar_blackbody=False,
                             stellar_grid_only=False, T_fac=None, fac_cov_frac=None,
                             logg_phot=4.5, logg_spot=None, logg_fac=None, feh=0.,
                             validate_T_grid=True,
                             **profile_kwargs):
        '''Get a :class:`.FitInfo` object filled with best guess values.  A few
        parameters are required, but others can be set to default values if you
        do not want to specify them.  Physical parameters use SI, except stellar logg
        (log10 cgs), [Fe/H] (dex), and offsets (ppm). For
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
        logg_phot, logg_spot, logg_fac, feh : float
            Stellar log10 gravity (cgs) and shared [Fe/H] (dex). Spots and
            faculae inherit logg_phot; defaults are 4.5 and solar metallicity.
            These can be given retrieval priors like other scalar parameters.
        T_fac : float, optional
            Facula effective temperature in K.
        fac_cov_frac : float, optional
            Facula area fraction; spot and facula fractions sum to at most one.
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
        offset_transit : float
            Additive model-depth offset in ppm, applied to rows selected by
            offset_start and offset_end (e.g. obs[offset_start:offset_end]).
            A positive value increases model depths; 500 means 500 ppm.
        offset_eclipse : float
            Same as above, but for eclipse depths.
        transit_offset_windows, eclipse_offset_windows : dict, optional
            Map offset parameter names (e.g. offset_nirspec) to half-open
            (start, end) index windows, or a list of disjoint windows. Each
            name defaults to zero and can receive a uniform or Gaussian fit
            prior. Initial values, prior bounds, and prior widths are in ppm.
            Supply initial values through keyword arguments. Positive
            offsets increase model depths. Different offsets add in overlaps.
        profile_type : string
            "isothermal", "parametric" (Madhusudhan & Seager 2009) or
            "radiative_solution" (Line et al 2013) T/P profile
            parameterizations.  This profile applies to the dayside, and is
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
            Its named sector values are added to the returned FitInfo.
        profile_kwargs : kwargs
            T/P profile arguments.  For "isothermal": T (K).  For "parametric":
            T0, log_P1, alpha1, alpha2, log_P3, T3. Pressures are
            log10(Pa). For "radiative_solution":
            T_star, Rs, a, Mp, Rp, beta, log_k_th, log_gamma, log_gamma2,
            alpha, and T_int (optional).  We recommend that T_star, Rs, a, and
            Mp be fixed, and that T_int be omitted (which sets it to 100 K).
            

        Returns
        -------
        fit_info : :class:`.FitInfo` object
            This object is used to indicate which parameters to fit for, which
            to fix, and what values all parameters should take.'''
        all_variables = locals().copy()
        del all_variables["profile_kwargs"]
        all_variables.update(profile_kwargs)
        if transit_terminator is not None:
            if not isinstance(transit_terminator, TwoSectorTerminator):
                raise TypeError(
                    "transit_terminator must be a TwoSectorTerminator")
            all_variables.update(transit_terminator.retrieval_defaults())
        
        fit_info = FitInfo(all_variables)
        return fit_info
