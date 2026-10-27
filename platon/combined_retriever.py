import json
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.special
import scipy.stats
import emcee
from dynesty import NestedSampler
from dynesty import plotting as dyplot
import dynesty.utils
import copy
import pickle

from .psis import psisloo
from .transit_depth_calculator import TransitDepthCalculator
from .eclipse_depth_calculator import EclipseDepthCalculator
from .fit_info import FitInfo

from .constants import METRES_TO_UM, M_jup, R_jup, R_earth, M_earth, R_sun
from ._params import _UniformParam, _GaussianParam
from .errors import AtmosphereError
from ._output_writer import write_param_estimates_file
from .TP_profile import Profile
from .terminator import TwoSectorTerminator, label_by_temperature, limb_time_shifts
from .retrieval_result import RetrievalResult
from .custom_dynesty_result import CustomDynestyResult

# Nodes spanning the prior range of an integrated-out error_excess.  The
# integrand is first evaluated at every EXCESS_REFINE-th node; only the cells
# of that coarse grid within EXCESS_MARGIN nats of its maximum (and their
# neighbours) are filled in to the full grid, since the rest of the integral
# is below exp(-EXCESS_MARGIN) of it.
N_EXCESS_NODES = 2049
EXCESS_REFINE = 32
EXCESS_MARGIN = 40.0
assert (N_EXCESS_NODES - 1) % EXCESS_REFINE == 0
# err**2 + s**2 at those nodes and half the sum of its log, which depend only
# on the data errors and the prior: kept for a few error arrays
_EXCESS_VARIANCES = {}


class CombinedRetriever:
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
        terminator_param = fit_info.all_params.get("transit_terminator")
        terminator = None if terminator_param is None else \
            terminator_param.best_guess

        if fit_info._get("time_shifts") is not None and (terminator is None or
                "sector1.fraction" in fit_info.fit_param_names or fit_info._get("sector1.fraction") != 0.5):
            raise ValueError("time_shifts require a transit_terminator with sector1.fraction fixed at 0.5")

        if terminator is not None:
            cloud_fraction = fit_info.all_params["cloud_fraction"].best_guess
            if cloud_fraction != 1 or \
               "cloud_fraction" in fit_info.fit_param_names:
                raise ValueError(
                    "cloud_fraction must be fixed at 1 for a "
                    "TwoSectorTerminator")
        
        if fit_info.all_params["n"].best_guess is None:
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
                    10**fit_info._get("log_cloudtop_P"))

        if terminator is not None:
            best = [fit_info.all_params[name].best_guess
                    for name in fit_info.fit_param_names]
            params = fit_info._interpret_param_array(best)
            sectors = terminator.sectors_from_params(params)
            P_quench = 10**params["log_P_quench"]
            for sector in sectors:
                calculator._validate_params(
                    sector.profile.temperatures, params["logZ"],
                    params["CO_ratio"], sector.cloudtop_pressure,
                    sector.profile.pressures, P_quench)
            for name in fit_info.fit_param_names:
                param = fit_info.all_params[name]
                if not isinstance(param, _UniformParam):
                    continue
                if name not in (
                        "logZ", "CO_ratio", "sector1.log_cloudtop_P",
                        "sector2.log_cloudtop_P"):
                    continue
                for limit in (param.low_lim, param.high_lim):
                    for label, sector in zip(("sector1", "sector2"), sectors):
                        logZ = limit if name == "logZ" else params["logZ"]
                        ratio = limit if name == "CO_ratio" else \
                            params["CO_ratio"]
                        cloudtop = 10**limit if \
                            name == f"{label}.log_cloudtop_P" else \
                            sector.cloudtop_pressure
                        calculator._validate_params(
                            sector.profile.temperatures, logZ, ratio, cloudtop,
                            sector.profile.pressures, P_quench)

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
    def _apply_offsets(depths, params_dict, kind):
        """Adds each named offset in params_dict[kind + "_offsets"] (kind is
        "transit", "eclipse" or "time_shift") to its index range of the model,
        in place."""
        offsets = params_dict.get(kind + "_offsets")
        if offsets is not None:
            for name, (start, end) in offsets.items():
                depths[start:end] += params_dict[name]

    @staticmethod
    def _marginalize(fit_info):
        """Returns a copy of fit_info that does not sample the fitted offsets
        or a fitted error_excess with a uniform prior; _ln_like integrates them
        out over their priors instead (see _nuisance_ln_like), so the evidence
        and the posterior of every other parameter are unchanged.  Returns
        fit_info itself if there is nothing to integrate out."""
        offsets = set()
        for kind in ("transit", "eclipse", "time_shift"):
            param = fit_info.all_params.get(kind + "_offsets")
            if param is not None and param.best_guess is not None:
                offsets.update(param.best_guess)
        marginal = [name for name in fit_info.fit_param_names
                    if name in offsets or (name == "error_excess" and isinstance(
                        fit_info.all_params[name], _UniformParam))]
        if not marginal:
            return fit_info
        reduced = copy.deepcopy(fit_info)
        reduced.fit_param_names = [name for name in fit_info.fit_param_names
                                   if name not in marginal]
        reduced.marginal_params = marginal
        return reduced

    @staticmethod
    def _log_ndtr_diff(high, low):
        """log(Phi(high) - Phi(low)) for high > low, accurate in both tails."""
        flip = low > 0
        high, low = np.where(flip, -low, high), np.where(flip, -high, low)
        log_high = scipy.special.log_ndtr(high)
        return log_high + np.log(-np.expm1(scipy.special.log_ndtr(low) - log_high))

    @staticmethod
    def _nuisance_ln_like(fit_info, params_dict, blocks):
        """Gaussian log likelihood with fit_info.marginal_params integrated out
        over their priors.  blocks maps each data kind to (model - data,
        errors), computed with those offsets and error_excess at 0.

        The model is linear in each offset, so its integral is closed form.
        error_excess, s, adds in quadrature to every transit and eclipse depth
        error, sqrt(err**2 + s**2); with s uniform it is integrated over
        N_EXCESS_NODES values of s, taking the log of the integrand as linear
        between them (_ln_cells).  The depths form one group when error_excess
        is integrated, and the remaining data (time shifts, or everything)
        another; given s the groups are independent, and each offset lies in
        one.  Returns the log likelihood and, per group, (excess name or None,
        s nodes, log integrand at the nodes, [(offset name, model - data,
        errors)] over each offset's data)."""
        marginal = fit_info.marginal_params
        excess = "error_excess" in marginal
        members = [[], []]  # (kind, residuals, errors) without / with the excess
        for kind, (residuals, errors) in blocks.items():
            members[excess and kind != "time_shift"].append((kind, residuals, errors))

        ln_like, groups = 0, []
        for g, group in enumerate(members):
            if not group:
                continue
            name = "error_excess" if g else None
            if name is None:
                s = np.zeros(1)
            else:
                prior = fit_info.all_params[name]
                full = np.linspace(prior.low_lim, prior.high_lim, N_EXCESS_NODES)
                coarse = np.arange(0, N_EXCESS_NODES, EXCESS_REFINE)
                ln_f, _ = CombinedRetriever._excess_group_terms(
                    fit_info, params_dict, group, full, coarse, coarse=True)
                keep = np.maximum(ln_f[:-1], ln_f[1:]) >= np.max(ln_f) - EXCESS_MARGIN
                keep = keep | np.r_[keep[1:], False] | np.r_[False, keep[:-1]]
                index = np.unique(np.concatenate([coarse] + [
                    np.arange(c * EXCESS_REFINE, (c + 1) * EXCESS_REFINE + 1)
                    for c in np.flatnonzero(keep)]))
                terms, group_offsets = CombinedRetriever._excess_group_terms(
                    fit_info, params_dict, group, full, index)
                # Cell widths in units of the full grid's spacing: log(1) = 0 where refined
                ln_like += scipy.special.logsumexp(CombinedRetriever._ln_cells(terms)
                                                   + np.log(np.diff(index))) \
                    - np.log(N_EXCESS_NODES - 1)
                groups.append((name, full[index], terms, group_offsets))
                continue
            if name is None:
                # Sequential sums, in this order, as the batched likelihood
                # reproduces them bit for bit
                chi2, ln_errors, counts = np.zeros(3)
                for kind, residuals, errors in group:
                    first = np.zeros(len(residuals), int)
                    chi2 += np.bincount(first, (residuals / errors)**2, 1)[0]
                    ln_errors += np.bincount(first, np.log(errors), 1)[0]
                    counts += len(residuals)
                terms = np.full(1, -0.5 * chi2 - ln_errors - 0.5 * counts * np.log(2 * np.pi))
            group_offsets = []
            for kind, residuals, errors in group:
                for offset, (start, end) in \
                        (params_dict.get(kind + "_offsets") or {}).items():
                    if offset not in marginal:
                        continue
                    r, e = residuals[start:end], errors[start:end]
                    if len(r) == 0:
                        raise ValueError("Offset {} covers no data".format(offset))
                    W, R = CombinedRetriever._offset_sums(r, e, s)
                    prior = fit_info.all_params[offset]
                    mean, std = R / W, 1 / np.sqrt(W)
                    terms = terms + 0.5 * R * mean + np.log(np.sqrt(2 * np.pi) * std)
                    if isinstance(prior, _GaussianParam):
                        terms += scipy.stats.norm.logpdf(
                            mean, prior.best_guess, np.hypot(std, prior.std))
                    else:
                        terms += CombinedRetriever._log_ndtr_diff(
                            (prior.high_lim - mean) / std, (prior.low_lim - mean) / std) \
                            - np.log(prior.high_lim - prior.low_lim)
                    group_offsets.append((offset, r, e))
            ln_like += terms[0]
            groups.append((name, s, terms, group_offsets))
        return ln_like, groups

    @staticmethod
    def _excess_group_terms(fit_info, params_dict, group, full, index, coarse=False):
        """Log integrand of the error-excess group at the nodes full[index]
        (sorted indices into the full node grid; coarse: index is every
        EXCESS_REFINE-th node), with its integrated offsets' closed-form terms,
        and [(offset, model - data, errors)]."""
        marginal = fit_info.marginal_params
        terms = np.zeros(len(index))
        rows = {}
        for kind, residuals, errors in group:
            entry = rows[kind] = CombinedRetriever._excess_cache(errors, full)
            inverse = entry["coarse"] if coarse else entry["inverse"][index]
            entry = rows[kind] = dict(entry, rows=inverse)
            terms += -0.5 * (inverse @ residuals**2) \
                - entry["half_ln"][index] - 0.5 * len(residuals) * np.log(2 * np.pi)
        group_offsets = []
        for kind, residuals, errors in group:
            entry = rows[kind]
            for offset, (start, end) in \
                    (params_dict.get(kind + "_offsets") or {}).items():
                if offset not in marginal:
                    continue
                r, e = residuals[start:end], errors[start:end]
                if len(r) == 0:
                    raise ValueError("Offset {} covers no data".format(offset))
                if (start, end) not in entry["sums"]:
                    entry["sums"][start, end] = entry["inverse"][:, start:end].sum(axis=1)
                W = entry["sums"][start, end][index]
                R = -(entry["rows"][:, start:end] @ r)
                prior = fit_info.all_params[offset]
                mean, std = R / W, 1 / np.sqrt(W)
                terms = terms + 0.5 * R * mean + np.log(np.sqrt(2 * np.pi) * std)
                if isinstance(prior, _GaussianParam):
                    terms += scipy.stats.norm.logpdf(
                        mean, prior.best_guess, np.hypot(std, prior.std))
                else:
                    terms += CombinedRetriever._log_ndtr_diff(
                        (prior.high_lim - mean) / std, (prior.low_lim - mean) / std) \
                        - np.log(prior.high_lim - prior.low_lim)
                group_offsets.append((offset, r, e))
        return terms, group_offsets

    @staticmethod
    def _nuisance_ln_like_rows(fit_info, params_dicts, blocks_rows):
        """_nuisance_ln_like's log likelihood for each of many rows of one fit.
        Each row refines its own error-excess nodes, so rows are integrated one
        at a time, exactly as _nuisance_ln_like integrates them."""
        return np.array([CombinedRetriever._nuisance_ln_like(fit_info, p, b)[0]
                         for p, b in zip(params_dicts, blocks_rows)])

    @staticmethod
    def _excess_cache(errors, full):
        """What the error-excess integrand needs at the full node grid, which
        depends only on the data errors and the prior, computed once per error
        array: 1/(errors**2 + s**2) as a C-contiguous (nodes x data) array and
        its coarse rows, half the sum of log(errors**2 + s**2) over the data,
        and each offset's per-node weight sum (filled in on first use)."""
        key = (np.asarray(errors).tobytes(), full[0], full[-1], len(full))
        entry = _EXCESS_VARIANCES.get(key)
        if entry is None:
            if len(_EXCESS_VARIANCES) >= 8:
                _EXCESS_VARIANCES.pop(next(iter(_EXCESS_VARIANCES)))
            variance = errors[None, :]**2 + full[:, None]**2
            inverse = np.ascontiguousarray(1 / variance)
            entry = _EXCESS_VARIANCES[key] = dict(
                inverse=inverse, coarse=np.ascontiguousarray(inverse[::EXCESS_REFINE]),
                half_ln=0.5 * np.sum(np.log(variance), axis=1), sums={})
        return entry

    @staticmethod
    def _offset_sums(residuals, errors, s):
        """W = sum 1/var and R = sum (data - model)/var over an offset's data,
        with var = errors**2 + s**2, for each value of the error excess s."""
        if np.all(s == 0):
            weights = errors**-2
            return np.full(len(s), weights.sum()), np.full(len(s), -(weights * residuals).sum())
        weights = 1 / (errors[None, :]**2 + np.asarray(s)[:, None]**2)
        return weights.sum(axis=1), -(weights * residuals[None, :]).sum(axis=1)

    @staticmethod
    def _ln_cells(ln_f):
        """ln of the integral of f over each cell between equally spaced
        nodes, in units of the spacing, with ln f linear in each cell.  This is
        exact where f rises exponentially, e.g. toward a prior edge."""
        d = np.diff(ln_f, axis=-1)   # along the nodes (the last axis)
        a = np.abs(d)
        small = a < 1e-8
        a = np.where(small, 1, a)
        return ln_f[..., :-1] + np.where(
            small, d / 2, np.maximum(d, 0) + np.log(-np.expm1(-a)) - np.log(a))

    @staticmethod
    def _draw_nuisance(fit_info, groups, size, mode=False):
        """size draws of each integrated-out parameter from its posterior given
        the other parameters, as a dict of arrays; with mode, its conditional
        mode instead (the most probable error_excess node, then the offsets'
        mode given it).  groups is from _nuisance_ln_like."""
        values = {}
        for name, x, ln_f, offsets in groups:
            if name is None:
                excess = np.zeros(size)
            elif mode:
                excess = np.full(size, x[np.argmax(ln_f)])
            else:
                # a cell, then a point in it from the log-linear density
                # cells of the (partly refined) node grid, weighted by their widths
                ln_cells = CombinedRetriever._ln_cells(ln_f) + np.log(np.diff(x))
                p = np.exp(ln_cells - ln_cells.max())
                cell = np.random.choice(len(p), size, p=p / p.sum())
                slope, u = np.diff(ln_f)[cell], np.random.random(size)
                with np.errstate(all="ignore"):
                    t = np.where(np.abs(slope) < 1e-8, u, np.where(
                        slope > 0, 1 + np.log(u + (1 - u) * np.exp(-slope)) / slope,
                        np.log1p(u * np.expm1(slope)) / slope))
                excess = x[cell] + t * (x[cell + 1] - x[cell])
            if name is not None:  # inside the open prior interval
                excess = values[name] = np.clip(
                    excess, np.nextafter(x[0], x[-1]), np.nextafter(x[-1], x[0]))
            for offset, r, e in offsets:
                prior = fit_info.all_params[offset]
                W, R = CombinedRetriever._offset_sums(r, e, excess)
                mean, std = R / W, 1 / np.sqrt(W)
                if isinstance(prior, _GaussianParam):
                    var = 1 / (std**-2 + prior.std**-2)
                    mean = var * (mean / std**2 + prior.best_guess / prior.std**2)
                    values[offset] = mean if mode else \
                        np.random.normal(mean, np.sqrt(var), size)
                elif mode:  # inside the open prior interval
                    values[offset] = np.clip(
                        mean, np.nextafter(prior.low_lim, prior.high_lim),
                        np.nextafter(prior.high_lim, prior.low_lim))
                else:
                    values[offset] = scipy.stats.truncnorm.rvs(
                        (prior.low_lim - mean) / std, (prior.high_lim - mean) / std,
                        loc=mean, scale=std, size=size)
        return values

    def _batched_ln_like(self, transit_calc, eclipse_calc, fit_info,
                         transit_depths, transit_errors, eclipse_depths,
                         eclipse_errors, zero_opacities=(), batch_size=256):
        """The batched likelihood for (batch, n_parameters) rows, or None.

        Where the model supports it (transit retrievals with a two-sector or
        a 1-D terminator and Guillot, isothermal or Madhusudhan & Seager
        parametric profiles), it returns exactly what _ln_like returns for
        each row, evaluating many rows per device call; otherwise samplers use
        _ln_like.
        """
        from ._batched_likelihood import BatchedLikelihood
        try:
            return BatchedLikelihood(
                self, transit_calc, eclipse_calc, fit_info,
                transit_depths, transit_errors, eclipse_depths, eclipse_errors,
                zero_opacities=zero_opacities, batch_size=batch_size)
        except (ValueError, AtmosphereError):
            return None

    def _with_nuisance(self, rows, fit_info, full_fit_info, like_args, mode=False):
        """Rows of fit_info's parameters as rows of full_fit_info's, each
        integrated-out parameter drawn from its posterior given the rest of
        its row (or, with mode, set to its conditional mode).  Each distinct
        row costs one model evaluation.  like_args = (transit_calc,
        eclipse_calc, transit_depths, transit_errors, eclipse_depths,
        eclipse_errors, zero_opacities)."""
        rows = np.array(rows, float, ndmin=2)
        if fit_info is full_fit_info:
            return rows
        transit_calc, eclipse_calc, *data, zero_opacities = like_args
        unique, inverse = np.unique(rows, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        order = np.argsort(inverse, kind="stable")
        result = np.empty((len(rows), len(full_fit_info.fit_param_names)))
        for row, where in zip(unique, np.split(
                order, np.cumsum(np.bincount(inverse))[:-1])):
            ln_like = self._ln_like(row, transit_calc, eclipse_calc, fit_info,
                                    *data, zero_opacities=zero_opacities)
            if not np.isfinite(ln_like):
                raise ValueError("Posterior sample {} has zero likelihood".format(row))
            values = {name: np.full(len(where), value)
                      for name, value in zip(fit_info.fit_param_names, row)}
            values.update(self._draw_nuisance(
                fit_info, self.last_nuisance, len(where), mode))
            result[where] = np.array(
                [values[name] for name in full_fit_info.fit_param_names]).T
        return result

    def _best_with_nuisance(self, best, fit_info, full_fit_info, like_args):
        """The best-fit row of fit_info's parameters as a row of
        full_fit_info's, with its integrated-out parameters at their
        conditional mode, and that row's ln probability."""
        best = self._with_nuisance(best, fit_info, full_fit_info, like_args,
                                   mode=True)[0]
        transit_calc, eclipse_calc, *data, zero_opacities = like_args
        return best, full_fit_info._ln_prior(best) + self._ln_like(
            best, transit_calc, eclipse_calc, full_fit_info, *data,
            zero_opacities=zero_opacities)

    def _pointwise_ln_like(self, params, fit_info, like_args):
        """Per-point ln likelihoods at params, from the cache if there."""
        pointwise = self.params_to_lnlike.get(tuple(params))
        if pointwise is None:
            transit_calc, eclipse_calc, *data, zero_opacities = like_args
            pointwise = self._ln_like(
                params, transit_calc, eclipse_calc, fit_info, *data,
                zero_opacities=zero_opacities, lnlike_per_point=True)
        return pointwise

    @staticmethod
    def _visit_hets(params, n_points):
        """T_het, het_cov_frac, T_het2, het2_cov_frac, as arrays with one value per point
        if any visit in transit_visits has its own value."""
        hets = {name: params[name] for name in ("T_het", "het_cov_frac", "T_het2", "het2_cov_frac")}
        visits = params.get("transit_visits") or {}
        for name in hets:
            own = [(r, params[v + "." + name]) for v, r in visits.items()
                   if params[v + "." + name] is not None]
            if own:
                shared = hets[name] if hets[name] is not None else \
                    params["T_star"] if name.startswith("T") else 0
                hets[name] = np.full(n_points, shared, float)
                for (start, end), value in own:
                    hets[name][start:end] = value
        return hets

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
        marginal = getattr(fit_info, "marginal_params", ())
        for name in marginal:
            params_dict[name] = 0
        blocks = {}
        
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
        T_het, het_cov_frac = params_dict["T_het"], params_dict["het_cov_frac"]
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
            if cloud_fraction != 1 or any(not 0 <= params_dict[name] <= 1 for name in
                    ("sector1.fraction", "sector1.cloud_fraction", "sector2.cloud_fraction")):
                return -np.inf

        vmrs = None
        gases = None
        if params_dict["fit_vmr"] or params_dict["fit_clr"]:
            on_eq = logZ is not None and CO_ratio is not None and params_dict["fit_vmr"]
            assert on_eq or (logZ is None and CO_ratio is None)
            gases = fit_info.gases
            per_sector = f'sector1.log_{gases[0]}' in params_dict or f'sector1.clr_{gases[0]}' in params_dict
            if per_sector and measured_eclipse_depths is not None:
                raise ValueError("Per-sector free abundances apply to transit depths only")
            for prefix in ("sector1.", "sector2.") if per_sector else ("",):
                if on_eq:
                    vmrs = [10.**params_dict[f'{prefix}log_{gas}'] for gas in gases]
                elif params_dict["fit_vmr"]:
                    vmrs = [10.**params_dict[f'{prefix}log_{gas}'] for gas in gases[:-1]]
                    vmrs.append(1 - np.sum(vmrs))
                    if vmrs[-1] < 0: return -np.inf
                else:
                    clrs = [params_dict[f'{prefix}clr_{gas}'] for gas in gases[:-1]]
                    vmrs = self.convert_clr_to_vmr(clrs)
                    if np.min(vmrs) < fit_info.clr_low_lim: return -np.inf
                if prefix:
                    params_dict[prefix + "vmrs"] = vmrs
        if "n" in params_dict and params_dict["n"] is not None and "log_k" in params_dict:
            ri = params_dict["n"] - 1j * 10**params_dict["log_k"]
        else:
            ri = None
            
        if Rs <= 0 or Mp <= 0:
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
                    **self._visit_hets(params_dict, len(measured_transit_depths)),
                    frac_scale_height=frac_scale_height, number_density=number_density,
                    part_size=part_size, ri=ri, P_quench=P_quench, full_output=ret_best_fit, zero_opacities=zero_opacities)

                self._apply_offsets(calculated_transit_depths, params_dict, "transit")
                residuals = calculated_transit_depths - measured_transit_depths
                scaled_errors = np.sqrt(measured_transit_errors**2 + error_excess**2)
                blocks["transit"] = residuals, scaled_errors
                ln_likelihood = np.append(ln_likelihood, -0.5 * (residuals**2 / scaled_errors**2 + np.log(2 * np.pi * scaled_errors**2)))

                if params_dict.get("time_shifts") is not None:
                    leading, trailing = transit_info_dict["cold_depths"], transit_info_dict["hot_depths"]
                    if not transit_profile.cold_leads:
                        leading, trailing = trailing, leading
                    coefficients = params_dict.get("time_shift_coefficients")
                    shifts = limb_time_shifts(leading, trailing, Rs, params_dict["a"], params_dict["period"], params_dict["b"]) \
                        if coefficients is None else coefficients * (np.sqrt(trailing) - np.sqrt(leading))
                    self._apply_offsets(shifts, params_dict, "time_shift")
                    residuals = shifts - params_dict["time_shifts"]
                    errors = params_dict["time_shift_errors"]
                    blocks["time_shift"] = residuals, errors
                    ln_likelihood = np.append(ln_likelihood, -0.5 * (residuals**2 / errors**2 + np.log(2 * np.pi * errors**2)))
                    if ret_best_fit:
                        transit_info_dict["time_shifts"] = shifts
                
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
                    T_het=T_het, het_cov_frac=het_cov_frac,
                    frac_scale_height=frac_scale_height, number_density=number_density,
                    part_size = part_size, ri=ri, P_quench=P_quench, full_output=ret_best_fit, zero_opacities=zero_opacities)
                self._apply_offsets(calculated_eclipse_depths, params_dict, "eclipse")
                residuals = calculated_eclipse_depths - measured_eclipse_depths
                scaled_errors = np.sqrt(measured_eclipse_errors**2 + error_excess**2)
                blocks["eclipse"] = residuals, scaled_errors
                ln_likelihood = np.append(ln_likelihood, -0.5 * (residuals**2 / scaled_errors**2 + np.log(2 * np.pi * scaled_errors**2)))

        except AtmosphereError as e:
            return -np.inf

        if marginal:
            ln_likelihood, self.last_nuisance = self._nuisance_ln_like(
                fit_info, params_dict, blocks)
        
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
            if marginal:
                return np.atleast_1d(ln_likelihood)
            self.params_to_lnlike[tuple(params)] = ln_likelihood
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
                transit_info["unbinned_depths"])
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

    def _ln_prob(self, params, transit_calc, eclipse_calc, fit_info, measured_transit_depths,
                 measured_transit_errors, measured_eclipse_depths,
                 measured_eclipse_errors, zero_opacities=[]):
        
        lnlike_per_point = self._ln_like(params, transit_calc, eclipse_calc, fit_info, measured_transit_depths,
                                measured_transit_errors, measured_eclipse_depths,
                                measured_eclipse_errors, zero_opacities=zero_opacities, lnlike_per_point=True)
        
        if not np.isscalar(lnlike_per_point):
            ln_like = lnlike_per_point.sum()
        else:
            assert(lnlike_per_point == -np.inf)
            ln_like = -np.inf

        return fit_info._ln_prior(params) + ln_like


    def run_emcee(self, transit_bins, transit_depths, transit_errors,
                  eclipse_bins, eclipse_depths, eclipse_errors,
                  fit_info, nwalkers=50,
                  nsteps=1000, include_condensation=True,
                  rad_method="xsec",
                  num_final_samples=100, zero_opacities=[],
                  marginalize_nuisance=True):
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
        marginalize_nuisance : bool, optional
            Integrate the fitted offsets, and error_excess if fitted with a
            uniform prior, out of the likelihood instead of sampling them
            (see the "Offsets and error excess" section of the docs).
            The returned samples include them, drawn from their posterior.

        Returns
        -------
        result : RetrievalResult object
        '''
        self.params_to_lnlike = {}
        transit_calc = None
        eclipse_calc = None

        if transit_bins is not None:
            transit_calc = TransitDepthCalculator(
                include_condensation=include_condensation, method=rad_method,
                **fit_info._stellar_grid_options())
            transit_calc.change_wavelength_bins(transit_bins)
            self._validate_params(fit_info, transit_calc)
        if eclipse_bins is not None:
            eclipse_calc = EclipseDepthCalculator(
                include_condensation=include_condensation, method=rad_method)
            eclipse_calc.change_wavelength_bins(eclipse_bins)       

        full_fit_info = fit_info
        if marginalize_nuisance:
            fit_info = self._marginalize(fit_info)
        like_args = (transit_calc, eclipse_calc, transit_depths, transit_errors,
                     eclipse_depths, eclipse_errors, zero_opacities)
        initial_positions = fit_info._generate_rand_param_arrays(nwalkers)

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
        best_ln_prob = np.max(sampler.flatlnprobability)
        flatchain = self._with_nuisance(
            sampler.flatchain, fit_info, full_fit_info, like_args)
        if fit_info is not full_fit_info:
            best_params_arr, best_ln_prob = self._best_with_nuisance(
                best_params_arr, fit_info, full_fit_info, like_args)
        fit_info = full_fit_info
        
        # Two-sector parameters are reported as cold/hot by temperature
        labels, samples = label_by_temperature(
            fit_info, np.vstack([flatchain, best_params_arr]))
        divisors, new_labels = self._get_divisors_labels(
            np.median(samples[:-1], axis=0), labels)
        
        write_param_estimates_file(
            samples[:-1] / divisors,
            samples[-1] / divisors,
            best_ln_prob,
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
        equal_samples = np.copy(flatchain)
        np.random.shuffle(equal_samples)
        self._init_random_samples(retrieval_result)
        for params in equal_samples[:num_final_samples]:
            ret = self._ln_like(
                params, transit_calc, eclipse_calc, fit_info,
                transit_depths, transit_errors,
                eclipse_depths, eclipse_errors,
                zero_opacities=zero_opacities, ret_best_fit=True)
            if ret == -np.inf: continue
            _, transit_info, _, eclipse_info = ret
            self._record_random_sample(
                retrieval_result, transit_info, eclipse_info,
                self._pointwise_ln_like(params, fit_info, like_args))
        retrieval_result.loo_total, retrieval_result.loos, retrieval_result.loo_ks = psisloo(np.array(retrieval_result.pointwise_lnlikes))
        return retrieval_result

    def _get_divisors_labels(self, medians, labels):
        divisors = np.ones(len(labels))
        new_labels = list(labels)
        
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
                      marginalize_nuisance=True, **dynesty_kwargs):
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
        marginalize_nuisance : bool, optional
            Integrate the fitted offsets, and error_excess if fitted with a
            uniform prior, out of the likelihood instead of sampling them
            (see the "Offsets and error excess" section of the docs).
            The returned samples include them, drawn from their posterior.
        **dynesty_kwargs : keyword arguments to pass to dynesty's NestedSampler

        Returns
        -------
        result : RetrievalResult object
        '''        
        self.params_to_lnlike = {}
        transit_calc = None
        eclipse_calc = None
        if transit_bins is not None:
            transit_calc = TransitDepthCalculator(
                include_condensation=include_condensation, method=rad_method,
                **fit_info._stellar_grid_options())
            transit_calc.change_wavelength_bins(transit_bins)
            self._validate_params(fit_info, transit_calc)
        if eclipse_bins is not None:
            eclipse_calc = EclipseDepthCalculator(
                include_condensation=include_condensation, method=rad_method)
            eclipse_calc.change_wavelength_bins(eclipse_bins)

        full_fit_info = fit_info
        if marginalize_nuisance:
            fit_info = self._marginalize(fit_info)
        like_args = (transit_calc, eclipse_calc, transit_depths, transit_errors,
                     eclipse_depths, eclipse_errors, zero_opacities)

        def transform_prior(cube):
            return fit_info._from_unit_interval_array(cube)

        def dynesty_ln_like(cube):
            lnlike_per_point = self._ln_like(cube, transit_calc, eclipse_calc, fit_info, transit_depths, transit_errors,
                                    eclipse_depths, eclipse_errors, zero_opacities=zero_opacities, lnlike_per_point=True)
            if not np.isscalar(lnlike_per_point):
                ln_like = lnlike_per_point.sum()
            else:
                assert(lnlike_per_point == -np.inf)
                ln_like = -np.inf
            
            if np.random.randint(100) == 0:
                print("\nEvaluated params: {}".format(self.pretty_print(fit_info)))
            return ln_like

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
        best_ln_prob = np.max(result.logp)
        equal_samples = self._with_nuisance(
            equal_samples, fit_info, full_fit_info, like_args)
        if fit_info is not full_fit_info:
            best_params_arr, best_ln_prob = self._best_with_nuisance(
                best_params_arr, fit_info, full_fit_info, like_args)
        fit_info = full_fit_info

        labels, samples = label_by_temperature(
            fit_info, np.vstack([equal_samples, best_params_arr]))
        divisors, new_labels = self._get_divisors_labels(
            np.median(samples[:-1], axis=0), labels)
        
        write_param_estimates_file(
            samples[:-1] / divisors,
            samples[-1] / divisors,
            best_ln_prob,
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

        self._init_random_samples(retrieval_result)
        for params in equal_samples[:num_final_samples]:
            _, transit_info, _, eclipse_info = self._ln_like(
                params, transit_calc, eclipse_calc, fit_info,
                transit_depths, transit_errors,
                eclipse_depths, eclipse_errors,
                zero_opacities=zero_opacities, ret_best_fit=True)
            self._record_random_sample(
                retrieval_result, transit_info, eclipse_info,
                self._pointwise_ln_like(params, fit_info, like_args))

        #Calculate LOO-CV scores
        retrieval_result.loo_total, retrieval_result.loos, retrieval_result.loo_ks = psisloo(np.array(retrieval_result.pointwise_lnlikes))
                    
        return retrieval_result


    def run_multinest(self, transit_bins, transit_depths, transit_errors,
                      eclipse_bins, eclipse_depths, eclipse_errors,
                      fit_info,
                      include_condensation=True, rad_method="xsec",
                      maxiter=None, maxcall=None, nlive=250,
                      num_final_samples=100, zero_opacities=[],
                      multinest_kwargs={}, marginalize_nuisance=True):
        """multinest_kwargs are forwarded to pymultinest.solve/run (e.g.
        sampling_efficiency, const_efficiency_mode, evidence_tolerance,
        multimodal, outputfiles_basename). maxiter sets MultiNest's max_iter
        (0 means unlimited). maxcall is unsupported and must be None.
        marginalize_nuisance is as for run_dynesty.
        """
        if maxcall is not None:
            raise ValueError(
                "MultiNest does not support maxcall; use maxiter instead")
        if maxiter is not None:
            if not isinstance(maxiter, (int, np.integer)) or maxiter < 0:
                raise ValueError("maxiter must be a non-negative integer")
            if "max_iter" in multinest_kwargs and \
               multinest_kwargs["max_iter"] != maxiter:
                raise ValueError("maxiter conflicts with multinest_kwargs['max_iter']")
        import pymultinest
        
        self.params_to_lnlike = {}
        transit_calc = None
        eclipse_calc = None
        if transit_bins is not None:
            transit_calc = TransitDepthCalculator(
                include_condensation=include_condensation, method=rad_method,
                **fit_info._stellar_grid_options())
            transit_calc.change_wavelength_bins(transit_bins)
            self._validate_params(fit_info, transit_calc)
        if eclipse_bins is not None:
            eclipse_calc = EclipseDepthCalculator(
                include_condensation=include_condensation, method=rad_method)
            eclipse_calc.change_wavelength_bins(eclipse_bins)

        full_fit_info = fit_info
        if marginalize_nuisance:
            fit_info = self._marginalize(fit_info)
        like_args = (transit_calc, eclipse_calc, transit_depths, transit_errors,
                     eclipse_depths, eclipse_errors, zero_opacities)

        def transform_prior(cube):
            return fit_info._from_unit_interval_array(cube)

        def multinest_ln_like(cube):
            lnlike_per_point = self._ln_like(cube, transit_calc, eclipse_calc, fit_info, transit_depths, transit_errors,
                                    eclipse_depths, eclipse_errors, zero_opacities=zero_opacities, lnlike_per_point=True)
            if not np.isscalar(lnlike_per_point):
                ln_like = lnlike_per_point.sum()
            else:
                assert(lnlike_per_point == -np.inf)
                ln_like = -np.inf
            
            if np.random.randint(100) == 0:
                print("\nEvaluated params: {}".format(self.pretty_print(fit_info)))
            return ln_like

        num_dim = fit_info._get_num_fit_params()
        solve_kwargs = dict(
            verbose=True, resume=False, n_live_points=nlive,
            outputfiles_basename="multinest_" + str(np.random.randint(1000)))
        solve_kwargs.update(multinest_kwargs)
        if maxiter is not None:
            solve_kwargs["max_iter"] = int(maxiter)
        basename = solve_kwargs["outputfiles_basename"]
        result = pymultinest.solve(LogLikelihood=multinest_ln_like, Prior=transform_prior,
                                   n_dims=num_dim, **solve_kwargs)
        a = pymultinest.Analyzer(outputfiles_basename=basename, n_params=num_dim)
        data = a.get_data()
        result["logz"] = np.atleast_1d(result["logZ"])
        result["samples"] = data[:,2:]
        result["logl"] = -0.5 * data[:,1]
        result["logp"] = result["logl"] + np.array(
            [fit_info._ln_prior(params) for params in result["samples"]])
        best_params_arr = result["samples"][np.argmax(result["logp"])]
        
        equal_samples = a.get_equal_weighted_posterior()[:,:-1]
        np.random.shuffle(equal_samples)
        best_ln_prob = np.max(result["logp"])
        equal_samples = self._with_nuisance(
            equal_samples, fit_info, full_fit_info, like_args)
        if fit_info is not full_fit_info:
            best_params_arr, best_ln_prob = self._best_with_nuisance(
                best_params_arr, fit_info, full_fit_info, like_args)
        fit_info = full_fit_info
        result["equal_samples"] = equal_samples
        
        labels, samples = label_by_temperature(
            fit_info, np.vstack([equal_samples, best_params_arr]))
        divisors, new_labels = self._get_divisors_labels(
            np.median(samples[:-1], axis=0), labels)
        
        write_param_estimates_file(
            samples[:-1] / divisors,
            samples[-1] / divisors,
            best_ln_prob,
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

        self._init_random_samples(retrieval_result)
        for params in equal_samples[:num_final_samples]:
            pointwise = self.params_to_lnlike.get(tuple(params))
            if pointwise is None:
                pointwise = self._ln_like(
                    params, transit_calc, eclipse_calc, fit_info,
                    transit_depths, transit_errors,
                    eclipse_depths, eclipse_errors,
                    zero_opacities=zero_opacities, lnlike_per_point=True)
            _, transit_info, _, eclipse_info = self._ln_like(
                params, transit_calc, eclipse_calc, fit_info,
                transit_depths, transit_errors,
                eclipse_depths, eclipse_errors,
                zero_opacities=zero_opacities, ret_best_fit=True)
            self._record_random_sample(
                retrieval_result, transit_info, eclipse_info, pointwise)

        #Calculate LOO-CV scores
        retrieval_result.loo_total, retrieval_result.loos, retrieval_result.loo_ks = psisloo(np.array(retrieval_result.pointwise_lnlikes))
        return retrieval_result

    def run_nautilus(self, transit_bins, transit_depths, transit_errors,
                     eclipse_bins, eclipse_depths, eclipse_errors,
                     fit_info, include_condensation=True, rad_method="xsec",
                     n_live=2000, n_eff=10000, n_networks=16,
                     discard_exploration=True, verbose=True,
                     num_final_samples=100, zero_opacities=(),
                     marginalize_nuisance=True, **nautilus_kwargs):
        """Run optional Nautilus nested sampling.

        Install support with ``pip install "platon[nautilus]"`` or
        ``pip install nautilus-sampler``. Extra keyword arguments are passed
        directly to :class:`nautilus.Sampler`.  marginalize_nuisance is as
        for run_dynesty; when it applies, the returned equal-weight samples
        are n_eff random rows of the resampled posterior, each with its
        integrated-out parameters drawn from their posterior.  Two-sector
        Guillot transit retrievals evaluate each batch of points at once
        (vectorized), with exactly the scalar likelihood's values.
        """
        try:
            from nautilus import Sampler
        except ImportError as error:
            raise ImportError(
                'run_nautilus requires the optional "nautilus-sampler" '
                'package. Install it with pip install "platon[nautilus]" '
                'or pip install nautilus-sampler.') from error

        self.params_to_lnlike = {}
        transit_calc = None
        eclipse_calc = None
        if transit_bins is not None:
            transit_calc = TransitDepthCalculator(
                include_condensation=include_condensation, method=rad_method,
                **fit_info._stellar_grid_options())
            transit_calc.change_wavelength_bins(transit_bins)
            self._validate_params(fit_info, transit_calc)
        if eclipse_bins is not None:
            eclipse_calc = EclipseDepthCalculator(
                include_condensation=include_condensation, method=rad_method)
            eclipse_calc.change_wavelength_bins(eclipse_bins)

        full_fit_info = fit_info
        if marginalize_nuisance:
            fit_info = self._marginalize(fit_info)
        like_args = (transit_calc, eclipse_calc, transit_depths, transit_errors,
                     eclipse_depths, eclipse_errors, zero_opacities)

        def transform_prior(cube):
            return fit_info._from_unit_interval_array(cube)

        # No per-point cache: Nautilus makes millions of calls, and the final
        # samples below recompute their pointwise likelihoods
        def nautilus_ln_like(params):
            return self._ln_like(
                params, transit_calc, eclipse_calc, fit_info,
                transit_depths, transit_errors, eclipse_depths, eclipse_errors,
                zero_opacities=zero_opacities)

        # Where it applies, evaluate Nautilus's batches of points at once;
        # the likelihood values, and so the run, are unchanged
        batched = self._batched_ln_like(
            transit_calc, eclipse_calc, fit_info, transit_depths,
            transit_errors, eclipse_depths, eclipse_errors, zero_opacities)
        if batched is not None:
            def transform_prior(cube):
                return np.array([fit_info._from_unit_interval_array(row)
                                 for row in cube])
            nautilus_ln_like = batched
            nautilus_kwargs["vectorized"] = True

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
        best_ln_prob = np.max(logp)
        if fit_info is not full_fit_info:
            equal_samples = equal_samples[:int(n_eff)]
        equal_samples = self._with_nuisance(
            equal_samples, fit_info, full_fit_info, like_args)
        if fit_info is not full_fit_info:
            best_params_arr, best_ln_prob = self._best_with_nuisance(
                best_params_arr, fit_info, full_fit_info, like_args)
        fit_info = full_fit_info
        labels, samples = label_by_temperature(
            fit_info, np.vstack([equal_samples, best_params_arr]))
        divisors, new_labels = self._get_divisors_labels(
            np.median(samples[:-1], axis=0), labels)
        write_param_estimates_file(
            samples[:-1] / divisors, samples[-1] / divisors,
            best_ln_prob, new_labels)

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

        self._init_random_samples(retrieval_result)
        for params in equal_samples[:num_final_samples]:
            pointwise = self.params_to_lnlike.get(tuple(params))
            if pointwise is None:
                pointwise = self._ln_like(
                    params, transit_calc, eclipse_calc, fit_info,
                    transit_depths, transit_errors,
                    eclipse_depths, eclipse_errors,
                    zero_opacities=zero_opacities, lnlike_per_point=True)
            _, transit_info, _, eclipse_info = self._ln_like(
                params, transit_calc, eclipse_calc, fit_info,
                transit_depths, transit_errors,
                eclipse_depths, eclipse_errors,
                zero_opacities=zero_opacities, ret_best_fit=True)
            self._record_random_sample(
                retrieval_result, transit_info, eclipse_info, pointwise)

        retrieval_result.loo_total, retrieval_result.loos, \
            retrieval_result.loo_ks = psisloo(
                np.array(retrieval_result.pointwise_lnlikes))
        return retrieval_result

    def run_inessai(self, transit_bins, transit_depths, transit_errors,
                    eclipse_bins, eclipse_depths, eclipse_errors,
                    fit_info, include_condensation=True, rad_method="xsec",
                    output="inessai_output", n_live=5000, target_ess=5000,
                    min_final_draws=1000000, max_final_draws=3000000,
                    seed=None, max_seconds=None, verbose=True,
                    num_final_samples=100, zero_opacities=(),
                    marginalize_nuisance=True, **nessai_kwargs):
        """Run importance nested sampling with normalizing flows (i-nessai).

        Install support with ``pip install "platon[inessai]"`` (nessai
        0.15.2 and PyTorch).  The run checkpoints in ``output``; on SIGTERM
        or SIGUSR1, or after ``max_seconds``, it stops at the next
        checkpoint and raises ``platon._inessai.Checkpointed`` (a SystemExit
        with status 75).  Calling it again with the same arguments resumes
        it.  ``seed=None`` draws a seed, which the output directory records.

        After the nested sampling converges, its mixture of flows is frozen
        and an independent importance sample of min_final_draws to
        max_final_draws points (enough for an effective sample size of
        target_ess, judged from 8,192 planning draws) gives the evidence and
        posterior.  The result has ``logz``, ``logz_err``, the effective
        sample size ``ess`` (also ``n_eff``), ``psis_ess``, the Pareto
        ``pareto_k`` diagnostic (above 0.7, the weights are too heavy-tailed
        for the evidence error to be trusted), the weighted final draws
        (``draws``, ``weights``, ``logw``, ``logl``, ``logp``) and ``n_calls``.
        ``samples`` holds ``ess`` (at most ``target_ess``) equal-weight rows
        of the full model, with integrated-out parameters drawn as for
        run_nautilus, followed by the best fit.  Extra keyword arguments
        override nessai's sampler options.
        """
        try:
            from . import _inessai
        except ImportError as error:
            raise ImportError(
                'run_inessai requires the optional "nessai" (0.15.2) and '
                '"torch" packages. Install them with pip install '
                '"platon[inessai]". ({})'.format(error)) from error

        self.params_to_lnlike = {}
        transit_calc = None
        eclipse_calc = None
        if transit_bins is not None:
            transit_calc = TransitDepthCalculator(
                include_condensation=include_condensation, method=rad_method,
                **fit_info._stellar_grid_options())
            transit_calc.change_wavelength_bins(transit_bins)
            self._validate_params(fit_info, transit_calc)
        if eclipse_bins is not None:
            eclipse_calc = EclipseDepthCalculator(
                include_condensation=include_condensation, method=rad_method)
            eclipse_calc.change_wavelength_bins(eclipse_bins)

        full_fit_info = fit_info
        if marginalize_nuisance:
            fit_info = self._marginalize(fit_info)
        like_args = (transit_calc, eclipse_calc, transit_depths, transit_errors,
                     eclipse_depths, eclipse_errors, zero_opacities)
        transform = _inessai.cube_transform(fit_info)
        batched = self._batched_ln_like(
            transit_calc, eclipse_calc, fit_info, transit_depths,
            transit_errors, eclipse_depths, eclipse_errors, zero_opacities)
        if batched is None:
            def batched(rows):
                return np.array([self._ln_like(
                    row, transit_calc, eclipse_calc, fit_info,
                    transit_depths, transit_errors, eclipse_depths,
                    eclipse_errors, zero_opacities=zero_opacities)
                    for row in rows])

        def cube_ln_like(cube):
            result = np.full(len(cube), -np.inf)
            valid = np.all(np.isfinite(cube) & (cube > 0) & (cube < 1), axis=1)
            if np.any(valid):
                values = np.asarray(batched(transform(cube[valid])), dtype=float)
                result[valid] = np.where(np.isfinite(values), values, -np.inf)
            return result

        # A drawn seed is recorded so that a resumed run reuses it
        seed_path = os.path.join(output, "seed.json")
        if seed is None and os.path.exists(seed_path):
            with open(seed_path) as f:
                seed = json.load(f)["seed"]
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % 2**31)
        if not os.path.exists(seed_path):
            os.makedirs(output, exist_ok=True)
            with open(seed_path, "w") as f:
                json.dump({"seed": int(seed)}, f)
        run = _inessai.sample(
            cube_ln_like, fit_info._get_num_fit_params(), output, n_live,
            int(seed), target_ess, min_final_draws, max_final_draws,
            max_seconds=max_seconds, verbose=verbose, nessai_kwargs=nessai_kwargs)

        stats = _inessai.importance_stats(run["logw"], int(run["n_draws"]))
        finite = np.isfinite(run["logw"])
        draws = transform(run["u"][finite])
        log_weights = run["logw"][finite]
        logl = run["logl"][finite]
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= np.sum(weights)
        logp = logl + _inessai.ln_prior(fit_info, draws)
        best_params_arr = draws[np.argmax(logp)]
        best_ln_prob = np.max(logp)

        # ess (at most target_ess) equal-weight rows: each needs a model
        # evaluation when parameters were integrated out
        n_rows = stats["ess"] if not target_ess else min(stats["ess"], target_ess)
        equal_samples = dynesty.utils.resample_equal(draws, weights)
        np.random.shuffle(equal_samples)
        equal_samples = equal_samples[:max(1, int(n_rows))]
        equal_samples = self._with_nuisance(
            equal_samples, fit_info, full_fit_info, like_args)
        if fit_info is not full_fit_info:
            best_params_arr, best_ln_prob = self._best_with_nuisance(
                best_params_arr, fit_info, full_fit_info, like_args)
        fit_info = full_fit_info
        labels, samples = label_by_temperature(
            fit_info, np.vstack([equal_samples, best_params_arr]))
        divisors, new_labels = self._get_divisors_labels(
            np.median(samples[:-1], axis=0), labels)
        write_param_estimates_file(
            samples[:-1] / divisors, samples[-1] / divisors,
            best_ln_prob, new_labels)

        best = self._ln_like(
            best_params_arr, transit_calc, eclipse_calc, fit_info,
            transit_depths, transit_errors, eclipse_depths, eclipse_errors,
            zero_opacities=zero_opacities, ret_best_fit=True)
        result = {
            "samples": samples,
            "draws": draws,
            "weights": weights,
            "logw": log_weights,
            "logl": logl,
            "logp": logp,
            "logz": np.atleast_1d(stats["logz"]),
            "logz_err": stats["logz_err"],
            "ess": stats["ess"],
            "n_eff": stats["ess"],
            "psis_ess": stats["psis_ess"],
            "pareto_k": stats["pareto_k"],
            "n_draws": stats["n_draws"],
            "n_calls": int(run["n_calls"]),
            "training_logz": float(run["training_logz"]),
            "success": True,
        }
        retrieval_result = RetrievalResult(
            result, "inessai", best_params_arr,
            transit_bins, transit_depths, transit_errors,
            eclipse_bins, eclipse_depths, eclipse_errors,
            best[0], best[1], best[2], best[3],
            fit_info, divisors, new_labels)

        self._init_random_samples(retrieval_result)
        for params in equal_samples[:num_final_samples]:
            pointwise = self._ln_like(
                params, transit_calc, eclipse_calc, fit_info,
                transit_depths, transit_errors,
                eclipse_depths, eclipse_errors,
                zero_opacities=zero_opacities, lnlike_per_point=True)
            _, transit_info, _, eclipse_info = self._ln_like(
                params, transit_calc, eclipse_calc, fit_info,
                transit_depths, transit_errors,
                eclipse_depths, eclipse_errors,
                zero_opacities=zero_opacities, ret_best_fit=True)
            self._record_random_sample(
                retrieval_result, transit_info, eclipse_info, pointwise)

        retrieval_result.loo_total, retrieval_result.loos, \
            retrieval_result.loo_ks = psisloo(
                np.array(retrieval_result.pointwise_lnlikes))
        return retrieval_result

    @staticmethod
    def get_default_fit_info(Rs, Mp, Rp, T=None, logZ=0, CO_ratio=0.53, log_CH4_mult=0,
                             free_retrieval=False,
                             log_cloudtop_P=np.inf, cloud_fraction=1,
                             log_scatt_factor=0,
                             scatt_slope=4, error_excess=0, T_star=None,
                             T_het=None, het_cov_frac=None, T_het2=None, het2_cov_frac=None,
                             frac_scale_height=1,
                             log_number_density=-np.inf, log_part_size=-6,
                             n=None, log_k=-np.inf,
                             log_P_quench=-99,
                             transit_offsets=None, eclipse_offsets=None,
                             fit_vmr=False, fit_clr=False,
                             profile_type = 'isothermal',
                             transit_profile_type = 'isothermal',
                             transit_terminator=None,
                             stellar_grid=None, logg_star=4.5, feh_star=0.,
                             transit_visits=None,
                             time_shifts=None, time_shift_errors=None, period=None, b=None,
                             time_shift_offsets=None, time_shift_coefficients=None,
                             **profile_kwargs):
        '''Get a :class:`.FitInfo` object filled with best guess values.  A few
        parameters are required, but others can be set to default values if you
        do not want to specify them.  All parameters are in SI.  For 
        information on the parameters not described below, see the documentation
        for :func:`~platon.transit_depth_calculator.TransitDepthCalculator.compute_depths` and :func:`~platon.eclipse_depth_calculator.EclipseDepthCalculator.compute_depths`

        Parameters
        ----------
        cloud_fraction : float
            Fraction of the terminator covered by clouds, between 0 and 1.
            Only affects transit depths; see
            :func:`~platon.transit_depth_calculator.TransitDepthCalculator.compute_depths`
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
            errors or scatter the model cannot explain; with a uniform prior
            the retrievers integrate it out of the likelihood.  Errors are
            only ever inflated this way, in quadrature, never by a factor.
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
        time_shifts, time_shift_errors : array, optional
            Fitted mid-transit times minus a reference time, and their errors,
            in seconds, one per transit point from the same bins and light-curve
            fits as the depths.  Require transit_terminator with sector1.fraction
            fixed at 0.5; sector1 is leading (morning), sector2 trailing (evening).
            Also require Rs and a (semi-major axis in m, passed as a keyword,
            as for Guillot profiles), period and b.
        period, b : float, optional
            Orbital period in seconds and impact parameter for time shifts.
        time_shift_offsets : dict, optional
            Like transit_offsets, in seconds, added to model shifts.  Each is
            the true mid-transit time minus the reference time for its range;
            fit it unless the reference is the true mid-transit time.

            The retrievers integrate every fitted offset, and error_excess if
            fitted with a uniform prior, out of the likelihood rather than
            sampling it (see their marginalize_nuisance argument).
        time_shift_coefficients : array, optional
            Calibrated shift per unit trailing-minus-leading radius, in s per
            stellar radius, one per transit point (e.g. from injecting a small
            limb asymmetry into the light-curve fits).  If given, these replace
            the Mukherjee et al. (2026) prescription.
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
        T_het, het_cov_frac, T_het2, het2_cov_frac : float, optional
            Temperatures and covering fractions of up to two unocculted
            stellar heterogeneities (see TransitDepthCalculator).  Eclipse
            depths use only the first.
        stellar_grid, logg_star, feh_star : optional
            Stellar spectra file and the (fixed) stellar log g and [Fe/H] it
            is interpolated to; see TransitDepthCalculator.
        transit_visits : dict, optional
            Visit names mapped to the (start, end) indices of their transit
            data, e.g. {"visit1": (0, 120), "visit2": (120, 176)}.  Each
            visit gets parameters such as "visit1.het_cov_frac" (also T_het, T_het2,
            het2_cov_frac), which default to None, meaning the shared value; fit
            them to let the heterogeneities differ between visits.
        transit_terminator : TwoSectorTerminator, optional
            A cold and hot terminator template for a 1.5-D transit retrieval.
            Its named sector values are added to the returned FitInfo.  For
            Guillot sectors, the shared T_star, Rs, a, Mp, Rp, log_k_th, and
            T_int are taken from the template; any of these also passed here
            must agree with it.
        profile_kwargs : kwargs
            T/P profile arguments.  For "isothermal": T_day.  For "parametric":
            T0, P1, alpha1, alpha2, P3, T3.  For "radiative_solution":
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

        range_names = {}
        for kind, what, ranges in (
                ("transit", "offsets", transit_offsets),
                ("eclipse", "offsets", eclipse_offsets),
                ("time_shift", "offsets", time_shift_offsets)):
            if ranges is None:
                continue
            for name, index_range in ranges.items():
                if name in all_variables and range_names.get(name) != what:
                    raise ValueError(
                        "Name {} in {} {} conflicts with an existing "
                        "parameter".format(name, kind, what))
                if len(index_range) != 2 or \
                   not 0 <= index_range[0] < index_range[1]:
                    raise ValueError(
                        "Range for {} must be (start, end) with "
                        "0 <= start < end".format(name))
                all_variables[name] = 0
                range_names[name] = what

            sorted_ranges = sorted(ranges.items(), key=lambda kv: kv[1][0])
            for (name1, range1), (name2, range2) in zip(
                    sorted_ranges, sorted_ranges[1:]):
                if range1[1] > range2[0]:
                    raise ValueError(
                        "{} {} {} {} and {} {} overlap".format(
                            kind, what, name1, tuple(range1), name2, tuple(range2)))
        
        for visit in transit_visits or {}:
            for name in ("T_het", "het_cov_frac", "T_het2", "het2_cov_frac"):
                all_variables["{}.{}".format(visit, name)] = None

        fit_info = FitInfo(all_variables)
        return fit_info
