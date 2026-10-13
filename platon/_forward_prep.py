"""Host-side preparation of ForwardConfig/ForwardInputs, shared by the
transit and eclipse depth calculators."""
import math
import inspect

import numpy as np

from . import _forward_model as fm
from ._forward_model import ForwardConfig, ForwardInputs
from ._atmosphere_solver import AtmosphereSolver
from .constants import Teff_sun


def _pack_scalars(**kwargs):
    shape = np.broadcast_shapes(*(np.shape(v) for v in kwargs.values()))
    scalars = np.zeros(shape + (fm.SC_N_SCALARS,))
    for name, value in kwargs.items():
        scalars[..., fm.SCALAR_INDEX[name]] = value
    return scalars


def _log(value, base10=False, floor=False):
    if floor:
        value = np.maximum(value, 1e-99) if np.ndim(value) else max(value, 1e-99)
    if np.ndim(value):
        return (np.log10 if base10 else np.log)(value)
    return math.log10(value) if base10 else math.log(value)


def prepare_forward_inputs(atm, *, star_radius, planet_mass, planet_radius,
                           P_profile, T_profile, logZ, CO_ratio, CH4_mult,
                           gases, vmrs, add_gas_absorption,
                           add_H_minus_absorption, add_scattering,
                           scattering_factor, scattering_slope,
                           scattering_ref_wavelength,
                           add_collisional_absorption, cloudtop_pressure,
                           custom_abundances, T_star, T_het, het_cov_frac,
                           ri, frac_scale_height, number_density, part_size,
                           part_size_std, P_quench, zero_opacities,
                           stellar_blackbody, bot_pressure,
                           min_abundance=1e-99, min_cross_sec=1e-99,
                           surface_pressure=np.inf, a_over_Rs=0.0,
                           surface_temp=None, redist=0.0, T_het2=None,
                           het2_cov_frac=None, _batch=False):
    """Host-side preparation shared by the transit and eclipse calculators.
    Returns (ForwardConfig, ForwardInputs, host bookkeeping dict)."""
    # bot_pressure is min(cloudtop_pressure, surface_pressure): the deepest
    # level light can reach, which is what must lie within the pressure grid
    if not _batch:
        atm._validate_params(T_profile, logZ, CO_ratio, bot_pressure)

    P_profile = np.asarray(P_profile, dtype=np.float64)
    T_profile = np.asarray(T_profile, dtype=np.float64)
    if not np.all(np.diff(P_profile, axis=-1) > 0):
        raise ValueError(
            "P_profile must be monotonically increasing in pressure")

    # Abundance mode
    vmrs_arr = None
    custom_log_abund = None
    gas_master_idx = ()
    if custom_abundances is None and logZ is not None and CO_ratio is not None:
        abund_mode = "eq"
        active_species = list(atm.abundance_getter.included_species)
    else:
        if logZ is not None or CO_ratio is not None:
            raise ValueError(
                "Must set logZ=None and CO_ratio=None to use custom_abundances")
        if custom_abundances is not None:
            if isinstance(custom_abundances, str):
                from .abundance_getter import AbundanceGetter
                custom_abundances = AbundanceGetter.from_file(custom_abundances)
            if not isinstance(custom_abundances, dict):
                raise ValueError("Unrecognized format for custom_abundances")
            abund_mode = "custom"
            custom_log_abund = atm.custom_abundances_to_log_master(
                custom_abundances, T_profile, P_profile)
            active_species = list(custom_abundances.keys())
        elif vmrs is not None and gases is not None:
            abund_mode = "vmr"
            for gas in gases:
                if gas not in atm.master_index:
                    raise ValueError("Unknown gas: {}".format(gas))
            gas_master_idx = tuple(int(atm.master_index[g]) for g in gases)
            vmrs_arr = np.asarray(vmrs, dtype=np.float32)
            active_species = list(gases)
        else:
            raise ValueError("Unrecognized format for custom_abundances")

    add_H_minus_absorption = bool(add_H_minus_absorption)
    if add_H_minus_absorption and \
       ("el" not in active_species or "H" not in active_species):
        raise ValueError(
            "add_H_minus_absorption requires 'el' and 'H' abundances, which "
            "are missing from the provided gases/custom_abundances")

    # Mie scattering
    eff_xsec = None
    mie_ref_P = 1.0
    use_mie = ri is not None
    if use_mie:
        if scattering_factor != 1 or scattering_slope != 4:
            raise ValueError(
                "Cannot use both parametric and Mie scattering at the same time")
        eff_xsec = atm.get_mie_eff_cross_section(
            ri, part_size, sigma=part_size_std)   # float32
        mie_ref_P = atm.get_mie_ref_pressure(P_profile, bot_pressure)

    opac_mask = np.ones(len(atm.raw["opac_names"]), dtype=np.float32)
    for name in zero_opacities:
        if name in atm.raw["opac_names"]:
            opac_mask[atm.raw["opac_names"].index(name)] = 0

    if _batch:
        above = P_profile < np.asarray(bot_pressure)[..., None]
        n_above = above.sum(axis=-1)
        shell_mask = above[:, 1:].astype(np.float32)
    else:
        n_above, shell_mask = atm.get_above_info(P_profile, bot_pressure)

    if T_het is None:
        T_het = T_star
    if het_cov_frac is None:
        het_cov_frac = 0.0
    T_het2 = T_star if T_het2 is None else T_het2
    het2_cov_frac = 0.0 if het2_cov_frac is None else het2_cov_frac

    # Static stellar-spectrum branches (comparison in float32 to match what
    # the device would compute)
    stellar_temps = atm.raw["stellar_temps"]
    stellar_in_grid = False if T_star is None or stellar_blackbody else \
        (stellar_temps[0] <= np.asarray(T_star, dtype=np.float32)) & \
        (np.asarray(T_star, dtype=np.float32) <= stellar_temps[-1])

    scalars = _pack_scalars(
        rs=star_radius, mp=planet_mass, rp=planet_radius,
        logz=0.0 if logZ is None else logZ,
        co=0.0 if CO_ratio is None else CO_ratio,
        log_ch4=_log(CH4_mult, base10=True, floor=True),
        scat_factor=scattering_factor, scat_slope=scattering_slope,
        scat_ref_um=scattering_ref_wavelength * 1e6,
        cloudtop=cloudtop_pressure,
        p_quench=P_quench,
        log10_p_quench=_log(P_quench, base10=True, floor=True),
        t_star=0.0 if T_star is None else T_star,
        t_het=0.0 if T_het is None else T_het,
        het_frac=het_cov_frac,
        t_het2=0.0 if T_het2 is None else T_het2, het2_frac=het2_cov_frac,
        fsh=frac_scale_height, num_den=number_density,
        ln_min_xsec=_log(min_cross_sec),
        log_min_abund=_log(min_abundance, base10=True),
        ref_pressure=atm.ref_pressure,
        t_star_hydro=Teff_sun if T_star is None else T_star,
        mie_ref_p=mie_ref_P,
        surface_p=surface_pressure,
        a_over_rs=a_over_Rs,
        surface_temp=0.0 if surface_temp is None else surface_temp,
        redist=redist,
    )

    # Pack all small per-call arrays into one vector: a single host-to-device
    # transfer per call (see ForwardInputs)
    packed = fm.pack_inputs(scalars, n_above - 1, T_profile, P_profile,
                            shell_mask, opac_mask)

    # el/H indices are only meaningful (and validated above) when H-
    # absorption is on; -1 makes any unintended use fail loudly downstream
    config_values = dict(
        n_layers=T_profile.shape[-1],
        abund_mode=abund_mode,
        gas_master_idx=gas_master_idx,
        ch4_idx=int(atm.master_index.get("CH4", -1)),
        el_idx=int(atm.master_index["el"]) if add_H_minus_absorption else -1,
        h_idx=int(atm.master_index["H"]) if add_H_minus_absorption else -1,
        add_gas=bool(add_gas_absorption),
        add_hminus=add_H_minus_absorption,
        add_scattering=bool(add_scattering),
        add_collisional=bool(add_collisional_absorption),
        # Layer sorting only pays off when layers span multiple T-grid rows
        # (see _opacity); exactly isothermal profiles skip it
        sort_layers=T_profile.max(axis=-1) > T_profile.min(axis=-1),
        use_mie=use_mie and add_scattering,
        has_t_star=T_star is not None,
        stellar_in_grid=stellar_in_grid,
        has_het=het_cov_frac != 0.0,
        has_het2=het2_cov_frac != 0.0,
    )

    if _batch:
        config = [ForwardConfig(**{
            k: (bool(v[i]) if isinstance(v, np.ndarray) else v)
            for k, v in config_values.items()}) for i in range(len(T_profile))]
    else:
        config = ForwardConfig(**config_values)

    inputs = ForwardInputs(
        packed=packed, vmrs=vmrs_arr, custom_log_abund=custom_log_abund,
        eff_xsec=eff_xsec)

    host = dict(n_above=n_above, active_species=active_species,
                P_profile=P_profile, T_profile=T_profile)
    if _batch:
        return [(cfg, ForwardInputs(*(None if x is None else x[i] for x in inputs)),
                 dict(host, n_above=int(n_above[i]), P_profile=P_profile[i],
                      T_profile=T_profile[i])) for i, cfg in enumerate(config)]
    return config, inputs, host


# Arguments that control Python branches; numerical parameters are stacked.
_PREP_STATIC = (
    "gases", "add_gas_absorption", "add_H_minus_absorption", "add_scattering",
    "add_collisional_absorption", "zero_opacities", "stellar_blackbody",
    "custom_abundances", "ri")
_PREP_DEFAULTS = {k: p.default for k, p in
                  inspect.signature(prepare_forward_inputs).parameters.items()
                  if p.default is not inspect.Parameter.empty}


def prepare_forward_inputs_batch(calls):
    """Vectorize scalar packing and masks; retain scalar Mie/custom prep.

    Validation errors are sent back to the requesting row, so the existing
    AtmosphereError handler determines validity. Configuration errors keep
    their original exception type and message.
    """
    results = [None] * len(calls)
    groups = {}
    for i, call in enumerate(calls):
        args = dict(_PREP_DEFAULTS, **call.kwargs)
        atm = call.args[0]
        if args["ri"] is not None or args["custom_abundances"] is not None or \
           args["min_cross_sec"] <= 0 or args["min_abundance"] <= 0 or \
           getattr(atm._validate_params, "__func__", None) is not AtmosphereSolver._validate_params or \
           getattr(atm.get_above_info, "__func__", None) is not AtmosphereSolver.get_above_info:
            try:
                results[i] = call.run()
            except Exception as error:
                results[i] = error
            continue
        key = tuple((k, tuple(v) if k in ("gases", "zero_opacities") and v is not None else v)
                    for k, v in args.items() if k in _PREP_STATIC)
        key += tuple((k, v is None) for k, v in args.items())
        key += (np.shape(args["T_profile"]),)
        groups.setdefault(key, []).append((i, args))
    for members in groups.values():
        first = members[0][1]
        kwargs = {k: (v if k in _PREP_STATIC or v is None or k == "_batch" else
                      np.asarray([a[k] for _, a in members]))
                  for k, v in first.items()}
        atm = calls[members[0][0]].args[0]
        T, P, bot = kwargs["T_profile"], kwargs["P_profile"], kwargs["bot_pressure"]
        bad = (T.min(axis=-1) < atm.min_temperature) | (T.max(axis=-1) > atm.max_temperature)
        for name, grid in (("logZ", atm.abundance_getter.logZs),
                           ("CO_ratio", atm.abundance_getter.CO_ratios)):
            if kwargs[name] is not None:
                bad |= (kwargs[name] < float(grid.min())) | (kwargs[name] > float(grid.max()))
        bad |= ~np.isinf(bot) & ((bot <= float(atm.P_grid.min())) | (bot > float(atm.P_grid.max())))
        bad |= ~np.all(np.diff(P, axis=-1) > 0, axis=-1)
        bad |= ~np.any(P < bot[:, None], axis=-1)
        for j in np.flatnonzero(bad):
            i = members[j][0]
            try:
                results[i] = calls[i].run()
            except Exception as error:
                results[i] = error
        members = [member for j, member in enumerate(members) if not bad[j]]
        if not members:
            continue
        kwargs = {k: v[~bad] if isinstance(v, np.ndarray) and k not in _PREP_STATIC else v
                  for k, v in kwargs.items()}
        kwargs["_batch"] = True
        try:
            values = prepare_forward_inputs(calls[members[0][0]].args[0], **kwargs)
        except Exception:
            # Preserve row-specific configuration errors, including ValueError.
            values = []
            for i, _ in members:
                try:
                    values.append(calls[i].run())
                except Exception as error:
                    values.append(error)
        for (i, _), value in zip(members, values):
            results[i] = value
    return results


def atm_info_dict(atm, out, host):
    """Build the backward-compatible full_output info dict entries shared by
    both calculators (arrays truncated to the above-cloud region)."""
    n = host["n_above"]
    atm_out = out.atm
    atm_abund = np.array(atm_out.atm_abund)      # (N, M)
    abundances = {}
    for name in host["active_species"]:
        idx = atm.master_index[name]
        abundances[name] = atm_abund[:n, idx]
    return dict(
        absorption_coeff_atm=np.array(atm_out.absorption_coeff_atm)[:n],
        radii=np.array(atm_out.radii)[:n],
        dr=np.array(atm_out.dr)[:n - 1],
        P_profile=np.array(host["P_profile"])[:n],
        T_profile=np.array(host["T_profile"])[:n],
        mu_profile=np.array(atm_out.mu_profile),
        atm_abundances=abundances,
    )
