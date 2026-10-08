"""Host-side preparation of ForwardConfig/ForwardInputs, shared by the
transit and eclipse depth calculators."""
import math

import numpy as np

from . import _forward_model as fm
from ._forward_model import ForwardConfig, ForwardInputs
from .constants import Teff_sun
from ._stellar_grid import stellar_components
from .errors import AtmosphereError


def _pack_scalars(**kwargs):
    scalars = np.zeros(fm.SC_N_SCALARS)
    for name, value in kwargs.items():
        scalars[fm.SCALAR_INDEX[name]] = value
    return scalars


# Each visit unrolls one more stellar spectrum and binning into the JIT core
MAX_VISITS = 20


def _visit_heterogeneities(atm, T_star, T_het, f_het, T_het2, f_het2,
                           per_bin_het):
    """Group wavelength bins by heterogeneity settings.

    Each of T_het, f_het, T_het2, f_het2 is a scalar (or None), or, when
    per_bin_het is allowed, an array with one value per wavelength bin.  An
    unset temperature means no contrast with the photosphere, and an unset
    fraction means no coverage.
    Returns the distinct (T_het, f_het, T_het2, f_het2) rows in order of
    first appearance, and the visit index of each bin (None for one visit).
    """
    values = (T_het, f_het, T_het2, f_het2)
    if all(np.ndim(value) == 0 for value in values):
        return [values], None
    if not per_bin_het:
        raise ValueError(
            "T_het, f_het, T_het2 and f_het2 must be scalars here; per-bin "
            "(per-visit) heterogeneities are supported for transit depths")
    if atm.wavelength_bins is None:
        raise ValueError(
            "Per-bin heterogeneities need wavelength bins; call "
            "change_wavelength_bins first")
    n_bins = len(atm.wavelength_bins)
    columns = []
    # Without a photosphere no heterogeneity may be active, so any value works
    no_contrast = 0. if T_star is None else T_star
    unset = (no_contrast, 0., no_contrast, 0.)
    for name, value, default in zip(("T_het", "f_het", "T_het2", "f_het2"),
                                    values, unset):
        column = np.asarray(default if value is None else value,
                            dtype=np.float64)
        if column.ndim not in (0, 1) or column.ndim == 1 and \
                column.shape != (n_bins,):
            raise ValueError(
                "{} must be a scalar or have one value per wavelength bin "
                "({}), not shape {}".format(name, n_bins, column.shape))
        columns.append(np.broadcast_to(column, (n_bins,)))
    rows = np.column_stack(columns)
    unique, first, inverse = np.unique(
        rows, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty(len(order), dtype=np.int32)
    rank[order] = np.arange(len(order))
    visit_rows = [tuple(float(x) for x in unique[i]) for i in order]
    if len(visit_rows) == 1:
        return visit_rows, None
    if len(visit_rows) > MAX_VISITS:
        raise ValueError(
            "T_het, f_het, T_het2 and f_het2 take {} different combinations "
            "of values across the bins; per-bin heterogeneities are meant "
            "for a few visits (at most {}), each of which adds to the "
            "compiled model".format(len(visit_rows), MAX_VISITS))
    return visit_rows, rank[inverse.ravel()].astype(np.int32)


def prepare_forward_inputs(atm, *, star_radius, planet_mass, planet_radius,
                           P_profile, T_profile, logZ, CO_ratio, CH4_mult,
                           gases, vmrs, add_gas_absorption,
                           add_H_minus_absorption, add_scattering,
                           scattering_factor, scattering_slope,
                           scattering_ref_wavelength,
                           add_collisional_absorption, cloudtop_pressure,
                           custom_abundances, T_star, T_het, f_het,
                           ri, frac_scale_height, number_density, part_size,
                           part_size_std, P_quench, zero_opacities,
                           stellar_blackbody, bot_pressure,
                           min_abundance=1e-99, min_cross_sec=1e-99,
                           surface_pressure=np.inf, a_over_Rs=0.0,
                           surface_temp=None, redist=0.0, T_het2=None,
                           f_het2=None, logg_star=4.5, logg_het=None,
                           logg_het2=None, feh_star=0., stellar_grid_only=False,
                           validate_T_grid=True, per_bin_het=False):
    """Host-side preparation shared by the transit and eclipse calculators.
    Returns (ForwardConfig, ForwardInputs, host bookkeeping dict)."""
    for name, value in (('star_radius', star_radius), ('planet_mass', planet_mass),
                        ('planet_radius', planet_radius)):
        if not np.isfinite(value) or value <= 0:
            raise AtmosphereError(f'{name} must be finite and positive')
    P_profile = np.asarray(P_profile, dtype=np.float64)
    T_profile = np.asarray(T_profile, dtype=np.float64)
    if (P_profile.ndim != 1 or len(P_profile) < 2 or
            T_profile.shape != P_profile.shape):
        raise ValueError("P_profile and T_profile must be matching 1D arrays with at least two layers")
    if (not np.all(np.isfinite(P_profile)) or np.any(P_profile <= 0) or
            not np.all(np.diff(P_profile) > 0)):
        raise ValueError(
            "P_profile must be finite, positive, and monotonically increasing in pressure")

    # bot_pressure is min(cloudtop_pressure, surface_pressure): the deepest
    # level light can reach, which is what must lie within the pressure grid
    atm._validate_params(T_profile, logZ, CO_ratio, bot_pressure,
                         validate_T_grid=validate_T_grid)

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
            vmrs_arr = np.asarray(vmrs, dtype=np.float64)
            if not len(gases) or vmrs_arr.shape != (len(gases),):
                raise ValueError("vmrs must have one value per gas")
            if len(set(gases)) != len(gases):
                raise ValueError("gases must not contain duplicate species")
            if (np.any(~np.isfinite(vmrs_arr)) or np.any(vmrs_arr < 0) or
                    np.any(vmrs_arr > 1) or not np.any(vmrs_arr > 0)):
                raise ValueError("vmrs must be finite fractions between zero and one with a positive total")
            for gas in gases:
                if gas not in atm.master_index:
                    raise ValueError("Unknown gas: {}".format(gas))
            gas_master_idx = tuple(int(atm.master_index[g]) for g in gases)
            vmrs_arr = vmrs_arr.astype(np.float32)
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

    n_above, shell_mask = atm.get_above_info(P_profile, bot_pressure)

    grid = atm.stellar_grid if T_star is not None and not stellar_blackbody else None
    visit_het, bin_visit = _visit_heterogeneities(
        atm, T_star, T_het, f_het, T_het2, f_het2, per_bin_het)
    visits = [stellar_components(
        grid, T_star, *het, logg_star, logg_het, logg_het2, feh_star,
        stellar_blackbody, stellar_grid_only) for het in visit_het]
    temps, gravities, fractions, in_grid = visits[0]
    T_star, T_het, T_het2 = temps
    f_het, f_het2 = fractions

    scalars = _pack_scalars(
        rs=star_radius, mp=planet_mass, rp=planet_radius,
        logz=0.0 if logZ is None else logZ,
        co=0.0 if CO_ratio is None else CO_ratio,
        log_ch4=math.log10(max(CH4_mult, 1e-99)),
        scat_factor=scattering_factor, scat_slope=scattering_slope,
        scat_ref_um=scattering_ref_wavelength * 1e6,
        cloudtop=cloudtop_pressure,
        p_quench=P_quench,
        log10_p_quench=math.log10(max(P_quench, 1e-99)),
        t_star=0.0 if T_star is None else T_star,
        t_het=0.0 if T_het is None else T_het, f_het=f_het,
        t_het2=0.0 if T_het2 is None else T_het2, f_het2=f_het2,
        fsh=frac_scale_height, num_den=number_density,
        ln_min_xsec=math.log(min_cross_sec),
        log_min_abund=math.log10(min_abundance),
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
    config = ForwardConfig(
        n_layers=len(P_profile),
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
        sort_layers=bool(T_profile.max() > T_profile.min()),
        use_mie=use_mie and add_scattering,
        has_t_star=T_star is not None,
        stellar_in_grid=in_grid[0],
        has_het=any(visit[2][0] != 0.0 for visit in visits),
        has_het2=any(visit[2][1] != 0.0 for visit in visits),
        het_in_grid=tuple(visit[3][1] for visit in visits),
        het2_in_grid=tuple(visit[3][2] for visit in visits),
        n_visits=len(visits),
    )

    # Decode/interpolate only the selected vertices on the host. The full
    # stellar cube never resides on the GPU; only three compact spectra per
    # visit do.  The photosphere is shared, so it is interpolated once.
    stellar_fluxes = None
    if any(any(visit[3]) for visit in visits):
        stellar_fluxes = np.zeros(
            (len(visits), 3, len(grid.wavelengths_m)), np.float32)
        if in_grid[0]:
            stellar_fluxes[:, 0] = grid.interpolate(T_star, gravities[0], feh_star)
        for v, (v_temps, v_gravities, _, v_in_grid) in enumerate(visits):
            for i in (1, 2):
                if v_in_grid[i]:
                    stellar_fluxes[v, i] = grid.interpolate(
                        v_temps[i], v_gravities[i], feh_star)

    inputs = ForwardInputs(
        packed=packed, vmrs=vmrs_arr, custom_log_abund=custom_log_abund,
        eff_xsec=eff_xsec, stellar_fluxes=stellar_fluxes,
        visit_het=None if len(visits) == 1 else np.array(
            [[v_temps[1], v_fractions[0], v_temps[2], v_fractions[1]]
             for v_temps, _, v_fractions, _ in visits], np.float32),
        bin_visit=bin_visit)

    host = dict(n_above=n_above, active_species=active_species,
                P_profile=P_profile, T_profile=T_profile)
    return config, inputs, host


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
