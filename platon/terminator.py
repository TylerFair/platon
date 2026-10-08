from dataclasses import dataclass

import numpy as np

from .TP_profile import Profile

# Guillot profile parameters that the cold and hot sectors must share
GUILLOT_SHARED_PARAMS = ("T_star", "Rs", "a", "Mp", "Rp", "log_k_th", "T_int")


@dataclass(frozen=True)
class TerminatorSector:
    """One homogeneous sector of a two-sector terminator."""

    profile: Profile
    cloudtop_pressure: float = np.inf
    scattering_factor: float = 1
    scattering_slope: float = 4

    def __post_init__(self):
        if not isinstance(self.profile, Profile):
            raise TypeError("profile must be a platon.TP_profile.Profile")
        if np.isnan(self.cloudtop_pressure) or self.cloudtop_pressure <= 0:
            raise ValueError("cloudtop_pressure must be positive")
        if not np.isfinite(self.scattering_factor) or self.scattering_factor <= 0:
            raise ValueError("scattering_factor must be positive")
        if not np.isfinite(self.scattering_slope):
            raise ValueError("scattering_slope must be finite")


# Retrievals sample the two sectors under these neutral names, then label
# each posterior sample's colder sector "cold" (see label_by_temperature)
SECTOR_LABELS = ("sector1", "sector2")
SECTOR_FRACTION = "sector1.fraction"
# Sectors are compared by their mean temperature over the layers between
# 0.1 mbar and 1 bar (Pa), roughly where transmission spectra are formed
COMPARISON_PRESSURES = (1e1, 1e5)


def sector_temperature(profile):
    """Mean temperature (K) of a profile between 0.1 mbar and 1 bar.  The
    default pressure grid is log-spaced, so this is a mean in log pressure."""
    pressures = np.asarray(profile.pressures)
    temperatures = np.asarray(profile.temperatures)
    low, high = COMPARISON_PRESSURES
    probed = (pressures >= low) & (pressures <= high)
    if not np.any(probed):
        return float(np.interp(np.log(np.sqrt(low * high)), np.log(pressures),
                               temperatures))
    return float(np.mean(temperatures[probed]))


@dataclass(frozen=True)
class TwoSectorTerminator:
    """A cold and hot terminator sector combined by projected area.  The cold
    sector is the one with the lower mean temperature between 0.1 mbar and
    1 bar (see sector_temperature)."""

    cold: TerminatorSector
    hot: TerminatorSector
    cold_fraction: float = 0.5

    def __post_init__(self):
        if not isinstance(self.cold, TerminatorSector) or \
           not isinstance(self.hot, TerminatorSector):
            raise TypeError("cold and hot must be TerminatorSector objects")
        if not 0 <= self.cold_fraction <= 1:
            raise ValueError("cold_fraction must be between 0 and 1")

        kind = self.cold.profile.profile_type
        if kind != self.hot.profile.profile_type or \
           kind not in ("isothermal", "guillot"):
            raise ValueError(
                "cold and hot profiles must use the same isothermal or "
                "Guillot parameterization")

        if sector_temperature(self.cold.profile) > \
           sector_temperature(self.hot.profile):
            raise ValueError(
                "cold profile must not be hotter than hot profile (compared "
                "by mean temperature between 0.1 mbar and 1 bar)")

        if kind == "guillot":
            for name in GUILLOT_SHARED_PARAMS:
                if not np.isclose(self.cold.profile.profile_params[name],
                                  self.hot.profile.profile_params[name]):
                    raise ValueError(
                        "Guillot sectors must share {}".format(
                            ", ".join(GUILLOT_SHARED_PARAMS)))

    @property
    def profile_type(self):
        return self.cold.profile.profile_type

    @property
    def sector_parameters(self):
        """Names of the parameters each sector has its own value of."""
        profile = ("T",) if self.profile_type == "isothermal" else \
            ("beta", "log_gamma")
        return profile + ("log_cloudtop_P", "log_scatt_factor", "scatt_slope")

    def retrieval_defaults(self):
        """Return the named values used to reconstruct this terminator: the
        cold sector as sector1 and the hot one as sector2."""
        values = {
            "transit_terminator": self,
            SECTOR_FRACTION: self.cold_fraction,
        }
        for label, sector in zip(SECTOR_LABELS, (self.cold, self.hot)):
            values[f"{label}.log_cloudtop_P"] = np.log10(
                sector.cloudtop_pressure)
            values[f"{label}.log_scatt_factor"] = np.log10(
                sector.scattering_factor)
            values[f"{label}.scatt_slope"] = sector.scattering_slope
            if self.profile_type == "isothermal":
                values[f"{label}.T"] = sector.profile.profile_params["T"]
            else:
                values[f"{label}.beta"] = \
                    sector.profile.profile_params["beta"]
                values[f"{label}.log_gamma"] = \
                    sector.profile.profile_params["log_gamma"]

        if self.profile_type == "guillot":
            for name in GUILLOT_SHARED_PARAMS:
                values[name] = self.cold.profile.profile_params[name]
        return values

    def sectors_from_params(self, params):
        """The (sector1, sector2) TerminatorSectors of a retrieval parameter
        dictionary, in sampling order.  Guillot sectors take T_star, Rs, a,
        Mp, Rp, log_k_th, and T_int from params, and beta and log_gamma from
        the sector-prefixed names (e.g. sector1.beta)."""
        sectors = []
        for label in SECTOR_LABELS:
            if self.profile_type == "isothermal":
                profile = Profile.isothermal(params[f"{label}.T"])
            else:
                profile = Profile.guillot(
                    params["T_star"], params["Rs"], params["a"],
                    params["Mp"], params["Rp"],
                    params[f"{label}.beta"],
                    params["log_k_th"],
                    params[f"{label}.log_gamma"],
                    params["T_int"])
            sectors.append(TerminatorSector(
                profile,
                10**params[f"{label}.log_cloudtop_P"],
                10**params[f"{label}.log_scatt_factor"],
                params[f"{label}.scatt_slope"]))
        return sectors

    def _sector1_is_hotter(self, sectors):
        return sector_temperature(sectors[0].profile) > \
            sector_temperature(sectors[1].profile)

    def from_params(self, params):
        """Build a terminator from a retrieval parameter dictionary, with
        whichever of sector1 and sector2 is colder as the cold sector."""
        sectors = self.sectors_from_params(params)
        fraction = params[SECTOR_FRACTION]
        if self._sector1_is_hotter(sectors):
            return TwoSectorTerminator(sectors[1], sectors[0], 1 - fraction)
        return TwoSectorTerminator(sectors[0], sectors[1], fraction)

    def labelled_values(self, params):
        """The sector parameters of a retrieval parameter dictionary named
        by temperature: cold.<name> and hot.<name> for each name in
        sector_parameters, and cold_fraction.  All of a sector's parameters
        move together, so each sample stays self-consistent."""
        swap = self._sector1_is_hotter(self.sectors_from_params(params))
        cold, hot = SECTOR_LABELS[::-1] if swap else SECTOR_LABELS
        values = {"{}.{}".format(new, name): params["{}.{}".format(old, name)]
                  for new, old in (("cold", cold), ("hot", hot))
                  for name in self.sector_parameters}
        fraction = params[SECTOR_FRACTION]
        values["cold_fraction"] = 1 - fraction if swap else fraction
        return values


def label_by_temperature(fit_info, samples):
    """Posterior samples with the two sectors named by temperature.

    Two-sector retrievals sample sector1 and sector2 with independent priors
    and no ordering.  For every sample, this names the sector with the lower
    mean temperature between 0.1 mbar and 1 bar "cold" and the other "hot",
    moving all of that sector's parameters (and its share of the terminator,
    which becomes cold_fraction) with it.  This post-hoc relabelling is the
    usual remedy for label switching in mixture models (e.g. Stephens 2000,
    J. R. Stat. Soc. B 62, 795).

    Returns (names, array); a fit without a terminator is returned as is.
    If only one sector of a parameter is fitted, both labelled versions are
    included, since the fixed value can belong to either.
    """
    samples = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    names = list(fit_info.fit_param_names)
    param = fit_info.all_params.get("transit_terminator")
    terminator = None if param is None else param.best_guess
    if terminator is None:
        return names, samples

    per_sector = {"{}.{}".format(label, name): name for label in SECTOR_LABELS
                  for name in terminator.sector_parameters}
    labelled = []
    for name in names:
        if name in per_sector:
            for new in ("cold.", "hot."):
                if new + per_sector[name] not in labelled:
                    labelled.append(new + per_sector[name])
        elif name == SECTOR_FRACTION:
            labelled.append("cold_fraction")
        else:
            labelled.append(name)
    result = np.empty((len(samples), len(labelled)))
    for i, row in enumerate(samples):
        params = fit_info._interpret_param_array(row)
        params.update(terminator.labelled_values(params))
        result[i] = [params[name] for name in labelled]
    return labelled, result
