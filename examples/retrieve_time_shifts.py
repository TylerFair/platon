"""Retrieve morning/evening atmospheres from depths and mid-transit times."""
import matplotlib.pyplot as plt
import numpy as np

from platon.combined_retriever import CombinedRetriever
from platon.constants import AU, M_jup, R_jup, R_sun
from platon.terminator import TerminatorSector, TwoSectorTerminator, limb_time_shifts
from platon.TP_profile import Profile
from platon.transit_depth_calculator import TransitDepthCalculator


# All quantities are in SI
Rs = 1.16 * R_sun
Mp = 0.73 * M_jup
Rp = 1.40 * R_jup
a, period, b = 0.047 * AU, 3.52 * 86400, 0.5
terminator = TwoSectorTerminator(
    TerminatorSector(Profile.isothermal(900), cloudtop_pressure=1e3),
    TerminatorSector(Profile.isothermal(1400), cloudtop_pressure=1e6))

# Synthetic JWST-like bins from 0.8 to 5.3 um
edges = np.geomspace(0.8e-6, 5.3e-6, 31)
bins = np.column_stack([edges[:-1], edges[1:]])
calculator = TransitDepthCalculator()
calculator.change_wavelength_bins(bins)
wavelengths, true_depths, info = calculator.compute_depths(
    terminator, Rs, Mp, Rp, T_star=6100)
true_shifts = limb_time_shifts(
    info["cold_depths"], info["hot_depths"], Rs, a, period, b)

rng = np.random.default_rng(42)
errors = np.full(len(bins), 50e-6)
time_shift_errors = np.full(len(bins), 5.)
depths = true_depths + rng.normal(0, errors)
time_shifts = true_shifts + 10 + rng.normal(0, time_shift_errors)

retriever = CombinedRetriever()
fit_info = retriever.get_default_fit_info(
    Rs=Rs, Mp=Mp, Rp=Rp, T_star=6100, transit_terminator=terminator,
    time_shifts=time_shifts, time_shift_errors=time_shift_errors,
    a=a, period=period, b=b,
    time_shift_offsets={"t0_offset": (0, len(bins))})
for sector in ("sector1", "sector2"):
    fit_info.add_uniform_fit_param(sector + ".T", 500, 2000)
    fit_info.add_uniform_fit_param(sector + ".log_cloudtop_P", 0, 7)
fit_info.add_uniform_fit_param("t0_offset", -100, 100)

result = retriever.run_dynesty(
    bins, depths, errors, None, None, None, fit_info, nlive=50)
model_shifts = result.best_fit_transit_dict["time_shifts"]
print("Wavelength (um), measured shift (s), best-fit shift (s)")
print(np.column_stack([1e6 * wavelengths, time_shifts, model_shifts]))
plt.errorbar(1e6 * wavelengths, time_shifts, yerr=time_shift_errors,
             fmt=".", label="data")
plt.plot(1e6 * wavelengths, model_shifts, label="best fit")
plt.xlabel("Wavelength (um)")
plt.ylabel("Mid-transit time shift (s)")
plt.legend()
plt.tight_layout()
plt.show()
