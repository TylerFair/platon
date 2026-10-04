# Synthetic truth: Rs=0.40 R_sun, Mp=0.30 M_jup, Rp=0.80 R_jup, T=700 K.
# Solar equilibrium chemistry; T_star=3400 K, logg=4.9, [Fe/H]=0.0.
# T_spot=3000 K, spot coverage=0.05, no faculae; G395H offset=+200 ppm.
import numpy as np
from pathlib import Path

from platon.transit_depth_calculator import TransitDepthCalculator
from platon.TP_profile import Profile
from platon.constants import M_jup, R_sun, R_jup

p = Profile()
p.set_isothermal(700)
calculator = TransitDepthCalculator()
rng = np.random.default_rng(42)
data_dir = Path(__file__).resolve().parent

# Explicit bin edges in um, R about 100; preserve the G395H detector gap
instruments = [("niriss_soss_example.csv", [(0.85, 2.80)], 150, 0),
               ("nirspec_g395h_example.csv", [(2.87, 3.72), (3.82, 5.18)], 100, 200)]
for filename, intervals, error, offset in instruments:
    bins = []
    for low, high in intervals:
        edges = np.geomspace(low, high, int(100*np.log(high/low)) + 1)
        bins.extend(zip(edges[:-1], edges[1:]))
    bins = np.round(np.array(bins), 4)
    calculator.change_wavelength_bins(1e-6*bins)
    _, depths, _ = calculator.compute_depths(
        p, 0.40*R_sun, 0.30*M_jup, 0.80*R_jup, logZ=0, CO_ratio=0.53,
        T_star=3400, T_spot=3000, spot_cov_frac=0.05, logg_phot=4.9, feh=0.0)
    noisy_depths = 1e6*depths + offset + rng.normal(0, error, len(depths))
    rows = np.column_stack((bins, noisy_depths, np.full(len(depths), error)))
    np.savetxt(data_dir / filename, rows, delimiter=",",
               header="wavelength_low,wavelength_high,depth,error", comments="",
               fmt=["%.4f", "%.4f", "%.1f", "%.1f"])
