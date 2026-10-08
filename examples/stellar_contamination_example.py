import numpy as np
import matplotlib.pyplot as plt

from platon.transit_depth_calculator import TransitDepthCalculator
from platon.TP_profile import Profile
from platon.constants import M_jup, R_sun, R_jup

# Made-up M dwarf and warm Saturn; all quantities in SI
Rs = 0.40 * R_sun
Mp = 0.30 * M_jup
Rp = 0.80 * R_jup
T = 700

p = Profile.isothermal(T)

# PRISM-like bins from 0.6 to 5.3 um, resolving power about 100
edges = np.geomspace(0.6e-6, 5.3e-6, int(100*np.log(5.3/0.6)) + 1)
bins = np.column_stack((edges[:-1], edges[1:]))

# NewEra downloads automatically on first use
calculator = TransitDepthCalculator()
calculator.change_wavelength_bins(bins)
# logg_star is log10 gravity in cgs; feh_star is [Fe/H] in dex.  Each
# heterogeneity may be cooler (spots) or hotter (faculae) than the photosphere
star = dict(T_star=3400, logg_star=4.9, feh_star=0.0)
wavelengths, clean, _ = calculator.compute_depths(
    p, Rs, Mp, Rp, logZ=0, CO_ratio=0.53, **star)
_, spotted, _ = calculator.compute_depths(
    p, Rs, Mp, Rp, logZ=0, CO_ratio=0.53,
    T_het=3000, f_het=0.05, **star)
_, mixed, _ = calculator.compute_depths(
    p, Rs, Mp, Rp, logZ=0, CO_ratio=0.53,
    T_het=3000, f_het=0.05, T_het2=3600, f_het2=0.03, **star)

plt.plot(1e6*wavelengths, 1e6*clean, label="Clean")
plt.plot(1e6*wavelengths, 1e6*spotted, label="5% at 3000 K (spots)")
plt.plot(1e6*wavelengths, 1e6*mixed, label="+ 3% at 3600 K (faculae)")
plt.xlabel("Wavelength (um)")
plt.ylabel("Transit depth (ppm)")
plt.legend()
plt.show()
