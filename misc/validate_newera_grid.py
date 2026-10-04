"""Measure compact-grid TLS errors against independent MSG flux evaluation.

Requires pymsg only for validation. Save the report with the bundled grid.
The scenarios measure a flat 1% planetary transit, 10% spots, 5% faculae.
They characterize this approximation; they are not a bound for all stars.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from platon._stellar_grid import load_stellar_grid


def validate(source, compact):
    import pymsg
    msg = pymsg.SpecGrid(str(source))
    grid = load_stellar_grid(compact)
    edges = np.linspace(msg.lam_min, msg.lam_max, 139501)
    centers_m = .5 * (edges[1:] + edges[:-1]) * 1e-10
    scenarios = ((3500., 2800., 4200., 4.5, 0.),
                 (5000., 4000., 6000., 4.5, 0.),
                 (6000., 4500., 7000., 4., -.5),
                 (8000., 6000., 10000., 4., .5),
                 (3000., 2500., 3500., 5., -1.))
    report = {'depth': .01, 'spot_fraction': .1, 'facula_fraction': .05,
              'comparison': 'MSG order=1, native 2 Angstrom bins, photon-weighted TLS',
              'platon_transit_comparison': 'F_lambda weights on a uniform log-wavelength grid, matching existing transit binning',
              'scenarios': []}
    for star, spot, fac, gravity, feh in scenarios:
        native, reduced = [], []
        for temp in (star, spot, fac):
            x = {'Teff': temp, '[Fe/H]': feh, 'log(g)': gravity}
            f = msg.flux(x=x, z=0., lam=edges, order=1) * 1e7
            native.append(f)
            reduced.append(np.interp(centers_m, grid.wavelengths_m, grid.interpolate(temp, gravity, feh)))
        native, reduced = np.asarray(native), np.asarray(reduced)
        mix_native = .85 * native[0] + .1 * native[1] + .05 * native[2]
        mix_reduced = .85 * reduced[0] + .1 * reduced[1] + .05 * reduced[2]
        # Native source has constant F_lambda within each source bin, so
        # integrate F_lambda * lambda analytically at arbitrary observation edges.
        def integral(flux, target, log_weight=False):
            cumulative = np.r_[0., np.cumsum(flux * (np.diff(np.log(edges)) if log_weight else .5 * np.diff(edges**2)))]
            idx = np.clip(np.searchsorted(edges, target, side='right') - 1, 0, len(flux) - 1)
            return cumulative[idx] + flux[idx] * (np.log(target / edges[idx]) if log_weight else .5 * (target**2 - edges[idx]**2))
        row = dict(T_star=star, T_spot=spot, T_fac=fac, logg=gravity, feh=feh, errors={}, platon_transit_errors={})
        for R in (100, 300, 1000):
            errors, platon_errors = [], []
            for phase in (0., .25, .5, .75):
                be = np.exp(np.arange(np.log(edges[0]) + phase / R, np.log(edges[-1]), 1 / R))
                ref = .01 * np.diff(integral(native[0], be)) / np.diff(integral(mix_native, be))
                approx = .01 * np.diff(integral(reduced[0], be)) / np.diff(integral(mix_reduced, be))
                errors.extend(np.abs(approx - ref) * 1e6)
                ref_log = .01 * np.diff(integral(native[0], be, True)) / np.diff(integral(mix_native, be, True))
                approx_log = .01 * np.diff(integral(reduced[0], be, True)) / np.diff(integral(mix_reduced, be, True))
                platon_errors.extend(np.abs(approx_log - ref_log) * 1e6)
            row['platon_transit_errors'][str(R)] = dict(max_ppm=float(np.max(platon_errors)), p95_ppm=float(np.percentile(platon_errors, 95)))
            row['errors'][str(R)] = dict(max_ppm=float(np.max(errors)), p95_ppm=float(np.percentile(errors, 95)))
        report['scenarios'].append(row)
    # Also confirm the builder's units and node mapping against MSG on compact edges.
    with np.load(compact) as data:
        be = data['wavelength_edges_m'] * 1e10
    # Avoid rounding a boundary outside the strict MSG wavelength domain.
    be[0], be[-1] = msg.lam_min, msg.lam_max
    reference = msg.flux(x={'Teff': 5000., '[Fe/H]': 0., 'log(g)': 4.5}, z=0., lam=be, order=1) * 1e7
    report['msg_node_max_relative_error'] = float(np.max(np.abs(grid.interpolate(5000., 4.5, 0.) / reference - 1)))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('compact', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.source, args.compact)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
