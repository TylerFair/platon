"""Small synthetic grids and profiles shared by offline tests."""
import json
from pathlib import Path

import numpy as np
import pytest
from misc.build_newera_grid import encode_log_flux
from platon._stellar_grid import grid_from_dict
from platon.constants import M_jup, R_jup, R_sun


_FIXTURES = Path(__file__).parent / 'fixtures'
_TLS_REFERENCE = json.loads((_FIXTURES / 'cool_star_tls_native.json').read_text())
_OPACITY_REFERENCE = json.loads((_FIXTURES / 'eight_gas_guillot_opacity.json').read_text())
STELLAR_PARAMS = _OPACITY_REFERENCE['stellar_parameters']
assert STELLAR_PARAMS == _TLS_REFERENCE['parameters']
RS = _OPACITY_REFERENCE['geometry']['Rs_Rsun'] * R_sun
MP = _OPACITY_REFERENCE['geometry']['Mp_Mjup'] * M_jup
RP = _OPACITY_REFERENCE['geometry']['Rp_Rjup'] * R_jup
TRACE_GASES = tuple(_OPACITY_REFERENCE['gases'])
TRACE_LOG_VMRS = np.asarray(_OPACITY_REFERENCE['log_vmrs'], dtype=float)


def synthetic_grid(encoded=False):
    temps = np.array([3000., 5000.])
    loggs = np.array([4., 5.])
    fehs = np.array([-.5, .5])
    waves = np.array([.7, 1., 2., 5.]) * 1e-6
    flux = (temps[:, None, None, None] * 1e6 +
            loggs[None, :, None, None] * 1e9 +
            fehs[None, None, :, None] * 1e9 +
            waves[None, None, None, :] * 1e15)
    data = dict(temperatures=temps, loggs=loggs, fehs=fehs,
                wavelengths_m=waves, spectra=flux)
    if encoded:
        q = np.empty(flux.shape, np.uint16)
        off, step = np.empty(flux.shape[:-1]), np.empty(flux.shape[:-1])
        for idx in np.ndindex(flux.shape[:-1]):
            q[idx], off[idx], step[idx] = encode_log_flux(flux[idx])
        del data['spectra']
        data.update(encoded_spectra=q, log_offset=off, log_scale=step)
    return grid_from_dict(data)


def make_native_h5(path):
    h5py = pytest.importorskip('h5py')
    with h5py.File(path, 'w') as f:
        f.attrs['REVISION'] = 1
        f.attrs['label'] = np.bytes_('Synthetic native test')
        vg = f.create_group('vgrid')
        for i, (label, values) in enumerate((('Teff', [3000., 5000.]),
                                            ('[Fe/H]', [0.]), ('log(g)', [4.5])), start=1):
            a = vg.create_group(f'axes[{i}]')
            a.attrs['label'] = np.bytes_(label)
            a.create_dataset('x', data=values)
        vg.create_dataset('v_lin_seq', data=[1, 2])
        ss = f.create_group('specsource')
        for i in (1, 2):
            sp = ss.create_group(f'specints[{i}]')
            sp.create_dataset('c', data=np.arange(1., 41.)[:, None] * i)
            limb = sp.create_group('limb')
            limb.attrs['law'] = np.bytes_('CONST')
            r = sp.create_group('range')
            r.attrs.update(TYPE=np.bytes_('lin_range_t'), x_0=6000., dx=2., n=41)


def tiny_profile(T=1000.):
    """Isothermal profile on 12 layers, for fast offline forward models."""
    from platon.TP_profile import Profile
    return Profile(np.geomspace(1e-4, 1e8, 12), np.full(12, T))
