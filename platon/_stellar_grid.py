"""Compact stellar grids: interpolate eight model vertices, then wavelengths.

Spectra remain on their own wavelength grid. uint16 log encoding is decoded
only for selected vertices. Calculators send interpolated spectra to the
device while the model cube stays in a lazy two-slice host cache.
"""
from dataclasses import dataclass
from collections import OrderedDict
from threading import RLock
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import pickle
import warnings

import numpy as np

from .errors import AtmosphereError


_blackbody_warnings = set()


class ShardedSpectra:
    """Read only two metallicity slices; each is an immutable uint16 array.

    The lock protects the LRU in threaded retrievals. Pickling drops cached
    slices and reconstructs the lock, so multiprocessing workers load lazily.
    """
    def __init__(self, paths, shape):
        self.paths = tuple(Path(p) for p in paths)
        self.shape = tuple(shape)
        self.dtype = np.dtype(np.uint16)
        self._cache = OrderedDict()
        self._lock = RLock()

    @property
    def nbytes(self):
        return int(np.prod(self.shape)) * self.dtype.itemsize

    @property
    def resident_nbytes(self):
        with self._lock:
            return sum(a.nbytes for a in self._cache.values())

    def __getitem__(self, index):
        t, g, z = index
        with self._lock:
            if z in self._cache:
                cube = self._cache.pop(z)
            else:
                with np.load(self.paths[z], allow_pickle=False) as data:
                    cube = data['encoded_spectra']
                expected = self.shape[:2] + self.shape[-1:]
                if cube.dtype != self.dtype or cube.shape != expected:
                    raise ValueError(f'Invalid stellar shard {self.paths[z]}')
                cube.setflags(write=False)
            self._cache[z] = cube
            while len(self._cache) > 2:
                self._cache.popitem(last=False)
            return cube[t, g]

    def __getstate__(self):
        return {'paths': self.paths, 'shape': self.shape}

    def __setstate__(self, state):
        self.__init__(state['paths'], state['shape'])


@dataclass(frozen=True)
class StellarGrid:
    temperatures: np.ndarray
    loggs: np.ndarray
    fehs: np.ndarray
    wavelengths_m: np.ndarray
    spectra: np.ndarray       # (T, logg, feh, wavelength), float32 or uint16
    valid: np.ndarray
    log_offset: np.ndarray | None = None
    log_scale: np.ndarray | None = None
    metadata: dict | None = None

    def validate(self, temperature, logg, feh):
        self._interpolation_vertices(temperature, logg, feh)

    def _interpolation_vertices(self, temperature, logg, feh):
        """Repair an isolated missing temperature row from its valid neighbors."""
        brackets = []
        ignored = (self.metadata or {}).get('ignored_parameter_axes', ())
        for name, value, axis in zip(('Teff', 'logg', 'feh'),
                                     (temperature, logg, feh),
                                     (self.temperatures, self.loggs, self.fehs)):
            if not np.isfinite(value) or (name not in ignored and
                                          (value < axis[0] or value > axis[-1])):
                raise AtmosphereError(f'Stellar {name}={value} is outside [{axis[0]}, {axis[-1]}]')
            brackets.append(axis_bracket(value, axis))
        def row_vertices(t):
            vertices = []
            for sides in product((0, 1), repeat=2):
                indices = (t,) + tuple(b[side] for b, side in zip(brackets[1:], sides))
                weight = np.prod([b[2] if side else 1 - b[2]
                                  for b, side in zip(brackets[1:], sides)])
                if weight > 0:
                    vertices.append((indices, weight))
            return vertices

        repair = (self.metadata or {}).get('missing_model_policy') == 'interpolate_isolated_temperature'
        t_rows = [(brackets[0][0], 1 - brackets[0][2]), (brackets[0][1], brackets[0][2])]

        def missing_model_error(*repair_rows):
            needed = [index for t, weight in t_rows if weight > 0
                      for index, _ in row_vertices(t)]
            needed += [index for row in repair_rows for index, _ in row]
            missing = sorted({index for index in needed if not self.valid[index]})
            nodes = ', '.join(f'({self.temperatures[t]:g} K, {self.loggs[g]:g}, {self.fehs[z]:g})'
                              for t, g, z in missing)
            return AtmosphereError(
                f'Requested stellar interpolation (Teff, logg, [Fe/H]) = '
                f'({temperature:g} K, {logg:g}, {feh:g}) includes missing NewEra '
                f'model nodes: {nodes}')

        result = []
        for t, weight in t_rows:
            if weight == 0:
                continue
            vertices = row_vertices(t)
            if not all(self.valid[index] for index, _ in vertices):
                if repair and 0 < t < len(self.temperatures) - 1:
                    lower, upper = row_vertices(t - 1), row_vertices(t + 1)
                    if all(self.valid[index] for index, _ in lower + upper):
                        fraction = ((self.temperatures[t] - self.temperatures[t - 1]) /
                                    (self.temperatures[t + 1] - self.temperatures[t - 1]))
                        vertices = ([(index, w * (1 - fraction)) for index, w in lower] +
                                    [(index, w * fraction) for index, w in upper])
                    else:
                        raise missing_model_error(lower, upper)
                else:
                    raise missing_model_error()
            result.extend((index, w * weight) for index, w in vertices)
        return result

    def interpolate(self, temperature, logg=4.5, feh=0.):
        result = np.zeros(len(self.wavelengths_m), np.float64)
        for indices, weight in self._interpolation_vertices(temperature, logg, feh):
            row = self.spectra[indices].astype(np.float64)
            if self.log_offset is not None:
                row = np.exp(self.log_offset[indices] + self.log_scale[indices] * row)
            result += weight * row
        return result


def axis_bracket(value, axis):
    if len(axis) == 1:
        return 0, 0, 0.
    lower = int(np.clip(np.searchsorted(axis, value, side='right') - 1, 0, len(axis) - 2))
    fraction = float(np.clip((value - axis[lower]) / (axis[lower + 1] - axis[lower]), 0., 1.))
    return lower, lower + 1, fraction


def wavelength_brackets(target, source):
    idx = np.clip(np.searchsorted(source, target, side='right') - 1, 0, len(source) - 2)
    frac = np.clip((target - source[idx]) / (source[idx + 1] - source[idx]), 0., 1.)
    return idx.astype(np.int32), frac.astype(np.float32)


def grid_from_dict(data, legacy_wavelengths=None):
    temps = np.asarray(data['temperatures'], np.float64)
    loggs = np.asarray(data.get('loggs', [4.5]), np.float64)
    fehs = np.asarray(data.get('fehs', [0.]), np.float64)
    waves = np.asarray(data.get('wavelengths_m', legacy_wavelengths), np.float64)
    encoded = 'encoded_spectra' in data
    spectra = data['encoded_spectra'] if encoded else data['spectra']
    if not isinstance(spectra, ShardedSpectra):
        spectra = np.asarray(spectra, dtype=np.uint16 if encoded else np.float32)
    # Legacy pickle layouts: (logg, T, wavelength) or (T, wavelength).
    ignored_axes = []
    if not isinstance(spectra, ShardedSpectra) and spectra.ndim == 2:
        ignored_axes = [name for name, key in (('logg', 'loggs'), ('feh', 'fehs'))
                        if key not in data]
        spectra = spectra[:, None, None, :]
    elif not isinstance(spectra, ShardedSpectra) and spectra.ndim == 3:
        if 'fehs' not in data:
            ignored_axes.append('feh')
        spectra = np.swapaxes(spectra, 0, 1)[:, :, None, :]
    expected = (len(temps), len(loggs), len(fehs), waves.size)
    if spectra.shape != expected:
        raise ValueError(f'Stellar spectra shape {spectra.shape}; expected {expected}')
    for name, axis in zip(('temperatures', 'loggs', 'fehs', 'wavelengths_m'),
                          (temps, loggs, fehs, waves)):
        if axis.ndim != 1 or len(axis) == 0 or np.any(~np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
            raise ValueError(f'Stellar {name} must be finite and strictly increasing')
    if len(waves) < 2 or temps[0] <= 0 or waves[0] <= 0:
        raise ValueError('Stellar temperatures and wavelengths must be positive; need two wavelengths')
    valid = np.asarray(data.get('valid', np.ones(expected[:-1], bool)), bool)
    if valid.shape != expected[:-1] or not valid.any():
        raise ValueError('Invalid stellar model mask')
    offset = np.asarray(data['log_offset'], np.float32) if encoded else None
    scale = np.asarray(data['log_scale'], np.float32) if encoded else None
    if encoded:
        if (offset.shape != valid.shape or scale.shape != valid.shape or
                np.any(~np.isfinite(offset[valid])) or np.any(~np.isfinite(scale[valid])) or
                np.any(scale[valid] < 0)):
            raise ValueError('Invalid stellar log encoding')
    else:
        # Inspect rows without constructing a second full cube.
        for index in zip(*np.where(valid)):
            if np.any(~np.isfinite(spectra[index])) or np.any(spectra[index] <= 0):
                raise ValueError('Stellar spectra must be finite and positive')
    metadata = data.get('metadata', {})
    if not isinstance(metadata, dict):
        metadata = json.loads(str(np.asarray(metadata).item()))
    if ignored_axes:
        metadata = dict(metadata, ignored_parameter_axes=ignored_axes)
    for arr in (temps, loggs, fehs, waves, spectra, valid, offset, scale):
        if isinstance(arr, np.ndarray):
            arr.setflags(write=False)
    return StellarGrid(temps, loggs, fehs, waves, spectra, valid, offset, scale, metadata)


def resolve_stellar_grid(stellar_grid='newera'):
    if stellar_grid is None or str(stellar_grid).lower() == 'newera':
        return Path(__file__).resolve().parent / 'stellar_data/newera_jwst.npz'
    if str(stellar_grid).lower() == 'phoenix':
        return Path(__file__).resolve().parent / 'data/stellar_spectra.pkl'
    path = Path(stellar_grid).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Stellar grid not found: {path}')
    return path


def load_stellar_grid(stellar_grid='newera'):
    if stellar_grid is None or str(stellar_grid).lower() == 'newera':
        from .stellar_grid import download_stellar_grid
        download_stellar_grid()
    path = resolve_stellar_grid(stellar_grid)
    stat = path.stat()
    return _load_grid_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4)
def _load_grid_cached(path, mtime_ns, size):
    if Path(path).suffix.lower() in {'.h5', '.hdf5'}:
        return load_native_stellar_grid(path)
    if Path(path).suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            values = dict(data)
        if 'shard_files' in values:
            filenames = values['shard_files']
            if any(Path(str(n)).name != str(n) for n in filenames):
                raise ValueError('Stellar shard names must be local filenames')
            shape = (len(values['temperatures']), len(values['loggs']),
                     len(values['fehs']), len(values['wavelengths_m']))
            if len(filenames) != shape[2]:
                raise ValueError('Expected one stellar shard per metallicity')
            values['encoded_spectra'] = ShardedSpectra(
                [Path(path).parent / str(n) for n in filenames], shape)
        return grid_from_dict(values)
    with open(path, 'rb') as stream:
        data = pickle.load(stream, encoding='latin1')
    waves = None
    if 'wavelengths_m' not in data:
        waves = np.load(Path(__file__).resolve().parent / 'data/low_res_lambdas.npy')
    return grid_from_dict(data, waves)


def stellar_components(grid, T_star, T_spot, spot_cov_frac, T_fac=None,
                       fac_cov_frac=None, logg_phot=4.5, logg_spot=None,
                       logg_fac=None, feh=0., blackbody=False, grid_only=False):
    """Validate stellar inputs on the host and select static JIT branches."""
    if blackbody and grid_only:
        raise ValueError('stellar_blackbody and stellar_grid_only cannot both be True')
    fractions = (0. if spot_cov_frac is None else spot_cov_frac,
                 0. if fac_cov_frac is None else fac_cov_frac)
    if (not np.all(np.isfinite(fractions)) or min(fractions) < 0 or sum(fractions) > 1):
        raise AtmosphereError('Stellar covering fractions must be finite, nonnegative, and sum to at most one')
    temps = (T_star, T_star if T_spot is None else T_spot,
             T_star if T_fac is None else T_fac)
    gravities = (logg_phot, logg_phot if logg_spot is None else logg_spot,
                 logg_phot if logg_fac is None else logg_fac)
    active = (T_star is not None, fractions[0] > 0, fractions[1] > 0)
    if T_star is None and any(active):
        raise AtmosphereError('T_star is required for stellar contamination')
    in_grid = []
    for component, temp, gravity, enabled in zip(
            ('T_star', 'T_spot', 'T_fac'),
            temps, gravities, active):
        if not enabled:
            in_grid.append(False)
            continue
        if not np.isfinite(temp) or temp <= 0:
            raise AtmosphereError('Stellar temperatures must be finite and positive')
        use_grid = not blackbody and grid.temperatures[0] <= temp <= grid.temperatures[-1]
        if not blackbody and not use_grid:
            minimum, maximum = grid.temperatures[[0, -1]]
            message = (f'{component} = {temp:g} K is outside the stellar grid '
                       f'({minimum:g}-{maximum:g} K)')
            if grid_only:
                raise AtmosphereError(message)
            # Trial temperatures vary during retrievals; warn only on the first
            # fallback for each component and grid temperature range.
            key = (component, minimum, maximum)
            if key not in _blackbody_warnings:
                warnings.warn(message + '; using a blackbody spectrum; '
                              'stellar_grid_only=True raises an error instead',
                              UserWarning, stacklevel=2)
                _blackbody_warnings.add(key)
        if use_grid:
            grid.validate(temp, gravity, feh)
        in_grid.append(bool(use_grid))
    return temps, gravities, fractions, tuple(in_grid)


class NativeSpectra:
    """Lazy reader for the native MSG HDF5 file; no pymsg/MSG dependency."""
    def __init__(self, path, model_indices, wavelength_edges, downsample):
        self.path = Path(path)
        self.model_indices = model_indices
        self.wavelength_edges = wavelength_edges
        self.downsample = downsample
        self.shape = model_indices.shape + (len(wavelength_edges) - 1,)
        self.dtype = np.dtype(np.float32)
        self._read_row = lru_cache(maxsize=16)(self._load_row)

    def __getitem__(self, index):
        return self._read_row(int(self.model_indices[index]))

    def _load_row(self, model):
        import h5py
        from ._stellar_rebin import conservative_rebin
        if model == 0:
            raise AtmosphereError('Missing native NewEra model')
        with h5py.File(self.path, 'r') as f:
            s = f[f'specsource/specints[{model}]']
            if s['limb'].attrs['law'] != b'CONST' or s['c'].shape[1] != 1:
                raise ValueError('Only CONST limb spectra are supported')
            r = s['range'].attrs
            if r['TYPE'] != b'lin_range_t':
                raise ValueError('Only linear native wavelength ranges are supported')
            source_edges = r['x_0'] + r['dx'] * np.arange(r['n'])
            # Integrate intensity over solid angle (pi) and convert cgs to SI (1e7).
            flux = conservative_rebin(s['c'][:, 0], source_edges, self.wavelength_edges) * (np.pi * 1e7)
        flux = flux.astype(np.float32)
        flux.setflags(write=False)
        return flux

    def __getstate__(self):
        return (self.path, self.model_indices, self.wavelength_edges, self.downsample)

    def __setstate__(self, state):
        self.__init__(*state)


def load_native_stellar_grid(path, downsample=1):
    """Read an MSG NewEra grid lazily; average blocks of native wavelength bins."""
    from numbers import Integral
    if not isinstance(downsample, Integral) or isinstance(downsample, bool) or downsample < 1:
        raise ValueError('downsample must be a positive integer')
    try:
        import h5py
    except ImportError as error:
        raise ImportError('Install platon[stellar-build] to read native HDF5 grids; pymsg is unnecessary') from error
    path = Path(path).expanduser().resolve()
    with h5py.File(path, 'r') as f:
        if int(f.attrs['REVISION']) != 1:
            raise ValueError('Unsupported MSG specgrid revision')
        vg = f['vgrid']
        labels = [vg[f'axes[{i}]'].attrs['label'].decode() for i in (1, 2, 3)]
        if labels != ['Teff', '[Fe/H]', 'log(g)']:
            raise ValueError(f'Unsupported MSG axes: {labels}')
        temps, fehs, loggs = [vg[f'axes[{i}]/x'][...] for i in (1, 2, 3)]
        model_indices = np.zeros((len(temps), len(loggs), len(fehs)), np.int32)
        for seq, linear in enumerate(vg['v_lin_seq'][...], start=1):
            t, z, g = np.unravel_index(int(linear) - 1, (len(temps), len(fehs), len(loggs)), order='F')
            model_indices[t, g, z] = seq
        r = f['specsource/specints[1]/range'].attrs
        source_edges = r['x_0'] + r['dx'] * np.arange(r['n'])
        edges = source_edges[::downsample]
        if edges[-1] != source_edges[-1]:
            edges = np.r_[edges, source_edges[-1]]
        if len(edges) < 3:
            raise ValueError('downsample must retain at least two wavelength bins')
        metadata = dict(model='PHOENIX NewEra JWST', source_label=f.attrs['label'].decode(),
                        source_file=path.name, native_downsample=int(downsample),
                        flux_units='W m-2 m-1', license='CC-BY-4.0',
                        missing_model_policy='interpolate_isolated_temperature')
    return StellarGrid(temps, loggs, fehs, .5 * (edges[:-1] + edges[1:]) * 1e-10,
                       NativeSpectra(path, model_indices, edges, downsample),
                       model_indices > 0, metadata=metadata)
