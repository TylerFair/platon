"""Public, MSG-free generation and download of stellar-grid caches."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from threading import RLock

import numpy as np

from ._stellar_grid import (load_stellar_grid as _load_stellar_grid,
                            load_native_stellar_grid as _load_native_stellar_grid)
from .errors import AtmosphereError


__all__ = ['generate_stellar_grid', 'download_stellar_grid']

_download_lock = RLock()


def generate_stellar_grid(output_path, *, logg=4.5, feh=0., temperatures=None,
                          loggs=None, fehs=None, source='newera', downsample=1,
                          missing_model_policy=None):
    """Save a standalone stellar grid with float32 flux in W m^-2 m^-1.

    Parameters
    ----------
    output_path : str or pathlib.Path
        Destination NPZ file.
    logg : float, optional
        Fixed log10 surface gravity in cgs (default 4.5); ignored if loggs is set.
    feh : float, optional
        Fixed [Fe/H] in dex (default 0); ignored if fehs is set.
    temperatures : array_like, optional
        Increasing temperatures in K; defaults to the source temperature axis.
        Missing models on that default axis remain masked.
    loggs : array_like, optional
        Increasing log10 surface gravities in cgs, retaining this grid axis.
    fehs : array_like, optional
        Increasing [Fe/H] values in dex, retaining this grid axis.
    source : str or pathlib.Path, optional
        'newera' (default, downloaded on first use), 'phoenix', or a local
        NPZ, pickle, or native NewEra HDF5 file. HDF5 requires h5py.
    downsample : int, optional
        Number of native HDF5 wavelength bins to average (default 1).
    missing_model_policy : str, optional
        'reject' or 'interpolate_isolated_temperature'; defaults to source
        metadata. Repair interpolates a missing row from its valid neighbors.

    Returns
    -------
    pathlib.Path
        Absolute path to the saved grid.
    """
    path = Path(output_path).expanduser()
    if path.suffix.lower() != '.npz':
        raise ValueError('output_path must end in .npz')
    native = Path(source).suffix.lower() in {'.h5', '.hdf5'}
    if not native and downsample != 1:
        raise ValueError('downsample applies only to native HDF5 sources')
    grid = _load_native_stellar_grid(source, downsample) if native else _load_stellar_grid(source)
    if missing_model_policy is not None:
        if missing_model_policy not in {'reject', 'interpolate_isolated_temperature'}:
            raise ValueError('Unknown missing_model_policy')
        grid = replace(grid, metadata=dict(grid.metadata or {},
                                           missing_model_policy=missing_model_policy))
    def axis(values, fallback, name):
        result = np.asarray(fallback if values is None else values, dtype=np.float64)
        if (result.ndim != 1 or len(result) == 0 or np.any(~np.isfinite(result)) or
                np.any(np.diff(result) <= 0)):
            raise ValueError(f'{name} must be a finite increasing array')
        return result
    temps = axis(temperatures, grid.temperatures, 'temperatures')
    gravity = axis(loggs, [logg], 'loggs')
    metallicity = axis(fehs, [feh], 'fehs')
    # Preserve missing models only on the default temperature axis.
    for values, source_axis, name in ((temps, grid.temperatures, 'Teff'),
                                     (gravity, grid.loggs, 'logg'),
                                     (metallicity, grid.fehs, 'feh')):
        ignored = (grid.metadata or {}).get('ignored_parameter_axes', ())
        if name not in ignored and (values[0] < source_axis[0] or values[-1] > source_axis[-1]):
            raise AtmosphereError(f'Stellar {name} is outside [{source_axis[0]}, {source_axis[-1]}]')
    valid = np.ones((len(temps), len(gravity), len(metallicity)), bool)
    strict_grid = replace(grid, metadata=dict(grid.metadata or {}, missing_model_policy='reject'))
    repaired_models = 0
    for i, t in enumerate(temps):
        for j, g in enumerate(gravity):
            for k, z in enumerate(metallicity):
                try:
                    grid.validate(t, g, z)
                except AtmosphereError:
                    if temperatures is not None:
                        raise
                    valid[i, j, k] = False
                if valid[i, j, k]:
                    try:
                        strict_grid.validate(t, g, z)
                    except AtmosphereError:
                        repaired_models += 1
    if not valid.any():
        raise AtmosphereError('No valid stellar models at the requested coordinates')
    spectra = np.zeros(valid.shape + (len(grid.wavelengths_m),), np.float32)
    for i, t in enumerate(temps):
        for j, g in enumerate(gravity):
            for k, z in enumerate(metallicity):
                if valid[i, j, k]:
                    spectra[i, j, k] = grid.interpolate(t, g, z)
    metadata = dict(grid.metadata or {})
    # Generated axes specify gravity and metallicity even for legacy sources.
    metadata.pop('ignored_parameter_axes', None)
    metadata.setdefault('missing_model_policy', 'reject')
    metadata['generated_repaired_models'] = repaired_models
    metadata['generated_from'] = 'newera' if source is None else Path(source).name
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: interrupts cannot leave an apparently valid truncated cache.
    handle, tmp_name = tempfile.mkstemp(prefix=path.stem + '-', suffix='.npz', dir=path.parent)
    os.close(handle)
    try:
        np.savez_compressed(tmp_name, temperatures=temps, loggs=gravity, fehs=metallicity,
                            wavelengths_m=grid.wavelengths_m, spectra=spectra,
                            valid=valid,
                            metadata=json.dumps(metadata, sort_keys=True))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path.resolve()


def download_stellar_grid(force=False):
    """Download, verify, and install the NewEra release on first use.

    Parameters
    ----------
    force : bool, optional
        Replace an existing installation (default False). Otherwise reuse
        the grid when its manifest and ten metallicity shards are present.

    Returns
    -------
    pathlib.Path
        Installed manifest, in the stellar_data folder of PLATON's data
        directory (next to the opacity data).
    """
    from ._get_data import _download_and_install
    from . import __stellar_grid_url__, __stellar_grid_sha256__
    from ._stellar_grid import resolve_stellar_grid, _load_grid_cached

    with _download_lock:
        path = resolve_stellar_grid()
        names = ('newera_jwst.npz',) + tuple(f'newera_jwst_feh_{i:02d}.npz' for i in range(10))
        if not force and all((path.parent / name).is_file() for name in names):
            return path
        # Earlier versions kept the grid in platon/stellar_data
        legacy = path.parent.parent.parent / 'stellar_data'
        if not force and not path.parent.exists() and \
                all((legacy / name).is_file() for name in names):
            path.parent.parent.mkdir(parents=True, exist_ok=True)
            os.replace(legacy, path.parent)
            print(f'Moved the NewEra stellar spectra from {legacy} to {path.parent}')
            return path
        print(f'Downloading NewEra stellar spectra (one time only) from {__stellar_grid_url__}')
        try:
            _download_and_install(__stellar_grid_url__, path.parent.parent,
                                  'stellar_data', __stellar_grid_sha256__,
                                  required_files=names)
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                f'Could not install NewEra stellar spectra: {error}. '
                'Run platon.stellar_grid.download_stellar_grid() where internet is available, '
                'or pass stellar_grid= a local file.') from error
        _load_grid_cached.cache_clear()
        return path
