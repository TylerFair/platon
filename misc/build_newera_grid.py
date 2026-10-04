"""Stream an MSG NewEra-JWST grid into a compact, distributable NPZ.

No pymsg at build/run time: requires h5py only to build. The supported MSG
revision stores piecewise-constant specific intensity and a CONST limb law.
The conservative integral below matches SpecGrid.flux(order=1) at grid nodes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from platon._stellar_rebin import conservative_rebin


def logarithmic_edges(lo, hi, resolving_power):
    if not np.isfinite(resolving_power) or resolving_power <= 0:
        raise ValueError("resolving_power must be positive")
    return np.geomspace(lo, hi, int(np.ceil(np.log(hi / lo) * resolving_power)) + 1)


def encode_log_flux(flux):
    """Row-local uint16 log encoding; decoding preserves relative precision."""
    if np.any(~np.isfinite(flux)) or np.any(flux <= 0):
        raise ValueError("Stellar flux must be finite and positive")
    log_flux = np.log(flux)
    offset = log_flux.min()
    scale = (log_flux.max() - offset) / 65535
    if scale == 0:
        scale = 1.
    encoded = np.rint((log_flux - offset) / scale).astype(np.uint16)
    return encoded, np.float32(offset), np.float32(scale)


def build(source, output, resolving_power=5000, native_downsample=None):
    import h5py
    source = Path(source)
    with h5py.File(source, 'r') as f:
        if int(f.attrs['REVISION']) != 1:
            raise ValueError("Unsupported MSG specgrid revision")
        vg = f['vgrid']
        labels = [vg[f'axes[{i}]'].attrs['label'].decode() for i in (1, 2, 3)]
        if labels != ['Teff', '[Fe/H]', 'log(g)']:
            raise ValueError(f"Unsupported MSG axes: {labels}")
        temperatures, fehs, loggs = [vg[f'axes[{i}]/x'][...] for i in (1, 2, 3)]
        shape = (len(temperatures), len(loggs), len(fehs))
        ss = f['specsource']
        if native_downsample is None:
            edges = logarithmic_edges(ss.attrs['lam_min'], ss.attrs['lam_max'], resolving_power)
        else:
            if not isinstance(native_downsample, int) or isinstance(native_downsample, bool) or native_downsample < 1:
                raise ValueError('native_downsample must be a positive integer')
            r = ss['specints[1]/range'].attrs
            native_edges = r['x_0'] + r['dx'] * np.arange(r['n'])
            edges = native_edges[::native_downsample]
            if edges[-1] != native_edges[-1]:
                edges = np.r_[edges, native_edges[-1]]
            if len(edges) < 3:
                raise ValueError('native_downsample must retain at least two wavelength bins')
        encoded = np.zeros(shape + (len(edges) - 1,), np.uint16)
        offsets = np.zeros(shape, np.float32)
        scales = np.zeros(shape, np.float32)
        valid = np.zeros(shape, bool)
        max_quantization = 0.
        for seq, linear in enumerate(vg['v_lin_seq'][...], start=1):
            t, z, g = np.unravel_index(int(linear) - 1, (len(temperatures), len(fehs), len(loggs)), order='F')
            s = ss[f'specints[{seq}]']
            if s['limb'].attrs['law'] != b'CONST' or s['c'].shape[1] != 1:
                raise ValueError("Only CONST limb spectra are supported")
            r = s['range'].attrs
            if r['TYPE'] != b'lin_range_t':
                raise ValueError("Only linear source wavelength ranges are supported")
            native_edges = r['x_0'] + r['dx'] * np.arange(r['n'])
            flux = conservative_rebin(s['c'][:, 0], native_edges, edges) * (np.pi * 1e7)
            q, off, step = encode_log_flux(flux)
            encoded[t, g, z] = q
            offsets[t, g, z], scales[t, g, z] = off, step
            valid[t, g, z] = True
            restored = np.exp(off + q.astype(np.float64) * step)
            max_quantization = max(max_quantization, float(np.max(np.abs(restored / flux - 1))))
            if seq % 1000 == 0:
                print(f'{seq}/{len(vg["v_lin_seq"])} models', flush=True)
        source_label = f.attrs['label'].decode()
    # Hash by streaming: never read the 3.5 GB source into memory.
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    metadata = dict(format_version=1, model='PHOENIX NewEra V3 JWST',
                    source_label=source_label, source_sha256=digest.hexdigest(),
                    source_url='https://doi.org/10.25592/uhhfdm.17935',
                    license='CC-BY-4.0', flux_units='W m-2 m-1',
                    resolving_power=resolving_power if native_downsample is None else None,
                    native_downsample=native_downsample,
                    quantization_max_relative_error=max_quantization,
                    missing_models=int(valid.size - valid.sum()))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shards = []
    for i in range(len(fehs)):
        name = f'{output.stem}_feh_{i:02d}.npz'
        shards.append(name)
        np.savez_compressed(output.parent / name, encoded_spectra=encoded[:, :, i, :])
    np.savez_compressed(output, temperatures=temperatures, loggs=loggs, fehs=fehs,
                        wavelengths_m=np.sqrt(edges[:-1] * edges[1:]) * 1e-10,
                        wavelength_edges_m=edges * 1e-10,
                        shard_files=np.asarray(shards), log_offset=offsets,
                        log_scale=scales, valid=valid,
                        metadata=json.dumps(metadata, sort_keys=True))
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    total_bytes = output.stat().st_size + sum((output.parent / name).stat().st_size for name in shards)
    print(f'Wrote {output} + {len(shards)} shards: {total_bytes / 1024**2:.1f} MiB', flush=True)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    resolution = parser.add_mutually_exclusive_group()
    resolution.add_argument('--resolving-power', type=float, default=5000)
    resolution.add_argument('--native-downsample', type=int)
    args = parser.parse_args()
    build(args.source, args.output, args.resolving_power, args.native_downsample)
