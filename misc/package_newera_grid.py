"""Rebuild the release asset: python misc/package_newera_grid.py OUTPUT.zip."""
import argparse
import hashlib
from pathlib import Path
import shutil
import zipfile


def package(output, source=None):
    source = Path(source) if source else Path(__file__).resolve().parents[1] / 'platon/stellar_data'
    names = ['README.md', 'newera_jwst.json', 'validation.json', 'newera_jwst.npz']
    names += [f'newera_jwst_feh_{i:02d}.npz' for i in range(10)]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(names):
            info = zipfile.ZipInfo(f'stellar_data/{name}', date_time=(2026, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            with (source / name).open('rb') as stream, archive.open(info, 'w') as member:
                shutil.copyfileobj(stream, member, length=1024**2)
    digest = hashlib.sha256()
    with output.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    print(f'SHA-256: {digest.hexdigest()}\nBytes: {output.stat().st_size}')
    return digest.hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source', type=Path, default=Path('platon/stellar_data'),
                        help='Directory containing the release grid files')
    args = parser.parse_args()
    package(args.output, args.source)
