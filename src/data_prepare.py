# -*- coding: utf-8 -*-
"""Convert raw .mat files into data/<dataset>/X.npy (all pixels, N x L, float32)
and y.npy (N, int64, 0 = background).

Usage:
    python3 src/data_prepare.py --dataset indian_pines
    python3 src/data_prepare.py --dataset paviau
    python3 src/data_prepare.py --dataset salinas

The raw .mat files must sit in the repo root (or the dataset dir). After the
first run, X.npy / y.npy are cached under data/<dataset>/ and later runs of
run_all.py reuse them.
"""
import sys
import numpy as np
import scipy.io as sio
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'

SOURCES = {
    'indian_pines': dict(mat=ROOT / 'Indian_pines_corrected.mat',
                         gt=ROOT / 'Indian_pines_gt.mat',
                         var='indian_pines_corrected', gtvar='indian_pines_gt'),
    'paviau': dict(mat=ROOT / 'PaviaU.mat', gt=ROOT / 'PaviaU_gt.mat',
                   var='paviaU', gtvar='paviaU_gt'),
    'salinas': dict(mat=ROOT / 'Salinas_corrected.mat', gt=ROOT / 'Salinas_gt.mat',
                    var=None, gtvar=None),
}


def _first_var(d, exclude=('__header__', '__version__', '__globals__')):
    return [k for k in d if k not in exclude][0]


def prepare(ds):
    if ds not in SOURCES:
        sys.exit(f'unknown dataset: {ds}')
    src = SOURCES[ds]
    if not src['mat'].exists() or not src['gt'].exists():
        sys.exit(f'missing raw .mat: {src["mat"]} or {src["gt"]} '
                 f'(put it in the repo root)')
    D, G = sio.loadmat(src['mat']), sio.loadmat(src['gt'])
    var = src['var'] or _first_var(D)
    gtvar = src['gtvar'] or _first_var(G)
    Xm, Gm = D[var], G[gtvar]
    if Xm.ndim != 3:
        sys.exit(f'{ds}: expected an H x W x L cube, got {Xm.shape}')
    H, W, L = Xm.shape
    X = Xm.reshape(-1, L).astype(np.float32)
    y = Gm.reshape(-1).astype(np.int64)
    out = DATA_DIR / ds
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'X.npy', X)
    np.save(out / 'y.npy', y)
    print(f'{ds}: X={X.shape} y={y.shape} labeled={(y > 0).sum()} '
          f'classes={len(np.unique(y)) - 1} -> {out}')


if __name__ == '__main__':
    arg = '--dataset='
    ds = next((a[len(arg):] for a in sys.argv[1:] if a.startswith(arg)),
              'salinas')
    prepare(ds)
