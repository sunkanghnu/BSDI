# -*- coding: utf-8 -*-
"""Shared dataset-level statistics, computed once and cached.

Several criteria need the same statistics, so they are computed here once,
cached to disk, and reused by later methods:

- Sigma: band covariance matrix (L x L) over all pixels. Used by MaxDet
  (maximum covariance determinant); its regularisation and Schur-complement
  greedy search only need Sigma.
- corr: correlation matrix (L x L) derived from Sigma. Used by DensityPeak,
  whose distance d_ij = 1 - |corr_ij| comes straight from it.

High-dimensional guard: when a statistic needs a pixel-level (N x N) matrix,
stats.max_pixels enables random pixel subsampling before computing it. The
current method set only uses band-level matrices (L x L, L <= 200), so this
is a safeguard for future pixel-level statistics.
"""
import numpy as np
from .paths import STATS_DIR
from . import cache as cache_utils


def _sigma_corr(X, max_pixels=None, rng_seed=42):
    """Band covariance and correlation matrices, with optional subsampling."""
    n = X.shape[0]
    if max_pixels is not None and n > max_pixels:
        rng = np.random.RandomState(rng_seed)
        keep = rng.choice(n, max_pixels, replace=False)
        X = X[keep]
    Xc = X - X.mean(0)
    n = X.shape[0]
    Sigma = (Xc.T @ Xc) / n
    sd = np.sqrt(np.clip(np.diag(Sigma), 1e-12, None))
    corr = Sigma / np.outer(sd, sd)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return Sigma, corr


def get_dataset_stats(ds_name, X_all, max_pixels=None, force=False, logger=None):
    """Return the shared statistics npz (loaded from cache when present):
       ['Sigma']  L x L covariance over all pixels
       ['corr']   L x L correlation matrix
    """
    path = STATS_DIR / f'{ds_name}_stats.npz'
    if not force:
        z = cache_utils.load_npz(path, logger)
        if z is not None:
            return z
    Sigma, corr = _sigma_corr(X_all, max_pixels=max_pixels)
    cache_utils.save_npz(path, {'Sigma': Sigma, 'corr': corr}, logger)
    return np.load(path)
