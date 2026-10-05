# -*- coding: utf-8 -*-
"""Density peak representative bands: unsupervised, inter-band relations.

Follows the density-peak idea of Rodriguez & Laio (2014, Science): band
distance d_ij = 1 - |corr_ij| defines similarity, then each band gets a local
density rho (Gaussian kernel) and a distance delta to the nearest higher-density
band. The representativeness score is gamma = rho * delta, and the top-k bands
are selected. No full clustering is performed, only peak identification.

Shared statistics: the correlation matrix is precomputed in src/stats.py and
injected through the constructor argument corr.
Not independent (independent=False): S2 is re-run on the remaining set with
that set's own correlation matrix.
"""
import numpy as np
from sklearn.metrics import pairwise_distances
from .base import Criterion


class DensityPeak(Criterion):
    name = 'DPC'
    supervised = False
    sample = 'all'
    independent = False

    def __init__(self, X=None, y=None, corr=None):
        super().__init__(X, y)
        self._corr = corr

    def _fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        # 1. pairwise distance matrix over bands
        distance_matrix = pairwise_distances(X.T, metric='euclidean')
        D = distance_matrix
        # 2. adaptive bandwidth
        sigma = np.sqrt(np.mean(D) / 30)
        # 3. local density rho
        rho = np.sum(np.exp(-D / (2 * sigma**2)), axis=1)
        # 4. distance delta to the nearest higher-density band
        delta = np.zeros_like(rho)
        max_density_idx = np.argmax(rho)

        for i in range(len(rho)):
            if i == max_density_idx:
                delta[i] = np.max(D[i])
            else:
                higher_density_idxs = np.where(rho > rho[i])[0]
                delta[i] = np.min(D[i, higher_density_idxs]) if len(higher_density_idxs) > 0 else 0

        # 5. representativeness gamma
        gamma = rho * delta

        return np.argsort(-gamma)
