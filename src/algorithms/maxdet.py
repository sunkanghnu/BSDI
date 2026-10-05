# -*- coding: utf-8 -*-
"""MaxDet (maximum covariance determinant): unsupervised, multi-band.

Greedily picks the k bands maximising the determinant of the subset covariance
matrix (D-optimality). At each step the band with the largest Schur complement
sigma_j is added and the inverse is updated incrementally, giving roughly
O(L^2 k^2).

Shared statistics: Sigma is precomputed in src/stats.py and injected through
the constructor argument sigma, so it is not recomputed per dataset.
Not independent (independent=False): S2 is re-run on the remaining set with
that set's own covariance.
"""
import numpy as np
from .base import Criterion


class MaxDet(Criterion):
    name = 'MCD'
    supervised = False
    sample = 'all'
    independent = False

    def __init__(self, X=None, y=None, sigma=None):
        super().__init__(X, y)
        self._sigma = sigma

    def _fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        L = X.shape[1]
        if self._sigma is not None:
            Sigma = self._sigma
        else:
            Xc = X - X.mean(0)
            n = X.shape[0]
            Sigma = (Xc.T @ Xc) / n
        eps = 1e-10 * np.trace(Sigma) / L
        Sigma = Sigma + eps * np.eye(L)          # regularise to avoid singularity

        S = []                                   # selected indices
        order = []
        first = int(np.argmax(np.diag(Sigma)))
        S.append(first); order.append(first)
        invS = 1.0 / Sigma[first, first] * np.eye(1)
        cand = np.array([i for i in range(L) if i != first])

        while len(order) < L:
            U = Sigma[np.ix_(cand, S)]            # (c,k)
            W = U @ invS                          # (c,k)
            sig = np.diag(Sigma)[cand] - np.sum(W * U, axis=1)
            j_pos = int(np.argmax(sig))
            if sig[j_pos] <= 0:                    # numerical guard: skip degenerate
                cand = np.delete(cand, j_pos)
                continue
            j = int(cand[j_pos])
            u = Sigma[j, S]                        # (k,)
            w = invS @ u                           # (k,)
            sigma_j = Sigma[j, j] - u @ w
            top = invS + np.outer(w, w) / sigma_j
            side = -w / sigma_j
            invS = np.vstack([np.hstack([top, side[:, None]]),
                              np.hstack([side, [1.0 / sigma_j]])])
            S.append(j); order.append(j)
            cand = np.delete(cand, j_pos)
        return np.asarray(order)
