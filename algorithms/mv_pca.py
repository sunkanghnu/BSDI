# -*- coding: utf-8 -*-
"""MVPCA (principal component loadings): unsupervised, full-component PCA on
the all-pixel matrix; the score of a band is the sum of its absolute loadings.

Not independent (independent=False): PCA loadings are determined by the
covariance structure of the full set, so scores change on the remaining set
and PCA must be re-run there to obtain S2.
"""
import numpy as np
from sklearn.decomposition import PCA
from .base import Criterion


class MVPCA(Criterion):
    name = 'MVPCA'
    supervised = False
    sample = 'all'
    independent = False

    def _fit(self, X, y=None):
        L = X.shape[1]
        pca = PCA(n_components=L).fit(X)
        ld = np.abs(pca.components_).sum(0)
        return np.argsort(-ld)
