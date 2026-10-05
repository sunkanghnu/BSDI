# -*- coding: utf-8 -*-
"""Entropy (histogram entropy): unsupervised, 32-bin histogram entropy per
band over all pixels. Single-band independent criterion (independent=True)."""
import numpy as np
from scipy.stats import entropy as sp_entropy
from .base import Criterion


class Entropy(Criterion):
    name = 'ETP'
    supervised = False
    sample = 'all'
    independent = True

    def _fit(self, X, y=None):
        L = X.shape[1]
        ent = np.array([sp_entropy(np.histogram(X[:, j], bins=32)[0])
                        for j in range(L)])
        return np.argsort(-ent)
