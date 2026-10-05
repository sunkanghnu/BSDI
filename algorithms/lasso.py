# -*- coding: utf-8 -*-
"""Lasso (L1-regularised linear regression) band selection: supervised,
training split.

Class labels are encoded as 0/1 (for multi-class, absolute coefficients are
aggregated one-vs-rest), an L1-penalised linear regression is fitted (the
sparsity-based criterion of Tibshirani, 1996), and bands are ranked by the
absolute value of their coefficients.

Not independent (independent=False): L1 coefficients are redistributed among
highly correlated bands, so refitting on the remaining set changes the ranking
and S2 must be re-run with the same criterion.
"""
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from .base import Criterion


class LassoBS(Criterion):
    name = 'LaASSO'
    supervised = True
    sample = 'train'
    independent = False

    def __init__(self, X=None, y=None, alpha=0.01, max_iter=1000, **kwargs):
        super().__init__(X, y, **kwargs)
        self.alpha = alpha
        self.max_iter = max_iter

    def _fit(self, X, y=None):
        """Return the full band ranking (best to worst); select(k) takes the
        top k."""
        X_s = StandardScaler().fit_transform(X)
        classes = np.unique(y)
        n_bands = X.shape[1]
        if len(classes) == 2:
            model = Lasso(alpha=self.alpha, max_iter=self.max_iter,
                          random_state=0)
            model.fit(X_s, (y == classes[1]).astype(np.float64))
            scores = np.abs(model.coef_)
        else:
            # one-vs-rest: sum the absolute coefficients across classes
            coef_sum = np.zeros(n_bands)
            for c in classes:
                model = Lasso(alpha=self.alpha, max_iter=self.max_iter,
                              random_state=0)
                model.fit(X_s, (y == c).astype(np.float64))
                coef_sum += np.abs(model.coef_)
            scores = coef_sum
        self.scores_ = np.asarray(scores, dtype=np.float64)
        return np.argsort(-scores)
