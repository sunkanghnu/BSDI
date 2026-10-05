# -*- coding: utf-8 -*-
"""Logistic-regression-based band selection: supervised, training split.

Fits an L1-penalised logistic regression (multi-class handled by a one-vs-rest
wrapper around liblinear) and ranks bands by the absolute coefficient value
(aggregated across classes by max), i.e. the sparse logistic band selection
scheme of Pant et al. (2014).

Not independent (independent=False): L1 coefficients depend on the set of
bands used in the fit, so refitting on the remaining set changes the ranking
and S2 must be re-run with the same criterion.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from .base import Criterion


class LogisticBS(Criterion):
    name = 'LR'
    supervised = True
    sample = 'train'
    independent = False

    def __init__(self, X=None, y=None, C=1.0, max_iter=1000, **kwargs):
        super().__init__(X, y, **kwargs)
        self.C = C
        self.max_iter = max_iter

    def _fit(self, X, y=None):
        """Return the full band ranking (best to worst); select(k) takes the
        top k."""
        X_s = StandardScaler().fit_transform(X)
        base = LogisticRegression(penalty='l1', solver='liblinear', C=self.C,
                                  max_iter=self.max_iter, random_state=0)
        model = OneVsRestClassifier(base)
        model.fit(X_s, y)
        coef = np.vstack([est.coef_ for est in model.estimators_])
        if coef.shape[0] == 1:
            # binary case: one-vs-rest yields a single estimator
            scores = np.abs(coef[0])
        else:
            # multi-class: take the max absolute coefficient over classes
            scores = np.abs(coef).max(axis=0)
        self.scores_ = np.asarray(scores, dtype=np.float64)
        return np.argsort(-scores)
