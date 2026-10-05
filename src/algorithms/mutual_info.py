# -*- coding: utf-8 -*-
"""Mutual information: supervised, sklearn mutual_info_classif on the training
split. Single-band independent criterion (independent=True): mutual
information is computed per band against the labels and does not depend on the
selected set."""
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from .base import Criterion


class MutualInfo(Criterion):
    name = 'MI'
    supervised = True
    sample = 'train'
    independent = True

    def _fit(self, X, y=None):
        mi = mutual_info_classif(X, y, random_state=42)
        return np.argsort(-mi)
