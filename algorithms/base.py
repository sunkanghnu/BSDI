# -*- coding: utf-8 -*-
"""Base class for band-selection criteria.

Conventions:
- One module per criterion, subclassing Criterion and implementing _fit(),
  which sets self.order_ (the full ranking, best to worst, length = L).
- independent marks whether the criterion scores each band on its own:
    * independent=True: single-band criteria (variance, entropy, mutual
      information, univariate F). Scores do not depend on the selected set, so
      the ranking of the remaining set equals the full ranking without its
      first k entries and S2 can be sliced from the cached ranking.
    * independent=False: multi-band / search criteria (LDA-SFS, mRMR, MVPCA,
      MaxDet, DensityPeak). Scores depend on the band set, so S2 must be
      refitted on the remaining set with the same criterion.
- Ranking caches (cache/rankings/<dataset>_<name>_<sample>.npy) are managed by
  the pipeline; Criterion itself never touches the disk.
- Shared statistics (covariance Sigma / correlation corr) can be injected by
  the caller through constructor arguments (MaxDet, DensityPeak) so they are
  not recomputed.
"""
import numpy as np


class Criterion:
    name = 'base'
    supervised = False          # whether the criterion needs labels
    sample = 'all'              # 'all' (all pixels) | 'train' (training split)
    independent = False         # single-band independent scoring?

    def __init__(self, X=None, y=None, **kwargs):
        self.X_ = np.asarray(X, dtype=np.float64) if X is not None else None
        self.y_ = y
        self.order_ = None

    def fit(self, X=None, y=None):
        X = self.X_ if X is None else np.asarray(X, dtype=np.float64)
        y = self.y_ if y is None else y
        self.order_ = np.asarray(self._fit(X, y))
        return self.order_

    def _fit(self, X, y=None):
        raise NotImplementedError

    def select(self, k):
        """Return the top-k bands of the full ranking (original indices)."""
        return self.order_[:k]

    def __repr__(self):
        return f'{self.name}[{self.sample},indep={self.independent}]'
