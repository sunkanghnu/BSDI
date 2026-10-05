# -*- coding: utf-8 -*-
"""mRMR (minimum redundancy maximum relevance): supervised, official
mrmr_selection library, training-split context.

Notes:
- mRMR selects greedily, so running L-1 steps once yields the full order
  except for the last band, which is appended at the end. Once the ranking is
  cached, any k is a prefix slice (this gives S1).
- Not independent (independent=False): the redundancy term depends on the
  already selected bands, so the remaining-set selection must be re-run.
- The official library requires a DataFrame input.

Refine (swap-based local search) is deliberately not used here. The incremental
mRMR criterion (F_j - mean|corr(candidate, selected)|) is not isomorphic to the
global objective J(S) = mean(F_S) - mean|corr|(S); using the global J as the
swap criterion pushes S1/S2 towards combinations with very high F but strong
mutual correlation, which lowered kNN accuracy by 8-12 percentage points on
Indian Pines and broke the D_Q test. mRMR therefore stays purely greedy, and
the diagnostic acts directly on the greedy path.
"""
import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from .base import Criterion


class MRMR(Criterion):
    name = 'mRMR'
    supervised = True
    sample = 'train'
    independent = False

    def _fit(self, X, y=None):
        L = X.shape[1]
        # n_jobs must stay 1. Internally mrmr-selection calls joblib Parallel
        # (loky processes by default) for f_classif and for the per-step
        # correlation, while this framework already runs methods in parallel
        # with the threading backend. Spawning child processes inside worker
        # threads stalls the mRMR progress bar on Windows (spawn overhead and
        # repeated re-imports), typically around 30%. n_jobs=1 makes joblib run
        # sequentially in a single worker, removing the nested parallelism.
        order = np.asarray(mrmr_classif(pd.DataFrame(X), y, L - 1, n_jobs=1))
        rem = np.setdiff1d(np.arange(L), order)
        return np.concatenate([order, rem])
