# -*- coding: utf-8 -*-
"""kNN classifier plus a band-subset -> accuracy cache table.

Conventions:
- The classifier is fixed to kNN (parameters from config.yaml).
- Accuracy of any band subset is looked up in the table first; on a miss it is
  computed and written back.
- The table is keyed by frozenset of band indices and persisted per dataset
  (cache/accuracy/<dataset>_acc.pkl), shared across methods, k values, and
  the random baseline.
- Parallel safety: with joblib each task holds a copy of the table. New entries
  live in _local_new and are returned to the parent process at the end, so
  concurrent tasks never write to the same file.

kNN implementation (performance-critical path):
sklearn's KDTree/BallTree/brute distance code holds the GIL in Cython, so it
cannot truly parallelise under the threading backend. Here Euclidean
distances are computed with numpy matrix products (BLAS, GIL released) in
chunks, followed by a top-k majority vote. Results match
sklearn's KNeighborsClassifier(euclidean, k=5) (verified by smoke tests), so
tasks run in parallel under the threading backend.

Return contract: score() returns (acc, y_pred). On a cache hit y_pred is None,
so the paired McNemar standard error is unavailable and the caller falls back
to the independent approximation.
"""
import numpy as np
from .paths import ACC_DIR
from . import cache as cache_utils

_NCLASS_CAP = 64          # class-count cap (IP=16 / BW=9 / PV=9 is enough)
_CHUNK = 1024             # test-sample chunk size. Peak memory is
                          # chunk x n_tr x 8B x 2 (D + int64 indices).
                          # With n_tr~10826 (Salinas), chunk=8192 peaks at
                          # ~1.4GB and parallel tasks trigger OOM (SIGKILL,
                          # no traceback). chunk=1024 peaks at ~88MB, which is
                          # safe at any parallelism level.


def _knn_predict(A, B, ytr, k, n_neighbors):
    """Chunked Euclidean distance + top-k majority vote; returns predictions.

    A: n_tr x k standardised training projections; B: n_te x k test
    projections; ytr: training labels; returns n_te predicted labels.
    """
    n_tr = A.shape[0]
    n_te = B.shape[0]
    nA = (A ** 2).sum(axis=1)
    nB = (B ** 2).sum(axis=1)
    yte_pred = np.empty(n_te, dtype=np.int64)
    ncls = max(int(ytr.max()) + 1, 2)
    for s in range(0, n_te, _CHUNK):
        e = min(s + _CHUNK, n_te)
        D = nB[s:e, None] - 2.0 * (B[s:e] @ A.T) + nA[None, :]
        idx = np.argpartition(D, n_neighbors, axis=1)[:, :n_neighbors]
        labels = ytr[idx]                                   # batch x k
        batch = e - s
        cnt = np.zeros((batch, min(ncls, _NCLASS_CAP)), dtype=np.int32)
        rows_ids = np.repeat(np.arange(batch), n_neighbors)
        np.add.at(cnt, (rows_ids, labels.ravel()), 1)
        yte_pred[s:e] = cnt.argmax(axis=1)
    return yte_pred


class AccuracyTable:
    def __init__(self, ds_name, force=False, logger=None):
        self.path = ACC_DIR / f'{ds_name}_acc.pkl'
        self.logger = logger
        if force:
            self.table = {}
        else:
            self.table = cache_utils.load_pkl(self.path, logger) or {}
        self._local_new = {}

    def score(self, bands, Xtr, Xte, ytr, yte, clf_params):
        """Look up / compute the accuracy of a band subset.

        Returns (accuracy, y_pred or None).
        """
        key = frozenset(int(i) for i in bands)
        if key in self.table:
            if self.logger is not None:
                self.logger.debug('[ACC-HIT] %s', _short(key))
            return float(self.table[key]), None
        if key in self._local_new:
            acc, _ = self._local_new[key]
            return acc, None
        n_neighbors = int(clf_params.get('n_neighbors', 5))
        bands_arr = np.asarray(list(key), dtype=np.int64)
        # standardise with training column stats; zero-variance columns -> 1
        mu = Xtr[:, bands_arr].mean(axis=0)
        sd = Xtr[:, bands_arr].std(axis=0)
        sd[sd == 0] = 1.0
        A = (Xtr[:, bands_arr] - mu) / sd
        B = (Xte[:, bands_arr] - mu) / sd
        y_pred = _knn_predict(A, B, ytr, len(bands_arr), n_neighbors)
        acc = float(np.mean(y_pred == yte))
        self._local_new[key] = (acc, y_pred)
        if self.logger is not None:
            self.logger.debug('[ACC-COMPUTE] %s -> %.4f', _short(key), acc)
        return acc, y_pred

    def merge_from(self, other):
        """Merge new entries from another table instance (parallel task)."""
        for k, (acc, _) in other._local_new.items():
            self.table[k] = acc

    def save(self, logger=None):
        cache_utils.save_pkl(self.path, self.table, logger or self.logger)

    def __len__(self):
        return len(self.table)


def _short(key, n=10):
    s = sorted(key)
    if len(s) <= n:
        return f'bands={s}'
    return f'bands={s[:n]}...({len(s)} bands)'
