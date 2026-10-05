# -*- coding: utf-8 -*-
"""Fisher-LDA (multivariate Fisher discriminant): supervised, greedy forward
selection driven by the classical multivariate criterion

    J(S) = trace(S_w^{-1} S_b)

Sequential forward selection: at each step add the unselected band j that
maximises J(S + {j}), which yields the full selection order. Not independent
(independent=False): once bands are selected the scatter matrices change, so
the remaining-set re-selection differs from a prefix of the full ranking and
must be re-run.

Implementation notes:
- The full within-class (S_w) and between-class (S_b) scatter matrices are
  precomputed once on the training split.
- Adding a band extends S_w by one row and column, so S_w^{-1} is updated with
  the block-inverse formula instead of an O(k^3) recomputation per step.
- S_w is ridge-regularised with lambda = 1e-4 * tr(S_w) / L to stay well
  conditioned.
- The first step degenerates to the univariate Fisher ratio J = S_b / S_w.

Swap-based refine(): a local search on top of the greedy result, reducing the
irreversibility of forward selection. For a subset S from select(k), all
(selected i, unselected j) swaps are scanned each round and any strictly
improving swap is applied (best improvement) until a full scan yields no gain.
The objective is isomorphic to the search criterion, J = trace(S_w^{-1} S_b),
with S_w / S_b taken from the statistics of the set this instance was built on
(full set or remaining set), so swap decisions match the context that produced
S. The result is monotone non-decreasing.
"""
import numpy as np
from .base import Criterion
from ..objectives import lda_objective, hill_climb_refine


class FisherLDA(Criterion):
    name = 'LDA'
    supervised = True
    sample = 'train'
    independent = False

    def _compute_scatters(self, X, y):
        """Full scatter matrices on the training split, ridge-regularised.
        Shared by _fit and _ensure_stats so both use identical statistics."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        L = X.shape[1]
        mu = X.mean(0)
        Sw_all = np.zeros((L, L))
        Sb_all = np.zeros((L, L))
        for c in np.unique(y):
            Xc = X[y == c]
            muc = Xc.mean(0)
            nc = Xc.shape[0]
            Xcc = Xc - muc
            Sw_all += Xcc.T @ Xcc                       # within-class scatter
            d = (muc - mu)[:, None]
            Sb_all += nc * (d @ d.T)                    # between-class scatter
        lam = 1e-4 * np.trace(Sw_all) / L
        return Sw_all + lam * np.eye(L), Sb_all         # ridge regularised

    def _ensure_stats(self):
        """Lazily build the statistics. When the rankings cache hits, fit() is
        never called, but refine() still needs the scatter matrices, so they
        are computed on demand with exactly the same convention as fit()."""
        if not hasattr(self, 'Sw_all_'):
            self.Sw_all_, self.Sb_all_ = self._compute_scatters(self.X_, self.y_)

    def _fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        L = X.shape[1]
        # full-set scatter matrices, reused by refine and by criterion values
        self.Sw_all_, self.Sb_all_ = self._compute_scatters(X, y)
        Sw_all, Sb_all = self.Sw_all_, self.Sb_all_

        # ---- sequential forward selection ----
        sel = []                                        # selected bands
        Ainv = np.zeros((0, 0))                         # S_w^{-1} on the selected block
        Sb_sel = np.zeros((0, 0))                       # S_b on the selected block
        Jcur = 0.0
        order = []
        unsel = np.arange(L)

        for _ in range(L - 1):
            Q = Sw_all[np.ix_(unsel, sel)]              # m x k
            s = Sw_all[unsel, unsel]                    # m
            R = Sb_all[np.ix_(unsel, sel)]              # m x k
            t = Sb_all[unsel, unsel]                    # m

            if len(sel) == 0:
                Jn = t / s
                best_idx = int(np.argmax(Jn))
                bestJ = Jn[best_idx]
                j = int(unsel[best_idx])
            else:
                W = Q @ Ainv
                d = s - np.sum(W * Q, axis=1)
                Jn = np.full(len(unsel), -np.inf)
                valid = d > 1e-12
                if np.any(valid):
                    Wv = W[valid]; Qv = Q[valid]; Rv = R[valid]
                    dv = d[valid]; tv = t[valid]
                    term = np.sum((Wv @ Sb_sel) * Wv, axis=1)
                    cross = 2.0 * np.sum(Wv * Rv, axis=1)
                    Jn[valid] = Jcur + term / dv - cross / dv + tv / dv
                best_idx = int(np.argmax(Jn))
                bestJ = Jn[best_idx]
                j = int(unsel[best_idx])

            # ---- block-inverse incremental update ----
            if len(sel) == 0:
                sjj = Sw_all[j, j]
                tjj = Sb_all[j, j]
                Ainv = np.array([[1.0 / sjj]])
                Sb_sel = np.array([[tjj]])
            else:
                q = Sw_all[j, sel]
                r = Sb_all[j, sel]
                sjj = Sw_all[j, j]
                tjj = Sb_all[j, j]
                Aq = Ainv @ q
                d = sjj - q @ Aq
                top = Ainv + np.outer(Aq, Aq) / d
                right = -Aq / d
                Ainv = np.block([[top, right[:, None]],
                                 [right[None, :], np.array([[1.0 / d]])]])
                Sb_sel = np.block([[Sb_sel, r[:, None]],
                                   [r[None, :], np.array([[tjj]])]])
            Jcur = bestJ
            sel.append(j)
            order.append(j)
            unsel = np.setdiff1d(unsel, [j])

        order.append(int(unsel[0]))                     # last band goes last
        return np.array(order)

    # ------------------------------------------------------------------
    # refine: swap-based local search on the forward-greedy result
    # ------------------------------------------------------------------
    def _objective(self, S):
        return lda_objective(S, self.Sw_all_, self.Sb_all_)

    def refine(self, S, max_rounds=10):
        self._ensure_stats()
        S = np.asarray(S, dtype=int)
        L = self.Sw_all_.shape[0]
        unsel = [j for j in range(L) if j not in set(S.tolist())]
        return hill_climb_refine(S, unsel, self._objective,
                                 max_rounds=max_rounds)
