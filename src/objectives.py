# -*- coding: utf-8 -*-
"""Criterion value J(S) for the re-selection diagnostic, computed in one
shared full-set context.

Used by run_criterion_values.py. Key idea: S1 and S2 must be scored with the
same statistics, otherwise D_J = J(S1) - J(S2) is not interpretable.

Per-method conventions:
- mRMR (path function): sum of the greedy-path scores of the first k bands.
  The score depends on the greedy path (denominator = redundancy with the
  previously selected bands), so only path prefixes can be scored: S1 uses the
  full-set path, S2 the path re-run on the remaining bands (both mapped back to
  full-set indices). F and |corr| do not depend on the candidate pool, so both
  paths are evaluated under the same precomputed statistics.
- Set functions (LDA / MCD): J depends only on the subset itself, so the
  full-set and remaining-set conventions coincide automatically.
- Weight-based (MVPCA / DPC / LASSO / LR / HOGSP / CAE): J_full sums
  full-set weights over S, giving one comparable scale.
- Single-band independent criteria (MI / ETP / NHMC): no set dependence.

Interpretation:
- Pure ranking methods (S1 = top-k by full-set score, i.e. MVPCA / DPC / LASSO
  / LR / HOGSP / CAE / MI / ETP): J_full(S1) >= J_full(S2) holds by
  construction, so the sign carries no information; the signal is in the
  relative magnitude D_J / J1 (near zero for flat probability fields such as
  CAE).
- Search / greedy methods (LDA-SFS, MCD, mRMR): no such guarantee, so the sign
  itself is diagnostic (a negative value means the first pass was not the
  criterion optimum, i.e. a genuine local optimum).
"""
import numpy as np


def dpc_gamma(X):
    """Density-peak gamma = rho * delta (Euclidean version)."""
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X.T, metric='euclidean')
    sigma = np.sqrt(np.mean(D) / 30)
    rho = np.sum(np.exp(-D / (2 * sigma ** 2)), axis=1)
    delta = np.zeros_like(rho)
    max_idx = int(np.argmax(rho))
    for i in range(len(rho)):
        if i == max_idx:
            delta[i] = np.max(D[i])
        else:
            hi = np.where(rho > rho[i])[0]
            delta[i] = np.min(D[i, hi]) if len(hi) > 0 else 0.0
    return rho * delta


def entropy_hist(X, bins=32):
    from scipy.stats import entropy as sp_entropy
    return np.array([sp_entropy(np.histogram(X[:, j], bins=bins)[0])
                     for j in range(X.shape[1])])


def mrmr_objective(S, F, corr, floor=0.001):
    """Set version of the MIQ criterion:
       J(S) = sum_{f in S} F(f) / mean over s in S, s != f of |corr(f, s)|.

    Scoring the final subset S directly (rather than a greedy path) keeps the
    quotient well behaved: the denominator averages the redundancy over the
    other k-1 members of S, so a single near-orthogonal band cannot blow the
    score up. With the path version, one band with denom=0.0023 reached a score
    of 7.9e5 and contributed 82% of J2 on PV at k=30; with the set version the
    denominator is the mean over 29 bands and D_J becomes stable again.

    Note the separation of concerns: band selection still uses the official
    greedy (path-based) library, while the quality of a final subset is scored
    with this set function, which uses the same F and |corr| measures and the
    same quotient form.
    """
    S = np.asarray(S, dtype=int)
    if len(S) == 0:
        return 0.0
    tot = 0.0
    for f in S:
        others = S[S != f]
        if len(others) == 0:
            denom = 1.0
        else:
            c = np.abs(corr[f, others])
            denom = float(np.nan_to_num(c, nan=floor).clip(floor).mean())
        tot += float(F[f]) / denom
    return tot


def lda_objective(S, Sw, Sb):
    """J(S) = trace(S_w[S,S]^{-1} S_b[S,S]), the LDA-SFS search criterion.

    Sw / Sb are full-set scatter matrices (training-set context, S_w
    ridge-regularised with lambda = 1e-4 * tr(S_w) / L). Used by the FisherLDA
    SFS search and by refine().
    """
    S = np.asarray(S, dtype=int)
    SwS = Sw[np.ix_(S, S)]
    SbS = Sb[np.ix_(S, S)]
    return float(np.trace(np.linalg.solve(SwS, SbS)))


def hill_climb_refine(S, unsel, objective, max_rounds=10, rel_tol=1e-9):
    """Generic swap-based local search (best-improvement hill climbing).

    Starting from S, each round scans all (selected i, unselected j) swaps and
    applies the one with the largest strictly positive gain (rel_tol suppresses
    floating-point noise). Stops when a full scan yields no positive gain.
    Returns the improved band set as an int array. Monotone non-decreasing, but
    not guaranteed globally optimal.
    """
    S = [int(x) for x in S]
    unsel = [int(x) for x in unsel]
    J = objective(np.asarray(S))
    for _ in range(max_rounds):
        best_i, best_j, best_dJ = -1, -1, 0.0
        thr = rel_tol * max(1.0, abs(J))
        for i in range(len(S)):
            for j in unsel:
                cand = S.copy()
                cand[i] = j
                Jc = objective(np.asarray(cand))
                dJ = Jc - J
                if dJ > best_dJ + thr:
                    best_i, best_j, best_dJ = i, j, dJ
        if best_i < 0:
            break
        cur = S[best_i]
        S[best_i] = best_j
        unsel.remove(best_j)
        unsel.append(cur)
        J += best_dJ
    return np.asarray(S, dtype=int)


def j_full(method, S, st, path=None):
    """Full-set J(S): S1 and S2 are scored under the same context.

    Methodological point: the criterion values of two selections are only
    comparable under one context. S1 comes from full-set statistics, while S2
    comes from re-running the criterion on the remaining bands; scoring S2
    with remaining-set statistics would mix rules and scales, so J2 always
    uses the full-set statistics.

    mRMR: J is the sum of greedy-path scores of the first k bands, is not a set
    function, and therefore needs the path (S1: full-set greedy path, S2:
    path re-run on the remaining bands, both mapped back to full-set indices).
    For the remaining methods J depends only on S itself (scatter blocks,
    covariance submatrix, single-column F and within-subset correlation, sums
    of full-set weights), so any S can be scored.
    """
    S = np.asarray(S, dtype=int)
    if method == 'mRMR':
        # Path-based criterion score; the path must be passed by the caller.
        if path is None:
            raise ValueError('mRMR J needs the greedy path (S1: full-set path, '
                             'S2: remaining-set path); the caller must pass it')
        return mrmr_objective(S, st['mRMR_F'], st['mRMR_corr'])
    if method == 'MVPCA':
        return float(st['MVPCA_load'][S].sum())
    if method == 'MCD':
        Sig = st['MCD_Sigma'][np.ix_(S, S)] + st['MCD_eps'] * np.eye(len(S))
        sign, ld = np.linalg.slogdet(Sig)
        return float(ld) if sign > 0 else float('-inf')
    if method == 'DPC':
        return float(st['DPC_gamma'][S].sum())
    if method == 'LDA':
        SwS = st['LDA_Sw'][np.ix_(S, S)]
        SbS = st['LDA_Sb'][np.ix_(S, S)]
        return float(np.trace(np.linalg.solve(SwS, SbS)))
    if method == 'LASSO':
        return float(st['LASSO_w'][S].sum())
    if method == 'LR':
        return float(st['LR_w'][S].sum())
    if method == 'HOGSP':
        return float(st['HOGSP_w'][S].sum())
    if method == 'CAE':
        return float(st['CAE_w'][S].sum())
    if method == 'MI':
        return float(st['MI_w'][S].sum())
    if method == 'ETP':
        return float(st['ETP_w'][S].sum())
    if method == 'NHMC':
        # J_n is a mean inter-class correlation (lower is better), so
        # D_J <= 0 holds by construction; the signal is in |D_J / J1|.
        return float(st['NHMC_w'][S].sum())
    raise ValueError(method)
