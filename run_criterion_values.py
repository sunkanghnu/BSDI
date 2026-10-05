#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Criterion consistency check: compute J(S1) and J(S2) for the cached
selections.

Background: a criterion only guarantees that the algorithm is optimal within
its own search space, i.e. J(S1) >= J(S2), where S1 is the top-k over the full
band set and S2 the top-k re-selected on the remaining set by the same
criterion. Whether the accuracy difference D_Q holds depends on accuracy Q
being a monotone proxy of the criterion J. This script checks the criterion-level
re-selection differential D_J = J(S1) - J(S2) separately, so that the two
claims (internal consistency of the criterion, and Q as a proxy for it) are
diagnosed independently.

Convention for J(S):
- mRMR: sum of the greedy-path scores of the first k bands, mirroring the
  per-step relevance/denominator score of the mrmr_selection library. The
  score depends on the greedy path (denominator = redundancy with the
  previously selected bands), so it is a path function, not a set function,
  and only path prefixes can be scored: S1 uses the full-set greedy path, S2
  the path re-run on the remaining set (mapped back to full-set indices). F
  and |corr| do not depend on the candidate pool, so both are evaluated under
  the same precomputed full-set statistics.
- Set functions (LDA / MCD): J depends only on the bands of S (scatter blocks,
  covariance submatrix), so the full-set and remaining-set conventions
  coincide and J_sel = J_full.
- Weight-based (MVPCA / DPC / LASSO / LR / HOGSP / CAE): the scores depend on
  the set of bands used to compute them (PCA loadings, gamma, regression
  coefficients, embedding scores, gate probabilities).
  * J_full(S) sums full-set weights over S (one comparable evaluation scale);
  * J_sel(S2) uses the statistics recomputed on the remaining set, exactly as
    in the algorithm's second selection, i.e. the true criterion value of that
    second selection.
- Single-band independent criteria (MI / ETP): no set dependence, so
  J_sel = J_full.

No selection is re-run: S1 / S2 are read from cache/selections. Criterion
values are cached as {ds}_{m}_k{k}.json under cache/criterion_values/, and a
cache hit skips the computation (including the mRMR remaining-set rerun). To
refresh one method / dataset, delete the corresponding json and rerun.
Usage: python3 run_criterion_values.py
Outputs: results/csv/<tag>_criterion_values.csv and
figures/fig_<tag>_criterion_delta.png
"""
import sys
import time
import csv
import json
import logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.paths import (ensure_dirs, SEL_DIR, CSV_DIR, FIG_DIR, LOG_DIR,
                       RANK_DIR, CRIT_DIR)
from src import datasets as ds_mod
from src.stats import get_dataset_stats
from src import cache as cache_utils
from src.algorithms.lasso import LassoBS
from src.algorithms.logistic_regression import LogisticBS
from src.algorithms.hogsp import HOGSP
from src.algorithms.concrete_vae import ConcreteAE
from src.algorithms.mrmr import MRMR
from src.algorithms.nhmc import NHMC
from src.objectives import (j_full, mrmr_objective,
                            dpc_gamma, entropy_hist)

# Datasets with complete caches by default (indian_pines / salinas). They can
# also be given on the command line:
#   python3 run_criterion_values.py indian_pines salinas
DEFAULT_DS = ['indian_pines', 'salinas', 'botswana', 'paviau', 'KSC']
TAG_MAP = {'indian_pines': 'IP', 'salinas': 'SL',
           'botswana': 'BW', 'paviau': 'PV', 'KSC': 'KSC'}
KS = [5, 10, 15, 20, 25, 30]
# 6 supervised + 6 unsupervised (same as METHODS in src/algorithms/__init__.py)
METHODS = ['mRMR', 'LDA', 'LASSO', 'LR', 'MI', 'NHMC',
           'MVPCA', 'MCD', 'DPC', 'HOGSP', 'CAE', 'ETP']
# Search / greedy methods: the sign of D_J is not guaranteed by construction
# (so it is diagnostic). The rest are pure ranking methods (D_J >= 0 by
# construction).
# Note: NHMC is also a pure ranking method, but its criterion J_n is an
# inter-class correlation (lower is better), so D_J = J(S1) - J(S2) <= 0 holds
# by construction, opposite to the other weight-based methods. The signal is
# in |D_J / J1| (strength of the criterion ranking), not in the sign.
SEARCH = {'mRMR', 'MCD', 'LDA'}

CLF = {'test_size': 0.8, 'random_state': 42}


def make_logger(tag):
    ensure_dirs()
    fpath = LOG_DIR / f'{tag}_criterion.log'
    logger = logging.getLogger(tag + '_crit')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S')
    fh = logging.FileHandler(fpath, encoding='utf-8')
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------- statistics


def build_full_stats(X_all, Xtr, ytr, stats_npz, logger, ds=''):
    """Full-set statistics for every method, computed once. Returns a dict.

    Heavy statistics (HOGSP / CAE / NHMC, all trained on the full data) are
    cached in cache/criterion_values/{ds}_weights.npz and loaded on a hit, so
    repeated runs do not recompute them.
    """
    st = {}
    L = X_all.shape[1]
    wpath = CRIT_DIR / f'{ds}_weights.npz'
    saved = dict(np.load(wpath)) if wpath.exists() else {}

    def cached(key, fn):
        if key in saved:
            logger.info('  weights: %-10s CACHE-HIT', key)
            return saved[key]
        logger.info('  weights: %-10s compute', key)
        v = fn()
        return v

    from sklearn.feature_selection import f_classif
    st['mRMR_F'] = cached('mRMR_F',
                          lambda: f_classif(Xtr, ytr)[0])
    st['mRMR_corr'] = cached('mRMR_corr',
                             lambda: np.corrcoef(np.asarray(Xtr, dtype=np.float64).T))

    from sklearn.decomposition import PCA
    st['MVPCA_load'] = cached('MVPCA_load', lambda: np.abs(
        PCA(n_components=L).fit(X_all).components_).sum(0))

    st['MCD_Sigma'] = np.asarray(stats_npz['Sigma'])   # already cached by stats
    st['MCD_eps'] = 1e-10 * np.trace(st['MCD_Sigma']) / L

    st['DPC_gamma'] = cached('DPC_gamma', lambda: dpc_gamma(X_all))

    def _lda_scatter():
        mu = Xtr.mean(0)
        Sw = np.zeros((L, L)); Sb = np.zeros((L, L))
        for c in np.unique(ytr):
            Xc = Xtr[ytr == c]
            muc = Xc.mean(0)
            Xcc = Xc - muc
            Sw += Xcc.T @ Xcc
            d = (muc - mu)[:, None]
            Sb += Xc.shape[0] * (d @ d.T)
        return Sw + (1e-4 * np.trace(Sw) / L) * np.eye(L), Sb
    Sw, Sb = cached('LDA_SwSb', _lda_scatter)
    st['LDA_Sw'], st['LDA_Sb'] = Sw, Sb

    st['LASSO_w'] = cached('LASSO_w', lambda: np.asarray(
        (lambda m: (m.fit(), m.scores_)[1])(LassoBS(Xtr, ytr))))
    st['LR_w'] = cached('LR_w', lambda: np.asarray(
        (lambda m: (m.fit(), m.scores_)[1])(LogisticBS(Xtr, ytr))))

    def _hogsp():
        m_ = HOGSP(X_all); m_.fit(); return np.asarray(m_.scores_)
    st['HOGSP_w'] = cached('HOGSP_w', _hogsp)

    def _cae():
        m_ = ConcreteAE(X_all); m_.fit(); return np.asarray(m_.probs_)
    st['CAE_w'] = cached('CAE_w', _cae)

    from sklearn.feature_selection import mutual_info_classif
    st['MI_w'] = cached('MI_w', lambda: mutual_info_classif(
        Xtr, ytr, random_state=42))
    st['ETP_w'] = cached('ETP_w', lambda: entropy_hist(X_all))

    def _nhmc():
        m_ = NHMC(Xtr, ytr, n_states=2, max_iter=80, n_jobs=4)
        m_.fit()
        return np.asarray(m_.scores_)          # J_n: inter-class correlation, lower is better
    st['NHMC_w'] = cached('NHMC_w', _nhmc)

    # write newly computed statistics back to the cache (the two MCD entries
    # come from the stats cache and are not written)
    new = {k: v for k, v in st.items()
           if k not in saved and k not in ('MCD_Sigma', 'MCD_eps')}
    if new:
        saved.update(new)
        np.savez(wpath, **saved)
        logger.info('  weights npz -> %s (%d keys)', wpath.name, len(saved))
    return st


# ---------------------------------------------------------------- J(S)


# ---------------------------------------------------------------- main flow
def parse_args(argv):
    """Parse the command line: dataset list plus cache refresh options.

    Usage:
      python3 run_criterion_values.py                 # default datasets, use cache
      python3 run_criterion_values.py --refresh       # force recomputation
      python3 run_criterion_values.py --refresh=mRMR  # recompute mRMR only
      python3 run_criterion_values.py botswana        # specific datasets
    """
    dss, refresh, refresh_method = [], False, None
    for a in argv:
        if a == '--refresh':
            refresh = True
        elif a.startswith('--refresh='):
            refresh, refresh_method = True, a.split('=', 1)[1]
        elif a.startswith('--'):
            raise SystemExit(f'unknown option: {a}')
        else:
            dss.append(a)
    return (dss or DEFAULT_DS), refresh, refresh_method


def main():
    import sys as _sys
    dss, refresh, refresh_method = parse_args(_sys.argv[1:])
    for ds in dss:
        tag = TAG_MAP.get(ds, ds.upper())
        run_one(ds, tag, refresh=refresh, refresh_method=refresh_method)


def run_one(ds, tag, refresh=False, refresh_method=None):
    ensure_dirs()
    logger = make_logger(tag)

    X_all, y_all, Xtr, Xte, ytr, yte = ds_mod.load_split(
        ds, test_size=CLF['test_size'], random_state=CLF['random_state'])
    L = X_all.shape[1]
    logger.info('==== criterion values (J(S1) vs J(S2)), %s ====', ds)
    logger.info('  X_all=%s labeled=%d L=%d',
                X_all.shape, int((y_all > 0).sum()), L)
    if refresh:
        if refresh_method:
            logger.info('  --refresh=%s: recomputing %s only, others use cache',
                        refresh_method, refresh_method)
        else:
            logger.info('  --refresh: forcing full recomputation')

    stats_npz = get_dataset_stats(ds, X_all, max_pixels=None,
                                  force=False, logger=logger)
    st = build_full_stats(X_all, Xtr, ytr, stats_npz, logger, ds=ds)
    st['_Xtr'] = Xtr   # mRMR subset correlations use the training split

    rows = []
    n_hit = n_calc = 0
    for m in METHODS:
        for k in KS:
            cpath = CRIT_DIR / f'{ds}_{m}_k{k}.json'
            do_cache = (not refresh) or (refresh_method is not None
                                         and m != refresh_method)
            if do_cache and cpath.exists():
                d = json.loads(cpath.read_text(encoding='utf-8'))
                J1, J2, dJ, rel = d['J1'], d['J2'], d['dJ'], d['rel']
                n_hit += 1
                logger.info('  [CACHE] %-12s k=%2d  hit %s', m, k,
                            cpath.name)
            else:
                n_calc += 1
                f1 = SEL_DIR / f'{ds}_{m}_k{k}_S1.npy'
                f2 = SEL_DIR / f'{ds}_{m}_k{k}_S2.npy'
                S1 = cache_utils.load_npy(f1, logger).astype(int)
                S2 = cache_utils.load_npy(f2, logger).astype(int)

                if m == 'mRMR':
                    # path convention: S1 = prefix of the full-set greedy path,
                    # S2 = prefix of the path re-run on the remaining set
                    order = np.load(RANK_DIR / f'{ds}_mRMR.npy')
                    rem = np.setdiff1d(np.arange(L), S1)
                    sub = MRMR(Xtr[:, rem], ytr)
                    sub.fit()
                    path2 = rem[sub.order_]         # remaining path -> full-set indices
                    J1 = j_full(m, S1, st, path=order)
                    J2 = j_full(m, S2, st, path=path2)
                else:
                    J1 = j_full(m, S1, st)
                    J2 = j_full(m, S2, st)      # same full-set context: fair comparison
                dJ = J1 - J2
                rel = (dJ / J1 * 100.0) if J1 > 0 else float('nan')
                cpath.write_text(json.dumps(
                    {'dataset': ds, 'method': m, 'k': k,
                     'J1': J1, 'J2': J2, 'dJ': dJ, 'rel': rel},
                    ensure_ascii=False), encoding='utf-8')

            rows.append([m, k, J1, J2, dJ, rel])
            logger.info('  [%s] %-12s k=%2d  J1=%+.4e  J2=%+.4e  '
                        'dJ=%+.4e  rel=%+.2f%%',
                        tag, m, k, J1, J2, dJ, rel)

    # CSV
    out = CSV_DIR / f'{tag.lower()}_criterion_values.csv'
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Dataset', 'Method', 'k', 'J1', 'J2', 'ΔJ', 'relΔJ(%)'])
        for r in rows:
            w.writerow([tag, r[0], r[1]] + [f'{v:.6e}' for v in r[2:5]]
                       + [f'{r[5]:.4f}'])
    logger.info('CSV -> %s (%d rows)', out.name, len(rows))
    logger.info('criterion cache: %d hit / %d computed (%d total)',
                n_hit, n_calc, n_hit + n_calc)

    plot(rows, tag, logger)
    logger.info('==== DONE ====')


def plot(rows, tag, logger):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    by_m = {m: [] for m in METHODS}
    for r in rows:
        by_m[r[0]].append(r)

    # method types: search / greedy (sign of dJ is diagnostic) vs pure ranking
    # (dJ >= 0 by construction)
    SEARCH = {'mRMR', 'MCD', 'LDA'}

    fig, axes = plt.subplots(3, 4, figsize=(15.5, 10.5))
    axes = axes.ravel()
    for i, m in enumerate(METHODS):
        ax = axes[i]
        rs = sorted(by_m[m], key=lambda r: r[1])
        ks = [r[1] for r in rs]
        dJ = [r[4] for r in rs]
        ax.axhline(0, color='0.35', lw=0.9)
        ax.plot(ks, dJ, 'o-', color='#d62728', lw=1.8, ms=5)
        ax.set_title(f'{m}  [{("search" if m in SEARCH else "ranking")}]',
                     fontsize=10.5, fontweight='bold')
        ax.set_xticks(KS)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.3)
        lim = np.max(np.abs(dJ)) * 1.15 if np.max(np.abs(dJ)) > 0 else 1.0
        ax.set_ylim(-lim, lim)
        if i in (8, 9, 10, 11):
            ax.set_xlabel('Number of bands k', fontsize=10)
        if i % 4 == 0:
            ax.set_ylabel('ΔJ = J(S1) − J(S2)', fontsize=10)

    axes[11].axis('off')
    axes[11].text(0.02, 0.98,
                  'All ΔJ use full-set statistics (S1/S2: same rule, same scale)\n\n'
                  '[search] search / greedy (LDA-SFS, MCD, mRMR): no sign\n'
                  '  guarantee; a negative value means the first selection was\n'
                  '  not criterion-optimal (a genuine local optimum)\n'
                  '[ranking] pure ranking (the other 8): ΔJ >= 0 by\n'
                  '  construction, so the sign carries no information; the\n'
                  '  signal is in the relative magnitude ΔJ/J1 (last CSV column)\n'
                  f'data: results/csv/{tag}_criterion_values.csv',
                  va='top', ha='left', fontsize=9.5, family='monospace')

    fig.suptitle(f'{tag}: criterion re-selection differential  '
                 'ΔJ = J(S1) − J(S2)  (same full-context statistics)',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outp = FIG_DIR / f'fig_{tag.lower()}_criterion_delta.png'
    fig.savefig(outp, dpi=170)
    logger.info('fig -> %s', outp.name)


if __name__ == '__main__':
    main()
