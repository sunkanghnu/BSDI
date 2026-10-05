# -*- coding: utf-8 -*-
"""Re-selection differential pipeline:
build S1/S2 -> accuracy (cached) -> D / SE / t -> verdict.

Cache layers:
1. Full rankings, cache/rankings/<ds>_<method>.npy
   Single-band independent criteria compute the full ranking once; any k is a
   prefix slice. S2 also comes from this ranking (the ranking of the remaining
   set equals the full ranking without its first k entries).
2. Selections, cache/selections/<ds>_<method>_k{k}_S1|S2.npy
   S1 and S2 are cached separately and reused unless recomputation is forced.
3. Accuracy table, cache/accuracy/<ds>_acc.pkl
   Band subset -> accuracy (frozenset key), shared across methods, k values,
   and the random baseline.
"""
import numpy as np
from .paths import RANK_DIR, SEL_DIR
from . import cache as cache_utils
from . import metrics
from .algorithms.base import Criterion


# ---------------------------------------------------------------- ranking cache
def get_or_compute_ranking(ds_name, crit, force=False, logger=None):
    """Return the full band ranking (L bands, best to worst), disk-cached."""
    path = RANK_DIR / f'{ds_name}_{crit.name}.npy'
    if not force:
        cached = cache_utils.load_npy(path, logger)
        if cached is not None:
            return cached
    logger.info('    ranking: computing %s on %s ...', crit.name, crit.sample)
    order = crit.fit()
    cache_utils.save_npy(path, order, logger)
    return order


# ---------------------------------------------------------------- S1 / S2
def _reselect_dependent(crit, S1, L):
    """Multi-band / search criteria: refit on the remaining set with the same
    criterion, take the top k, and map back to original band indices.
    Statistics of the remaining set (covariance / correlation / scatter) are
    recomputed."""
    rem = np.setdiff1d(np.arange(L), S1)
    X_rem = crit.X_[:, rem]
    sub = crit.__class__(X_rem, crit.y_)       # no shared stats: the set changed
    sub.fit()
    s2 = sub.select(len(S1))
    if hasattr(sub, 'refine'):
        s2 = sub.refine(s2)                    # swap-based local search
    return np.asarray(rem)[s2]


def get_or_compute_selections(ds_name, method_name, k, crit, L,
                              force=False, logger=None):
    """Build and cache S1 / S2. Returns (S1, S2)."""
    f1 = SEL_DIR / f'{ds_name}_{method_name}_k{k}_S1.npy'
    f2 = SEL_DIR / f'{ds_name}_{method_name}_k{k}_S2.npy'
    if not force:
        c1, c2 = cache_utils.load_npy(f1, logger), cache_utils.load_npy(f2, logger)
        if c1 is not None and c2 is not None:
            return c1.astype(int), c2.astype(int)

    S1 = crit.select(k)
    if hasattr(crit, 'refine'):
        S1 = crit.refine(S1)                   # swap-based local search
    if crit.independent:
        # independent criterion: the remaining-set ranking is the full
        # ranking without its first k entries
        S2 = crit.order_[k:2 * k]
        logger.info('    S2: %s independent, direct slice from full ranking',
                    method_name)
    else:
        S2 = _reselect_dependent(crit, S1, L)
        logger.info('    S2: %s dependent, re-run on remaining %d bands',
                    method_name, L - k)
    cache_utils.save_npy(f1, S1, logger)
    cache_utils.save_npy(f2, S2, logger)
    return S1.astype(int), S2.astype(int)


# ---------------------------------------------------------------- single method
def run_method(ds_cfg, method_name, X_all, Xtr, Xte, ytr, yte,
               acc_table, stats, cfg, logger):
    """Run one method on one dataset over all k values.

    Returns (rows, acc_table); the table carries the entries added by this
    task, which the caller merges.
    """
    tag = ds_cfg.get('tag', ds_cfg['name'])
    from .algorithms import REGISTRY
    cls = REGISTRY[method_name]
    L = X_all.shape[1]
    if cls.supervised:
        crit = cls(Xtr, ytr)
    else:
        crit = cls(X_all, None)
        if method_name == 'MaxDet':
            crit = cls(X_all, None, sigma=stats['Sigma'])
        elif method_name == 'DensityPeak':
            crit = cls(X_all, None, corr=stats['corr'])

    # 1) full ranking (cached)
    order = get_or_compute_ranking(ds_cfg['name'], crit,
                                   force=cfg['project']['force_ranking'],
                                   logger=logger)
    crit.order_ = order

    clf_params = cfg['classifier']
    n_test = Xte.shape[0]
    rows = []
    for k in cfg['ks']:
        # 2) S1 / S2 (cached)
        S1, S2 = get_or_compute_selections(
            ds_cfg['name'], method_name, k, crit, L,
            force=cfg['project']['force_recompute'], logger=logger)
        # 3) accuracy (cache lookup or compute)
        q1, pred1 = acc_table.score(S1, Xtr, Xte, ytr, yte, clf_params)
        q2, pred2 = acc_table.score(S2, Xtr, Xte, ytr, yte, clf_params)
        delta = q1 - q2
        # 4) inference
        se_ind = metrics.se_indep(q1, q2, n_test)
        t_ind = metrics.t_stat(delta, se_ind)
        if pred1 is not None and pred2 is not None:
            se_mc, b, c = metrics.se_mcnemar(pred1, pred2, yte)
            t_mc = metrics.t_stat(delta, se_mc)
        else:
            se_mc, t_mc = float('nan'), float('nan')
        v = metrics.verdict(t_ind)
        rows.append([method_name, k, q1, q2, delta,
                     se_ind, t_ind, se_mc, t_mc, v])
        logger.info('  [%s] %-12s k=%2d  Q1=%.4f  Q2=%.4f  D=%+.4f  '
                    't_ind=%6.2f  t_mc=%6.2f  %s',
                    tag, method_name, k, q1, q2, delta, t_ind, t_mc, v)
    return rows, acc_table


# ---------------------------------------------------------------- random baseline
def run_random_k(ds_cfg, k, Xtr, Xte, ytr, yte, acc_table,
                 cfg, logger, n_groups=None, seed=None):
    """Accuracy of B random subsets of the same size (reference display only,
    never a verdict condition). One parallel task per k value; every subset
    goes through the shared accuracy table. Returns the table instance.
    """
    from .paths import RAND_DIR
    tag = ds_cfg.get('tag', ds_cfg['name'])
    name = ds_cfg['name']
    L_all = Xtr.shape[1]
    n_groups = n_groups or cfg['random_baseline']['n_groups']
    seed = seed or cfg['random_baseline']['seed']
    out = RAND_DIR / f'{name}_k{k}_q1.npy'
    force = cfg['project'].get('force_recompute', False)
    if out.exists() and not force:
        q1s = cache_utils.load_npy(out, logger)
        logger.info('  [%s] random baseline k=%2d  CACHE-HIT '
                    '(med=%.4f p97.5=%.4f), skip recompute',
                    tag, k, float(np.median(q1s)),
                    float(np.percentile(q1s, 97.5)))
        return None, acc_table
    rng = np.random.RandomState(seed)
    clf_params = cfg['classifier']
    q1s = np.empty(n_groups)
    for b in range(n_groups):
        bands = rng.choice(L_all, k, replace=False)
        acc, _ = acc_table.score(bands, Xtr, Xte, ytr, yte, clf_params)
        q1s[b] = acc
    cache_utils.save_npy(out, q1s, logger)
    logger.info('  [%s] random baseline k=%2d  med=%.4f  p97.5=%.4f',
                tag, k, float(np.median(q1s)), float(np.percentile(q1s, 97.5)))
    return None, acc_table
