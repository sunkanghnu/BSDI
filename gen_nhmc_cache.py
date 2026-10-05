# -*- coding: utf-8 -*-
"""Generate the rankings + selections caches for the NHMC method, reusing the
pipeline interface.

Only NHMC is handled: it is an independent criterion, so the full ranking is
computed once and S1 / S2 are sliced from it without re-running on the
remaining set.
Usage: python3 gen_nhmc_cache.py [indian_pines salinas ...]
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.paths import ensure_dirs, LOG_DIR
from src import datasets as ds_mod
from src.pipeline import get_or_compute_ranking, get_or_compute_selections
from src.algorithms import REGISTRY


def main():
    dss = sys.argv[1:] or ['indian_pines', 'salinas']
    ensure_dirs()
    logger = logging.getLogger('nhmc_cache')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_DIR / 'nhmc_cache.log', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s %(message)s',
                                      datefmt='%H:%M:%S'))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())

    cls = REGISTRY['NHMC']
    ks = [5, 10, 15, 20, 25, 30]
    for ds in dss:
        X_all, y_all, Xtr, Xte, ytr, yte = ds_mod.load_split(ds)
        L = X_all.shape[1]
        crit = cls(Xtr, ytr)                     # supervised: training split
        logger.info('==== %s (L=%d) NHMC ranking ... ====', ds, L)
        order = get_or_compute_ranking(ds, crit, force=False, logger=logger)
        crit.order_ = order
        for k in ks:
            S1, S2 = get_or_compute_selections(
                ds, 'NHMC', k, crit, L, force=False, logger=logger)
            logger.info('  %s k=%2d S1[:5]=%s S2[:5]=%s',
                        ds, k, S1[:5].tolist(), S2[:5].tolist())
    logger.info('==== DONE ====')


if __name__ == '__main__':
    main()
