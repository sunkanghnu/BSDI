# -*- coding: utf-8 -*-
"""Main entry point: run the full re-selection differential experiment.

Usage:
    python3 run_all.py                 # run everything in config.yaml
    python3 run_all.py --datasets IP   # only Indian Pines
    python3 run_all.py --force         # overwrite selection/ranking/stats caches
    python3 run_all.py --jobs 4        # override the worker count

Flow (dataset by dataset):
    1. check data (data/<ds>/X.npy, y.npy; run src/data_prepare.py if missing)
    2. shared statistics (covariance / correlation, cached)
    3. methods in parallel (joblib): full ranking -> S1/S2 per k -> accuracy
       -> D / SE / t
    4. random baseline (B random subsets, cached)
    5. summary CSVs: results/csv/<ds>_k<k>.csv and results/csv/all_datasets.csv

Any cache hit skips the computation and is logged as CACHE-HIT.
"""
import argparse
import csv
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.paths import ensure_dirs, CSV_DIR, LOG_DIR
from src.config import load_config, dataset_cfg
from src import datasets as ds_mod
from src.stats import get_dataset_stats
from src.classifier import AccuracyTable
from src.pipeline import run_method, run_random_k
from src.algorithms import REGISTRY

# dataset short name -> directory name
DS_ALIAS = {'IP': 'indian_pines', 'BW': 'botswana', 'PU': 'paviau'}


def make_logger(tag):
    ensure_dirs()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fpath = LOG_DIR / f'{tag}_{ts}.log'
    logger = logging.getLogger(tag)
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


def run_dataset(ds_name, cfg, logger, jobs):
    """Full pipeline for one dataset (shared stats -> parallel methods ->
    random baseline -> CSV)."""
    tag = dataset_cfg(ds_name, cfg).get('tag', ds_name)
    t0 = time.time()
    logger.info('==== dataset %s (%s) ====', ds_name, tag)

    if not ds_mod.check_data(ds_name):
        logger.error('data/%s/X.npy or y.npy is missing; run '
                     'python3 src/data_prepare.py --dataset %s first',
                     ds_name, ds_name)
        raise FileNotFoundError(f'data missing for {ds_name}')

    X_all, y_all, Xtr, Xte, ytr, yte = ds_mod.load_split(
        ds_name, test_size=cfg['classifier']['test_size'],
        random_state=cfg['classifier']['random_state'])
    L = X_all.shape[1]
    logger.info('  X_all=%s  labeled=%d  L=%d  n_test=%d',
                X_all.shape, (y_all > 0).sum(), L, Xte.shape[0])

    # 2) shared statistics (cached)
    stats = get_dataset_stats(ds_name, X_all,
                              max_pixels=cfg['stats']['max_pixels'],
                              force=cfg['project']['force_stats'],
                              logger=logger)
    logger.info('  shared stats ready (Sigma/corr %dx%d)',
                stats['Sigma'].shape[0], stats['Sigma'].shape[1])

    # 3) accuracy table (one per dataset, shared across methods, k, baseline)
    acc_table = AccuracyTable(ds_name,
                              force=cfg['project']['force_recompute'],
                              logger=logger)
    logger.info('  accuracy table entries: %d', len(acc_table))

    # 4) methods in parallel: one task per method (all k values run serially
    #    inside the task)
    methods = [m for m in cfg['methods'] if m in REGISTRY]
    tasks = [delayed(run_method)(dataset_cfg(ds_name, cfg), m,
                                 X_all, Xtr, Xte, ytr, yte,
                                 acc_table, stats, cfg, logger)
             for m in methods]

    # the random baseline is split into one task per k (parallel at the outer
    # level; each returns an accuracy table holding its new entries)
    for k in cfg['ks']:
        tasks.append(delayed(run_random_k)(
            dataset_cfg(ds_name, cfg), k, Xtr, Xte, ytr, yte,
            acc_table, cfg, logger))
    backend = cfg['project'].get('backend', 'loky')
    results = Parallel(n_jobs=jobs, backend=backend, verbose=0)(tasks)

    # 5) merge the new accuracy entries from all tasks and write to disk
    #    (the threading backend shares one table, loky gives copies; both are
    #    merged through _local_new)
    n_methods = len(methods)
    method_results = results[:n_methods]
    for _, tbl in results:
        acc_table.merge_from(tbl)
    acc_table.save(logger)
    logger.info('  accuracy table entries after run: %d', len(acc_table))

    # 6) summary CSVs: one per dataset per k, plus the combined file
    all_rows = [row for tbl in method_results for row in tbl[0]]
    write_csvs(ds_name, tag, all_rows, cfg, logger)
    logger.info('==== dataset %s done in %.1f s ====', ds_name, time.time() - t0)
    return all_rows


def write_csvs(ds_name, tag, rows, cfg, logger):
    """rows: [method, k, q1, q2, delta, se_ind, t_ind, se_mc, t_mc, verdict]"""
    from src.paths import RAND_DIR
    rand_med, rand_p975 = {}, {}
    for k in cfg['ks']:
        p = RAND_DIR / f'{ds_name}_k{k}_q1.npy'
        if p.exists():
            q1s = np.load(p)
            rand_med[k], rand_p975[k] = (
                float(np.median(q1s)), float(np.percentile(q1s, 97.5)))
        else:
            rand_med[k], rand_p975[k] = float('nan'), float('nan')
    for k in cfg['ks']:
        out = CSV_DIR / f'{ds_name}_k{k}.csv'
        with open(out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Method', 'Q(S1)', 'Q(S2)', 'ΔQ', 'SE_ind', 't_ind',
                        'SE_mc', 't_mc', 'Verdict', 'Q1_rand_med',
                        'Q1_rand_p975'])
            for r in rows:
                if r[1] != k:
                    continue
                w.writerow([r[0], f'{r[2]:.4f}', f'{r[3]:.4f}', f'{r[4]:+.4f}',
                            f'{r[5]:.4f}', f'{r[6]:.2f}', f'{r[7]:.4f}',
                            f'{r[8]:.2f}', r[9],
                            f'{rand_med[k]:.4f}', f'{rand_p975[k]:.4f}'])
        logger.info('  CSV -> %s', out.name)

    # combined summary
    allp = CSV_DIR / 'all_datasets.csv'
    is_new = not allp.exists()
    with open(allp, 'a', newline='') as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(['Dataset', 'Method', 'k', 'Q(S1)', 'Q(S2)', 'ΔQ',
                        'SE_ind', 't_ind', 'SE_mc', 't_mc', 'Verdict',
                        'Q1_rand_med', 'Q1_rand_p975'])
        for r in rows:
            k = r[1]
            w.writerow([tag, r[0], k, f'{r[2]:.4f}', f'{r[3]:.4f}',
                        f'{r[4]:+.4f}', f'{r[5]:.4f}', f'{r[6]:.2f}',
                        f'{r[7]:.4f}', f'{r[8]:.2f}', r[9],
                        f'{rand_med[k]:.4f}', f'{rand_p975[k]:.4f}'])
    logger.info('  CSV append -> %s', allp.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default=None,
                    help='comma-separated dataset short names or directories, '
                         'e.g. IP,PU or indian_pines')
    ap.add_argument('--force', action='store_true',
                    help='overwrite selection/ranking/stats/accuracy caches')
    ap.add_argument('--jobs', type=int, default=None, help='number of workers')
    args = ap.parse_args()

    ensure_dirs()
    cfg = load_config()
    if args.force:
        cfg['project']['force_recompute'] = True
        cfg['project']['force_ranking'] = True
        cfg['project']['force_stats'] = True
    jobs = args.jobs or cfg['project']['parallel_jobs']
    logger = make_logger('run_all')
    logger.info('config: datasets=%s methods=%d ks=%s jobs=%s force=%s',
                [d['name'] for d in cfg['datasets']],
                len(cfg['methods']), cfg['ks'], jobs,
                cfg['project']['force_recompute'])

    if args.datasets:
        names = []
        for tok in args.datasets.split(','):
            tok = tok.strip()
            names.append(DS_ALIAS.get(tok.upper(), tok))
    else:
        names = [d['name'] for d in cfg['datasets']]

    # the combined file is rebuilt at the start of every run (cache hits make
    # reruns cheap)
    allp = CSV_DIR / 'all_datasets.csv'
    if allp.exists():
        allp.unlink()
        logger.info('reset %s (this run regenerates the full summary)', allp.name)

    t_all = time.time()
    for name in names:
        run_dataset(name, cfg, logger, jobs)
    logger.info('==== ALL DONE in %.1f s ====', time.time() - t_all)


if __name__ == '__main__':
    main()
