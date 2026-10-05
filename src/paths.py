# -*- coding: utf-8 -*-
"""Project paths. All scripts use paths relative to the repo root.

Layout:
    repo_root/
    config.yaml            # run configuration
    run_all.py             # main entry point
    plot_results.py        # figures (reads results/csv)
    data/<dataset>/        # X.npy (all pixels, N x L) / y.npy (N, 0 = background)
    cache/
        stats/             # shared statistics (covariance, correlation)
        rankings/          # cached full band rankings
        selections/        # cached S1 / S2 selections
        criterion_values/  # cached criterion values {ds}_{m}_k{k}.json
        accuracy/          # band subset -> accuracy lookup table
        random/            # random baseline accuracies (B random subsets)
    results/
        csv/               # summary CSV (one per dataset per k)
        logs/              # run logs
    figures/               # output figures
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / 'data'
CACHE_DIR = ROOT / 'cache'
STATS_DIR = CACHE_DIR / 'stats'
RANK_DIR = CACHE_DIR / 'rankings'
SEL_DIR = CACHE_DIR / 'selections'
CRIT_DIR = CACHE_DIR / 'criterion_values'
ACC_DIR = CACHE_DIR / 'accuracy'
RAND_DIR = CACHE_DIR / 'random'
RESULT_DIR = ROOT / 'results'
CSV_DIR = RESULT_DIR / 'csv'
LOG_DIR = RESULT_DIR / 'logs'
FIG_DIR = ROOT / 'figures'
CONFIG_PATH = ROOT / 'config.yaml'


def ensure_dirs():
    for d in (DATA_DIR, STATS_DIR, RANK_DIR, SEL_DIR, CRIT_DIR, ACC_DIR,
              RAND_DIR, CSV_DIR, LOG_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
