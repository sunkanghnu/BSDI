# -*- coding: utf-8 -*-
"""Load config.yaml and expose it as a dict with defaults filled in."""
import yaml
from .paths import CONFIG_PATH

DEFAULTS = {
    'project': {'seed': 42, 'parallel_jobs': -1,
                'force_recompute': False, 'force_ranking': False,
                'force_stats': False},
    'classifier': {'name': 'knn', 'n_neighbors': 5, 'test_size': 0.8,
                   'random_state': 42, 'metric': 'euclidean'},
    'datasets': [],
    'methods': [],
    'ks': [5, 10, 15, 20, 25, 30],
    'random_baseline': {'n_groups': 100, 'seed': 42},
    'stats': {'max_pixels': None},
}


def load_config(path=None):
    cfg = dict(DEFAULTS)
    path = path or CONFIG_PATH
    if path.exists():
        user = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        _deep_update(cfg, user)
    return cfg


def _deep_update(base, patch):
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def dataset_cfg(name, cfg):
    for d in cfg['datasets']:
        if d['name'] == name:
            return d
    return {'name': name, 'tag': name.upper()}
