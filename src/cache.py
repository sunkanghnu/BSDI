# -*- coding: utf-8 -*-
"""Thin cache layer for npy / npz / pickle: look up first, compute later.

Callers build the cache paths from project-relative dirs (see paths.py);
this module only provides uniform load/save plus hit logging. The logger is
injected by the caller; without it, logging is silent.
"""
import os
import pickle
import numpy as np


def load_npy(path, logger=None):
    path = str(path)
    if os.path.exists(path):
        _log(logger, 'CACHE-HIT', f'npy {path}')
        return np.load(path)
    return None


def save_npy(path, arr, logger=None):
    path = str(path)
    np.save(path, arr)
    _log(logger, 'CACHE-WRITE', f'npy {path}')


def load_npz(path, logger=None):
    path = str(path)
    if os.path.exists(path):
        _log(logger, 'CACHE-HIT', f'npz {path}')
        return np.load(path)
    return None


def save_npz(path, obj, logger=None):
    path = str(path)
    np.savez(path, **obj)
    _log(logger, 'CACHE-WRITE', f'npz {path}')


def load_pkl(path, logger=None):
    path = str(path)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            _log(logger, 'CACHE-HIT', f'pkl {path}')
            return pickle.load(f)
    return None


def save_pkl(path, obj, logger=None):
    path = str(path)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    _log(logger, 'CACHE-WRITE', f'pkl {path}')


def _log(logger, tag, msg):
    if logger is not None:
        logger.info('[%s] %s', tag, msg)
