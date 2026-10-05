# -*- coding: utf-8 -*-
"""Dataset loading: data/<dataset>/X.npy (all pixels, N x L) and y.npy (N, 0 = background).

- X_all / y_all: all pixels (the context used by unsupervised criteria).
- Labelled pixels Xm / ym: only pixels with label > 0.
- Train/test split: fixed seed, stratified by class. Supervised criteria and
  the classifier share the same split.

Run src/data_prepare.py first to convert the raw .mat files to X.npy / y.npy.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from .paths import DATA_DIR


def dataset_dir(name):
    return DATA_DIR / name


def check_data(name):
    d = dataset_dir(name)
    return (d / 'X.npy').exists() and (d / 'y.npy').exists()


def load_raw(name):
    """Return all-pixel X (N x L) and y (N, int)."""
    d = dataset_dir(name)
    X = np.load(d / 'X.npy')
    y = np.load(d / 'y.npy')
    return X, y


def load_split(name, test_size=0.8, random_state=42):
    """Return (X_all, y_all, X_train, X_test, y_train, y_test).

    The split is taken over labelled pixels only; X_all holds all pixels and
    is meant for unsupervised criteria.
    """
    X, y = load_raw(name)
    m = y > 0
    Xm, ym = X[m], y[m]
    Xtr, Xte, ytr, yte = train_test_split(
        Xm, ym, test_size=test_size, stratify=ym, random_state=random_state)
    return X, y, Xtr, Xte, ytr, yte
