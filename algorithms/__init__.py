# -*- coding: utf-8 -*-
"""Algorithm registry: name -> (criterion class, supervision type).

To add a method: create a module under algorithms/ subclassing Criterion,
import it here and append it to METHODS; to run it, also add an entry with the
same name to the methods list in config.yaml.
"""
from .base import Criterion
from .logistic_regression import LogisticBS
from .entropy import Entropy
from .mutual_info import MutualInfo
from .lasso import LassoBS
from .fisher_lda import FisherLDA
from .mrmr import MRMR
from .mv_pca import MVPCA
from .maxdet import MaxDet
from .density_peak import DensityPeak
from .hogsp import HOGSP
from .concrete_vae import ConcreteAE
from .nhmc import NHMC

# (method name, supervision type, criterion class)
# supervised (6): mRMR / LDA / LASSO / LR / MI / NHMC
# unsupervised (6): MVPCA / MCD / DPC / HOGSP / CAE / ETP
METHODS = [
    ('mRMR', 'sup', MRMR),
    ('LDA', 'sup', FisherLDA),
    ('LASSO', 'sup', LassoBS),
    ('LR', 'sup', LogisticBS),
    ('MI', 'sup', MutualInfo),
    ('NHMC', 'sup', NHMC),
    ('MVPCA', 'unsup', MVPCA),
    ('MCD', 'unsup', MaxDet),
    ('DPC', 'unsup', DensityPeak),
    ('HOGSP', 'unsup', HOGSP),
    ('CAE', 'unsup', ConcreteAE),
    ('ETP', 'unsup', Entropy),
]

REGISTRY = {name: cls for name, _, cls in METHODS}

__all__ = ['Criterion', 'METHODS', 'REGISTRY',
           'MRMR', 'LDA', 'LASSO', 'LR', 'MI', 'NHMC',
           'MVPCA', 'MCD', 'DPC', 'HOGSP', 'CAE', 'ETP']
