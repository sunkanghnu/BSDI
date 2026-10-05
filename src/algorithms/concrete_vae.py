# -*- coding: utf-8 -*-
"""ConcreteAE (Dropout Concrete Autoencoder): unsupervised deep-learning band
selection.

Registered class: subclasses Criterion, with the kernel living in the
dl_concrete/ package (models.py + trainer.py).

Principle (see https://ieeexplore.ieee.org/abstract/document/10976710 and
dl-selection / iancovert/dl-selection):
- every band gets a learned gate (BinConcrete relaxation sampling, with an
  annealed temperature approaching a hard 0/1 mask);
- the masked input is reconstructed by an MLP decoder (self-supervised,
  independent of any classifier);
- loss = reconstruction MSE + lambda * mean number of open gates (sparsity);
  the open probabilities probs are the band importance scores, and the final
  ranking sorts by -probs.

Set-dependent criterion (independent=False): gates and decoder are trained
jointly on the full band set, so removing the first k bands and retraining
changes probs. S2 must be re-run on the remaining set.

CPU adaptations: device is fixed to cpu; max_pixels bounds the training size;
the reconstruction loss is MSE (the original code defaults to CrossEntropyLoss,
which is inappropriate for continuous spectra); the decoder input is
input_dim rather than a deterministic top-k slice, since the latter gives the
gates no reconstruction gradient.
"""
import numpy as np
from .base import Criterion
from .dl_concrete import train_concrete


class ConcreteAE(Criterion):
    name = 'CAE'
    supervised = False
    sample = 'all'
    independent = False

    def __init__(self, X=None, y=None, *, max_pixels=10000, epochs=40,
                 batch_size=256, lr=1e-3, hidden_dim=(128,),
                 start_temperature=1.0, end_temperature=1e-3, lam1=0.005,
                 seed=0, **kwargs):
        super().__init__(X, y, **kwargs)
        self.max_pixels = int(max_pixels)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.hidden_dim = tuple(hidden_dim)
        self.start_temperature = float(start_temperature)
        self.end_temperature = float(end_temperature)
        self.lam1 = float(lam1)
        self.seed = int(seed)

    def _fit(self, X, y=None):
        """Return the full band ranking (best to worst); select(k) takes the
        top k."""
        X = np.asarray(X, dtype=np.float64)

        # pixel subsampling with a fixed seed for reproducibility; the default
        # cap of 10000 keeps CPU training manageable
        if self.max_pixels and len(X) > self.max_pixels:
            rng = np.random.RandomState(self.seed)
            X = X[rng.choice(len(X), self.max_pixels, replace=False)]

        # per-band min-max scaling to [0, 1] (matches minmax_scale in the
        # original code)
        xmin = X.min(0)
        span = X.max(0) - xmin
        span[span == 0] = 1.0
        X = (X - xmin) / span

        probs = train_concrete(
            X,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            start_temperature=self.start_temperature,
            end_temperature=self.end_temperature,
            lam1=self.lam1,
            seed=self.seed,
        )
        self.probs_ = np.asarray(probs, dtype=np.float64)
        return np.argsort(-probs)
