# -*- coding: utf-8 -*-
"""Training loop for the Dropout Concrete Autoencoder (CPU version)."""
import numpy as np
import torch
from .models import ConcreteVAE


def train_concrete(X, *, hidden_dim=(128,), lr=1e-3, epochs=40, batch_size=256,
                   start_temperature=1.0, end_temperature=1e-3, lam1=0.005,
                   seed=0, verbose=True):
    """Train the Concrete AE and return the gate open probabilities probs
    (length = L).

    X : (n_samples, n_bands) float32, already scaled to [0, 1].
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = np.ascontiguousarray(X, dtype=np.float32)

    model = ConcreteVAE(X.shape[1], X.shape[1], list(hidden_dim), lam1=lam1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = len(X)
    n_batches = max(1, n // batch_size)
    # temperature annealing: multiplied by r at every training step, decaying
    # exponentially from start to end. The denominator is the total number of
    # steps n_batches*epochs (the original code divided by batch_size once
    # more).
    r = (end_temperature / start_temperature) ** (1.0 / (n_batches * epochs))
    model.sampled_layer.temperature = start_temperature

    # step learning rate (milestones from the original config); simply never
    # triggers when epochs is small
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[10, 25, 35], gamma=0.1)

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        total = 0.0
        for bi in range(n_batches):
            idx = perm[bi * batch_size:(bi + 1) * batch_size]
            xb = torch.from_numpy(X[idx])
            opt.zero_grad()
            _, m, loss, penalty = model(xb)
            loss.backward()
            opt.step()
            model.sampled_layer.temperature *= r
            total += loss.item()
        scheduler.step()

        if verbose:
            probs = model.sampled_layer.probs.detach().numpy()
            top = np.argsort(-probs)[:10]
            print('[CAE epoch {:02d}] loss={:.4e} penalty={:.4f} '
                  'temp={:.4f} top10={}'.format(
                      epoch + 1, total / n_batches, penalty.item(),
                      model.sampled_layer.temperature, top.tolist()))

    return model.sampled_layer.probs.detach().numpy()
