# -*- coding: utf-8 -*-
"""Dropout Concrete Autoencoder kernel (CPU version).

Ported from Hyperspectral-master (https://ieeexplore.ieee.org/abstract/document/10976710,
"Dropout Concrete Autoencoder for Band Selection on Hyperspectral Image Scenes"),
following the Concrete AE structure of dl-selection (iancovert/dl-selection).

Fixes applied to the original code, all implemented in this file:
1. get_inds: the original used argsort(descending=True)[-k:], i.e. the k bands
   with the *smallest* probabilities (clearly a typo); it now takes the top-k
   by probability.
2. forward: the original computed a mask-weighted input sampled_x but never
   used it, feeding the decoder a deterministic top-k slice instead (a numpy
   index, so the gate logits received no reconstruction gradient and were
   driven only by the sparsity penalty). This is now a standard Concrete AE:
   the reconstruction loss acts on the mask-weighted input and the gates get
   gradients through the Concrete reparameterisation.
3. The decoder input dimension is input_dim instead of selected_num, matching
   the mask-weighted input.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = np.finfo(float).eps


def clamp_probs(probs):
    eps = torch.finfo(probs.dtype).eps
    return torch.clamp(probs, min=eps, max=1 - eps)


def bernoulli_concrete_sample(logits, temperature, shape=torch.Size([])):
    """Relaxed sampling from the BinConcrete (Bernoulli Concrete) distribution.

    See Maddison et al. (2017) Eq. 16. The sampling path is differentiable
    (reparameterised) and is used during training as a differentiable binary
    mask.
    """
    uniform_shape = torch.Size(shape) + logits.shape
    u = clamp_probs(torch.rand(uniform_shape, dtype=torch.float32,
                               device=logits.device))
    return torch.sigmoid((F.logsigmoid(logits + EPSILON)
                          - F.logsigmoid(-logits + EPSILON)
                          + torch.log(u + EPSILON)
                          - torch.log(1 - u + EPSILON)) / temperature)


class ConcreteGates(nn.Module):
    """One binary gate per band: logits -> open probability probs.

    Training uses BinConcrete sampling with an annealed temperature to
    approximate a hard 0/1 mask (dropout semantics), so the network only
    "sees" the selected bands. After training, bands are ranked by probs.
    """

    def __init__(self, input_size, k, temperature=1.0, init=0.99,
                 implicit_temp=0.2):
        super().__init__()
        # init_logit makes the initial probs approx. init (gates nearly open)
        init_logit = -torch.log(1 / torch.tensor(init) - 1) * implicit_temp
        self.logits = nn.Parameter(torch.full(
            (input_size,), init_logit, dtype=torch.float32, requires_grad=True))
        self.input_size = input_size
        self.temperature = temperature
        self.implicit_temp = implicit_temp
        self.k = k

    @property
    def probs(self):
        """Gate open probability per band, i.e. the importance score in [0, 1]."""
        return torch.sigmoid(self.logits / self.implicit_temp)

    def sample(self, n_samples=None, sample_shape=None):
        if n_samples:
            sample_shape = torch.Size([n_samples])
        return bernoulli_concrete_sample(
            self.logits / self.implicit_temp, self.temperature, sample_shape)

    def forward(self, x, n_samples=None):
        n = n_samples if n_samples else 1
        m = self.sample(sample_shape=(n, len(x)))     # (n,B,L)
        x = x * m
        if not n_samples:
            x = x.squeeze(0)
            m = m.squeeze(0)
        return x, m

    def get_inds(self, num_features=None, threshold=None):
        """Return the selected band indices (ascending).

        Fix: take the top-k by probability. The original [-k:] (k smallest
        probabilities) was a typo and selected the least discriminative bands.
        """
        if num_features:
            inds = torch.argsort(self.probs, descending=True)[:num_features]
        elif threshold:
            inds = (self.probs > threshold).nonzero()[:, 0]
        else:
            raise ValueError('num_features or threshold must be specified')
        return torch.sort(inds)[0].cpu().numpy()

    def get_left_inds(self, num_features=None):
        selected_inds = self.get_inds(num_features)
        whole = set(range(self.input_size))
        return sorted(whole.symmetric_difference(set(selected_inds)))


class Decoder(nn.Module):
    """MLP decoder reconstructing the original spectrum from the mask-weighted
    input (sigmoid output in [0, 1])."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        dims = [input_dim] + list(hidden_dim) + [output_dim]
        self.fc_layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.activation = nn.ReLU()

    def forward(self, x):
        for fc in self.fc_layers[:-1]:
            x = self.activation(fc(x))
        return torch.sigmoid(self.fc_layers[-1](x))


class ConcreteVAE(nn.Module):
    """Dropout Concrete Autoencoder: gate layer + decoder + sparsity penalty."""

    def __init__(self, input_dim, output_dim, hidden_dim, lam1=0.005):
        super().__init__()
        self.sampled_layer = ConcreteGates(input_dim, k=None)
        self.decoder = Decoder(input_dim, hidden_dim, output_dim)
        self.lam1 = lam1

    def forward(self, x):
        # mask-weighted input (gates are Concrete-reparameterised, so gradients
        # reach the logits)
        sampled_x, m = self.sampled_layer(x)
        y = self.decoder(sampled_x)
        loss_rec = F.mse_loss(y, x)                    # reconstruction loss (MSE)
        penalty = torch.mean(torch.sum(m, dim=-1))     # sparsity: number of open gates
        loss = loss_rec + self.lam1 * penalty
        return y, m, loss, penalty
