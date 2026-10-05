# -*- coding: utf-8 -*-
"""Dropout Concrete Autoencoder (deep-learning band selection) kernel package.

models.py : model definitions (ConcreteGates / Decoder / ConcreteVAE, CPU)
trainer.py: training loop returning the gate open probabilities probs

Called from outside through ConcreteAE(Criterion) in
src/algorithms/concrete_vae.py.
"""
from .models import ConcreteGates, Decoder, ConcreteVAE
from .trainer import train_concrete

__all__ = ['ConcreteGates', 'Decoder', 'ConcreteVAE', 'train_concrete']
