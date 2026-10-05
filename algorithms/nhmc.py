# -*- coding: utf-8 -*-
"""NHMC (Feng et al., IEEE TGRS 2017): supervised band selection based on a
statistical wavelet model.

Procedure (following the paper):
1. A UWT (undecimated Haar wavelet transform) turns each training spectrum
   into a scale x band coefficient matrix;
2. one NHMC per band (non-homogeneous hidden Markov chain with a zero-mean
   GMM, k states) is trained, with EM (Baum-Welch) estimating the initial
   state distribution, inter-scale transition matrices and per-state variances;
3. Viterbi decoding gives a state-label matrix per training sample (a
   "semantic feature");
4. labels are averaged per class to form class-level features, and the mean
   pairwise normalised inter-class correlation J_n is computed per band;
5. bands are ranked by ascending J_n (lower inter-class correlation is better),
   giving the full order_.

Criterion properties: supervised=True, sample='train', independent=True. J_n
is computed per band column from class-level labels and does not depend on the
candidate pool, so the ranking of the remaining set equals the full ranking
without its first k entries and S2 can be sliced from the cached ranking.

Cost control: one small HMM per band (the sequence is the scale chain), EM in
the log domain, bands processed in parallel with joblib. The default is k=2
states because the paper reports less than 1% accuracy difference from 8
states, and the UWT level is floor(log2(L)) including the approximation level.
"""
import numpy as np
import pywt
from joblib import Parallel, delayed
from .base import Criterion

__all__ = ['NHMC']


# ---------------------------------------------------------------- helpers
def _logsumexp(a, axis):
    amax = np.max(a, axis=axis, keepdims=True)
    amax_s = np.squeeze(amax, axis=axis)
    return np.log(np.sum(np.exp(a - amax), axis=axis)) + amax_s


def _pad_to_pow2(x, n2):
    """Reflect-pad to n2 (pywt.swt requires len % 2**level == 0)."""
    if len(x) == n2:
        return x
    return np.pad(x, (0, n2 - len(x)), mode='reflect')


def _uwt_coeffs(X, level):
    """Haar UWT coefficient tensor, returned as (n_samples, n_scales, n_bands)
    with n_scales = level + 1 (detail bands plus the coarsest approximation).
    Each spectrum is reflect-padded to 2**level, transformed with the
    undecimated wavelet transform, and each level is trimmed back to the
    original length. Note that pywt>=1.8 returns swt as a list of (cA, cD)
    tuples per level.
    """
    n, L = X.shape
    n2 = 1 << level
    scales = []
    for x in X:
        coeffs = pywt.swt(_pad_to_pow2(x, n2), 'haar', level=level)
        arr = [cD[:L] for _, cD in coeffs] + [coeffs[-1][0][:L]]
        scales.append(np.stack(arr))                       # (n_scales, L)
    out = np.stack(scales)                                  # (n, n_scales, L)
    # standardise per (scale, band) column: scale magnitudes differ by many
    # orders of magnitude (~1e2-1e8) and EM degenerates to a single state
    # without this. J only depends on the discrete state labels, so it is
    # unaffected.
    mu = out.mean(axis=0, keepdims=True)
    sd = out.std(axis=0, keepdims=True)
    return (out - mu) / (sd + 1e-8)


# ---------------------------------------------------------------- NHMC core
def _log_gauss(o, var, eps=1e-12):
    """log N(o; 0, var), returned as (N, T, K)."""
    var = np.maximum(var, eps)
    return -0.5 * (np.log(2 * np.pi * var)[None, :, :]
                   + (o[:, :, None] ** 2) / var[None, :, :])


def _baum_welch(O, n_states=2, max_iter=80, tol=1e-5, seed=0):
    """Train one band-level NHMC (Baum-Welch in the log domain).

    O : (N, T) observations (N independent sequences, T = scale-chain length).
    Returns (pi, A, var):
        pi     : (K,) prior over the first scale
        A      : (T-1, K, K) transitions, A_s[i, j] = p(S_s=i | S_{s-1}=j)
        var    : (T, K) zero-mean Gaussian variance per scale and state
    """
    rng = np.random.RandomState(seed)
    N, T = O.shape
    K = n_states
    v = float(np.var(O)) + 1e-12
    var = np.empty((T, K))
    var[:] = (np.array([0.25 * v, 2.0 * v])[:K])[None, :]     # small/large init
    A = np.full((T - 1, K, K), 0.1 / (K - 1) if K > 1 else 0.0)
    for t in range(T - 1):
        np.fill_diagonal(A[t], 0.9)
    pi = np.full(K, 1.0 / K)
    logA = np.log(A)

    prev_ll = -np.inf
    for it in range(max_iter):
        logB = _log_gauss(O, var)                # (N, T, K)
        # ---- E step: forward / backward in the log domain ----
        logalpha = np.empty((N, T, K))
        logalpha[:, 0, :] = np.log(pi)[None, :] + logB[:, 0, :]
        for s in range(1, T):
            tmp = logalpha[:, s - 1, :][:, None, :] + logA[s - 1][None, :, :]
            logalpha[:, s, :] = _logsumexp(tmp, axis=2) + logB[:, s, :]
        logbeta = np.zeros((N, T, K))
        for s in range(T - 1, 0, -1):
            tmp = (logA[s - 1][None, :, :] + logB[:, s, :][:, :, None]
                   + logbeta[:, s, :][:, :, None])
            logbeta[:, s - 1, :] = _logsumexp(tmp, axis=1)
        logZ = _logsumexp(logalpha[:, T - 1, :], axis=1)     # (N,)
        ll = float(np.mean(logZ))
        if it > 0 and (ll - prev_ll) < tol * abs(prev_ll):
            break
        prev_ll = ll

        # ---- M step (Dirichlet smoothing avoids absorbing states) ----
        loggamma = logalpha + logbeta - logZ[:, None, None]
        gamma = np.exp(loggamma)                              # (N, T, K)
        # xi[n, s, i, j] is proportional to
        # alpha_s(i) A_{s+1}[j, i] b_{s+1}(j) beta_{s+1}(j), s = 0..T-2
        xi = np.empty((N, T - 1, K, K))
        for s in range(T - 1):
            tmp = (logalpha[:, s, :][:, :, None]
                   + logA[s][None, :, :]
                   + logB[:, s + 1, :][:, None, :]
                   + logbeta[:, s + 1, :][:, None, :]
                   - logZ[:, None, None])
            xi[:, s] = np.exp(tmp)                            # (N, i, j)
        a_prior = 1.0                                        # pseudo-count
        pi = (gamma[:, 0, :].sum(axis=0) + a_prior)
        pi = pi / pi.sum()
        for s in range(T - 1):
            num = xi[:, s, :, :].sum(axis=0) + a_prior        # (i, j)
            den = gamma[:, s, :].sum(axis=0) + a_prior * K    # (i,)
            A[s] = num / (den[:, None])
        for s in range(T):
            num = (gamma[:, s, :] * O[:, s, None] ** 2).sum(axis=0)
            den = gamma[:, s, :].sum(axis=0) + a_prior * K
            var[s] = num / den
        var = np.maximum(var, 1e-3)                           # variance floor
        logA = np.log(np.maximum(A, 1e-12))
    return pi, A, var


def _viterbi_batch(O, pi, A, var):
    """Batched Viterbi decoding; returns (N, T) state labels."""
    N, T = O.shape
    K = pi.size
    logA = np.log(np.maximum(A, 1e-12))
    logB = _log_gauss(O, var)                     # (N, T, K)
    logdelta = np.empty((N, T, K))
    psi = np.empty((N, T, K), dtype=int)
    logdelta[:, 0, :] = np.log(np.maximum(pi, 1e-12))[None, :] + logB[:, 0, :]
    for s in range(1, T):
        tmp = logdelta[:, s - 1, :][:, None, :] + logA[s - 1][None, :, :]
        psi[:, s, :] = np.argmax(tmp, axis=2)
        logdelta[:, s, :] = np.max(tmp, axis=2) + logB[:, s, :]
    states = np.empty((N, T), dtype=int)
    states[:, T - 1] = np.argmax(logdelta[:, T - 1, :], axis=1)
    for s in range(T - 2, -1, -1):
        states[:, s] = psi[np.arange(N), s + 1, states[:, s + 1]]
    return states


# ---------------------------------------------------------------- criterion
class NHMC(Criterion):
    name = 'NHMC'
    supervised = True
    sample = 'train'
    independent = True

    def __init__(self, X=None, y=None, n_states=2, max_iter=80,
                 n_jobs=4, seed=0, **kwargs):
        super().__init__(X, y, **kwargs)
        self.n_states = n_states
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.seed = seed
        self.scores_ = None

    def _fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n, L = X.shape
        level = int(np.ceil(np.log2(L)))
        W = _uwt_coeffs(X, level)                             # (n, T, L)
        T = W.shape[1]

        def train_one(b):
            return _baum_welch(W[:, :, b], self.n_states,
                               self.max_iter, seed=self.seed)

        params = Parallel(n_jobs=self.n_jobs)(
            delayed(train_one)(b) for b in range(L))

        def decode_one(b):
            pi, A, var = params[b]
            return _viterbi_batch(W[:, :, b], pi, A, var)

        labels = Parallel(n_jobs=self.n_jobs)(
            delayed(decode_one)(b) for b in range(L))
        labels = np.stack(labels, axis=2)                     # (n, T, L)

        # class-level features: mean label per class, band and scale
        classes = np.unique(y)
        feat = np.stack([
            labels[y == c].mean(axis=0) for c in classes])    # (C, T, L)
        # mean pairwise normalised inter-class correlation J_n over scales
        ln = feat / (np.sqrt((feat ** 2).sum(axis=1,
                                              keepdims=True)) + 1e-12)
        J = np.zeros(L)
        C = len(classes)
        for p in range(C):
            for q in range(p + 1, C):
                J += (ln[p] * ln[q]).sum(axis=0)
        J = J / (C * (C - 1) / 2.0)
        self.scores_ = np.asarray(J, dtype=np.float64)        # correlation, lower is better
        return np.argsort(J)                                  # ascending: low correlation first
