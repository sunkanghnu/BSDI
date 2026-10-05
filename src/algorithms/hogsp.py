# -*- coding: utf-8 -*-
"""HOGSP (High-Order Graph-based Spectral band Selection): unsupervised,
set-dependent criterion (independent=False).

Ported from a MATLAB/C implementation (hogrsp-main) and adapted to the
Criterion interface of this framework.

Core idea: treat every band as an embeddable variable and learn a band
embedding H in R^{L x k} (one row per band) together with a sample
self-representation matrix S in R^{n x n}, minimising

    min_{H,S}  ||Y - YS||_F^2 + alpha * sum_i ||H_i||_2
              + gamma * ||S - F||_F^2
              + eta * tr(H' X L_f X' H) + lam * ||H' G H - I||_F^2

where Y = H'X is the sample representation in the band subspace, F is the
higher-order graph adjacency (accumulated T-step random-walk reachability), L_f
its Laplacian, and G = XX'/n is the band covariance. Alternating optimisation:
with H fixed, S is solved column-wise as a non-negative quadratic program; with
S fixed, H is updated in closed form by generalised eigendecomposition (the k
smallest generalised eigenvectors plus a scale shrinkage). The final band score
is the row norm ||H_i||_2, and the descending ranking is the full order.

Region samples (the original method relies on ERS superpixels and therefore
needs a spatial cube, while this framework only passes a pixel matrix):
- default: k-means spectral clustering on the pixel matrix, using cluster
  centres as region samples (spectral pseudo-regions, no spatial information
  needed, so the method runs standalone);
- if img_shape=(H, W) is injected and skimage is available, SLIC superpixels
  (spatial + spectral) are used instead, which is closer to the original ERS
  region partition.
"""
import numpy as np
from scipy import linalg
from scipy.optimize import nnls
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from .base import Criterion


class HOGSP(Criterion):
    name = 'HOGSP'
    supervised = False
    sample = 'all'
    independent = False

    def __init__(self, X=None, y=None, *, subspace_dim=20, knn_k=3,
                 high_order_T=3, sigma=1.0, alpha=0.002, lam=50.0, eta=0.01,
                 gamma=0.005, max_iter=50, tol=1e-5, n_regions=200,
                 img_shape=None, seed=0, **kwargs):
        super().__init__(X, y, **kwargs)
        self.subspace_dim = int(subspace_dim)
        self.knn_k = int(knn_k)
        self.high_order_T = int(high_order_T)
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_regions = int(n_regions)
        self.img_shape = img_shape
        self.seed = int(seed)

    # ---------------------------------------------------------- region samples
    def _build_regions(self, X):
        """Turn the pixel matrix into a region-level sample matrix (L x n_r)."""
        n, L = X.shape
        if self.img_shape is not None and len(self.img_shape) == 2:
            H_, W_ = self.img_shape
            if H_ * W_ == n:
                try:
                    from skimage.segmentation import slic
                    cube3 = X.reshape(H_, W_, L)
                    seg = slic(cube3, n_segments=self.n_regions, compactness=10,
                               start_label=1)
                    ids = np.unique(seg)
                    rows = [X[seg.ravel() == s].mean(0)
                            for s in ids if np.any(seg.ravel() == s)]
                    Xr = np.vstack(rows)
                    if Xr.shape[0] >= 5:
                        return Xr.T  # L x n_r
                except Exception:
                    pass  # skimage unavailable or failed: fall back to k-means
        # default: k-means spectral clustering, "pseudo-region" means
        km = KMeans(n_clusters=self.n_regions, random_state=self.seed,
                    n_init=3).fit(X)
        Xr = km.cluster_centers_  # n_r x L
        return Xr.T              # L x n_r

    # ---------------------------------------------------------- graph construction
    def _knn_graph(self, Xm):
        """Xm: L x n (one sample per column) -> Gaussian-kernel kNN adjacency
        (symmetric, zero diagonal, normalised)."""
        n = Xm.shape[1]
        Xt = Xm.T
        nbrs = NearestNeighbors(n_neighbors=self.knn_k + 1,
                                 metric='euclidean').fit(Xt)
        dists, idx = nbrs.kneighbors(Xt)
        C = np.zeros((n, n))
        for i in range(n):
            for jj in range(1, self.knn_k + 1):   # skip self
                j = idx[i, jj]
                C[i, j] = np.exp(-dists[i, jj] ** 2 / (2 * self.sigma ** 2))
        C = (C + C.T) / 2
        np.fill_diagonal(C, 0.0)
        mx = C.max()
        if mx > 0:
            C /= mx
        return C

    def _high_order_graph(self, C):
        """F = sum_{t=1..T} symmetrised C_norm^t ;  L_f = D - F."""
        n = C.shape[0]
        C = np.maximum(C, 0.0)
        C = (C + C.T) / 2
        np.fill_diagonal(C, 0.0)
        row_sum = C.sum(1)
        row_sum[row_sum == 0] = 1.0
        C_norm = C / row_sum[:, None]

        F = np.zeros((n, n))
        Cp = C_norm
        for _ in range(self.high_order_T):
            Ct = (Cp + Cp.T) / 2
            np.fill_diagonal(Ct, 0.0)
            F = F + Ct
            Cp = Cp @ C_norm
        F = np.maximum(F, 0.0)
        F = (F + F.T) / 2
        np.fill_diagonal(F, 0.0)
        mx = F.max()
        if mx > 0:
            F /= mx
        Lf = np.diag(F.sum(1)) - F
        return F, Lf

    # ---------------------------------------------------------- optimisation
    def _update_S(self, Xm, H, F):
        """With H fixed, solve min u'Ru - 2q'u, u >= 0 column by column
        (R = Yt'Yt + gamma*I)."""
        n = Xm.shape[1]
        Y = H.T @ Xm                    # k x n
        S = np.zeros((n, n))
        eye = np.eye(n - 1)
        for j in range(n):
            idx = np.ones(n, dtype=bool)
            idx[j] = False
            Yt = Y[:, idx]              # k x (n-1)
            yj = Y[:, j]
            fj = F[idx, j]
            R = Yt.T @ Yt + self.gamma * eye
            q = Yt.T @ yj + self.gamma * fj
            # R = L L' (Cholesky) -> min u'Ru-2q'u = ||L'u - L^{-1}q||^2
            Lc = np.linalg.cholesky(R)
            b = linalg.solve_triangular(Lc, q, lower=True)
            u = nnls(Lc.T, b)[0]
            s_j = np.zeros(n)
            s_j[idx] = u
            S[:, j] = s_j
        S = np.maximum(S, 0.0)
        np.fill_diagonal(S, 0.0)
        return S

    def _update_H(self, Xm, S, Lf, H_old, G):
        """With S fixed, update H by generalised eigendecomposition (the k
        smallest eigenvectors plus a scale shrinkage)."""
        d = Xm.shape[0]
        k = self.subspace_dim
        In = np.eye(Xm.shape[1])
        M = (In - S) @ (In - S).T
        row_norm = np.sqrt(np.sum(H_old ** 2, axis=1))
        D_H = np.diag(1.0 / (2.0 * np.sqrt(row_norm ** 2 + 1e-8)))
        Q = Xm @ M @ Xm.T + self.eta * Xm @ Lf @ Xm.T + self.alpha * D_H
        Q = (Q + Q.T) / 2
        G = (G + G.T) / 2

        mu, V = linalg.eigh(Q, G, subset_by_index=[0, k - 1])  # k smallest
        mu = np.real(mu)
        V = np.real(V)
        # G-normalisation
        for i in range(k):
            denom = np.sqrt(V[:, i] @ G @ V[:, i])
            if denom > 0:
                V[:, i] /= denom
        s2 = 1.0 - mu / (2.0 * self.lam)
        s2 = np.maximum(s2, 0.0)
        s2 = np.maximum(s2, 1e-3 ** 2)   # scale floor
        s = np.sqrt(s2)
        return V @ np.diag(s)

    def _objective(self, Xm, H, S, F, Lf, G):
        Y = H.T @ Xm
        rec = np.linalg.norm(Y - Y @ S, 'fro') ** 2
        sparse = np.sum(np.sqrt(np.sum(H ** 2, axis=1)))
        graph = np.linalg.norm(S - F, 'fro') ** 2
        lap = np.trace(H.T @ Xm @ Lf @ Xm.T @ H)
        k = H.shape[1]
        orth = np.linalg.norm(H.T @ G @ H - np.eye(k), 'fro') ** 2
        return rec + self.alpha * sparse + self.gamma * graph \
            + self.eta * lap + self.lam * orth

    # ---------------------------------------------------------- main entry
    def _fit(self, X, y=None):
        """Return the full band ranking (best to worst); select(k) takes the
        top k."""
        X = np.asarray(X, dtype=np.float64)
        # global min-max normalisation (matches normalize_cube in the original)
        xmin, xmax = X.min(), X.max()
        if xmax - xmin > 1e-12:
            X = (X - xmin) / (xmax - xmin)

        Xm = self._build_regions(X)          # L x n_r
        d, n = Xm.shape
        k = min(self.subspace_dim, d)
        if self.knn_k >= n:
            self.knn_k = max(1, n // 2)

        # first-order kNN graph -> higher-order graph
        C = self._knn_graph(Xm)
        F, Lf = self._high_order_graph(C)

        # init: G (band covariance), H (PCA loadings, G-normalised), S (= F)
        G = Xm @ Xm.T / n + 1e-6 * np.eye(d)
        pca = PCA(n_components=k).fit(Xm.T)
        H = pca.components_.T                # d x k
        for i in range(k):
            denom = np.sqrt(H[:, i] @ G @ H[:, i])
            if denom > 0:
                H[:, i] /= denom
        S = np.maximum(F, 0.0)
        np.fill_diagonal(S, 0.0)

        # alternating optimisation
        prev_loss = np.inf
        for _ in range(self.max_iter):
            S = self._update_S(Xm, H, F)
            H = self._update_H(Xm, S, Lf, H, G)
            loss = self._objective(Xm, H, S, F, Lf, G)
            rel = abs(prev_loss - loss) / max(abs(prev_loss), 1e-12)
            if rel < self.tol:
                break
            prev_loss = loss

        score = np.sqrt(np.sum(H ** 2, axis=1))   # one row per band
        self.scores_ = np.asarray(score, dtype=np.float64)
        return np.argsort(-score)
