"""Service-equivalent context transform and feature maps for contextual_execution.

The service returns offsets = clip(phi(T(context)) @ W, -1, 1). T clips the context at +-3
support standard deviations (population, ddof 0) in the learner's own coordinates:

  linear basis (default, no normalisation): PCA-whitened coordinates of the raw support set
  quadratic basis with normalize_context:   per-feature z-scores of the support set

phi is [1, T(f)] for the linear basis and [1, z, z_i z_j (i <= j)] on the clipped z-scores for the
quadratic basis. These rules reproduced the prod API to ~1e-15 (api-sweep, voltgt and rs-vardecomp
runs), which is what lets a retained map be exported as a small, exact, offline policy.
"""
import itertools
import math

import numpy as np

CLIP_Z = 3.0


class Basis:
    KINDS = ("linear", "quadratic")

    def __init__(self, support, kind="linear", clip_z=CLIP_Z):
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}")
        S = np.asarray(support, float)
        self.kind = kind
        self.clip_z = float(clip_z)
        self.n, self.d = S.shape
        self.mu = S.mean(0)
        self.sd = S.std(0)
        _, s, Vt = np.linalg.svd(S - self.mu, full_matrices=False)
        self.Vt = Vt
        self.sdp = s / math.sqrt(len(S))
        self.iu = np.triu_indices(self.d)
        self.P = 1 + self.d if kind == "linear" else 1 + self.d + self.d * (self.d + 1) // 2

    # -- coordinates ---------------------------------------------------------------
    def z_coord(self, F):
        return (np.atleast_2d(np.asarray(F, float)) - self.mu) / self.sd

    def z_pca(self, F):
        return (np.atleast_2d(np.asarray(F, float)) - self.mu) @ self.Vt.T / self.sdp

    def z_max(self, F):
        """Largest |z| in both metrics: a conservative 'inside the support' measure."""
        return np.maximum(np.abs(self.z_coord(F)).max(1), np.abs(self.z_pca(F)).max(1))

    def transform(self, F):
        F = np.atleast_2d(np.asarray(F, float))
        if self.kind == "linear":
            Z = np.clip(self.z_pca(F), -self.clip_z, self.clip_z)
            return self.mu + (Z * self.sdp) @ self.Vt
        return self.mu + self.sd * np.clip(self.z_coord(F), -self.clip_z, self.clip_z)

    def phi(self, F):
        G = self.transform(F)
        one = np.ones((len(G), 1))
        if self.kind == "linear":
            return np.hstack([one, G])
        Z = (G - self.mu) / self.sd
        Q = np.einsum("ni,nj->nij", Z, Z)[:, self.iu[0], self.iu[1]]
        return np.hstack([one, Z, Q])

    def offsets(self, W, F):
        return np.clip(self.phi(F) @ np.asarray(W, float), -1, 1)

    # -- serialisation -----------------------------------------------------------
    def to_dict(self):
        return {"kind": self.kind, "clip_z": self.clip_z, "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "Vt": self.Vt.tolist(), "sdp": self.sdp.tolist()}

    @classmethod
    def from_dict(cls, d):
        b = cls.__new__(cls)
        b.kind, b.clip_z = d["kind"], float(d["clip_z"])
        b.mu, b.sd = np.asarray(d["mu"], float), np.asarray(d["sd"], float)
        b.Vt, b.sdp = np.asarray(d["Vt"], float), np.asarray(d["sdp"], float)
        b.d = len(b.mu)
        b.n = None
        b.iu = np.triu_indices(b.d)
        b.P = 1 + b.d if b.kind == "linear" else 1 + b.d + b.d * (b.d + 1) // 2
        return b


def fit_map(X, Y):
    """Least squares for the map-then-clip model, per output coordinate on its unclipped values.
    X: (n, P) design rows phi(F); Y: (n, m) observed offsets. Returns (W, max residual, determined)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    W = np.zeros((X.shape[1], Y.shape[1]))
    resid, determined = 0.0, True
    for m in range(Y.shape[1]):
        ok = np.abs(Y[:, m]) < 1 - 1e-12
        if ok.sum() < X.shape[1] or np.linalg.matrix_rank(X[ok]) < X.shape[1]:
            determined = False
            if ok.sum() == 0:
                continue
        W[:, m] = np.linalg.lstsq(X[ok], Y[ok, m], rcond=None)[0]
        resid = max(resid, float(np.abs(X[ok] @ W[:, m] - Y[ok, m]).max()))
    return W, resid, determined


def d_optimal_subset(X, P, max_exhaustive=20000):
    """Indices of a P-row subset of X with the largest |det| (exhaustive when small, else greedy)."""
    X = np.asarray(X, float)
    n = len(X)
    if n < P:
        return None
    if math.comb(n, P) <= max_exhaustive:
        best, best_s = -np.inf, None
        for s in itertools.combinations(range(n), P):
            sign, v = np.linalg.slogdet(X[list(s)])
            if sign != 0 and v > best + 1e-12:
                best, best_s = v, list(s)
        return best_s
    chosen = []
    for _ in range(P):  # greedy volume growth
        best, best_i = -np.inf, None
        for i in range(n):
            if i in chosen:
                continue
            M = X[chosen + [i]]
            v = np.linalg.slogdet(M @ M.T)[1]
            if v > best:
                best, best_i = v, i
        chosen.append(best_i)
    return chosen
