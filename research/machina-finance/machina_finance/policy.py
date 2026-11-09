"""Frozen policies: a base schedule plus an exported retained refinement map. No API needed at runtime.

actions(context) = clip(base + clip(phi(T(features)) @ W, -1, 1), -1, 1)

The task's decoder turns actions into the real controls (trade fractions per slot, quotes, bands,
equity weights). Policies are plain JSON so they can be reviewed, diffed and audited.
"""
import json

import numpy as np

from . import tasks as T
from .basis import Basis

FORMAT = "machina-finance-policy/1"


class FrozenPolicy:
    def __init__(self, task_name, base, basis, W, meta=None):
        self.task_name = task_name
        self.task = T.get_task(task_name)
        self.base = np.asarray(base, float).reshape(-1, self.task.A)
        self.basis = basis
        self.W = np.asarray(W, float)
        self.meta = dict(meta or {})
        if self.W.shape != (basis.P, self.base.size):
            raise ValueError(f"W shape {self.W.shape} != ({basis.P}, {self.base.size})")

    # -- use ---------------------------------------------------------------------
    def offsets(self, features):
        return self.basis.offsets(self.W, [features])[0].reshape(self.base.shape)

    def actions(self, features):
        return np.clip(self.base + self.offsets(features), -1, 1)

    def controls(self, ctx_or_features):
        """Decoded controls for a context dict or a raw feature vector."""
        f = ctx_or_features["feat"] if isinstance(ctx_or_features, dict) else ctx_or_features
        return self.task.decode(T.as_rows(self.actions(f), self.task.H, self.task.A))

    def score(self, ctx):
        return float(self.task.score(self.actions(ctx["feat"]), ctx))

    def evaluate(self, contexts, best=None):
        """Mean score on contexts; with best-known scores, also the share of best (mean / mean best)."""
        s = np.array([self.score(c) for c in contexts])
        out = {"n": len(s), "scores": s.tolist(), "mean": float(s.mean())}
        if best is not None:  # dict {cid: score} or a list aligned with contexts
            b = np.array([best[c["cid"]] for c in contexts] if isinstance(best, dict) else best, float)
            out["share_of_best"] = float(s.mean() / b.mean())
        return out

    # -- io ----------------------------------------------------------------------
    def to_dict(self):
        return {"format": FORMAT, "task": self.task_name, "feature_names": list(self.task.feature_names),
                "base": self.base.tolist(), "basis": self.basis.to_dict(), "W": self.W.tolist(), "meta": self.meta}

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def from_dict(cls, d):
        if d.get("format") != FORMAT:
            raise ValueError(f"unsupported policy format {d.get('format')!r}")
        return cls(d["task"], d["base"], Basis.from_dict(d["basis"]), d["W"], d.get("meta"))

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))
