"""Local same-budget baselines (no API). Use them to check that Machina beats plain black-box search.

  cem_acquisition / powell_acquisition: search one fixed context with the same number of evaluations
  contextual_cem: CEM over linear (or quadratic) offset maps with the same updates x contexts as refine()
  oracle_fit: least-squares map fitted to best-known actions on the training set (a ceiling for the basis)
Note: our CEM/Powell are not tuned. Powell matched Machina's final level on larger baskets at 1,000
evaluations; a tuned contextual baseline is still on the to-do list (README).
"""
import numpy as np

from .pipeline import compose


def phi(f, quad=False):
    f = np.asarray(f, float)
    if not quad:
        return np.concatenate([[1.0], f])
    iu = np.triu_indices(len(f))
    return np.concatenate([[1.0], f, np.outer(f, f)[iu]])


def cem_acquisition(task, ctx, evals, seed, pop=10, elite=3):
    D = task.H * task.A
    rng = np.random.default_rng(seed)
    mu, sd = rng.uniform(-1, 1, D), np.full(D, 0.5)
    curve = []
    while len(curve) < evals:
        X = np.clip(mu + sd * rng.standard_normal((pop, D)), -1, 1)
        F = np.array([task.score(x.reshape(task.H, task.A), ctx) for x in X])
        curve += F.tolist()
        i = np.argsort(-F)[:elite]
        mu, sd = X[i].mean(0), np.maximum(X[i].std(0), 0.03)
    return {"curve": curve[:evals], "best": float(max(curve[:evals])),
            "final": float(task.score(mu.reshape(task.H, task.A), ctx))}


def powell_acquisition(task, ctx, evals, seed):
    from scipy.optimize import minimize
    D = task.H * task.A
    x0 = np.random.default_rng(100 + seed).uniform(-1, 1, D)
    curve = []

    def f(x):
        s = float(task.score(np.clip(x, -1, 1).reshape(task.H, task.A), ctx))
        curve.append(s)
        return -s
    r = minimize(f, x0, method="Powell", bounds=[(-1, 1)] * D, options={"maxfev": evals})
    return {"curve": curve[:evals], "best": float(max(curve[:evals])),
            "final": float(task.score(np.clip(r.x, -1, 1).reshape(task.H, task.A), ctx))}


def contextual_cem(task, base, train, eval_contexts, updates=20, quad=False, seed=31, n_ctx=16, pop=8, elite=2,
                   sigma=0.1):
    base = np.asarray(base, float)
    P = len(phi(train[0]["feat"], quad))
    rng = np.random.default_rng(seed + 7)
    mu, sd = np.zeros((base.size, P)), np.full((base.size, P), sigma)
    for _ in range(updates):
        ctxs = [train[i] for i in rng.choice(len(train), n_ctx, replace=False)]
        Phi = np.array([phi(c["feat"], quad) for c in ctxs])
        W = mu + sd * rng.standard_normal((pop, base.size, P))
        F = [np.mean([task.score(compose(base, w @ ph), c) for c, ph in zip(ctxs, Phi)]) for w in W]
        i = np.argsort(-np.array(F))[:elite]
        mu, sd = W[i].mean(0), np.maximum(W[i].std(0), 0.01)
    return [float(task.score(compose(base, mu @ phi(c["feat"], quad)), c)) for c in eval_contexts]


def oracle_fit(task, base, train, best_actions, eval_contexts, quad=False):
    """best_actions: {cid: H x A actions} for the training contexts."""
    base = np.asarray(base, float)
    X = np.array([phi(c["feat"], quad) for c in train])
    Y = np.array([(np.asarray(best_actions[c["cid"]], float) - base).ravel() for c in train])
    W = np.linalg.lstsq(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ Y, rcond=None)[0].T
    return [float(task.score(compose(base, W @ phi(c["feat"], quad)), c)) for c in eval_contexts]
