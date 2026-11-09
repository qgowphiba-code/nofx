"""Best-known schedules (computed locally, never sent to the API) and textbook references.

Share of best = mean score / mean best-known score over a context set; 0 = naive benchmark.
  ac / ow / vwap / two- and three-leg: exact KKT optimum of the quadratic cost (true optimum)
  market making: CEM search started from Avellaneda-Stoikov and the naive quote
  hedge bands:   CEM search started from Whalley-Wilmott bands
  glide path:    L-BFGS-B from three starts
  alpha:         QP optimum then a local polish in action space
"""
import numpy as np

from . import tasks as T


def cem_maximize(f, x0_list, dim, iters=25, pop=32, elite=6, sigma=0.3, seed=0):
    rng = np.random.default_rng(seed)
    best_x, best_f = None, -np.inf
    for x0 in x0_list:
        fx = f(x0)
        if fx > best_f:
            best_x, best_f = x0.copy(), fx
    mu, sd = best_x.copy(), np.full(dim, sigma)
    for _ in range(iters):
        X = np.clip(mu + sd * rng.standard_normal((pop, dim)), -1, 1)
        F = np.array([f(x) for x in X])
        i = np.argsort(-F)[:elite]
        if F[i[0]] > best_f:
            best_x, best_f = X[i[0]].copy(), F[i[0]]
        mu = 0.7 * X[i].mean(0) + 0.3 * mu
        sd = np.maximum(0.7 * X[i].std(0) + 0.3 * sd, 0.02)
    return best_x, best_f


def best_known(task, ctx):
    """(actions H x A, score, references dict) for one context."""
    H, A = task.H, task.A
    ref = {}
    name = task.name
    if name in ("ac_liquidation", "ow_transient", "vwap_profile"):
        n = task.oracle(ctx)
        a = T.weights_to_actions(n)[:, None]
        ref["true_opt_score"] = (task.naive_obj(ctx) - task.objective(n, ctx)) / abs(task.naive_obj(ctx))
    elif name in ("two_asset_transition", "three_asset_transition"):
        n = task.oracle(ctx)
        a = task.oracle_to_actions(n)
        ref["true_opt_score"] = (task.naive_obj(ctx) - task.objective(n, ctx)) / abs(task.naive_obj(ctx))
    elif name == "glide_path":
        a = (2 * task.oracle(ctx) - 1)[:, None]
    elif name == "market_making":
        a_as = task.sched_to_actions(task.as_schedule(ctx["p"]))
        ref["as_score"] = task.score(a_as, ctx)
        a_naive = task.sched_to_actions(np.tile([1.0 / ctx["p"]["k"], 0.0], (H, 1)))
        x, _ = cem_maximize(lambda x: task.score(x.reshape(H, A), ctx), [a_as.ravel(), a_naive.ravel()], H * A,
                            seed=T.cid_seed(ctx["cid"]) % 1000)
        a = x.reshape(H, A)
    elif name == "hedge_bands":
        a_ww = task.bands_to_actions(task.ww_schedule(ctx["p"]))
        ref["ww_score"] = task.score(a_ww, ctx)
        x, _ = cem_maximize(lambda x: task.score(x.reshape(H, A), ctx),
                            [a_ww.ravel(), np.zeros(H * A), np.full(H * A, -0.5)], H * A,
                            seed=T.cid_seed(ctx["cid"]) % 1000)
        a = x.reshape(H, A)
    elif name == "alpha_liquidation":
        a, n = task.best_actions(ctx)
        ref["qp_score"] = (task.naive_obj(ctx) - task.objective(n, ctx)) / task.scale(ctx)
        blind = T.weights_to_actions(T.ACLiquidation.oracle(task, ctx))[:, None]
        ref["blind_ac_score"] = float(task.score(blind, ctx))
    elif hasattr(task, "best_actions"):  # your own task with its own optimiser
        a = np.asarray(task.best_actions(ctx), float)
    else:  # your own task without one: generic CEM search from the naive schedule (slower, approximate)
        naive = task.naive_actions(ctx)
        x0 = np.zeros(H * A) if naive is None else np.asarray(naive, float).ravel()
        x, _ = cem_maximize(lambda x: task.score(x.reshape(H, A), ctx), [x0], H * A, seed=T.cid_seed(ctx["cid"]) % 1000)
        a = x.reshape(H, A)
        ref["approximate"] = True
    return np.asarray(a, float).reshape(H, A), float(task.score(a, ctx)), ref


def textbook(task, ctx):
    """Score of the textbook rule for the task, where one exists (None otherwise)."""
    if task.name == "market_making":
        return float(task.score(task.sched_to_actions(task.as_schedule(ctx["p"])), ctx))  # Avellaneda-Stoikov
    if task.name == "hedge_bands":
        return float(task.score(task.bands_to_actions(task.ww_schedule(ctx["p"])), ctx))  # Whalley-Wilmott
    if task.name == "alpha_liquidation":
        return float(task.score(T.weights_to_actions(T.ACLiquidation.oracle(task, ctx))[:, None], ctx))  # signal-blind AC
    return None
