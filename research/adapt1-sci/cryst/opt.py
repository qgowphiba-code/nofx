"""Same-budget black-box optimizers over the command vector (local baselines; numpy allowed here)."""
import math, random
import numpy as np


def clip(x):
    return [max(-1.0, min(1.0, float(v))) for v in x]


def random_search(f, n, budget, seed):
    r = random.Random(seed); best, hist = -1, []
    for _ in range(budget):
        y = f([r.uniform(-1, 1) for _ in range(n)]); best = max(best, y); hist.append(best)
    return hist


def one_plus_one(f, n, budget, seed, sigma=0.3, x0=None):
    r = random.Random(seed)
    x = x0[:] if x0 else [r.uniform(-1, 1) for _ in range(n)]
    fx = f(x); hist = [fx]; best = fx
    while len(hist) < budget:
        y = clip([v + sigma * r.gauss(0, 1) for v in x]); fy = f(y)
        if fy >= fx:
            x, fx = y, fy; sigma *= math.exp(0.8)   # 1/5 success rule (Rechenberg, Kern et al. form)
        else:
            sigma *= math.exp(-0.2)
        sigma = min(max(sigma, 1e-4), 2.0)
        best = max(best, fx); hist.append(best)
    return hist


def cmaes(f, n, budget, seed, sigma0=0.5, x0=None, lam=None):
    """Standard (mu/mu_w, lambda)-CMA-ES (Hansen 2016 tutorial defaults); maximizes f on clipped points."""
    rng = np.random.default_rng(seed)
    lam = lam or 4 + int(3 * math.log(n)); mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); w /= w.sum(); mueff = 1 / (w ** 2).sum()
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n); cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff); cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs; chin = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    m = np.array(x0 if x0 is not None else rng.uniform(-0.5, 0.5, n)); sigma = sigma0
    C = np.eye(n); pc = np.zeros(n); ps = np.zeros(n); B = np.eye(n); D = np.ones(n); evals, best, hist, eig = 0, -1, [], 0
    while evals < budget:
        z = rng.standard_normal((lam, n)); y = z @ (B * D).T; x = m + sigma * y
        fit = []
        for xi in x:
            if evals >= budget: break
            v = f(clip(xi)); evals += 1; best = max(best, v); hist.append(best); fit.append(v)
        if len(fit) < lam: break
        idx = np.argsort(fit)[::-1][:mu]
        yw = (w[:, None] * y[idx]).sum(0); m = m + sigma * yw
        m = np.clip(m, -1.5, 1.5)
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (B @ ((B.T @ yw) / D))
        hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * evals / lam)) / chin < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * yw
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * (w[:, None, None] * np.einsum('ki,kj->kij', y[idx], y[idx])).sum(0)
        sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / chin - 1)); sigma = min(sigma, 2.0)
        if evals - eig > lam / (c1 + cmu) / n / 10:
            eig = evals; C = np.triu(C) + np.triu(C, 1).T; Dsq, B = np.linalg.eigh(C); D = np.sqrt(np.maximum(Dsq, 1e-20))
        if sigma * D.max() < 1e-6: break
    while len(hist) < budget: hist.append(best)
    return hist


def cmaes_best(f, n, budget, seed, restarts=4, sigma0=0.5):
    """Ceiling search: best of several CMA-ES restarts, returns (best value, best x)."""
    best = (-1, None)
    for r in range(restarts):
        holder = {'x': None, 'v': -1}
        def g(x):
            v = f(x)
            if v > holder['v']: holder['v'], holder['x'] = v, x
            return v
        cmaes(g, n, budget // restarts, seed * 100 + r, sigma0=sigma0)
        if holder['v'] > best[0]: best = (holder['v'], holder['x'])
    return best
