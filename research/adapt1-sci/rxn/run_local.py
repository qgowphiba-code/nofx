"""Local comparators for Exp B: random, GP-BO per reaction, GP-BO with transfer (same info, same budget)."""
import sys, json, math, random
import numpy as np
sys.path.insert(0, 'rxn')
import sim

NEP, BUDGET, SEEDS = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
K = len(sim.CONDITIONS)
COND_X = np.array([[1.0 if c == j else 0.0 for j in range(4)] + [(T - 60) / 30, (math.log(tau) - math.log(8)) / math.log(4)] for c, T, tau in sim.CONDITIONS])

def measured(rx, k):
    return sim.measure(rx, k, random.Random(rx['id'] * 1000 + k))

def gp_post(X, y, Xs, ls, noise=0.05):
    mu, sd = y.mean(), y.std() + 1e-9
    yn = (y - mu) / sd
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    Kxx = np.exp(-0.5 * d2 / ls ** 2) + noise * np.eye(len(X))
    L = np.linalg.cholesky(Kxx)
    a = np.linalg.solve(L.T, np.linalg.solve(L, yn))
    ks = np.exp(-0.5 * ((Xs[:, None, :] - X[None, :, :]) ** 2).sum(-1) / ls ** 2)
    m = ks @ a
    v = np.linalg.solve(L, ks.T)
    var = np.clip(1.0 - (v * v).sum(0), 1e-9, None)
    lml = -0.5 * yn @ a - np.log(np.diag(L)).sum()
    return m * sd + mu, np.sqrt(var) * sd, lml

def choose_ei(X, y, Xs, best):
    fits = [gp_post(X, y, Xs, ls) for ls in (0.5, 0.8, 1.2, 2.0)]
    m, s, _ = max(fits, key=lambda t: t[2])
    z = (m - best - 0.01) / s
    ei = (m - best - 0.01) * 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2))) + s * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return int(np.argmax(ei))

def run(method, seed):
    r = random.Random(seed); hist_X, hist_y, out = [], [], []
    for e in range(NEP):
        rx = sim.make_reaction(e); g = sim.grid(rx); gmax = max(g); desc = np.array(rx['descriptors'])
        tested, ys, best_true, curve = [], [], 0.0, []
        for step in range(BUDGET):
            untested = [k for k in range(K) if k not in tested]
            if method == 'random' or (method == 'gp_episode' and step == 0) or (method == 'gp_transfer' and not hist_y and step == 0):
                k = r.choice(untested)
            elif method == 'gp_episode':
                X = COND_X[tested]; Xs = COND_X[untested]
                k = untested[choose_ei(X, np.array(ys), Xs, max(ys))]
            else:  # gp_transfer: descriptors + condition, all earlier reactions + this one
                Xc = np.hstack([np.tile(desc, (len(tested), 1)), COND_X[tested]]) if tested else np.zeros((0, 9))
                X = np.vstack([np.array(hist_X).reshape(-1, 9), Xc]) if hist_X else Xc
                y = np.array(hist_y + ys)
                Xs = np.hstack([np.tile(desc, (len(untested), 1)), COND_X[untested]])
                incumbent = max(ys) if ys else float(np.max(gp_post(X, y, Xs, 1.0)[0]))
                k = untested[choose_ei(X, y, Xs, incumbent)]
            tested.append(k); ys.append(measured(rx, k)); best_true = max(best_true, g[k]); curve.append(best_true / gmax)
        hist_X += [list(desc) + list(COND_X[k]) for k in tested]; hist_y += ys
        out.append({'method': method, 'seed': seed, 'episode': e, 'family': rx['family'], 'curve': curve})
    return out

for method in ('random', 'gp_episode', 'gp_transfer'):
    for s in range(SEEDS if method != 'gp_transfer' else max(1, SEEDS // 2)):
        for rec in run(method, s):
            print(json.dumps(rec), flush=True)
