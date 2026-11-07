"""Local comparators on the HTE datasets: random, GP-EI per substrate (one-hot), GP-EI with transfer (substrate + condition one-hot)."""
import sys, json, math, random
import numpy as np
sys.path.insert(0, 'hte')
import data
DS, BUDGET = sys.argv[1], int(sys.argv[2]); ORDERS = [int(x) for x in sys.argv[3].split(',')]; RSEEDS = int(sys.argv[4]); ONLY = sys.argv[5].split(',') if len(sys.argv) > 5 else None
D = data.load(DS)
levels = [sorted({c[i] for c in D['conditions']}) for i in range(3)]
def onehot(c):
    v = []
    for i in range(3): v += [1.0 if c[i] == l else 0.0 for l in levels[i]]
    return v
CX = {c: np.array(onehot(c)) for c in D['conditions']}

def gp_post(X, y, Xs, ls, noise=0.02):
    mu, sd = y.mean(), y.std() + 1e-6
    yn = (y - mu) / sd
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    K = np.exp(-0.5 * d2 / ls ** 2) + noise * np.eye(len(X))
    L = np.linalg.cholesky(K); a = np.linalg.solve(L.T, np.linalg.solve(L, yn))
    ks = np.exp(-0.5 * ((Xs[:, None, :] - X[None, :, :]) ** 2).sum(-1) / ls ** 2)
    v = np.linalg.solve(L, ks.T)
    return ks @ a * sd + mu, np.sqrt(np.clip(1 - (v * v).sum(0), 1e-9, None)) * sd, -0.5 * yn @ a - np.log(np.diag(L)).sum()

ERF = np.vectorize(math.erf)
def ei_pick(X, y, Xs, best):
    m, s, _ = max((gp_post(X, y, Xs, ls) for ls in (0.7, 1.0, 1.5, 2.5)), key=lambda t: t[2])
    z = (m - best - 0.01) / s
    ei = (m - best - 0.01) * 0.5 * (1 + ERF(z / math.sqrt(2))) + s * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return int(np.argmax(ei))

def run(method, order, seed):
    subs = list(D['substrates']); random.Random(order).shuffle(subs)
    r = random.Random(seed); HX, HY = [], []
    for e, sub in enumerate(subs):
        avail = sorted(c for c in D['conditions'] if (sub, c) in D['yield'])
        gmax = max(D['yield'][(sub, c)] for c in avail); sx = np.array(D['context'][sub])
        tested, ys, best, curve = [], [], 0.0, []
        for step in range(BUDGET):
            un = [c for c in avail if c not in tested]
            if method == 'random' or (step == 0 and (method == 'gp_substrate' or not HY)):
                c = r.choice(un)
            elif method == 'gp_substrate':
                c = un[ei_pick(np.array([CX[t] for t in tested]), np.array(ys), np.array([CX[t] for t in un]), max(ys))]
            else:
                Xc = [np.concatenate([sx, CX[t]]) for t in tested]
                X = np.array(HX + Xc); y = np.array(HY + ys)
                Xs = np.array([np.concatenate([sx, CX[t]]) for t in un])
                inc = max(ys) if ys else float(np.max(gp_post(X, y, Xs, 1.0)[0]))
                c = un[ei_pick(X, y, Xs, inc)]
            tested.append(c); ys.append(D['yield'][(sub, c)]); best = max(best, ys[-1]); curve.append(best / gmax)
        HX += [np.concatenate([sx, CX[t]]).tolist() for t in tested]; HY += ys
        HX = [list(v) for v in HX]
        print(json.dumps({'ds': DS, 'method': method, 'order': order, 'seed': seed, 'episode': e, 'curve': curve}), flush=True)

for order in ORDERS:
    if not ONLY or 'random' in ONLY:
        for s in range(RSEEDS): run('random', order, s)
    if not ONLY or 'gp_substrate' in ONLY:
        for s in range(max(1, RSEEDS // 3)): run('gp_substrate', order, s)
    if not ONLY or 'gp_transfer' in ONLY:
        run('gp_transfer', order, 0)
