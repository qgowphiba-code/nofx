import sys, json, math, random
import numpy as np
sys.path.insert(0, 'pool'); import data
NAME, B, METHOD, SEEDS = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
X, y = data.load(NAME); X = np.array(X); y = np.array(y); N = len(y); ymax = y.max()
ERF = np.vectorize(math.erf)
for seed in range(SEEDS):
    r = random.Random(seed); tested = []
    if METHOD == 'random':
        tested = r.sample(range(N), B)
    else:
        tested = r.sample(range(N), 5)
        while len(tested) < B:
            Xt, yt = X[tested], y[tested]; mu, sd = yt.mean(), yt.std() + 1e-9; yn = (yt - mu) / sd
            un = [i for i in range(N) if i not in set(tested)]; Xs = X[un]; best = None
            for ls in (0.1, 0.2, 0.4, 0.8):
                K = np.exp(-0.5 * ((Xt[:, None] - Xt[None]) ** 2).sum(-1) / ls ** 2) + 0.01 * np.eye(len(Xt))
                L = np.linalg.cholesky(K); a = np.linalg.solve(L.T, np.linalg.solve(L, yn)); lml = -0.5 * yn @ a - np.log(np.diag(L)).sum()
                if best is None or lml > best[0]: best = (lml, ls, L, a)
            _, ls, L, a = best
            ks = np.exp(-0.5 * ((Xs[:, None] - Xt[None]) ** 2).sum(-1) / ls ** 2); m = ks @ a
            v = np.linalg.solve(L, ks.T); s = np.sqrt(np.clip(1 - (v * v).sum(0), 1e-9, None)); z = (m - yn.max()) / s
            ei = (m - yn.max()) * 0.5 * (1 + ERF(z / math.sqrt(2))) + s * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
            tested.append(un[int(np.argmax(ei))])
    best, curve = -1e9, {}
    for n, i in enumerate(tested, 1):
        best = max(best, y[i])
        if n in (10, 25, 50, 100): curve[n] = float(best / ymax)
    rank1 = next((n for n, i in enumerate(tested, 1) if y[i] >= ymax), None)
    print(json.dumps({'dataset': NAME, 'method': METHOD, 'seed': seed, 'curve': curve, 'found_max_at': rank1}), flush=True)
