"""Local baselines on GB1 at a matched measurement budget (no repeated variants):
random screening; directed evolution (single-step recombination of the best single mutants, and iterated greedy walk);
GP-EI on one-hot sequences (machine-learning-guided design, sequential BO); MLDE (Wittmann et al. 2021 style: 384 random,
ridge on one-hot + pairwise one-hot, then measure the top predictions)."""
import sys, json, math, random, itertools
import numpy as np
sys.path.insert(0, 'gb1')
import land as L
B = int(sys.argv[1]); METHOD = sys.argv[2]; SEEDS = int(sys.argv[3])
CHECK = [n for n in (24, 48, 96, 192, 384, 500, 1000) if n <= B]
def onehot(v):
    x = np.zeros(80); [x.__setitem__(20 * i + L.AA.index(a), 1.0) for i, a in enumerate(v)]; return x
def report(seed, seq):
    best, out = 0.0, {}
    for n, v in enumerate(seq[:B], 1):
        best = max(best, L.fit(v))
        if n in CHECK: out[n] = best
    print(json.dumps({'method': METHOD, 'seed': seed, 'best': out, 'found_global': 'FWAA' in seq[:B], 'n': min(len(seq), B)}), flush=True)
def singles(v):
    return [v[:i] + a + v[i + 1:] for i in range(4) for a in L.AA if a != v[i]]
for seed in range(SEEDS):
    r = random.Random(seed); seq, seen = [], set()
    def add(v):
        if v not in seen: seen.add(v); seq.append(v)
    if METHOD == 'random':
        allv = [''.join(t) for t in itertools.product(L.AA, repeat=4)]; r.shuffle(allv); seq = allv[:B]
    elif METHOD == 'de_recomb':          # all single mutants of WT, then combine the best residue at each position
        for v in singles(L.WT): add(v)
        bestaa = [max(L.AA, key=lambda a: L.fit(L.WT[:i] + a + L.WT[i + 1:])) for i in range(4)]; add(''.join(bestaa))
    elif METHOD == 'de_walk':            # iterated greedy walk: scan single mutants of the incumbent, move if better
        cur = L.WT; add(cur)
        while len(seq) < B:
            cands = [v for v in singles(cur) if v not in seen]; r.shuffle(cands)
            if not cands: break
            for v in cands: add(v)
            nxt = max(seen, key=L.fit)
            if nxt == cur: break
            cur = nxt
    elif METHOD == 'mlde':               # 384 random training variants, ridge on one-hot + pairwise, then top predictions
        allv = [''.join(t) for t in itertools.product(L.AA, repeat=4)]
        train = r.sample(allv, min(384, B)); [add(v) for v in train]
        def feat(v):
            f = list(onehot(v))
            for (i, j) in itertools.combinations(range(4), 2):
                z = [0.0] * 400; z[20 * L.AA.index(v[i]) + L.AA.index(v[j])] = 1.0; f += z
            return f
        X = np.array([feat(v) for v in train]); y = np.array([L.fit(v) for v in train])
        w = np.linalg.solve(X.T @ X + 1.0 * np.eye(X.shape[1]), X.T @ y)
        pool = [v for v in allv if v not in seen]; P = np.array([feat(v) for v in pool]) @ w
        for k in np.argsort(-P)[:B - len(seq)]: add(pool[k])
    elif METHOD == 'gp_ei':
        allv = [''.join(t) for t in itertools.product(L.AA, repeat=4)]
        for v in r.sample(allv, 20): add(v)
        while len(seq) < B:
            X = np.array([onehot(v) for v in seq]); y = np.array([L.fit(v) for v in seq]); mu, sd = y.mean(), y.std() + 1e-9; yn = (y - mu) / sd
            top = sorted(seq, key=L.fit, reverse=True)[:3]
            pool = list({v for t in top for v in singles(t) if v not in seen} | {v for v in r.sample(allv, 2000) if v not in seen})
            Xs = np.array([onehot(v) for v in pool]); best = None
            for ls in (1.0, 1.5, 2.5):
                K = np.exp(-0.5 * ((X[:, None] - X[None]) ** 2).sum(-1) / ls ** 2) + 0.05 * np.eye(len(X))
                Lc = np.linalg.cholesky(K); a = np.linalg.solve(Lc.T, np.linalg.solve(Lc, yn)); lml = -0.5 * yn @ a - np.log(np.diag(Lc)).sum()
                if best is None or lml > best[0]: best = (lml, ls, Lc, a)
            _, ls, Lc, a = best
            ks = np.exp(-0.5 * ((Xs[:, None] - X[None]) ** 2).sum(-1) / ls ** 2); m = ks @ a
            vv = np.linalg.solve(Lc, ks.T); s = np.sqrt(np.clip(1 - (vv * vv).sum(0), 1e-9, None))
            z = (m - yn.max()) / s; ei = (m - yn.max()) * 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2))) + s * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
            add(pool[int(np.argmax(ei))])
    report(seed, seq)
