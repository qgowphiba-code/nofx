"""Score Discovery's selected predictors and standard neighbourhood-selection baselines against the Sachs consensus network."""
import csv, json, glob, math, random, statistics as st
import numpy as np
rows = list(csv.DictReader(open('data/raw/sachs/cyto_full_data.csv')))
names = list(rows[0].keys()); safe = [n.replace('/', '_').lower() for n in names]; S = {n: s for n, s in zip(names, safe)}
edges = {frozenset((S[a.strip('"')] if a.strip('"') in S else a.strip('"').lower(), S[b.strip('"')] if b.strip('"') in S else b.strip('"').lower()))
         for a, b in (l.strip().split(',') for l in list(open('data/raw/sachs/cyto_full_target.csv'))[1:])}
nb = {v: {u for e in edges if v in e for u in e if u != v} for v in safe}
def score_sets(sel):           # sel: target -> selected predictor names
    prec = [len(set(sel[t]) & nb[t]) / max(1, len(sel[t])) for t in sel]
    pred = {frozenset((t, p)) for t in sel for p in sel[t]}
    tp = len(pred & edges); P = tp / max(1, len(pred)); R = tp / len(edges)
    return st.mean(prec), P, R, (2 * P * R / (P + R) if P + R else 0.0)
def baselines(seed, N, K):
    cells = random.Random(seed).sample(rows, N)
    X = np.array([[math.log(float(r[n]) + 1.0) for n in names] for r in cells]); X = (X - X.mean(0)) / X.std(0)
    C = np.corrcoef(X.T); Pm = np.linalg.inv(C); pc = -Pm / np.sqrt(np.outer(np.diag(Pm), np.diag(Pm)))
    out = {}
    for name, M in (('correlation', np.abs(C)), ('partial_correlation', np.abs(pc))):
        out[name] = {safe[i]: [safe[j] for j in np.argsort(-M[i]) if j != i][:K] for i in range(len(names))}
    las = {}
    for i in range(len(names)):            # Lasso path order: first K predictors to enter (coordinate descent on a decreasing lambda grid)
        y = X[:, i]; Z = np.delete(X, i, axis=1); idx = [j for j in range(len(names)) if j != i]; order = []
        for lam in np.geomspace(1.0, 1e-3, 60):
            b = np.zeros(Z.shape[1])
            for _ in range(100):
                for j in range(Z.shape[1]):
                    r = y - Z @ b + Z[:, j] * b[j]; z = Z[:, j] @ r / len(y)
                    b[j] = np.sign(z) * max(abs(z) - lam, 0)
            for j in np.nonzero(b)[0]:
                if idx[j] not in order: order.append(idx[j])
            if len(order) >= K: break
        las[safe[i]] = [safe[j] for j in order[:K]]
    out['lasso_neighbourhood'] = las
    rr = random.Random(seed + 7); out['random'] = {t: rr.sample([u for u in safe if u != t], K) for t in safe}
    return out
print(f'consensus network: {len(edges)} edges; mean neighbours per protein {st.mean(len(v) for v in nb.values()):.2f}')
print('| method | K | seeds | precision of selected predictors | skeleton precision | recall | F1 |'); print('|---|---:|---:|---:|---:|---:|---:|')
for K in (3, 5):
    files = sorted(glob.glob(f'runs/sachs/discovery_k{K}_s*.jsonl'))
    res = {'Adapt-1 Discovery': [], 'correlation': [], 'partial_correlation': [], 'lasso_neighbourhood': [], 'random': []}
    for f in files:
        d = [json.loads(l) for l in open(f)]
        if len(d) < 11: continue
        seed = d[0]['seed']
        sel = {r['target']: [p.replace('values.', '') for p in (r['input_paths'] or [])] for r in d}
        res['Adapt-1 Discovery'].append(score_sets(sel))
        for m, s2 in baselines(seed, 1000, K).items(): res[m].append(score_sets(s2))
    for m, v in res.items():
        if v: print(f'| {m} | {K} | {len(v)} | ' + ' | '.join(f'{st.mean(x[i]):.2f}' for i in range(4)) + ' |')
