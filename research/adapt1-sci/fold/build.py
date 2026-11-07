"""Enumerate conformations, build the designable-target bank with exact ceilings (numpy for the exhaustive part)."""
import sys, json, itertools
import numpy as np
sys.path.insert(0, 'fold')
import lattice2d as L
confs = L.enumerate_confs()
C = np.zeros((len(confs), len(L.PAIRS)), dtype=np.int16)
for a, c in enumerate(confs):
    for p in L.contacts(c): C[a, L.PAIR_IX[p]] = 1
seqs = np.array(list(itertools.product([0, 1], repeat=L.N)), dtype=np.int16)      # 1 = H
HH = np.array([[s[i] * s[j] for (i, j) in L.PAIRS] for s in seqs], dtype=np.int16)
E = -(C @ HH.T); Emin = E.min(0)
w = np.exp(-(E - Emin) / L.TEMP); P = w / w.sum(0)
uniq = (E == Emin).sum(0) == 1
targets = sorted({int(np.argmin(E[:, s])) for s in np.where(uniq)[0]})
bank = []
for a in targets:
    conf = confs[a]; cs = L.contacts(conf)
    deg = [0] * L.N
    for i, j in cs: deg[i] += 1; deg[j] += 1
    burial = np.array([[1 if d > 0 else 0 for d in deg]])
    bidx = int((burial[0] * (2 ** np.arange(L.N - 1, -1, -1))).sum())
    best = int(np.argmax(P[a]))
    bank.append({'conf_index': a, 'conf': conf, 'contacts': cs, 'degree': deg, 'ceiling': float(P[a, best]),
                 'best_seq': ''.join('PH'[v] for v in seqs[best]), 'random_mean': float(P[a].mean()),
                 'burial_heuristic': float(P[a, bidx]), 'burial_seq': ''.join('PH'[v] for v in burial[0])})
print('confs', len(confs), 'targets', len(bank))
print('ceiling mean %.3f | burial heuristic mean %.3f | random-seq mean %.4f' % (
    np.mean([b['ceiling'] for b in bank]), np.mean([b['burial_heuristic'] for b in bank]), np.mean([b['random_mean'] for b in bank])))
json.dump({'N': L.N, 'TEMP': L.TEMP, 'targets': bank}, open('fold/bank.json', 'w'))
np.save('fold/C.npy', C)
