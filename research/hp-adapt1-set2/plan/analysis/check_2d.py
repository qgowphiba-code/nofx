"""Offline 2D vs 3D baselines on the seven Liu panel sequences (they are the classic HP benchmark sequences).
2D best-known energies (standard benchmark values): 9, 9, 8, 14, 23, 21, 36 (total 120).
argv[1] = data/liu_panel.json. No network."""
import json, random, statistics as st, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from fasthp_nd import Fold, fold, upper_bound

panel = json.load(open(sys.argv[1]))['cases']
seqs = [(c['case_id'].replace('liu2025_', ''), c['sequence']) for c in panel]
BEST2D = {'seq1': 9, 'seq2': 9, 'seq3': 8, 'seq4': 14, 'seq5': 23, 'seq6': 21, 'seq7': 36}

# validation: 3D with stage_m.py seeding (fresh Random(len(chain)) per decision) must give 68 / 113 / 134
def stage_m_fold(seq, depth):
    f = Fold(seq, 3)
    while f.legal(): f.push(f.plan(depth, random.Random(len(f.chain))))
    return f.final_contacts()
print('3D reproduction (expect [68, 113, 134]):', [sum(stage_m_fold(s, d) for _, s in seqs) for d in (1, 2, 3)])

def dist(dim, depth, nseeds, neutral=False, base=4000):
    tot, traps, per = [], 0, {n: [] for n, _ in seqs}
    for k in range(nseeds):
        rng = random.Random(base + k); t = 0
        for n, s in seqs:
            f = fold(s, dim, depth, rng, 0.0, neutral); c = f.final_contacts(); traps += not f.complete(); per[n].append(c); t += c
        tot.append(t)
    q = sorted(tot)
    return {'mean': round(st.mean(tot), 1), 'sd': round(st.pstdev(tot), 1), 'p5': q[int(.05 * len(q))], 'p95': q[int(.95 * len(q)) - 1],
            'trapped_fold_rate': round(traps / (nseeds * len(seqs)), 3), 'per_chain_mean': {n: round(st.mean(v), 1) for n, v in per.items()}}

for dim in (3, 2):
    print(f'== {dim}D  ub per chain:', {n: upper_bound(s, dim) for n, s in seqs}, '| best-known total', 171 if dim == 3 else 120)
    for depth, ns in ((1, 200), (2, 200), (3, 200), (4, 60), (5, 20)):
        print(f'  contact heuristic depth {depth}:', json.dumps(dist(dim, depth, ns)))
    print('  neutral-leaf search depth 3:', json.dumps(dist(dim, 3, 100, neutral=True)))

print('== 2D best-of-k search-only (depth 3, eps 0.1, first attempt greedy), 10 runs')
K = (1, 8, 32, 128); curves = {n: {k: [] for k in K} for n, _ in seqs}
for r in range(10):
    for n, s in seqs:
        rng = random.Random(90 + r); best = 0
        for a in range(1, K[-1] + 1):
            best = max(best, fold(s, 2, 3, rng, 0.0 if a == 1 else 0.1).final_contacts())
            if a in K: curves[n][a].append(best)
for k in K:
    tot = [sum(curves[n][k][r] for n, _ in seqs) for r in range(10)]
    print(f'  k={k:3d}: total mean {st.mean(tot):.1f} [{min(tot)}-{max(tot)}]', {n: round(st.mean(curves[n][k]), 1) for n, _ in seqs}, 'best-known', BEST2D)
