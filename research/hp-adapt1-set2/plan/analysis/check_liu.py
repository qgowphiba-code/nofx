"""Offline checks on the Liu panel contact-heuristic baseline (no network).
argv[1] = path to data/liu_panel.json (read as data only)."""
import json, random, statistics as st, sys, time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from fasthp import greedy_fold, upper_bound

panel = json.load(open(sys.argv[1]))['cases']
seqs = [(c['case_id'].replace('liu2025_', ''), c['sequence'], c['reference_contacts'], c['published_method_best_contacts']) for c in panel]

# 1) reproduce stage_m.py contact_D1..D3 per chain (results/m-mixed-length/results.json)
reported = {'seq1': (2, 8, 9), 'seq2': (6, 11, 13), 'seq3': (0, 2, 5), 'seq4': (7, 7, 12), 'seq5': (13, 23, 24), 'seq6': (11, 22, 26), 'seq7': (29, 40, 45)}
print('== reproduction of stage M contact_D1/D2/D3 (tie rng = Random(len(chain)) per decision)')
ok = True; tot = [0, 0, 0]
for name, seq, ref, pub in seqs:
    got = tuple(greedy_fold(seq, d).final_contacts() for d in (1, 2, 3))
    tot = [a + b for a, b in zip(tot, got)]
    ok &= got == reported[name]
    print(f'  {name} L{len(seq)} ub={upper_bound(seq)} got {got} reported {reported[name]}')
print('  totals', tot, 'match' if ok else 'MISMATCH')

# 2) distribution over random tie-breaks (persistent rng per fold, eps 0)
print('== greedy contact heuristic, random tie-breaks: panel total over seeds')
dist = {}
for depth, nseeds in ((1, 300), (2, 300), (3, 300), (4, 100), (5, 30)):
    t0 = time.time(); totals = []; per = {n: [] for n, *_ in seqs}
    for s in range(nseeds):
        rng = random.Random(1000 + s); t = 0
        for name, seq, *_ in seqs:
            c = greedy_fold(seq, depth, 'rng', rng, 0.0).final_contacts(); per[name].append(c); t += c
        totals.append(t)
    q = sorted(totals)
    dist[depth] = {'mean': round(st.mean(totals), 1), 'sd': round(st.pstdev(totals), 1), 'min': q[0], 'p5': q[int(.05 * len(q))], 'median': q[len(q) // 2],
                   'p95': q[int(.95 * len(q)) - 1], 'max': q[-1], 'seeds': nseeds,
                   'per_chain_mean': {n: round(st.mean(v), 1) for n, v in per.items()}, 'seconds': round(time.time() - t0)}
    print(f'  depth {depth}:', json.dumps(dist[depth]))

# 3) search-only control as in stage_d.py: depth 3 + eps 0.1, first attempt greedy; best-so-far per chain
print('== best-of-k search-only control (depth 3, eps 0.1, first attempt eps 0), 10 independent control runs')
K = (1, 8, 16, 24, 32, 64, 128)
curves = {n: {k: [] for k in K} for n, *_ in seqs}
for run in range(10):
    for name, seq, *_ in seqs:
        rng = random.Random(77 + run); best = 0
        for a in range(1, K[-1] + 1):
            best = max(best, greedy_fold(seq, 3, 'rng', rng, 0.0 if a == 1 else 0.1).final_contacts())
            if a in K: curves[name][a].append(best)
for k in K:
    tot = [sum(curves[n][k][r] for n, *_ in seqs) for r in range(10)]
    print(f'  k={k:3d} panel total mean {st.mean(tot):.1f} [min {min(tot)}, max {max(tot)}] per chain mean',
          {n: round(st.mean(curves[n][k]), 1) for n, *_ in seqs})
print('  reference_contacts total', sum(r for *_, r, _ in seqs), '| paper-method best total', sum(p for *_, p in seqs))

# 4) exact reproduction of the stage_d.py control stream (rng_c = Random(5150)), to validate against the logs
print('== stage_d control-stream reproduction (best_ctrl after each round of 8)')
for name, depth, rounds in (('seq1', 2, 8), ('seq4', 2, 6), ('seq5', 3, 4)):
    seq = next(s for n, s, *_ in seqs if n == name); rng = random.Random(5150); best = 0; out = []
    for r in range(rounds):
        for k in range(8):
            best = max(best, greedy_fold(seq, depth, 'rng', rng, 0.1 if k else 0.0).final_contacts())
        out.append(best)
    print(f'  {name} depth {depth}: {out}')
