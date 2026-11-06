"""Offline: contact heuristic on the Stage A/P short-chain oracle set, first-tie (as stage_p.py) vs fair ties.
Imports the reviewed copies in ./vendor (pure computation, no network)."""
import json, random, statistics as st, sys, time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / 'vendor'))
t0 = time.time()
from grading import grade, eval_seqs, optimum  # builds the 216-state oracle set from fixed seeds
from lattice_env import State, plan
print('oracle set built', len(grade), 'states;', sum(len(g['opt']) < len(g['children']) for g in grade), 'informative;', f'{time.time()-t0:.0f}s')


def cval(s, d):
    if s.terminal: return s.utility()
    if d == 0: return s.partial_score()
    return max(cval(s.step(a), d - 1) for a in s.legal)


info = [g for g in grade if len(g['opt']) < len(g['children'])]
opt_total = sum(round((optimum[ci] - .5) * 2 * State.initial(s).upper_bound) for ci, s in enumerate(eval_seqs))
print('optimum greedy total', opt_total)
for depth in (1, 2, 3):
    first = []; fair = []; reg_fair = []
    for g in info:
        s = g['state']; vals = [(a, cval(s.step(a), depth - 1)) for a in s.legal]
        m = max(v for _, v in vals); ties = [a for a, v in vals if v == m]
        first.append(float(ties[0] in g['opt']))
        fair.append(sum(a in g['opt'] for a in ties) / len(ties))
        reg_fair.append(sum(g['vstar'] - g['child_vstar'][a] for a in ties) / len(ties))
    # greedy folds: first-tie (stage_p) and random tie-break distribution
    def fold(rng):
        tot = 0
        for seq in eval_seqs:
            s0 = State.initial(seq)
            while not s0.terminal: s0 = s0.step(plan(s0, depth, 'C2', rng)[0])
            tot += s0.contacts if s0.complete else 0
        return tot
    g_first = fold(None); g_rand = [fold(random.Random(500 + i)) for i in range(200)]
    print(f'depth {depth}: move acc first-tie {st.mean(first):.3f} | fair (expected over ties) {st.mean(fair):.3f} regret {st.mean(reg_fair):.4f}'
          f' | greedy first-tie {g_first}/{opt_total}, random-tie mean {st.mean(g_rand):.1f} [p5 {sorted(g_rand)[10]}, p95 {sorted(g_rand)[189]}] of {opt_total}')
n = len(info)
for p in (0.66, 0.81, 0.93):
    print(f'binomial SE at p={p}: {((p*(1-p))/n)**.5:.3f}  (n={n}); one decision = {1/n:.4f}')
