"""Offline: 'search without learned values' control on the Liu panel (lattice_env arm C1: unfinished leaves = neutral 0.5;
only terminal leaves inside the horizon score contacts, trapped = 0). argv[1] = data/liu_panel.json."""
import json, random, statistics as st, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from fasthp import Fold

def value(f, d):
    if len(f.chain) == f.n: return f.contacts
    lg = f.legal()
    if not lg: return -1
    if d == 0: return 0  # neutral leaf (utility 0.5 == zero contacts)
    best = -2
    for m in lg:
        f.push(m); v = value(f, d - 1); f.pop(); best = max(best, v)
    return best

def fold(seq, depth, rng):
    f = Fold(seq)
    while True:
        lg = f.legal()
        if not lg: break
        vals = []
        for m in lg:
            f.push(m); vals.append(value(f, depth - 1)); f.pop()
        mx = max(vals); f.push(rng.choice([m for m, v in zip(lg, vals) if v == mx]))
    return f.final_contacts()

panel = json.load(open(sys.argv[1]))['cases']
for depth in (1, 3):
    tot = [sum(fold(c['sequence'], depth, random.Random(3000 + s)) for c in panel) for s in range(100)]
    q = sorted(tot)
    print(f'neutral-leaf search depth {depth}: mean {st.mean(tot):.1f} sd {st.pstdev(tot):.1f} [p5 {q[5]}, p95 {q[94]}] (100 seeds)')
