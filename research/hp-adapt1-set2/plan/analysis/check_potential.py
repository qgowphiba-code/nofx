"""Offline: does a cheap hand-made tie-break (compactness) raise the depth-3 contact-heuristic bar on the Liu panel?
Leaf score = (contacts, -spread) compared lexicographically; trapped = (-1, 0). argv[1] = data/liu_panel.json."""
import json, random, statistics as st, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from fasthp import Fold

panel = json.load(open(sys.argv[1]))['cases']


def spread(f, h_only):
    pts = [p for i, p in enumerate(f.chain) if (not h_only or f.seq[i] == 'H')]
    if len(pts) < 2: return 0.
    c = [sum(p[k] for p in pts) / len(pts) for k in range(3)]
    return sum((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2 for p in pts) / len(pts)


def value(f, depth, h_only):
    if len(f.chain) == f.n: return (f.contacts, -spread(f, h_only))
    lg = f.legal()
    if not lg: return (-1, 0.)
    if depth == 0: return (f.contacts, -spread(f, h_only))
    best = None
    for d in lg:
        f.push(d); v = value(f, depth - 1, h_only); f.pop()
        if best is None or v > best: best = v
    return best


def fold(seq, depth, rng, h_only):
    f = Fold(seq)
    while True:
        lg = f.legal()
        if not lg: break
        vals = []
        for d in lg:
            f.push(d); vals.append(value(f, depth - 1, h_only)); f.pop()
        m = max(vals); f.push(rng.choice([d for d, v in zip(lg, vals) if v == m]))
    return f.final_contacts()


for label, h_only in (('H-core compactness tie-break', True), ('whole-chain compactness tie-break', False)):
    totals = []
    for s in range(40):
        rng = random.Random(2000 + s); totals.append(sum(fold(c['sequence'], 3, rng, h_only) for c in panel))
    q = sorted(totals)
    print(f'depth 3, {label}: mean {st.mean(totals):.1f} sd {st.pstdev(totals):.1f} [min {q[0]}, p5 {q[2]}, p95 {q[-3]}, max {q[-1]}] (40 seeds)')
