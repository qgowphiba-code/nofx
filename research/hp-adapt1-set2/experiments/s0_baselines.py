"""S0 / C-base: frozen baseline cards (plan 2.3), computed locally before any API run.

For every chain of a suite (Eval-L, Eval-Liu; 2D suites with --dim 2):
  - contact heuristic at depths 1-5, 100 tie-break seeds each (per-chain contacts per seed);
  - H-core compactness tie-break variant at depth 3 (100 seeds);
  - search without learned values (neutral leaves) at depth 3 (100 seeds);
  - best-of-k search-only curves, k = 1..128, depth 3, eps 0.1 (first attempt greedy), 10 runs.
Tie-break rng per (chain, seed) is Random('<suite>-<kind>-<chain>-<seed>') (string seeds are stable).
Uses plan/analysis/fasthp_nd.py (validated: reproduces 68/113/134). Writes suites/baselines_<suite>.json.
Usage: s0_baselines.py <suite> [--dim 2] [--workers 3]
"""
import argparse
import json
import random
import statistics as st
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hpx import suites as SU  # noqa: E402
from hpx.vendor import ANALYSIS  # noqa: E402,F401
from fasthp_nd import Fold, fold  # noqa: E402

DEPTH_SEEDS = {1: 100, 2: 100, 3: 100, 4: 100, 5: 100}
K = (1, 2, 4, 8, 16, 32, 64, 128)


def spread_h(f):
    pts = [p for i, p in enumerate(f.chain) if f.seq[i] == 'H']
    if len(pts) < 2:
        return 0.
    c = [sum(p[k] for p in pts) / len(pts) for k in range(f.dim)]
    return sum(sum((p[k] - c[k]) ** 2 for k in range(f.dim)) for p in pts) / len(pts)


def compact_value(f, depth):
    if len(f.chain) == f.n:
        return (f.contacts, -spread_h(f))
    lg = f.legal()
    if not lg:
        return (-1, 0.)
    if depth == 0:
        return (f.contacts, -spread_h(f))
    best = None
    for d in lg:
        f.push(d)
        v = compact_value(f, depth - 1)
        f.pop()
        if best is None or v > best:
            best = v
    return best


def compact_fold(seq, dim, depth, rng):
    f = Fold(seq, dim)
    while True:
        lg = f.legal()
        if not lg:
            break
        vals = []
        for d in lg:
            f.push(d)
            vals.append(compact_value(f, depth - 1))
            f.pop()
        m = max(vals)
        f.push(rng.choice([d for d, v in zip(lg, vals) if v == m]))
    return f


def job(args):
    suite, kind, ci, seq, dim, depth, seeds = args
    out = []
    t0 = time.time()
    if kind == 'bestofk':
        for r in range(seeds):
            rng = random.Random(f'{suite}-bestofk-{ci}-{r}')
            best, curve = 0, []
            for a in range(1, K[-1] + 1):
                f = fold(seq, dim, 3, rng, 0.0 if a == 1 else 0.1)
                best = max(best, f.final_contacts())
                if a in K:
                    curve.append(best)
            out.append(curve)
    else:
        for s in range(seeds):
            rng = random.Random(f'{suite}-{kind}{depth}-{ci}-{s}')
            if kind == 'heuristic':
                f = fold(seq, dim, depth, rng)
            elif kind == 'neutral':
                f = fold(seq, dim, depth, rng, neutral=True)
            else:
                f = compact_fold(seq, dim, depth, rng)
            out.append([f.final_contacts(), int(not f.complete())])
    return (kind, depth, ci, out, round(time.time() - t0, 1))


def summarize(per_chain_lists):
    """per_chain_lists[ci] = per-seed contacts. Totals over chains, seed-aligned."""
    n = len(per_chain_lists[0])
    totals = sorted(sum(pc[s] for pc in per_chain_lists) for s in range(n))
    return {'mean_total': round(st.mean(totals), 2), 'sd_total': round(st.pstdev(totals), 2),
            'p5_total': totals[int(.05 * n)], 'p95_total': totals[int(.95 * n) - 1], 'seeds': n,
            'per_chain_mean': [round(st.mean(pc), 3) for pc in per_chain_lists],
            'per_chain_sd': [round(st.pstdev(pc), 3) for pc in per_chain_lists]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('suite')
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    data = SU.load(a.suite)
    seqs, dim = data['sequences'], data['dim']
    jobs = []
    for ci, seq in enumerate(seqs):
        for depth, ns in DEPTH_SEEDS.items():
            jobs.append((a.suite, 'heuristic', ci, seq, dim, depth, ns))
        jobs.append((a.suite, 'compact', ci, seq, dim, 3, 100))
        jobs.append((a.suite, 'neutral', ci, seq, dim, 3, 100))
        jobs.append((a.suite, 'bestofk', ci, seq, dim, 3, 10))
    jobs.sort(key=lambda j: -(len(j[3]) * (5 ** j[5] if j[1] != 'bestofk' else 128 * 125)))
    t0 = time.time()
    res = {}
    with Pool(a.workers) as pool:
        for kind, depth, ci, out, secs in pool.imap_unordered(job, jobs):
            res[(kind, depth, ci)] = out
            if len(res) % 25 == 0:
                print(f'{len(res)}/{len(jobs)} jobs, {time.time() - t0:.0f}s', flush=True)
    card = {'suite': a.suite, 'dim': dim, 'sequences': seqs, 'suite_sha256': json.loads((SU.SUITES / 'MANIFEST.json').read_text())[a.suite]['sha256'],
            'tie_rng': "Random('<suite>-<kind><depth>-<chain>-<seed>')", 'heuristic': {}, 'K': list(K)}
    for depth in DEPTH_SEEDS:
        pcs = [[c for c, _ in res[('heuristic', depth, ci)]] for ci in range(len(seqs))]
        card['heuristic'][str(depth)] = {**summarize(pcs), 'per_chain_seeds': pcs,
                                          'trapped_rate': sum(t for ci in range(len(seqs)) for _, t in res[('heuristic', depth, ci)]) / (len(seqs) * len(pcs[0]))}
    for kind in ('compact', 'neutral'):
        pcs = [[c for c, _ in res[(kind, 3, ci)]] for ci in range(len(seqs))]
        card[f'{kind}_depth3'] = {**summarize(pcs), 'per_chain_seeds': pcs}
    curves = [res[('bestofk', 3, ci)] for ci in range(len(seqs))]
    card['best_of_k'] = {'runs': 10, 'eps': 0.1, 'depth': 3,
                         'mean_total_by_k': {str(k): round(st.mean(sum(curves[ci][r][j] for ci in range(len(seqs))) for r in range(10)), 2)
                                             for j, k in enumerate(K)},
                         'per_chain_mean_by_k': [[round(st.mean(curves[ci][r][j] for r in range(10)), 2) for j in range(len(K))]
                                                 for ci in range(len(seqs))]}
    card['seconds'] = round(time.time() - t0)
    sha = SU.freeze(f'baselines_{a.suite}', card)
    print(f'baselines_{a.suite}: sha256 {sha} ({card["seconds"]}s)')
    for depth in DEPTH_SEEDS:
        h = card['heuristic'][str(depth)]
        print(f'  heuristic depth {depth}: {h["mean_total"]} +- {h["sd_total"]} [p5 {h["p5_total"]}, p95 {h["p95_total"]}] trapped {h["trapped_rate"]:.3f}')
    for kind in ('compact', 'neutral'):
        h = card[f'{kind}_depth3']
        print(f'  {kind} depth 3: {h["mean_total"]} +- {h["sd_total"]}')
    print('  best of k:', card['best_of_k']['mean_total_by_k'])


if __name__ == '__main__':
    main()
