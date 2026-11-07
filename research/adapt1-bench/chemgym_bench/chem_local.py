"""Local side of the ChemGymRL reaction-bench test (executor CHEM-1, per target).
  gate       per-target optimum by CMA-ES (4,000 evaluations, 3 restarts) vs the heuristic and random: the headroom check
  random:S / es11:S / cmaes:S   same-budget comparators at N attempts for every target, starting from u = 0 (mid-range)
  heuristic  the benchmark's own policy per target
usage: chem_local.py BENCH WHAT N OUT.jsonl      (BENCH wurtz | fict)
"""
import sys, json, statistics as st
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'common'))
import numpy as np
import chem_plant as P
import opt

BENCH, WHAT, N, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
ADIM = P.BENCHES[BENCH]['adim']; TARGETS = P.BENCHES[BENCH]['targets']
D = P.H * ADIM


def emit(rec):
    with open(OUT, 'a') as fh:
        fh.write(json.dumps(rec) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k not in ('best_curve', 'u')}), flush=True)


def gate_target(target):
    import cma
    env = P.make_env(BENCH); best, best_u = -1e9, None
    rng = np.random.default_rng(abs(hash(target)) % 10000)
    for r in range(3):
        es = cma.CMAEvolutionStrategy(np.zeros(D) if r == 0 else rng.uniform(-0.6, 0.6, D), 0.5,
                                      {'seed': 1 + r, 'verbose': -9, 'bounds': [-1, 1], 'maxfevals': 4000 // 3})
        while not es.stop():
            X = es.ask(); F = [-P.run(np.array(x).reshape(P.H, ADIM), target, env, bench=BENCH)['return'] for x in X]; es.tell(X, F)
            i = int(np.argmin(F))
            if -F[i] > best:
                best, best_u = -F[i], list(map(float, X[i]))
    h = P.run(P.heuristic_u(BENCH, target), target, env, bench=BENCH)['return']
    rnd = [P.run(rng.uniform(-1, 1, (P.H, ADIM)), target, env, bench=BENCH)['return'] for _ in range(50)]
    return {'what': 'gate', 'bench': BENCH, 'target': target, 'cmaes_4000_best': best, 'heuristic': h, 'random_mean': st.mean(rnd), 'random_best_of_50': max(rnd), 'u': best_u}


def comparator(args):
    method, seed, target = args
    env = P.make_env(BENCH); hist = []

    def f(x):
        q = P.run(np.array(x).reshape(P.H, ADIM), target, env, bench=BENCH)['quality']; hist.append(q); return q
    if method == 'random':
        opt.random_search(f, D, N, seed)
    elif method == 'es11':
        opt.one_plus_one(f, D, N, seed, sigma=0.3, x0=[0.0] * D)
    elif method == 'cmaes':
        opt.cmaes(f, D, N, seed, sigma0=0.5, x0=[0.0] * D)
    hist = hist[:N]; best = [max(hist[:i + 1]) for i in range(len(hist))]
    return {'what': 'comparator', 'bench': BENCH, 'method': method, 'seed': seed, 'target': target, 'attempts': len(hist), 'best_curve': [round(b, 4) for b in best], 'best_q': best[-1]}


if __name__ == '__main__':
    if WHAT == 'gate':
        with Pool(3) as pool:
            for rec in pool.imap_unordered(gate_target, TARGETS):
                emit(rec)
    elif WHAT == 'heuristic':
        env = P.make_env(BENCH)
        for t in TARGETS:
            r = P.run(P.heuristic_u(BENCH, t), t, env, bench=BENCH); emit({'what': 'heuristic', 'bench': BENCH, 'target': t, 'return': r['return'], 'quality': r['quality']})
    else:
        method, seed = WHAT.split(':'); seed = int(seed)
        with Pool(3) as pool:
            for rec in pool.imap_unordered(comparator, [(method, seed, t) for t in TARGETS]):
                emit(rec)
