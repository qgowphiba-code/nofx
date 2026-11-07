"""Local side of the BeerFMT refinement test: base alone, per-context ceiling, best constant per context, and same-budget
learners trained on the TRAIN contexts with the same 7,680 executions:
  cmaes:SEED / nes:SEED   linear context -> offset policy (theta: H x 4; offsets = theta0 + theta . vec(c)), minibatches of 16
  robust:SEED             context-free profile optimized by CMA-ES on the mean quality over contexts (same executions)
usage: beer_local_refine.py WHAT BASE_JSON OUT.jsonl   with WHAT in base | ceiling | constant | cmaes:S | nes:S | robust:S
"""
import sys, json, random, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'common'))
import numpy as np
import beer_plant as B
import beer_contexts as C
import opt

WHAT, BASE_JSON, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
base = [float(v) for v in json.load(open(BASE_JSON))['u']]
BUDGET_EXEC, MB = 30 * 8 * 32, 16


def ret(u, x0):
    return B.run(B.profile_from_u(u), x0)['return']


def emit(rec):
    with open(OUT, 'a') as fh:
        fh.write(json.dumps(rec) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k not in ('per_context', 'theta', 'u')}))


if WHAT == 'base':
    vals = [ret(base, x0) for x0 in C.HELD]
    emit({'what': 'base_on_heldout', 'mean_return': st.mean(vals), 'mean_quality': st.mean(B.quality(v) for v in vals),
          'finished_frac': sum(v > 0 for v in vals) / len(vals), 'per_context': vals})
elif WHAT == 'ceiling':
    import cma
    vals = []
    for j, x0 in enumerate(C.HELD):
        best = -1e9
        for r in range(3):
            es = cma.CMAEvolutionStrategy(np.zeros(B.H) if r == 0 else np.random.default_rng(100 * j + r).uniform(-0.6, 0.6, B.H), 0.5,
                                          {'seed': 1 + 100 * j + r, 'verbose': -9, 'bounds': [-1, 1], 'maxfevals': 4000 // 3})
            while not es.stop():
                X = es.ask(); F = [-ret(x, x0) for x in X]; es.tell(X, F); best = max(best, -min(F))
        vals.append(best); print(f'ceiling ctx {j}: {best}', flush=True)
    emit({'what': 'ceiling_on_heldout', 'mean_return': st.mean(vals), 'mean_quality': st.mean(B.quality(v) for v in vals), 'per_context': vals})
elif WHAT == 'constant':
    vals = [max(B.run(np.full(200, T), x0)['return'] for T in np.arange(13.0, 15.01, 0.1)) for x0 in C.HELD]
    emit({'what': 'best_constant_on_heldout', 'mean_return': st.mean(vals), 'mean_quality': st.mean(B.quality(v) for v in vals), 'per_context': vals})
else:
    method, seed = WHAT.split(':'); seed = int(seed)
    r = random.Random(seed)
    if method == 'robust':
        n = B.H

        def policy(theta, x0):
            return [max(-1.0, min(1.0, b + t)) for b, t in zip(base, theta)]
    else:
        n = B.H * 4

        def policy(theta, x0):
            v = C.vec(x0); th = np.array(theta).reshape(B.H, 4)
            return [max(-1.0, min(1.0, b + th[k, 0] + th[k, 1] * v[0] + th[k, 2] * v[1] + th[k, 3] * v[2])) for k, b in enumerate(base)]

    def f(theta):
        batch = [C.TRAIN[r.randrange(len(C.TRAIN))] for _ in range(MB)]
        return st.mean(B.quality(ret(policy(theta, x0), x0)) for x0 in batch)
    best = {'v': -1, 'x': [0.0] * n}

    def g(theta):
        v = f(theta)
        if v > best['v']:
            best['v'], best['x'] = v, list(theta)
        return v
    n_eval = BUDGET_EXEC // MB
    if method in ('cmaes', 'robust'):
        opt.cmaes(g, n, n_eval, seed, sigma0=0.1, x0=[0.0] * n)
    elif method == 'nes':
        opt.nes(g, n, n_eval, seed, sigma=0.1, lr=0.05, pop=16, x0=[0.0] * n)
    else:
        raise SystemExit(method)
    held = [ret(policy(best['x'], x0), x0) for x0 in C.HELD]
    emit({'what': f'contextual_{method}' if method != 'robust' else 'robust_profile', 'seed': seed, 'executions': n_eval * MB,
          'mean_return': st.mean(held), 'mean_quality': st.mean(B.quality(v) for v in held), 'finished_frac': sum(v > 0 for v in held) / len(held),
          'per_context': held, 'theta': best['x']})
