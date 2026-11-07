"""Local headroom gate for BeerFMT before any API call.

1. Light control: best constant temperature (fine sweep), nominal context.
2. Nominal-optimal 20-stage profile by CMA-ES (restarts) on the nominal context.
3. Per-context optima on K sampled contexts (SMPL's +-10 % initial yeast and sugar) vs the nominal profile on the same contexts.
Gate (pre-registered in PREREG.md): run the Machina acquisition test only if the best profile beats the best constant by
>= 3 % of the achievable range; run refinement only if the nominal profile loses >= 3 % against per-context optima.
usage: beer_gate.py OUT.json [K] [EVALS]
"""
import sys, json, numpy as np, cma
from multiprocessing import Pool
sys.path.insert(0, __import__('os').path.dirname(__file__))
import beer_plant as B
OUT, K, EVALS = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8, int(sys.argv[3]) if len(sys.argv) > 3 else 4000


def ret_u(u, x0):
    return B.run(B.profile_from_u(u), x0)['return']


def cmaes(x0, seed, evals, restarts=3, sigma=0.5):
    best = (-1e9, None)
    rng = np.random.default_rng(seed)
    for r in range(restarts):
        start = rng.uniform(-0.6, 0.6, B.H) if r else np.zeros(B.H)
        es = cma.CMAEvolutionStrategy(start, sigma, {'seed': seed * 100 + r + 1, 'verbose': -9, 'bounds': [-1, 1], 'maxfevals': evals // restarts})
        while not es.stop():
            X = es.ask(); F = [-ret_u(x, x0) for x in X]; es.tell(X, F)
            i = int(np.argmin(F))
            if -F[i] > best[0]: best = (-F[i], list(map(float, X[i])))
    return best


def per_context(args):
    j, x0 = args
    v, u = cmaes(np.array(x0), 1000 + j, EVALS)
    return j, v, u


if __name__ == '__main__':
    res = {}
    sweep = {f'{T:.2f}': B.run(np.full(200, T))['return'] for T in np.arange(13.0, 15.01, 0.05)}
    bestT = max(sweep, key=sweep.get)
    res['constant_sweep_best'] = {'T': float(bestT), 'return': sweep[bestT]}
    print('best constant T', bestT, sweep[bestT], flush=True)
    v, u = cmaes(B.NOMINAL_X0, 7, EVALS * 2, restarts=4)
    res['nominal_profile'] = {'return': v, 'u': u, 'temps': [float(t) for t in B.profile_from_u(u)[::B.STAGE_HOURS]]}
    print('nominal CMA-ES profile return', v, 'steps', B.run(B.profile_from_u(u))['steps'], flush=True)
    rng = np.random.default_rng(2026)
    ctxs = [sample.tolist() for sample in (B.sample_context(rng) for _ in range(K))]
    with Pool(min(4, K)) as pool:
        opt = pool.map(per_context, list(enumerate(ctxs)))
    rows = []
    for j, vj, uj in opt:
        nom = ret_u(u, np.array(ctxs[j]))
        bestc = max(B.run(np.full(200, T), np.array(ctxs[j]))['return'] for T in np.arange(13.0, 15.01, 0.1))
        rows.append({'j': j, 'ctx': ctxs[j], 'per_context_opt': vj, 'nominal_profile': nom, 'best_constant': bestc, 'u_opt': uj})
        print(f'ctx {j}: opt {vj:.1f} nominal-profile {nom:.1f} best-constant {bestc:.1f}', flush=True)
    res['contexts'] = rows
    res['summary'] = {'mean_per_context_opt': float(np.mean([r['per_context_opt'] for r in rows])),
                      'mean_nominal_profile': float(np.mean([r['nominal_profile'] for r in rows])),
                      'mean_best_constant': float(np.mean([r['best_constant'] for r in rows])), 'range_for_pct': 600.0}
    print(json.dumps(res['summary']))
    json.dump(res, open(OUT, 'w'), indent=1)
