"""Local side of the PenSim refinement test (executor PEN-2 around the acquired base): base alone and the default recipe on
the HELD batches; the C2 gate (top-8 acquired recipes on 16 gate contexts: does the best recipe depend on the batch?);
same-budget learners trained on TRAIN with 1,920 executions: cmaes:SEED (linear context -> offset policy, minibatch 12)
and robust:SEED (context-free offset). Simulations run on 3 worker processes.
usage: pensim_local_refine.py WHAT BASE_JSON OUT.jsonl   with WHAT in base | default | gate:ACQ_DIR | cmaes:S | robust:S
"""
import sys, json, random, statistics as st
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'common'))
import numpy as np
import pensim_plant as P
import pensim_contexts as C
import opt

WHAT, BASE_JSON, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
base = np.array(json.load(open(BASE_JSON))['u'], dtype=float)
BUDGET_EXEC, MB, WORKERS = 20 * 8 * 12, 12, 3


def sim(args):
    u, seed_ref = args
    r = P.run(np.clip(np.array(u, dtype=float), -1, 1), seed_ref)
    return r['quality'], r['yield'], r['steps']


def emit(rec):
    with open(OUT, 'a') as fh:
        fh.write(json.dumps(rec) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k not in ('per_context', 'theta', 'table')}))


if __name__ == '__main__':
    pool = Pool(WORKERS)
    if WHAT == 'base':
        res = pool.map(sim, [(base, s) for s in C.HELD])
        emit({'what': 'base_on_heldout', 'mean_yield': st.mean(r[1] for r in res), 'mean_quality': st.mean(r[0] for r in res),
              'completed_frac': sum(r[2] >= 1150 for r in res) / len(res), 'per_context': [r[1] for r in res]})
    elif WHAT == 'default':
        res = pool.map(sim, [(np.zeros((P.H, P.ADIM)), s) for s in C.HELD])
        emit({'what': 'default_recipe_on_heldout', 'mean_yield': st.mean(r[1] for r in res), 'mean_quality': st.mean(r[0] for r in res),
              'completed_frac': sum(r[2] >= 1150 for r in res) / len(res), 'per_context': [r[1] for r in res]})
    elif WHAT.startswith('gate:'):
        acq_dir = WHAT.split(':', 1)[1]
        rows = sorted((json.loads(l) for l in open(Path(acq_dir) / 'attempts.jsonl')), key=lambda r: -r['q'])
        top, seen = [], set()
        for r in rows:
            key = json.dumps(r['u'])
            if key not in seen and r['steps'] >= 1150:
                seen.add(key); top.append(r['u'])
            if len(top) == 8:
                break
        res = pool.map(sim, [(u, s) for u in top for s in C.GATE])
        table = np.array([r[1] for r in res]).reshape(len(top), len(C.GATE))
        winners = table.argmax(axis=0)
        frac_not_nominal = float(np.mean(winners != 0))
        emit({'what': 'c2_gate', 'recipes': len(top), 'contexts': len(C.GATE), 'winner_per_context': winners.tolist(),
              'frac_contexts_where_best_recipe_is_not_the_top_acquired': frac_not_nominal, 'passes': frac_not_nominal >= 0.25,
              'table': table.round(1).tolist(), 'spread_of_top_recipe_over_contexts': [float(table[0].min()), float(table[0].max())]})
    else:
        method, seed = WHAT.split(':'); seed = int(seed)
        r = random.Random(seed)
        D = P.H * P.ADIM
        if method == 'robust':
            n = D

            def policy(theta, s):
                return np.clip(base + np.array(theta).reshape(P.H, P.ADIM), -1, 1)
        else:
            n = D * 9       # offset = theta0 + theta . vec(c) (8 context features), per command coordinate

            def policy(theta, s):
                v = np.array(C.vec(s)); th = np.array(theta).reshape(P.H, P.ADIM, 9)
                return np.clip(base + th[:, :, 0] + np.tensordot(th[:, :, 1:], v, axes=([2], [0])), -1, 1)

        def f(theta):
            batch = [C.TRAIN[r.randrange(len(C.TRAIN))] for _ in range(MB)]
            res = pool.map(sim, [(policy(theta, s), s) for s in batch])
            return st.mean(x[0] for x in res)
        best = {'v': -1, 'x': [0.0] * n}

        def g(theta):
            v = f(theta)
            if v > best['v']:
                best['v'], best['x'] = v, list(theta)
            print(f'{WHAT} eval {len(hist) + 1} q={v:.3f} best={best["v"]:.3f}', flush=True); hist.append(v)
            return v
        hist = []
        n_eval = BUDGET_EXEC // MB
        opt.cmaes(g, n, n_eval, seed, sigma0=0.05, x0=[0.0] * n)
        res = pool.map(sim, [(policy(best['x'], s), s) for s in C.HELD])
        emit({'what': 'contextual_cmaes' if method == 'cmaes' else 'robust_recipe', 'seed': seed, 'executions': n_eval * MB,
              'mean_yield': st.mean(x[1] for x in res), 'mean_quality': st.mean(x[0] for x in res), 'completed_frac': sum(x[2] >= 1150 for x in res) / len(res),
              'per_context': [x[1] for x in res], 'theta': best['x']})
    pool.close()
