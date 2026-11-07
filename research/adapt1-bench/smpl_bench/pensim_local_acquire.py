"""Same-budget black-box comparators for PenSim acquisition at the nominal batch (seed_ref 1000): random search, (1+1)-ES,
CMA-ES over the same u in [-1, 1]^(23 x 6) (executor PEN-1), starting from u = 0 like the beer comparators; plus the default
recipe scored on seeds 1-5 (light control). Records the best-so-far quality after every attempt.
usage: pensim_local_acquire.py METHOD SEED ATTEMPTS OUT.jsonl      (METHOD random | es11 | cmaes | default)
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'common'))
import numpy as np
import pensim_plant as P
import opt

METHOD, SEED, N, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
SEED_REF = 1000
D = P.H * P.ADIM
hist = []


def f(uflat):
    u = np.asarray(uflat, dtype=float).reshape(P.H, P.ADIM)
    r = P.run(u, SEED_REF)
    hist.append(r['quality'])
    if len(hist) % 25 == 0:
        print(f'{METHOD} s{SEED} n={len(hist)} best={max(hist):.3f}', flush=True)
    return r['quality']


if METHOD == 'default':
    e = P.make_env(); u0 = P.default_recipe_u(e)
    rows = [P.run(u0, s) for s in range(1, 6)]
    with open(OUT, 'a') as fh:
        fh.write(json.dumps({'method': 'default_recipe', 'seeds': [1, 2, 3, 4, 5], 'yields': [r['yield'] for r in rows],
                             'qualities': [r['quality'] for r in rows], 'nominal_seed_ref_yield': P.run(u0, SEED_REF)['yield']}) + '\n')
    print('default recipe yields', [round(r['yield']) for r in rows])
    raise SystemExit
# Start point (PREREG amendment 3): seeds < 100 start from u = 0, which under PEN-2 is the default recipe (the light control
# handed to the comparator); seeds >= 100 start from a uniform random u in [-1, 1]^D, i.e. with no recipe knowledge, like
# Machina's first proposals. Both variants are reported.
x0 = [0.0] * D if SEED < 100 else [float(v) for v in np.random.default_rng(SEED).uniform(-1, 1, D)]
START = 'default_recipe' if SEED < 100 else 'random_init'
if METHOD == 'random':
    opt.random_search(f, D, N, SEED)
elif METHOD == 'es11':
    opt.one_plus_one(f, D, N, SEED, sigma=0.3, x0=x0)
elif METHOD == 'cmaes':
    opt.cmaes(f, D, N, SEED, sigma0=0.5, x0=x0)
else:
    raise SystemExit(METHOD)
hist = hist[:N]
best = [max(hist[:i + 1]) for i in range(len(hist))]
with open(OUT, 'a') as fh:
    fh.write(json.dumps({'method': METHOD, 'seed': SEED, 'start': START, 'attempts': len(hist), 'best_curve': [round(b, 4) for b in best],
                         'best_q': best[-1], 'best_yield': best[-1] * (P.Y_SCALE + 100) - 100}) + '\n')
print(METHOD, SEED, START, 'best q', round(best[-1], 4), 'yield', round(best[-1] * (P.Y_SCALE + 100) - 100))
