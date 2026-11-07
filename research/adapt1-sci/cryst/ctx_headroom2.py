"""Exploratory (post-hoc) context: measured growth-inhibiting impurity. Does the optimal profile now depend on context?"""
import sys, json, random
sys.path.insert(0, 'cryst')
import sim, opt
from bank import compound, NOMINAL
ci = int(sys.argv[1]); base = compound(ci)
def comp_at(imp):
    c = dict(base); c['kg'] = base['kg'] * (1 - 0.75 * imp); c['kb'] = base['kb'] * (1 + 3.0 * imp); return c
r = random.Random(4242)
ctxs = [r.uniform(0.0, 1.0) for _ in range(10)]
f0 = lambda u: sim.simulate(u, comp_at(0.3), NOMINAL)['quality']
v0, x0 = opt.cmaes_best(f0, sim.H, 3000, seed=ci, restarts=3)
gaps = []
for j, imp in enumerate(ctxs):
    f = lambda u: sim.simulate(u, comp_at(imp), NOMINAL)['quality']
    vc, xc = opt.cmaes_best(f, sim.H, 2000, seed=500 + j, restarts=2)
    gaps.append(vc - f(x0)); print(json.dumps({'impurity': round(imp, 2), 'nominal_opt_here': round(f(x0), 3), 'ctx_opt': round(vc, 3)}), flush=True)
print(json.dumps({'compound': ci, 'mean_gap': round(sum(gaps) / len(gaps), 4), 'max_gap': round(max(gaps), 4)}))
