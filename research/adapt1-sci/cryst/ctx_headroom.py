"""Refinement headroom: how much does the nominal-optimal profile lose on other batch contexts?"""
import sys, json, math, random
sys.path.insert(0, 'cryst')
import sim, opt
from bank import compound, NOMINAL, contexts
ci = int(sys.argv[1])
comp = compound(ci)
fn = lambda u: sim.simulate(u, comp, NOMINAL)['quality']
v0, x0 = opt.cmaes_best(fn, sim.H, 4000, seed=ci, restarts=4)
rows = []
for j, c in enumerate(contexts(12, 555)):
    f = lambda u: sim.simulate(u, comp, c)['quality']
    vc, xc = opt.cmaes_best(f, sim.H, 2000, seed=1000 + j, restarts=2)
    rows.append({'ctx': {k: round(v, 3) for k, v in c.items()}, 'nominal_opt_here': round(f(x0), 3), 'linear': round(f([0.0] * sim.H), 3), 'ctx_opt': round(vc, 3)})
    print(json.dumps(rows[-1]), flush=True)
gap = sum(r['ctx_opt'] - r['nominal_opt_here'] for r in rows) / len(rows)
print(json.dumps({'compound': ci, 'nominal_opt': round(v0, 3), 'mean_gap_nominal_vs_ctx_opt': round(gap, 4),
                  'mean_nominal_here': round(sum(r['nominal_opt_here'] for r in rows) / len(rows), 3),
                  'mean_ctx_opt': round(sum(r['ctx_opt'] for r in rows) / len(rows), 3)}))
