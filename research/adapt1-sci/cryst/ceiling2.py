"""Best-known reference per compound: long CMA-ES restarts, plus restarts seeded from every method's best profile."""
import sys, json, glob
sys.path.insert(0, 'cryst')
import sim, opt
from bank import compound, NOMINAL
c = int(sys.argv[1]); comp = compound(c); f = lambda u: sim.simulate(u, comp, NOMINAL)['quality']
v, x = opt.cmaes_best(f, sim.H, 20000, seed=77 + c, restarts=4, sigma0=0.5)
seeds = []
for p in glob.glob(f'runs/cryst/machina-c{c}-*/summary.json'):
    seeds.append(json.load(open(p))['best_u'])
for s in seeds:
    holder = {'v': -1, 'x': None}
    def g(u):
        q = f(u)
        if q > holder['v']: holder['v'], holder['x'] = q, u
        return q
    opt.cmaes(g, sim.H, 3000, seed=1, sigma0=0.1, x0=s)
    if holder['v'] > v: v, x = holder['v'], holder['x']
print(json.dumps({'compound': c, 'best_known': v, 'u': x, 'seeded_from': len(seeds)}))
