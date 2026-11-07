"""Same-budget local search methods on the same compound/context (no service)."""
import sys, json
sys.path.insert(0, 'cryst')
import sim, opt
from bank import compound, NOMINAL
COMPS = [int(x) for x in sys.argv[1].split(',')]; BUDGET = int(sys.argv[2]); SEEDS = int(sys.argv[3])
for c in COMPS:
    comp = compound(c); f = lambda u: sim.simulate(u, comp, NOMINAL)['quality']
    for name, fn in (('random', opt.random_search), ('es11', opt.one_plus_one), ('cmaes', opt.cmaes)):
        for s in range(SEEDS):
            h = fn(f, sim.H, BUDGET, s)
            print(json.dumps({'compound': c, 'method': name, 'seed': s, 'curve': {n: round(h[n - 1], 5) for n in (16, 32, 64, 128, 256, 400, 600, 800) if n <= BUDGET}}), flush=True)
