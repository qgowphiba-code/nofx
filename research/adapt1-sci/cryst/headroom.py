"""Local headroom check at the nominal batch context: floor, textbook diagnostics, ceiling."""
import sys, json, math, time
sys.path.insert(0, 'cryst')
import sim, opt
from bank import compound, NOMINAL

def cubic_u():
    d = [((k + 1) / sim.H) ** 3 - (k / sim.H) ** 3 for k in range(sim.H)]
    l = [math.log(x) for x in d]; m = (max(l) + min(l)) / 2
    return [max(-1, min(1, (x - m) / sim.BETA)) for x in l]

rows = []
for i in range(int(sys.argv[1]), int(sys.argv[2])):
    comp = compound(i); f = lambda u: sim.simulate(u, comp, NOMINAL)['quality']
    t = time.time()
    lin, cub = f([0.0] * sim.H), f(cubic_u())
    rnd = opt.random_search(f, sim.H, 200, 1)
    rnd_mean = sum(f([__import__('random').Random(k).uniform(-1, 1) for _ in range(sim.H)]) for k in range(50)) / 50
    ceil, xbest = opt.cmaes_best(f, sim.H, 4000, seed=i, restarts=4)
    rows.append({'compound': i, 'random_mean': round(rnd_mean, 3), 'linear': round(lin, 3), 'cubic': round(cub, 3),
                 'random_best200': round(rnd[-1], 3), 'ceiling': round(ceil, 3), 'u_best': [round(v, 2) for v in xbest]})
    print(json.dumps(rows[-1]), '%.0fs' % (time.time() - t), flush=True)
