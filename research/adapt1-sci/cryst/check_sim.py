import sys, time, math
sys.path.insert(0, 'cryst')
import sim
from bank import compound, NOMINAL
H = sim.H
def cubic_u(T0):
    d = [((k + 1) / H) ** 3 - (k / H) ** 3 for k in range(H)]
    l = [math.log(x) for x in d]; m = (max(l) + min(l)) / 2
    return [max(-1, min(1, (x - m) / sim.BETA)) for x in l]
for i in range(6):
    comp = compound(i)
    t = time.time()
    lin = sim.simulate([0.0] * H, comp, NOMINAL)
    cub = sim.simulate(cubic_u(50), comp, NOMINAL)
    dt = time.time() - t
    print(i, {k: round(v, 3) for k, v in comp.items() if k != 'id'})
    for name, o in (('linear', lin), ('cubic', cub)):
        print('   %-7s q=%.3f fines=%.3f left=%.3f size=%.0f smax=%.3f' % (name, o['quality'], o['fines_frac'], o['left_dissolved'], o['final_size'], o['max_supersat']))
    print('   2 sims %.3fs' % dt)
# step-size convergence
comp = compound(0)
for dtv in (0.2, 0.1, 0.05, 0.025):
    sim.DT = dtv
    print('DT', dtv, round(sim.simulate([0.0] * H, comp, NOMINAL)['quality'], 5), round(sim.simulate(cubic_u(50), comp, NOMINAL)['quality'], 5))
