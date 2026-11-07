import sys, json
sys.path.insert(0, 'dynbench'); sys.path.insert(0, 'cryst')
import problems, opt
PB, BUDGET, SEEDS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); H = 20
N = H * problems.PROBLEMS[PB].get('adim', 1) if PB in problems.PROBLEMS else H
f = lambda u: problems.simulate(PB, u, H)['J']
METHODS = [('random', opt.random_search), ('es11', opt.one_plus_one), ('cmaes', opt.cmaes)] + ([('nes', opt.nes)] if len(sys.argv) > 4 else [])
for name, fn in METHODS:
    for s in range(SEEDS):
        h = fn(f, N, BUDGET, s)
        print(json.dumps({'problem': PB, 'method': name, 'seed': s, 'curve': {n: h[n - 1] for n in (16, 32, 64, 128, 256, 400, 600, 1000, 1500, 5000, 25000, 35000) if n <= BUDGET}}), flush=True)
