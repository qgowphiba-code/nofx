import sys, json
sys.path.insert(0, 'dynbench'); sys.path.insert(0, 'cryst')
import problems, opt
PB, BUDGET, SEEDS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); H = 20
f = lambda u: problems.simulate(PB, u, H)['J']
for name, fn in (('random', opt.random_search), ('es11', opt.one_plus_one), ('cmaes', opt.cmaes)):
    for s in range(SEEDS):
        h = fn(f, H, BUDGET, s)
        print(json.dumps({'problem': PB, 'method': name, 'seed': s, 'curve': {n: h[n - 1] for n in (16, 32, 64, 128, 256, 400, 600) if n <= BUDGET}}), flush=True)
