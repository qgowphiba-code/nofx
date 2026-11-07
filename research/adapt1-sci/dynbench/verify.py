"""Verify formulations: does a strong optimizer on the H-stage discretization reproduce the published optimum?"""
import sys, json, time
sys.path.insert(0, 'dynbench'); sys.path.insert(0, 'cryst')
import problems, opt
name = sys.argv[1]; H = 20
f = lambda u: problems.simulate(name, u, H)['J']
t = time.time()
v, x = opt.cmaes_best(f, H, 12000, seed=3, restarts=4, sigma0=0.5)
P = problems.PROBLEMS[name]
ctrl = [round(P['lo'] + (P['hi'] - P['lo']) * (max(-1, min(1, a)) + 1) / 2, 3) for a in x]
print(json.dumps({'problem': name, 'H': H, 'best_J': round(v, 6), 'published': P['J_lit'], 'ratio': round(v / P['J_lit'], 4),
                  'random_J_mean': round(sum(f([__import__('random').Random(k * 7 + i).uniform(-1, 1) for i in range(H)]) for k in range(50)) / 50, 5),
                  'control': ctrl, 'seconds': round(time.time() - t)}))
