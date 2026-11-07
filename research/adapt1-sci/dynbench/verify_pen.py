import sys, json, time, random
sys.path.insert(0, 'dynbench'); sys.path.insert(0, 'cryst')
import penicillin as P, opt
t = time.time(); f = lambda u: P.J(u)
v, x = opt.cmaes_best(f, P.H, 6000, seed=4, restarts=3, sigma0=0.5)
print(json.dumps({'best_J': v, 'published': '87.7-88.0', 'feed': [round(25 * (a + 1), 2) for a in x], 'random_mean': sum(f([random.Random(k * 31 + i).uniform(-1, 1) for i in range(P.H)]) for k in range(20)) / 20,
                  'sec': round(time.time() - t), 'stepcheck': [P.J(x, sub=100), P.J(x, sub=400)]}))
