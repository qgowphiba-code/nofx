import sys, json, statistics as st
sys.path.insert(0, 'dynbench'); sys.path.insert(0, 'cryst')
import ctxtasks, opt
T = ctxtasks.Task(sys.argv[1]); lo, hi = map(int, sys.argv[2].split('-'))
for j in range(lo, min(hi, len(T.HELD))):
    c = T.HELD[j]
    v, x = opt.cmaes_best(lambda u: T.J(u, c), ctxtasks.H * T.adim, 1200 if T.name == 'pen2' else 2400, seed=j, restarts=2)
    print(json.dumps({'task': T.name, 'j': j, 'ctx': c, 'ceiling': v, 'base': T.J(T.base, c)}), flush=True)
