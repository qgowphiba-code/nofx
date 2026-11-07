"""General contextual learners on a ctxtasks task: linear context->offset policy trained by CMA-ES or NES, minibatch 32.
usage: scale2.py TASK METHOD SEED MAX_EXEC [MODE]
MODE plain (default): held-out mean J at 7680, 15360, 30720, 61440 executions.
MODE drift: train 5120 executions on the original plant, then the plant drifts (pr2drift); report held-out right after the
change and after 1280, 2560, 5120 further executions (continued), plus a from-scratch restart on the drifted plant."""
import sys, json, random, statistics as st
import numpy as np
sys.path.insert(0, 'dynbench'); sys.path.insert(0, 'cryst')
import ctxtasks, opt
TASK, METHOD, SEED, MAX_EXEC = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); MODE = sys.argv[5] if len(sys.argv) > 5 else 'plain'
T = ctxtasks.Task(TASK); MB = 32; d = len(T.vec(T.HELD[0])); A = T.adim; NP = ctxtasks.H * A * (d + 1)
def policy(theta, c):
    v = np.array([1.0] + T.vec(c)); th = np.asarray(theta).reshape(ctxtasks.H, A, d + 1)
    if A == 1:
        return [max(-1.0, min(1.0, b + float(th[k, 0] @ v))) for k, b in enumerate(T.base)]
    return [[max(-1.0, min(1.0, bb + float(th[k, j] @ v))) for j, bb in enumerate(br)] for k, br in enumerate(T.base)]
def heldout(theta, drift):
    return st.mean(T.J(policy(theta, c), c, drift) for c in T.HELD)

def train(theta0, budget, drift, checks, label, rseed):
    r = random.Random(rseed); state = {'execs': 0, 'best': (-1, np.array(theta0, dtype=float)), 'i': 0}
    def f(theta):
        batch = [T.TRAIN[r.randrange(len(T.TRAIN))] for _ in range(MB)]
        v = st.mean(T.J(policy(theta, c), c, drift) for c in batch) / T.norm
        state['execs'] += MB
        if v > state['best'][0]: state['best'] = (v, np.array(theta, dtype=float))
        while state['i'] < len(checks) and state['execs'] >= checks[state['i']]:
            print(json.dumps({'task': TASK, 'method': METHOD, 'seed': SEED, 'phase': label, 'executions': checks[state['i']],
                              'heldout_mean_J': heldout(state['best'][1], drift)}), flush=True)
            state['i'] += 1
        return v
    if METHOD == 'cmaes':
        opt.cmaes(f, NP, budget // MB, rseed, sigma0=0.1, x0=list(theta0))
    else:
        rng = np.random.default_rng(rseed); th = np.array(theta0, dtype=float); sigma, lr, pop = 0.05, 0.02, 16
        while state['execs'] < budget - 2 * pop * MB:
            eps = rng.standard_normal((pop, NP)); rets = []
            for e in eps: rets += [f(th + sigma * e), f(th - sigma * e)]
            rk = np.argsort(np.argsort(rets)).astype(float); rk = rk / (len(rk) - 1) - 0.5
            th = th + lr * sum((rk[2 * i] - rk[2 * i + 1]) * eps[i] for i in range(pop)) / (2 * pop * sigma)
            f(th)
    return state['best'][1]

if MODE == 'plain':
    train(np.zeros(NP), MAX_EXEC, False, [c for c in (7680, 15360, 30720, 61440) if c <= MAX_EXEC], 'plain', SEED)
else:
    th = train(np.zeros(NP), 5120, False, [5120], 'before_change', SEED)
    print(json.dumps({'task': TASK, 'method': METHOD, 'seed': SEED, 'phase': 'right_after_change', 'executions': 0, 'heldout_mean_J': heldout(th, True)}), flush=True)
    train(th, 5120, True, [1280, 2560, 5120], 'continued_after_change', SEED + 100)
    train(np.zeros(NP), 5120, True, [1280, 2560, 5120], 'restart_after_change', SEED + 200)
