"""Data multiplier: how many batch executions does a general contextual learner need to match Machina refinement?
Linear context->offset policy (20 x 3 parameters) on Park-Ramirez with measured batch variation; minibatches of 32
training batches per evaluation (the same per-update batch Machina used). Methods: CMA-ES; NES/OpenAI-ES with antithetic
sampling and rank-normalized fitness (Salimans et al. 2017). Held-out mean J recorded at checkpoints."""
import sys, json, math, random, statistics as st
import numpy as np
sys.path.insert(0, 'dynbench'); sys.path.insert(0, 'cryst')
import pr_ctx as P, opt
METHOD, SEED, MAX_EXEC = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); MB = 32
CHECK = [7680, 15360, 30720, 61440]
base, _, _ = P.machina_base(); r = random.Random(SEED)
def policy(theta, c):
    v = P.vec(c); th = np.asarray(theta).reshape(P.H, 3)
    return [max(-1.0, min(1.0, b + th[k, 0] + th[k, 1] * v[0] + th[k, 2] * v[1])) for k, b in enumerate(base)]
def heldout(theta):
    return st.mean(P.J(policy(theta, c), c['sf'], c['x40']) for c in P.HELD)
state = {'execs': 0, 'best': (-1, np.zeros(P.H * 3)), 'next': 0}
def f(theta):
    batch = [P.TRAIN[r.randrange(len(P.TRAIN))] for _ in range(MB)]
    v = st.mean(P.J(policy(theta, c), c['sf'], c['x40']) for c in batch) / P.J_PUB
    state['execs'] += MB
    if v > state['best'][0]: state['best'] = (v, np.array(theta, dtype=float))
    while state['next'] < len(CHECK) and state['execs'] >= CHECK[state['next']]:
        print(json.dumps({'method': METHOD, 'seed': SEED, 'executions': CHECK[state['next']], 'heldout_mean_J': heldout(state['best'][1])}), flush=True)
        state['next'] += 1
    return v
n_eval = MAX_EXEC // MB
if METHOD == 'cmaes':
    opt.cmaes(f, P.H * 3, n_eval, SEED, sigma0=0.1, x0=[0.0] * (P.H * 3))
else:   # NES / OpenAI-ES: 2*16 antithetic perturbations per step, Adam-free plain SGD on rank-normalized returns
    rng = np.random.default_rng(SEED); th = np.zeros(P.H * 3); sigma, lr, pop = 0.05, 0.02, 16
    while state['execs'] < MAX_EXEC:
        eps = rng.standard_normal((pop, th.size)); rets = []
        for e in eps:
            rets += [f(th + sigma * e), f(th - sigma * e)]
            if state['execs'] >= MAX_EXEC: break
        if len(rets) < 2 * pop: break
        rk = np.argsort(np.argsort(rets)).astype(float); rk = rk / (len(rk) - 1) - 0.5
        g = sum((rk[2 * i] - rk[2 * i + 1]) * eps[i] for i in range(pop)) / (2 * pop * sigma)
        th = th + lr * g
        if f(th) and False: pass
