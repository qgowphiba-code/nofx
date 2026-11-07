"""Local side of the refinement test: base alone, per-context ceiling, and same-budget contextual learners."""
import sys, json, statistics as st
sys.path.insert(0, 'dynbench'); sys.path.insert(0, 'cryst')
import numpy as np, random
import pr_ctx as P, opt
what = sys.argv[1]
base, bJ, _ = P.machina_base()
if what == 'base':
    vals = [P.J(base, c['sf'], c['x40']) for c in P.HELD]
    print(json.dumps({'what': 'base_on_heldout', 'mean_J': st.mean(vals), 'per_context': vals, 'base_nominal_J': bJ}))
elif what.startswith('ceiling'):
    lo, hi = map(int, what.split(':')[1].split('-'))
    for j in range(lo, hi):
        c = P.HELD[j]
        v, x = opt.cmaes_best(lambda u: P.J(u, c['sf'], c['x40']), P.H, 2400, seed=j, restarts=2)
        v2 = P.J(base, c['sf'], c['x40'])
        print(json.dumps({'what': 'ceiling', 'j': j, 'ctx': c, 'J': v, 'base_J': v2}), flush=True)
else:   # contextual linear policy offsets = theta0 + theta1*c1 + theta2*c2, trained on TRAIN with minibatches, same executions
    method, seed = what.split(':'); seed = int(seed)
    BUDGET_EXEC, MB = 30 * 8 * 32, 16
    r = random.Random(seed)
    def policy(theta, c):
        v = P.vec(c); th = np.array(theta).reshape(P.H, 3)
        return [max(-1.0, min(1.0, b + th[k, 0] + th[k, 1] * v[0] + th[k, 2] * v[1])) for k, b in enumerate(base)]
    def f(theta):
        batch = [P.TRAIN[r.randrange(len(P.TRAIN))] for _ in range(MB)]
        return st.mean(P.J(policy(theta, c), c['sf'], c['x40']) for c in batch) / P.J_PUB
    best = {'v': -1, 'x': [0.0] * (P.H * 3)}
    def g(theta):
        v = f(theta)
        if v > best['v']: best['v'], best['x'] = v, list(theta)
        return v
    n_eval = BUDGET_EXEC // MB
    if method == 'cmaes':
        opt.cmaes(g, P.H * 3, n_eval, seed, sigma0=0.1, x0=[0.0] * (P.H * 3))
    else:
        opt.one_plus_one(g, P.H * 3, n_eval, seed, sigma=0.1, x0=[0.0] * (P.H * 3))
    held = [P.J(policy(best['x'], c), c['sf'], c['x40']) for c in P.HELD]
    print(json.dumps({'what': 'contextual_' + method, 'seed': seed, 'executions': n_eval * MB, 'heldout_mean_J': st.mean(held), 'per_context': held}))
