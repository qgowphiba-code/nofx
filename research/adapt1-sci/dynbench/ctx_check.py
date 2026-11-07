"""Refinement headroom on Park-Ramirez: does the best feed profile depend on the measured feed concentration?
The published problem fixes the feed substrate concentration at 20 (the '20' in x4' = ... + u (20 - x4)/x5).
Here it varies per batch in [12, 28] and is measured before the batch (context). Everything else is the published model."""
import sys, json, math, random
sys.path.insert(0, 'dynbench'); sys.path.insert(0, 'cryst')
import problems, opt
H = 20

def make_f(sf):
    def f(x, u):
        x1, x2, x3, x4, x5 = x
        g3 = 21.87 * x4 / ((x4 + 0.4) * (x4 + 62.5)); g2 = x4 * math.exp(-5 * x4) / (0.1 + x4); g1 = 4.75 * g3 / (0.12 + g3)
        return [g1 * (x2 - x1) - u * x1 / x5, g2 * x3 - u * x2 / x5, g3 * x3 - u * x3 / x5, -7.3 * g3 * x3 + u * (sf - x4) / x5, u]
    return f

def J(u, sf):
    x = [0.0, 0.0, 1.0, 5.0, 1.0]; h = 15.0 / H / 60
    for k in range(H):
        c = 2.0 * (max(-1, min(1, u[k])) + 1) / 2
        x = problems._rk4(make_f(sf), x, c, h, 60)
    return x[0] * x[4]

v20, x20 = opt.cmaes_best(lambda u: J(u, 20.0), H, 6000, seed=1, restarts=3)
rows = []
for sf in (12.0, 15.0, 18.0, 22.0, 25.0, 28.0):
    vc, xc = opt.cmaes_best(lambda u: J(u, sf), H, 6000, seed=2, restarts=3)
    rows.append({'feed_conc': sf, 'nominal_profile_here': round(J(x20, sf), 3), 'own_optimum': round(vc, 3),
                 'loss_pct': round(100 * (vc - J(x20, sf)) / vc, 2)})
    print(json.dumps(rows[-1]), flush=True)
print(json.dumps({'nominal_opt_at_20': round(v20, 3), 'mean_loss_pct': round(sum(r['loss_pct'] for r in rows) / len(rows), 2)}))
