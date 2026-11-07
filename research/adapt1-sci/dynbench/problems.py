"""Published open-loop dynamic-optimization benchmarks (piecewise-constant control over H equal stages; stdlib).

batch    Consecutive reactions A->B->C in a batch reactor, temperature profile (Ray 1981; Luus; APMonitor/GAMS lib).
         x1' = -k1 x1^2, x2' = k1 x1^2 - k2 x2, k1 = 4000 exp(-2500/T), k2 = 6.2e5 exp(-5000/T), 298 <= T <= 398,
         x(0) = (1, 0), tf = 1, maximize x2(tf). Published optimum 0.6108.
catmix   Catalyst mixing in a tubular reactor (Gunn & Thomas 1965; COPS #14).
         x1' = u (10 x2 - x1), x2' = u (x1 - 10 x2) - (1 - u) x2, 0 <= u <= 1, x(0) = (1, 0), tf = 1,
         maximize 1 - x1(tf) - x2(tf). Analytic optimum 0.048056.
parkram  Fed-batch production of secreted protein (Park & Ramirez 1988; Luus 1995; Banga et al.).
         x1' = g1 (x2 - x1) - u x1/x5, x2' = g2 x3 - u x2/x5, x3' = g3 x3 - u x3/x5,
         x4' = -7.3 g3 x3 + u (20 - x4)/x5, x5' = u; g3 = 21.87 x4/((x4 + 0.4)(x4 + 62.5)),
         g2 = x4 exp(-5 x4)/(0.1 + x4), g1 = 4.75 g3/(0.12 + g3); 0 <= u <= 2, x(0) = (0, 0, 1, 5, 1), tf = 15,
         maximize x1(tf) x5(tf). Published optimum about 32.6-32.7 (continuous control).
Command u_k in [-1, 1] maps linearly onto the control bounds.
"""
import math

def _rk4(f, x, u, h, n):
    for _ in range(n):
        k1 = f(x, u); k2 = f([a + h / 2 * b for a, b in zip(x, k1)], u)
        k3 = f([a + h / 2 * b for a, b in zip(x, k2)], u); k4 = f([a + h * b for a, b in zip(x, k3)], u)
        x = [a + h / 6 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(x, k1, k2, k3, k4)]
    return x

def f_batch(x, T):
    k1 = 4000.0 * math.exp(-2500.0 / T); k2 = 6.2e5 * math.exp(-5000.0 / T)
    return [-k1 * x[0] ** 2, k1 * x[0] ** 2 - k2 * x[1]]

def f_catmix(x, u):
    return [u * (10 * x[1] - x[0]), u * (x[0] - 10 * x[1]) - (1 - u) * x[1]]

def f_parkram(x, u):
    x1, x2, x3, x4, x5 = x
    g3 = 21.87 * x4 / ((x4 + 0.4) * (x4 + 62.5)); g2 = x4 * math.exp(-5 * x4) / (0.1 + x4); g1 = 4.75 * g3 / (0.12 + g3)
    return [g1 * (x2 - x1) - u * x1 / x5, g2 * x3 - u * x2 / x5, g3 * x3 - u * x3 / x5, -7.3 * g3 * x3 + u * (20 - x4) / x5, u]

PROBLEMS = {
    'batch':   {'f': f_batch, 'x0': [1.0, 0.0], 'lo': 298.0, 'hi': 398.0, 'tf': 1.0, 'sub': 40, 'J': lambda x: x[1], 'J_lit': 0.6108,
                'scale': [1.0, 1.0]},
    'catmix':  {'f': f_catmix, 'x0': [1.0, 0.0], 'lo': 0.0, 'hi': 1.0, 'tf': 1.0, 'sub': 40, 'J': lambda x: 1 - x[0] - x[1], 'J_lit': 0.048056,
                'scale': [1.0, 0.1]},
    'parkram': {'f': f_parkram, 'x0': [0.0, 0.0, 1.0, 5.0, 1.0], 'lo': 0.0, 'hi': 2.0, 'tf': 15.0, 'sub': 60, 'J': lambda x: x[0] * x[4], 'J_lit': 32.68,
                'scale': [2.0, 2.0, 10.0, 10.0, 15.0]},
}

def simulate(name, u, H=20):
    P = PROBLEMS[name]; assert len(u) == H
    x = list(P['x0']); h = P['tf'] / H / P['sub']; states = [[a / s for a, s in zip(x, P['scale'])]]
    for k in range(H):
        c = P['lo'] + (P['hi'] - P['lo']) * (max(-1.0, min(1.0, u[k])) + 1) / 2
        x = _rk4(P['f'], x, c, h, P['sub'])
        states.append([max(-1e6, min(1e6, a / s)) for a, s in zip(x, P['scale'])])
    J = P['J'](x)
    return {'J': J, 'score': max(0.0, min(1.0, J / P['J_lit'])), 'states': states}
