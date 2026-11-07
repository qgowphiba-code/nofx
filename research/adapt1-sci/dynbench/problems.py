"""Published open-loop dynamic-optimization benchmarks (piecewise-constant control over H equal stages; stdlib).

batch    Consecutive reactions A->B->C in a batch reactor, temperature profile (Ray 1981; Luus; APMonitor/GAMS lib).
         x1' = -k1 x1^2, x2' = k1 x1^2 - k2 x2, k1 = 4000 exp(-2500/T), k2 = 6.2e5 exp(-5000/T), 298 <= T <= 398,
         x(0) = (1, 0), tf = 1, maximize x2(tf). Published optimum 0.6108.
catmix   Catalyst mixing in a tubular reactor (Gunn & Thomas 1965; COPS #14).
         x1' = u (10 x2 - x1), x2' = u (x1 - 10 x2) - (1 - u) x2, 0 <= u <= 1, x(0) = (1, 0), tf = 1,
         maximize 1 - x1(tf) - x2(tf). Analytic optimum 0.048056.
penicillin  Fed-batch penicillin (Cuthrell & Biegler 1989), volume <= 10 L; published optimum about 87.7-88.0 (see penicillin.py).
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

def f_penicillin(x, u):
    x1, x2, x3, x4 = x; x3p = max(x3, 0.0)
    if x4 >= 10.0: u = 0.0                        # volume cap: the executor shuts off the feed when the reactor is full
    h1 = 0.11 * x3p / (0.006 * x1 + x3p + 1e-12); h2 = 0.0055 * x3p / (0.0001 + x3p * (1 + 10 * x3p))
    return [h1 * x1 - u * x1 / (500 * x4), h2 * x1 - 0.01 * x2 - u * x2 / (500 * x4),
            -h1 * x1 / 0.47 - h2 * x1 / 1.2 - 0.029 * x1 * x3p / (0.0001 + x3p) + u / x4 * (1 - x3 / 500), u / 500]

PROBLEMS = {
    'batch':   {'f': f_batch, 'x0': [1.0, 0.0], 'lo': 298.0, 'hi': 398.0, 'tf': 1.0, 'sub': 40, 'J': lambda x: x[1], 'J_lit': 0.6108,
                'scale': [1.0, 1.0]},
    'catmix':  {'f': f_catmix, 'x0': [1.0, 0.0], 'lo': 0.0, 'hi': 1.0, 'tf': 1.0, 'sub': 40, 'J': lambda x: 1 - x[0] - x[1], 'J_lit': 0.048056,
                'scale': [1.0, 0.1]},
    'penicillin': {'f': f_penicillin, 'x0': [1.5, 0.0, 0.0, 7.0], 'lo': 0.0, 'hi': 50.0, 'tf': 132.0, 'sub': 200, 'J': lambda x: x[1] * x[3],
                   'J_lit': 88.0, 'scale': [40.0, 10.0, 25.0, 10.0]},
    'parkram': {'f': f_parkram, 'x0': [0.0, 0.0, 1.0, 5.0, 1.0], 'lo': 0.0, 'hi': 2.0, 'tf': 15.0, 'sub': 60, 'J': lambda x: x[0] * x[4], 'J_lit': 32.68,
                'scale': [2.0, 2.0, 10.0, 10.0, 15.0]},
}

CUSTOM = {}
def _insulin_nominal(u):
    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'varied'))
    import insulin_task as IT
    q, G, ri = IT.run(u, IT.NOMINAL)
    st = [[k / 20, G[min(len(G) - 1, k * 15)] / 150.0, (G[min(len(G) - 1, k * 15)] - G[max(0, k * 15 - 15)]) / 50.0, 0.0] for k in range(21)]
    st[0] = [0.0, G[0] / 150.0, 0.0, 0.0]
    return {'J': q, 'score': q, 'states': st}
CUSTOM['insulin'] = _insulin_nominal
PROBLEMS['insulin'] = {'x0': [0.0, 139.0 / 150.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0], 'J_lit': 1.0}

def _arm_nominal(u):
    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'varied'))
    import arm_task as AT
    q, st = AT.run(u, AT.NOMINAL)
    return {'J': q, 'score': q, 'states': st}
CUSTOM['arm'] = _arm_nominal
PROBLEMS['arm'] = {'x0': None, 'scale': None, 'J_lit': 1.0, 'adim': 2, 'state_dims': 6, 'goal_indices': [4, 5], 'goal': [1.0, 0.5]}

def simulate(name, u, H=20):
    if name in CUSTOM:
        return CUSTOM[name](u)
    P = PROBLEMS[name]; assert len(u) == H
    x = list(P['x0']); h = P['tf'] / H / P['sub']; states = [[a / s for a, s in zip(x, P['scale'])]]
    for k in range(H):
        c = P['lo'] + (P['hi'] - P['lo']) * (max(-1.0, min(1.0, u[k])) + 1) / 2
        x = _rk4(P['f'], x, c, h, P['sub'])
        if name == 'penicillin': x[2] = max(x[2], 0.0)
        states.append([max(-1e6, min(1e6, a / s)) for a, s in zip(x, P['scale'])])
    J = P['J'](x)
    return {'J': J, 'score': max(0.0, min(1.0, J / P['J_lit'])), 'states': states}
