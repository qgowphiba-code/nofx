"""Park-Ramirez with measured batch-to-batch variation (our variant of the published problem).
Context per batch, measured before it starts: feed substrate concentration sf in [12, 28] (published value 20) and
initial glucose x4(0) in [3, 7] (published value 5). Everything else is the published model; J = x1(tf) x5(tf)."""
import math, random, json, glob
from pathlib import Path
import problems
H = 20
J_PUB = 32.68

def J(u, sf, x40):
    def f(x, c):
        x1, x2, x3, x4, x5 = x
        g3 = 21.87 * x4 / ((x4 + 0.4) * (x4 + 62.5)); g2 = x4 * math.exp(-5 * x4) / (0.1 + x4); g1 = 4.75 * g3 / (0.12 + g3)
        return [g1 * (x2 - x1) - c * x1 / x5, g2 * x3 - c * x2 / x5, g3 * x3 - c * x3 / x5, -7.3 * g3 * x3 + c * (sf - x4) / x5, c]
    x = [0.0, 0.0, 1.0, x40, 1.0]; h = 15.0 / H / 60
    for k in range(H):
        x = problems._rk4(f, x, 2.0 * (max(-1.0, min(1.0, u[k])) + 1) / 2, h, 60)
    return x[0] * x[4]

def contexts(n, seed):
    r = random.Random(seed)
    return [{'sf': r.uniform(12.0, 28.0), 'x40': r.uniform(3.0, 7.0)} for _ in range(n)]

def vec(c):
    return [(c['sf'] - 20.0) / 8.0, (c['x40'] - 5.0) / 2.0]

SUPPORT = contexts(64, 101)
TRAIN = contexts(256, 202)
HELD = contexts(32, 303)

def machina_base():
    """Machina's best acquired profile at the published (nominal) conditions, from the acquisition runs."""
    best = max((json.load(open(p)) for p in glob.glob(str(Path(__file__).resolve().parents[1] / 'runs/dyn/machina-parkram-*/summary.json'))),
               key=lambda d: d['best_J'])
    return best['best_u'], best['best_J'], best['domain_id']
