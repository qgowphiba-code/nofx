"""Hidden compound kinetics and observable batch contexts (fixed seeds)."""
import random, math

def compound(i):
    r = random.Random(7000 + i)
    return {'id': i,
            's0': 0.10, 's1': r.uniform(0.025, 0.035),
            'kg': math.exp(r.uniform(math.log(120), math.log(450))), 'g': r.uniform(1.0, 2.0),
            'kb': math.exp(r.uniform(math.log(2e6), math.log(4e7))), 'kb_exp': r.uniform(1.5, 3.0),
            'kp': math.exp(r.uniform(math.log(1e8), math.log(1e10))), 'kp_exp': r.uniform(5.0, 8.0)}

NOMINAL = {'seed_frac': 0.02, 'seed_size': 100.0, 'T0': 50.0}

def contexts(n, seed):
    r = random.Random(seed)
    return [{'seed_frac': r.uniform(0.005, 0.04), 'seed_size': r.uniform(60.0, 160.0), 'T0': r.uniform(45.0, 55.0)}
            for _ in range(n)]

def ctx_vector(c):
    """Measured context as sent to Machina (scaled to about [-1, 1])."""
    return [(math.log(c['seed_frac']) - math.log(0.0141)) / 1.04, (c['seed_size'] - 110.0) / 50.0, (c['T0'] - 50.0) / 5.0]
