"""Experiment-level reaction-condition optimization with hidden kinetics (stdlib only).

One experiment = one condition from a grid: catalyst (4) x temperature (4) x residence time (3) = 48 conditions.
The measured outcome is the yield of product P, with 2% (absolute) Gaussian measurement noise.

Hidden kinetics per reaction (isothermal, well mixed, first order):
  A -> P   k1 = a_c * k1ref * arr(E1)         (catalyst activity a_c)
  P -> D   k2 = k2ref * arr(E2)               (product degradation)
  A -> S   k3 = s_c * k3ref * arr(E3)         (side reaction, catalyst selectivity s_c)
  catalyst deactivation: k1 and k3 decay as exp(-d_c * t)
  arr(E) = exp(-E/R (1/T - 1/Tref))

Reactions come from four recurring families (different activation-energy patterns and catalyst effects);
each reaction jitters its family's parameters. Two observable substrate descriptors are noisy signals of the
family; a third descriptor is pure noise. Learners see the descriptors, never the family or the kinetics.
"""
import math, random

CATALYSTS = ['cat0', 'cat1', 'cat2', 'cat3']
TEMPS = [30.0, 50.0, 70.0, 90.0]
TAUS = [2.0, 8.0, 30.0]
CONDITIONS = [(c, T, tau) for c in range(4) for T in TEMPS for tau in TAUS]
R, TREF = 8.314e-3, 333.15        # kJ/mol/K, 60 C
NOISE = 0.02

# family -> (centre of descriptors, kinetic pattern)
FAMILIES = {
    0: {'desc': (-1.0, -1.0), 'k1': 0.08, 'E1': 60, 'k2': 0.030, 'E2': 90, 'k3': 0.01, 'E3': 50, 'act': (1.0, 2.5, 0.6, 1.4), 'sel': (1.0, 1.0, 1.0, 1.0), 'deact': (0.0, 0.02, 0.0, 0.0)},
    1: {'desc': (1.0, -1.0), 'k1': 0.05, 'E1': 45, 'k2': 0.002, 'E2': 60, 'k3': 0.03, 'E3': 95, 'act': (1.0, 1.0, 1.0, 1.0), 'sel': (2.0, 0.5, 1.0, 3.0), 'deact': (0.0, 0.0, 0.0, 0.0)},
    2: {'desc': (-1.0, 1.0), 'k1': 0.20, 'E1': 70, 'k2': 0.004, 'E2': 50, 'k3': 0.01, 'E3': 60, 'act': (1.5, 1.0, 2.0, 0.8), 'sel': (1.0, 1.0, 1.0, 1.0), 'deact': (0.15, 0.05, 0.30, 0.01)},
    3: {'desc': (1.0, 1.0), 'k1': 0.006, 'E1': 80, 'k2': 0.001, 'E2': 60, 'k3': 0.002, 'E3': 70, 'act': (0.7, 1.0, 1.6, 2.2), 'sel': (1.0, 1.0, 1.0, 1.0), 'deact': (0.0, 0.0, 0.0, 0.003)},
}


def arr(E, T):
    return math.exp(-E / R * (1.0 / (T + 273.15) - 1.0 / TREF))


def make_reaction(i, seed_base=91000):
    r = random.Random(seed_base + i)
    fam = r.randrange(4); F = FAMILIES[fam]
    j = lambda x, s=0.35: x * math.exp(r.gauss(0, s))
    rx = {'id': i, 'family': fam,
          'k1': j(F['k1']), 'E1': F['E1'] + r.gauss(0, 6), 'k2': j(F['k2']), 'E2': F['E2'] + r.gauss(0, 6),
          'k3': j(F['k3']), 'E3': F['E3'] + r.gauss(0, 6),
          'act': [j(a, 0.25) for a in F['act']], 'sel': [j(s, 0.25) for s in F['sel']],
          'deact': [j(d, 0.25) if d > 0 else 0.0 for d in F['deact']],
          'descriptors': [F['desc'][0] + r.gauss(0, 0.6), F['desc'][1] + r.gauss(0, 0.6), r.gauss(0, 1.0)]}
    return rx


def true_yield(rx, cond):
    c, T, tau = CONDITIONS[cond] if isinstance(cond, int) else cond
    k1 = rx['act'][c] * rx['k1'] * arr(rx['E1'], T)
    k2 = rx['k2'] * arr(rx['E2'], T)
    k3 = rx['sel'][c] * rx['k3'] * arr(rx['E3'], T)
    d = rx['deact'][c]
    A, P, n = 1.0, 0.0, 400
    h = tau / n
    t = 0.0
    def f(A, P, t):
        g = math.exp(-d * t)
        return -(k1 + k3) * g * A, k1 * g * A - k2 * P
    for _ in range(n):
        a1, p1 = f(A, P, t); a2, p2 = f(A + h * a1 / 2, P + h * p1 / 2, t + h / 2)
        a3, p3 = f(A + h * a2 / 2, P + h * p2 / 2, t + h / 2); a4, p4 = f(A + h * a3, P + h * p3, t + h)
        A += h * (a1 + 2 * a2 + 2 * a3 + a4) / 6; P += h * (p1 + 2 * p2 + 2 * p3 + p4) / 6; t += h
    return max(0.0, min(1.0, P))


def measure(rx, cond, rng):
    return max(0.0, min(1.0, true_yield(rx, cond) + rng.gauss(0, NOISE)))


def grid(rx):
    return [true_yield(rx, k) for k in range(len(CONDITIONS))]
