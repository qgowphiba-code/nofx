"""UVA/Padova type-1 diabetes model (Dalla Man et al. 2007; Kovatchev et al. 2009), reimplemented from the open-source
simglucose package (jxx123/simglucose: t1dpatient.py model, vpatient_params.csv, Quest.csv) for the 30 virtual patients.
Meal eaten at 5 g/min; insulin in U/min; plasma glucose G = x4/Vg (mg/dL). Risk: Kovatchev symmetrized BG risk index."""
import csv, math
from pathlib import Path
RAW = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'simglucose'
_P = {r['Name']: r for r in csv.DictReader(open(RAW / 'vpatient_params.csv'))}
_Q = {r['Name']: r for r in csv.DictReader(open(RAW / 'Quest.csv'))}
NAMES = list(_P)
def params(name):
    return {k: float(v) for k, v in _P[name].items() if k not in ('Name',)}
def quest(name):
    return {k: float(v) for k, v in _Q[name].items() if k != 'Name'}
X0_KEYS = [k for k in _P[NAMES[0]] if k.startswith('x0_')]

def deriv(x, cho_mg, ins_pmolkg, p, last_qsto, last_food):
    d = [0.0] * 13
    qsto = x[0] + x[1]; Dbar = last_qsto + last_food * 1000
    d[0] = -p['kmax'] * x[0] + cho_mg
    if Dbar > 0:
        aa = 5 / (2 * Dbar * (1 - p['b'])); cc = 5 / (2 * Dbar * p['d'])
        kgut = p['kmin'] + (p['kmax'] - p['kmin']) / 2 * (math.tanh(aa * (qsto - p['b'] * Dbar)) - math.tanh(cc * (qsto - p['d'] * Dbar)) + 2)
    else:
        kgut = p['kmax']
    d[1] = p['kmax'] * x[0] - x[1] * kgut
    d[2] = kgut * x[1] - p['kabs'] * x[2]
    Rat = p['f'] * p['kabs'] * x[2] / p['BW']
    EGPt = p['kp1'] - p['kp2'] * x[3] - p['kp3'] * x[8]
    Et = p['ke1'] * (x[3] - p['ke2']) if x[3] > p['ke2'] else 0.0
    d[3] = max(EGPt, 0) + Rat - p['Fsnc'] - Et - p['k1'] * x[3] + p['k2'] * x[4]
    d[3] = d[3] if x[3] >= 0 else 0.0
    Uidt = (p['Vm0'] + p['Vmx'] * x[6]) * x[4] / (p['Km0'] + x[4])
    d[4] = -Uidt + p['k1'] * x[3] - p['k2'] * x[4]; d[4] = d[4] if x[4] >= 0 else 0.0
    d[5] = -(p['m2'] + p['m4']) * x[5] + p['m1'] * x[9] + p['ka1'] * x[10] + p['ka2'] * x[11]
    It = x[5] / p['Vi']; d[5] = d[5] if x[5] >= 0 else 0.0
    d[6] = -p['p2u'] * x[6] + p['p2u'] * (It - p['Ib'])
    d[7] = -p['ki'] * (x[7] - It)
    d[8] = -p['ki'] * (x[8] - x[7])
    d[9] = -(p['m1'] + p['m30']) * x[9] + p['m2'] * x[5]; d[9] = d[9] if x[9] >= 0 else 0.0
    d[10] = ins_pmolkg - (p['ka1'] + p['kd']) * x[10]; d[10] = d[10] if x[10] >= 0 else 0.0
    d[11] = p['kd'] * x[10] - p['ka2'] * x[11]; d[11] = d[11] if x[11] >= 0 else 0.0
    d[12] = -p['ksc'] * x[12] + p['ksc'] * x[3]; d[12] = d[12] if x[12] >= 0 else 0.0
    return d

def basal_rate(p):
    return p['u2ss'] * p['BW'] / 6000.0          # U/min

def simulate(name, meal_g, insulin_fn, minutes, sub=2):
    """insulin_fn(t_min) -> U/min. Returns plasma glucose trace (mg/dL) sampled every minute."""
    p = params(name); x = [p[k] for k in X0_KEYS]
    planned, eating, last_qsto, last_food, prev_cho = meal_g, False, 0.0, 0.0, 0.0
    G = []; h = 1.0 / sub
    for t in range(minutes):
        to_eat = min(5.0, planned) if planned > 0 else 0.0; planned = max(0.0, planned - to_eat)
        if to_eat > 0 and prev_cho <= 0:
            last_qsto = x[0] + x[1]; last_food = 0.0; eating = True
        if eating: last_food += to_eat
        if to_eat <= 0 and prev_cho > 0: eating = False
        prev_cho = to_eat
        ins = insulin_fn(t) * 6000.0 / p['BW']; cho = to_eat * 1000.0
        for _ in range(sub):
            k1 = deriv(x, cho, ins, p, last_qsto, last_food)
            k2 = deriv([a + h / 2 * b for a, b in zip(x, k1)], cho, ins, p, last_qsto, last_food)
            k3 = deriv([a + h / 2 * b for a, b in zip(x, k2)], cho, ins, p, last_qsto, last_food)
            k4 = deriv([a + h * b for a, b in zip(x, k3)], cho, ins, p, last_qsto, last_food)
            x = [a + h / 6 * (b + 2 * c + 2 * e + f) for a, b, c, e, f in zip(x, k1, k2, k3, k4)]
        G.append(x[3] / p['Vg'])
    return G

def risk(bg):
    if bg <= 20: return 100.0
    if bg >= 600: return 100.0
    U = 1.509 * (math.log(bg) ** 1.084 - 5.381); return 10 * U * U

def risk_index(G):
    return sum(risk(g) for g in G) / len(G)
