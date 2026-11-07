"""Post-meal insulin dosing on the UVA/Padova virtual population.
Control: 20 commands, one per 15 min over 5 h; command u in [-1, 1] -> infusion = basal x 64^u (U/min).
Outcome: quality = exp(-RI / 10), RI = mean Kovatchev risk index of plasma glucose over the 5 h (hypo and hyper both cost).
Context (measured before the meal): log body weight, log carb ratio, log correction factor, log total daily insulin, meal size.
Train patients: adult/adolescent #001-#007, child #001-#006 (20). Held out: adult/adolescent #008-#010, child #007-#010 (10)."""
import math, random
import uvap as U
H = 20
HELD_P = [f'adult#{i:03d}' for i in (8, 9, 10)] + [f'adolescent#{i:03d}' for i in (8, 9, 10)] + [f'child#{i:03d}' for i in (7, 8, 9, 10)]
TRAIN_P = [n for n in U.NAMES if n not in HELD_P]
NOMINAL = {'patient': 'adult#001', 'meal': 60.0}

def run(u, ctx):
    p = U.params(ctx['patient']); b = U.basal_rate(p)
    rates = [b * 64.0 ** max(-1.0, min(1.0, x)) for x in u]
    G = U.simulate(ctx['patient'], ctx['meal'], lambda t: rates[min(H - 1, t // 15)], 300)
    ri = U.risk_index(G)
    return math.exp(-ri / 10.0), G, ri

def J(u, ctx, drift=False):
    return run(u, ctx)[0]

def vec(c):
    q = U.quest(c['patient']); p = U.params(c['patient'])
    return [math.log(p['BW'] / 50) / 0.6, math.log(q['CR'] / 12) / 0.6, math.log(q['CF'] / 40) / 0.8, math.log(q['TDI'] / 35) / 0.7, (c['meal'] - 60) / 30]

def sample_train(r):
    return {'patient': r.choice(TRAIN_P), 'meal': r.uniform(30, 90)}

HELD = [{'patient': n, 'meal': m} for n in HELD_P for m in (40.0, 60.0, 80.0)]

def calculator_u(ctx):
    """Diagnostic only: standard bolus calculator (meal / CR delivered over the first 15 min) on top of basal."""
    p = U.params(ctx['patient']); q = U.quest(ctx['patient']); b = U.basal_rate(p)
    r0 = b + ctx['meal'] / q['CR'] / 15.0
    return [min(1.0, math.log(r0 / b) / math.log(64))] + [0.0] * (H - 1)

def states(G, u, ctx):
    p = U.params(ctx['patient'])
    return [[k / H, G[min(len(G) - 1, k * 15)] / 150.0, 0.0, 0.0] for k in range(H + 1)]
