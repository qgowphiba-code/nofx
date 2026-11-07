"""Planar two-link arm (Li & Todorov 2004 human-arm model; values as reproduced in arXiv 1908.07615):
l1 = 0.30 m, l2 = 0.33 m, I1 = 0.025, I2 = 0.045 kg m^2, m2 = 1.0 kg, s2 = 0.16 m, joint friction B = [[0.05, 0.025], [0.025, 0.05]].
M(q) = [[a1 + 2 a2 cos q2, a3 + a2 cos q2], [a3 + a2 cos q2, a3]], a1 = I1 + I2' + m2' l1^2, a2 = l1 (m2 s2 + mp l2), a3 = I2';
C(q, w) w = a2 sin q2 [-w2 (2 w1 + w2), w1^2]; a point payload mp at the hand adds mp l2^2 to I2.
Task: from rest at q = (pi/4, pi/2), move the hand to a goal and stop there at T = 0.5 s; 20 commands of 25 ms, each two joint
torques (u in [-1, 1] -> 8 N m). quality = 1 / (1 + (dist / 1 cm)^2 + (speed / 0.1 m/s)^2).
Context (measured before the move): goal offset (dx, dy) from the nominal goal, payload mass."""
import math, random
H, D, DT, SUB, TMAX = 20, 2, 0.025, 10, 8.0
L1, L2, I1, I2, M2, S2 = 0.30, 0.33, 0.025, 0.045, 1.0, 0.16
B11, B12 = 0.05, 0.025
Q0 = (math.pi / 4, math.pi / 2)
def hand(q):
    return (L1 * math.cos(q[0]) + L2 * math.cos(q[0] + q[1]), L1 * math.sin(q[0]) + L2 * math.sin(q[0] + q[1]))
START = hand(Q0)
NOM_GOAL = (START[0] + 0.10, START[1] + 0.05)
NOMINAL = {'dx': 0.0, 'dy': 0.0, 'mp': 0.5}

def step(s, tau, mp):
    q1, q2, w1, w2 = s
    i2 = I2 + mp * L2 * L2; a1 = I1 + i2 + (M2 + mp) * L1 * L1; a2 = L1 * (M2 * S2 + mp * L2); a3 = i2
    c, sn = math.cos(q2), math.sin(q2)
    m11, m12, m22 = a1 + 2 * a2 * c, a3 + a2 * c, a3
    h1 = -a2 * sn * w2 * (2 * w1 + w2) + B11 * w1 + B12 * w2
    h2 = a2 * sn * w1 * w1 + B12 * w1 + B11 * w2
    r1, r2 = tau[0] - h1, tau[1] - h2
    det = m11 * m22 - m12 * m12
    return [w1, w2, (m22 * r1 - m12 * r2) / det, (-m12 * r1 + m11 * r2) / det]

def run(u, ctx):
    """u: H rows of D commands (or a flat list of H*D). Returns (quality, states)."""
    rows = u if isinstance(u[0], (list, tuple)) else [u[2 * k:2 * k + 2] for k in range(H)]
    goal = (NOM_GOAL[0] + ctx['dx'], NOM_GOAL[1] + ctx['dy']); mp = ctx['mp']
    s = [Q0[0], Q0[1], 0.0, 0.0]; h = DT / SUB; states = []
    def obs(s):
        x, y = hand(s); return [s[0], s[1], s[2] / 5, s[3] / 5, (x - START[0]) / 0.1, (y - START[1]) / 0.1]
    states.append(obs(s))
    for k in range(H):
        tau = [TMAX * max(-1.0, min(1.0, v)) for v in rows[k]]
        for _ in range(SUB):
            k1 = step(s, tau, mp); k2 = step([a + h / 2 * b for a, b in zip(s, k1)], tau, mp)
            k3 = step([a + h / 2 * b for a, b in zip(s, k2)], tau, mp); k4 = step([a + h * b for a, b in zip(s, k3)], tau, mp)
            s = [a + h / 6 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(s, k1, k2, k3, k4)]
        states.append(obs(s))
    x, y = hand(s)
    dist = math.hypot(x - goal[0], y - goal[1])
    l1v = (-L1 * math.sin(s[0]) - L2 * math.sin(s[0] + s[1])) * s[2] - L2 * math.sin(s[0] + s[1]) * s[3]
    l2v = (L1 * math.cos(s[0]) + L2 * math.cos(s[0] + s[1])) * s[2] + L2 * math.cos(s[0] + s[1]) * s[3]
    speed = math.hypot(l1v, l2v)
    return 1.0 / (1.0 + (dist / 0.01) ** 2 + (speed / 0.1) ** 2), states

def J(u, ctx, drift=False):
    return run(u, ctx)[0]
def vec(c):
    return [c['dx'] / 0.06, c['dy'] / 0.06, (c['mp'] - 0.75) / 0.75]
def sample_train(r):
    return {'dx': r.uniform(-0.06, 0.06), 'dy': r.uniform(-0.06, 0.06), 'mp': r.uniform(0.0, 1.5)}
HELD = [sample_train(random.Random(9000 + i)) for i in range(32)]
