"""Batch reactor RL case study of Petsagkourakis, del Rio-Chanona et al. (Comput. Chem. Eng. 2020), as defined in the authors'
code (OptiMaL-PSE-Lab/REINFORCE-PSE, src/models/ode.py, defaults): two controls U1, U2 in [0, 5], t in (0, 1), 20 intervals,
y(0) = (1, 0), maximize y2(1).
simple (approximate model):  y1' = -(U1 + a U1^2) y1 + d U2;                 y2' = (b U1 - c U2) y1;  (a, b, c, d) = (0.5, 1, 1, 1)
complex ("plant"):           y1' = -(U1 + a U2^2) y1 + d U2 y2 / (y1 + y2);  y2' = (b U1 - c U2) y1;  (a, b, c, d) = (0.5, 1, 0.7, 0.5)
Published RL budget (code defaults): REINFORCE, RNN policy, 100 episodes x 250 iterations on the simple model, then 100 x 100 on
the complex model (transfer), 5 seeds."""
H, D, SUB = 20, 2, 80     # 80 RK4 sub-steps per interval + non-negative states: coarser steps let optimizers exploit
                          # an integration blow-up in the complex model's y2/(y1+y2) term (a spurious 'optimum' of 36)
def f(model, y, u):
    y1, y2 = y; U1, U2 = u
    if model == 'simple':
        a, b, c, d = 0.5, 1.0, 1.0, 1.0
        return [-(U1 + a * U1 ** 2) * y1 + d * U2, (b * U1 - c * U2) * y1]
    a, b, c, d = 0.5, 1.0, 0.7, 0.5
    return [-(U1 + a * U2 ** 2) * y1 + d * U2 * y2 / (y1 + y2 + 1e-12), (b * U1 - c * U2) * y1]
def run(model, u, y10=1.0):
    rows = u if isinstance(u[0], (list, tuple)) else [u[2 * k:2 * k + 2] for k in range(H)]
    y = [y10, 0.0]; h = 1.0 / H / SUB; states = [[y[0], y[1], 0.0]]
    for k in range(H):
        c = [2.5 * (max(-1.0, min(1.0, v)) + 1) for v in rows[k]]
        for _ in range(SUB):
            k1 = f(model, y, c); k2 = f(model, [a + h / 2 * b for a, b in zip(y, k1)], c)
            k3 = f(model, [a + h / 2 * b for a, b in zip(y, k2)], c); k4 = f(model, [a + h * b for a, b in zip(y, k3)], c)
            y = [max(0.0, a + h / 6 * (b + 2 * cc + 2 * dd + e)) for a, b, cc, dd, e in zip(y, k1, k2, k3, k4)]
        states.append([y[0], y[1], (k + 1) / H])
    return y[1], states
