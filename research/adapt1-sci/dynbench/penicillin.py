"""Fed-batch penicillin fermentation (Cuthrell & Biegler 1989; also Banga et al.; Luus), fixed final time version.
x1 biomass, x2 penicillin, x3 substrate, x4 volume; feed rate u in [0, 50]; tf = 132 h; maximize x2(tf) x4(tf).
x1' = h1 x1 - u x1/(500 x4); x2' = h2 x1 - 0.01 x2 - u x2/(500 x4);
x3' = -h1 x1/0.47 - h2 x1/1.2 - 0.029 x1 x3/(0.0001 + x3) + u/x4 (1 - x3/500); x4' = u/500;
h1 = 0.11 x3/(0.006 x1 + x3); h2 = 0.0055 x3/(0.0001 + x3 (1 + 10 x3)); x(0) = (1.5, 0, 0, 7).
Published optimum about 87.7-88.0. Path constraint x4 <= 10 L: the executor shuts off the feed when the reactor is full. sf = feed substrate concentration (published 500)."""
import problems
H = 20
def J(u, sf=500.0, x10=1.5, sub=200):
    def f(x, c):
        x1, x2, x3, x4 = x; x3p = max(x3, 0.0)
        if x4 >= 10.0: c = 0.0
        h1 = 0.11 * x3p / (0.006 * x1 + x3p + 1e-12); h2 = 0.0055 * x3p / (0.0001 + x3p * (1 + 10 * x3p))
        return [h1 * x1 - c * x1 / (sf * x4), h2 * x1 - 0.01 * x2 - c * x2 / (sf * x4),
                -h1 * x1 / 0.47 - h2 * x1 / 1.2 - 0.029 * x1 * x3p / (0.0001 + x3p) + c / x4 * (1 - x3 / sf), c / sf]
    x = [x10, 0.0, 0.0, 7.0]; h = 132.0 / H / sub
    for k in range(H):
        x = problems._rk4(f, x, 50.0 * (max(-1.0, min(1.0, u[k])) + 1) / 2, h, sub)
        x[2] = max(x[2], 0.0)
    return x[1] * x[3]
