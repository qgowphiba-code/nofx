"""Seeded batch cooling crystallization with hidden kinetics (stdlib only).

Per gram of solvent. Concentration C [g/g], temperature T [degC], time [min], size [um].
  solubility   C*(T) = s0 * exp(s1 * (T - 20))
  supersat     S = (C - C*) / C*
  growth       G = kg * S^g                      (size independent, seeds and nuclei alike)
  secondary    B2 = kb * S^b * M                 (M = crystal mass, g/g; attrition/contact nucleation)
  primary      B1 = kp * S^p                     (steep, burst-like at high S)
Seeds are monodisperse (number Ns, size L). Nuclei are tracked by moments m0..m3 (born at size ~0).
  dL/dt = G, dm0 = B1+B2, dm1 = G m0, dm2 = 2 G m1, dm3 = 3 G m2
  dC/dt = -3 rho_kv G (Ns L^2 + m2)

Control: H commands u_k in [-1, 1]. Each interval of equal length lowers T by
dT_k = (T0 - Tf) * exp(beta u_k) / sum_j exp(beta u_j): monotone cooling, fixed batch time, fixed end point.

Outcome `quality` = mass grown on seeds / theoretical yield (C0 - C*(Tf)), in [0, 1]. Mass that goes into
nuclei (fines) or stays dissolved (unfinished desupersaturation) is lost.
Per-interval measured outcome (optional): exp(-new fines count in the interval / N_REF), an FBRM-style fines signal.

The kinetics (kg, g, kb, b, kp, p) and the solubility slope are hidden from every learner. A batch's measured
context (seed load, seed size, start temperature) is observable before the batch.
"""
import math

RHO_KV = 1.3e-12 * 0.52      # g/um^3 (density 1.3 g/cm3, volume shape factor 0.52)
TF, TB = 20.0, 120.0         # end temperature [C], batch time [min]
H = 16                       # commands per batch
BETA = 3.5
DT = 0.2                     # integration step [min] (0.2 vs 0.025 differs by < 1e-4 in quality)
N_REF = 2.0e4                # fines-count scale for the per-interval signal (#/g solvent)
S_SCALE = 0.2                # supersaturation scale for the reported state


def profile(u, T0):
    w = [math.exp(BETA * max(-1.0, min(1.0, x))) for x in u]
    s = sum(w)
    return [(T0 - TF) * x / s for x in w]


def solubility(T, comp):
    return comp['s0'] * math.exp(comp['s1'] * (T - 20.0))


def simulate(u, comp, ctx, trace=False):
    """u: H commands; comp: hidden kinetics; ctx: dict(seed_frac, seed_size, T0). Returns dict."""
    assert len(u) == H
    T0, ls0 = ctx['T0'], ctx['seed_size']
    c_star0 = solubility(T0, comp)
    C = c_star0                                   # saturated at T0
    ytheo = C - solubility(TF, comp)
    seed_mass = ctx['seed_frac'] * ytheo
    Ns = seed_mass / (RHO_KV * ls0 ** 3)
    L = ls0
    m0 = m1 = m2 = m3 = 0.0
    kg, g, kb, b, kp, p = comp['kg'], comp['g'], comp['kb'], comp['kb_exp'], comp['kp'], comp['kp_exp']
    drops = profile(u, T0)
    seg = TB / H
    n_sub = int(round(seg / DT))
    h = seg / n_sub
    T = T0
    states = [[0.0, 1.0, 0.0, 0.0]]
    step_out, smax = [], 0.0

    def deriv(C, L, m0, m1, m2, T):
        cs = solubility(T, comp)
        S = (C - cs) / cs
        if S <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, S
        G = kg * S ** g
        M = RHO_KV * (Ns * L ** 3 + m3_cur[0]) - seed_mass + seed_mass   # crystal mass incl. seeds
        B = kb * S ** b * M + kp * S ** p
        dC = -3.0 * RHO_KV * G * (Ns * L * L + m2)
        return dC, G, B, G * m0, 2.0 * G * m1, 3.0 * G * m2, S

    m3_cur = [0.0]
    for k in range(H):
        rate = drops[k] / seg
        m0_start = m0
        for _ in range(n_sub):
            m3_cur[0] = m3
            k1 = deriv(C, L, m0, m1, m2, T)
            k2 = deriv(C + 0.5 * h * k1[0], L + 0.5 * h * k1[1], m0 + 0.5 * h * k1[2], m1 + 0.5 * h * k1[3], m2 + 0.5 * h * k1[4], T - 0.5 * h * rate)
            k3 = deriv(C + 0.5 * h * k2[0], L + 0.5 * h * k2[1], m0 + 0.5 * h * k2[2], m1 + 0.5 * h * k2[3], m2 + 0.5 * h * k2[4], T - 0.5 * h * rate)
            k4 = deriv(C + h * k3[0], L + h * k3[1], m0 + h * k3[2], m1 + h * k3[3], m2 + h * k3[4], T - h * rate)
            C += h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
            L += h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
            m0 += h * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
            m1 += h * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6
            m2 += h * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4]) / 6
            m3 += h * (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5]) / 6
            T -= h * rate
            smax = max(smax, k1[6])
            cs = solubility(T, comp)
            if C < cs:          # growth cannot overshoot saturation within a step
                C = cs
        cs = solubility(T, comp)
        S = (C - cs) / cs
        grown = (c_star0 - C)                                  # mass left solution so far
        states.append([(k + 1) / H, (T - TF) / (T0 - TF), max(0.0, min(1.0, S / S_SCALE)), max(0.0, min(1.0, grown / ytheo))])
        step_out.append(math.exp(-(m0 - m0_start) / N_REF))
    seed_grown = RHO_KV * Ns * (L ** 3 - ls0 ** 3)
    fines = RHO_KV * m3
    q = max(0.0, min(1.0, seed_grown / ytheo))
    out = {'quality': q, 'fines_frac': fines / ytheo, 'left_dissolved': max(0.0, C - solubility(TF, comp)) / ytheo,
           'final_size': L, 'nuclei_count': m0, 'max_supersat': smax, 'states': states, 'step_outcomes': step_out}
    return out
