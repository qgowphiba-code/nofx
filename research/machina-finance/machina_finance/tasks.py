"""Finance control tasks kept for further Machina work (simulators, context features, scores).

Every task is a committed numeric schedule (a few continuous controls per step) scored once at the
end, with measurable pre-trade context and a repeatable simulator. Scores are normalised so the
naive benchmark scores 0 and higher is better: score = (obj_naive - obj) / |obj_naive| for costs.

These definitions are byte-for-byte the ones used in the prod runs (same context seeds, same
feature scalings), so saved policies reproduce the reported numbers exactly. Dropped tasks
(real S&P daily accumulation, real-data vol targeting, baskets) are not included; see README.
"""
import hashlib
import math

import numpy as np
from scipy.optimize import minimize

ALPHA = 3.0  # log-weight decoder scale: weight = exp(ALPHA * a), a in [-1, 1]


def cid_seed(cid):
    return int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16)


def lhs(rng, n, d):
    """Latin hypercube in [-1, 1]^d."""
    u = np.empty((n, d))
    for j in range(d):
        u[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return 2 * u - 1


def lerp_log(u, lo, hi):
    return float(np.exp(np.log(lo) + (u + 1) / 2 * (np.log(hi) - np.log(lo))))


def lerp(u, lo, hi):
    return float(lo + (u + 1) / 2 * (hi - lo))


def as_rows(actions, H, A):
    a = np.asarray(actions, dtype=float).reshape(-1, A)[:H]
    return np.clip(a, -1, 1)


def schedule_weights(a_col, H):
    """Log-weight decoder. Missing (padded) slots trade nothing."""
    w = np.zeros(H)
    k = len(a_col)
    w[:k] = np.exp(ALPHA * np.clip(a_col, -1, 1))
    return w / w.sum()


def weights_to_actions(n):
    n = np.maximum(np.asarray(n, float), 1e-12)
    a = np.log(n) / ALPHA
    a = a - (a.max() + a.min()) / 2  # centre, then clip
    return np.clip(a, -1, 1)


def kkt_min(Q, b, Aeq, beq):
    """min n'Qn - 2b'n  s.t. Aeq n = beq."""
    m = Aeq.shape[0]
    K = np.block([[2 * Q, Aeq.T], [Aeq, np.zeros((m, m))]])
    rhs = np.concatenate([2 * b, beq])
    return np.linalg.solve(K, rhs)[: Q.shape[0]]


class Task:
    name = "base"
    H = 10
    A = 1
    d = 3
    feature_names = []
    kind = "cost"  # 'cost' -> lower obj better; 'utility' -> higher better
    pad = "zero"

    # --- contexts -----------------------------------------------------
    def make_contexts(self, n, seed, prefix):
        rng = np.random.default_rng(seed)
        U = lhs(rng, n, self.d)
        out = []
        for i, u in enumerate(U):
            p = self.params_from_u(u)
            cid = f"{prefix}-{i:03d}"
            out.append({"cid": cid, "u": u.tolist(), "p": p, "feat": self.features(p)})
        return out

    def nominal(self):
        p = self.params_from_u(np.zeros(self.d))
        return {"cid": "nominal", "u": [0.0] * self.d, "p": p, "feat": self.features(p)}

    # --- scoring --------------------------------------------------------
    def obj_actions(self, actions, ctx):
        return self.objective(self.decode(as_rows(actions, self.H, self.A)), ctx)

    def score(self, actions, ctx):
        o = self.obj_actions(actions, ctx)
        on = self.naive_obj(ctx)
        if self.kind == "cost":
            return (on - o) / abs(on)
        return (o - on) / abs(on)

    def naive_obj(self, ctx):
        key = ("naive", ctx["cid"], tuple(np.round(ctx["u"], 9)))
        cache = self.__dict__.setdefault("_cache", {})
        if key not in cache:
            cache[key] = self.objective(self.decode(self.naive_actions(ctx)), ctx)
        return cache[key]

    def naive_actions(self, ctx=None):
        return np.zeros((self.H, self.A))

    def acq_state(self, ctx, k):
        return [1.0 - k / self.H] + list(ctx["feat"])


# ---------------------------------------------------------------------------
# T1 Almgren-Chriss liquidation
# ---------------------------------------------------------------------------
class ACLiquidation(Task):
    name = "ac_liquidation"
    H, A, d = 10, 1, 3
    feature_names = ["log_sigma", "log_eta", "log_lambda"]
    T = 1.0

    def params_from_u(self, u):
        sigma = lerp_log(u[0], 0.01, 0.04)
        eta = lerp_log(u[1], 1e-3, 1e-2)
        kT = lerp_log(u[2], 0.5, 6.0)  # urgency drawn first, lambda implied
        lam = (kT / self.T) ** 2 * eta / sigma ** 2
        return {"sigma": sigma, "eta": eta, "lam": lam}

    def features(self, p):
        return [
            (math.log(p["sigma"]) - math.log(0.02)) / math.log(2),
            (math.log(p["eta"]) - math.log(math.sqrt(1e-5))) / math.log(math.sqrt(10)),
            (math.log(p["lam"]) - 3.17) / 5.03,
        ]

    def decode(self, a):
        return schedule_weights(a[:, 0], self.H)

    def _quad(self, p):
        N, tau = self.H, self.T / self.H
        L = np.tril(np.ones((N - 1, N)))
        c = p["lam"] * p["sigma"] ** 2 * tau
        Q = p["eta"] / tau * np.eye(N) + c * L.T @ L
        b = c * L.T @ np.ones(N - 1)
        const = c * (N - 1)
        return Q, b, const

    def objective(self, n, ctx):
        Q, b, const = self._quad(ctx["p"])
        return float((n @ Q @ n - 2 * b @ n + const) * 1e4)  # bps of notional

    def oracle(self, ctx):
        Q, b, _ = self._quad(ctx["p"])
        n = kkt_min(Q, b, np.ones((1, self.H)), np.ones(1))
        return n


# ---------------------------------------------------------------------------
# T2 Obizhaeva-Wang transient impact (propagator) liquidation
# ---------------------------------------------------------------------------
class OWTransient(Task):
    name = "ow_transient"
    H, A, d = 10, 1, 3
    feature_names = ["log_halflife", "log_temp_ratio", "log_lambda"]
    T = 1.0
    kappa = 5e-3
    sigma = 0.02

    def params_from_u(self, u):
        return {
            "halflife": lerp_log(u[0], 0.05, 3.0),
            "temp_ratio": lerp_log(u[1], 0.01, 0.3),
            "lam": lerp_log(u[2], 0.005, 1.0),
        }

    def features(self, p):
        return [
            (math.log(p["halflife"]) - math.log(math.sqrt(0.15))) / math.log(math.sqrt(60)),
            (math.log(p["temp_ratio"]) - math.log(math.sqrt(0.003))) / math.log(math.sqrt(30)),
            (math.log(p["lam"]) - math.log(math.sqrt(0.005))) / math.log(math.sqrt(200)),
        ]

    def decode(self, a):
        return schedule_weights(a[:, 0], self.H)

    def _quad(self, p):
        N, tau = self.H, self.T / self.H
        rho = math.log(2) / p["halflife"]
        t = np.arange(N) * tau
        G = self.kappa * np.exp(-rho * np.abs(t[:, None] - t[None, :]))
        eta = p["temp_ratio"] * self.kappa * tau
        L = np.tril(np.ones((N - 1, N)))
        c = p["lam"] * self.sigma ** 2 * tau
        Q = 0.5 * G + eta / tau * np.eye(N) + c * L.T @ L
        b = c * L.T @ np.ones(N - 1)
        return Q, b, c * (N - 1)

    def objective(self, n, ctx):
        Q, b, const = self._quad(ctx["p"])
        return float((n @ Q @ n - 2 * b @ n + const) * 1e4)

    def oracle(self, ctx):
        Q, b, _ = self._quad(ctx["p"])
        return kkt_min(Q, b, np.ones((1, self.H)), np.ones(1))


# ---------------------------------------------------------------------------
# T3 Two-asset portfolio transition (sell A, buy B), coordinated legs
# ---------------------------------------------------------------------------
class TwoAssetTransition(Task):
    name = "two_asset_transition"
    H, A, d = 8, 2, 4
    feature_names = ["rho", "log_eta_ratio", "log_lambda", "log_sigma_ratio"]
    T = 1.0
    sigA = 0.02
    etaA = 3e-3

    def params_from_u(self, u):
        return {
            "rho": lerp(u[0], -0.3, 0.9),
            "eta_ratio": lerp_log(u[1], 0.25, 4.0),
            "lam": lerp_log(u[2], 2.0, 200.0),
            "sig_ratio": lerp_log(u[3], 0.5, 2.0),
        }

    def features(self, p):
        return [
            (p["rho"] - 0.3) / 0.6,
            math.log(p["eta_ratio"]) / math.log(4.0),
            (math.log(p["lam"]) - math.log(20.0)) / math.log(10.0),
            math.log(p["sig_ratio"]) / math.log(2.0),
        ]

    x0 = np.array([1.0, -1.0])

    def decode(self, a):
        nA = schedule_weights(a[:, 0], self.H) * self.x0[0]
        nB = schedule_weights(a[:, 1], self.H) * self.x0[1]
        return np.concatenate([nA, nB])

    def _quad(self, p):
        N, tau = self.H, self.T / self.H
        sA, sB = self.sigA, self.sigA * p["sig_ratio"]
        S = np.array([[sA ** 2, p["rho"] * sA * sB], [p["rho"] * sA * sB, sB ** 2]])
        L = np.tril(np.ones((N - 1, N)))
        # x = c + M n, x stacked [xA(1..N-1); xB(1..N-1)]
        M = np.block([[-L, np.zeros_like(L)], [np.zeros_like(L), -L]])
        cvec = np.concatenate([np.full(N - 1, self.x0[0]), np.full(N - 1, self.x0[1])])
        SS = np.kron(S, np.eye(N - 1))
        lamtau = p["lam"] * tau
        D = np.diag(np.concatenate([np.full(N, self.etaA / tau),
                                    np.full(N, self.etaA * p["eta_ratio"] / tau)]))
        Q = D + lamtau * M.T @ SS @ M
        b = -lamtau * M.T @ SS @ cvec
        const = lamtau * cvec @ SS @ cvec
        return Q, b, const

    def objective(self, n, ctx):
        Q, b, const = self._quad(ctx["p"])
        return float((n @ Q @ n - 2 * b @ n + const) * 1e4)

    def oracle(self, ctx):
        Q, b, _ = self._quad(ctx["p"])
        N = self.H
        Aeq = np.zeros((2, 2 * N))
        Aeq[0, :N] = 1
        Aeq[1, N:] = 1
        return kkt_min(Q, b, Aeq, self.x0.copy())

    def oracle_to_actions(self, n):
        N = self.H
        return np.stack([weights_to_actions(n[:N] / self.x0[0]),
                         weights_to_actions(n[N:] / self.x0[1])], axis=1)


# ---------------------------------------------------------------------------
# T4 VWAP execution with day-type volume-profile context
# ---------------------------------------------------------------------------
class VWAPProfile(Task):
    name = "vwap_profile"
    H, A, d = 13, 1, 4
    feature_names = ["open_share", "close_share", "u_curvature", "log_size"]
    M = 400
    sig_day = 0.015
    eta = 0.1   # impact (fraction) per unit participation
    lam = 0.02  # per bps^2
    noise = 0.35

    def params_from_u(self, u):
        return {
            "open": lerp(u[0], 0.03, 0.15),
            "close": lerp(u[1], 0.04, 0.45),
            "curv": lerp(u[2], 0.3, 3.0),
            "size": lerp_log(u[3], 0.01, 0.2),
        }

    def features(self, p):
        return [(p["open"] - 0.09) / 0.06, (p["close"] - 0.245) / 0.205,
                (p["curv"] - 1.65) / 1.35, (math.log(p["size"]) - math.log(math.sqrt(0.002))) / math.log(math.sqrt(20))]

    def profile(self, p):
        N = self.H
        k = np.arange(N)
        U = 1 + p["curv"] * ((k - (N - 1) / 2) / ((N - 1) / 2)) ** 2
        U = U / U.sum()
        prof = (1 - p["open"] - p["close"]) * U
        prof[0] += p["open"]
        prof[-1] += p["close"]
        return prof

    def decode(self, a):
        return schedule_weights(a[:, 0], self.H)

    def naive_actions(self, ctx=None):
        avg = self.profile(self.params_from_u(np.zeros(self.d)))
        return weights_to_actions(avg)[:, None]

    def _moments(self, ctx):
        cache = self.__dict__.setdefault("_mcache", {})
        key = ctx["cid"] + str(np.round(ctx["u"], 9).tolist())
        if key in cache:
            return cache[key]
        p = ctx["p"]
        rng = np.random.default_rng(cid_seed(ctx["cid"]) ^ 0x5157)
        N, M = self.H, self.M
        prof = self.profile(p)
        z = rng.standard_normal((M, N))
        v = prof * np.exp(self.noise * z - 0.5 * self.noise ** 2)
        v = v / v.sum(axis=1, keepdims=True)
        r = rng.standard_normal((M, N)) * self.sig_day / math.sqrt(N)
        P = np.exp(np.cumsum(r, axis=1))  # price per bin, P0 = 1
        # tracking error TE_s = (n - v_s)' P_s ; E[TE^2] = n'An - 2b'n + c
        Aq = (P[:, :, None] * P[:, None, :]).mean(axis=0)
        vP = (v * P).sum(axis=1)
        bq = (P * vP[:, None]).mean(axis=0)
        cq = (vP ** 2).mean()
        inv_v = (1.0 / v).mean(axis=0)
        cache[key] = (Aq, bq, cq, inv_v)
        return cache[key]

    def _quad(self, ctx):
        Aq, bq, cq, inv_v = self._moments(ctx)
        p = ctx["p"]
        s = 1e4
        Q = self.lam * s * s * Aq + s * self.eta * p["size"] * np.diag(inv_v)
        b = self.lam * s * s * bq
        return Q, b, self.lam * s * s * cq

    def objective(self, n, ctx):
        Q, b, const = self._quad(ctx)
        return float(n @ Q @ n - 2 * b @ n + const)

    def oracle(self, ctx):
        Q, b, _ = self._quad(ctx)
        n = kkt_min(Q, b, np.ones((1, self.H)), np.ones(1))
        return np.maximum(n, 1e-9) / np.maximum(n, 1e-9).sum()


# ---------------------------------------------------------------------------
# T5 Market making: Avellaneda-Stoikov quoting schedule (spread, skew)
# ---------------------------------------------------------------------------
class MarketMaking(Task):
    name = "market_making"
    H, A, d = 8, 2, 4
    feature_names = ["log_sigma", "log_A", "log_k", "log_gamma"]
    kind = "utility"
    substeps = 25
    M = 300
    liq_pen = 0.05
    pad = "repeat"

    def params_from_u(self, u):
        return {"sigma": lerp_log(u[0], 1.0, 3.0), "A": lerp_log(u[1], 70, 280),
                "k": lerp_log(u[2], 0.75, 3.0), "gamma": lerp_log(u[3], 0.02, 0.3)}

    def features(self, p):
        return [math.log(p["sigma"] / math.sqrt(3.0)) / math.log(math.sqrt(3.0)),
                math.log(p["A"] / 140) / math.log(2),
                math.log(p["k"] / 1.5) / math.log(2),
                math.log(p["gamma"] / math.sqrt(0.006)) / math.log(math.sqrt(15))]

    def decode(self, a):
        H = self.H
        if len(a) < H:
            a = np.vstack([a, np.repeat(a[-1:], H - len(a), axis=0)])
        delta = np.exp(np.log(0.15) + (a[:, 0] + 1) / 2 * (np.log(6.0) - np.log(0.15)))
        skew = 3.0 * ((a[:, 1] + 1) / 2) ** 2
        return np.stack([delta, skew], axis=1)

    def _paths(self, ctx):
        cache = self.__dict__.setdefault("_pcache", {})
        key = ctx["cid"]
        if key not in cache:
            rng = np.random.default_rng(cid_seed(ctx["cid"]) ^ 0x3A3A)
            n = self.H * self.substeps
            cache[key] = (rng.standard_normal((n, self.M)), rng.random((n, self.M)), rng.random((n, self.M)))
        return cache[key]

    def objective(self, sched, ctx):
        p = ctx["p"]
        Z, Ub, Ua = self._paths(ctx)
        n = self.H * self.substeps
        dt = 1.0 / n
        S = np.full(self.M, 100.0)
        q = np.zeros(self.M)
        cash = np.zeros(self.M)
        sq = p["sigma"] * math.sqrt(dt)
        for t in range(n):
            delta, skew = sched[t // self.substeps]
            db = np.maximum(delta + skew * q, 0.0)
            da = np.maximum(delta - skew * q, 0.0)
            hb = Ub[t] < np.minimum(p["A"] * np.exp(-p["k"] * db) * dt, 1.0)
            ha = Ua[t] < np.minimum(p["A"] * np.exp(-p["k"] * da) * dt, 1.0)
            cash += np.where(hb, -(S - db), 0.0) + np.where(ha, S + da, 0.0)
            q += hb.astype(float) - ha.astype(float)
            S = S + sq * Z[t]
        pnl = cash + q * S - self.liq_pen * q ** 2
        return float(pnl.mean() - 0.5 * p["gamma"] * pnl.var())

    def naive_actions(self, ctx=None):
        return None  # context-dependent; handled in naive_obj

    def naive_obj(self, ctx):
        cache = self.__dict__.setdefault("_ncache", {})
        if ctx["cid"] not in cache:
            p = ctx["p"]
            sched = np.tile([1.0 / p["k"], 0.0], (self.H, 1))
            cache[ctx["cid"]] = self.objective(sched, ctx)
        return cache[ctx["cid"]]

    def as_schedule(self, p):
        tm = 1 - (np.arange(self.H) + 0.5) / self.H
        g, s2 = p["gamma"], p["sigma"] ** 2
        delta = 0.5 * g * s2 * tm + (1 / g) * np.log(1 + g / p["k"])
        skew = g * s2 * tm
        return np.stack([delta, skew], axis=1)

    def sched_to_actions(self, sched):
        d = np.clip(sched[:, 0], 0.15, 6.0)
        a0 = 2 * (np.log(d) - np.log(0.15)) / (np.log(6.0) - np.log(0.15)) - 1
        a1 = 2 * np.sqrt(np.clip(sched[:, 1], 0, 3.0) / 3.0) - 1
        return np.clip(np.stack([a0, a1], axis=1), -1, 1)


# ---------------------------------------------------------------------------
# T6 Lifecycle glide path with labour income (CRRA certainty equivalent)
# ---------------------------------------------------------------------------
class GlidePath(Task):
    name = "glide_path"
    H, A, d = 10, 1, 4
    feature_names = ["log_gamma", "premium", "vol", "w0"]
    kind = "utility"
    years = 40
    r = 0.02
    g = 0.01
    M = 4000
    pad = "repeat"

    def params_from_u(self, u):
        return {"gamma": lerp_log(u[0], 2.0, 10.0), "ep": lerp(u[1], 0.02, 0.07),
                "vol": lerp(u[2], 0.12, 0.24), "w0": lerp(u[3], 0.0, 10.0)}

    def features(self, p):
        return [math.log(p["gamma"] / math.sqrt(20)) / math.log(math.sqrt(5)),
                (p["ep"] - 0.045) / 0.025, (p["vol"] - 0.18) / 0.06, (p["w0"] - 5) / 5]

    def decode(self, a):
        k = a[:, 0]
        if len(k) < self.H:
            k = np.concatenate([k, np.repeat(k[-1:], self.H - len(k))])
        return (k + 1) / 2  # equity weight per 4-year knot, in [0, 1]

    def _z(self, ctx):
        cache = self.__dict__.setdefault("_zcache", {})
        if ctx["cid"] not in cache:
            rng = np.random.default_rng(cid_seed(ctx["cid"]) ^ 0x6161)
            cache[ctx["cid"]] = rng.standard_normal((self.years, self.M))
        return cache[ctx["cid"]]

    def objective(self, pi_knots, ctx):
        p = ctx["p"]
        Z = self._z(ctx)
        W = np.full(self.M, p["w0"])
        per = self.years // self.H
        mu_log = math.log(1 + self.r) + p["ep"] - 0.5 * p["vol"] ** 2
        for t in range(self.years):
            pi = pi_knots[t // per]
            R = np.exp(mu_log + p["vol"] * Z[t]) - 1
            W = (W + (1 + self.g) ** t) * (1 + self.r + pi * (R - self.r))
        gm = p["gamma"]
        Wn = W / np.median(W)
        eu = np.mean(Wn ** (1 - gm))
        ce = eu ** (1 / (1 - gm)) * np.median(W)
        return float(ce)

    def naive_actions(self, ctx=None):
        return np.full((self.H, 1), 0.2)  # constant 60% equity

    def oracle(self, ctx):
        from scipy.optimize import minimize
        f = lambda x: -self.objective(np.clip(x, 0, 1), ctx)
        best = None
        for x0 in (np.full(self.H, 0.6), np.linspace(0.9, 0.3, self.H), np.full(self.H, 0.95)):
            r = minimize(f, x0, method="L-BFGS-B", bounds=[(0, 1)] * self.H,
                         options={"maxiter": 200})
            if best is None or r.fun < best.fun:
                best = r
        return np.clip(best.x, 0, 1)


# ---------------------------------------------------------------------------
# T7 Option delta-hedging band schedule with transaction costs
# ---------------------------------------------------------------------------
class HedgeBands(Task):
    name = "hedge_bands"
    H, A, d = 8, 1, 4
    feature_names = ["log_vol", "log_cost", "moneyness", "log_gamma"]
    days = 40
    M = 1500
    pad = "repeat"

    def params_from_u(self, u):
        return {"vol": lerp_log(u[0], 0.15, 0.5), "cost": lerp_log(u[1], 5e-4, 4e-3),
                "K": lerp(u[2], 90.0, 110.0), "gamma": lerp_log(u[3], 0.02, 0.5)}

    def features(self, p):
        return [math.log(p["vol"] / math.sqrt(0.075)) / math.log(math.sqrt(10 / 3)),
                math.log(p["cost"] / math.sqrt(2e-6)) / math.log(math.sqrt(8)),
                (p["K"] - 100) / 10, math.log(p["gamma"] / 0.1) / math.log(5)]

    def decode(self, a):
        k = a[:, 0]
        if len(k) < self.H:
            k = np.concatenate([k, np.repeat(k[-1:], self.H - len(k))])
        return 0.3 * ((k + 1) / 2) ** 1.5  # band half-width in delta units

    @staticmethod
    def _ncdf(x):
        from scipy.special import ndtr
        return ndtr(x)

    def _paths(self, ctx):
        cache = self.__dict__.setdefault("_hcache", {})
        if ctx["cid"] not in cache:
            p = ctx["p"]
            rng = np.random.default_rng(cid_seed(ctx["cid"]) ^ 0x7E7E)
            dt = 1 / 252
            z = rng.standard_normal((self.days, self.M))
            logS = np.vstack([np.zeros(self.M), np.cumsum(-0.5 * p["vol"] ** 2 * dt + p["vol"] * math.sqrt(dt) * z, axis=0)])
            S = 100 * np.exp(logS)
            tau = (self.days - np.arange(self.days + 1)) / 252
            delta = np.empty_like(S)
            for t in range(self.days):
                d1 = (np.log(S[t] / p["K"]) + 0.5 * p["vol"] ** 2 * tau[t]) / (p["vol"] * math.sqrt(tau[t]))
                delta[t] = self._ncdf(d1)
            delta[-1] = (S[-1] > p["K"]).astype(float)
            d1 = (math.log(100 / p["K"]) + 0.5 * p["vol"] ** 2 * tau[0]) / (p["vol"] * math.sqrt(tau[0]))
            d2 = d1 - p["vol"] * math.sqrt(tau[0])
            prem = 100 * self._ncdf(d1) - p["K"] * self._ncdf(d2)
            cache[ctx["cid"]] = (S, delta, float(prem))
        return cache[ctx["cid"]]

    def objective(self, bands, ctx):
        p = ctx["p"]
        S, delta, prem = self._paths(ctx)
        per = self.days // self.H
        h = delta[0].copy()
        cost = p["cost"] * h * S[0]
        gains = np.zeros(self.M)
        for t in range(1, self.days + 1):
            gains += h * (S[t] - S[t - 1])
            if t == self.days:
                break
            b = bands[min(t // per, self.H - 1)]
            dev = h - delta[t]
            target = np.where(dev > b, delta[t] + b, np.where(dev < -b, delta[t] - b, h))
            cost += p["cost"] * np.abs(target - h) * S[t]
            h = target
        cost += p["cost"] * np.abs(h) * S[-1]  # unwind
        payoff = np.maximum(S[-1] - p["K"], 0)
        pnl = prem + gains - cost - payoff
        return float(-pnl.mean() + 0.5 * p["gamma"] * pnl.var())  # loss (lower better)

    def ww_schedule(self, p):
        """Whalley-Wilmott band from ATM-forward gamma at bucket midpoints."""
        out = []
        per = self.days // self.H
        for k in range(self.H):
            tau = (self.days - (k + 0.5) * per) / 252
            d1 = (math.log(100 / p["K"]) + 0.5 * p["vol"] ** 2 * tau) / (p["vol"] * math.sqrt(tau))
            gam = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) / (100 * p["vol"] * math.sqrt(tau))
            out.append((1.5 * p["cost"] * 100 * gam ** 2 / p["gamma"]) ** (1 / 3))
        return np.array(out)

    def bands_to_actions(self, b):
        return np.clip(2 * (np.clip(b, 0, 0.3) / 0.3) ** (1 / 1.5) - 1, -1, 1)[:, None]

    def naive_actions(self, ctx=None):
        return np.full((self.H, 1), -1.0)  # zero band: rebalance to delta daily


TASKS = {c.name: c for c in [ACLiquidation, OWTransient, TwoAssetTransition, VWAPProfile,
                             MarketMaking, GlidePath, HedgeBands]}


# ---------------------------------------------------------------------------
# T9 Three-leg portfolio switch (sell A, buy B and C): three coordinated controls
# ---------------------------------------------------------------------------
class ThreeAssetTransition(Task):
    name = "three_asset_transition"
    H, A, d = 8, 3, 4
    feature_names = ["rho_ab", "rho_bc", "log_eta_ratio", "log_lambda"]
    T = 1.0
    sig = 0.02
    etaA = 3e-3
    x0 = np.array([1.0, -0.5, -0.5])

    def params_from_u(self, u):
        return {"rho_ab": lerp(u[0], -0.2, 0.9), "rho_bc": lerp(u[1], -0.2, 0.9),
                "eta_ratio": lerp_log(u[2], 0.25, 4.0), "lam": lerp_log(u[3], 2.0, 200.0)}

    def features(self, p):
        return [(p["rho_ab"] - 0.35) / 0.55, (p["rho_bc"] - 0.35) / 0.55,
                math.log(p["eta_ratio"]) / math.log(4.0), (math.log(p["lam"]) - math.log(20.0)) / math.log(10.0)]

    def decode(self, a):
        return np.concatenate([schedule_weights(a[:, i], self.H) * self.x0[i] for i in range(3)])

    def _quad(self, p):
        N, tau = self.H, self.T / self.H
        rab, rbc = p["rho_ab"], p["rho_bc"]
        rac = rab * rbc  # implied A-C correlation keeps the matrix valid
        C = np.array([[1, rab, rac], [rab, 1, rbc], [rac, rbc, 1]])
        S = self.sig ** 2 * C
        L = np.tril(np.ones((N - 1, N)))
        Z = np.zeros_like(L)
        M = np.block([[-L, Z, Z], [Z, -L, Z], [Z, Z, -L]])
        cvec = np.concatenate([np.full(N - 1, x) for x in self.x0])
        SS = np.kron(S, np.eye(N - 1))
        lamtau = p["lam"] * tau
        eta = [self.etaA, self.etaA * p["eta_ratio"], self.etaA * math.sqrt(p["eta_ratio"])]
        D = np.diag(np.concatenate([np.full(N, e / tau) for e in eta]))
        Q = D + lamtau * M.T @ SS @ M
        b = -lamtau * M.T @ SS @ cvec
        return Q, b, lamtau * cvec @ SS @ cvec

    def objective(self, n, ctx):
        Q, b, const = self._quad(ctx["p"])
        return float((n @ Q @ n - 2 * b @ n + const) * 1e4)

    def oracle(self, ctx):
        Q, b, _ = self._quad(ctx["p"])
        N = self.H
        Aeq = np.zeros((3, 3 * N))
        for i in range(3):
            Aeq[i, i * N:(i + 1) * N] = 1
        return kkt_min(Q, b, Aeq, self.x0.copy())

    def oracle_to_actions(self, n):
        N = self.H
        return np.stack([weights_to_actions(n[i * N:(i + 1) * N] / self.x0[i]) for i in range(3)], axis=1)


TASKS[ThreeAssetTransition.name] = ThreeAssetTransition


# ---------------------------------------------------------------------------
# T10 AC liquidation with a pre-trade alpha signal (the optimal shape flips with the signal)
# ---------------------------------------------------------------------------
class AlphaLiquidation(ACLiquidation):
    """cost (bps) = eta/tau sum n^2 + lam sig^2 tau sum x^2 - mu tau sum x, x = holdings after each slot.
    Signal a = mu / (2 lam sig^2) is the drift-justified holding level: a < 0 sells faster, a > ~0.6 back-loads.
    Score = (cost_TWAP - cost) / cost_TWAP(mu = 0), so the scale never collapses to ~0."""
    name = "alpha_liquidation"
    d = 4
    feature_names = ["log_sigma", "log_eta", "log_lambda", "alpha_hold"]

    def params_from_u(self, u):
        p = super().params_from_u(u[:3])
        p["a"] = lerp(u[3], -0.75, 1.5)
        p["mu"] = p["a"] * 2 * p["lam"] * p["sigma"] ** 2
        return p

    def features(self, p):
        return super().features(p) + [(p["a"] - 0.375) / 1.125]

    def _qlin(self, p):
        Q, b, const = self._quad(p)
        N, tau = self.H, self.T / self.H
        L = np.tril(np.ones((N - 1, N)))
        b2 = b - 0.5 * p["mu"] * tau * L.T @ np.ones(N - 1)
        return Q, b2, const - p["mu"] * tau * (N - 1)

    def objective(self, n, ctx):
        Q, b, c = self._qlin(ctx["p"])
        return float((n @ Q @ n - 2 * b @ n + c) * 1e4)

    def scale(self, ctx):
        Q, b, c = self._quad(ctx["p"])
        n = np.full(self.H, 1 / self.H)
        return float((n @ Q @ n - 2 * b @ n + c) * 1e4)

    def score(self, actions, ctx):
        return (self.naive_obj(ctx) - self.obj_actions(actions, ctx)) / self.scale(ctx)

    def qp(self, ctx):
        Q, b, _ = self._qlin(ctx["p"])
        N = self.H
        r = minimize(lambda n: n @ Q @ n - 2 * b @ n, np.full(N, 1 / N), jac=lambda n: 2 * Q @ n - 2 * b,
                     method="SLSQP", bounds=[(0, 1)] * N,
                     constraints=[{"type": "eq", "fun": lambda n: n.sum() - 1, "jac": lambda n: np.ones(N)}],
                     options={"ftol": 1e-14, "maxiter": 500})
        return np.maximum(r.x, 0) / np.maximum(r.x, 0).sum()

    def best_actions(self, ctx):
        n = self.qp(ctx)
        a0 = weights_to_actions(np.maximum(n, 1e-12))
        r = minimize(lambda a: -self.score(a[:, None], ctx), a0, method="L-BFGS-B", bounds=[(-1, 1)] * self.H)
        a = r.x if -r.fun > self.score(a0[:, None], ctx) else a0
        return a[:, None], n


TASKS[AlphaLiquidation.name] = AlphaLiquidation

# Single-asset liquidation tasks report remaining inventory as acquisition progress; the rest report time.
INVENTORY_PROGRESS = {"ac_liquidation", "ow_transient", "vwap_profile", "alpha_liquidation"}


def get_task(name):
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; kept tasks: {sorted(TASKS)}")
    return TASKS[name]()


def standard_contexts(task):
    """The context sets used in every prod run: train 48 (seed 101), support 24 (seed 202),
    held-out eval 32 (seed 303) and the nominal context (all features 0 in parameter space)."""
    return {"train": task.make_contexts(48, 101, "tr"), "support": task.make_contexts(24, 202, "sp"),
            "eval": task.make_contexts(32, 303, "ev"), "nominal": task.nominal()}


def context_from_u(task, u, cid="custom"):
    """A context from normalised parameters u in [-1, 1]^d (0 = nominal)."""
    u = np.asarray(u, float)
    p = task.params_from_u(u)
    return {"cid": cid, "u": u.tolist(), "p": p, "feat": task.features(p)}


def trace_states(task, actions, ctx):
    """T + 1 state rows for an episode_credit observation: [progress] + context features."""
    a = np.asarray(actions, float)
    rows = len(a)
    if task.name in INVENTORY_PROGRESS:
        n = task.decode(as_rows(a, task.H, task.A))
        rem = 1 - np.cumsum(n)[:rows]
        prog = [1.0] + [float(max(r, 0.0)) for r in rem]
    else:
        prog = [1.0 - k / rows for k in range(rows + 1)]
    return [[p] + list(ctx["feat"]) for p in prog]
