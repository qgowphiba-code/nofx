"""Stand-alone copy of SMPL's BeerFMTEnv plant (smpl/envs/beerfmtenv.py, SMPL 1.5.x) for local gates and baselines.

Equations, constants, step integration and termination rules are transcribed from the SMPL source so that the Machina
executor and the local comparators run the identical plant without importing gym. `check_against_smpl.py` verifies this
copy against the installed SMPL environment trajectory-for-trajectory.
Episode: hourly steps, at most 200; action = temperature in [9, 16] C held for one hour.
Return (SMPL reward_function_standard, dense): -1 per step until done; +200 on early submission (finished); -200 on
leaving the observation box; an unfinished batch simply ends after 200 steps (return -200, see run()).
Quality for Machina = (return + 400) / 600, clipped to [0, 1]: 0.333 for a batch that never finishes, 0.867 for a finish at 81 h.
"""
import math
import numpy as np
from scipy.integrate import odeint

MAX_STEPS = 200
INIT_SUGAR = 130
BIOMASS_end_threshold = 0.5
BIOMASS_end_change_threshold = 0.01
SUGAR_end_threshold = 0.5
ZERO_C_IN_K = 273.15
MAX_OBS = np.array([15, 15, 15, 150, 150, 10, 10, MAX_STEPS], dtype=float)
MIN_OBS = np.zeros(8)
T_MIN, T_MAX = 9.0, 16.0
ERROR_REWARD = -float(MAX_STEPS)
NOMINAL_X0 = np.array([0, 2, 2, INIT_SUGAR, 0, 0, 0, 0], dtype=float)


def beer_ode(points, t, sets):
    X_A, X_L, X_D, S, EtOH, DY, EA = points
    S0, T = sets
    k_x = 0.5 * S0
    u_x0 = math.exp(108.31 - 31934.09 / T)
    Y_EA = math.exp(89.92 - 26589 / T)
    u_s0 = math.exp(-41.92 + 11654.64 / T)
    u_L = math.exp(30.72 - 9501.54 / T)
    u_DY = 0.000127672
    u_AB = 0.00113864
    u_DT = math.exp(130.16 - 38313 / T)
    u_SD0 = math.exp(33.82 - 10033.28 / T)
    u_e0 = math.exp(3.27 - 1267.24 / T)
    k_s = math.exp(-119.63 + 34203.95 / T)
    u_x = u_x0 * S / (k_x + EtOH)
    u_SD = u_SD0 * 0.5 * S0 / (0.5 * S0 + EtOH)
    u_s = u_s0 * S / (k_s + S)
    u_e = u_e0 * S / (k_s + S)
    f = 1 - EtOH / (0.5 * S0)
    dXAt = u_x * X_A - u_DT * X_A + u_L * X_L
    dXLt = -u_L * X_L
    dXDt = -u_SD * X_D + u_DT * X_A
    dSt = -u_s * X_A
    dEtOHt = f * u_e * X_A
    dDYt = u_DY * S * X_A - u_AB * DY * EtOH
    dEAt = Y_EA * u_x * X_A
    return [dXAt, dXLt, dXDt, dSt, dEtOHt, dDYt, dEAt]


def sample_context(rng):
    """SMPL sample_initial_state: X_L, X_D around 2 (+-10 %), sugar around 130 (+-10 %), uniform."""
    return np.array([0, rng.uniform(1.8, 2.2), rng.uniform(1.8, 2.2), rng.uniform(INIT_SUGAR * 0.9, INIT_SUGAR * 1.1), 0, 0, 0, 0])


def run(temps_by_hour, x0=NOMINAL_X0, record=False):
    """Simulate with the per-hour temperature schedule (len >= MAX_STEPS; extra entries ignored). Returns dict."""
    obs = np.array(x0, dtype=np.float32).astype(float)
    prev_biomass = obs[1] + obs[2] + obs[0]  # SMPL initialises prev_biomass from the initial observation in reset()
    total = 0.0
    traj = [obs.copy()] if record else None
    for k in range(MAX_STEPS):
        T = float(np.clip(temps_by_hour[k], T_MIN, T_MAX))
        t = np.arange(0 + k, 1 + k, BIOMASS_end_change_threshold)
        sol = odeint(beer_ode, tuple(obs[:7]), t, args=([INIT_SUGAR, T + ZERO_C_IN_K],))
        obs = np.array(list(sol[-1, :]) + [k + 1], dtype=np.float32).astype(float)
        if record:
            traj.append(obs.copy())
        # reward_function_standard
        beyond = bool(np.any(obs > MAX_OBS) or np.any(obs < MIN_OBS))
        if beyond:
            reward = ERROR_REWARD
            total += reward
            return {'return': total, 'steps': k + 1, 'finished': False, 'reason': 'beyond_box', 'traj': traj}
        step_count = k  # SMPL evaluates done with the pre-increment step_count
        X_A, X_L, X_D, S, EtOH, DY, EA, _ = obs
        biomass = X_A + X_L + X_D
        finished = (biomass < BIOMASS_end_threshold or abs(biomass - prev_biomass) < BIOMASS_end_change_threshold) and S < SUGAR_end_threshold and EtOH > 50.0
        # SMPL's own -200 timeout branch is unreachable under its 200-step limit (step_count 199 < 200 at the last step),
        # so an unfinished batch ends after 200 steps of -1 (return -200), exactly as the SMPL environment behaves.
        timeout = (step_count + 1) >= MAX_STEPS
        if finished:
            reward = -ERROR_REWARD
        else:
            reward = -1.0
        prev_biomass = biomass
        total += reward
        if finished:
            return {'return': total, 'steps': k + 1, 'finished': True, 'reason': 'finished', 'traj': traj}
        if timeout:
            return {'return': total, 'steps': k + 1, 'finished': False, 'reason': 'timeout', 'traj': traj}
    return {'return': total, 'steps': MAX_STEPS, 'finished': False, 'reason': 'timeout', 'traj': traj}


H = 20            # Machina stages
STAGE_HOURS = MAX_STEPS // H


def profile_from_u(u):
    """u in [-1, 1]^H -> hourly temperatures (piecewise constant, STAGE_HOURS per stage)."""
    temps = [T_MIN + (float(np.clip(v, -1, 1)) + 1) / 2 * (T_MAX - T_MIN) for v in u]
    return np.repeat(np.array(temps), STAGE_HOURS)


def quality(ret):
    return float(np.clip((ret + 400.0) / 600.0, 0.0, 1.0))
