"""SMPL PenSimEnv (IndPenSim via PenSimPy) as a batch executor for Machina and the local comparators.

The SMPL benchmark code is imported from a read-only clone (SMPL_REPO) with two import shims (SHIMS): `mzutils` (import
names only) and `fastodeint` (PenSimPy's compiled integrator is not on PyPI; the shim runs the simulator's own pure-Python
ODE, and with fast=False it is never called). Everything else is SMPL/PenSimPy code unchanged.

Executor PEN-1: H = 23 stages of 10 h (50 steps of 12 min); 6 piecewise-constant controls in SMPL's action order
(discharge, sugar feed Fs, soybean oil Foil, aeration Fg, back pressure, water Fw), command u in [-1, 1] mapped linearly
onto SMPL's action bounds; PAA follows SMPL's default recipe (as SMPL itself does). Outcome = batch yield (sum of SMPL's
per-step yield reward, kg penicillin); quality = yield / 4500, clipped to [0, 1] (the published GP-EI dataset tops out at 4080).
Context (measured at reset, PenSimPy's X0 draw for the batch): initial S, DO2, X, V, Wt, pH, T, PAA, scaled to about [0, 1]
over their +-3 sigma ranges. Hidden per-batch variation (growth-rate parameters, kLa, PAA concentration, disturbance
profiles) stays hidden, as in the plant.
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
np.float = float; np.bool8 = np.bool_
SCRATCH = os.environ.get('ADAPT1_SCRATCH', '/tmp/claude-0/-home-user-nofx/7c289125-8882-5b5c-9bcd-654fe7300b9f/scratchpad')
SHIMS, SMPL_REPO = os.path.join(SCRATCH, 'shims'), os.path.join(SCRATCH, 'repos', 'smpl')
for p in (SHIMS, SMPL_REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
from smpl.envs.pensimenv import PenSimEnvGym, NUM_STEPS, STEP_IN_MINUTES
from pensimpy.examples.recipe import Recipe, RecipeCombo
from pensimpy.data.constants import FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA
from pensimpy.data.constants import (FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE, PRESS_DEFAULT_PROFILE,
                                     DISCHARGE_DEFAULT_PROFILE, WATER_DEFAULT_PROFILE, PAA_DEFAULT_PROFILE)

H = 23
STAGE_STEPS = NUM_STEPS // H          # 50 steps = 10 h
assert STAGE_STEPS * H == NUM_STEPS
ADIM = 6
Y_SCALE = 4500.0
CTX_NAMES = ['S', 'DO2', 'X', 'V', 'Wt', 'pH', 'T', 'PAA']
CTX_MEAN = np.array([1.0, 15.0, 0.5, 5.8e4, 6.2e4, 6.5, 297.0, 1400.0])
CTX_SD = np.array([0.1, 0.5, 0.11, 500.0, 500.0, 0.1, 0.5, 50.0])    # X: 0.5 +- 0.05 (initial_conds) then +- 0.1


def make_env(seed=0):
    recipe = RecipeCombo(recipe_dict={FS: Recipe(FS_DEFAULT_PROFILE, FS), FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
                                      FG: Recipe(FG_DEFAULT_PROFILE, FG), PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
                                      DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE), WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
                                      PAA: Recipe(PAA_DEFAULT_PROFILE, PAA)})
    return PenSimEnvGym(recipe_combo=recipe, normalize=False, dense_reward=True, fast=False, random_seed=seed)


ENV = None


def env():
    global ENV
    if ENV is None:
        ENV = make_env()
    return ENV


def context_of(e):
    x0 = e.x0
    ph = -np.log10(x0.pH) if x0.pH > 0 else 6.5          # PenSimPy stores pH as the H+ concentration
    raw = np.array([x0.S, x0.DO2, x0.X, x0.V, x0.Wt, ph, x0.T, x0.PAA], dtype=float)
    return raw, [float(v) for v in (raw - CTX_MEAN) / (6 * CTX_SD) + 0.5]


_ANCHORS = None


def anchors(e):
    """PEN-2 anchors: the default recipe's stage means in absolute u-space (see default_recipe_u)."""
    global _ANCHORS
    if _ANCHORS is None:
        _ANCHORS = np.array(default_recipe_u(e), dtype=float)
    return _ANCHORS


def actions_from_u(u_row, e, k=None):
    """PEN-2 (used for every method): u = 0 is the default recipe at stage k; u > 0 moves linearly towards SMPL's upper
    action bound, u < 0 towards the lower bound. k=None gives the PEN-1 absolute mapping (u = 0 is mid-range), kept for
    reference only."""
    u = np.clip(np.asarray(u_row, dtype=float), -1, 1)
    if k is not None:
        a0 = np.clip(anchors(e)[k], -1, 1)
        u = np.where(u >= 0, a0 + u * (1 - a0), a0 + u * (a0 + 1))
    return (e.min_actions + (u + 1) / 2 * (e.max_actions - e.min_actions)).astype(np.float32)


def default_recipe_u(e):
    """The SMPL/PenSimPy default recipe expressed as a stage-wise u matrix (mean setpoint within each stage)."""
    rows = []
    for k in range(H):
        acc = np.zeros(ADIM)
        for s in range(STAGE_STEPS):
            v = e.recipe_combo.get_values_dict_at((k * STAGE_STEPS + s) * STEP_IN_MINUTES)
            acc += np.array([v['discharge'], v['Fs'], v['Foil'], v['Fg'], v['pressure'], v['Fw']], dtype=float)
        a = acc / STAGE_STEPS
        rows.append(list((a - e.min_actions) / (e.max_actions - e.min_actions) * 2 - 1))
    return rows


def run(u_matrix, random_seed_ref, record_states=False, e=None):
    """Execute one batch. u_matrix: H x 6 commands in [-1, 1]. Returns dict(yield, quality, steps, states, context, error).
    A fresh SMPL environment (and recipe object) is built for every batch: the recipe's setpoint pointer is not reset by
    SMPL's reset(), and a second batch on the same object fails at its first step (observed: batches alternate ok/failed)."""
    e = make_env()
    e.reset(random_seed_ref=random_seed_ref)
    raw_ctx, ctx = context_of(e)
    tot = 0.0; n = 0; states = []; err = None
    try:
        for k in range(H):
            a = actions_from_u(u_matrix[k], e, k)
            for s in range(STAGE_STEPS):
                obs, r, done, info = e.step(a); tot += r; n += 1
                if done:
                    break
            if record_states:
                states.append([float(v) for v in np.asarray(obs, dtype=float) / e.max_observations])
            if done:
                break
    except Exception as ex:               # SMPL returns error_reward -100 for solver failures; record and treat as failed batch
        err = repr(ex)[:200]
    # quality = SMPL's own batch return shifted into [0, 1]: a batch that fails at its first step scores 0, a completed batch
    # with yield Y scores (Y + 100) / 4600 (= Y / 4500 within 0.3 %), and a batch that fails after producing some penicillin
    # keeps the credit for what it produced (SMPL's -100 failure penalty is included in `tot`).
    return {'yield': float(tot), 'quality': float(np.clip((float(tot) + 100.0) / (Y_SCALE + 100.0), 0, 1)), 'steps': n, 'states': states,
            'context': ctx, 'raw_context': [float(v) for v in raw_ctx], 'error': err}


if __name__ == '__main__':
    e = env()
    t0 = time.time()
    u0 = default_recipe_u(e)
    for s in (1, 2, 3):
        r = run(u0, s, e=e)
        print(f'default recipe seed {s}: yield {r["yield"]:.1f} q {r["quality"]:.3f} steps {r["steps"]} err {r["error"]} ctx {np.round(r["raw_context"], 2)} ({time.time() - t0:.0f}s)', flush=True)
    rng = np.random.default_rng(0)
    for s in (1, 2):
        u = rng.uniform(-1, 1, (H, ADIM))
        r = run(u, s, e=e)
        print(f'random recipe seed {s}: yield {r["yield"]:.1f} steps {r["steps"]} err {r["error"]} ({time.time() - t0:.0f}s)', flush=True)
