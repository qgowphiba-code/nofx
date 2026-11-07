"""ChemGymRL reaction bench (ChemGymRL-v1, Beeler et al. 2023/2024) as a batch executor for Machina and local comparators.

The benchmark code is imported from a read-only clone (CHEMGYM_REPO). Two NumPy alias shims are set for gym 0.26.
Environment: GenWurtzReact-v1 (Wurtz coupling: 1-, 2-, 3-chlorohexane + Na in diethyl ether, 7 possible target products).
One episode = 20 steps; action = 6 values in [0, 1] (temperature change, volume change, four reagent additions); the reward
is terminal only: moles of the target minus moles of the undesired material in the vessel. The target is drawn at reset in the
original environment; here it is pinned per batch so that a recipe can be acquired per target (set_target).
Executor CHEM-1: the 20 x 6 action sequence is the open-loop recipe (the plant is deterministic given the target), u in
[-1, 1] maps linearly onto the action box [0, 1]. Quality = return clipped to [0, 1] for the organic targets; for NaCl the
return is divided by 3 (3 mol Na available).
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
np.bool8 = np.bool_; np.float = float
SCRATCH = os.environ.get('ADAPT1_SCRATCH', '/tmp/claude-0/-home-user-nofx/7c289125-8882-5b5c-9bcd-654fe7300b9f/scratchpad')
CHEMGYM_REPO = os.path.join(SCRATCH, 'repos', 'chemgymrl')
if CHEMGYM_REPO not in sys.path:
    sys.path.insert(0, CHEMGYM_REPO)
import gym, chemistrylab  # noqa: E402

H, ADIM = 20, 6
WURTZ_TARGETS = ["dodecane", "5-methylundecane", "4-ethyldecane", "5,6-dimethyldecane", "4-ethyl-5-methylnonane", "4,5-diethyloctane", "NaCl"]


def make_env(env_id='GenWurtzReact-v1'):
    return gym.make(env_id).unwrapped


def set_target(env, target):
    """Pin the target material (the original env redraws it at reset)."""
    env.reaction.desired = target
    env.target_material = target
    if hasattr(env.reaction, 'target_material'):
        env.reaction.target_material = target


def heuristic_action(target):
    """The benchmark's own hand-written reaction policy (Heuristic_Policies/wurtz_reaction_policy.py): heat, add the
    reagents that form the target, every step."""
    a = np.zeros(ADIM, dtype=np.float32); a[0] = 1.0
    add = {"dodecane": [2, 5], "5-methylundecane": [2, 3, 5], "4-ethyldecane": [2, 4, 5], "5,6-dimethyldecane": [3, 5],
           "4-ethyl-5-methylnonane": [3, 4, 5], "4,5-diethyloctane": [4, 5], "NaCl": [2, 3, 4, 5]}[target]
    for i in add:
        a[i] = 1.0
    return a


def actions_from_u(u_row):
    return ((np.clip(np.asarray(u_row, dtype=float), -1, 1) + 1) / 2).astype(np.float32)


def run(u_matrix, target, env=None, record_states=False, env_id='GenWurtzReact-v1'):
    """Execute one 20-step batch with the open-loop recipe u_matrix (H x 6). Returns dict(return, quality, steps, states)."""
    env = env or make_env(env_id)
    set_target(env, target)
    s = env.reset()
    if isinstance(s, tuple):
        s = s[0]
    set_target(env, target)        # reset redraws the target; pin it again before the first step
    states = []; tot = 0.0; done = False; k = 0
    while not done and k < H:
        out = env.step(actions_from_u(u_matrix[k])); s, r, done = out[0], float(out[1]), bool(out[2]); tot += r; k += 1
        if record_states:
            states.append([float(v) for v in np.asarray(s[:7], dtype=float)])   # T, V, P, four in-hand amounts
    scale = 3.0 if target == 'NaCl' else 1.0
    return {'return': tot, 'quality': float(np.clip(tot / scale, 0.0, 1.0)), 'steps': k, 'states': states}


def heuristic_u(target):
    return np.array([heuristic_action(target) * 2 - 1] * H)


if __name__ == '__main__':
    import time
    env = make_env(); rng = np.random.default_rng(0); t0 = time.time()
    for tgt in WURTZ_TARGETS:
        h = run(heuristic_u(tgt), tgt, env)
        rnd = [run(rng.uniform(-1, 1, (H, ADIM)), tgt, env)['return'] for _ in range(5)]
        print(f'{tgt:24s} heuristic return {h["return"]:.3f} q {h["quality"]:.3f} | random mean {np.mean(rnd):.3f}  ({time.time() - t0:.1f}s)')
