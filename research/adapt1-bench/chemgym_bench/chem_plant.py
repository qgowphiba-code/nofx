"""ChemGymRL reaction benches (ChemGymRL-v1, Beeler et al. 2023/2024) as batch executors for Machina and local comparators.

The benchmark code is imported from a read-only clone (CHEMGYM_REPO). Two NumPy alias shims are set for gym 0.26.
Benches:
  wurtz  GenWurtzReact-v1: Wurtz coupling (1-, 2-, 3-chlorohexane + Na in diethyl ether), 7 targets, 6 controls
         (temperature change, volume change, four reagent additions); reward = moles of target (NaCl scaled by 1/3).
  fict   FictReact-v2: the fictional reaction (A, B, D + water), 4 targets E, F, G, I with E as the material to avoid for the
         other targets, 7 controls; reward = moles of target minus moles of E. Target I needs a two-stage recipe (the
         benchmark's own heuristic switches reagents after step 9).
One episode = 20 steps; actions in [0, 1]; the reward is terminal only. The target is drawn at reset in the original
environment; here it is pinned per batch so that a recipe can be acquired per target (set_target).
Executor CHEM-1: the 20 x ADIM action sequence is the open-loop recipe (the plant is deterministic given the target),
u in [-1, 1] maps linearly onto the action box [0, 1]. Quality in [0, 1]: wurtz = return (NaCl / 3); fict = (return + 0.5) / 1.5.
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

H = 20
BENCHES = {
    'wurtz': {'env_id': 'GenWurtzReact-v1', 'adim': 6, 'n_targets': 7, 'obs_state_dims': 7,
              'targets': ["dodecane", "5-methylundecane", "4-ethyldecane", "5,6-dimethyldecane", "4-ethyl-5-methylnonane", "4,5-diethyloctane", "NaCl"]},
    'fict': {'env_id': 'FictReact-v2', 'adim': 7, 'n_targets': 4, 'obs_state_dims': 8, 'targets': ['E', 'F', 'G', 'I']},
}
WURTZ_TARGETS = BENCHES['wurtz']['targets']


def make_env(bench='wurtz'):
    return gym.make(BENCHES[bench]['env_id']).unwrapped


def set_target(env, target):
    """Pin the target material (the original env redraws it at reset)."""
    env.reaction.desired = target
    env.target_material = target
    if hasattr(env.reaction, 'target_material'):
        env.reaction.target_material = target


def heuristic_action(bench, target, step=0):
    """The benchmark's own hand-written reaction policies (Heuristic_Policies/*_reaction_policy.py)."""
    a = np.zeros(BENCHES[bench]['adim'], dtype=np.float32); a[0] = 1.0
    if bench == 'wurtz':
        add = {"dodecane": [2, 5], "5-methylundecane": [2, 3, 5], "4-ethyldecane": [2, 4, 5], "5,6-dimethyldecane": [3, 5],
               "4-ethyl-5-methylnonane": [3, 4, 5], "4,5-diethyloctane": [4, 5], "NaCl": [2, 3, 4, 5]}[target]
    else:
        add = {'E': [2, 3], 'F': [2, 4], 'G': [3, 4], 'I': ([2, 4] if step < 9 else [3, 4])}[target]
    for i in add:
        a[i] = 1.0
    return a


def heuristic_u(bench, target):
    return np.array([heuristic_action(bench, target, k) * 2 - 1 for k in range(H)])


def actions_from_u(u_row):
    return ((np.clip(np.asarray(u_row, dtype=float), -1, 1) + 1) / 2).astype(np.float32)


def quality(bench, ret, target):
    if bench == 'wurtz':
        return float(np.clip(ret / (3.0 if target == 'NaCl' else 1.0), 0.0, 1.0))
    return float(np.clip((ret + 0.5) / 1.5, 0.0, 1.0))


def run(u_matrix, target, env=None, record_states=False, bench='wurtz'):
    """Execute one 20-step batch with the open-loop recipe u_matrix (H x ADIM). Returns dict(return, quality, steps, states)."""
    env = env or make_env(bench)
    set_target(env, target)
    s = env.reset()
    if isinstance(s, tuple):
        s = s[0]
    set_target(env, target)        # reset redraws the target; pin it again before the first step
    sd = BENCHES[bench]['obs_state_dims']
    states = []; tot = 0.0; done = False; k = 0
    while not done and k < H:
        out = env.step(actions_from_u(u_matrix[k])); s, r, done = out[0], float(out[1]), bool(out[2]); tot += r; k += 1
        if record_states:
            states.append([float(v) for v in np.asarray(s[:sd], dtype=float)])   # T, V, P, in-hand amounts
    return {'return': tot, 'quality': quality(bench, tot, target), 'steps': k, 'states': states}


if __name__ == '__main__':
    import time
    for bench in ('wurtz', 'fict'):
        env = make_env(bench); rng = np.random.default_rng(0); t0 = time.time()
        for tgt in BENCHES[bench]['targets']:
            h = run(heuristic_u(bench, tgt), tgt, env, bench=bench)
            rnd = [run(rng.uniform(-1, 1, (H, BENCHES[bench]['adim'])), tgt, env, bench=bench)['return'] for _ in range(5)]
            print(f'{bench:5s} {tgt:24s} heuristic return {h["return"]:.3f} q {h["quality"]:.3f} | random mean {np.mean(rnd):.3f}  ({time.time() - t0:.1f}s)')
