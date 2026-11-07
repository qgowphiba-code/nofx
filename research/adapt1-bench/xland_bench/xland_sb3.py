"""Deep-RL comparator on the XLand macro-action task: SB3 PPO over the same decision features and the same 31 macros
(inapplicable macros are no-ops with reward 0, exactly as for Adapt-1's fallbacks). Meta-training across the tier's
TRAIN rulesets (indices >= 1000), each episode = one ruleset for K trials; evaluation on the EVAL rulesets used by the
Adapt-1 runs (indices 0..N-1) with the same layout seeds. The features include trial index and last-macro outcome, so the
MLP policy can condition on within-episode history the same way Adapt-1's context does (no recurrence).
usage: xland_sb3.py train TIER SEED MACRO_STEPS OUT_DIR | xland_sb3.py eval TIER MODEL.zip FIRST N SEED OUT.jsonl
"""
import sys, json, time, os, random
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import numpy as np
import gymnasium as gym
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xland_macro as X

POLICIES = X.MacroXLand.policy_names()


def features(e):
    v = e.context()
    for i, c in enumerate(e.slots):
        v[f'slot{i}_tile'] = (c[0] / 12.0) if c else 0.0
        v[f'slot{i}_color'] = (c[1] / 11.0) if c else 0.0
    return v


class MacroGym(gym.Env):
    def __init__(self, tier, ruleset_ids, seed=0, fixed_seed_by_ruleset=False):
        self.bench = X.load_tier(tier); self.ids = list(ruleset_ids); self.rng = random.Random(seed); self.env = X.MacroXLand()
        self.fixed = fixed_seed_by_ruleset; self.seed_base = seed
        self.env.start(self.bench.get_ruleset(self.ids[0]), 0); self.keys = sorted(features(self.env).keys())
        self.observation_space = gym.spaces.Box(-1.0, 2.0, (len(self.keys),), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(POLICIES))
        self.cur = None

    def _obs(self):
        v = features(self.env); return np.array([v[k] for k in self.keys], dtype=np.float32)

    def reset(self, seed=None, options=None):
        while True:
            idx = self.rng.choice(self.ids) if not self.fixed else self.ids[self.cur_i]
            try:
                self.env.start(self.bench.get_ruleset(idx), (self.seed_base * 100000 + idx) if self.fixed else self.rng.randrange(1 << 30)); break
            except ValueError:
                if self.fixed:
                    self.cur_i += 1; continue
        self.cur = idx; self.n = 0
        return self._obs(), {}

    def step(self, a):
        r, ok, _ = self.env.macro(POLICIES[int(a)]); self.n += 1
        done = self.env.episode_done() or self.n > 12 * self.env.K + 5
        return self._obs(), float(r), bool(done), False, {'ok': ok}


if __name__ == '__main__':
    mode = sys.argv[1]
    import torch; torch.set_num_threads(1)
    from stable_baselines3 import PPO
    if mode == 'train':
        TIER, SEED, STEPS, OUT = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5]); OUT.mkdir(parents=True, exist_ok=True)
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import BaseCallback
        venv = DummyVecEnv([lambda: MacroGym(TIER, range(1000, 3000), SEED)])
        log = open(OUT / f'episodes-ppo-{TIER}-s{SEED}.jsonl', 'w', buffering=1); t0 = time.time()

        class Logger(BaseCallback):
            def __init__(self):
                super().__init__(); self.tot = 0.0; self.ep = 0; self.steps = 0

            def _on_step(self):
                self.tot += float(self.locals['rewards'][0]); self.steps += 1
                if bool(self.locals['dones'][0]):
                    log.write(json.dumps({'episode': self.ep, 'return': self.tot, 'steps': self.steps, 'elapsed_s': round(time.time() - t0, 1)}) + '\n'); self.tot = 0.0; self.ep += 1
                return True
        model = PPO('MlpPolicy', venv, n_steps=512, batch_size=64, gamma=0.99, ent_coef=0.01, seed=SEED, verbose=0, device='cpu')
        model.learn(total_timesteps=STEPS, callback=Logger()); model.save(OUT / f'model-ppo-{TIER}-s{SEED}.zip')
        print('trained', TIER, SEED, STEPS, f'{time.time() - t0:.0f}s')
    else:
        TIER, MODEL, FIRST, N, SEED, OUT = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
        model = PPO.load(MODEL, device='cpu'); g = MacroGym(TIER, list(range(FIRST, FIRST + N)), SEED, fixed_seed_by_ruleset=True)
        with open(OUT, 'a') as fh:
            for i in range(N):
                g.cur_i = i
                try:
                    o, _ = g.reset()
                except IndexError:
                    break
                if g.cur_i >= N:
                    break
                done = False; n = 0
                while not done:
                    a, _ = model.predict(o, deterministic=True); o, r, done, _, _ = g.step(a); n += 1
                fh.write(json.dumps({'ruleset': g.cur, 'ppo_trials': list(g.env.trial_rewards) + [0.0] * (g.env.K - len(g.env.trial_rewards)), 'macros': n, 'model': os.path.basename(MODEL)}) + '\n')
        print('evaluated', N, 'rulesets')
