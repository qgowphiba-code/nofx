"""Closed-loop deep-RL comparator on SMPL PenSimEnv (SB3 PPO) at its full budget, on SMPL's own protocol: 1,150 steps per
batch, 6 normalized actions in [-1, 1] (SMPL normalize=True), 9 normalized observations, dense per-step yield reward,
a fresh batch (random seed_ref) every episode. A fresh SMPL environment object is built per episode (recipe pointer bug).
Logs per-episode return (= batch yield incl. SMPL's -100 failure penalty) and finally rolls the policy out on the HELD batches.
usage: pensim_sb3.py train SEED STEPS OUT_DIR
"""
import sys, json, time, os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import numpy as np
import gymnasium as gym
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pensim_plant as P
import pensim_contexts as C


class PenGym(gym.Env):
    def __init__(self, seed=0, fixed_seed_ref=None):
        self.rng = np.random.default_rng(seed); self.fixed = fixed_seed_ref
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (9,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32)
        self.e = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.e = P.make_env(); self.e.normalize = True
        sr = self.fixed if self.fixed is not None else int(self.rng.integers(10000, 20000))
        obs = self.e.reset(random_seed_ref=sr)
        return np.clip(np.asarray(obs, dtype=np.float32), -1, 1), {}

    def step(self, action):
        try:
            obs, r, done, info = self.e.step(np.asarray(action, dtype=np.float32))
        except Exception:
            return np.zeros(9, dtype=np.float32), -100.0, True, False, {'error': True}
        return np.clip(np.nan_to_num(np.asarray(obs, dtype=np.float32)), -1, 1), float(r), bool(done), False, {}


def rollout(model, seed_ref):
    env = PenGym(fixed_seed_ref=seed_ref); o, _ = env.reset(); tot = 0.0; done = False; n = 0
    while not done:
        a, _ = model.predict(o, deterministic=True); o, r, done, _, _ = env.step(a); tot += r; n += 1
    return tot, n


if __name__ == '__main__':
    _, SEED, STEPS, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    OUT.mkdir(parents=True, exist_ok=True)
    import torch; torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    venv = DummyVecEnv([lambda: PenGym(SEED)])
    log = open(OUT / f'episodes-ppo-s{SEED}.jsonl', 'w', buffering=1); t0 = time.time()

    class Logger(BaseCallback):
        def __init__(self):
            super().__init__(); self.tot = 0.0; self.len = 0; self.ep = 0; self.steps = 0

        def _on_step(self):
            self.tot += float(self.locals['rewards'][0]); self.len += 1; self.steps += 1
            if bool(self.locals['dones'][0]):
                log.write(json.dumps({'episode': self.ep, 'return': self.tot, 'len': self.len, 'steps': self.steps, 'elapsed_s': round(time.time() - t0, 1)}) + '\n')
                self.tot = 0.0; self.len = 0; self.ep += 1
            return True
    model = PPO('MlpPolicy', venv, n_steps=2048, batch_size=64, gamma=0.999, seed=SEED, verbose=0, device='cpu')
    model.learn(total_timesteps=STEPS, callback=Logger())
    model.save(OUT / f'model-ppo-s{SEED}.zip')
    held = [rollout(model, s) for s in C.HELD]
    json.dump({'algo': 'ppo', 'seed': SEED, 'steps': STEPS, 'heldout_yields': [h[0] for h in held], 'heldout_mean_yield': float(np.mean([h[0] for h in held])),
               'completed_frac': float(np.mean([h[1] >= 1150 for h in held])), 'wall_s': round(time.time() - t0)}, open(OUT / f'heldout-ppo-s{SEED}.json', 'w'), indent=1)
    print('ppo', SEED, 'held-out mean yield', round(float(np.mean([h[0] for h in held]))))
