"""Closed-loop deep-RL comparators on BeerFMT (SB3 PPO / SAC), trained at their full budget on SMPL's own episode
protocol: hourly steps, temperature action in [9, 16] C, observation = SMPL's 8 variables scaled to [-1, 1] (SMPL
normalize=True convention), initial conditions redrawn every episode from SMPL's +-10 % ranges, dense SMPL reward.
Logs the per-episode return online and saves the final policy; eval mode scores a saved policy on the fixed HELD contexts
with learning off (the same batches the refinement arms are scored on).
usage: beer_sb3.py train ALGO SEED STEPS OUT_DIR | beer_sb3.py eval MODEL.zip OUT.json
"""
import sys, json, time, os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import numpy as np
import gymnasium as gym
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import beer_plant as B
import beer_contexts as C


class BeerGym(gym.Env):
    """Faithful to SMPL BeerFMTEnvGym(normalize=True, dense_reward=True): same plant, step, reward and termination."""
    def __init__(self, seed=0, fixed_x0=None):
        self.rng = np.random.default_rng(seed)
        self.fixed_x0 = fixed_x0
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (8,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

    def _norm(self, obs):
        return ((np.asarray(obs, dtype=float) - B.MIN_OBS) / (B.MAX_OBS - B.MIN_OBS) * 2 - 1).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.x0 = np.array(self.fixed_x0, dtype=float) if self.fixed_x0 is not None else B.sample_context(self.rng)
        self.obs = np.array(self.x0, dtype=np.float32).astype(float)
        self.prev_biomass = self.obs[0] + self.obs[1] + self.obs[2]
        self.k = 0
        return self._norm(self.obs), {}

    def step(self, action):
        T = B.T_MIN + (float(np.clip(action[0], -1, 1)) + 1) / 2 * (B.T_MAX - B.T_MIN)
        from scipy.integrate import odeint
        t = np.arange(self.k, self.k + 1, B.BIOMASS_end_change_threshold)
        sol = odeint(B.beer_ode, tuple(self.obs[:7]), t, args=([B.INIT_SUGAR, T + B.ZERO_C_IN_K],))
        obs = np.array(list(sol[-1, :]) + [self.k + 1], dtype=np.float32).astype(float)
        beyond = bool(np.any(obs > B.MAX_OBS) or np.any(obs < B.MIN_OBS))
        if beyond:
            self.obs = obs; self.k += 1
            return self._norm(np.clip(obs, B.MIN_OBS, B.MAX_OBS)), B.ERROR_REWARD, True, False, {'reason': 'beyond_box'}
        X_A, X_L, X_D, S, EtOH, DY, EA, _ = obs
        biomass = X_A + X_L + X_D
        finished = (biomass < B.BIOMASS_end_threshold or abs(biomass - self.prev_biomass) < B.BIOMASS_end_change_threshold) and S < B.SUGAR_end_threshold and EtOH > 50.0
        timeout = (self.k + 1) >= B.MAX_STEPS
        reward = -B.ERROR_REWARD if finished else -1.0      # same reachable rule as SMPL: no extra penalty at the 200-step limit
        self.prev_biomass = biomass
        self.obs = obs; self.k += 1
        return self._norm(obs), float(reward), bool(finished or timeout), False, {'reason': 'finished' if finished else ('timeout' if timeout else '')}


def rollout(model, x0):
    env = BeerGym(fixed_x0=x0)
    o, _ = env.reset(); tot = 0.0; done = False
    while not done:
        a, _ = model.predict(o, deterministic=True)
        o, r, done, _, _ = env.step(a); tot += r
    return tot


if __name__ == '__main__':
    mode = sys.argv[1]
    import torch; torch.set_num_threads(1)
    from stable_baselines3 import PPO, SAC
    if mode == 'train':
        ALGO, SEED, STEPS, OUT = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
        OUT.mkdir(parents=True, exist_ok=True)
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
        venv = DummyVecEnv([lambda: BeerGym(SEED)])
        log = open(OUT / f'episodes-{ALGO}-s{SEED}.jsonl', 'w', buffering=1); t0 = time.time()

        class Logger(BaseCallback):
            def __init__(self):
                super().__init__(); self.tot = 0.0; self.len = 0; self.ep = 0; self.steps = 0

            def _on_step(self):
                self.tot += float(self.locals['rewards'][0]); self.len += 1; self.steps += 1
                if bool(self.locals['dones'][0]):
                    log.write(json.dumps({'episode': self.ep, 'return': self.tot, 'len': self.len, 'steps': self.steps, 'elapsed_s': round(time.time() - t0, 1)}) + '\n')
                    self.tot = 0.0; self.len = 0; self.ep += 1
                return True
        kw = {'seed': SEED, 'verbose': 0, 'device': 'cpu'}
        model = PPO('MlpPolicy', venv, n_steps=2048, batch_size=64, gamma=0.99, **kw) if ALGO == 'ppo' else SAC('MlpPolicy', venv, gamma=0.99, learning_starts=5000, **kw)
        model.learn(total_timesteps=STEPS, callback=Logger())
        model.save(OUT / f'model-{ALGO}-s{SEED}.zip')
        held = [rollout(model, x0) for x0 in C.HELD]
        json.dump({'algo': ALGO, 'seed': SEED, 'steps': STEPS, 'heldout_returns': held, 'heldout_mean_return': float(np.mean(held)),
                   'heldout_mean_quality': float(np.mean([B.quality(r) for r in held])), 'wall_s': round(time.time() - t0)},
                  open(OUT / f'heldout-{ALGO}-s{SEED}.json', 'w'), indent=1)
        print(ALGO, SEED, 'held-out mean return', round(float(np.mean(held)), 1))
    else:
        model_path, OUT = sys.argv[2], sys.argv[3]
        model = (PPO if 'ppo' in model_path else SAC).load(model_path, device='cpu')
        held = [rollout(model, x0) for x0 in C.HELD]
        json.dump({'model': model_path, 'heldout_returns': held, 'heldout_mean_return': float(np.mean(held))}, open(OUT, 'w'), indent=1)
        print('held-out mean return', round(float(np.mean(held)), 1))
