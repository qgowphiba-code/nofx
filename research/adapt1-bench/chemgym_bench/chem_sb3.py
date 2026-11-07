"""Deep-RL comparators on the ChemGymRL Wurtz reaction bench at their published-style budgets, on the benchmark's own
protocol: the target is redrawn every episode and encoded in the observation (closed loop, 214-dim observation, 6 actions).
Algorithms as in the paper: PPO, A2C, SAC, TD3 (SB3 defaults; the paper used a [32, 32] MLP and 100k steps for PPO).
Logs the per-episode return online and finally evaluates the deterministic policy on every target (5 rollouts each; the
plant is deterministic given the target, so 1 would do).
usage: chem_sb3.py ALGO SEED STEPS OUT_DIR
"""
import sys, json, time, os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import numpy as np
import gymnasium as gymn
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chem_plant as P


class ChemGym(gymn.Env):
    def __init__(self, seed=0, fixed_target=None):
        self.env = P.make_env(); self.fixed = fixed_target
        self.observation_space = gymn.spaces.Box(0.0, 1.0, self.env.observation_space.shape, dtype=np.float32)
        self.action_space = gymn.spaces.Box(0.0, 1.0, (P.ADIM,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if self.fixed is not None:
            P.set_target(self.env, self.fixed)
        s = self.env.reset(); s = s[0] if isinstance(s, tuple) else s
        if self.fixed is not None:
            P.set_target(self.env, self.fixed)
            s = np.array(s, dtype=np.float32); s[-7:] = 0.0; s[-7 + P.WURTZ_TARGETS.index(self.fixed)] = 1.0
        return np.clip(np.asarray(s, dtype=np.float32), 0, 1), {}

    def step(self, a):
        out = self.env.step(np.clip(np.asarray(a, dtype=np.float32), 0, 1))
        return np.clip(np.asarray(out[0], dtype=np.float32), 0, 1), float(out[1]), bool(out[2]), False, {}


def evaluate(model):
    res = {}
    for t in P.WURTZ_TARGETS:
        env = ChemGym(fixed_target=t); rets = []
        for _ in range(3):
            o, _ = env.reset(); done = False; R = 0.0
            while not done:
                a, _ = model.predict(o, deterministic=True); o, r, done, _, _ = env.step(a); R += r
            rets.append(R)
        res[t] = float(np.mean(rets))
    return res


if __name__ == '__main__':
    ALGO, SEED, STEPS, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    OUT.mkdir(parents=True, exist_ok=True)
    import torch; torch.set_num_threads(1)
    from stable_baselines3 import PPO, A2C, SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    venv = DummyVecEnv([lambda: ChemGym(SEED)])
    tag = f'{ALGO}-s{SEED}-{STEPS}'
    log = open(OUT / f'episodes-{tag}.jsonl', 'w', buffering=1); t0 = time.time()

    class Logger(BaseCallback):
        def __init__(self):
            super().__init__(); self.tot = 0.0; self.ep = 0; self.steps = 0

        def _on_step(self):
            self.tot += float(self.locals['rewards'][0]); self.steps += 1
            if bool(self.locals['dones'][0]):
                obs = self.locals['new_obs'][0] if 'new_obs' in self.locals else None
                log.write(json.dumps({'episode': self.ep, 'return': self.tot, 'steps': self.steps, 'elapsed_s': round(time.time() - t0, 1)}) + '\n')
                self.tot = 0.0; self.ep += 1
            return True
    kw = {'seed': SEED, 'verbose': 0, 'device': 'cpu', 'policy_kwargs': dict(net_arch=[32, 32])}
    model = {'ppo': lambda: PPO('MlpPolicy', venv, **kw), 'a2c': lambda: A2C('MlpPolicy', venv, **kw),
             'sac': lambda: SAC('MlpPolicy', venv, learning_starts=1000, **kw), 'td3': lambda: TD3('MlpPolicy', venv, learning_starts=1000, **kw)}[ALGO]()
    model.learn(total_timesteps=STEPS, callback=Logger())
    model.save(OUT / f'model-{tag}.zip')
    ev = evaluate(model)
    json.dump({'algo': ALGO, 'seed': SEED, 'steps': STEPS, 'per_target_return': ev, 'mean_organic': float(np.mean([ev[t] for t in P.WURTZ_TARGETS[:-1]])),
               'wall_s': round(time.time() - t0)}, open(OUT / f'eval-{tag}.json', 'w'), indent=1)
    print(tag, 'per-target', {k: round(v, 3) for k, v in ev.items()})
