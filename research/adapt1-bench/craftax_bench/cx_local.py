"""Local comparators for Craftax-Classic-Symbolic with the capped episodes used for Adapt-1 (CRAFTAX_EP_CAP):
  random SEED EPISODES OUT.jsonl        uniform random actions
  ppo SEED STEPS OUT_DIR                SB3 PPO (MLP) on the same cx_env features, single environment (slow but same information);
                                        evaluation on 20 capped episodes afterwards
The published Craftax-Classic baselines (PPO-RNN, 1e9 steps, 10,000-step episodes) are quoted in the report as context.
usage: cx_local.py random|ppo ...
"""
import sys, json, time, os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cx_env

WHAT, SEED = sys.argv[1], int(sys.argv[2])
if WHAT == 'random':
    EPISODES, OUT = int(sys.argv[3]), sys.argv[4]
    t = cx_env.CraftaxTask(SEED); rng = np.random.default_rng(SEED)
    with open(OUT, 'a') as fh:
        for ep in range(EPISODES):
            t.reset(SEED * 100000 + ep); done = False
            while not done:
                _, r, done, _ = t.step(rng.integers(t.n_actions))
            fh.write(json.dumps({'method': 'random', 'seed': SEED, 'episode': ep, 'return': t.ret, 'n_achievements': len(t.achievements()), 'achievements': t.achievements(), 'len': t.t}) + '\n')
    print('random done')
else:
    import gymnasium as gym, torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    STEPS, OUT = int(sys.argv[3]), Path(sys.argv[4]); OUT.mkdir(parents=True, exist_ok=True)

    class CXGym(gym.Env):
        def __init__(self, seed):
            self.t = cx_env.CraftaxTask(seed); self.keys = sorted(self.t.reset(0).keys()); self.ep = 0; self.seed0 = seed
            self.observation_space = gym.spaces.Box(-2.0, 2.0, (len(self.keys),), dtype=np.float32); self.action_space = gym.spaces.Discrete(self.t.n_actions)

        def _o(self, f):
            return np.array([f[k] for k in self.keys], dtype=np.float32)

        def reset(self, seed=None, options=None):
            self.ep += 1; return self._o(self.t.reset(self.seed0 * 100000 + 500000 + self.ep)), {}

        def step(self, a):
            f, r, done, info = self.t.step(int(a)); return self._o(f), float(r), bool(done), False, {}
    venv = DummyVecEnv([lambda: CXGym(SEED)]); log = open(OUT / f'episodes-ppo-s{SEED}.jsonl', 'w', buffering=1); t0 = time.time()

    class Logger(BaseCallback):
        def __init__(self):
            super().__init__(); self.tot = 0.0; self.ep = 0; self.steps = 0

        def _on_step(self):
            self.tot += float(self.locals['rewards'][0]); self.steps += 1
            if bool(self.locals['dones'][0]):
                log.write(json.dumps({'episode': self.ep, 'return': self.tot, 'steps': self.steps, 'elapsed_s': round(time.time() - t0, 1)}) + '\n'); self.tot = 0.0; self.ep += 1
            return True
    model = PPO('MlpPolicy', venv, n_steps=2048, batch_size=64, gamma=0.99, ent_coef=0.01, seed=SEED, verbose=0, device='cpu')
    model.learn(total_timesteps=STEPS, callback=Logger()); model.save(OUT / f'model-ppo-s{SEED}.zip')
    g = CXGym(SEED); g.ep = -1; rets = []; nach = []
    for ep in range(20):
        o, _ = g.reset(); done = False
        while not done:
            a, _ = model.predict(o, deterministic=True); o, r, done, _, _ = g.step(a)
        rets.append(g.t.ret); nach.append(len(g.t.achievements()))
    json.dump({'seed': SEED, 'steps': STEPS, 'eval_mean_return': float(np.mean(rets)), 'eval_mean_achievements': float(np.mean(nach)), 'wall_s': round(time.time() - t0)},
              open(OUT / f'eval-ppo-s{SEED}.json', 'w'), indent=1)
    print('ppo', SEED, 'eval mean achievements', np.mean(nach))
