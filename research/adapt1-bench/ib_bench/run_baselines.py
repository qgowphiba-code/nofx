"""Local comparators on the Industrial Benchmark at their full budgets: SB3 PPO and SAC (continuous), DQN (the benchmark's
27 discrete steerings), plus do-nothing and uniform random. Online learning curves are logged per episode (EP_LEN steps),
the same quantity that is logged for Adapt-1.
usage: run_baselines.py ALGO SETPOINT DRIFT(0/1) SEED STEPS OUT_DIR
"""
import sys, json, time, os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ib_env

ALGO, SP, DRIFT, SEED, STEPS, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3] == '1', int(sys.argv[4]), int(sys.argv[5]), Path(sys.argv[6])
OUT.mkdir(parents=True, exist_ok=True)
tag = f'{ALGO}-p{SP}{"d" if DRIFT else ""}-s{SEED}'
log = open(OUT / f'episodes-{tag}.jsonl', 'w', buffering=1)
t0 = time.time()


def write_episode(ep, rewards, steps_so_far):
    log.write(json.dumps({'episode': ep, 'mean_reward': float(np.mean(rewards)), 'len': len(rewards), 'steps': steps_so_far,
                          'elapsed_s': round(time.time() - t0, 1)}) + '\n')


if ALGO in ('zero', 'random'):
    env = ib_env.IB(SP, DRIFT, 'continuous', SEED)
    rng = np.random.default_rng(SEED); steps = 0; ep = 0
    while steps < STEPS:
        env.reset(); rs = []
        for _ in range(ib_env.EP_LEN):
            a = np.zeros(3) if ALGO == 'zero' else rng.uniform(-1, 1, 3)
            _, r, done, _ = env.step(a); rs.append(r); steps += 1
            if done: break
        write_episode(ep, rs, steps); ep += 1
else:
    import torch; torch.set_num_threads(1)
    from stable_baselines3 import PPO, SAC, DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    discrete = ALGO == 'dqn'
    venv = DummyVecEnv([lambda: ib_env.IBGymnasium(SP, DRIFT, discrete, SEED)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=0.99)
    venv.seed(SEED)

    class EpisodeLogger(BaseCallback):
        def __init__(self):
            super().__init__(); self.rs = []; self.ep = 0; self.steps = 0

        def _on_step(self):
            self.rs.append(float(self.training_env.get_original_reward()[0])); self.steps += 1
            if bool(self.locals['dones'][0]):
                write_episode(self.ep, self.rs, self.steps); self.rs = []; self.ep += 1
            return True

    kw = {'seed': SEED, 'verbose': 0, 'device': 'cpu'}
    if ALGO == 'ppo':
        model = PPO('MlpPolicy', venv, n_steps=2048, batch_size=64, gamma=0.99, **kw)
    elif ALGO == 'sac':
        model = SAC('MlpPolicy', venv, gamma=0.99, learning_starts=5000, **kw)
    elif ALGO == 'dqn':
        model = DQN('MlpPolicy', venv, gamma=0.99, learning_starts=5000, exploration_fraction=0.2, exploration_final_eps=0.05,
                    target_update_interval=1000, train_freq=4, **kw)
    else:
        raise SystemExit(f'unknown algo {ALGO}')
    model.learn(total_timesteps=STEPS, callback=EpisodeLogger())
log.close()
print(json.dumps({'tag': tag, 'steps': STEPS, 'wall_s': round(time.time() - t0)}))
