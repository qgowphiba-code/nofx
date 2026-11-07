"""Local comparators for the MiniHack skill tasks, same features, same seeds:
  random                       uniform random action
  script                       the task's known key sequence (use item, confirm, then greedy walk to the exit / zap at the
                               monster): a knowledgeable reference, not a learner
  ppo (train/eval)             SB3 PPO (MLP) on the same ~48 features at STEPS environment steps, then 200 evaluation episodes
usage: mh_local.py TASK random|script SEED EPISODES OUT.jsonl
       mh_local.py TASK ppo SEED STEPS OUT_DIR
"""
import sys, json, time, os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mh_env

TASK, WHAT, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3])


def script_policy(t, step_state):
    """Use the item (QUAFF/PUTON/ZAP + FIRE [+ READ for the ring hand]), then step towards the exit; WoD: pick up, zap east."""
    a = t.actions; f = t.features(); msg = t.message().lower()
    if t.task in ('potion', 'ring', 'lavacross'):
        if 'what do you want' in msg or 'which ring' in msg:
            return a.index('READ') if ('which ring' in msg and 'READ' in a) else a.index('FIRE')
        if not step_state.get('used'):
            step_state['used'] = True
            for verb in ('QUAFF', 'PUTON', 'ZAP', 'WEAR', 'APPLY', 'READ'):
                if verb in a and (t.task != 'lavacross' or f['inv_grew'] > 0 or f['inv_potion'] + f['inv_ring'] + f['inv_wand'] > 0):
                    return a.index(verb)
            if t.task == 'lavacross':
                step_state['used'] = False
                if f['item_vis'] and (f['item_dx'] or f['item_dy']):
                    return move_towards(a, f['item_dx'], f['item_dy'])
                return a.index('PICKUP')
        return move_towards(a, f['exit_dx'], f['exit_dy']) if f['exit_vis'] else a.index('E')
    # wand of death
    if 'in what direction' in msg:
        return a.index('E')
    if 'what do you want' in msg:
        return a.index('FIRE')
    if f['inv_wand'] == 0:
        if f['item_vis'] and (f['item_dx'] or f['item_dy']):
            return move_towards(a, f['item_dx'], f['item_dy'])
        return a.index('PICKUP')
    if f['monster_vis'] and abs(f['monster_dy']) < 1e-9 and f['monster_dx'] > 0 and not step_state.get('zapped'):
        step_state['zapped'] = True; return a.index('ZAP')
    return move_towards(a, f['exit_dx'], f['exit_dy']) if f['exit_vis'] else a.index('E')


def move_towards(a, dx, dy):
    name = ('N' if dy < 0 else 'S' if dy > 0 else '') + ('E' if dx > 0 else 'W' if dx < 0 else '')
    if name in ('', 'N', 'S'):
        name = name or 'E'
    return a.index(name if name in a else 'E')


if WHAT in ('random', 'script'):
    EPISODES, OUT = int(sys.argv[4]), sys.argv[5]
    t = mh_env.MHTask(TASK, SEED); rng = np.random.default_rng(SEED)
    with open(OUT, 'a') as fh:
        for ep in range(EPISODES):
            t.reset(seed=SEED * 100000 + ep); done = False; st = {}
            while not done:
                a = rng.integers(t.n_actions) if WHAT == 'random' else script_policy(t, st)
                _, r, done, _ = t.step(a)
            fh.write(json.dumps({'method': WHAT, 'task': TASK, 'seed': SEED, 'episode': ep, 'return': t.ret, 'win': int(t.ret > 0), 'len': t.step_n}) + '\n')
    print(WHAT, TASK, 'done', EPISODES, 'episodes')
else:
    import gymnasium as gym, torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    STEPS, OUT = int(sys.argv[4]), Path(sys.argv[5]); OUT.mkdir(parents=True, exist_ok=True)

    class MHGym(gym.Env):
        def __init__(self, seed):
            self.t = mh_env.MHTask(TASK, seed); self.keys = sorted(self.t.features().keys()); self.ep = 0; self.seed0 = seed
            self.observation_space = gym.spaces.Box(-2.0, 2.0, (len(self.keys),), dtype=np.float32); self.action_space = gym.spaces.Discrete(self.t.n_actions)

        def _o(self, f):
            return np.array([f[k] for k in self.keys], dtype=np.float32)

        def reset(self, seed=None, options=None):
            self.ep += 1; return self._o(self.t.reset(seed=self.seed0 * 100000 + 500000 + self.ep)), {}

        def step(self, a):
            f, r, done, info = self.t.step(int(a)); return self._o(f), float(r), bool(done), False, {}
    venv = DummyVecEnv([lambda: MHGym(SEED)]); log = open(OUT / f'episodes-ppo-{TASK}-s{SEED}.jsonl', 'w', buffering=1); t0 = time.time()

    class Logger(BaseCallback):
        def __init__(self):
            super().__init__(); self.tot = 0.0; self.ep = 0; self.steps = 0

        def _on_step(self):
            self.tot += float(self.locals['rewards'][0]); self.steps += 1
            if bool(self.locals['dones'][0]):
                log.write(json.dumps({'episode': self.ep, 'return': self.tot, 'win': int(self.tot > 0), 'steps': self.steps, 'elapsed_s': round(time.time() - t0, 1)}) + '\n'); self.tot = 0.0; self.ep += 1
            return True
    model = PPO('MlpPolicy', venv, n_steps=2048, batch_size=64, gamma=0.99, ent_coef=0.01, seed=SEED, verbose=0, device='cpu')
    model.learn(total_timesteps=STEPS, callback=Logger()); model.save(OUT / f'model-ppo-{TASK}-s{SEED}.zip')
    g = MHGym(SEED); g.ep = -1; wins = []; lens = []
    for ep in range(200):
        o, _ = g.reset(); done = False
        while not done:
            a, _ = model.predict(o, deterministic=True); o, r, done, _, _ = g.step(a)
        wins.append(int(g.t.ret > 0)); lens.append(g.t.step_n)
    json.dump({'task': TASK, 'seed': SEED, 'steps': STEPS, 'eval_win_rate': float(np.mean(wins)), 'eval_mean_len': float(np.mean(lens)), 'wall_s': round(time.time() - t0)},
              open(OUT / f'eval-ppo-{TASK}-s{SEED}.json', 'w'), indent=1)
    print('ppo', TASK, SEED, 'eval win rate', np.mean(wins))
