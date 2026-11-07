"""MiniHack skill tasks (MiniHack 1.0.2 / NLE 1.3.0, Samvelyan et al. 2021) with a compact symbolic feature extractor for the
Adapt-1 sequential route and the same-information PPO comparator. The environment is untouched: the Restricted action sets
are the benchmark's own; items keep NetHack's randomised appearances ("an effervescent potion"), so which item does what is
latent and has to be discovered by using it.

Tasks (all *-Restricted-v0*): LavaCross-Levitate-Potion-Inv (11 actions), LavaCross-Levitate-Ring-Inv (12),
LavaCross (16: a random item type lies on the floor), WoD-Medium (11: wand of death on the floor, a minotaur blocks the
corridor), WoD-Hard (11). Reward: +1 on reaching the goal, 0 otherwise (death ends the episode with 0). Step cap 250.

Features (about 40 scalars, all from the current observation): agent position, offsets to the exit, to the nearest floor item
and to the nearest monster when visible; the 8-neighbourhood lava mask; inventory class flags (potion, ring, wand, armour,
tool) and whether the inventory grew this episode; levitation condition; a coarse class of the last game message; HP fraction;
step fraction; the last action taken (one-hot over the task's actions).
"""
import os, re
import numpy as np
import gymnasium as gym
import minihack  # noqa: F401  (registers the environments)
from nle import nethack as nh

TASKS = {
    'potion': 'MiniHack-LavaCross-Levitate-Potion-Inv-Restricted-v0',
    'ring': 'MiniHack-LavaCross-Levitate-Ring-Inv-Restricted-v0',
    'lavacross': 'MiniHack-LavaCross-Restricted-v0',
    'wod_medium': 'MiniHack-WoD-Medium-Restricted-v0',
    'wod_hard': 'MiniHack-WoD-Hard-Restricted-v0',
}
OBS_KEYS = ('chars', 'message', 'blstats', 'inv_oclasses', 'inv_strs', 'glyphs')
MSG_CLASSES = ['none', 'float', 'nothing', 'burn', 'see_here', 'prompt', 'kill', 'other']
# NetHack object classes (nle.nethack): 4 wand, 7 potion, 8 ring, 3 armor, 6 tool, 9 scroll
CLASS_FLAGS = {'potion': 7, 'ring': 8, 'wand': 4, 'armor': 3, 'tool': 6, 'scroll': 9}
BL_X, BL_Y, BL_HP, BL_HPMAX, BL_CONDITION = nh.NLE_BL_X, nh.NLE_BL_Y, nh.NLE_BL_HP, nh.NLE_BL_HPMAX, nh.NLE_BL_CONDITION
BL_MASK_LEV = getattr(nh, 'BL_MASK_LEV', 0x00000040)


def msg_class(msg):
    m = msg.lower()
    if not m.strip():
        return 'none'
    if 'float' in m or 'levitat' in m:
        return 'float'
    if 'nothing happens' in m or 'nothing seems to happen' in m:
        return 'nothing'
    if 'burn' in m or 'lava' in m or 'fire' in m:
        return 'burn'
    if 'you see here' in m or 'there is' in m:
        return 'see_here'
    if 'what do you want' in m or 'in what direction' in m or '?' in m:
        return 'prompt'
    if 'kill' in m or 'destroy' in m or 'dies' in m:
        return 'kill'
    return 'other'


class MHTask:
    def __init__(self, task, seed=0):
        self.task = task; self.env_id = TASKS[task]
        self.env = gym.make(self.env_id, observation_keys=OBS_KEYS)
        self.actions = [a.name if hasattr(a, 'name') else str(a) for a in self.env.unwrapped.actions]
        self.n_actions = len(self.actions); self.seed = seed; self.max_steps = 250
        self.reset(seed)

    def reset(self, seed=None):
        self.obs, _ = self.env.reset(seed=seed)
        self.step_n = 0; self.last_action = -1; self.inv0 = self._inv_count(); self.done = False; self.ret = 0.0
        return self.features()

    def step(self, a):
        self.obs, r, term, trunc, info = self.env.step(int(a))
        self.step_n += 1; self.last_action = int(a); self.ret += float(r)
        self.done = bool(term or trunc) or self.step_n >= self.max_steps
        return self.features(), float(r), self.done, info

    # ---- feature extraction ----------------------------------------------------------------------------------------
    def _inv_count(self):
        return int(sum(1 for c in self.obs['inv_oclasses'] if c != 0 and c != 255))

    def _chars(self):
        return np.array(self.obs['chars'])

    def _agent_xy(self):
        bl = self.obs['blstats']; return int(bl[BL_X]), int(bl[BL_Y])

    def _find(self, chars, symbols):
        ys, xs = np.where(np.isin(chars, [ord(s) for s in symbols])); return list(zip(xs.tolist(), ys.tolist()))

    def _offset_to_nearest(self, chars, symbols, ax, ay, scale=20.0):
        pts = self._find(chars, symbols)
        pts = [(x, y) for x, y in pts if (x, y) != (ax, ay)]
        if not pts:
            return 0.0, 0.0, 0.0
        x, y = min(pts, key=lambda p: abs(p[0] - ax) + abs(p[1] - ay))
        return (x - ax) / scale, (y - ay) / scale, 1.0

    def features(self):
        ch = self._chars(); ax, ay = self._agent_xy(); v = {}
        v['ax'] = ax / 79.0; v['ay'] = ay / 21.0
        for name, syms in (('exit', '>'), ('item', '/!=[("%*'), ('monster', 'HhdDAaBbCcEeFfGgIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz@&;:')):
            dx, dy, vis = self._offset_to_nearest(ch, syms, ax, ay)
            v[f'{name}_dx'], v[f'{name}_dy'], v[f'{name}_vis'] = dx, dy, vis
        k = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                yy, xx = ay + dy, ax + dx
                v[f'lava{k}'] = 1.0 if (0 <= yy < ch.shape[0] and 0 <= xx < ch.shape[1] and ch[yy, xx] == ord('}')) else 0.0; k += 1
        classes = set(int(c) for c in self.obs['inv_oclasses'] if c not in (0, 255))
        for name, cid in CLASS_FLAGS.items():
            v[f'inv_{name}'] = 1.0 if cid in classes else 0.0
        v['inv_grew'] = 1.0 if self._inv_count() > self.inv0 else 0.0
        bl = self.obs['blstats']
        v['levitating'] = 1.0 if (int(bl[BL_CONDITION]) & BL_MASK_LEV) else 0.0
        v['hp_frac'] = float(bl[BL_HP]) / max(1.0, float(bl[BL_HPMAX]))
        msg = bytes(self.obs['message']).decode(errors='ignore').strip('\x00')
        mc = msg_class(msg)
        for c in MSG_CLASSES:
            v[f'msg_{c}'] = 1.0 if c == mc else 0.0
        v['step_frac'] = self.step_n / self.max_steps
        for i in range(self.n_actions):
            v[f'last_a{i}'] = 1.0 if i == self.last_action else 0.0
        return v

    def message(self):
        return bytes(self.obs['message']).decode(errors='ignore').strip('\x00')


if __name__ == '__main__':
    import sys, time
    task = sys.argv[1] if len(sys.argv) > 1 else 'potion'
    t = MHTask(task, 0); rng = np.random.default_rng(0)
    print(task, t.env_id, 'actions', t.actions, '| features', len(t.features()))
    wins = 0; lens = []; t0 = time.time()
    for ep in range(30):
        f = t.reset(seed=ep); done = False
        while not done:
            f, r, done, info = t.step(rng.integers(t.n_actions))
        wins += t.ret > 0; lens.append(t.step_n)
    print(f'random: {wins}/30 wins, mean length {np.mean(lens):.1f}, {(time.time() - t0) / sum(lens) * 1000:.1f} ms/step')
    # scripted probe: use the item then walk east
    f = t.reset(seed=1); print('start msg:', t.message()); use = [i for i, a in enumerate(t.actions) if a in ('QUAFF', 'PUTON', 'ZAP', 'WEAR', 'APPLY', 'READ')]
    for a in use[:1]:
        f, r, done, info = t.step(a); print('after', t.actions[a], '->', t.message()[:100], '| levitating', f['levitating'], 'done', done)
    for _ in range(12):
        if done: break
        f, r, done, info = t.step(t.actions.index('E'));
    print('after walking east: return', t.ret, 'steps', t.step_n, 'msg', t.message()[:80])
