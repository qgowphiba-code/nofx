"""Craftax-Classic-Symbolic (Matthews et al. 2024) with a compact structured context for the Adapt-1 sequential route and the
same-information PPO comparator. The environment is the installed `craftax` package, untouched (JAX, jitted step); the
context is read from the environment's own state (inventory, player stats, achievements) and from the symbolic map around
the player (counts of each block type in the 9 x 7 view and offsets to the nearest instance of the blocks that matter).
Reward: Craftax's own (+1 for each achievement unlocked the first time in the episode, minus a small health penalty).
Episode cap: EP_CAP steps (Craftax's own cap is 10,000; the cap here is set by the API budget and recorded in PREREG G).
"""
import os
import numpy as np
import jax, jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name
import craftax.craftax_classic.constants as C

ENV_NAME = 'Craftax-Classic-Symbolic-v1'
ACTIONS = [a.name for a in C.Action]                     # 17
BLOCKS = [b.name for b in C.BlockType]                   # 17
ACHIEVEMENTS = [a.name for a in C.Achievement]           # 22
INV = ['wood', 'stone', 'coal', 'iron', 'diamond', 'sapling', 'wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe', 'wood_sword', 'stone_sword', 'iron_sword']
NEAR = ['TREE', 'STONE', 'COAL', 'IRON', 'DIAMOND', 'WATER', 'CRAFTING_TABLE', 'FURNACE', 'GRASS', 'LAVA']
VIEW_H, VIEW_W = 7, 9                                    # obs view (rows, cols) around the player
EP_CAP = int(os.environ.get('CRAFTAX_EP_CAP', '500'))


class CraftaxTask:
    def __init__(self, seed=0):
        self.env = make_craftax_env_from_name(ENV_NAME, auto_reset=False); self.params = self.env.default_params
        self.reset_fn = jax.jit(self.env.reset); self.step_fn = jax.jit(self.env.step)
        self.key = jax.random.PRNGKey(seed); self.n_actions = len(ACTIONS)
        self.block_idx = {b: i for i, b in enumerate(BLOCKS)}

    def reset(self, seed=None):
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)
        self.key, k = jax.random.split(self.key); self.obs, self.state = self.reset_fn(k, self.params)
        self.t = 0; self.ret = 0.0; self.done = False; self.last_action = -1; self.ach0 = np.array(self.state.achievements)
        return self.features()

    def step(self, a):
        self.key, k = jax.random.split(self.key)
        self.obs, self.state, r, done, info = self.step_fn(k, self.state, int(a), self.params)
        self.t += 1; self.ret += float(r); self.last_action = int(a)
        self.done = bool(done) or self.t >= EP_CAP
        return self.features(), float(r), self.done, info

    def achievements(self):
        return [ACHIEVEMENTS[i] for i, v in enumerate(np.array(self.state.achievements)) if v]

    def features(self):
        s = self.state; v = {}
        inv = s.inventory
        for name in INV:
            v[f'inv_{name}'] = float(min(9, int(getattr(inv, name)))) / 9.0
        v['health'] = float(s.player_health) / 9.0; v['food'] = float(s.player_food) / 9.0; v['drink'] = float(s.player_drink) / 9.0; v['energy'] = float(s.player_energy) / 9.0
        v['sleeping'] = float(bool(s.is_sleeping)); v['daylight'] = float(s.light_level)
        pos = np.array(s.player_position); m = np.array(s.map); H, W = m.shape
        y0, x0 = int(pos[0]), int(pos[1])
        v['pos_y'] = y0 / H; v['pos_x'] = x0 / W
        view = m[max(0, y0 - VIEW_H // 2): y0 + VIEW_H // 2 + 1, max(0, x0 - VIEW_W // 2): x0 + VIEW_W // 2 + 1]
        counts = np.bincount(view.ravel(), minlength=len(BLOCKS))
        for b in BLOCKS:
            v[f'cnt_{b.lower()}'] = float(min(10, counts[self.block_idx[b]])) / 10.0
        R = 12
        win = m[max(0, y0 - R): y0 + R + 1, max(0, x0 - R): x0 + R + 1]; oy, ox = max(0, y0 - R), max(0, x0 - R)
        for b in NEAR:
            ys, xs = np.where(win == self.block_idx[b])
            if len(ys):
                d = np.abs(ys + oy - y0) + np.abs(xs + ox - x0); i = int(np.argmin(d))
                v[f'near_{b.lower()}_dy'] = float(ys[i] + oy - y0) / R; v[f'near_{b.lower()}_dx'] = float(xs[i] + ox - x0) / R; v[f'near_{b.lower()}_vis'] = 1.0
            else:
                v[f'near_{b.lower()}_dy'] = 0.0; v[f'near_{b.lower()}_dx'] = 0.0; v[f'near_{b.lower()}_vis'] = 0.0
        for mob, lim in (('zombies', 3), ('cows', 3), ('skeletons', 2)):
            grp = getattr(s, mob); mask = np.array(grp.mask); p = np.array(grp.position)
            d = [abs(int(p[i][0]) - y0) + abs(int(p[i][1]) - x0) for i in range(len(mask)) if mask[i]]
            v[f'{mob}_near'] = 1.0 / (1.0 + min(d)) if d else 0.0
        dy, dx = np.array(C.DIRECTIONS)[int(s.player_direction)] if hasattr(C, 'DIRECTIONS') else (0, 0)
        v['dir_y'], v['dir_x'] = float(dy), float(dx)
        fy, fx = y0 + int(dy), x0 + int(dx)
        front = int(m[fy, fx]) if 0 <= fy < H and 0 <= fx < W else -1
        for b in BLOCKS:
            v[f'front_{b.lower()}'] = 1.0 if front == self.block_idx[b] else 0.0
        ach = np.array(s.achievements)
        for i, a in enumerate(ACHIEVEMENTS):
            v[f'ach_{a.lower()}'] = float(ach[i])
        v['t_frac'] = self.t / EP_CAP
        for i in range(self.n_actions):
            v[f'last_a{i}'] = 1.0 if i == self.last_action else 0.0
        return v


if __name__ == '__main__':
    import time
    t = CraftaxTask(0); f = t.reset(0); print('features', len(f), '| actions', ACTIONS)
    rng = np.random.default_rng(0); t0 = time.time(); rets = []; achs = []
    for ep in range(3):
        f = t.reset(ep); done = False
        while not done:
            f, r, done, info = t.step(rng.integers(t.n_actions))
        rets.append(t.ret); achs.append(len(t.achievements()))
    print(f'random: returns {rets}, achievements {achs}, {(time.time() - t0) / (3 * EP_CAP) * 1000:.1f} ms/step incl. features')
