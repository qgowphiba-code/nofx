"""XLand-MiniGrid (xminigrid 0.9.3, Nikulin et al. 2024) driven through macro-actions, for the Adapt-1 sequential route and the
local comparators. The environment itself is untouched: every macro is executed as a sequence of the environment's own
primitive actions (forward, turn, pick up, put down), planned by breadth-first search on the current grid.

Episode = one hidden ruleset played for K trials on fresh layouts (the meta-RL protocol of the benchmark; goal and rules are
never shown to the agent). Trial reward is the environment's: 1 - 0.9 * steps / max_steps when the goal is reached, else 0.

Macros, over object CLASSES (tile, colour) rather than object instances:
  pick <class>   walk next to an object of that class and pick it up (triggers AgentHold rules/goal)
  goto <class>   end a forward move next to an object of that class (triggers AgentNear rules/goal)
  put <class>    put the held object down on a free floor cell adjacent to an object of that class (TileNear rules/goal)
  drop           put the held object down anywhere
Policy slots = the classes present at the start of the episode plus RESERVE slots for classes that first appear later
(rule products), assigned in order of first appearance. A macro that is not applicable (class absent, nothing held, no
reachable cell) executes nothing and returns reward 0 with success False.
"""
import os, sys, json
from collections import deque
import numpy as np
import jax, jax.numpy as jnp
import xminigrid
from xminigrid.benchmarks import load_benchmark_from_path
from xminigrid.core.constants import Tiles, Colors
from xminigrid.core import grid as G
from xminigrid.types import RuleSet
import dataclasses

TILE_NAME = {f.default: f.name.lower() for f in dataclasses.fields(Tiles)}
COLOR_NAME = {f.default: f.name.lower() for f in dataclasses.fields(Colors)}
WALKABLE = set(int(v) for v in np.array(G.WALKABLE)); PICKABLE = set(int(v) for v in np.array(G.PICKABLE)); FREE = set(int(v) for v in np.array(G.FREE_TO_PUT_DOWN))
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # up, right, down, left (xminigrid DIRECTIONS)
FWD, CW, CCW, PICK, PUT = 0, 1, 2, 3, 4
OPS = ('pick', 'goto', 'put')
RESERVE = 2
MAX_SLOTS = 10            # fixed policy set: pick/goto/put x 10 slots + drop = 31 policies; rulesets with > 8 initial classes are skipped
SCRATCH = os.environ.get('ADAPT1_SCRATCH', '/tmp/claude-0/-home-user-nofx/7c289125-8882-5b5c-9bcd-654fe7300b9f/scratchpad')
RULESETS_DIR = os.path.join(SCRATCH, 'xland_rulesets')


def cname(c):
    return f'{COLOR_NAME.get(c[1], c[1])}_{TILE_NAME.get(c[0], c[0])}'


def load_tier(tier):
    return load_benchmark_from_path(os.path.join(RULESETS_DIR, f'{tier}_pf'))


class MacroXLand:
    def __init__(self, env_name='XLand-MiniGrid-R1-9x9', trials=6):
        self.env, self.params0 = xminigrid.make(env_name)
        self.reset_fn = jax.jit(self.env.reset); self.step_fn = jax.jit(self.env.step)
        self.K = trials; self.max_steps = int(self.params0.max_steps)

    # ---- episode / trial control ------------------------------------------------------------------------------------
    def start(self, ruleset: RuleSet, seed: int):
        self.params = self.params0.replace(ruleset=ruleset)
        self.key = jax.random.PRNGKey(seed); self.trial = 0; self.trial_rewards = []; self.prims_total = 0
        init = [tuple(int(v) for v in t) for t in np.array(ruleset.init_tiles) if int(t[0]) != 0]
        self.slots = []
        for c in init:
            if c not in self.slots:
                self.slots.append(c)
        self.n_init = len(self.slots)
        if self.n_init > MAX_SLOTS - RESERVE:
            raise ValueError(f'ruleset has {self.n_init} initial classes > {MAX_SLOTS - RESERVE}')
        self.slots += [None] * (MAX_SLOTS - self.n_init)
        self.ever_seen = set(init)
        self._new_trial()

    @staticmethod
    def policy_names():
        return [f'{op}_{i}' for i in range(MAX_SLOTS) for op in OPS] + ['drop']

    def _new_trial(self):
        self.key, k = jax.random.split(self.key); self.ts = self.reset_fn(self.params, k)
        self.steps = 0; self.trial_reward = 0.0; self.trial_done = False; self.macros_in_trial = 0; self.last_success = 0; self.last_reward = 0.0
        self._register_new_classes()

    def _register_new_classes(self):
        for c in self.classes_present():
            if c not in self.ever_seen:
                self.ever_seen.add(c)
                if None in self.slots:
                    self.slots[self.slots.index(None)] = c
        p = self.pocket()
        if p[0] != 0 and p not in self.ever_seen:
            self.ever_seen.add(p)
            if None in self.slots:
                self.slots[self.slots.index(None)] = p

    def episode_done(self):
        return self.trial >= self.K

    # ---- state readers ----------------------------------------------------------------------------------------------
    def grid(self):
        return np.array(self.ts.state.grid)

    def agent(self):
        pos = tuple(int(v) for v in np.array(self.ts.state.agent.position)); return pos, int(self.ts.state.agent.direction)

    def pocket(self):
        return tuple(int(v) for v in np.array(self.ts.state.agent.pocket))

    def objects(self):
        g = self.grid(); out = []
        for y in range(g.shape[0]):
            for x in range(g.shape[1]):
                if int(g[y, x, 0]) in PICKABLE:
                    out.append(((y, x), (int(g[y, x, 0]), int(g[y, x, 1]))))
        return out

    def classes_present(self):
        return [c for _, c in self.objects()]

    def policies(self):
        return self.policy_names()

    def context(self):
        """Decision-time features (same vector for Adapt-1 and the local learners). Nothing about rules or goal."""
        present = self.classes_present(); pocket = self.pocket(); v = {}
        for i, c in enumerate(self.slots):
            v[f'slot{i}_present'] = float(min(2, present.count(c))) / 2.0 if c is not None else 0.0
            v[f'slot{i}_held'] = 1.0 if (c is not None and pocket == c) else 0.0
            v[f'slot{i}_known'] = 1.0 if c is not None else 0.0
        v['holding'] = 1.0 if pocket[0] != 0 else 0.0
        v['trial_frac'] = self.trial / max(1, self.K)
        v['steps_frac'] = self.steps / self.max_steps
        v['macros_in_trial'] = min(1.0, self.macros_in_trial / 12.0)
        v['last_success'] = float(self.last_success)
        v['last_reward'] = float(self.last_reward)
        return v

    # ---- primitive execution ----------------------------------------------------------------------------------------
    def prim(self, a):
        self.ts = self.step_fn(self.params, self.ts, a); self.steps += 1; self.prims_total += 1
        r = float(self.ts.reward); self.trial_reward += r
        if int(self.ts.step_type) == 2:      # LAST: goal reached or step budget exhausted
            self.trial_done = True
        return r

    def _walkable(self, g, y, x):
        return 0 <= y < g.shape[0] and 0 <= x < g.shape[1] and int(g[y, x, 0]) in WALKABLE

    def _bfs(self, goal_test):
        """BFS over (y, x, dir) from the agent's state using forward/turn moves; goal_test(y, x, d, g) -> bool.
        Returns the primitive action list or None."""
        g = self.grid(); (y0, x0), d0 = self.agent()
        start = (y0, x0, d0); prev = {start: None}; q = deque([start])
        while q:
            s = q.popleft(); y, x, d = s
            if goal_test(y, x, d, g):
                acts = []
                while prev[s] is not None:
                    s, a = prev[s]; acts.append(a)
                return acts[::-1]
            for a, ns in ((CW, (y, x, (d + 1) % 4)), (CCW, (y, x, (d - 1) % 4)), (FWD, (y + DIRS[d][0], x + DIRS[d][1], d))):
                if a == FWD and not self._walkable(g, ns[0], ns[1]):
                    continue
                if ns not in prev:
                    prev[ns] = (s, a); q.append(ns)
        return None

    def _run(self, acts):
        r = 0.0
        for a in acts:
            r += self.prim(a)
            if self.trial_done:
                break
        return r

    def _positions_of(self, cls):
        return [p for p, c in self.objects() if c == cls]

    # ---- macros -------------------------------------------------------------------------------------------------------
    def macro(self, name):
        """Execute one macro. Returns (reward, success, primitive_count)."""
        if self.trial_done or self.episode_done():
            return 0.0, False, 0
        p0 = self.prims_total; r = 0.0; ok = False
        if name == 'drop':
            if self.pocket()[0] != 0:
                acts = self._bfs(lambda y, x, d, g: self._walkable(g, y + DIRS[d][0], x + DIRS[d][1]) and int(g[y + DIRS[d][0], x + DIRS[d][1], 0]) in FREE)
                if acts is not None:
                    r += self._run(acts + [PUT]); ok = True
        else:
            op, i = name.rsplit('_', 1); cls = self.slots[int(i)]
            targets = self._positions_of(cls) if cls is not None else []
            if targets:
                tset = set(targets)
                if op == 'pick':
                    acts = self._bfs(lambda y, x, d, g: (y + DIRS[d][0], x + DIRS[d][1]) in tset)
                    if acts is not None and self.pocket()[0] == 0:
                        r += self._run(acts + [PICK]); ok = True
                elif op == 'goto':
                    def adj(y, x):
                        return any((y + dy, x + dx) in tset for dy, dx in DIRS)
                    acts = self._bfs(lambda y, x, d, g: adj(y, x))
                    if acts is not None:
                        if not acts or acts[-1] != FWD:       # the AgentNear check fires on a forward move: step away and back
                            (y, x), d = self.agent(); g = self.grid(); extra = None
                            for turn, nd in ((None, d), (CW, (d + 1) % 4), (CCW, (d - 1) % 4), (CW, (d + 2) % 4)):
                                ny, nx = y + DIRS[nd][0], x + DIRS[nd][1]
                                if self._walkable(g, ny, nx):
                                    turns = [] if turn is None else ([CW] if nd == (d + 1) % 4 else [CCW] if nd == (d - 1) % 4 else [CW, CW])
                                    extra = turns + [FWD, CW, CW, FWD]; break
                            acts = acts + (extra or [])
                        r += self._run(acts); ok = True
                elif op == 'put':
                    if self.pocket()[0] != 0:
                        def front_ok(y, x, d, g):
                            fy, fx = y + DIRS[d][0], x + DIRS[d][1]
                            return self._walkable(g, fy, fx) and int(g[fy, fx, 0]) in FREE and any((fy + dy, fx + dx) in tset for dy, dx in DIRS)
                        acts = self._bfs(front_ok)
                        if acts is not None:
                            r += self._run(acts + [PUT]); ok = True
        self.macros_in_trial += 1; self.last_success = int(ok); self.last_reward = r
        self._register_new_classes()
        if self.trial_done:
            self.trial_rewards.append(self.trial_reward); self.trial += 1
            if not self.episode_done():
                self._new_trial()
        return r, ok, self.prims_total - p0


# ---- reference policies ----------------------------------------------------------------------------------------------
def random_policy(env, rng):
    return rng.choice(env.policies())


class SystematicExplorer:
    """Scripted comparator with no knowledge of rules: tries every untried macro in a fixed order (goto, pick, put-near pairs),
    remembers the macro sequence that produced reward in a trial and replays it in later trials."""
    def __init__(self, env):
        self.env = env; self.tried = set(); self.success_seq = None; self.cur = []

    def act(self):
        env = self.env
        if self.success_seq is not None and len(self.cur) < len(self.success_seq):
            a = self.success_seq[len(self.cur)]; self.cur.append(a); return a
        present = set(env.classes_present()); pocket = env.pocket()
        cands = []
        for i, c in enumerate(env.slots):
            if c is None or c not in present:
                continue
            cands.append(f'goto_{i}')
            if pocket[0] == 0:
                cands.append(f'pick_{i}')
            else:
                cands.append(f'put_{i}')
        order = [a for a in cands if a not in self.tried] or cands or ['drop']
        a = order[0]; self.tried.add(a); self.cur.append(a); return a

    def observe(self, reward, trial_ended):
        if trial_ended:
            if reward > 0 and self.success_seq is None:
                self.success_seq = list(self.cur)
            self.cur = []


def decode_ruleset(rs):
    g = np.array(rs.goal); gt = int(g[0])
    goal = {'type': {1: 'hold', 3: 'near', 4: 'tilenear'}.get(gt, gt), 'tiles': [tuple(int(v) for v in g[1:3])] + ([tuple(int(v) for v in g[3:5])] if gt == 4 else [])}
    rules = []
    for r in np.array(rs.rules):
        k = int(r[0])
        if k == 1: rules.append({'type': 'hold', 'in': [tuple(int(v) for v in r[1:3])], 'out': tuple(int(v) for v in r[3:5])})
        elif k == 2: rules.append({'type': 'near', 'in': [tuple(int(v) for v in r[1:3])], 'out': tuple(int(v) for v in r[3:5])})
        elif k == 3: rules.append({'type': 'tilenear', 'in': [tuple(int(v) for v in r[1:3]), tuple(int(v) for v in r[3:5])], 'out': tuple(int(v) for v in r[5:7])})
    return goal, rules


class Oracle:
    """Ceiling: knows the goal and rules; plans the shortest macro sequence (depth <= 3 productions)."""
    def __init__(self, env, ruleset):
        self.env = env; self.goal, self.rules = decode_ruleset(ruleset); self.plan = None

    def _slot(self, cls):
        return self.env.slots.index(cls) if cls in self.env.slots else None

    def _produce(self, cls, depth):
        """macro list that makes an object of class cls present (or held, for hold-rules), or None."""
        if cls in self.env.classes_present() or self.env.pocket() == cls:
            return []
        if depth == 0:
            return None
        for r in self.rules:
            if r['out'] != cls:
                continue
            if r['type'] == 'hold':
                pre = self._produce(r['in'][0], depth - 1)
                if pre is not None and self._slot(r['in'][0]) is not None:
                    return pre + [f"pick_{self._slot(r['in'][0])}", 'drop']
            elif r['type'] == 'near':
                pre = self._produce(r['in'][0], depth - 1)
                if pre is not None and self._slot(r['in'][0]) is not None:
                    return pre + [f"goto_{self._slot(r['in'][0])}"]
            elif r['type'] == 'tilenear':
                a, b = r['in']; pa = self._produce(a, depth - 1); pb = self._produce(b, depth - 1)
                if pa is not None and pb is not None and self._slot(a) is not None and self._slot(b) is not None:
                    return pa + pb + [f"pick_{self._slot(a)}", f"put_{self._slot(b)}"]
        return None

    def act(self):
        if not self.plan:
            g = self.goal; t = g['tiles']
            if g['type'] == 'hold':
                pre = self._produce(t[0], 3); self.plan = (pre or []) + ([f"pick_{self._slot(t[0])}"] if self._slot(t[0]) is not None else ['drop'])
            elif g['type'] == 'near':
                pre = self._produce(t[0], 3); self.plan = (pre or []) + ([f"goto_{self._slot(t[0])}"] if self._slot(t[0]) is not None else ['drop'])
            else:
                pa = self._produce(t[0], 3) or []; pb = self._produce(t[1], 3) or []
                sa, sb = self._slot(t[0]), self._slot(t[1])
                self.plan = pa + pb + ([f'pick_{sa}', f'put_{sb}'] if sa is not None and sb is not None else ['drop'])
        return self.plan.pop(0)


def play_episode(env, ruleset, seed, policy, on_result=None):
    """Run one K-trial episode; policy(env) -> macro name. Returns dict with trial rewards and macro count."""
    env.start(ruleset, seed); macros = 0
    while not env.episode_done():
        tr_before = env.trial
        a = policy(env)
        r, ok, n = env.macro(a); macros += 1
        if on_result:
            on_result(a, r, ok, env.trial > tr_before)
        if macros > 12 * env.K + 5:      # safety: a policy that never ends a trial
            break
    return {'trial_rewards': list(env.trial_rewards) + [0.0] * (env.K - len(env.trial_rewards)), 'macros': macros, 'prims': env.prims_total}


if __name__ == '__main__':
    import time
    tier = sys.argv[1] if len(sys.argv) > 1 else 'small'
    b = load_tier(tier); env = MacroXLand(); rng = np.random.default_rng(0)
    print(tier, 'rulesets', b.num_rulesets(), '| walkable', WALKABLE, 'pickable', PICKABLE, 'free', FREE)
    t0 = time.time()
    for i in range(3):
        rs = b.get_ruleset(i); goal, rules = decode_ruleset(rs)
        print(f'#{i} goal {goal["type"]} {[cname(c) for c in goal["tiles"]]} rules {[(r["type"], [cname(c) for c in r["in"]], cname(r["out"])) for r in rules]} objects {[cname(c) for c in env.slots[:env.n_init]] if False else ""}')
        res_r = play_episode(env, rs, 1, lambda e: random_policy(e, rng))
        ex = SystematicExplorer(env); res_e = play_episode(env, rs, 1, lambda e: ex.act(), on_result=lambda a, r, ok, te: ex.observe(r, te))
        env.start(rs, 1); orc = Oracle(env, rs); res_o = play_episode(env, rs, 1, lambda e: orc.act())
        print(f'   random {res_r["trial_rewards"]} | explorer {res_e["trial_rewards"]} | oracle {res_o["trial_rewards"]} ({res_o["macros"]} macros) {time.time() - t0:.0f}s')
