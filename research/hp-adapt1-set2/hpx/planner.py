"""C-plan: client-side depth-d MAX search with batched leaf evaluation across parallel folds.

Semantics are lattice_env.plan's: a complete successor scores 0.5 + 0.5*c/ub exactly, a trapped one
scores 0, and an unfinished node at depth 0 is a leaf scored by the evaluator. With the Heuristic
evaluator this reproduces lattice_env.plan (arm C2) move for move (tests/test_planner.py: 68/113/134).

Every decision of every active fold is expanded first; all leaves are then scored in one evaluator
call, which a learned evaluator turns into as few `value/predict` requests as the payload limits allow.
"""
import hashlib
import json
import random

from .hpenv import BASIC, Fold


# ------------------------------------------------------------------------------------- evaluators
class Heuristic:
    """Contact heuristic: the leaf's own partial score."""
    name = 'heuristic'

    def payload(self, fold):
        return fold.score()

    def evaluate(self, payloads):
        return list(payloads)


class Neutral:
    """Search without learned values: every unfinished leaf is 0.5 (lattice_env arm C1)."""
    name = 'neutral'

    def payload(self, fold):
        return None

    def evaluate(self, payloads):
        return [.5] * len(payloads)


class Model:
    """Learned leaf values through a batch predict function, cached by encoding.

    predict_fn(list_of_inputs) -> list_of_floats. `conditions` selects the condition set (BASIC for R0)."""

    def __init__(self, predict_fn, conditions=BASIC, name='model', cache=None):
        self.predict_fn, self.conditions, self.name = predict_fn, conditions, name
        self.cache = {} if cache is None else cache
        self.stats = {'leaves': 0, 'unique_sent': 0, 'cache_hits': 0, 'predict_batches': 0}

    def payload(self, fold):
        obs = fold.observation(self.conditions)
        key = hashlib.sha1(json.dumps(obs, separators=(',', ':')).encode()).hexdigest()
        return key, obs

    def evaluate(self, payloads):
        self.stats['leaves'] += len(payloads)
        todo = {}
        for key, obs in payloads:
            if key not in self.cache and key not in todo:
                todo[key] = obs
        self.stats['cache_hits'] += len(payloads) - len(todo)
        if todo:
            keys = list(todo)
            vals = self.predict_fn([todo[k] for k in keys])
            self.stats['unique_sent'] += len(keys)
            self.stats['predict_batches'] += 1
            for k, v in zip(keys, vals):
                self.cache[k] = float(v)
        return [self.cache[k] for k, _ in payloads]


class Blend:
    """Leaf blend (1 - lam) * heuristic + lam * learned (plan R3; lam = 0 is the heuristic itself)."""

    def __init__(self, model, lam):
        self.model, self.lam = model, lam
        self.name = f'blend{lam:g}({model.name})'

    def payload(self, fold):
        return fold.score(), (self.model.payload(fold) if self.lam else None)

    def evaluate(self, payloads):
        if not self.lam:
            return [h for h, _ in payloads]
        v = self.model.evaluate([m for _, m in payloads])
        return [(1 - self.lam) * h + self.lam * x for (h, _), x in zip(payloads, v)]


# ------------------------------------------------------------------------------------- search
def _build(fold, d, ev, leaves):
    if fold.complete:
        return fold.score()
    legal = fold.legal()
    if not legal:
        return 0.
    if d == 0:
        leaves.append(ev.payload(fold))
        return ~(len(leaves) - 1)          # leaf marker: negative int (distinct from float values)
    kids = []
    for a in legal:
        fold.push(a)
        kids.append(_build(fold, d - 1, ev, leaves))
        fold.pop()
    return kids


def _value(node, vals):
    if isinstance(node, float):
        return node
    if isinstance(node, int):
        return vals[~node]
    return max(_value(k, vals) for k in node)


def expand(fold, depth, ev):
    """Root expansion for one decision: (legal actions, per-action subtrees, leaf payloads)."""
    legal = fold.legal()
    leaves, subtrees = [], []
    for a in legal:
        fold.push(a)
        subtrees.append(_build(fold, depth - 1, ev, leaves))
        fold.pop()
    return legal, subtrees, leaves


def decide(folds, depth, ev):
    """One decision for each fold. Returns per fold (legal, action_values, n_leaves)."""
    exps = [expand(f, depth, ev) for f in folds]
    flat = [p for _, _, leaves in exps for p in leaves]
    vals = ev.evaluate(flat) if flat else []
    out, k = [], 0
    for legal, subtrees, leaves in exps:
        v = vals[k:k + len(leaves)]
        k += len(leaves)
        out.append((legal, [_value(t, v) for t in subtrees], len(leaves)))
    return out


def search_value(folds, depth, ev):
    """u = Search_depth(s): the backed-up max at the root (exact for terminal roots)."""
    live = [i for i, f in enumerate(folds) if not f.terminal]
    res = decide([folds[i] for i in live], depth, ev)
    out = [f.utility() if f.terminal else None for f in folds]
    for i, (_, vals, _) in zip(live, res):
        out[i] = max(vals)
    return out


def choose(legal, vals, tie, rng, placed):
    m = max(vals)
    ties = [a for a, v in zip(legal, vals) if v == m]
    if tie == 'first':
        return ties[0], len(ties)
    if tie == 'stage_m':
        return random.Random(placed).choice(ties), len(ties)
    return rng.choice(ties), len(ties)


def run_folds(seqs, depth, ev, rngs=None, eps=0.0, tie='rng', dim=3, record=False):
    """Greedy (or epsilon-greedy) folds of every sequence, decided in lockstep.

    rngs: one random.Random per sequence (tie-breaks and epsilon). With eps > 0 the search still runs
    at every step (so u is recorded for every state), then the epsilon coin is drawn, as in plan B1.
    Returns a list of dicts: final Fold, contacts, complete, and (if record) per-step trajectory."""
    folds = [Fold(s, dim) for s in seqs]
    rngs = rngs or [random.Random(i) for i in range(len(seqs))]
    trajs = [[] for _ in seqs]
    leaves = [0] * len(seqs)
    ties_seen = [0] * len(seqs)
    while True:
        active = [i for i, f in enumerate(folds) if not f.terminal]
        if not active:
            break
        res = decide([folds[i] for i in active], depth, ev)
        for i, (legal, vals, nl) in zip(active, res):
            f, rng = folds[i], rngs[i]
            leaves[i] += nl
            explored = False
            if eps and rng.random() < eps:
                a, explored = rng.choice(legal), True
            else:
                a, nt = choose(legal, vals, tie, rng, len(f.chain))
                ties_seen[i] += nt > 1
            if record:
                trajs[i].append({'chain': [list(p) for p in f.chain], 'step': len(f.chain) - 2, 'u': max(vals),
                                 'action': list(a), 'explored': explored,
                                 'action_values': [[list(x), v] for x, v in zip(legal, vals)]})
            f.push(a)
    out = []
    for i, f in enumerate(folds):
        r = {'sequence': seqs[i], 'contacts': f.final_contacts(), 'complete': f.complete, 'utility': f.utility(),
             'leaf_evaluations': leaves[i], 'decisions': len(f.chain) - 2, 'tie_decisions': ties_seen[i],
             'chain': [list(p) for p in f.chain]}
        if record:
            r['trajectory'] = trajs[i]
        out.append(r)
    return out
