"""E0: gravity four-in-a-row on a small board (default 5 wide x 4 high), exact solver, heuristic,
object-graph encoding, and a batched minimax planner (MAX when the first player moves, MIN otherwise).

Values are from the first player's (X's) view: win 1, draw 1/2, loss 0 (plan section 9).
A position is a tuple of column strings, bottom to top, e.g. ('XO', '', 'X', '', '').
"""
import random
import sys

W, H = 5, 4


def lines(w=W, h=H):
    out = []
    for r in range(h):
        for c in range(w - 3):
            out.append(tuple((c + i, r) for i in range(4)))
    for c in range(w):
        for r in range(h - 3):
            out.append(tuple((c, r + i) for i in range(4)))
    for c in range(w - 3):
        for r in range(h - 3):
            out.append(tuple((c + i, r + i) for i in range(4)))
            out.append(tuple((c + 3 - i, r + i) for i in range(4)))
    return tuple(out)


LINES = lines()
LINES_THROUGH = {}
for _ln in LINES:
    for _cell in _ln:
        LINES_THROUGH.setdefault(_cell, []).append(_ln)


class Pos:
    __slots__ = ('cols', 'w', 'h', 'n', 'last', '_winner')

    def __init__(self, cols=None, w=W, h=H, last=None):
        self.w, self.h = w, h
        self.cols = tuple(cols) if cols is not None else ('',) * w
        self.n = sum(len(c) for c in self.cols)
        self.last = last
        self._winner = None

    @property
    def to_move(self):
        return 'X' if self.n % 2 == 0 else 'O'

    def cell(self, c, r):
        col = self.cols[c]
        return col[r] if r < len(col) else None

    def winner(self):
        """'X', 'O' or None. With a known last move only its lines can be complete; a position rebuilt from
        column strings (last unknown) scans every line."""
        if self._winner is None:
            self._winner = ''
            if self.last is not None:
                c = self.last
                r = len(self.cols[c]) - 1
                cand = LINES_THROUGH[(c, r)]
            else:
                cand = LINES
            for ln in cand:
                first = self.cell(*ln[0])
                if first and all(self.cell(x, y) == first for x, y in ln[1:]):
                    self._winner = first
                    break
        return self._winner or None

    @property
    def terminal(self):
        return self.winner() is not None or self.n == self.w * self.h

    def value(self):
        """Exact terminal value from X's view."""
        w = self.winner()
        return 1. if w == 'X' else 0. if w == 'O' else .5

    def legal(self):
        if self.terminal:
            return []
        return [c for c in range(self.w) if len(self.cols[c]) < self.h]

    def play(self, c):
        cols = list(self.cols)
        cols[c] = cols[c] + self.to_move
        return Pos(cols, self.w, self.h, last=c)

    def key(self):
        return '|'.join(self.cols)

    # ------------------------------------------------------------------ heuristic and encoding
    def heuristic(self, weights=(0., 1., 4., 16.)):
        """Weighted count of open lines, mapped to [0, 1] from X's view (terminal positions exact)."""
        if self.terminal:
            return self.value()
        s = 0.
        for ln in LINES:
            cells = [self.cell(x, y) for x, y in ln]
            x, o = cells.count('X'), cells.count('O')
            if x and not o:
                s += weights[x]
            elif o and not x:
                s -= weights[o]
        return min(1., max(0., .5 + s / 100.))

    def observation(self):
        """Cells as objects (X, O, empty) at [col, row, 0]; relations link like pieces adjacent along the
        4 line directions; conditions: side to move and move fraction."""
        objects, relations = [], []
        for c in range(self.w):
            for r in range(self.h):
                p = self.cell(c, r)
                objects.append({'id': f'{c}_{r}', 'type': p or ('E_playable' if r == len(self.cols[c]) else 'E'),
                                'position': [c, r, 0]})
        for c in range(self.w):
            for r in range(self.h):
                p = self.cell(c, r)
                if p is None:
                    continue
                for dc, dr, t in ((1, 0, 'h'), (0, 1, 'v'), (1, 1, 'd1'), (-1, 1, 'd2')):
                    c2, r2 = c + dc, r + dr
                    if 0 <= c2 < self.w and 0 <= r2 < self.h and self.cell(c2, r2) == p:
                        relations.append({'source': f'{c}_{r}', 'target': f'{c2}_{r2}', 'type': f'{p}_{t}'})
        return {'objects': objects, 'relations': relations, 'include_positions': True, 'include_moments': False,
                'conditions': {'x_to_move': float(self.to_move == 'X'), 'move_fraction': self.n / (self.w * self.h)},
                'policy_features': {}}


class Solver:
    """Exact minimax values from X's view for every position reachable from the start (memoized)."""

    def __init__(self, w=W, h=H):
        self.w, self.h = w, h
        self.memo = {}
        sys.setrecursionlimit(10000)

    def value(self, pos):
        k = pos.key()
        v = self.memo.get(k)
        if v is not None:
            return v
        if pos.terminal:
            v = pos.value()
        else:
            vals = [self.value(pos.play(c)) for c in pos.legal()]
            v = max(vals) if pos.to_move == 'X' else min(vals)
        self.memo[k] = v
        return v

    def best_moves(self, pos):
        vals = {c: self.value(pos.play(c)) for c in pos.legal()}
        target = max(vals.values()) if pos.to_move == 'X' else min(vals.values())
        return [c for c, v in vals.items() if v == target], vals


def random_game(rng, w=W, h=H):
    p = Pos(w=w, h=h)
    out = [p]
    while not p.terminal:
        p = p.play(rng.choice(p.legal()))
        out.append(p)
    return out


# ------------------------------------------------------------------------------------- planner
class GameHeuristic:
    name = 'line_heuristic'

    def payload(self, pos):
        return pos.heuristic()

    def evaluate(self, payloads):
        return list(payloads)


class GameNeutral:
    name = 'neutral'

    def payload(self, pos):
        return None

    def evaluate(self, payloads):
        return [.5] * len(payloads)


def _build(pos, d, ev, leaves):
    if pos.terminal:
        return pos.value()
    if d == 0:
        leaves.append(ev.payload(pos))
        return ~(len(leaves) - 1)
    return (pos.to_move == 'X', [_build(pos.play(c), d - 1, ev, leaves) for c in pos.legal()])


def _value(node, vals):
    if isinstance(node, float):
        return node
    if isinstance(node, int):
        return vals[~node]
    maximizing, kids = node
    vs = [_value(k, vals) for k in kids]
    return max(vs) if maximizing else min(vs)


def decide(positions, depth, ev):
    """Per position: (legal moves, child values from X's view, n leaves)."""
    exps = []
    for p in positions:
        leaves = []
        legal = p.legal()
        subs = [_build(p.play(c), depth - 1, ev, leaves) for c in legal]
        exps.append((legal, subs, leaves))
    flat = [x for _, _, lv in exps for x in lv]
    vals = ev.evaluate(flat) if flat else []
    out, k = [], 0
    for legal, subs, lv in exps:
        v = vals[k:k + len(lv)]
        k += len(lv)
        out.append((legal, [_value(s, v) for s in subs], len(lv)))
    return out


def choose(pos, legal, vals, rng):
    target = max(vals) if pos.to_move == 'X' else min(vals)
    return rng.choice([c for c, v in zip(legal, vals) if v == target])


def search_value(positions, depth, ev):
    live = [i for i, p in enumerate(positions) if not p.terminal]
    res = decide([positions[i] for i in live], depth, ev)
    out = [p.value() if p.terminal else None for p in positions]
    for i, (legal, vals, _) in zip(live, res):
        out[i] = max(vals) if positions[i].to_move == 'X' else min(vals)
    return out


class GameModel:
    """Learned leaf values via a batch predict function, cached by encoding (cf. planner.Model)."""

    def __init__(self, predict_fn, name='model'):
        self.predict_fn, self.name = predict_fn, name
        self.cache = {}
        self.stats = {'leaves': 0, 'unique_sent': 0}

    def payload(self, pos):
        return pos.key(), pos

    def evaluate(self, payloads):
        self.stats['leaves'] += len(payloads)
        todo = {}
        for k, p in payloads:
            if k not in self.cache and k not in todo:
                todo[k] = p.observation()
        if todo:
            keys = list(todo)
            for k, v in zip(keys, self.predict_fn([todo[k] for k in keys])):
                self.cache[k] = float(v)
            self.stats['unique_sent'] += len(keys)
        return [self.cache[k] for k, _ in payloads]


# ------------------------------------------------------------------------------------- suite and grading
def build_suite(solver, n_random=300, n_optimal=40, seed=82100, size=800):
    """Held-out positions: random games + solver-optimal lines (random tie-breaks); informative = not
    every legal move is optimal for the side to move. Positions are stored by column strings."""
    rng = random.Random(seed)
    seen, pool = set(), []

    def add(p):
        if p.terminal or p.key() in seen:
            return
        seen.add(p.key())
        best, vals = solver.best_moves(p)
        if len(best) < len(vals):
            pool.append({'cols': list(p.cols), 'value': solver.value(p), 'best': best,
                         'child_values': {str(c): v for c, v in vals.items()}})
    for _ in range(n_random):
        for p in random_game(rng)[:-1]:
            add(p)
    for _ in range(n_optimal):
        p = Pos()
        while not p.terminal:
            add(p)
            best, _ = solver.best_moves(p)
            p = p.play(rng.choice(best))
    rng.shuffle(pool)
    return {'w': W, 'h': H, 'seed': seed, 'positions': sorted(pool[:size], key=lambda d: (sum(map(len, d['cols'])), d['cols']))}


def score(suite, ev, depth):
    """Best-move rate vs the solver (ties averaged), regret, and W/D/L accuracy of the position's own value."""
    pos = [Pos(d['cols']) for d in suite['positions']]
    res = decide(pos, depth, ev)
    hits, regrets = [], []
    for p, d, (legal, vals, _) in zip(pos, suite['positions'], res):
        target = max(vals) if p.to_move == 'X' else min(vals)
        chosen = [c for c, v in zip(legal, vals) if v == target]
        hits.append(sum(c in d['best'] for c in chosen) / len(chosen))
        sign = 1 if p.to_move == 'X' else -1
        regrets.append(sum(sign * (d['value'] - d['child_values'][str(c)]) for c in chosen) / len(chosen))
    own = ev.evaluate([ev.payload(p) for p in pos])
    wdl = [min((0., .5, 1.), key=lambda x: abs(x - v)) == d['value'] for v, d in zip(own, suite['positions'])]
    return {'depth': depth, 'best_move_rate': sum(hits) / len(hits), 'mean_regret': sum(regrets) / len(regrets),
            'wdl_accuracy': sum(wdl) / len(wdl), 'n': len(hits)}


def episodes(n_games, rng):
    return [[p.cols for p in random_game(rng)] for _ in range(n_games)]


if __name__ == '__main__':
    import time
    t0 = time.time()
    s = Solver()
    v = s.value(Pos())
    print(f'{W}x{H}: start value {v}, {len(s.memo)} positions, {time.time() - t0:.1f}s')
    print('best first moves', s.best_moves(Pos()))
    print('example random game length', len(random_game(random.Random(1))))
