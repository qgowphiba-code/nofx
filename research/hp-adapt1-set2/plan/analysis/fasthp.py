"""Fast, incremental re-implementation of lattice_env.plan (arm 'C2', contact heuristic) for offline checks.

Score convention: trapped terminal = -1, otherwise current contacts. This is a monotone map of the package's
utility (0 for trapped, 0.5 + 0.5*c/ub otherwise), so argmax sets and ties are identical.
Legal-move order matches lattice_env.directions(3), so rng.choice(ties) reproduces the package exactly.
"""
import random

DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def upper_bound(seq, dim=3):
    n = len(seq)
    return max(1, sum(2 * dim - int(i > 0) - int(i < n - 1) for i, t in enumerate(seq) if t == 'H') // 2)


class Fold:
    def __init__(self, seq):
        self.seq = seq; self.n = len(seq)
        self.chain = [(0, 0, 0), (1, 0, 0)]
        self.occ = {(0, 0, 0): 0, (1, 0, 0): 1}
        self.contacts = 0
        self.gains = []

    def legal(self):
        if len(self.chain) == self.n: return []
        x, y, z = self.chain[-1]
        return [d for d in DIRS if (x + d[0], y + d[1], z + d[2]) not in self.occ]

    def push(self, d):
        x, y, z = self.chain[-1]; p = (x + d[0], y + d[1], z + d[2]); idx = len(self.chain)
        g = 0
        if self.seq[idx] == 'H':
            for e in DIRS:
                j = self.occ.get((p[0] + e[0], p[1] + e[1], p[2] + e[2]))
                if j is not None and j < idx - 1 and self.seq[j] == 'H': g += 1
        self.occ[p] = idx; self.chain.append(p); self.contacts += g; self.gains.append(g)

    def pop(self):
        p = self.chain.pop(); del self.occ[p]; self.contacts -= self.gains.pop()

    def complete(self): return len(self.chain) == self.n

    def value(self, depth):
        """lattice_env.plan's value(): terminal first (complete -> contacts, trapped -> -1), then depth-0 leaf."""
        if len(self.chain) == self.n: return self.contacts
        lg = self.legal()
        if not lg: return -1
        if depth == 0: return self.contacts
        best = -2
        for d in lg:
            self.push(d); v = self.value(depth - 1); self.pop()
            if v > best: best = v
        return best

    def plan(self, depth, rng=None):
        lg = self.legal(); vals = []
        for d in lg:
            self.push(d); vals.append(self.value(depth - 1)); self.pop()
        m = max(vals); ties = [d for d, v in zip(lg, vals) if v == m]
        return ties[0] if rng is None else rng.choice(ties)

    def final_contacts(self):
        return self.contacts if self.complete() else 0


def greedy_fold(seq, depth, tie='stage_m', rng=None, eps=0.0):
    """tie: 'stage_m' -> fresh random.Random(len(chain)) per decision (stage_m.py / stage_l.py);
            'first'   -> ties[0] (stage_p.py, lattice_env.plan with rng=None);
            'rng'     -> the given persistent rng (stage_d.py control: also used for epsilon)."""
    f = Fold(seq)
    while True:
        lg = f.legal()
        if not lg: break
        if tie == 'rng':
            if rng.random() < eps: f.push(rng.choice(lg)); continue
            f.push(f.plan(depth, rng))
        elif tie == 'first':
            f.push(f.plan(depth, None))
        else:
            f.push(f.plan(depth, random.Random(len(f.chain))))
    return f
