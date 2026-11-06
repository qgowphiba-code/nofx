"""Dimension-generic version of fasthp.py (2D square or 3D cubic HP) for offline baselines.

Same conventions as lattice_env: first bond fixed along +x, legal moves in directions() order, utility
0.5 + 0.5*contacts/ub for complete folds and 0 for trapped ones (here scored as contacts, trapped = -1,
a monotone map with identical argmax sets and ties).
"""
import random


def directions(dim):
    return tuple(tuple(s if j == a else 0 for j in range(dim)) for a in range(dim) for s in (1, -1))


def upper_bound(seq, dim):
    n = len(seq)
    return max(1, sum(2 * dim - int(i > 0) - int(i < n - 1) for i, t in enumerate(seq) if t == 'H') // 2)


class Fold:
    def __init__(self, seq, dim):
        self.seq = seq; self.n = len(seq); self.dim = dim; self.dirs = directions(dim)
        o = (0,) * dim; e = (1,) + (0,) * (dim - 1)
        self.chain = [o, e]; self.occ = {o: 0, e: 1}; self.contacts = 0; self.gains = []

    def _add(self, p, d): return tuple(a + b for a, b in zip(p, d))

    def legal(self):
        if len(self.chain) == self.n: return []
        t = self.chain[-1]
        return [d for d in self.dirs if self._add(t, d) not in self.occ]

    def push(self, d):
        p = self._add(self.chain[-1], d); idx = len(self.chain); g = 0
        if self.seq[idx] == 'H':
            for e in self.dirs:
                j = self.occ.get(self._add(p, e))
                if j is not None and j < idx - 1 and self.seq[j] == 'H': g += 1
        self.occ[p] = idx; self.chain.append(p); self.contacts += g; self.gains.append(g)

    def pop(self):
        p = self.chain.pop(); del self.occ[p]; self.contacts -= self.gains.pop()

    def value(self, depth, neutral=False):
        if len(self.chain) == self.n: return self.contacts
        lg = self.legal()
        if not lg: return -1
        if depth == 0: return 0 if neutral else self.contacts
        best = -2
        for d in lg:
            self.push(d); v = self.value(depth - 1, neutral); self.pop()
            if v > best: best = v
        return best

    def plan(self, depth, rng, neutral=False):
        lg = self.legal(); vals = []
        for d in lg:
            self.push(d); vals.append(self.value(depth - 1, neutral)); self.pop()
        m = max(vals); ties = [d for d, v in zip(lg, vals) if v == m]
        return rng.choice(ties)

    def complete(self): return len(self.chain) == self.n

    def final_contacts(self): return self.contacts if self.complete() else 0


def fold(seq, dim, depth, rng, eps=0.0, neutral=False):
    f = Fold(seq, dim)
    while True:
        lg = f.legal()
        if not lg: break
        if eps and rng.random() < eps: f.push(rng.choice(lg))
        else: f.push(f.plan(depth, rng, neutral))
    return f
