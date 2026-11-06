"""Exact short-chain oracle, 2D or 3D: V*(s) (best reachable utility) and V_rand(s) (uniform random
completion), by exhaustive DFS from s. Matches vendor hp.Oracle in 3D (tests/test_oracle.py)."""
from .hpenv import Fold, directions


class Oracle:
    def __init__(self, sequence, dim=3):
        self.sequence, self.dim, self.n = sequence, dim, len(sequence)
        self.dirs = directions(dim)
        self.ub = Fold(sequence, dim).ub
        self.cache = {}

    def values(self, chain):
        """(V*, V_rand) of the state whose placed chain is `chain` (tuple of points)."""
        chain = tuple(map(tuple, chain))
        if chain in self.cache:
            return self.cache[chain]
        f = Fold(self.sequence, self.dim, list(chain))
        if f.terminal:
            out = (f.utility(), f.utility())
        else:
            seq, n, ub, dirs = self.sequence, self.n, self.ub, self.dirs
            ch, occ = list(f.chain), dict(f.occ)

            def visit(contacts):
                idx = len(ch)
                if idx == n:
                    u = .5 + .5 * contacts / ub
                    return u, u
                tip = ch[-1]
                nexts = [p for p in (tuple(a + b for a, b in zip(tip, d)) for d in dirs) if p not in occ]
                if not nexts:
                    return 0., 0.
                best, tot = 0., 0.
                for p in nexts:
                    gain = 0
                    if seq[idx] == 'H':
                        for d in dirs:
                            j = occ.get(tuple(a + b for a, b in zip(p, d)))
                            if j is not None and j < idx - 1 and seq[j] == 'H':
                                gain += 1
                    occ[p] = idx
                    ch.append(p)
                    b, r = visit(contacts + gain)
                    ch.pop()
                    del occ[p]
                    best = b if b > best else best
                    tot += r
                return best, tot / len(nexts)
            out = visit(f.contacts)
        self.cache[chain] = out
        return out

    def vstar(self, chain):
        return self.values(chain)[0]

    def vrand(self, chain):
        return self.values(chain)[1]

    def children(self, chain):
        """[(action, child_chain, V*(child))] in legal order."""
        f = Fold(self.sequence, self.dim, list(chain))
        out = []
        for a in f.legal():
            f.push(a)
            c = tuple(f.chain)
            out.append((a, c, self.vstar(c)))
            f.pop()
        return out

    def optimal_moves(self, chain):
        kids = self.children(chain)
        m = max(v for _, _, v in kids)
        return [a for a, _, v in kids if abs(v - m) < 1e-12], kids

    def optimum_contacts(self):
        v = self.vstar(Fold(self.sequence, self.dim).key())
        return round((v - .5) * 2 * self.ub) if v > 0 else 0
