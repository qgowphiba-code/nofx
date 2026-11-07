"""2D square-lattice HP inverse folding: exact enumeration of all conformations of a length-N chain.

Conformations are enumerated up to symmetry (first step +x, first turn, if any, to +y).
A sequence s in {H, P}^N has energy E(c) = -(number of non-bonded H-H contacts in conformation c).
Design reward for a target conformation t: Boltzmann probability of t at temperature TEMP,
p(t) = exp(-E(t)/TEMP) / sum_c exp(-E(c)/TEMP). It is 1 only when t is the unique ground state, with a gap.
"""
import math
N = 12
TEMP = 0.3
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def enumerate_confs(n=N):
    out = []
    def rec(path, occ, turned):
        if len(path) == n:
            out.append(tuple(path)); return
        x, y = path[-1]
        for d, (dx, dy) in enumerate(DIRS):
            p = (x + dx, y + dy)
            if p in occ: continue
            if not turned and dy != 0 and dy < 0: continue      # first turn goes +y
            path.append(p); occ.add(p)
            rec(path, occ, turned or dy != 0)
            path.pop(); occ.discard(p)
    rec([(0, 0), (1, 0)], {(0, 0), (1, 0)}, False)
    return out


def contacts(conf):
    pos = {p: i for i, p in enumerate(conf)}
    cs = []
    for i, (x, y) in enumerate(conf):
        for dx, dy in DIRS:
            j = pos.get((x + dx, y + dy))
            if j is not None and j > i + 1:
                cs.append((i, j))
    return cs


PAIRS = [(i, j) for i in range(N) for j in range(i + 2, N)]
PAIR_IX = {p: k for k, p in enumerate(PAIRS)}
