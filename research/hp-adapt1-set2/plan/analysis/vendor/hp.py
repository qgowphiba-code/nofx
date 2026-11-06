"""Local HP helpers: fair object-graph observation (object4_contacts arm, as in Nov 4-5),
exact oracle values for short chains, random folds. No network."""
import random
from functools import lru_cache
from lattice_env import State, distance, sequences

BASIC = ('placed_fraction', 'remaining_fraction', 'tip_is_hydrophobic', 'next_is_hydrophobic')


def sub(a, b): return tuple(x - y for x, y in zip(a, b))
def dot(a, b): return sum(x * y for x, y in zip(a, b))
def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def observation(s):
    """Same encoding as representation_source.observation(s, 'object4_contacts')."""
    f = s.features()
    tip = s.chain[-1]; x = sub(tip, s.chain[-2]); y = None
    for a, b in reversed(list(zip(s.chain, s.chain[1:]))):
        v = sub(b, a)
        if dot(x, v) == 0: y = v; break
    if y is None: y = next(v for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1)) if dot(x, v) == 0)
    z = cross(x, y); basis = (x, y, z)
    objects = []
    for i, typ in enumerate(s.sequence):
        placed = i < len(s.chain)
        kind = typ + ('_tip' if i == len(s.chain) - 1 else '' if placed else '_unplaced')
        pos = [dot(sub(s.chain[i], tip), b) for b in basis] if placed else [0, 0, 0]
        objects.append({'id': str(i), 'type': kind, 'position': pos})
    relations = [{'source': str(i), 'target': str(i + 1), 'type': 'next'} for i in range(len(objects) - 1)]
    for i, p in enumerate(s.chain):
        for j, q in enumerate(s.chain[:max(0, i - 1)]):
            if s.sequence[i] == s.sequence[j] == 'H' and distance(p, q) == 1:
                relations.append({'source': str(j), 'target': str(i), 'type': 'hh_contact'})
    return {'objects': objects, 'relations': relations, 'include_positions': True,
            'include_moments': False, 'conditions': {k: f[k] for k in BASIC}, 'policy_features': {}}


DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


class Oracle:
    """Exact V*(s) (best reachable utility) and V_rand(s) (uniform random completion), one DFS per state."""
    def __init__(self, sequence):
        self.sequence = sequence
        self.n = len(sequence)
        self.ub = State.initial(sequence).upper_bound
        self.cache = {}

    def values(self, s):
        key = s.chain
        if key in self.cache: return self.cache[key]
        seq, n, ub = self.sequence, self.n, self.ub
        chain = list(s.chain); occ = {p: i for i, p in enumerate(chain)}

        def visit(contacts):
            idx = len(chain)
            if idx == n:
                u = .5 + .5 * contacts / ub
                return u, u
            tip = chain[-1]
            nexts = [(tip[0]+d[0], tip[1]+d[1], tip[2]+d[2]) for d in DIRS]
            nexts = [p for p in nexts if p not in occ]
            if not nexts: return 0., 0.
            best = 0.; tot = 0.
            for p in nexts:
                gain = 0
                if seq[idx] == 'H':
                    for d in DIRS:
                        j = occ.get((p[0]+d[0], p[1]+d[1], p[2]+d[2]))
                        if j is not None and j < idx - 1 and seq[j] == 'H': gain += 1
                occ[p] = idx; chain.append(p)
                b, r = visit(contacts + gain)
                chain.pop(); del occ[p]
                best = max(best, b); tot += r
            return best, tot / len(nexts)

        out = (s.utility(), s.utility()) if s.terminal else visit(s.contacts)
        self.cache[key] = out
        return out

    def vstar(self, s): return self.values(s)[0]
    def vrand(self, s): return self.values(s)[1]

    def optimal_moves(self, s):
        vals = {a: self.vstar(s.step(a)) for a in s.legal}
        m = max(vals.values())
        return {a for a, v in vals.items() if abs(v - m) < 1e-12}, vals


def random_fold(sequence, rng):
    s = State.initial(sequence); states = [s]
    while not s.terminal:
        s = s.step(rng.choice(s.legal)); states.append(s)
    return states


def chains(seed, n, lengths):
    return sequences(seed, [lengths[i % len(lengths)] for i in range(n)])
