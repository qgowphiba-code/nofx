"""Fast incremental HP lattice fold (2D square / 3D cubic) with set-1-exact encodings.

`Fold` mirrors vendor lattice_env.State (first bond fixed along +x, legal moves in directions() order,
utility 0.5 + 0.5*contacts/ub for a complete fold and 0 for a trapped one). `observation()` reproduces
vendor hp.observation byte for byte in 3D (tests/test_hpenv.py), and `features()` reproduces
State.features() exactly. 2D uses the same object graph with a 2D tip frame (third coordinate 0).
"""
import math

BASIC = ('placed_fraction', 'remaining_fraction', 'tip_is_hydrophobic', 'next_is_hydrophobic')
FEATURES = ('placed_fraction', 'remaining_fraction', 'hydrophobic_fraction', 'placed_hydrophobic_fraction',
            'remaining_hydrophobic_fraction', 'contact_capacity_fraction', 'h_contact_degree_fraction',
            'legal_direction_fraction', 'tip_h_neighbor_fraction', 'tip_is_hydrophobic', 'next_is_hydrophobic',
            'mean_occupied_degree_fraction', 'radius_over_chain_length', 'end_to_end_over_chain_length')


def directions(dim):
    return tuple(tuple(s if j == a else 0 for j in range(dim)) for a in range(dim) for s in (1, -1))


def upper_bound(seq, dim=3):
    n = len(seq)
    return max(1, sum(2 * dim - int(i > 0) - int(i < n - 1) for i, t in enumerate(seq) if t == 'H') // 2)


def _add(p, d):
    return tuple(a + b for a, b in zip(p, d))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


class Fold:
    __slots__ = ('seq', 'n', 'dim', 'dirs', 'chain', 'occ', 'contacts', 'gains', 'hh', 'ub', 'h_total')

    def __init__(self, seq, dim=3, chain=None):
        self.seq, self.n, self.dim = seq, len(seq), dim
        self.dirs = directions(dim)
        self.ub = upper_bound(seq, dim)
        self.h_total = seq.count('H')
        o, e = (0,) * dim, (1,) + (0,) * (dim - 1)
        self.chain, self.occ = [o, e], {o: 0, e: 1}
        self.contacts, self.gains, self.hh = 0, [], []
        if chain is not None:
            chain = [tuple(p) for p in chain]
            assert tuple(chain[:2]) == (o, e), 'chains must start with the fixed first bond'
            for p in chain[2:]:
                self.push(_sub(p, self.chain[-1]))

    # ------------------------------------------------------------- moves
    def legal(self):
        if len(self.chain) == self.n:
            return []
        t = self.chain[-1]
        return [d for d in self.dirs if _add(t, d) not in self.occ]

    def push(self, d):
        p = _add(self.chain[-1], d)
        idx = len(self.chain)
        assert p not in self.occ and sum(map(abs, d)) == 1
        js = []
        if self.seq[idx] == 'H':
            for e in self.dirs:
                j = self.occ.get(_add(p, e))
                if j is not None and j < idx - 1 and self.seq[j] == 'H':
                    js.append(j)
        js.sort()
        self.occ[p] = idx
        self.chain.append(p)
        self.contacts += len(js)
        self.gains.append(len(js))
        self.hh.extend((j, idx) for j in js)

    def pop(self):
        p = self.chain.pop()
        del self.occ[p]
        g = self.gains.pop()
        self.contacts -= g
        if g:
            del self.hh[-g:]

    def copy(self):
        f = Fold.__new__(Fold)
        f.seq, f.n, f.dim, f.dirs, f.ub, f.h_total = self.seq, self.n, self.dim, self.dirs, self.ub, self.h_total
        f.chain, f.occ = list(self.chain), dict(self.occ)
        f.contacts, f.gains, f.hh = self.contacts, list(self.gains), list(self.hh)
        return f

    # ------------------------------------------------------------- status and values
    @property
    def complete(self):
        return len(self.chain) == self.n

    @property
    def terminal(self):
        return self.complete or not self.legal()

    @property
    def trapped(self):
        return not self.complete and not self.legal()

    def score(self):
        """0.5 + 0.5*contacts/ub (lattice_env partial_score for an untrapped state)."""
        return .5 + .5 * self.contacts / self.ub

    def utility(self):
        """Exact terminal value: complete -> score, trapped -> 0. Only valid at terminal states."""
        return self.score() if self.complete else 0.

    def partial_score(self):
        return 0. if self.trapped else self.score()

    def final_contacts(self):
        return self.contacts if self.complete else 0

    def key(self):
        return tuple(self.chain)

    # ------------------------------------------------------------- encodings
    def features(self, names=FEATURES):
        """Exactly lattice_env.State.features() (same arithmetic order), restricted to `names`."""
        n, placed, dim, seq = self.n, len(self.chain), self.dim, self.seq
        remaining = n - placed
        out = {}
        need = set(names)
        if 'placed_fraction' in need: out['placed_fraction'] = placed / n
        if 'remaining_fraction' in need: out['remaining_fraction'] = remaining / n
        if 'hydrophobic_fraction' in need: out['hydrophobic_fraction'] = self.h_total / n
        if 'placed_hydrophobic_fraction' in need: out['placed_hydrophobic_fraction'] = seq[:placed].count('H') / placed
        if 'remaining_hydrophobic_fraction' in need:
            out['remaining_hydrophobic_fraction'] = seq[placed:].count('H') / max(1, remaining)
        if 'contact_capacity_fraction' in need: out['contact_capacity_fraction'] = self.contacts / self.ub
        if 'h_contact_degree_fraction' in need:
            out['h_contact_degree_fraction'] = 2 * self.contacts / max(1, (2 * dim - 2) * self.h_total + 2)
        if 'legal_direction_fraction' in need: out['legal_direction_fraction'] = len(self.legal()) / (2 * dim)
        if 'tip_h_neighbor_fraction' in need:
            tip, nb = self.chain[-1], 0
            for d in self.dirs:
                index = self.occ.get(_add(tip, d))
                if index is not None:
                    nb += index < placed - 2 and seq[index] == 'H'
            out['tip_h_neighbor_fraction'] = nb / (2 * dim - 1)
        if 'tip_is_hydrophobic' in need: out['tip_is_hydrophobic'] = float(seq[placed - 1] == 'H')
        if 'next_is_hydrophobic' in need: out['next_is_hydrophobic'] = float(remaining > 0 and seq[placed] == 'H')
        if 'mean_occupied_degree_fraction' in need:
            occ = self.occ
            degrees = [sum(_add(p, d) in occ for d in self.dirs) for p in self.chain]
            out['mean_occupied_degree_fraction'] = sum(degrees) / (placed * 2 * dim)
        if 'radius_over_chain_length' in need:
            center = tuple(sum(p[k] for p in self.chain) / placed for k in range(dim))
            radius = math.sqrt(sum(sum((p[k] - center[k]) ** 2 for k in range(dim)) for p in self.chain) / placed)
            out['radius_over_chain_length'] = radius / n
        if 'end_to_end_over_chain_length' in need:
            out['end_to_end_over_chain_length'] = sum(abs(a - b) for a, b in zip(self.chain[0], self.chain[-1])) / n
        return {k: out[k] for k in names}

    def frame(self):
        """Tip frame: x = last bond, y = most recent bond perpendicular to x, z = x cross y (3D)."""
        ch = self.chain
        x = _sub(ch[-1], ch[-2])
        y = None
        for i in range(len(ch) - 1, 0, -1):
            v = _sub(ch[i], ch[i - 1])
            if _dot(x, v) == 0:
                y = v
                break
        if self.dim == 3:
            if y is None:
                y = next(v for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1)) if _dot(x, v) == 0)
            return (x, y, _cross(x, y))
        if y is None:
            y = (-x[1], x[0])
        return (x, y)

    def observation(self, conditions=BASIC):
        """Set-1 object graph (hp.observation, arm object4_contacts) in the chain-tip frame."""
        ch, seq, placed = self.chain, self.seq, len(self.chain)
        tip = ch[-1]
        basis = self.frame()
        objects = []
        for i, typ in enumerate(seq):
            if i < placed:
                r = _sub(ch[i], tip)
                pos = [_dot(r, b) for b in basis]
                if self.dim == 2:
                    pos.append(0)
                kind = typ + ('_tip' if i == placed - 1 else '')
            else:
                pos, kind = [0, 0, 0], typ + '_unplaced'
            objects.append({'id': str(i), 'type': kind, 'position': pos})
        relations = [{'source': str(i), 'target': str(i + 1), 'type': 'next'} for i in range(self.n - 1)]
        relations.extend({'source': str(j), 'target': str(i), 'type': 'hh_contact'} for j, i in self.hh)
        return {'objects': objects, 'relations': relations, 'include_positions': True,
                'include_moments': False, 'conditions': self.features(conditions), 'policy_features': {}}


def from_state(state):
    """Fold from a vendor lattice_env.State (or any object with sequence/chain/dim)."""
    return Fold(state.sequence, state.dim, list(state.chain))
