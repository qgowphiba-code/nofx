"""C-target: value rows with explicit provenance (plan 2.1).

For a recorded state s_k of an episode with final utility U (terminal at index T):
    zbar_k = 1/2 + (U - 1/2) * gamma**(T - k)          (gamma = 1 for HP)
    u_k    = Search_d(s_k; V_theta)                     (backed-up max, exact terminal values)
    y_k    = clip(lam * u_k + (1 - lam) * zbar_k, 0, 1)
Every row is sent as a provided `target`; u, zbar, lam, d, gamma, the assessor sha, episode and step are
kept in `metadata.prov` and in the local record. Record ids encode arm, lam, d and the assessor sha prefix,
so changed content never reuses an id (value/observe answers 409 to that).
"""
import random

from .hpenv import BASIC, Fold
from .planner import Heuristic, decide, search_value


class Episode:
    """A recorded fold: chains s_0..s_T (s_T terminal) of one sequence, plus collection-time search values."""

    def __init__(self, episode_id, sequence, chains, dim=3, generation=0, u=None, collector=None):
        self.id, self.sequence, self.dim = episode_id, sequence, dim
        self.chains = [tuple(map(tuple, c)) for c in chains]
        self.generation, self.u, self.collector = generation, u, collector
        last = Fold(sequence, dim, list(self.chains[-1]))
        assert last.terminal, 'an episode ends at a terminal state'
        self.U = last.utility()
        self.contacts = last.final_contacts()

    @property
    def T(self):
        return len(self.chains) - 1

    def fold(self, k):
        return Fold(self.sequence, self.dim, list(self.chains[k]))

    def zbar(self, k, gamma=1.0):
        return .5 + (self.U - .5) * gamma ** (self.T - k)

    def to_json(self):
        return {'id': self.id, 'sequence': self.sequence, 'dim': self.dim, 'generation': self.generation,
                'chains': [[list(p) for p in c] for c in self.chains], 'u': self.u, 'collector': self.collector,
                'U': self.U, 'contacts': self.contacts}

    @classmethod
    def from_json(cls, d):
        return cls(d['id'], d['sequence'], d['chains'], d['dim'], d.get('generation', 0), d.get('u'), d.get('collector'))


def random_episodes(seqs, folds_per_chain, rng, prefix, dim=3, order='chain_major', generation=0):
    """Random folds (uniform legal moves), chain-major by default."""
    plan = ([(ci, j) for ci in range(len(seqs)) for j in range(folds_per_chain)] if order == 'chain_major'
            else [(i % len(seqs), i // len(seqs)) for i in range(len(seqs) * folds_per_chain)])
    eps = []
    for ci, j in plan:
        f = Fold(seqs[ci], dim)
        chains = [f.key()]
        while not f.terminal:
            f.push(rng.choice(f.legal()))
            chains.append(f.key())
        eps.append(Episode(f'{prefix}-c{ci}-f{j}', seqs[ci], chains, dim, generation))
    return eps


def _clip(x):
    return min(1., max(0., x))


def _state_fields(fold, conditions):
    obs = fold.observation(conditions)
    return {k: obs[k] for k in ('objects', 'relations', 'conditions', 'include_positions', 'include_moments', 'policy_features')}


def terminal_rows(episodes, arm='A0', gamma=1.0, conditions=BASIC):
    """lam = 0 rows: target = zbar_k (= U for gamma = 1). One row per non-terminal recorded state."""
    rows = []
    for e in episodes:
        for k in range(e.T):
            z = e.zbar(k, gamma)
            rows.append({'record_id': f'{arm}-g{gamma:g}-{e.id}-s{k}', 'episode_id': e.id, 'step': k,
                         **_state_fields(e.fold(k), conditions), 'target': _clip(z),
                         'metadata': {'prov': {'arm': arm, 'lam': 0, 'zbar': z, 'U': e.U, 'gamma': gamma,
                                               'generation': e.generation}}})
    return rows


def assess(episodes, depth, ev, ks=None):
    """u_k = Search_depth(s_k; ev) for every non-terminal recorded state (batched across all states)."""
    folds, index = [], []
    for e in episodes:
        for k in (range(e.T) if ks is None else ks(e)):
            folds.append(e.fold(k))
            index.append((e.id, k))
    vals = search_value(folds, depth, ev) if folds else []
    return {key: v for key, v in zip(index, vals)}


def blended_rows(episodes, lam, depth, u, assessor_sha, arm='BL', gamma=1.0, conditions=BASIC):
    """y_k = clip(lam*u_k + (1-lam)*zbar_k). u: {(episode_id, k): value} from assess() or collection."""
    tag = f'{arm}-l{lam:g}-d{depth}-a{assessor_sha[:10]}-g{gamma:g}'
    rows = []
    for e in episodes:
        for k in range(e.T):
            uk, z = u[(e.id, k)], e.zbar(k, gamma)
            y = _clip(lam * uk + (1 - lam) * z)
            rows.append({'record_id': f'{tag}-{e.id}-s{k}', 'episode_id': e.id, 'step': k,
                         **_state_fields(e.fold(k), conditions), 'target': y,
                         'metadata': {'prov': {'arm': arm, 'lam': lam, 'd': depth, 'u': uk, 'zbar': z, 'U': e.U,
                                               'gamma': gamma, 'assessor': assessor_sha, 'generation': e.generation}}})
    return rows


def sibling_rows(episodes, depth, ev, assessor_sha, arm='BLG', stride=1, conditions=BASIC):
    """BLG: one row per legal child c of s_k, target = Search_{depth-1}(c; V) (exact if c is terminal),
    grouped by metadata.comparison_group = parent. Uses every `stride`-th state."""
    items = []
    for e in episodes:
        for k in range(0, e.T, stride):
            f = e.fold(k)
            for j, a in enumerate(f.legal()):
                f.push(a)
                items.append((e, k, j, f.copy()))
                f.pop()
    live = [it[3] for it in items if not it[3].terminal]
    if depth - 1 >= 1:
        vals = iter(search_value(live, depth - 1, ev))
    else:
        vals = iter(ev.evaluate([ev.payload(c) for c in live]))
    tag = f'{arm}-d{depth}-a{assessor_sha[:10]}'
    rows = []
    for e, k, j, c in items:
        v = c.utility() if c.terminal else next(vals)
        rows.append({'record_id': f'{tag}-{e.id}-s{k}-c{j}', 'episode_id': e.id, 'step': k + 1,
                     **_state_fields(c, conditions), 'target': _clip(v),
                     'metadata': {'comparison_group': f'{e.id}-s{k}',
                                  'prov': {'arm': arm, 'd': depth, 'exact': c.terminal, 'assessor': assessor_sha}}})
    return rows


def continuation_sibling_rows(episodes, stride=3, arm='ROB', conditions=BASIC):
    """Set 1's robust recipe rows: for every `stride`-th state s_k, one row per legal child c carrying a
    server-side `continuation` over c's legal successors (exact terminal_value where finished), grouped by
    parent. A terminal child gets its exact utility as a provided target."""
    rows = []
    for e in episodes:
        for k in range(0, e.T, stride):
            f = e.fold(k)
            for j, a in enumerate(f.legal()):
                f.push(a)
                row = {'record_id': f'{arm}-s{stride}-{e.id}-s{k}-c{j}', 'episode_id': e.id, 'step': k + 1,
                       **_state_fields(f, conditions), 'metadata': {'comparison_group': f'{e.id}-s{k}'}}
                if f.terminal:
                    row['target'] = f.utility()
                else:
                    succ = []
                    for b in f.legal():
                        f.push(b)
                        s = _state_fields(f, conditions)
                        if f.terminal:
                            s['terminal_value'] = f.utility()
                        succ.append(s)
                        f.pop()
                    row['continuation'] = {'maximizing': True, 'discount': 1.0, 'neutral': .5, 'successors': succ}
                rows.append(row)
                f.pop()
    return rows


def validation_episodes(episodes, fit_index, fraction=0.2, base_seed=80700):
    """20% of episodes by Random(80700 + fit_index), stratified by generation (plan Appendix A)."""
    rng = random.Random(base_seed + fit_index)
    by_gen = {}
    for e in episodes:
        by_gen.setdefault(e.generation, []).append(e.id)
    out = []
    for g in sorted(by_gen):
        ids = sorted(by_gen[g])
        out.extend(rng.sample(ids, max(1, round(fraction * len(ids)))))
    return sorted(out)


def heuristic_assessor():
    return Heuristic()


__all__ = ['Episode', 'random_episodes', 'terminal_rows', 'assess', 'blended_rows', 'sibling_rows',
           'continuation_sibling_rows', 'validation_episodes', 'decide']
