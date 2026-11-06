import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hpx.hpenv import Fold  # noqa: E402
from hpx.planner import Blend, Heuristic, Model, Neutral, decide, run_folds, search_value  # noqa: E402
from hpx.vendor import lattice_env, liu_panel  # noqa: E402
from fasthp_nd import fold as nd_fold  # noqa: E402  (plan/analysis, validated offline baseline)


def obs_heuristic(inputs):
    """Mock learned values: the contact heuristic recomputed from the 3D observation alone."""
    out = []
    for obs in inputs:
        types = [o['type'][0] for o in obs['objects']]
        n = len(types)
        ub = max(1, sum(6 - int(i > 0) - int(i < n - 1) for i, t in enumerate(types) if t == 'H') // 2)
        c = sum(r['type'] == 'hh_contact' for r in obs['relations'])
        out.append(.5 + .5 * c / ub)
    return out


class TestPlanner(unittest.TestCase):
    def test_reproduces_set1_liu_totals(self):
        """Stage M seeding (fresh Random(len(chain)) per decision): 68 / 113 / 134 (plan Appendix C)."""
        seqs = [s for _, s in liu_panel()]
        totals = [sum(r['contacts'] for r in run_folds(seqs, d, Heuristic(), tie='stage_m')) for d in (1, 2, 3)]
        self.assertEqual(totals, [68, 113, 134])

    def test_matches_lattice_env_plan_with_rng(self):
        for seq in lattice_env.sequences(4242, [12, 20, 30]):
            for depth in (1, 2, 3):
                ref = lattice_env.episode(seq, 3, 'C2', depth, seed=17)[0]
                got = run_folds([seq], depth, Heuristic(), rngs=[random.Random(17)])[0]
                self.assertEqual(got['contacts'], ref['contacts'])
                moves = [[b - a for a, b in zip(p, q)] for p, q in zip(got['chain'][1:], got['chain'][2:])]
                self.assertEqual(moves, ref['actions'])

    def test_matches_fasthp_nd_with_eps_2d_and_3d(self):
        for dim in (2, 3):
            for seq in lattice_env.sequences(4343, [16, 24]):
                for seed in range(3):
                    ref = nd_fold(seq, dim, 2, random.Random(seed), 0.2)
                    got = run_folds([seq], 2, Heuristic(), rngs=[random.Random(seed)], eps=0.2, dim=dim)[0]
                    self.assertEqual(got['chain'], [list(p) for p in ref.chain])

    def test_model_with_heuristic_values_equals_heuristic(self):
        seqs = [s for _, s in liu_panel()][:4]
        a = run_folds(seqs, 2, Heuristic(), rngs=[random.Random(i) for i in range(4)])
        m = Model(obs_heuristic)
        b = run_folds(seqs, 2, m, rngs=[random.Random(i) for i in range(4)])
        self.assertEqual([r['chain'] for r in a], [r['chain'] for r in b])
        self.assertGreater(m.stats['cache_hits'], 0)
        self.assertLessEqual(m.stats['unique_sent'], m.stats['leaves'])

    def test_blend_zero_is_heuristic_and_neutral_baseline(self):
        seq = liu_panel()[3][1]
        a = run_folds([seq], 3, Heuristic(), rngs=[random.Random(3)])[0]
        b = run_folds([seq], 3, Blend(Model(obs_heuristic), 0.0), rngs=[random.Random(3)])[0]
        self.assertEqual(a['chain'], b['chain'])
        n = run_folds([seq], 1, Neutral(), rngs=[random.Random(3)])[0]
        self.assertTrue(n['complete'] or n['contacts'] == 0)

    def test_search_value_depth1_is_max_child(self):
        seq = lattice_env.sequences(4444, [14])[0]
        f = Fold(seq)
        for a in ((0, 1, 0), (-1, 0, 0)):
            f.push(a)
        u = search_value([f], 1, Heuristic())[0]
        kids = []
        for a in f.legal():
            f.push(a)
            kids.append(f.utility() if f.terminal else f.score())
            f.pop()
        self.assertEqual(u, max(kids))
        legal, vals, nl = decide([f], 1, Heuristic())[0]
        self.assertEqual(max(vals), u)
        self.assertEqual(nl, len(legal))


if __name__ == '__main__':
    unittest.main()
