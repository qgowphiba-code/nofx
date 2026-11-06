import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hpx import targets as T  # noqa: E402
from hpx.planner import Heuristic  # noqa: E402
from hpx.suites import set1_chains  # noqa: E402


class TestTargets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train, _ = set1_chains()
        cls.eps = T.random_episodes(train, 8, random.Random(61107), 'a3d1')

    def test_stage_a_reconstruction_has_816_rows(self):
        self.assertEqual(len(self.eps), 96)
        self.assertEqual(len(T.terminal_rows(self.eps)), 816)

    def test_lambda0_is_outcome_and_lambda1_d1_is_continuation(self):
        h = Heuristic()
        u = T.assess(self.eps[:10], 1, h)
        z0 = T.blended_rows(self.eps[:10], 0.0, 1, u, 'a' * 64)
        self.assertTrue(all(r['target'] == next(e.U for e in self.eps if e.id == r['episode_id']) for r in z0))
        b1 = T.blended_rows(self.eps[:10], 1.0, 1, u, 'a' * 64)
        for r in b1[:40]:
            e = next(e for e in self.eps if e.id == r['episode_id'])
            f = e.fold(r['step'])
            kids = []
            for a in f.legal():
                f.push(a)
                kids.append(f.utility() if f.terminal else f.score())
                f.pop()
            self.assertEqual(r['target'], max(kids))
        mix = T.blended_rows(self.eps[:10], 0.75, 1, u, 'a' * 64)
        for r in mix:
            p = r['metadata']['prov']
            self.assertAlmostEqual(r['target'], min(1, max(0, .75 * p['u'] + .25 * p['zbar'])), places=12)

    def test_ids_encode_settings_and_are_unique(self):
        u = T.assess(self.eps[:5], 1, Heuristic())
        a = T.blended_rows(self.eps[:5], .75, 1, u, 'a' * 64)
        b = T.blended_rows(self.eps[:5], .5, 1, u, 'a' * 64)
        c = T.blended_rows(self.eps[:5], .75, 1, u, 'b' * 64)
        ids = [r['record_id'] for r in a + b + c + T.terminal_rows(self.eps[:5])]
        self.assertEqual(len(ids), len(set(ids)))

    def test_sibling_and_continuation_rows(self):
        sib = T.sibling_rows(self.eps[:6], 2, Heuristic(), 'c' * 64, stride=3)
        groups = {}
        for r in sib:
            groups.setdefault(r['metadata']['comparison_group'], []).append(r)
        self.assertTrue(all(len(g) >= 1 for g in groups.values()))
        cont = T.continuation_sibling_rows(self.eps[:6], stride=3)
        self.assertEqual(len(cont), len(sib))
        for r in cont:
            self.assertEqual(sum(k in r for k in ('target', 'outcome', 'continuation')), 1)
            if 'continuation' in r:
                self.assertTrue(1 <= len(r['continuation']['successors']) <= 256)

    def test_validation_split(self):
        v1 = T.validation_episodes(self.eps, 0)
        self.assertEqual(v1, T.validation_episodes(self.eps, 0))
        self.assertNotEqual(v1, T.validation_episodes(self.eps, 1))
        self.assertEqual(len(v1), round(0.2 * 96))

    def test_episode_json_roundtrip(self):
        e = self.eps[3]
        e2 = T.Episode.from_json(e.to_json())
        self.assertEqual((e2.chains, e2.U, e2.T), (e.chains, e.U, e.T))


if __name__ == '__main__':
    unittest.main()
