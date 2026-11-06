import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hpx.oracle import Oracle  # noqa: E402
from hpx.planner import Heuristic  # noqa: E402
from hpx import suites  # noqa: E402
from hpx.vendor import hp, lattice_env  # noqa: E402


class TestOracle(unittest.TestCase):
    def test_matches_vendor_oracle_3d(self):
        seq = lattice_env.sequences(61107, [10, 11] * 9)[12]
        ref = hp.Oracle(seq)
        mine = Oracle(seq)
        rng = random.Random(3)
        for _ in range(3):
            for s in hp.random_fold(seq, rng):
                self.assertEqual(ref.values(s), mine.values(s.chain))

    def test_2d_small_bruteforce(self):
        seq = 'HPHPPHHPHH'
        o = Oracle(seq, 2)
        s0 = lattice_env.State.initial(seq, 2)
        best = -1

        def walk(s):
            nonlocal best
            if s.terminal:
                best = max(best, s.contacts if s.complete else -1)
                return
            for a in s.legal:
                walk(s.step(a))
        walk(s0)
        self.assertEqual(o.optimum_contacts(), best)


class TestEvalS(unittest.TestCase):
    """The set-1 group of Eval-S must be exactly set 1's grading set; heuristic rates match check_short.out."""

    @classmethod
    def setUpClass(cls):
        eval1, _ = suites.eval_s_chains()
        cls.suite = suites.build_short_suite([('set1', 61108, eval1)], dim=3)

    def test_set1_group_matches_grading(self):
        g = __import__('hpx.vendor', fromlist=['grading']).grading()
        ref = [(ci, st.chain, x['vstar'], x['vrand'], sorted(x['opt'])) for x in g.grade for ci, st in [(x['ci'], x['state'])]]
        mine = [(d['ci'], tuple(map(tuple, d['chain'])), d['vstar'], d['vrand'], sorted(map(tuple, d['opt'])))
                for d in self.suite['decisions']]
        self.assertEqual(ref, mine)
        self.assertEqual(len(mine), 216)

    def test_heuristic_fair_rates(self):
        got = [round(suites.score_short(self.suite, Heuristic(), d, state_values=False)['summary']['all']['best_move_rate'], 3)
               for d in (1, 2, 3)]
        self.assertEqual(got, [0.720, 0.862, 0.922])
        self.assertEqual(suites.score_short(self.suite, Heuristic(), 1)['summary']['all']['n_informative'], 151)


if __name__ == '__main__':
    unittest.main()
