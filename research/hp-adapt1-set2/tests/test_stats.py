import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hpx import stats  # noqa: E402


class TestStats(unittest.TestCase):
    def test_wilcoxon_exact_small(self):
        # all 8 positive, distinct: p = 2/2^8
        r = stats.wilcoxon_signed_rank([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertAlmostEqual(r['p'], 2 / 256)
        self.assertEqual(stats.wilcoxon_signed_rank([0, 0])['p'], 1.0)

    def test_wilcoxon_normal_symmetric(self):
        rng = random.Random(1)
        d = [rng.choice([-2, -1, 1, 2]) for _ in range(60)]
        r = stats.wilcoxon_signed_rank(d)
        self.assertTrue(0 <= r['p'] <= 1)
        r2 = stats.wilcoxon_signed_rank([x + 3 for x in d])
        self.assertLess(r2['p'], 1e-6)

    def test_holm(self):
        self.assertEqual(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_bootstrap_and_rule(self):
        m, lo, hi = stats.bootstrap_mean_ci([1.0] * 10)
        self.assertEqual((m, lo, hi), (1.0, 1.0, 1.0))
        rng = random.Random(2)
        lin = [[1 + rng.gauss(0, 1) for _ in range(40)] for _ in range(3)]
        res = stats.decision_rule(lin)
        self.assertTrue(res['holds'])
        res0 = stats.decision_rule([[rng.gauss(0, 1) for _ in range(40)] for _ in range(3)])
        self.assertIn(res0['holds'], (True, False))

    def test_paired_rate(self):
        a = {0: [1, 1, 0, 1], 1: [0.5, 1, 1]}
        b = {0: [0, 1, 0, 1], 1: [0.5, 0, 1]}
        point, lo, hi = stats.bootstrap_paired_rate(a, b, n_boot=2000)
        self.assertAlmostEqual(point, 2 / 7)
        self.assertLessEqual(lo, point)
        self.assertGreaterEqual(hi, point)


if __name__ == '__main__':
    unittest.main()
