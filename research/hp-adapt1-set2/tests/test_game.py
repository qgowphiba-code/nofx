import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hpx import game as G  # noqa: E402


class TestGame(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = G.Solver()
        cls.solver.value(G.Pos())

    def test_lines_and_winner(self):
        self.assertEqual(len(G.LINES), 17)   # 5x4: 8 horizontal, 5 vertical, 4 diagonal
        p = G.Pos()
        for c in (0, 0, 1, 1, 2, 2, 3):      # X plays 0,1,2,3 on the bottom row
            p = p.play(c)
        self.assertEqual(p.winner(), 'X')
        self.assertTrue(p.terminal)
        self.assertEqual(p.value(), 1.0)

    def test_winner_matches_bruteforce(self):
        rng = random.Random(3)
        for _ in range(300):
            for p in G.random_game(rng):
                brute = None
                for ln in G.LINES:
                    cells = [p.cell(x, y) for x, y in ln]
                    if cells[0] and all(x == cells[0] for x in cells):
                        brute = cells[0]
                self.assertEqual(p.winner(), brute)

    def test_full_depth_minimax_equals_solver(self):
        rng = random.Random(4)
        checked = 0
        for _ in range(200):
            game = G.random_game(rng)
            for p in game:
                if not p.terminal and p.w * p.h - p.n <= 5:
                    depth = p.w * p.h - p.n
                    self.assertEqual(G.search_value([p], depth, G.GameNeutral())[0], self.solver.value(p))
                    checked += 1
        self.assertGreater(checked, 50)

    def test_suite_and_scores(self):
        suite = G.build_suite(self.solver, n_random=30, n_optimal=5, size=100)
        self.assertTrue(all(len(d['best']) < len(d['child_values']) for d in suite['positions']))
        s1 = G.score(suite, G.GameHeuristic(), 1)
        s2 = G.score(suite, G.GameHeuristic(), 2)
        self.assertTrue(0 <= s1['best_move_rate'] <= 1 and 0 <= s2['best_move_rate'] <= 1)
        exact = G.score(suite, G.GameModel(lambda xs: [0.5] * len(xs)), 1)
        self.assertEqual(exact['n'], len(suite['positions']))


if __name__ == '__main__':
    unittest.main()
