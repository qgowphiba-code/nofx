import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hpx.hpenv import Fold, FEATURES, BASIC, upper_bound, from_state  # noqa: E402
from hpx.vendor import lattice_env, hp, liu_panel  # noqa: E402


def random_walk(seq, dim, rng):
    s = lattice_env.State.initial(seq, dim)
    f = Fold(seq, dim)
    pairs = [(s, f.copy())]
    while not s.terminal:
        a = rng.choice(s.legal)
        s = s.step(a)
        f.push(a)
        pairs.append((s, f.copy()))
    return pairs


class TestFoldMatchesVendor(unittest.TestCase):
    def test_3d_observation_features_values(self):
        rng = random.Random(5)
        seqs = lattice_env.sequences(9001, [10, 11, 14, 20, 24, 36, 48, 60]) + [s for _, s in liu_panel()]
        n_states = 0
        for seq in seqs:
            for _ in range(3):
                for s, f in random_walk(seq, 3, rng):
                    n_states += 1
                    self.assertEqual(f.contacts, s.contacts)
                    self.assertEqual(tuple(f.legal()), tuple(s.legal))
                    self.assertEqual(f.terminal, s.terminal)
                    self.assertEqual(f.partial_score(), s.partial_score())
                    if s.terminal:
                        self.assertEqual(f.utility(), s.utility())
                    self.assertEqual(f.features(), s.features())
                    self.assertEqual(json.dumps(f.observation(), separators=(',', ':')),
                                     json.dumps(hp.observation(s), separators=(',', ':')))
        self.assertGreater(n_states, 1000)

    def test_2d_matches_state(self):
        rng = random.Random(6)
        for seq in lattice_env.sequences(9002, [12, 16, 25, 40]):
            for _ in range(4):
                for s, f in random_walk(seq, 2, rng):
                    self.assertEqual(f.contacts, s.contacts)
                    self.assertEqual(tuple(f.legal()), tuple(s.legal))
                    self.assertEqual(f.features(), s.features())
                    self.assertEqual(f.partial_score(), s.partial_score())
                    obs = f.observation(FEATURES)
                    self.assertEqual(len(obs['objects']), len(seq))
                    self.assertTrue(all(len(o['position']) == 3 and o['position'][2] == 0 for o in obs['objects']))
                    self.assertEqual(sum(r['type'] == 'hh_contact' for r in obs['relations']), s.contacts)

    def test_pop_restores(self):
        rng = random.Random(7)
        seq = lattice_env.sequences(9003, [40])[0]
        f = Fold(seq)
        for _ in range(20):
            f.push(rng.choice(f.legal()))
        before = (list(f.chain), dict(f.occ), f.contacts, list(f.hh), f.observation())
        for d in f.legal():
            f.push(d)
            for e in f.legal()[:2]:
                f.push(e)
                f.pop()
            f.pop()
        self.assertEqual(before, (list(f.chain), dict(f.occ), f.contacts, list(f.hh), f.observation()))

    def test_from_state_and_ub(self):
        s = lattice_env.State.initial('HPHPPHHPHPPHPHHPPHPH')
        for a in ((0, 1, 0), (-1, 0, 0), (0, 0, 1)):
            s = s.step(a)
        f = from_state(s)
        self.assertEqual(tuple(f.chain), s.chain)
        self.assertEqual(f.ub, s.upper_bound)
        self.assertEqual(upper_bound('HPHPPHHPHPPHPHHPPHPH', 2), lattice_env.State.initial('HPHPPHHPHPPHPHHPPHPH', 2).upper_bound)
        self.assertEqual(set(BASIC) <= set(FEATURES), True)


if __name__ == '__main__':
    unittest.main()
