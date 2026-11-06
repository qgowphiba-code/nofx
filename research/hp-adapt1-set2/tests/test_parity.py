"""Dry run of the T0.4 comparison against the mock: native graph planning == client-side search."""
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from hpx import settings as S  # noqa: E402
from hpx.evaluate import model_evaluator  # noqa: E402
from hpx.hpenv import Fold  # noqa: E402
from hpx.mock import mock_client  # noqa: E402
from hpx.planner import decide  # noqa: E402
from hpx.vendor import lattice_env  # noqa: E402
from t04_parity import ACT, graph  # noqa: E402
from tests.test_client import rows_for  # noqa: E402


class TestParityDryRun(unittest.TestCase):
    def test_native_equals_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, _ = mock_client(Path(tmp) / 'run', tmp)
            d = c.create_domain('t04', 'dry', 0)
            c.configure(d, S.value_config())
            c.observe(d, rows_for(lattice_env.sequences(1, [10, 11]), 6, random.Random(1)))
            sha = c.train(d, 'a0', S.fit_options(), install=True)['sha256']
            ev = model_evaluator(c, d, 'installed', sha)
            rng = random.Random(5)
            states = []
            for seq in lattice_env.sequences(7, [11, 20, 36]):
                f = Fold(seq)
                for _ in range(rng.randrange(1, len(seq) - 3)):
                    if f.terminal:
                        break
                    f.push(rng.choice(f.legal()))
                if not f.terminal:
                    states.append(f)
            for depth in (1, 2, 3):
                for f, (legal, vals, _) in zip(states, decide(states, depth, ev)):
                    root, nodes = graph(f, depth)
                    resp = c.plan(d, {'relation': S.RELATION, 'policy': S.POLICY, 'evaluator_source': 'value_model',
                                      'expected_model_sha256': sha, 'root': root, 'nodes': nodes,
                                      'options': {'max_depth': depth}})
                    m = max(vals)
                    self.assertIn(resp['plan']['action'], {ACT(a) for a, x in zip(legal, vals) if x == m})
                    self.assertLess(abs(resp['plan']['value'] - m), 1e-12)
            c.delete_domain(d)


if __name__ == '__main__':
    unittest.main()
