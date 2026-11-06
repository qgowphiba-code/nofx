import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hpx import settings as S  # noqa: E402
from hpx.client import ApiError, BuildChanged, Client, GuardError, Stopped, compact, MAX_BODY  # noqa: E402
from hpx.hpenv import Fold  # noqa: E402
from hpx.mock import mock_client  # noqa: E402
from hpx.vendor import lattice_env  # noqa: E402


def rows_for(seqs, n_folds, rng, prefix='r'):
    rows = []
    for e in range(n_folds):
        f = Fold(seqs[e % len(seqs)])
        states = [f.copy()]
        while not f.terminal:
            f.push(rng.choice(f.legal()))
            states.append(f.copy())
        u = states[-1].utility()
        for k, s in enumerate(states[:-1]):
            rows.append({'record_id': f'{prefix}-e{e}-s{k}', 'episode_id': f'{prefix}-e{e}', 'step': k,
                         **s.observation(), 'target': u})
    return rows


class TestClientWithMock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name) / 'run'
        self.c, self.t = mock_client(self.run, self.tmp.name)
        self.rows = rows_for(lattice_env.sequences(1, [10, 11]), 8, random.Random(2))

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_cycle(self):
        c = self.c
        c.check_build('start')
        d = c.create_domain('test', 'arm', 1)
        self.assertTrue(d.startswith('hpx2-test-arm-1-'))
        c.configure(d, S.value_config())
        res = c.observe(d, self.rows)
        self.assertEqual(res['accepted'], len(self.rows))
        self.assertEqual(c.observe(d, self.rows[:7])['duplicates'], 7)
        with self.assertRaises(ApiError) as e:
            c.observe(d, [{**self.rows[0], 'target': 0.01}])
        self.assertEqual(e.exception.status, 409)
        fit = c.train(d, 'fit-1', S.fit_options(validation_episode_ids=['r-e0']), install=True)
        sha = fit['sha256']
        vals = c.predict(d, [{k: r[k] for k in ('objects', 'relations', 'conditions')} for r in self.rows[:5]], 'installed', sha)
        self.assertEqual(len(vals), 5)
        cand = c.train(d, 'fit-2', S.fit_options(warm_start=True), install=False, warm_start_source='installed',
                       expected_initial_model_sha256=sha)
        self.assertNotEqual(cand['sha256'], sha)
        self.assertEqual(Client.model_sha(cand['state'], 'installed'), sha)
        with self.assertRaises(ApiError):
            c.predict(d, [{'objects': []}], 'candidate', sha)    # wrong expected sha -> 409
        c.predict(d, [{'objects': []}], 'candidate', cand['sha256'])
        c.check_build('end')
        c.delete_domain(d)
        self.assertEqual(c._registry(), [])
        self.assertTrue((self.run / 'journal.jsonl').exists())

    def test_prefix_guard_and_live_limit(self):
        with self.assertRaises(GuardError):
            self.c.state('alchemy-authored-20261106T191111Z-c9c70d')
        ds = [self.c.create_domain('t', 'a', i) for i in range(4)]
        with self.assertRaises(GuardError):
            self.c.create_domain('t', 'a', 5)
        for d in ds:
            self.c.delete_domain(d)

    def test_stop_file(self):
        (self.run / 'STOP').write_text('stop')
        with self.assertRaises(Stopped):
            self.c.version()

    def test_build_change_aborts(self):
        self.c.check_build('start')
        self.t.commit = '1' * 40
        with self.assertRaises(BuildChanged):
            self.c.check_build('end')

    def test_ambiguous_writes_reconcile(self):
        c, t = self.c, self.t
        t.faults['POST api/v1/domains'] = [(504, 'after')]
        d = c.create_domain('t', 'a', 0)                  # applied, then 504 -> reconcile via GET
        self.assertIn(d, t.domains)
        c.configure(d, S.value_config())
        t.faults['POST domains/{id}/value/observe'] = [(502, 'after')]
        res = c.observe(d, self.rows[:30])                # resend identical -> duplicates
        self.assertEqual(res['accepted'] + res['duplicates'], 30)
        t.faults['POST domains/{id}/value/train'] = [(502, 'after')]
        c.train(d, 'fit-amb', S.fit_options(), install=True)
        self.assertEqual(t.calls.count('POST domains/{id}/value/train'), 1)   # reconciled, not resubmitted
        t.faults['POST domains/{id}/value/predict'] = [(503, 'before'), (429, 'before')]
        sha = Client.model_sha(c.state(d), 'installed')
        self.assertEqual(len(c.predict(d, [{'objects': []}], 'installed', sha)), 1)
        c.delete_domain(d)

    def test_unchanged_sha_is_refused(self):
        c = self.c
        d = c.create_domain('t', 'a', 0)
        c.configure(d, S.value_config())
        c.observe(d, self.rows)
        c.train(d, 'same', S.fit_options(), install=False)
        with self.assertRaises(GuardError):
            c.train(d, 'same', S.fit_options(), install=False)     # identical fit -> identical candidate sha
        c.delete_domain(d)

    def test_predict_chunks_respect_limits(self):
        c, t = self.c, self.t
        d = c.create_domain('t', 'a', 0)
        c.configure(d, S.value_config())
        c.observe(d, self.rows)
        sha = c.train(d, 'f', S.fit_options(), install=True)['sha256']
        seq = lattice_env.sequences(3, [60])[0]
        f = Fold(seq)
        rng = random.Random(4)
        big = []
        while len(big) < 600:
            if f.terminal:
                f = Fold(seq)
            f.push(rng.choice(f.legal()))
            big.append(f.observation())
        before = t.calls.count('POST domains/{id}/value/predict')
        vals = c.predict(d, big, 'installed', sha)
        n_calls = t.calls.count('POST domains/{id}/value/predict') - before
        self.assertEqual(len(vals), 600)
        sizes = sum(len(compact(x)) for x in big)
        self.assertGreaterEqual(n_calls, max(3, sizes // MAX_BODY))
        with self.assertRaises(GuardError):
            c.predict(d, big[:2], 'installed', None)
        c.delete_domain(d)


if __name__ == '__main__':
    unittest.main()
