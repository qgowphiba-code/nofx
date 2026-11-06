"""A3b: blended targets on own-policy data, a mini Track B with exact grading (plan section 5).

On A3's draw-1 data (96 random folds, Random(61107)): A0 (cold, install: it stays the A0 collector) and
BL(lam*, d*) (candidate: the BL collector; u from A0). One own-policy generation per arm: 96 folds (8 per
set-1 training chain) at depth 2, eps 0.15, collected by the arm's own model (fold rngs
Random('a3b-<arm>-<i>'): the plan names no seed); BL's new rows use u from its own model (the depth-2
search values recorded during collection when d* = 2). Both arms are then refit cold on all their rows
and graded on Eval-S. Prediction: BL keeps a lead of at least +0.03 after the own-policy generation.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hpx import settings as S, suites as SU, targets as T  # noqa: E402
from hpx.evaluate import grade_short, model_evaluator  # noqa: E402
from hpx.planner import run_folds  # noqa: E402
from hpx.runlog import ROOT, Run, connect, sha256_json  # noqa: E402


def collect(arm, ev, chains, sha):
    seqs = [chains[i // 8] for i in range(96)]
    res = run_folds(seqs, 2, ev, rngs=[random.Random(f'a3b-{arm}-{i}') for i in range(96)], eps=0.15, record=True)
    eps, u2 = [], {}
    for i, r in enumerate(res):
        chs = [tuple(map(tuple, s['chain'])) for s in r['trajectory']] + [tuple(map(tuple, r['chain']))]
        e = T.Episode(f'a3b{arm}-c{i // 8}-f{i % 8}', seqs[i], chs, generation=1, collector=sha)
        eps.append(e)
        for k, s in enumerate(r['trajectory']):
            u2[(e.id, k)] = s['u']
    return eps, u2


def main():
    cfg = json.loads(Path(__import__('os').environ.get('HPX2_TRACK_B', ROOT / 'manifests' / 'track_b.json')).read_text())
    lam, dstar = cfg['lam_star'], cfg['d_star']
    run = Run('A3b')
    c = connect(run)
    v = c.check_build('start')
    suite = SU.load('eval_s')
    train, _ = SU.set1_chains()
    eps0 = T.random_episodes(train, 8, random.Random(61107), 'a3d1')
    val0 = T.validation_episodes(eps0, fit_index=0)
    results = {'lam_star': lam, 'd_star': dstar, 'arms': {}}
    d = c.create_domain('a3b', 'all', 0)
    run.note(host=S.HOST, build=v, domain=d, track_b=cfg)

    def grade(name, src, sha, extra=None):
        ev = model_evaluator(c, d, src, sha, name=name)
        g = grade_short(ev, suite, depths=(1, 2, 3), greedy_depths=(1,))
        results['arms'][name] = {'sha256': sha, 'grade': g, **(extra or {})}
        run.log(f'{name:4s} d1 {g["depth1"]["all"]["best_move_rate"]:.3f} d2 {g["depth2"]["all"]["best_move_rate"]:.3f} '
                f'd3 {g["depth3"]["all"]["best_move_rate"]:.3f} corr V* {g["depth1"]["all"]["corr_vstar"]:.3f}')
        run.write_results(results)
        return ev
    try:
        c.configure(d, S.value_config('object_graph'))
        a0_rows = T.terminal_rows(eps0)
        c.observe(d, a0_rows)
        a0 = c.train(d, 'a3b-a0', S.fit_options(validation_episode_ids=val0), install=True,
                     observation_ids=[r['record_id'] for r in a0_rows], name='a0')
        ev_a0 = grade('A0', 'installed', a0['sha256'])
        u0 = T.assess(eps0, dstar, ev_a0)
        bl_rows = T.blended_rows(eps0, lam, dstar, u0, a0['sha256'])
        c.observe(d, bl_rows)
        bl = c.train(d, 'a3b-bl', S.fit_options(validation_episode_ids=val0), install=False,
                     observation_ids=[r['record_id'] for r in bl_rows], name='bl')
        ev_bl = grade('BL', 'candidate', bl['sha256'])

        bl_eps, bl_u2 = collect('BL', ev_bl, train, bl['sha256'])
        a0_eps, _ = collect('A0', ev_a0, train, a0['sha256'])
        results['collection'] = {'A0_mean_contacts': sum(e.contacts for e in a0_eps) / 96,
                                 'BL_mean_contacts': sum(e.contacts for e in bl_eps) / 96,
                                 'episodes_sha256': sha256_json([e.to_json() for e in a0_eps + bl_eps])}
        run.log('collection', results['collection'])
        bl_u = bl_u2 if dstar == 2 else T.assess(bl_eps, dstar, ev_bl)
        a0_new = T.terminal_rows(a0_eps)
        bl_new = T.blended_rows(bl_eps, lam, dstar, bl_u, bl['sha256'])
        c.observe(d, a0_new + bl_new)
        val_a0 = T.validation_episodes(eps0 + a0_eps, fit_index=600)
        val_bl = T.validation_episodes(eps0 + bl_eps, fit_index=601)
        a0b = c.train(d, 'a3b-a0-gen1', S.fit_options(validation_episode_ids=val_a0), install=False,
                      observation_ids=[r['record_id'] for r in a0_rows + a0_new], name='a0-gen1')
        grade('A0g1', 'candidate', a0b['sha256'], {'n_rows': len(a0_rows) + len(a0_new)})
        blb = c.train(d, 'a3b-bl-gen1', S.fit_options(validation_episode_ids=val_bl), install=False,
                      observation_ids=[r['record_id'] for r in bl_rows + bl_new], name='bl-gen1')
        grade('BLg1', 'candidate', blb['sha256'], {'n_rows': len(bl_rows) + len(bl_new)})
        r = {k: results['arms'][k]['grade']['depth1']['all']['best_move_rate'] for k in ('A0', 'BL', 'A0g1', 'BLg1')}
        results['decision'] = {'rates_depth1': r, 'lead_before': r['BL'] - r['A0'], 'lead_after': r['BLg1'] - r['A0g1'],
                               'prediction_holds': r['BLg1'] - r['A0g1'] >= 0.03}
        run.log('A3b', results['decision'])
    finally:
        c.delete_domain(d)
        results['version_end'] = c.check_build('end')
        results['requests'] = c.counts
        run.write_results(results)
        run.finish(requests=c.counts['requests'])


if __name__ == '__main__':
    main()
