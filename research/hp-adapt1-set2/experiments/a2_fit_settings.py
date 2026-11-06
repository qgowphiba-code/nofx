"""A2: fit-settings probe (plan section 5). Why did set-1 recipe fits on own-policy data keep epoch 0?

Data: T0.3's Stage A rows (816 terminal + s3 continuation siblings) plus 96 own-policy folds (8 per set-1
training chain) collected by the robust model at depth 1, eps 0.2 (fold rngs Random(80800*1000 + i): the
plan names no seed). Own-policy rows use the same set-1 recipe (terminal rows + s3 continuation siblings).

Arrangement (as T0.3): A0 is installed and stays the continuation reference; the robust model ROB is the
candidate (fits that select continuation rows must be install=false: af6670b answers 422 "selected
observation continuation requires install=false or application-outcome promotion"). Warm arms start from
the candidate ROB (warm_start_source candidate, bound by sha); ROB is refit into the candidate slot before
each later warm arm (refits are functionally identical, T0.3). Five candidate arms on all rows:
  warm_pred     warm from ROB, selection prediction                (set 1's setting)
  cold          cold init (installed A0 kept as the continuation reference)
  warm_adapt    warm from ROB, selection_objective adaptation_prediction
  warm_retain   warm from ROB, function_retention_weight 1
  cold_240      cold, 240 epochs
Measures: Eval-S best-move rate (depths 1-3), selected_epoch, optimization_applied. Decision: f* = best;
if the spread is under 0.03, prefer the simplest (cold).
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hpx import settings as S, suites as SU, targets as T  # noqa: E402
from hpx.client import ApiError  # noqa: E402
from hpx.evaluate import grade_short, model_evaluator  # noqa: E402
from hpx.planner import run_folds  # noqa: E402
from hpx.runlog import Run, connect, sha256_json  # noqa: E402

ARMS = [
    ('warm_pred', dict(warm_start=True)),
    ('warm_adapt', dict(warm_start=True, selection_objective='adaptation_prediction')),
    ('warm_retain', dict(warm_start=True, function_retention_weight=1)),
    ('cold', dict(warm_start=False)),
    ('cold_240', dict(warm_start=False, epochs=240)),
]


def main():
    run = Run('A2')
    c = connect(run)
    v = c.check_build('start')
    suite = SU.load('eval_s')
    train, _ = SU.set1_chains()
    eps = T.random_episodes(train, 8, random.Random(61107), 'a3d1')
    a0_rows = T.terminal_rows(eps)
    rob_rows = T.continuation_sibling_rows(eps, stride=3)
    stage_a_ids = [r['record_id'] for r in a0_rows + rob_rows]
    val0 = T.validation_episodes(eps, fit_index=0)
    rob_opts = S.fit_options(warm_start=True, validation_episode_ids=val0, **S.COMPARISON)
    results = {'arms': {}, 'rob_refits': []}
    d = c.create_domain('a2', 'all', 0)
    run.note(host=S.HOST, build=v, domain=d, seeds={'stage_a': 61107, 'own_folds': '80800*1000+i', 'validation': [80700, 80701]})
    try:
        c.configure(d, S.value_config('object_graph'))
        c.observe(d, a0_rows + rob_rows)
        a0 = c.train(d, 'a2-a0', S.fit_options(validation_episode_ids=val0), install=True,
                     observation_ids=[r['record_id'] for r in a0_rows], name='a0')

        def fit_rob(tag):
            f = c.train(d, f'a2-rob-{tag}', rob_opts, install=False, warm_start_source='installed',
                        expected_initial_model_sha256=a0['sha256'], observation_ids=stage_a_ids, name=f'rob-{tag}')
            results['rob_refits'].append({'tag': tag, 'sha256': f['sha256']})
            return f['sha256']
        rob_sha = fit_rob('0')
        ev_rob = model_evaluator(c, d, 'candidate', rob_sha, name='ROB')
        g = grade_short(ev_rob, suite, depths=(1,), greedy=False)
        results['ROB'] = {'sha256': rob_sha, 'rate_depth1': g['depth1']['all']['best_move_rate']}
        run.log(f'A0 {a0["sha256"][:12]} installed; ROB candidate {rob_sha[:12]}: Eval-S d1 '
                f'{g["depth1"]["all"]["best_move_rate"]:.3f} (T0.3: 0.818)')

        seqs = [train[i // 8] for i in range(96)]
        folds = run_folds(seqs, 1, ev_rob, rngs=[random.Random(80800 * 1000 + i) for i in range(96)], eps=0.2, record=True)
        own = []
        for i, f in enumerate(folds):
            chs = [tuple(map(tuple, s['chain'])) for s in f['trajectory']] + [tuple(map(tuple, f['chain']))]
            own.append(T.Episode(f'a2own-c{i // 8}-f{i % 8}', seqs[i], chs, generation=1, collector=rob_sha))
        results['own_collection'] = {'mean_contacts': sum(e.contacts for e in own) / len(own),
                                     'complete': sum(f['complete'] for f in folds),
                                     'episodes_sha256': sha256_json([e.to_json() for e in own])}
        run.log('own-policy folds: mean contacts', round(results['own_collection']['mean_contacts'], 2))
        own_rows = T.terminal_rows(own) + T.continuation_sibling_rows(own, stride=3)
        c.observe(d, own_rows)
        all_ids = stage_a_ids + [r['record_id'] for r in own_rows]
        val = T.validation_episodes(eps + own, fit_index=1)
        results['n_rows'] = len(all_ids)
        results['validation_episode_ids'] = val
        first_warm = True
        for name, over in ARMS:
            opts = S.fit_options(validation_episode_ids=val, **{**S.COMPARISON, **over})
            extra = {}
            if over.get('warm_start'):
                if not first_warm:
                    rob_sha = fit_rob(name)
                first_warm = False
                extra = {'warm_start_source': 'candidate', 'expected_initial_model_sha256': rob_sha}
            try:
                fit = c.train(d, f'a2-{name}', opts, install=False, observation_ids=all_ids, name=name, **extra)
            except ApiError as e:
                if e.status != 422:
                    raise
                results['arms'][name] = {'refused': True, 'status': e.status, 'detail': e.body, 'options': opts}
                run.log(f'{name:12s} REFUSED (422): {str(e.body)[:200]}')
                run.write_results(results)
                continue
            rep = fit['state']['training']['report']
            ev = model_evaluator(c, d, 'candidate', fit['sha256'], name=name)
            g = grade_short(ev, suite, depths=(1, 2, 3), greedy_depths=(1,))
            cont = rep.get('continuation') or {}
            results['arms'][name] = {'sha256': fit['sha256'], 'fit_seconds': fit['elapsed'], 'options': opts, 'report': rep,
                                     'selected_epoch': rep.get('selected_epoch'), 'optimization_applied': cont.get('optimization_applied'),
                                     'initial_model_sha256': cont.get('initial_model_sha256'),
                                     'regularization_target': rep.get('regularization_target'), 'grade': g}
            run.log(f'{name:12s} d1 {g["depth1"]["all"]["best_move_rate"]:.3f} d2 {g["depth2"]["all"]["best_move_rate"]:.3f} '
                    f'd3 {g["depth3"]["all"]["best_move_rate"]:.3f} corr V* {g["depth1"]["all"]["corr_vstar"]:.3f} | selected_epoch '
                    f'{rep.get("selected_epoch")} optimization_applied {cont.get("optimization_applied")} reg {rep.get("regularization_target")} | {fit["elapsed"]}s')
            run.write_results(results)
        rates = {n: results['arms'][n]['grade']['depth1']['all']['best_move_rate'] for n, _ in ARMS
                 if not results['arms'][n].get('refused')}
        best = max(rates, key=rates.get)
        spread = max(rates.values()) - min(rates.values())
        results['decision'] = {'rates_depth1': rates, 'best': best, 'spread': spread,
                               'f_star': 'cold' if spread < 0.03 else best,
                               'rule': 'best by Eval-S depth-1 rate; if the spread is under 0.03 prefer the simplest (cold)'}
        run.log('decision', results['decision'])
    finally:
        c.delete_domain(d)
        results['version_end'] = c.check_build('end')
        results['requests'] = c.counts
        run.write_results(results)
        run.finish(requests=c.counts['requests'])


if __name__ == '__main__':
    main()
