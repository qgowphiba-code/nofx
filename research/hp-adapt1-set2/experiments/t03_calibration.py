"""T0.3: calibration replicate of set-1 Stage A (plan section 4).

Rebuilds Stage A from seeds: 96 random folds (Random(61107), chain-major, 8 per training chain) on set 1's
12 training chains -> 816 terminal rows (target = U); plus set 1's robust recipe: sibling rows with
server-side `continuation` labels on every 3rd state (the "s3 sibling subset"), grouped by parent.
Fits A0 (cold, install true), a replicate of A0 as a candidate (determinism probe), then ROB (warm from A0,
squared comparison, prediction selection, install false), and grades them on Eval-S.

Not bit-identical to set 1: stage_a.py is not in the package, so record ids, row order, validation
episodes and the exact s3 rule are reconstructions. Compare with 0.662 / 0.821 directionally.
The domain is kept live for T0.4 (it needs this A0 model); t04_parity.py deletes it.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hpx import settings as S, suites as SU, targets as T  # noqa: E402
from hpx.client import Client  # noqa: E402
from hpx.evaluate import grade_short, model_evaluator  # noqa: E402
from hpx.planner import Heuristic  # noqa: E402
from hpx.runlog import Run, connect, sha256_json  # noqa: E402


def saved_fit(run, name, c, domain, slot):
    """On resume: a finished fit's sha from its saved response, checked against live state."""
    p = run.dir / 'responses' / f'train-{name}.json'
    if not p.exists():
        return None
    saved = json.loads(p.read_text())
    sha = Client.model_sha(saved['final_state'], slot)
    live = Client.model_sha(c.state(domain), slot)
    if sha != live:
        run.log(f'resume: saved {name} sha {sha[:12]} is no longer the live {slot} ({str(live)[:12]}); refitting')
        return None
    return {'sha256': sha, 'elapsed': saved['elapsed'], 'state': saved['final_state'], 'resumed': True}


def max_prediction_gap(c, domain, suite, a0_sha, rep_sha, n=512):
    """Functional equality of two fits: max |V_A0 - V_rep| over n Eval-S states and their children."""
    from hpx.hpenv import Fold
    seqs = {ch['ci']: ch['sequence'] for ch in suite['chains']}
    inputs = []
    for d in suite['decisions']:
        f = Fold(seqs[d['ci']], 3, d['chain'])
        inputs.append(f.observation())
        if len(inputs) >= n:
            break
    a = c.predict(domain, inputs, 'installed', a0_sha)
    b = c.predict(domain, inputs, 'candidate', rep_sha)
    gaps = [abs(x - y) for x, y in zip(a, b)]
    return {'n': len(gaps), 'max_abs_diff': max(gaps), 'mean_abs_diff': sum(gaps) / len(gaps)}


def main():
    resume = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run = Run('T0.3', name=resume.name if resume else None)
    c = connect(run)
    v = c.check_build('start')
    suite = SU.load('eval_s')
    train, _ = SU.set1_chains()
    eps = T.random_episodes(train, 8, random.Random(61107), 'a3d1')
    a0_rows = T.terminal_rows(eps)
    rob_rows = T.continuation_sibling_rows(eps, stride=3)
    val = T.validation_episodes(eps, fit_index=0)
    (run.dir / 'episodes.json').write_text(json.dumps([e.to_json() for e in eps]))
    run.note(host=S.HOST, build=v, seeds={'folds': 61107, 'validation': 80700}, n_a0_rows=len(a0_rows),
             n_rob_rows=len(rob_rows), validation_episode_ids=val, rows_sha256={'a0': sha256_json(a0_rows),
             'rob': sha256_json(rob_rows)}, eval_s_sha256=json.loads((SU.SUITES / 'MANIFEST.json').read_text())['eval_s'])
    run.log(f'rows: A0 {len(a0_rows)}, ROB siblings {len(rob_rows)} '
            f'({sum("continuation" in r for r in rob_rows)} continuation, {sum("target" in r for r in rob_rows)} exact)')
    results = {'heuristic': grade_short(Heuristic(), suite, depths=(1, 2, 3))}
    run.log('heuristic depth1', results['heuristic']['depth1']['set1']['best_move_rate'], results['heuristic']['depth1']['all']['best_move_rate'])
    if resume:
        created = [json.loads(line) for line in (run.dir / 'journal.jsonl').read_text().splitlines()]
        domain = [r['domain_id'] for r in created if r.get('kind') == 'domain_created'][-1]
        run.log('resuming on live domain', domain)
    else:
        domain = c.create_domain('t03', 'cal', 0)
        c.configure(domain, S.value_config('object_graph'))
    run.note(domain=domain)
    obs = c.observe(domain, a0_rows)
    run.log('observed A0 rows', {k: v for k, v in obs.items() if k != 'last'})
    a0_opts = S.fit_options(warm_start=False, validation_episode_ids=val)
    a0 = saved_fit(run, 'a0', c, domain, 'installed') or c.train(
        domain, 't03-a0', a0_opts, install=True, observation_ids=[r['record_id'] for r in a0_rows], name='a0')
    run.log('A0 fitted', a0['sha256'], f'{a0["elapsed"]}s', 'selected_epoch', a0['state']['training']['report'].get('selected_epoch'),
            '(resumed)' if a0.get('resumed') else '')
    ev = model_evaluator(c, domain, 'installed', a0['sha256'])
    results['A0'] = {'sha256': a0['sha256'], 'fit_seconds': a0['elapsed'], 'report': a0['state']['training']['report'],
                     'grade': grade_short(ev, suite, depths=(1, 2, 3))}
    g = results['A0']['grade']['depth1']
    run.log(f"A0 depth1 best-move: set1 {g['set1']['best_move_rate']:.3f} (pred 0.662) all {g['all']['best_move_rate']:.3f} "
            f"corr V* {g['all']['corr_vstar']:.3f}; predict stats {ev.stats}")

    rep = saved_fit(run, 'a0-replicate', c, domain, 'candidate') or c.train(
        domain, 't03-a0-replicate', a0_opts, install=False, observation_ids=[r['record_id'] for r in a0_rows],
        name='a0-replicate')
    results['A0_replicate'] = {'sha256': rep['sha256'], 'identical_sha_to_A0': rep['sha256'] == a0['sha256'],
                               'fit_seconds': rep['elapsed'],
                               'selected_epoch': rep['state']['training']['report'].get('selected_epoch')}
    if rep['sha256'] != a0['sha256']:
        results['A0_replicate']['prediction_gap'] = max_prediction_gap(c, domain, suite, a0['sha256'], rep['sha256'])
        evr = model_evaluator(c, domain, 'candidate', rep['sha256'])
        results['A0_replicate']['grade'] = grade_short(evr, suite, depths=(1,), greedy=False)
        run.log('A0 replicate depth1 rate', round(results['A0_replicate']['grade']['depth1']['all']['best_move_rate'], 3))
    run.log('A0 replicate: identical sha', results['A0_replicate']['identical_sha_to_A0'],
            '| prediction gap', results['A0_replicate'].get('prediction_gap'))

    obs = c.observe(domain, rob_rows)
    run.log('observed ROB rows', {k: v for k, v in obs.items() if k != 'last'})
    rob_opts = S.fit_options(warm_start=True, validation_episode_ids=val, **S.COMPARISON)
    rob = c.train(domain, 't03-rob', rob_opts, install=False, warm_start_source='installed',
                  expected_initial_model_sha256=a0['sha256'],
                  observation_ids=[r['record_id'] for r in a0_rows + rob_rows], name='rob')
    rr = rob['state']['training']['report']
    run.log('ROB fitted', rob['sha256'], f'{rob["elapsed"]}s', 'selected_epoch', rr.get('selected_epoch'),
            'target_refresh', json.dumps(rr.get('target_refresh'))[:300])
    evb = model_evaluator(c, domain, 'candidate', rob['sha256'])
    results['ROB'] = {'sha256': rob['sha256'], 'fit_seconds': rob['elapsed'], 'report': rr,
                      'grade': grade_short(evb, suite, depths=(1, 2, 3))}
    g = results['ROB']['grade']['depth1']
    run.log(f"ROB depth1 best-move: set1 {g['set1']['best_move_rate']:.3f} (pred 0.821) all {g['all']['best_move_rate']:.3f} "
            f"corr V* {g['all']['corr_vstar']:.3f}")
    results['domain_kept_for_T0.4'] = domain
    results['version_end'] = c.check_build('end')
    results['requests'] = c.counts
    run.write_results(results)
    run.finish(requests=c.counts['requests'], a0_sha=a0['sha256'], rob_sha=rob['sha256'])
    run.log('done; requests', c.counts['requests'], 'domain kept for T0.4:', domain)


if __name__ == '__main__':
    main()
