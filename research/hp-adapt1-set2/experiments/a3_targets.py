"""A3: target construction, oracle-graded (plan section 5) -- the key cheap experiment.

Data: 3 draws of 96 random folds on set 1's 12 training chains (draw 1 = Random(61107), set 1's folds as
reconstructed in T0.3; draws 2-3 = Random(80601), Random(80602)). Assessor = each draw's A0 model.
Arms, draw 1: A0 (lam 0); ROB (set 1's recipe); BL(lam, d), lam in {0.5, 0.75, 1}, d in {1, 2};
BL-iter (second round, best BL as assessor); BLG (best BL + sibling rows). Draws 2-3: A0, ROB, the best two
BL arms of draw 1, BL-iter, BLG.
One domain per draw (A0 stays installed as the continuation reference / assessor; every other arm is a
candidate graded at depths 1-3 on Eval-S while it holds the candidate slot).
Pre-registered: lam*, d* = best BL arm by Eval-S depth-1 best-move rate averaged over the 3 draws;
siblings used iff BLG beats that BL arm by >= 0.03 on the same average; if no BL arm beats A0 by >= 0.03,
Track B runs A0 against ROB-style targets.
Usage: a3_targets.py [draws...]   e.g. a3_targets.py 1   then   a3_targets.py 2 3
"""
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hpx import settings as S, suites as SU, targets as T  # noqa: E402
from hpx.evaluate import grade_short, model_evaluator  # noqa: E402
from hpx.runlog import RUNS, MOCK, Run, connect, sha256_json  # noqa: E402

BL_GRID = [(0.5, 1), (0.75, 1), (1.0, 1), (0.5, 2), (0.75, 2), (1.0, 2)]
DRAW_SEEDS = {1: 61107, 2: 80601, 3: 80602}


def bl_name(lam, d):
    return f'BL({lam:g},{d})'


def rate(g):
    return g['depth1']['all']['best_move_rate']


def run_draw(run, c, draw, suite, bl_arms):
    train, _ = SU.set1_chains()
    eps = T.random_episodes(train, 8, random.Random(DRAW_SEEDS[draw]), f'a3d{draw}')
    val = T.validation_episodes(eps, fit_index=200 + draw)
    out = {'validation_episode_ids': val, 'episodes_sha256': sha256_json([e.to_json() for e in eps]), 'arms': {}}
    d = c.create_domain('a3', 'all', draw)
    out['domain'] = d
    log = lambda *m: run.log(f'[draw {draw}]', *m)  # noqa: E731

    def fit_and_grade(name, rows_ids, opts, install=False, **kw):
        fit = c.train(d, f'a3-d{draw}-{name}', opts, install=install, observation_ids=rows_ids,
                      name=f'd{draw}-{name}', **kw)
        rep = fit['state']['training']['report']
        ev = model_evaluator(c, d, 'installed' if install else 'candidate', fit['sha256'], name=name)
        g = grade_short(ev, suite, depths=(1, 2, 3), greedy_depths=(1,))
        out['arms'][name] = {'sha256': fit['sha256'], 'fit_seconds': fit['elapsed'], 'n_rows': len(rows_ids),
                             'selected_epoch': rep.get('selected_epoch'), 'validation_skill': rep.get('validation_skill'),
                             'report': rep, 'grade': g, 'predict_stats': dict(ev.stats)}
        log(f'{name:12s} rate d1 {rate(g):.3f} (set1 {g["depth1"]["set1"]["best_move_rate"]:.3f}) d2 '
            f'{g["depth2"]["all"]["best_move_rate"]:.3f} d3 {g["depth3"]["all"]["best_move_rate"]:.3f} | corr V* '
            f'{g["depth1"]["all"]["corr_vstar"]:.3f} V_rand {g["depth1"]["all"]["corr_vrand"]:.3f} | greedy '
            f'{g["greedy_depth1"]["all"]["greedy_contacts"]}/{g["greedy_depth1"]["all"]["optimal_contacts"]} | '
            f'epoch {rep.get("selected_epoch")} | {fit["elapsed"]}s')
        run.write_results(out, f'results_draw{draw}.json')
        return fit, ev

    try:
        c.configure(d, S.value_config('object_graph'))
        a0_rows = T.terminal_rows(eps)
        c.observe(d, a0_rows)
        a0_ids = [r['record_id'] for r in a0_rows]
        a0, ev_a0 = fit_and_grade('A0', a0_ids, S.fit_options(validation_episode_ids=val), install=True)
        a0_sha = a0['sha256']

        rob_rows = T.continuation_sibling_rows(eps, stride=3)
        c.observe(d, rob_rows)
        fit_and_grade('ROB', a0_ids + [r['record_id'] for r in rob_rows],
                      S.fit_options(warm_start=True, validation_episode_ids=val, **S.COMPARISON),
                      warm_start_source='installed', expected_initial_model_sha256=a0_sha)

        u = {dd: T.assess(eps, dd, ev_a0) for dd in sorted({dd for _, dd in bl_arms})}
        bl_rows = {}
        for lam, dd in bl_arms:
            name = bl_name(lam, dd)
            bl_rows[name] = T.blended_rows(eps, lam, dd, u[dd], a0_sha)
            c.observe(d, bl_rows[name])
            fit_and_grade(name, [r['record_id'] for r in bl_rows[name]], S.fit_options(validation_episode_ids=val))
        best = max((bl_name(lam, dd) for lam, dd in bl_arms), key=lambda n: rate(out['arms'][n]['grade']))
        lam_b, d_b = next((lam, dd) for lam, dd in bl_arms if bl_name(lam, dd) == best)
        out['best_bl_this_draw'] = best
        log('best BL this draw:', best)

        # BL-iter: the best BL model as the assessor (refit as candidate if another arm holds the slot)
        st_now = c.state(d)
        best_sha = out['arms'][best]['sha256']
        if c.model_sha(st_now, 'candidate') != best_sha:
            refit = c.train(d, f'a3-d{draw}-{best}-refit', S.fit_options(validation_episode_ids=val), install=False,
                            observation_ids=[r['record_id'] for r in bl_rows[best]], name=f'd{draw}-{best}-refit')
            out['best_bl_refit'] = {'sha256': refit['sha256'], 'identical': refit['sha256'] == best_sha}
            log('refit of best BL identical:', refit['sha256'] == best_sha)
            best_sha = refit['sha256']
        ev_best = model_evaluator(c, d, 'candidate', best_sha, name=best)
        u_it = T.assess(eps, d_b, ev_best)
        it_rows = T.blended_rows(eps, lam_b, d_b, u_it, best_sha, arm='BLit')
        c.observe(d, it_rows)
        fit_and_grade('BL-iter', [r['record_id'] for r in it_rows], S.fit_options(validation_episode_ids=val))

        # BLG: best BL rows + sibling rows (target = Search_{d-1}(c; A0), exact where terminal), comparison
        sib = T.sibling_rows(eps, d_b, ev_a0, a0_sha, stride=1)
        c.observe(d, sib)
        fit_and_grade('BLG', [r['record_id'] for r in bl_rows[best] + sib],
                      S.fit_options(validation_episode_ids=val, **S.COMPARISON))
    finally:
        c.delete_domain(d)
        log('deleted', d)
    return out


def main():
    draws = [int(x) for x in sys.argv[1:]] or [1, 2, 3]
    run = Run('A3')
    c = connect(run)
    v = c.check_build('start')
    suite = SU.load('eval_s')
    run.note(host=S.HOST, build=v, draws=draws, bl_grid=BL_GRID, draw_seeds=DRAW_SEEDS,
             eval_s_sha256=json.loads((SU.SUITES / 'MANIFEST.json').read_text())['eval_s']['sha256'])
    results = {}
    for draw in draws:
        if draw == 1:
            arms = BL_GRID
        else:
            prev = sorted((RUNS / ('_mock' if MOCK else '') / 'A3').glob('*/results_draw1.json'))
            d1 = json.loads(prev[-1].read_text())
            ranked = sorted((n for n in d1['arms'] if n.startswith('BL(')), key=lambda n: -rate(d1['arms'][n]['grade']))
            arms = [next(x for x in BL_GRID if bl_name(*x) == n) for n in ranked[:2]]
            run.log(f'draw {draw}: BL arms from draw 1 ranking {ranked[:2]}')
        results[draw] = run_draw(run, c, draw, suite, arms)
    results['version_end'] = c.check_build('end')
    results['requests'] = c.counts
    run.write_results(results)
    run.finish(requests=c.counts['requests'])


def summarize(paths):
    """Average Eval-S depth-1 rates over draws and apply the pre-registered A3 decision rule."""
    draws = [json.loads(Path(p).read_text()) for p in paths]
    names = sorted(set.intersection(*[set(d['arms']) for d in draws]))
    mean = {n: st.mean(rate(d['arms'][n]['grade']) for d in draws) for n in names}
    bls = [n for n in names if n.startswith('BL(')]
    best = max(bls, key=mean.get)
    # plan 5 A3: "If no BL arm beats A0 by at least 0.03 averaged over draws, Track B runs A0 against
    # ROB-style targets instead." The plan's BL arms are the BL(lam, d) grid, BLG ("the best BL plus
    # sibling rows") and BL-iter; both the grid-only and the full readings are reported.
    family = [n for n in names if n.startswith('BL')]
    deltas = {n: mean[n] - mean['A0'] for n in family}
    return {'mean_rate_depth1': mean, 'lam_d_star': best,
            'bl_grid_beats_a0_by_0.03': mean[best] - mean['A0'] >= 0.03,
            'bl_arm_deltas_vs_a0': deltas,
            'bl_beats_a0_by_0.03': any(d >= 0.03 for d in deltas.values()),
            'use_siblings': mean.get('BLG', 0) - mean[best] >= 0.03}


if __name__ == '__main__':
    main()
