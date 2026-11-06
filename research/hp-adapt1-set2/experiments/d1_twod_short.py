"""D1: 2D short chains, oracle-graded (plan section 8).

A3's arms (A0, ROB, best BL = lam*, d* from G1, BLG) on the 12 2D training chains of length 14-16
(sequences(81001, [14,15,16]*8)[:12]) with 96 random folds (8 per chain), 2 data draws (Random(81003),
Random(81004): the plan names no draw seeds). Graded on Eval-S-2D (12 held-out chains, exact 2D oracle)
at depths 1-3, against the 2D contact heuristic.
Gate GD: run D2 if D1's best arm beats the 2D heuristic at depth 1 or depth 3 on Eval-S-2D.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hpx import settings as S, suites as SU, targets as T  # noqa: E402
from hpx.evaluate import grade_short, model_evaluator  # noqa: E402
from hpx.planner import Heuristic  # noqa: E402
from hpx.runlog import ROOT, Run, connect, sha256_json  # noqa: E402

DRAW_SEEDS = {1: 81003, 2: 81004}


def rate(g, d=1):
    return g[f'depth{d}']['all']['best_move_rate']


def run_draw(run, c, draw, suite, cfg):
    train, _ = SU.twod_short_chains()
    eps = T.random_episodes(train, 8, random.Random(DRAW_SEEDS[draw]), f'd1d{draw}', dim=2)
    val = T.validation_episodes(eps, fit_index=400 + draw)
    out = {'validation_episode_ids': val, 'episodes_sha256': sha256_json([e.to_json() for e in eps]), 'arms': {}}
    d = c.create_domain('d1', 'all', draw)
    out['domain'] = d

    def fit_and_grade(name, ids, opts, install=False, **kw):
        fit = c.train(d, f'd1-d{draw}-{name}', opts, install=install, observation_ids=ids, name=f'd{draw}-{name}', **kw)
        rep = fit['state']['training']['report']
        ev = model_evaluator(c, d, 'installed' if install else 'candidate', fit['sha256'], name=name)
        g = grade_short(ev, suite, depths=(1, 2, 3), greedy_depths=(1, 3))
        out['arms'][name] = {'sha256': fit['sha256'], 'fit_seconds': fit['elapsed'], 'n_rows': len(ids),
                             'selected_epoch': rep.get('selected_epoch'), 'report': rep, 'grade': g}
        run.log(f'[draw {draw}] {name:10s} d1 {rate(g, 1):.3f} d2 {rate(g, 2):.3f} d3 {rate(g, 3):.3f} | corr V* '
                f'{g["depth1"]["all"]["corr_vstar"]:.3f} | greedy d1 {g["greedy_depth1"]["all"]["greedy_contacts"]}'
                f'/{g["greedy_depth1"]["all"]["optimal_contacts"]} d3 {g["greedy_depth3"]["all"]["greedy_contacts"]} | {fit["elapsed"]}s')
        run.write_results(out, f'results_draw{draw}.json')
        return fit, ev
    try:
        c.configure(d, S.value_config('object_graph'))
        a0_rows = T.terminal_rows(eps)
        c.observe(d, a0_rows)
        a0_ids = [r['record_id'] for r in a0_rows]
        a0, ev_a0 = fit_and_grade('A0', a0_ids, S.fit_options(validation_episode_ids=val), install=True)
        rob_rows = T.continuation_sibling_rows(eps, stride=3)
        c.observe(d, rob_rows)
        fit_and_grade('ROB', a0_ids + [r['record_id'] for r in rob_rows],
                      S.fit_options(warm_start=True, validation_episode_ids=val, **S.COMPARISON),
                      warm_start_source='installed', expected_initial_model_sha256=a0['sha256'])
        lam, dd = cfg['lam_star'], cfg['d_star']
        u = T.assess(eps, dd, ev_a0)
        bl_rows = T.blended_rows(eps, lam, dd, u, a0['sha256'])
        c.observe(d, bl_rows)
        fit_and_grade('BL', [r['record_id'] for r in bl_rows], S.fit_options(validation_episode_ids=val))
        sib = T.sibling_rows(eps, dd, ev_a0, a0['sha256'], stride=1)
        c.observe(d, sib)
        fit_and_grade('BLG', [r['record_id'] for r in bl_rows + sib], S.fit_options(validation_episode_ids=val, **S.COMPARISON))
    finally:
        c.delete_domain(d)
    return out


def main():
    cfg = json.loads(Path(__import__('os').environ.get('HPX2_TRACK_B', ROOT / 'manifests' / 'track_b.json')).read_text())
    run = Run('D1')
    c = connect(run)
    v = c.check_build('start')
    suite = SU.load('eval_s_2d')
    heur = {'grade': grade_short(Heuristic(), suite, depths=(1, 2, 3), greedy_depths=(1, 3))}
    run.log('2D heuristic', {f'd{d}': round(rate(heur['grade'], d), 3) for d in (1, 2, 3)})
    run.note(host=S.HOST, build=v, draw_seeds=DRAW_SEEDS, track_b=cfg,
             eval_s_2d_sha256=json.loads((SU.SUITES / 'MANIFEST.json').read_text())['eval_s_2d']['sha256'])
    results = {'heuristic': heur}
    for draw in DRAW_SEEDS:
        results[draw] = run_draw(run, c, draw, suite, cfg)
        run.write_results(results)
    arms = ['A0', 'ROB', 'BL', 'BLG']
    mean = {a: {d: sum(rate(results[k]['arms'][a]['grade'], d) for k in DRAW_SEEDS) / len(DRAW_SEEDS) for d in (1, 2, 3)}
            for a in arms}
    best = max(arms, key=lambda a: mean[a][1])
    results['GD'] = {'mean_rates': mean, 'heuristic': {d: rate(heur['grade'], d) for d in (1, 2, 3)}, 'best_arm': best,
                     'run_D2': mean[best][1] > rate(heur['grade'], 1) or mean[best][3] > rate(heur['grade'], 3)}
    run.log('GD', results['GD'])
    results['version_end'] = c.check_build('end')
    results['requests'] = c.counts
    run.write_results(results)
    run.finish(requests=c.counts['requests'])


if __name__ == '__main__':
    main()
