"""E1: game value learning, solver-graded (plan section 9). Gravity four-in-a-row, 5 wide x 4 high.

3 draws of 600 random games (Random(82001 + draw)). Targets from X's view; zbar uses gamma = 0.99.
Arms: A0 (lam 0), BL (lam* from G1, depth 2 with MIN nodes, assessor = the draw's A0), BLG (BL rows +
sibling rows, target = Search_1(child; A0), comparison). Graded on 800 informative held-out positions
(suites/game_e1.json: random games Random(82100) + solver-optimal lines): best-move rate vs the solver at
depths 1 and 2, regret, W/D/L accuracy; compared with the line heuristic and neutral-leaf search.
Gate GE: run E2 if the best learned arm's depth-2 best-move rate >= the heuristic's at depth 2.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hpx import game as G, settings as S, suites as SU  # noqa: E402
from hpx.runlog import ROOT, Run, connect, sha256_json  # noqa: E402

GAMMA = 0.99
FIELDS = ('objects', 'relations', 'conditions', 'include_positions', 'include_moments', 'policy_features')


def state_fields(p):
    o = p.observation()
    return {k: o[k] for k in FIELDS}


def zbar(U, T, k):
    return .5 + (U - .5) * GAMMA ** (T - k)


def build_rows(games, draw, lam=0.0, u=None, assessor=None):
    rows = []
    for gi, cols_seq in enumerate(games):
        pos = [G.Pos(c) for c in cols_seq]
        U, T = pos[-1].value(), len(pos) - 1
        for k, p in enumerate(pos[:-1]):
            z = zbar(U, T, k)
            if lam:
                y = min(1., max(0., lam * u[(gi, k)] + (1 - lam) * z))
                rid = f'E1BL-l{lam:g}-d2-a{assessor[:10]}-d{draw}-g{gi}-s{k}'
            else:
                y, rid = z, f'E1A0-g{GAMMA:g}-d{draw}-g{gi}-s{k}'
            rows.append({'record_id': rid, 'episode_id': f'e1d{draw}-g{gi}', 'step': k, **state_fields(p), 'target': y,
                         'metadata': {'prov': {'lam': lam, 'zbar': z, 'U': U, 'gamma': GAMMA,
                                               **({'u': u[(gi, k)], 'assessor': assessor} if lam else {})}}})
    return rows


def sibling_stride(games, n_other, cap=19500):
    """Smallest state stride keeping BL rows + sibling rows within the 20,000-id per-fit limit (plan 12)."""
    for s in range(1, 20):
        n = sum(len(G.Pos(cols).legal()) for g in games for k, cols in enumerate(g[:-1]) if k % s == 0)
        if n_other + n <= cap:
            return s, n
    raise ValueError('no stride fits')


def sibling_rows(games, draw, ev, assessor, stride=1):
    items = []
    for gi, cols_seq in enumerate(games):
        for k, cols in enumerate(cols_seq[:-1]):
            if k % stride:
                continue
            p = G.Pos(cols)
            for c in p.legal():
                items.append((gi, k, c, p.play(c)))
    vals = G.search_value([it[3] for it in items], 1, ev)
    return [{'record_id': f'E1SIB-d1-a{assessor[:10]}-d{draw}-g{gi}-s{k}-c{c}', 'episode_id': f'e1d{draw}-g{gi}',
             'step': k + 1, **state_fields(child), 'target': v,
             'metadata': {'comparison_group': f'e1d{draw}-g{gi}-s{k}', 'prov': {'exact': child.terminal}}}
            for (gi, k, c, child), v in zip(items, vals)]


def assess(games, ev, depth=2):
    keys, positions = [], []
    for gi, cols_seq in enumerate(games):
        for k, cols in enumerate(cols_seq[:-1]):
            keys.append((gi, k))
            positions.append(G.Pos(cols))
    return dict(zip(keys, G.search_value(positions, depth, ev)))


def validation(draw, n_games):
    rng = random.Random(80700 + 500 + draw)
    return sorted(f'e1d{draw}-g{gi}' for gi in rng.sample(range(n_games), round(0.2 * n_games)))


def main():
    cfg = json.loads(Path(__import__('os').environ.get('HPX2_TRACK_B', ROOT / 'manifests' / 'track_b.json')).read_text())
    lam = cfg['lam_star'] or 0.75
    run = Run('E1')
    c = connect(run)
    v = c.check_build('start')
    if not (SU.SUITES / 'game_e1.json').exists():
        solver = G.Solver()
        solver.value(G.Pos())
        SU.freeze('game_e1', G.build_suite(solver))
    suite = SU.load('game_e1')
    results = {'baselines': {}, 'draws': {}}
    for name, ev in (('heuristic', G.GameHeuristic()), ('neutral', G.GameNeutral())):
        results['baselines'][name] = {f'depth{d}': G.score(suite, ev, d) for d in (1, 2)}
    run.log('baselines', json.dumps(results['baselines']))
    run.note(host=S.HOST, build=v, lam=lam, gamma=GAMMA, suite_sha256=json.loads((SU.SUITES / 'MANIFEST.json').read_text())['game_e1'])
    for draw in (1, 2, 3):
        games = G.episodes(600, random.Random(82001 + draw))
        val = validation(draw, len(games))
        out = {'n_games': len(games), 'games_sha256': sha256_json(games), 'arms': {}}
        d = c.create_domain('e1', 'all', draw)
        try:
            c.configure(d, {**S.value_config('object_graph'), 'max_samples': 60000})

            def fit_grade(name, rows_ids, opts, install=False):
                fit = c.train(d, f'e1-d{draw}-{name}', opts, install=install, observation_ids=rows_ids, name=f'd{draw}-{name}')
                src = 'installed' if install else 'candidate'
                ev = G.GameModel(lambda xs, s=fit['sha256'], src=src: c.predict(d, xs, src, s), name=name)
                g = {f'depth{dd}': G.score(suite, ev, dd) for dd in (1, 2)}
                out['arms'][name] = {'sha256': fit['sha256'], 'fit_seconds': fit['elapsed'], 'n_rows': len(rows_ids),
                                     'selected_epoch': fit['state']['training']['report'].get('selected_epoch'), 'grade': g}
                run.log(f'[draw {draw}] {name:4s} best-move d1 {g["depth1"]["best_move_rate"]:.3f} d2 {g["depth2"]["best_move_rate"]:.3f} '
                        f'| W/D/L acc {g["depth1"]["wdl_accuracy"]:.3f} | regret d2 {g["depth2"]["mean_regret"]:.4f} | {fit["elapsed"]}s')
                return fit, ev
            a0_rows = build_rows(games, draw)
            c.observe(d, a0_rows)
            a0, ev_a0 = fit_grade('A0', [r['record_id'] for r in a0_rows], S.fit_options(validation_episode_ids=val), install=True)
            u = assess(games, ev_a0, 2)
            bl_rows = build_rows(games, draw, lam, u, a0['sha256'])
            c.observe(d, bl_rows)
            fit_grade('BL', [r['record_id'] for r in bl_rows], S.fit_options(validation_episode_ids=val))
            stride, n_sib = sibling_stride(games, len(bl_rows))
            out['sibling_stride'] = stride
            run.log(f'[draw {draw}] sibling rows: every {stride} state(s), {n_sib} rows')
            sib = sibling_rows(games, draw, ev_a0, a0['sha256'], stride)
            c.observe(d, sib)
            fit_grade('BLG', [r['record_id'] for r in bl_rows + sib], S.fit_options(validation_episode_ids=val, **S.COMPARISON))
        finally:
            c.delete_domain(d)
        results['draws'][draw] = out
        run.write_results(results)
    arms = ['A0', 'BL', 'BLG']
    mean2 = {a: sum(results['draws'][k]['arms'][a]['grade']['depth2']['best_move_rate'] for k in (1, 2, 3)) / 3 for a in arms}
    h2 = results['baselines']['heuristic']['depth2']['best_move_rate']
    best = max(arms, key=mean2.get)
    results['GE'] = {'mean_depth2': mean2, 'heuristic_depth2': h2, 'best_arm': best, 'run_E2': mean2[best] >= h2}
    run.log('GE', results['GE'])
    results['version_end'] = c.check_build('end')
    results['requests'] = c.counts
    run.write_results(results)
    run.finish(requests=c.counts['requests'])


if __name__ == '__main__':
    main()
