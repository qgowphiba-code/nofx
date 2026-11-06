"""A1: representation probe (plan section 5). Can the value model carry the contact heuristic?

Data: 60 chains of length 24-60 (sequences(80501, 24..60 cycling)), 120 random folds + 60 heuristic-guided
folds (depth 1, eps 0.3), Random(80502). Synthetic target = the heuristic's own partial score.
Arms: R0 object graph + 4 BASIC conditions; R1 object graph + 14 features(); R2 projected + 14 features
(mlp_v2). R0/R1 share one object-graph domain (distinct record ids); R2 has its own projected domain.
Measures: sibling agreement on 1,000 held-out states (best predicted child among the heuristic's best),
error in contact units, depth-3 planning on Eval-Liu and 20 Eval-L chains vs the heuristic's distribution.
Held-out states come from 20 further chains (sequences(80503, ...), folds Random(80504)): new seeds, since
the plan does not name a held-out set.
"""
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hpx import settings as S, suites as SU, targets as T  # noqa: E402
from hpx.evaluate import long_eval, model_evaluator  # noqa: E402
from hpx.hpenv import BASIC, FEATURES, Fold  # noqa: E402
from hpx.planner import Heuristic, decide, run_folds  # noqa: E402
from hpx.runlog import Run, connect, sha256_json  # noqa: E402
from hpx.vendor import lattice_env, liu_panel  # noqa: E402

LENGTHS = [24 + (i % 37) for i in range(60)]


def chains(seed, n):
    excl = {lattice_env.sequence_key(s) for s in SU.load('eval_l')['sequences'] + [s for _, s in liu_panel()]}
    return lattice_env.sequences(seed, LENGTHS[:n], excluded=excl)


def episodes(seqs, rng, prefix):
    eps = T.random_episodes(seqs, 2, rng, prefix + 'r')
    for ci, seq in enumerate(seqs):
        r = run_folds([seq], 1, Heuristic(), rngs=[rng], eps=0.3, record=True)[0]
        chs = [tuple(map(tuple, step['chain'])) for step in r['trajectory']] + [tuple(map(tuple, r['chain']))]
        eps.append(T.Episode(f'{prefix}g-c{ci}', seq, chs))
    return eps


def heuristic_rows(eps, arm, conditions):
    rows = []
    for e in eps:
        for k in range(e.T):
            f = e.fold(k)
            obs = f.observation(conditions)
            rows.append({'record_id': f'A1-{arm}-{e.id}-s{k}', 'episode_id': e.id, 'step': k,
                         **{x: obs[x] for x in ('objects', 'relations', 'conditions', 'include_positions',
                                                'include_moments', 'policy_features')},
                         'target': f.score(), 'metadata': {'prov': {'arm': arm, 'target': 'heuristic_partial_score'}}})
    return rows


def sibling_agreement(ev, states):
    res_m = decide(states, 1, ev)
    res_h = decide(states, 1, Heuristic())
    agree, agree_inf, err_c, err_kids = [], [], [], []
    for f, (legal, vm, _), (_, vh, _) in zip(states, res_m, res_h):
        hb = {a for a, v in zip(legal, vh) if v == max(vh)}
        mb = [a for a, v in zip(legal, vm) if v == max(vm)]
        hit = sum(a in hb for a in mb) / len(mb)
        agree.append(hit)
        if len(hb) < len(legal):
            agree_inf.append(hit)
        for a, x, y in zip(legal, vm, vh):
            err_kids.append(abs(x - y) * 2 * f.ub)
    sv = ev.evaluate([ev.payload(f) for f in states])
    err_c = [abs(v - f.score()) * 2 * f.ub for v, f in zip(sv, states)]
    return {'agreement_all': st.mean(agree), 'agreement_informative': st.mean(agree_inf), 'n_informative': len(agree_inf),
            'n_states': len(states), 'mae_contacts_state': st.mean(err_c), 'mae_contacts_children': st.mean(err_kids),
            'p90_err_contacts_state': sorted(err_c)[int(.9 * len(err_c))]}


def main():
    run = Run('A1')
    c = connect(run)
    v = c.check_build('start')
    rng = random.Random(80502)
    train_seqs = chains(80501, 60)
    eps = episodes(train_seqs, rng, 'a1')
    hrng = random.Random(80504)
    held_seqs = [s for s in chains(80503, 60) if lattice_env.sequence_key(s) not in {lattice_env.sequence_key(x) for x in train_seqs}][:20]
    held_eps = episodes(held_seqs, hrng, 'a1h')
    pool = [(e, k) for e in held_eps for k in range(e.T) if len(e.fold(k).legal()) >= 2]
    held = [e.fold(k) for e, k in hrng.sample(pool, 1000)]
    val = T.validation_episodes(eps, fit_index=100)
    liu = [s for _, s in liu_panel()]
    evl = SU.load('eval_l')['sequences'][::2]
    base_liu = SU.load('baselines_eval_liu')['heuristic']['3']
    rows = {'R0': heuristic_rows(eps, 'R0', BASIC), 'R1': heuristic_rows(eps, 'R1', FEATURES), 'R2': heuristic_rows(eps, 'R2', FEATURES)}
    run.note(host=S.HOST, build=v, n_states=len(rows['R0']), n_episodes=len(eps), validation_episode_ids=val,
             rows_sha256={k: sha256_json(r) for k, r in rows.items()}, train_lengths=LENGTHS, held_out={'seed_chains': 80503, 'seed_folds': 80504})
    run.log(f'{len(eps)} episodes, {len(rows["R0"])} states per arm; held-out {len(held)} states; heuristic Liu depth 3 '
            f'{base_liu["mean_total"]} [p5 {base_liu["p5_total"]}, p95 {base_liu["p95_total"]}]')
    heur = {'liu': long_eval(Heuristic(), liu, 3, seed=1), 'eval_l20': long_eval(Heuristic(), evl, 3, seed=1)}
    results = {'heuristic_one_seed': heur, 'heuristic_card_liu_depth3': {k: base_liu[k] for k in ('mean_total', 'sd_total', 'p5_total', 'p95_total')}}
    plan = [('R0', 'object_graph', S.OBJECT_BASE, BASIC), ('R1', 'object_graph', S.OBJECT_BASE, FEATURES),
            ('R2', 'projected', S.PROJECTED_BASE, FEATURES)]
    domains = {}
    try:
        for i, (arm, rep, base, conds) in enumerate(plan):
            if rep not in domains:
                domains[rep] = c.create_domain('a1', 'og' if rep == 'object_graph' else 'proj', 0)
                c.configure(domains[rep], S.value_config(rep))
            d = domains[rep]
            c.observe(d, rows[arm])
            fit = c.train(d, f'a1-{arm}', S.fit_options(base, warm_start=False, validation_episode_ids=val), install=False,
                          observation_ids=[r['record_id'] for r in rows[arm]], name=arm)
            rep_ = fit['state']['training']['report']
            run.log(f'{arm} fitted {fit["sha256"][:12]} in {fit["elapsed"]}s, selected_epoch {rep_.get("selected_epoch")}, '
                    f'validation_skill {rep_.get("validation_skill")}')
            ev = model_evaluator(c, d, 'candidate', fit['sha256'], conditions=conds, name=arm)
            sib = sibling_agreement(ev, held)
            run.log(f'{arm} sibling agreement {sib["agreement_all"]:.3f} (informative {sib["agreement_informative"]:.3f}, '
                    f'n={sib["n_informative"]}); MAE {sib["mae_contacts_state"]:.2f} contacts')
            lv = long_eval(ev, liu, 3, seed=1)
            run.log(f'{arm} Liu depth 3 total {lv["total"]} (heuristic band {base_liu["p5_total"]}-{base_liu["p95_total"]}) {lv["per_chain"]}')
            ll = long_eval(ev, evl, 3, seed=1)
            card = SU.load('baselines_eval_l')['heuristic']['3']['per_chain_mean'][::2] if (SU.SUITES / 'baselines_eval_l.json').exists() else None
            run.log(f'{arm} Eval-L(20) depth 3 total {ll["total"]}' + (f' vs heuristic mean {sum(card):.1f}' if card else ''))
            results[arm] = {'sha256': fit['sha256'], 'domain': d, 'fit_seconds': fit['elapsed'], 'report': rep_,
                            'sibling': sib, 'liu_depth3': lv, 'eval_l20_depth3': ll, 'predict_stats': ev.stats,
                            'in_band_liu': base_liu['p5_total'] <= lv['total'] <= base_liu['p95_total'],
                            'below_p5_liu': lv['total'] < base_liu['p5_total']}
            run.write_results(results)
        qual = [a for a, *_ in plan if results[a]['liu_depth3']['total'] >= base_liu['p5_total']
                and results[a]['sibling']['agreement_all'] >= 0.95]
        results['decision'] = {'qualifying': qual, 'r_star': qual[0] if qual else max(
            (a for a, *_ in plan), key=lambda a: (results[a]['liu_depth3']['total'], results[a]['sibling']['agreement_all'])),
            'rule': 'cheapest representation with Liu depth-3 total inside the heuristic band (>= p5) and agreement >= 0.95; '
                    'R0 kept if it qualifies; else the best by Liu total'}
        run.log('decision', results['decision'])
    finally:
        for d in domains.values():
            c.delete_domain(d)
        results['version_end'] = c.check_build('end')
        results['requests'] = c.counts
        run.write_results(results)
        run.finish(requests=c.counts['requests'])


if __name__ == '__main__':
    main()
