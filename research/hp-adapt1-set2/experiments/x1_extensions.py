"""X1 (exploratory, not in the plan): native search extensions, graded exactly on short chains.

Every planning call in the program so far used max_extensions 0. Here, on 100 informative Eval-S decisions
(Random(80950) sample) with one learned evaluator (A0 on A3 draw-1 data, installed), native planning is run
on a supplied depth-4 graph under:
  plain depth 3 (must equal the client-side search: sanity check), plain depth 4,
  depth 3 + max_extensions in {2, 4, 8} x extension_mode in {all, competitive, equilibrium}.
Extensions can only expand into supplied nodes, so the graph is built to depth 4 (deeper graphs exceed the
1 MiB body limit); nodes carry only non-default fields. Decisions whose depth-4 body exceeds the 900 KiB guard
are skipped (count recorded). Grading: native action among the oracle-optimal moves; reported with
evaluations and extensions used, next to client-side depth 3/4 and the heuristic on the same decisions.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hpx import settings as S, suites as SU, targets as T  # noqa: E402
from hpx.client import MAX_BODY, compact  # noqa: E402
from hpx.evaluate import model_evaluator  # noqa: E402
from hpx.hpenv import Fold  # noqa: E402
from hpx.planner import Heuristic, decide  # noqa: E402
from hpx.runlog import MOCK, Run, connect  # noqa: E402

ACT = lambda a: ','.join(map(str, a))  # noqa: E731
CONFIGS = [('plain', 3, 0, 'all'), ('plain', 4, 0, 'all')] + [
    ('ext', 3, e, m) for m in ('all', 'competitive', 'equilibrium') for e in (2, 4, 8)]


def lean_graph(fold, depth):
    nodes, counter = [], [0]

    def add(d):
        nid = f'n{counter[0]}'
        counter[0] += 1
        o = fold.observation()
        node = {'id': nid, 'maximizing': True, 'objects': o['objects'], 'relations': o['relations'],
                'conditions': o['conditions']}
        nodes.append(node)
        if fold.terminal:
            node['terminal_value'] = fold.utility()
        elif d > 0:
            node['children'] = []
            for a in fold.legal():
                fold.push(a)
                node['children'].append({'action': ACT(a), 'target': add(d - 1)})
                fold.pop()
        return nid
    add(depth)
    return {k: v for k, v in nodes[0].items() if k != 'children'}, nodes


def main():
    run = Run('X1')
    c = connect(run)
    v = c.check_build('start')
    suite = SU.load('eval_s')
    seqs = {ch['ci']: ch['sequence'] for ch in suite['chains']}
    informative = [d for d in suite['decisions'] if len(d['opt']) < len(d['children'])]
    rng = random.Random(80950)
    pool = rng.sample(informative, len(informative))
    train, _ = SU.set1_chains()
    eps = T.random_episodes(train, 8, random.Random(61107), 'a3d1')
    rows = T.terminal_rows(eps)
    d = c.create_domain('x1', 'ext', 0)
    run.note(host=S.HOST, build=v, domain=d, configs=CONFIGS, decision_seed=80950)
    results = {'configs': {}, 'skipped_for_size': 0}
    try:
        c.configure(d, S.value_config('object_graph'))
        c.observe(d, rows)
        fit = c.train(d, 'x1-a0', S.fit_options(validation_episode_ids=T.validation_episodes(eps, fit_index=0)),
                      install=True, observation_ids=[r['record_id'] for r in rows], name='a0')
        sha = fit['sha256']
        ev = model_evaluator(c, d, 'installed', sha)
        chosen, graphs = [], []
        for dec in pool:
            f = Fold(seqs[dec['ci']], 3, dec['chain'])
            root, nodes = lean_graph(f, 4)
            body = {'session_id': 'ignored', 'relation': S.RELATION, 'policy': S.POLICY, 'root': root, 'nodes': nodes,
                    'evaluator_source': 'value_model', 'expected_model_sha256': sha, 'options': {}}
            if len(compact(body).encode()) > MAX_BODY - 4096:
                results['skipped_for_size'] += 1
                continue
            chosen.append((dec, f))
            graphs.append((root, nodes))
            if len(chosen) == 100:
                break
        run.log(f'{len(chosen)} decisions (skipped {results["skipped_for_size"]} over the size guard); model {sha[:12]}')
        opt = [{tuple(a) for a in dec['opt']} for dec, _ in chosen]
        folds = [f for _, f in chosen]
        ref = {}
        for name, evaluator in (('client_model', ev), ('heuristic', Heuristic())):
            for depth in (3, 4):
                res = decide(folds, depth, evaluator)
                hits = []
                for (legal, vals, _), o in zip(res, opt):
                    m = max(vals)
                    best = [tuple(a) for a, x in zip(legal, vals) if x == m]
                    hits.append(sum(a in o for a in best) / len(best))
                ref[f'{name}_depth{depth}'] = {'best_move_rate': sum(hits) / len(hits),
                                               'actions': [[ACT(a) for a, x in zip(legal, vals) if x == max(vals)] for legal, vals, _ in res],
                                               'values': [max(vals) for _, vals, _ in res]}
                run.log(f'{name} depth {depth}: best-move {ref[f"{name}_depth{depth}"]["best_move_rate"]:.3f}')
        results['reference'] = {k: {'best_move_rate': x['best_move_rate']} for k, x in ref.items()}
        for kind, depth, ext, mode in CONFIGS:
            key = f'{kind}-d{depth}-e{ext}-{mode}'
            hits, evals, exts, cexts, parity = [], [], [], [], []
            for i, ((root, nodes), o) in enumerate(zip(graphs, opt)):
                resp = c.plan(d, {'relation': S.RELATION, 'policy': S.POLICY, 'evaluator_source': 'value_model',
                                  'expected_model_sha256': sha, 'root': root, 'nodes': nodes,
                                  'options': {'max_depth': depth, 'max_extensions': ext, 'extension_mode': mode,
                                              'max_evaluations': 100000, 'exploration_probability': 0,
                                              'cache_values': False, 'reuse_tree': False, 'timeout_seconds': 120}})
                p = resp.get('plan') or {}
                a = tuple(int(x) for x in str(p.get('action')).split(','))
                hits.append(float(a in o))
                evals.append(p.get('evaluations') or 0)
                exts.append(p.get('extensions') or 0)
                cexts.append(p.get('completed_extensions') or 0)
                if kind == 'plain':
                    r = ref[f'client_model_depth{depth}']
                    parity.append(p.get('action') in r['actions'][i] and abs((p.get('value') or 0) - r['values'][i]) < 1e-6)
            out = {'best_move_rate': sum(hits) / len(hits), 'mean_evaluations': sum(evals) / len(evals),
                   'mean_extensions': sum(exts) / len(exts), 'mean_completed_extensions': sum(cexts) / len(cexts),
                   'n': len(hits)}
            if parity:
                out['parity_with_client'] = sum(parity) / len(parity)
            results['configs'][key] = out
            run.log(f'{key:24s} best-move {out["best_move_rate"]:.3f} | evals {out["mean_evaluations"]:.0f} | '
                    f'extensions {out["mean_extensions"]:.2f} (completed {out["mean_completed_extensions"]:.2f})'
                    + (f' | parity {out["parity_with_client"]:.2f}' if parity else ''))
            run.write_results(results)
    finally:
        c.delete_domain(d)
        results['version_end'] = c.check_build('end')
        results['requests'] = c.counts
        run.write_results(results)
        run.finish(requests=c.counts['requests'])


if __name__ == '__main__':
    if MOCK:
        print('note: the mock ignores extensions; this dry run checks plumbing only')
    main()
