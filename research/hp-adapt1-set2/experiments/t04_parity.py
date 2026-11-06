"""T0.4: client-side vs native planning (plan section 4).

With the T0.3 A0 model, for 100 short and 30 long states at depths 1-3, compare native planning's
plan.action / plan.value with the client-side maximum over predicted leaves.
Pass: the native action is in the client argmax set (exact ties) and |dvalue| < 1e-6 in every case.
Long states whose depth-3 graph body exceeds the 900 KiB guard are replaced (count recorded).
Deletes the T0.3 domain at the end (verified 404).
Usage: t04_parity.py [T0.3 run dir]   (default: latest)
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hpx import settings as S, suites as SU  # noqa: E402
from hpx.client import MAX_BODY, compact  # noqa: E402
from hpx.evaluate import model_evaluator  # noqa: E402
from hpx.hpenv import Fold  # noqa: E402
from hpx.planner import decide  # noqa: E402
from hpx.runlog import RUNS, MOCK, Run, connect  # noqa: E402

ACT = lambda a: ','.join(map(str, a))  # noqa: E731


def graph(fold, depth):
    """Native planning graph to `depth`: terminal nodes carry exact terminal_value; depth-0 nodes are leaves."""
    nodes, counter = [], [0]

    def add(d):
        nid = f'n{counter[0]}'
        counter[0] += 1
        node = {'id': nid, 'maximizing': True, **fold.observation()}
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
    root = {k: v for k, v in nodes[0].items() if k != 'children'}
    return root, nodes


def main():
    t03 = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((RUNS / ('_mock' if MOCK else '') / 'T0.3').glob('*/results.json'))[-1].parent
    prev = json.loads((t03 / 'results.json').read_text())
    domain, sha = prev['domain_kept_for_T0.4'], prev['A0']['sha256']
    if MOCK:
        raise SystemExit('T0.4 dry run lives in tests/test_parity.py (mock state does not persist across processes)')
    run = Run('T0.4')
    c = connect(run)
    v = c.check_build('start')
    run.note(host=S.HOST, build=v, domain=domain, model_sha256=sha, t03_run=str(t03))
    ev = model_evaluator(c, domain, 'installed', sha)
    rng = random.Random(80903)
    suite = SU.load('eval_s')
    seqs = {ch['ci']: ch['sequence'] for ch in suite['chains']}
    short = [Fold(seqs[d['ci']], 3, d['chain']) for d in rng.sample(suite['decisions'], 100)]
    long_seqs = SU.load('eval_l')['sequences']
    longs, skipped = [], 0
    while len(longs) < 30:
        seq = rng.choice(long_seqs)
        f = Fold(seq)
        stop = rng.randrange(2, len(seq) - 4)
        while len(f.chain) < stop and not f.terminal:
            f.push(rng.choice(f.legal()))
        if f.terminal:
            continue
        root, nodes = graph(f, 3)
        size = len(compact({'session_id': 'ignored', 'relation': S.RELATION, 'policy': S.POLICY, 'root': root,
                            'nodes': nodes, 'evaluator_source': 'value_model', 'expected_model_sha256': sha,
                            'options': {}}).encode())
        if size > MAX_BODY - 2048:
            skipped += 1
            continue
        longs.append(f)
    run.log(f'states: {len(short)} short, {len(longs)} long ({skipped} long states replaced: depth-3 body over guard)')
    cases, fails = [], []
    for kind, states in (('short', short), ('long', longs)):
        for depth in (1, 2, 3):
            res = decide(states, depth, ev)
            for i, (f, (legal, vals, nl)) in enumerate(zip(states, res)):
                root, nodes = graph(f, depth)
                body = {'relation': S.RELATION, 'policy': S.POLICY, 'evaluator_source': 'value_model',
                        'expected_model_sha256': sha, 'root': root, 'nodes': nodes,
                        'options': {'max_depth': depth, 'max_evaluations': 100000, 'exploration_probability': 0,
                                    'cache_values': False, 'reuse_tree': False, 'timeout_seconds': 120}}
                resp = c.plan(domain, body)
                p = resp.get('plan') or {}
                served = (resp.get('evaluator') or {}).get('model_sha256')
                m = max(vals)
                best = {ACT(a) for a, x in zip(legal, vals) if x == m}
                case = {'kind': kind, 'i': i, 'depth': depth, 'L': len(f.seq), 'placed': len(f.chain),
                        'native_action': p.get('action'), 'native_value': p.get('value'), 'client_value': m,
                        'client_best': sorted(best), 'dvalue': abs((p.get('value') or 0) - m),
                        'native_evaluations': p.get('evaluations'), 'client_leaves': nl,
                        'completed_depth': p.get('completed_depth'), 'budget_exhausted': p.get('budget_exhausted'),
                        'deadline_reached': p.get('deadline_reached'), 'served_sha_ok': served == sha}
                case['pass'] = (case['native_action'] in best and case['dvalue'] < 1e-6 and case['served_sha_ok'])
                cases.append(case)
                if not case['pass']:
                    fails.append(case)
                    run.log('MISMATCH', json.dumps(case))
            ok = sum(x['pass'] for x in cases if x['kind'] == kind and x['depth'] == depth)
            n = sum(1 for x in cases if x['kind'] == kind and x['depth'] == depth)
            run.log(f'{kind} depth {depth}: {ok}/{n} pass; max |dvalue| '
                    f'{max(x["dvalue"] for x in cases if x["kind"] == kind and x["depth"] == depth):.3g}')
    passed = not fails
    run.log('T0.4', 'PASS' if passed else f'FAIL ({len(fails)} cases)', '| predict stats', ev.stats)
    c.delete_domain(domain)
    run.log('deleted T0.3 domain', domain)
    out = {'pass': passed, 'n_cases': len(cases), 'n_fail': len(fails), 'long_replaced_for_size': skipped,
           'max_dvalue': max(x['dvalue'] for x in cases), 'cases': cases, 'predict_stats': ev.stats,
           'version_end': c.check_build('end'), 'requests': c.counts}
    run.write_results(out)
    run.finish(passed=passed, requests=c.counts['requests'])


if __name__ == '__main__':
    main()
