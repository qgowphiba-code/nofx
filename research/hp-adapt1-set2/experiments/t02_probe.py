"""T0.2: host, build, limits and throughput (plan section 4).

On one scratch domain: configure, observe 50 small rows, one tiny fit (timed), predict 256 inputs,
oversized predicts around the 1 MiB boundary, one depth-1 planning call, 200 timed predict calls of
125 L50 inputs with 2 in flight; then delete the domain and verify 404. Saves every response shape.
"""
import json
import random
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hpx import settings as S  # noqa: E402
from hpx.client import Client, compact  # noqa: E402
from hpx.hpenv import Fold  # noqa: E402
from hpx.runlog import Run  # noqa: E402
from hpx.vendor import lattice_env  # noqa: E402


def shape(obj, depth=0, max_depth=4):
    if isinstance(obj, dict):
        if depth >= max_depth:
            return '{...}'
        return {k: shape(v, depth + 1, max_depth) for k, v in obj.items()}
    if isinstance(obj, list):
        return [shape(obj[0], depth + 1, max_depth), f'len={len(obj)}'] if obj else []
    return type(obj).__name__ if not isinstance(obj, (str, int, float, bool)) or isinstance(obj, bool) else (
        obj if isinstance(obj, str) and len(obj) <= 80 else type(obj).__name__)


def random_fold_states(seq, rng, dim=3):
    f = Fold(seq, dim)
    out = [f.copy()]
    while not f.terminal:
        f.push(rng.choice(f.legal()))
        out.append(f.copy())
    return out


def main():
    run = Run('T0.2')
    c = Client(run.dir, host=S.HOST)
    rng = random.Random(S.SEEDS['t02_probe'])
    out = {}
    out['version_start'] = c.check_build('start')
    run.log('build', out['version_start'])
    domain = c.create_domain('t02', 'probe', 0)
    run.note(host=S.HOST, domain=domain, build=out['version_start'])
    run.log('domain', domain)
    try:
        cfg = c.configure(domain, S.value_config('object_graph'))
        out['configure_shape'] = shape(cfg)
        run.log('configure', json.dumps(cfg)[:400])

        # 50 small rows: random folds of 6 L10-11 chains, target = final utility (lambda = 0 rows)
        seqs = lattice_env.sequences(S.SEEDS['t02_probe'], [10, 11] * 3)
        rows = []
        for e in range(12):
            seq = seqs[e % len(seqs)]
            states = random_fold_states(seq, rng)
            u = states[-1].utility()
            for k, f in enumerate(states[:-1]):
                rows.append({'record_id': f't02-e{e}-s{k}', 'episode_id': f't02-e{e}', 'step': k,
                             **f.observation(), 'target': u})
        rows = rows[:50]
        episodes = sorted({r['episode_id'] for r in rows})
        t0 = time.monotonic()
        obs = c.observe(domain, rows)
        out['observe'] = {'seconds': round(time.monotonic() - t0, 2), 'counts': {k: v for k, v in obs.items() if k != 'last'},
                          'shape': shape(obs['last'])}
        run.log('observe', out['observe'])
        # replay one chunk: must be all duplicates
        rep = c.observe(domain, rows[:5])
        out['observe_replay'] = {k: v for k, v in rep.items() if k != 'last'}
        run.log('observe replay', out['observe_replay'])
        # changed content under an existing id: expect 409 (unretried probe)
        status, data = c.raw('POST', c._v(domain, 'observe'),
                             {'session_id': 'ignored', 'observations': [{**rows[0], 'target': 0.123}]})
        out['observe_conflict'] = {'status': status, 'body': data}
        run.log('observe conflict', status, str(data)[:300])

        st0 = c.state(domain)
        out['state_before_fit_shape'] = shape(st0)
        c.save('state-before-fit', st0)

        # tiny fit: cold, install true, 20 epochs, explicit validation episodes
        val = episodes[:2]
        opts = S.fit_options(epochs=20, validation_episode_ids=val)
        req_id = 't02-tiny-fit-1'
        body = {'session_id': 'ignored', 'request_id': req_id, 'install': True, 'options': opts}
        t0 = time.monotonic()
        with c.fit_lock():
            status, ack = c.raw('POST', c._v(domain, 'train'), body)
            run.log('train ack', status, json.dumps(ack)[:500])
            out['train_ack'] = {'status': status, 'body': ack}
            polls = []
            while True:
                time.sleep(3)
                s = c.state(domain)
                tr = s.get('training') or {}
                polls.append({'t': round(time.monotonic() - t0, 1), 'status': tr.get('status'),
                              'request_id': tr.get('request_id'), 'top_status': s.get('status')})
                if str(tr.get('status', '')).lower() not in ('running', 'queued', 'pending', 'started', 'accepted', '') \
                        or time.monotonic() - t0 > 1800:
                    break
        out['fit_seconds'] = round(time.monotonic() - t0, 1)
        out['fit_polls'] = polls
        c.save('state-after-fit', s)
        out['state_after_fit_shape'] = shape(s, max_depth=6)
        run.log('fit finished', out['fit_seconds'], 's; final training status', polls[-1])
        installed = Client.model_sha(s, 'installed')
        out['installed_sha'] = installed
        run.log('installed sha', installed)

        # predict 256 short inputs
        inputs = [r['objects'] and {k: r[k] for k in ('objects', 'relations', 'conditions', 'include_positions',
                                                          'include_moments', 'policy_features')} for r in rows]
        inputs = (inputs * 6)[:256]
        body = {'session_id': 'ignored', 'model_source': 'installed', 'expected_model_sha256': installed, 'inputs': inputs}
        t0 = time.monotonic()
        status, pred = c.raw('POST', c._v(domain, 'predict'), body)
        out['predict_256'] = {'status': status, 'seconds': round(time.monotonic() - t0, 2), 'bytes': len(compact(body)),
                              'shape': shape(pred)}
        c.save('predict-256', pred)
        run.log('predict 256', out['predict_256'])
        # wrong expected sha: expect a refusal
        status, bad = c.raw('POST', c._v(domain, 'predict'), {**body, 'inputs': inputs[:2], 'expected_model_sha256': '0' * 64})
        out['predict_wrong_sha'] = {'status': status, 'body': bad}
        run.log('predict wrong sha', status, str(bad)[:300])
        # candidate source with no candidate staged
        status, cand = c.raw('POST', c._v(domain, 'predict'), {**body, 'inputs': inputs[:2], 'model_source': 'candidate'})
        out['predict_candidate_none'] = {'status': status, 'body': cand}
        run.log('predict candidate (none staged)', status, str(cand)[:300])

        # L50 inputs for size probes and timing
        l50 = lattice_env.sequences(S.SEEDS['t02_probe'] + 1, [50])[0]
        big = []
        while len(big) < 400:
            big.extend(f.observation() for f in random_fold_states(l50, rng)[20:-1])
        sizes = [len(compact(x)) for x in big]
        out['l50_obs_bytes'] = {'mean': round(st.mean(sizes)), 'max': max(sizes)}

        def pbody(n):
            return {'session_id': 'ignored', 'model_source': 'installed', 'expected_model_sha256': installed, 'inputs': big[:n]}
        probes = []
        for target_kib in (980, 1030, 1100):
            n = 1
            while len(compact(pbody(n + 1)).encode()) <= target_kib * 1024 and n + 1 <= 256:
                n += 1
            b = pbody(n)
            nbytes = len(compact(b).encode())
            status, resp = c.raw('POST', c._v(domain, 'predict'), b, size_guard=False)
            probes.append({'target_kib': target_kib, 'n_inputs': n, 'bytes': nbytes, 'kib': round(nbytes / 1024, 1),
                           'status': status, 'body': resp if status >= 300 else 'ok'})
            run.log('size probe', probes[-1])
        out['size_probes'] = probes

        # depth-1 native planning call on an L50 state
        f = random_fold_states(l50, rng)[25]
        root = {'id': 'n0', 'maximizing': True, **f.observation()}
        nodes = [dict(root, children=[])]
        for i, d in enumerate(f.legal()):
            f.push(d)
            node = {'id': f'n{i + 1}', 'maximizing': True, **f.observation()}
            if f.terminal:
                node['terminal_value'] = f.utility()
            nodes.append(node)
            nodes[0]['children'].append({'action': ','.join(map(str, d)), 'target': node['id']})
            f.pop()
        pb = {'relation': S.RELATION, 'policy': S.POLICY, 'evaluator_source': 'value_model',
              'expected_model_sha256': installed, 'root': root, 'nodes': nodes,
              'options': {'max_depth': 1, 'exploration_probability': 0, 'cache_values': False, 'reuse_tree': False}}
        t0 = time.monotonic()
        status, plan = c.raw('POST', f'/api/v1/domains/{domain}/planning/adversarial', {'session_id': 'ignored', **pb})
        out['plan_depth1'] = {'status': status, 'seconds': round(time.monotonic() - t0, 2), 'shape': shape(plan, max_depth=6)}
        c.save('plan-depth1', {'request_options': pb['options'], 'response': plan})
        run.log('plan depth 1', status, json.dumps(plan)[:800])
        # the same leaves through predict, for a first parity look
        leaf_inputs = [{k: v for k, v in n.items() if k not in ('id', 'maximizing', 'terminal_value', 'children')}
                       for n in nodes[1:]]
        lv = c.predict(domain, leaf_inputs, 'installed', installed)
        out['plan_depth1_leaf_predictions'] = lv
        run.log('leaf predictions', lv)

        # throughput: 200 predict calls of 125 L50 inputs, 2 in flight
        n125 = pbody(125)
        nbytes = len(compact(n125).encode())
        lat = []
        errs = []

        def one(i):
            t = time.monotonic()
            try:
                d = c.read('POST', c._v(domain, 'predict'), n125)
                assert len(d['predictions']) == 125
                return time.monotonic() - t
            except Exception as e:  # noqa: BLE001 - recorded, not fatal for a throughput probe
                errs.append(repr(e)[:200])
                return None
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as ex:
            lat = [x for x in ex.map(one, range(200)) if x is not None]
        wall = time.monotonic() - t0
        lat.sort()
        out['throughput'] = {'calls': 200, 'ok': len(lat), 'errors': errs[:10], 'n_errors': len(errs), 'body_kib': round(nbytes / 1024),
                             'wall_seconds': round(wall, 1), 'calls_per_second': round(len(lat) / wall, 3),
                             'inputs_per_second': round(125 * len(lat) / wall, 1),
                             'latency_p50': round(lat[len(lat) // 2], 3) if lat else None,
                             'latency_p90': round(lat[int(.9 * len(lat))], 3) if lat else None,
                             'retries': c.counts['retries']}
        run.log('throughput', out['throughput'])
        out['state_final_shape'] = shape(c.state(domain), max_depth=3)
    finally:
        try:
            c.delete_domain(domain)
            out['deleted_verified_404'] = True
            run.log('deleted', domain)
        except Exception as e:  # noqa: BLE001
            out['delete_error'] = repr(e)
            run.log('DELETE FAILED', repr(e))
        out['version_end'] = c.check_build('end')
        out['requests'] = c.counts
        run.write_results(out)
        run.finish(requests=c.counts['requests'])
        run.log('done; requests', c.counts['requests'])


if __name__ == '__main__':
    main()
