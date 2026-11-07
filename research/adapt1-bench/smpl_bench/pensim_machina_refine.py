"""Machina contextual refinement (contextual_execution) on SMPL PenSimEnv with PenSimPy's own batch-to-batch variation.

usage: pensim_machina_refine.py ARM UPDATES RADIUS BASE_JSON OUT_ROOT
  ARM shared | linear. Base = best acquired recipe (BASE_JSON: {"u": [[..6]] x 23}). Executor PEN-2 on u = clip(base + offset).
Per update: 8 candidates x 12 training contexts (6 discovery, 6 confirmation) = 96 batches, simulated with 3 worker processes;
frozen held-out evaluation (16 unseen batches) at updates 0, 5, 10, 20. Objective = quality (pensim_plant.run).
"""
import sys, json, time, random
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'common'))
import numpy as np
import pensim_plant as P
import pensim_contexts as C
from client import Client, Ambiguous, new_id

ARM, UPD, RADIUS, BASE_JSON, ROOT = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
CANDS, BATCH, EVAL_AT, WORKERS = 8, 12, {0, 5, 10, 20, UPD}, 3
base_rec = json.load(open(BASE_JSON)); base = np.array(base_rec['u'], dtype=float)
tag = f'{ARM}-r{RADIUS:g}'
run = ROOT / f'refine-pensim-{tag}'; run.mkdir(parents=True, exist_ok=True)
log = open(run / 'log.txt', 'a', buffering=1)


def say(*a):
    m = ' '.join(str(x) for x in a); print(m, flush=True); log.write(m + '\n')


def sim(args):
    offsets, seed_ref = args
    u = np.clip(base + np.array(offsets, dtype=float), -1, 1)
    r = P.run(u, seed_ref)
    return r['quality'], r['yield'], r['steps']


if __name__ == '__main__':
    api = Client(run / 'requests.jsonl')
    for s in C.TRAIN + C.HELD:
        C.vec(s)
    cfg = {'mechanism': 'contextual_execution', 'horizon': P.H, 'action_dimensions': P.ADIM, 'contexts': [C.vec(s) for s in C.TRAIN],
           'objective_names': ['quality'], 'candidate_count': CANDS, 'max_contexts': BATCH, 'seed': 18, 'radius': RADIUS,
           'retained_variance': 0.99, 'contextual': ARM != 'shared', 'action_trust_region': True, 'normalize_context': True}
    if ARM != 'shared':
        cfg['context_basis'] = ARM
    did = new_id(f'bench-pensim-ref-{ARM}')
    api.post('/domains', {'domain_id': did, 'description': f'SMPL PenSim contextual refinement {tag}'})
    assert api.traj(did, 'state').get('status') == 'not_configured'
    conf = api.traj(did, 'configure', {'config': cfg})
    (run / 'ledger.json').write_text(json.dumps({'domain_id': did, 'config_sent': cfg, 'configure_response': conf, 'base': base_rec}, indent=1))
    pool = Pool(WORKERS)

    def heldout(k):
        offs = [api.traj(did, 'evaluate', {'context': C.vec(s)})['offsets'] for s in C.HELD]
        res = pool.map(sim, list(zip(offs, C.HELD)))
        fp = api.traj(did, 'state').get('fingerprint')
        rec = {'update': k, 'heldout_mean_quality': sum(r[0] for r in res) / len(res), 'heldout_mean_yield': sum(r[1] for r in res) / len(res),
               'completed_frac': sum(r[2] >= 1150 for r in res) / len(res), 'per_context': [r[1] for r in res], 'per_context_q': [r[0] for r in res], 'fingerprint': fp}
        with open(run / 'heldout.jsonl', 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        say(f'held-out after {k} updates: mean yield {rec["heldout_mean_yield"]:.0f} quality {rec["heldout_mean_quality"]:.3f} completed {rec["completed_frac"]:.2f}')

    rng = random.Random(7)
    ids = {f'tr{j}': s for j, s in enumerate(C.TRAIN)}
    heldout(0)
    version = conf.get('version', 0); t0 = time.time()
    for k in range(1, UPD + 1):
        batch = rng.sample(sorted(ids), BATCH); disc = batch[::2]
        body = {'state': C.vec(ids[batch[0]]), 'goal': [0.0] * P.ADIM, 'mode': 'explore', 'request_id': f'{did}-u{k}', 'context_ids': batch, 'discovery_context_ids': disc}
        try:
            p = api.traj(did, 'propose', body)
        except Ambiguous:
            if api.traj(did, 'state').get('pending_decisions', 1) != 0:
                raise SystemExit(f'propose unknown with pending decision at update {k}')
            body['request_id'] += '-retry'; p = api.traj(did, 'propose', body)
        dec, nc = p['decision_id'], len(p['candidates'])
        jobs, keys = [], []
        for i in range(nc):
            for cid in batch:
                e = api.traj(did, 'evaluate', {'context': C.vec(ids[cid]), 'candidate_index': i, 'decision_id': dec})
                jobs.append((e['offsets'], ids[cid])); keys.append((i, cid))
        res = pool.map(sim, jobs)
        outcomes = [{'candidate_index': i, 'context_id': cid, 'objectives': [r[0]]} for (i, cid), r in zip(keys, res)]
        try:
            o = api.traj(did, 'observe', {'decision_id': dec, 'outcomes': outcomes})
        except Ambiguous:
            s = api.traj(did, 'state')
            o = {'version': s['version'], 'reconciled': True} if s.get('version') == version + 1 else api.traj(did, 'observe', {'decision_id': dec, 'outcomes': outcomes})
        version = o.get('version', version + 1)
        upd = o.get('update') or {}
        with open(run / 'updates.jsonl', 'a') as fh:
            fh.write(json.dumps({'update': k, 'candidates': nc, 'nominee': upd.get('nominee'), 'selected': upd.get('selected'),
                                 'confirmation_gain': upd.get('confirmation_gain'), 'radius': o.get('radius', upd.get('radius')),
                                 'next_radius': o.get('next_radius', upd.get('next_radius')), 'mean_q': sum(x['objectives'][0] for x in outcomes) / len(outcomes),
                                 'mean_yield': sum(r[1] for r in res) / len(res), 'raw': {kk: v for kk, v in o.items() if kk != 'decision_id'}}) + '\n')
        say(f'update {k}: selected {upd.get("selected")} gain {upd.get("confirmation_gain")} mean yield {sum(r[1] for r in res) / len(res):.0f} calls {api.calls} {time.time() - t0:.0f}s')
        if k in EVAL_AT:
            heldout(k)
    (run / 'summary.json').write_text(json.dumps({'domain_id': did, 'arm': ARM, 'radius': RADIUS, 'updates': UPD, 'executions': UPD * CANDS * BATCH,
                                                 'api_calls': api.calls, 'wall_s': round(time.time() - t0)}, indent=1))
    say('deleted', json.dumps(api.delete_domain(did))); api.close(); pool.close()
