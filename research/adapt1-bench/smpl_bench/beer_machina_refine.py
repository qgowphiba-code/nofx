"""Machina contextual refinement (contextual_execution) on SMPL BeerFMT with SMPL's own batch-to-batch variation.

usage: beer_machina_refine.py ARM UPDATES RADIUS BASE_JSON OUT_ROOT
  ARM shared (contextual false) | linear (context_basis linear), with normalize_context and action_trust_region
Base = the best acquired profile (BASE_JSON: {"u": [...20], "source": ...}). Executor: u = clip(base + offset, -1, 1) -> BEER-1.
Objective quality = (SMPL return + 400) / 600 in [0, 1]. Per update: 8 candidates x 32 training contexts (16 discovery, 16
confirmation), one evaluate call per pair, then one complete observe. Frozen held-out evaluation (32 unseen contexts,
evaluate without candidate) at updates 0, 5, 10, 20, 30.
"""
import sys, json, time, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'common'))
import numpy as np
import beer_plant as B
import beer_contexts as C
from client import Client, Ambiguous, new_id

ARM, UPD, RADIUS, BASE_JSON, ROOT = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
CANDS, BATCH, EVAL_AT = 8, 32, {0, 5, 10, 20, 30, UPD}
base_rec = json.load(open(BASE_JSON)); base = [float(v) for v in base_rec['u']]
tag = f'{ARM}-r{RADIUS:g}'
run = ROOT / f'refine-beer-{tag}'; run.mkdir(parents=True, exist_ok=True)
api = Client(run / 'requests.jsonl')
log = open(run / 'log.txt', 'a', buffering=1)


def say(*a):
    m = ' '.join(str(x) for x in a); print(m, flush=True); log.write(m + '\n')


cfg = {'mechanism': 'contextual_execution', 'horizon': B.H, 'action_dimensions': 1, 'contexts': [C.vec(c) for c in C.TRAIN],
       'objective_names': ['quality'], 'candidate_count': CANDS, 'max_contexts': BATCH, 'seed': 18, 'radius': RADIUS,
       'retained_variance': 0.99, 'contextual': ARM != 'shared', 'action_trust_region': True, 'normalize_context': True}
if ARM != 'shared':
    cfg['context_basis'] = ARM
did = new_id(f'bench-beer-ref-{ARM}')
api.post('/domains', {'domain_id': did, 'description': f'SMPL BeerFMT contextual refinement {tag}'})
assert api.traj(did, 'state').get('status') == 'not_configured'
conf = api.traj(did, 'configure', {'config': cfg})
(run / 'ledger.json').write_text(json.dumps({'domain_id': did, 'config_sent': cfg, 'configure_response': conf, 'base': base_rec}, indent=1))


def execute(offsets, x0):
    u = [max(-1.0, min(1.0, b + o[0])) for b, o in zip(base, offsets)]
    return B.run(B.profile_from_u(u), x0)['return']


def heldout(k):
    vals = []
    for x0 in C.HELD:
        e = api.traj(did, 'evaluate', {'context': C.vec(x0)})
        vals.append(execute(e['offsets'], x0))
    fp = api.traj(did, 'state').get('fingerprint')
    base_vals = [B.run(B.profile_from_u(base), x0)['return'] for x0 in C.HELD]
    rec = {'update': k, 'heldout_mean_return': sum(vals) / len(vals), 'heldout_mean_quality': sum(B.quality(v) for v in vals) / len(vals),
           'finished_frac': sum(v > 0 for v in vals) / len(vals), 'per_context': vals, 'base_per_context': base_vals, 'fingerprint': fp}
    with open(run / 'heldout.jsonl', 'a') as fh:
        fh.write(json.dumps(rec) + '\n')
    say(f'held-out after {k} updates: mean return {rec["heldout_mean_return"]:.1f} quality {rec["heldout_mean_quality"]:.3f} finished {rec["finished_frac"]:.2f} (base alone {sum(base_vals) / len(base_vals):.1f})')


rng = random.Random(7)
ids = {f'tr{j}': x0 for j, x0 in enumerate(C.TRAIN)}
heldout(0)
version = conf.get('version', 0)
t0 = time.time()
for k in range(1, UPD + 1):
    batch = rng.sample(sorted(ids), BATCH); disc = batch[::2]
    body = {'state': C.vec(ids[batch[0]]), 'goal': [0.0], 'mode': 'explore', 'request_id': f'{did}-u{k}', 'context_ids': batch, 'discovery_context_ids': disc}
    try:
        p = api.traj(did, 'propose', body)
    except Ambiguous:
        if api.traj(did, 'state').get('pending_decisions', 1) != 0:
            raise SystemExit(f'propose unknown with pending decision at update {k}')
        body['request_id'] += '-retry'; p = api.traj(did, 'propose', body)
    dec, nc = p['decision_id'], len(p['candidates'])
    outcomes = []
    for i in range(nc):
        for cid in batch:
            e = api.traj(did, 'evaluate', {'context': C.vec(ids[cid]), 'candidate_index': i, 'decision_id': dec})
            q = B.quality(execute(e['offsets'], ids[cid]))
            outcomes.append({'candidate_index': i, 'context_id': cid, 'objectives': [q]})
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
                             'next_radius': o.get('next_radius', upd.get('next_radius')),
                             'mean_q': sum(x['objectives'][0] for x in outcomes) / len(outcomes),
                             'raw': {kk: v for kk, v in o.items() if kk not in ('decision_id',)}}) + '\n')
    say(f'update {k}: selected {upd.get("selected")} gain {upd.get("confirmation_gain")} calls {api.calls} {time.time() - t0:.0f}s')
    if k in EVAL_AT:
        heldout(k)
(run / 'summary.json').write_text(json.dumps({'domain_id': did, 'arm': ARM, 'radius': RADIUS, 'updates': UPD, 'executions': UPD * CANDS * BATCH,
                                             'api_calls': api.calls, 'wall_s': round(time.time() - t0)}, indent=1))
say('deleted', json.dumps(api.delete_domain(did))); api.close()
