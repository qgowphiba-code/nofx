"""Machina contextual refinement on a ctxtasks task, optionally with an unmeasured drift switched on mid-run.

usage: refine2.py TASK ARM UPDATES RADIUS [DRIFT_AT]
Same protocol as refine_machina.py (8 candidates x 32 training batches per update, frozen held-out evaluation on 32
unseen batches). With DRIFT_AT, the plant changes after that many updates (pr2drift: the feed pump delivers 0.7x the
commanded rate, not measured); held-out evaluation is repeated right after the change, before any new update.
"""
import sys, json, time, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'common'))
import ctxtasks
from client import Client, Ambiguous, new_id
TASK, ARM, UPD, RADIUS = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
DRIFT_AT = int(sys.argv[5]) if len(sys.argv) > 5 else None
T = ctxtasks.Task(TASK); CANDS, BATCH = 8, 32
EVAL_AT = {0, 5, 10, 20, 30, 40, UPD} | ({DRIFT_AT + 5, DRIFT_AT + 10, DRIFT_AT + 20} if DRIFT_AT else set())
tag = f'{TASK}-{ARM}-r{RADIUS:g}' + (f'-drift{DRIFT_AT}' if DRIFT_AT else '')
run = HERE.parent / 'runs' / 'dyn' / f'refine2-{tag}'; run.mkdir(parents=True, exist_ok=True)
api = Client(run / 'requests.jsonl')
log = open(run / 'log.txt', 'a', buffering=1)
def say(*a):
    m = ' '.join(str(x) for x in a); print(m, flush=True); log.write(m + '\n')
cfg = {'mechanism': 'contextual_execution', 'horizon': ctxtasks.H, 'action_dimensions': T.adim, 'contexts': [T.vec(c) for c in T.SUPPORT],
       'objective_names': ['quality'], 'candidate_count': CANDS, 'max_contexts': BATCH, 'seed': 18, 'radius': RADIUS,
       'retained_variance': 0.99, 'contextual': ARM != 'shared', 'action_trust_region': True, 'normalize_context': True}
if ARM != 'shared': cfg['context_basis'] = ARM
did = new_id(f'sci-r2-{TASK}-{ARM}')
api.post('/domains', {'domain_id': did, 'description': f'contextual refinement {tag}'})
assert api.traj(did, 'state').get('status') == 'not_configured'
conf = api.traj(did, 'configure', {'config': cfg})
(run / 'ledger.json').write_text(json.dumps({'domain_id': did, 'config_sent': cfg, 'configure_response': conf, 'base_source': T.base_src,
                                             'base_nominal_J': T.baseJ, 'base_u': T.base}, indent=1))
drift = False
def execute(offsets, c):
    if T.adim == 1:
        return T.J([max(-1.0, min(1.0, b + o[0])) for b, o in zip(T.base, offsets)], c, drift)
    return T.J([[max(-1.0, min(1.0, b + o)) for b, o in zip(br, orow)] for br, orow in zip(T.base, offsets)], c, drift)
def heldout(k, label=''):
    vals = [execute(api.traj(did, 'evaluate', {'context': T.vec(c)})['offsets'], c) for c in T.HELD]
    basev = [T.J(T.base, c, drift) for c in T.HELD]
    rec = {'update': k, 'drift': drift, 'label': label, 'heldout_mean_J': sum(vals) / len(vals), 'base_mean_J': sum(basev) / len(basev), 'per_context': vals}
    with open(run / 'heldout.jsonl', 'a') as fh: fh.write(json.dumps(rec) + '\n')
    say(f'[{tag}] held-out after {k} updates{" (drift)" if drift else ""}{label}: mean J {rec["heldout_mean_J"]:.4g} (base alone {rec["base_mean_J"]:.4g})')
rng = random.Random(7); ids = {f'tr{j}': c for j, c in enumerate(T.TRAIN)}
heldout(0); version = conf.get('version', 0)
for k in range(1, UPD + 1):
    batch = rng.sample(sorted(ids), BATCH); disc = batch[::2]
    body = {'state': T.vec(ids[batch[0]]), 'goal': [0.0], 'mode': 'explore', 'request_id': f'{did}-u{k}', 'context_ids': batch, 'discovery_context_ids': disc}
    try:
        p = api.traj(did, 'propose', body)
    except Ambiguous:
        if api.traj(did, 'state').get('pending_decisions', 1) != 0: raise SystemExit(f'propose unknown with pending decision at update {k}')
        body['request_id'] += '-retry'; p = api.traj(did, 'propose', body)
    dec, nc = p['decision_id'], len(p['candidates']); outcomes = []
    for i in range(nc):
        for cid in batch:
            e = api.traj(did, 'evaluate', {'context': T.vec(ids[cid]), 'candidate_index': i, 'decision_id': dec})
            outcomes.append({'candidate_index': i, 'context_id': cid, 'objectives': [max(0.0, min(1.0, execute(e['offsets'], ids[cid]) / T.norm))]})
    try:
        o = api.traj(did, 'observe', {'decision_id': dec, 'outcomes': outcomes})
    except Ambiguous:
        s = api.traj(did, 'state'); o = {'version': s['version'], 'reconciled': True} if s.get('version') == version + 1 else api.traj(did, 'observe', {'decision_id': dec, 'outcomes': outcomes})
    version = o.get('version', version + 1); upd = o.get('update') or {}
    with open(run / 'updates.jsonl', 'a') as fh:
        fh.write(json.dumps({'update': k, 'drift': drift, 'selected': upd.get('selected'), 'confirmation_gain': upd.get('confirmation_gain'),
                             'radius': upd.get('action_search_radius'), 'next_radius': upd.get('next_action_search_radius'),
                             'mean_q': sum(x['objectives'][0] for x in outcomes) / len(outcomes)}) + '\n')
    if k in EVAL_AT: heldout(k)
    if DRIFT_AT and k == DRIFT_AT:
        drift = True; heldout(k, ' right after the change')
(run / 'summary.json').write_text(json.dumps({'domain_id': did, 'task': TASK, 'arm': ARM, 'radius': RADIUS, 'updates': UPD, 'drift_at': DRIFT_AT,
                                             'executions': UPD * CANDS * BATCH, 'api_calls': api.calls}, indent=1))
say('deleted', json.dumps(api.delete_domain(did))); api.close()
