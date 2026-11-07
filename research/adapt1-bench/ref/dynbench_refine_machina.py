"""Machina contextual refinement (contextual_execution) on Park-Ramirez with measured batch-to-batch variation.

usage: refine_machina.py ARM UPDATES RADIUS
  ARM shared (contextual false) | linear | quadratic (context_basis), all with normalize_context and action_trust_region
Base = Machina's best acquired profile at the published conditions (acquisition -> refinement; structural stage skipped,
which the docs allow). Executor: u = clip(base + offset, -1, 1). Objective quality = J / 32.68 (published optimum).
Per update: 8 candidates x 32 training contexts (16 discovery, 16 confirmation), one evaluate call per pair, then one
complete observe. Frozen held-out evaluation (32 unseen contexts, evaluate without candidate) at updates 0, 5, 10, 20, 30.
"""
import sys, json, time, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'common'))
import pr_ctx as P
from client import Client, Ambiguous, new_id
ARM, UPD, RADIUS = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
CANDS, BATCH, EVAL_AT = 8, 32, {0, 5, 10, 20, 30, UPD}
base, baseJ, base_src = P.machina_base()
tag = f'{ARM}-r{RADIUS:g}'
run = HERE.parent / 'runs' / 'dyn' / f'refine-parkram-{tag}'; run.mkdir(parents=True, exist_ok=True)
api = Client(run / 'requests.jsonl')
log = open(run / 'log.txt', 'a', buffering=1)
def say(*a):
    m = ' '.join(str(x) for x in a); print(m, flush=True); log.write(m + '\n')
cfg = {'mechanism': 'contextual_execution', 'horizon': P.H, 'action_dimensions': 1, 'contexts': [P.vec(c) for c in P.SUPPORT],
       'objective_names': ['quality'], 'candidate_count': CANDS, 'max_contexts': BATCH, 'seed': 18, 'radius': RADIUS,
       'retained_variance': 0.99, 'contextual': ARM != 'shared', 'action_trust_region': True, 'normalize_context': True}
if ARM != 'shared':
    cfg['context_basis'] = ARM
did = new_id(f'sci-ref-pr-{ARM}')
api.post('/domains', {'domain_id': did, 'description': f'Park-Ramirez contextual refinement {tag}'})
assert api.traj(did, 'state').get('status') == 'not_configured'
conf = api.traj(did, 'configure', {'config': cfg})
(run / 'ledger.json').write_text(json.dumps({'domain_id': did, 'config_sent': cfg, 'configure_response': conf, 'base_source': base_src,
                                             'base_nominal_J': baseJ, 'base_u': base}, indent=1))
def execute(offsets, c):
    u = [max(-1.0, min(1.0, b + o[0])) for b, o in zip(base, offsets)]
    return P.J(u, c['sf'], c['x40'])
def heldout(k):
    vals = []
    for c in P.HELD:
        e = api.traj(did, 'evaluate', {'context': P.vec(c)})
        vals.append(execute(e['offsets'], c))
    fp = api.traj(did, 'state').get('fingerprint')
    rec = {'update': k, 'heldout_mean_J': sum(vals) / len(vals), 'per_context': vals, 'fingerprint': fp}
    with open(run / 'heldout.jsonl', 'a') as fh: fh.write(json.dumps(rec) + '\n')
    say(f'held-out after {k} updates: mean J {rec["heldout_mean_J"]:.3f} (base alone {sum(P.J(base, c["sf"], c["x40"]) for c in P.HELD) / len(P.HELD):.3f})')
rng = random.Random(7)
ids = {f'tr{j}': c for j, c in enumerate(P.TRAIN)}
heldout(0)
version = conf.get('version', 0)
for k in range(1, UPD + 1):
    batch = rng.sample(sorted(ids), BATCH); disc = batch[::2]
    body = {'state': P.vec(ids[batch[0]]), 'goal': [0.0], 'mode': 'explore', 'request_id': f'{did}-u{k}', 'context_ids': batch, 'discovery_context_ids': disc}
    try:
        p = api.traj(did, 'propose', body)
    except Ambiguous:
        if api.traj(did, 'state').get('pending_decisions', 1) != 0: raise SystemExit(f'propose unknown with pending decision at update {k}')
        body['request_id'] += '-retry'; p = api.traj(did, 'propose', body)
    dec, nc = p['decision_id'], len(p['candidates'])
    outcomes = []
    for i in range(nc):
        for cid in batch:
            e = api.traj(did, 'evaluate', {'context': P.vec(ids[cid]), 'candidate_index': i, 'decision_id': dec})
            q = max(0.0, min(1.0, execute(e['offsets'], ids[cid]) / P.J_PUB))
            outcomes.append({'candidate_index': i, 'context_id': cid, 'objectives': [q]})
    try:
        o = api.traj(did, 'observe', {'decision_id': dec, 'outcomes': outcomes})
    except Ambiguous:
        s = api.traj(did, 'state'); o = {'version': s['version'], 'reconciled': True} if s.get('version') == version + 1 else api.traj(did, 'observe', {'decision_id': dec, 'outcomes': outcomes})
    version = o.get('version', version + 1)
    upd = o.get('update') or {}
    with open(run / 'updates.jsonl', 'a') as fh:
        fh.write(json.dumps({'update': k, 'candidates': nc, 'nominee': upd.get('nominee'), 'selected': upd.get('selected'),
                             'confirmation_gain': upd.get('confirmation_gain'), 'radius': o.get('radius', upd.get('radius')),
                             'next_radius': o.get('next_radius', upd.get('next_radius')), 'mean_q': sum(x['objectives'][0] for x in outcomes) / len(outcomes),
                             'raw': {kk: v for kk, v in o.items() if kk not in ('decision_id',)}}) + '\n')
    say(f'update {k}: selected {upd.get("selected")} gain {upd.get("confirmation_gain")} calls {api.calls}')
    if k in EVAL_AT: heldout(k)
(run / 'summary.json').write_text(json.dumps({'domain_id': did, 'arm': ARM, 'radius': RADIUS, 'updates': UPD, 'executions': UPD * CANDS * BATCH,
                                             'api_calls': api.calls}, indent=1))
say('deleted', json.dumps(api.delete_domain(did))); api.close()
