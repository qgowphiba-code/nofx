"""Machina acquisition (episode_credit) on SMPL PenSimEnv (IndPenSim) at a fixed nominal batch (seed_ref 1000).

usage: pensim_machina_acquire.py ARM ATTEMPTS OUT_ROOT [SEED_REF]
Executor PEN-1 (pensim_plant.py): 23 stages of 10 h, 6 piecewise-constant controls, u in [-1, 1]^(23 x 6) mapped onto
SMPL's action bounds. Outcome = quality = batch yield / 4500 (failed batches score 0). State rows = SMPL observation / box
maximum at stage boundaries for the executed prefix.
"""
import sys, json, math, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'common'))
import numpy as np
import pensim_plant as P
from client import Client, Ambiguous, new_id

ARM, N, ROOT = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
SEED_REF = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
H, ADIM, SD = P.H, P.ADIM, 9
run = ROOT / f'machina-pensim-{ARM}-r1'; run.mkdir(parents=True, exist_ok=True)
api = Client(run / 'requests.jsonl')
cfg = {'mechanism': 'episode_credit', 'state_dimensions': SD, 'action_dimensions': ADIM, 'goal_indices': [0],
       'horizon': H, 'capacity': 512, 'max_pending': 1, 'seed': 1, 'outcome_mode': 'reward'}
if ARM == 'profile':
    cfg.update({'coherent_edits': True, 'global_recall': True, 'executed_prefix_mutation': True, 'sequence_compaction': True,
                'outcome_tier_exploration': True, 'retain_prefix_failures': False})
did = new_id(f'bench-pensim-{ARM}')
api.post('/domains', {'domain_id': did, 'description': f'SMPL PenSimEnv 6-control batch recipe, Machina {ARM} (PEN-1)'})
assert api.traj(did, 'state').get('status') == 'not_configured'
conf = api.traj(did, 'configure', {'config': cfg})
(run / 'ledger.json').write_text(json.dumps({'domain_id': did, 'config_sent': cfg, 'configure_response': conf, 'seed_ref': SEED_REF}, indent=1))
e0 = P.make_env(); e0.reset(random_seed_ref=SEED_REF)
init = [0.0] * SD                          # SMPL's reset observation is all zeros (no measurement before the batch starts)
GOAL = [1.0]
best, best_u, version = -1.0, None, conf.get('version', 0)
out = open(run / 'attempts.jsonl', 'a'); t0 = time.time()
for k in range(N):
    bp = {'state': init, 'goal': GOAL, 'mode': 'explore', 'request_id': f'{did}-a{k}', 'seed': k}
    try:
        p = api.traj(did, 'propose', bp)
    except Ambiguous:
        if api.traj(did, 'state').get('pending_decisions', 1) != 0:
            raise SystemExit(f'propose unknown with pending decision at k={k}')
        bp['request_id'] += '-retry'; p = api.traj(did, 'propose', bp)
    acts = p['actions']
    assert 1 <= len(acts) <= H and all(len(r) == ADIM and all(math.isfinite(v) for v in r) for r in acts)
    u = [[max(-1.0, min(1.0, v)) for v in r] for r in acts] + [[0.0] * ADIM for _ in range(H - len(acts))]
    res = P.run(u, SEED_REF, record_states=True)
    n_stages = max(1, len(res['states']))
    states = [init] + res['states'][:n_stages]
    actions = u[:n_stages]
    q = res['quality']
    body = {'decision_id': p['decision_id'], 'states': states, 'actions': actions, 'outcome': q}
    try:
        o = api.traj(did, 'observe', body)
    except Ambiguous:
        s = api.traj(did, 'state'); o = {'version': s['version']} if s.get('version') == version + 1 else api.traj(did, 'observe', body)
    version = o.get('version', version + 1)
    if q > best: best, best_u = q, u
    out.write(json.dumps({'k': k, 'proposed_len': len(acts), 'u': [[round(x, 4) for x in r] for r in u], 'yield': res['yield'], 'steps': res['steps'],
                          'error': res['error'], 'q': q, 'best': best, 'improved_parent': o.get('improved_parent'), 'version': version}) + '\n'); out.flush()
    if k % 10 == 0 or k == N - 1:
        print(f'pensim {ARM} k={k} yield={res["yield"]:.0f} steps={res["steps"]} q={q:.3f} best={best:.3f} calls={api.calls} {time.time() - t0:.0f}s', flush=True)
(run / 'final_state.json').write_text(json.dumps(api.traj(did, 'state'), indent=1))
(run / 'summary.json').write_text(json.dumps({'domain_id': did, 'arm': ARM, 'attempts': N, 'seed_ref': SEED_REF, 'best_q': best, 'best_yield': best * P.Y_SCALE,
                                             'best_u': best_u, 'api_calls': api.calls, 'wall_s': round(time.time() - t0)}, indent=1))
print('deleted', json.dumps(api.delete_domain(did))); api.close()
