"""Machina acquisition (episode_credit) on SMPL BeerFMT at the nominal initial conditions.

usage: beer_machina_acquire.py ARM STREAM ATTEMPTS OUT_ROOT
  ARM default   native defaults (terminal outcome only)
      profile   docs optional profile without progress-guided mutation (one whole-batch outcome)
Executor BEER-1: 20 stages of 10 h, piecewise-constant temperature; command u in [-1, 1] maps linearly to [9, 16] C.
Outcome = quality in [0, 1] = (SMPL return + 400) / 600. State rows = SMPL observation / box maximum at stage boundaries
for the executed prefix (the batch ends when SMPL declares it finished, timed out, or out of the observation box).
"""
import sys, json, math, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'common'))
import numpy as np
import beer_plant as B
from client import Client, Ambiguous, new_id

ARM, STREAM, N, ROOT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
H, SD = B.H, 8
run = ROOT / f'machina-beer-{ARM}-r{STREAM}'; run.mkdir(parents=True, exist_ok=True)
api = Client(run / 'requests.jsonl')
cfg = {'mechanism': 'episode_credit', 'state_dimensions': SD, 'action_dimensions': 1, 'goal_indices': [0],
       'horizon': H, 'capacity': 512, 'max_pending': 1, 'seed': 1, 'outcome_mode': 'reward'}
if ARM == 'profile':
    cfg.update({'coherent_edits': True, 'global_recall': True, 'executed_prefix_mutation': True, 'sequence_compaction': True,
                'outcome_tier_exploration': True, 'retain_prefix_failures': False})
did = new_id(f'bench-beer-{ARM}-r{STREAM}')
api.post('/domains', {'domain_id': did, 'description': f'SMPL BeerFMT batch temperature profile, Machina {ARM} (BEER-1)'})
assert api.traj(did, 'state').get('status') == 'not_configured'
conf = api.traj(did, 'configure', {'config': cfg})
(run / 'ledger.json').write_text(json.dumps({'domain_id': did, 'config_sent': cfg, 'configure_response': conf}, indent=1))
x0 = B.NOMINAL_X0
init = [float(v) for v in x0 / B.MAX_OBS]
GOAL = [1.0]


def execute(u):
    temps = B.profile_from_u(u)
    res = B.run(temps, x0, record=True)
    traj = res['traj']                                   # hourly observations, index 0 = initial
    n_stages = math.ceil(res['steps'] / B.STAGE_HOURS)   # stages at least partly executed
    states = [init] + [[float(v) for v in traj[min(len(traj) - 1, (k + 1) * B.STAGE_HOURS)] / B.MAX_OBS] for k in range(n_stages)]
    actions = [[float(u[k])] for k in range(n_stages)]
    return res, states, actions


best, best_u, version = -1.0, None, conf.get('version', 0)
out = open(run / 'attempts.jsonl', 'a'); t0 = time.time()
for k in range(N):
    bp = {'state': init, 'goal': GOAL, 'mode': 'explore', 'request_id': f'{did}-a{k}', 'seed': 1_000_000 * (STREAM - 1) + k}
    try:
        p = api.traj(did, 'propose', bp)
    except Ambiguous:
        if api.traj(did, 'state').get('pending_decisions', 1) != 0:
            raise SystemExit(f'propose unknown with pending decision at k={k}')
        bp['request_id'] += '-retry'; p = api.traj(did, 'propose', bp)
    acts = p['actions']
    assert 1 <= len(acts) <= H and all(len(r) == 1 and math.isfinite(r[0]) for r in acts)
    u = [max(-1.0, min(1.0, r[0])) for r in acts] + [0.0] * (H - len(acts))   # pad with neutral mid-range (12.5 C)
    res, states, actions = execute(u)
    q = B.quality(res['return'])
    body = {'decision_id': p['decision_id'], 'states': states, 'actions': actions, 'outcome': q}
    try:
        o = api.traj(did, 'observe', body)
    except Ambiguous:
        s = api.traj(did, 'state'); o = {'version': s['version']} if s.get('version') == version + 1 else api.traj(did, 'observe', body)
    version = o.get('version', version + 1)
    if q > best: best, best_u = q, u
    out.write(json.dumps({'k': k, 'proposed_len': len(acts), 'u': [round(x, 5) for x in u], 'return': res['return'], 'steps': res['steps'],
                          'reason': res['reason'], 'q': q, 'best': best, 'improved_parent': o.get('improved_parent'), 'version': version}) + '\n'); out.flush()
    if k % 50 == 0 or k == N - 1:
        print(f'beer {ARM} r{STREAM} k={k} return={res["return"]:.0f} q={q:.3f} best={best:.3f} calls={api.calls} {time.time() - t0:.0f}s', flush=True)
(run / 'final_state.json').write_text(json.dumps(api.traj(did, 'state'), indent=1))
(run / 'summary.json').write_text(json.dumps({'domain_id': did, 'arm': ARM, 'stream': STREAM, 'attempts': N, 'best_q': best, 'best_return': best * 600 - 400,
                                             'best_u': best_u, 'api_calls': api.calls, 'wall_s': round(time.time() - t0)}, indent=1))
print('deleted', json.dumps(api.delete_domain(did))); api.close()
