"""Machina acquisition on published dynamic-optimization benchmarks.

usage: run_machina.py PROBLEM ARM STREAM ATTEMPTS
  ARM default   native defaults (terminal outcome only)
      profile   docs optional profile WITHOUT progress-guided mutation (these benchmarks have no per-step outcome to measure)
Executor DYN-1: 20 equal stages, piecewise-constant control, command u in [-1, 1] mapped linearly onto the control bounds.
Outcome = J / J_published (clipped to [0, 1]). State rows = scaled ODE states at stage boundaries.
"""
import sys, json, math, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'common'))
import problems
from client import Client, Ambiguous, new_id
PB, ARM, STREAM, N = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); H = 20
P = problems.PROBLEMS[PB]
run = HERE.parent / 'runs' / 'dyn' / f'machina-{PB}-{ARM}-r{STREAM}'; run.mkdir(parents=True, exist_ok=True)
api = Client(run / 'requests.jsonl.xz')
ADIM = P.get('adim', 1)
cfg = {'mechanism': 'episode_credit', 'state_dimensions': P.get('state_dims') or len(P['x0']), 'action_dimensions': ADIM, 'goal_indices': P.get('goal_indices', [0]),
       'horizon': H, 'capacity': 512, 'max_pending': 1, 'seed': 1, 'outcome_mode': 'reward'}
if ARM == 'profile':
    cfg.update({'coherent_edits': True, 'global_recall': True, 'executed_prefix_mutation': True, 'sequence_compaction': True,
                'outcome_tier_exploration': True, 'retain_prefix_failures': False})
did = new_id(f'sci-dyn-{PB}-{ARM}-r{STREAM}')
api.post('/domains', {'domain_id': did, 'description': f'{PB} open-loop control benchmark, Machina {ARM} (DYN-1)'})
assert api.traj(did, 'state').get('status') == 'not_configured'
conf = api.traj(did, 'configure', {'config': cfg})
(run / 'ledger.json').write_text(json.dumps({'domain_id': did, 'config_sent': cfg, 'configure_response': conf}, indent=1))
init = problems.simulate(PB, [[0.0] * ADIM] * H if ADIM > 1 else [0.0] * H, H)['states'][0] if P.get('x0') is None else [a / s for a, s in zip(P['x0'], P['scale'])]
GOAL = P.get('goal', [1.0])
best, best_u, version = -1.0, None, conf.get('version', 0); out = open(run / 'attempts.jsonl', 'a'); t0 = time.time()
for k in range(N):
    bp = {'state': init, 'goal': GOAL, 'mode': 'explore', 'request_id': f'{did}-a{k}', 'seed': 1_000_000 * (STREAM - 1) + k}
    try:
        p = api.traj(did, 'propose', bp)
    except Ambiguous:
        if api.traj(did, 'state').get('pending_decisions', 1) != 0: raise SystemExit(f'propose unknown with pending decision at k={k}')
        bp['request_id'] += '-retry'; p = api.traj(did, 'propose', bp)
    acts = p['actions']; assert 1 <= len(acts) <= H and all(len(r) == ADIM and all(math.isfinite(v) for v in r) for r in acts)
    if ADIM == 1:
        u = [max(-1.0, min(1.0, r[0])) for r in acts] + [0.0] * (H - len(acts))
    else:
        u = [[max(-1.0, min(1.0, v)) for v in r] for r in acts] + [[0.0] * ADIM for _ in range(H - len(acts))]
    res = problems.simulate(PB, u, H)
    body = {'decision_id': p['decision_id'], 'states': res['states'], 'actions': [[x] for x in u] if ADIM == 1 else u, 'outcome': res['score']}
    body['states'][0] = init
    try:
        o = api.traj(did, 'observe', body)
    except Ambiguous:
        s = api.traj(did, 'state'); o = {'version': s['version']} if s.get('version') == version + 1 else api.traj(did, 'observe', body)
    version = o.get('version', version + 1)
    if res['J'] > best: best, best_u = res['J'], u
    out.write(json.dumps({'k': k, 'proposed_len': len(acts), 'u': u if ADIM > 1 else [round(x, 5) for x in u], 'J': res['J'], 'best': best,
                          'improved_parent': o.get('improved_parent'), 'version': version}) + '\n'); out.flush()
    if k % 100 == 0 or k == N - 1: print(f'{PB} {ARM} r{STREAM} k={k} J={res["J"]:.5f} best={best:.5f} calls={api.calls} {time.time() - t0:.0f}s', flush=True)
(run / 'final_state.json').write_text(json.dumps(api.traj(did, 'state'), indent=1))
(run / 'summary.json').write_text(json.dumps({'domain_id': did, 'problem': PB, 'arm': ARM, 'stream': STREAM, 'attempts': N, 'best_J': best,
                                             'best_u': best_u, 'api_calls': api.calls}, indent=1))
print('deleted', json.dumps(api.delete_domain(did))); api.close()
