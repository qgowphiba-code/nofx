"""Machina acquisition (episode_credit) on batch cooling crystallization, one compound at the nominal context.

usage: run_machina.py COMPOUND ARM STREAM ATTEMPTS
  ARM     default  native defaults, terminal outcome only
          profile  the docs' optional acquisition profile + measured per-interval fines signal (step_outcomes)
  STREAM  1, 2, ... proposal seed stream (seed = 1e6 * (STREAM - 1) + attempt)

Executor contract CRYST-1 (fixed): one command per 7.5-min interval; u clipped to [-1, 1]; interval cooling
dT_k = (T0 - Tf) * exp(3.5 u_k) / sum_j exp(3.5 u_j). A proposal shorter than 16 commands is padded with neutral
u = 0 commands, which are executed and reported. Outcome = quality (seed-grown mass / theoretical yield).
State row = [time fraction, (T - Tf)/(T0 - Tf), supersaturation / 0.2, crystallized fraction]; goal index 3 -> 1.0.
"""
import sys, json, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import sim
from bank import compound, NOMINAL
from client import Client, Ambiguous, new_id

COMP, ARM, STREAM, N = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
assert ARM in ('default', 'profile')
comp = compound(COMP)
run = Path(__file__).resolve().parents[1] / 'runs' / 'cryst' / f'machina-c{COMP}-{ARM}-r{STREAM}'
run.mkdir(parents=True, exist_ok=True)
api = Client(run / 'requests.jsonl.xz')
log = open(run / 'log.txt', 'a')
def say(*a):
    m = ' '.join(str(x) for x in a); print(m, flush=True); log.write(m + '\n'); log.flush()

cfg = {'mechanism': 'episode_credit', 'state_dimensions': 4, 'action_dimensions': 1, 'goal_indices': [3],
       'horizon': sim.H, 'capacity': 512, 'max_pending': 1, 'seed': 1, 'outcome_mode': 'reward'}
if ARM == 'profile':
    cfg.update({'coherent_edits': True, 'global_recall': True, 'executed_prefix_mutation': True, 'progress_guided_mutation': True,
                'sequence_compaction': True, 'outcome_tier_exploration': True, 'retain_prefix_failures': False})
did = new_id(f'sci-cryst-c{COMP}-{ARM}-r{STREAM}')
api.post('/domains', {'domain_id': did, 'description': f'cooling crystallization compound {COMP} Machina {ARM} (CRYST-1)'})
st = api.traj(did, 'state'); assert st.get('status') == 'not_configured', st
conf = api.traj(did, 'configure', {'config': cfg})
(run / 'ledger.json').write_text(json.dumps({'domain_id': did, 'compound': COMP, 'arm': ARM, 'stream': STREAM, 'config_sent': cfg,
                                             'configure_response': conf}, indent=1))
say('domain', did, 'resolved', json.dumps(conf.get('config')))
init = [0.0, 1.0, 0.0, 0.0]
best, best_u, version = -1.0, None, conf.get('version', 0)
out = open(run / 'attempts.jsonl', 'a')
t0 = time.time()
for k in range(N):
    body_p = {'state': init, 'goal': [1.0], 'mode': 'explore', 'request_id': f'{did}-a{k}', 'seed': 1_000_000 * (STREAM - 1) + k}
    try:
        p = api.traj(did, 'propose', body_p)
    except Ambiguous:
        s0 = api.traj(did, 'state')
        if s0.get('pending_decisions', 1) != 0:
            raise SystemExit(f'propose outcome unknown with a pending decision at k={k}: stop and reconcile')
        body_p['request_id'] += '-retry'
        p = api.traj(did, 'propose', body_p)
    acts = p['actions']
    assert 1 <= len(acts) <= sim.H and all(len(r) == 1 and math.isfinite(r[0]) for r in acts), acts
    u = [max(-1.0, min(1.0, r[0])) for r in acts] + [0.0] * (sim.H - len(acts))
    res = sim.simulate(u, comp, NOMINAL)
    body = {'decision_id': p['decision_id'], 'states': res['states'], 'actions': [[x] for x in u], 'outcome': res['quality']}
    body['states'][0] = init
    if ARM == 'profile':
        body['step_outcomes'] = res['step_outcomes']
    try:
        o = api.traj(did, 'observe', body)
    except Ambiguous:
        s = api.traj(did, 'state')
        if s.get('version') == version + 1:
            o = {'status': 'reconciled_applied', 'version': s['version']}
        else:
            o = api.traj(did, 'observe', body)
    version = o.get('version', version + 1)
    if res['quality'] > best:
        best, best_u = res['quality'], u
    out.write(json.dumps({'k': k, 'decision_id': p['decision_id'], 'proposed_len': len(acts), 'u': [round(x, 5) for x in u],
                          'quality': res['quality'], 'fines_frac': res['fines_frac'], 'left': res['left_dissolved'], 'best': best,
                          'improved_parent': o.get('improved_parent'), 'retained': o.get('retained_trajectories'), 'version': version,
                          'parent': p.get('parent_id', p.get('parent')), 'source': p.get('source')}) + '\n'); out.flush()
    if k % 50 == 0 or k == N - 1:
        say(f'k={k} q={res["quality"]:.4f} best={best:.4f} calls={api.calls} {time.time() - t0:.0f}s')
final_state = api.traj(did, 'state')
records = api.traj(did, 'records', {'limit': 64, 'offset': 0, 'best_only': True})
(run / 'final_state.json').write_text(json.dumps(final_state, indent=1))
(run / 'final_records.json').write_text(json.dumps(records, indent=1))
(run / 'summary.json').write_text(json.dumps({'domain_id': did, 'compound': COMP, 'arm': ARM, 'stream': STREAM, 'attempts': N,
                                              'best': best, 'best_u': best_u, 'api_calls': api.calls, 'api_seconds': round(api.seconds, 1)}, indent=1))
d = api.delete_domain(did)
say('deleted', json.dumps(d))
api.close()
