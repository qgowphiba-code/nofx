"""Machina acquisition (episode_credit) on the ChemGymRL Wurtz reaction bench, one Domain per target material, on the
production API.

usage: chem_machina_acquire.py BENCH ARM TARGET ATTEMPTS OUT_ROOT [BASE_URL]      (BENCH wurtz | fict)
Executor CHEM-1: 20 steps x 6 controls (temperature change, volume change, four reagent additions), u in [-1, 1] -> [0, 1].
Outcome = quality (chem_plant.run). State rows = the bench's 7 thermodynamic/in-hand observation entries after each step.
"""
import sys, json, math, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'common'))
import numpy as np
import chem_plant as P
from client import Client, Ambiguous, new_id

BENCH, ARM, TARGET, N, ROOT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), Path(sys.argv[5])
BASE = sys.argv[6] if len(sys.argv) > 6 else 'https://rei-neuroadapt-api.reilabs.org/api/v1'
H, ADIM, SD = P.H, P.BENCHES[BENCH]['adim'], P.BENCHES[BENCH]['obs_state_dims']
tslug = TARGET.replace(',', '').replace(' ', '')
run = ROOT / f'machina-chem-{BENCH}-{ARM}-{tslug}'; run.mkdir(parents=True, exist_ok=True)
api = Client(run / 'requests.jsonl', base=BASE)
cfg = {'mechanism': 'episode_credit', 'state_dimensions': SD, 'action_dimensions': ADIM, 'goal_indices': [0],
       'horizon': H, 'capacity': 512, 'max_pending': 1, 'seed': 1, 'outcome_mode': 'reward'}
if ARM == 'profile':
    cfg.update({'coherent_edits': True, 'global_recall': True, 'executed_prefix_mutation': True, 'sequence_compaction': True,
                'outcome_tier_exploration': True, 'retain_prefix_failures': False})
did = new_id(f'bench-chem-{BENCH}-{ARM}-{tslug[:10]}')
api.create_domain({'domain_id': did, 'description': f'ChemGymRL {P.BENCHES[BENCH]["env_id"]} target {TARGET}, Machina {ARM} (CHEM-1)'})
assert api.traj(did, 'state').get('status') == 'not_configured'
conf = api.traj(did, 'configure', {'config': cfg})
(run / 'ledger.json').write_text(json.dumps({'domain_id': did, 'config_sent': cfg, 'configure_response': conf, 'bench': BENCH, 'target': TARGET, 'base_url': BASE}, indent=1))
env = P.make_env(BENCH)
P.set_target(env, TARGET); s0 = env.reset(); s0 = s0[0] if isinstance(s0, tuple) else s0
init = [float(v) for v in np.asarray(s0[:SD], dtype=float)]
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
    res = P.run(u, TARGET, env, record_states=True, bench=BENCH)
    states = [init] + res['states'][:res['steps']]
    body = {'decision_id': p['decision_id'], 'states': states, 'actions': u[:res['steps']], 'outcome': res['quality']}
    try:
        o = api.traj(did, 'observe', body)
    except Ambiguous:
        s = api.traj(did, 'state'); o = {'version': s['version']} if s.get('version') == version + 1 else api.traj(did, 'observe', body)
    version = o.get('version', version + 1)
    if res['quality'] > best: best, best_u = res['quality'], u
    out.write(json.dumps({'k': k, 'proposed_len': len(acts), 'u': [[round(x, 4) for x in r] for r in u], 'return': res['return'], 'q': res['quality'],
                          'best': best, 'improved_parent': o.get('improved_parent'), 'version': version}) + '\n'); out.flush()
    if k % 50 == 0 or k == N - 1:
        print(f'chem {BENCH} {ARM} {TARGET} k={k} return={res["return"]:.3f} best={best:.3f} calls={api.calls} {time.time() - t0:.0f}s', flush=True)
(run / 'final_state.json').write_text(json.dumps(api.traj(did, 'state'), indent=1))
(run / 'summary.json').write_text(json.dumps({'domain_id': did, 'bench': BENCH, 'arm': ARM, 'target': TARGET, 'attempts': N, 'best_q': best, 'best_u': best_u,
                                             'api_calls': api.calls, 'wall_s': round(time.time() - t0)}, indent=1))
print('deleted', json.dumps(api.delete_domain(did))); api.close()
