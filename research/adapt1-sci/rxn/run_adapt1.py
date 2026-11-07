"""Adapt-1 contextual policy route on experiment-level reaction optimization.

usage: run_adapt1.py ARM EPISODES BUDGET [EXPLORATION]
  ARM ctx    context = three measured substrate descriptors
      noctx  context = a constant (no substrate information); tests adaptation from feedback alone
One query = one experiment. Conditions already run in this episode are ineligible: if Adapt-1 selects one, the
runner executes Adapt-1's highest-ranked untested condition and records the override (client-side eligibility,
as the docs require). Feedback goes to the condition actually executed. Reward = measured yield (2 % noise),
fixed per (reaction, condition) so every method sees the same measurement for the same experiment.
"""
import sys, json, time, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'common'))
import sim, decl
from client import Client, new_id, Ambiguous

ARM, NEP, BUDGET = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
EXPL = sys.argv[4] if len(sys.argv) > 4 else 'thompson'
TAG = sys.argv[5] if len(sys.argv) > 5 else ''
POL = [decl.policy_id(k) for k in range(len(sim.CONDITIONS))]
IDX = {p: k for k, p in enumerate(POL)}

def measured(rx, k):
    return sim.measure(rx, k, random.Random(rx['id'] * 1000 + k))

run = HERE.parent / 'runs' / 'rxn' / f'adapt1-{ARM}-{EXPL}{TAG}'
run.mkdir(parents=True, exist_ok=True)
api = Client(run / 'requests.jsonl.xz')
import os
RESUME = os.environ.get('RESUME_DOMAIN')
if RESUME:   # continue a run whose last feedback write failed and was reconciled as not applied
    did = RESUME
    prev = [json.loads(l) for l in open(run / 'trace.jsonl')]
    start_ep = max(r['episode'] for r in prev)
    done_eps = {json.loads(l)['episode'] for l in open(run / 'episodes.jsonl')}
    assert start_ep not in done_eps
    trace = open(run / 'trace.jsonl', 'a'); eps = open(run / 'episodes.jsonl', 'a')
    overrides = sum(r['override'] is not None for r in prev); gaps = sum(r['feedback_applied'] is not True for r in prev)
    partial = [r for r in prev if r['episode'] == start_ep]
    applied = api.call('GET', f'/domains/{did}', None, mutation=False)['event_count']
    assert applied == len(prev), (applied, len(prev))
    with open(run / 'RESUMED.txt', 'a') as fh:
        fh.write(json.dumps({'resumed_at_episode': start_ep, 'step': len(partial), 'event_count': applied, 'trace_rows': len(prev)}) + '\n')
else:
    did = new_id(f'sci-rxn-{ARM}-{EXPL}')
    d = decl.declaration(did, context=(ARM == 'ctx'), exploration=EXPL)
    echo = api.post('/domains', d)
    (run / 'declaration.json').write_text(json.dumps(d, indent=1)); (run / 'server_declaration.json').write_text(json.dumps(echo, indent=1))
    trace = open(run / 'trace.jsonl', 'w'); eps = open(run / 'episodes.jsonl', 'w')
    start_ep, partial, overrides, gaps = 0, [], 0, 0
applied_count = api.call('GET', f'/domains/{did}', None, mutation=False).get('event_count') or 0
t0 = time.time()
for e in range(start_ep, NEP):
    rx = sim.make_reaction(e); g = sim.grid(rx); gmax = max(g)
    ctx = {'d1': round(rx['descriptors'][0], 4), 'd2': round(rx['descriptors'][1], 4), 'd3': round(rx['descriptors'][2], 4)} if ARM == 'ctx' else {'bias': 1.0}
    tested, best_true, curve = set(), 0.0, []
    first = 0
    if e == start_ep and partial:
        for r0 in partial:
            k0 = IDX[r0['executed']]; tested.add(k0); best_true = max(best_true, g[k0]); curve.append(best_true / gmax)
        first = len(partial)
    for step in range(first, BUDGET):
        q = api.post(f'/domains/{did}/query', {'session_id': 'ignored', 'question': 'Choose the next experiment condition.',
                     'relation': decl.RELATION, 'context': {'event_type': 'state', 'values': ctx},
                     'allow_exploration': True, 'top_k': len(POL)}, mutation=True)
        sel = q.get('selection') or {}
        chosen = sel.get('selected_policy')
        ranked = q.get('ranked_hypotheses') or []
        execd, override = chosen, None
        if chosen not in IDX or IDX[chosen] in tested:
            override = 'already_tested' if chosen in IDX else 'abstained_or_unknown'
            execd = next((h['policy'] for h in ranked if h.get('policy') in IDX and IDX[h['policy']] not in tested), None)
            if execd is None:
                execd = next(p for p in POL if IDX[p] not in tested)
            overrides += 1
        k = IDX[execd]; tested.add(k)
        y = measured(rx, k)
        fb = {'session_id': 'ignored', 'outcome': 'success', 'feedback_kind': 'execution', 'relation': decl.RELATION, 'policy': execd,
              'values': {'yield': y}, 'metadata': {'episode_id': f'rx{e}', 'step': step}}
        if q.get('decision_id'):
            fb['decision_id'] = q['decision_id']
        try:
            r = api.post(f'/domains/{did}/feedback', fb)
        except Ambiguous:
            n = api.call('GET', f'/domains/{did}', None, mutation=False).get('event_count')
            r = {'credit_assignment': {'contextual_learning_applied': True, 'reconciled': True}} if n == applied_count + 1 else api.post(f'/domains/{did}/feedback', fb)
        applied_count += 1
        applied = (r.get('credit_assignment') or {}).get('contextual_learning_applied')
        gaps += applied is not True
        best_true = max(best_true, g[k]); curve.append(best_true / gmax)
        hx = next((h for h in ranked if h.get('policy') == execd), {})
        cp = hx.get('contextual_policy') or {}
        lb = q.get('latent_belief') or {}
        trace.write(json.dumps({'episode': e, 'family': rx['family'], 'step': step, 'selected': chosen, 'executed': execd, 'override': override,
                                'selected_by': sel.get('selected_by'), 'mode': sel.get('mode'), 'warning': sel.get('warning'),
                                'measured': y, 'true': g[k], 'grid_max': gmax, 'rank_of_true_best': None,
                                'exp_reward_executed': cp.get('expected_reward'), 'eff_obs_executed': cp.get('effective_observations'),
                                'active_regime': cp.get('active_regime_id'), 'latent': {kk: lb.get(kk) for kk in ('active_regime_id', 'regime_count', 'status') if kk in lb},
                                'feedback_applied': applied}) + '\n')
    trace.flush()
    rec = {'episode': e, 'family': rx['family'], 'grid_max': gmax, 'curve': curve, 'api_calls': api.calls, 'elapsed': round(time.time() - t0)}
    eps.write(json.dumps(rec) + '\n'); eps.flush()
    print(json.dumps({'episode': e, 'fam': rx['family'], 'best@1': round(curve[0], 3), 'best@5': round(curve[min(4, BUDGET - 1)], 3),
                      'best@end': round(curve[-1], 3), 'overrides': overrides, 'gaps': gaps, 'calls': api.calls, 's': round(time.time() - t0)}), flush=True)
state = {}
try:
    state = api.call('GET', f'/domains/{did}', None, mutation=False)
except Exception as ex:
    state = {'error': repr(ex)}
(run / 'final_domain.json').write_text(json.dumps(state, indent=1)[:5_000_000])
(run / 'summary.json').write_text(json.dumps({'domain_id': did, 'arm': ARM, 'exploration': EXPL, 'episodes': NEP, 'budget': BUDGET,
                                             'overrides': overrides, 'feedback_gaps': gaps, 'api_calls': api.calls, 'api_seconds': round(api.seconds, 1)}, indent=1))
print('deleted', json.dumps(api.delete_domain(did)))
api.close()
