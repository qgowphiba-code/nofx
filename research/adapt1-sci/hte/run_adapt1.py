"""Adapt-1 contextual policy route on published HTE reaction datasets (one query = one experiment).

usage: run_adapt1.py DATASET ORDER BUDGET
  DATASET aryl_amination | suzuki;  ORDER = seed of the substrate order;  BUDGET = experiments per substrate
One Domain spans all substrates of the dataset in the given order (retained state across substrates is the point).
Conditions already run for this substrate, or absent from the dataset for it, are ineligible: if Adapt-1 selects one,
the runner executes Adapt-1's highest-ranked eligible condition and records the override.
"""
import sys, json, time, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'common'))
import data
from client import Client, new_id, Ambiguous
DS, ORDER, BUDGET = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
D = data.load(DS); REL = 'run_condition'
POL = {data.policy_id(c): c for c in D['conditions']}
subs = list(D['substrates']); random.Random(ORDER).shuffle(subs)

def slim(path, resp):
    if path.endswith('/query') and isinstance(resp, dict):
        rk = resp.get('ranked_hypotheses') or []
        return {'decision_id': resp.get('decision_id'), 'selection': resp.get('selection'), 'state_version': resp.get('state_version'),
                'latent_belief': resp.get('latent_belief'),
                'ranked_top10': [{'policy': h.get('policy'), 'selection_score': h.get('selection_score'),
                                  'expected_reward': (h.get('contextual_policy') or {}).get('expected_reward'),
                                  'eff_obs': (h.get('contextual_policy') or {}).get('effective_observations')} for h in rk[:10]]}
    return resp

run = HERE.parent / 'runs' / 'hte' / f'adapt1-{DS}-o{ORDER}'; run.mkdir(parents=True, exist_ok=True)
api = Client(run / 'requests.jsonl', slim=slim)
did = new_id(f'sci-hte-{DS[:5]}-o{ORDER}')
hyps = [{'name': 'run ' + p, 'relation': REL, 'policy': p, 'policy_features': dict(zip(D['cond_fields'], c)),
         'when': [], 'predicts': ['measured yield'], 'falsified_by': [], 'weight': 1.0} for p, c in POL.items()]
learning = {'enabled': True,
    'context': {'feature_paths': ['values.' + n for n in D['context_names']], 'event_types': ['state'], 'max_samples': 4096},
    'reward': {'aggregation': 'weighted_mean', 'components': [{'field': 'values.yield', 'goal': 'maximize', 'min': 0.0, 'max': 1.0, 'weight': 1.0, 'required': True}]},
    'policy': {'decay_half_life_seconds': 0.0, 'transfer_strength': 0.0, 'action_transfer_strength': 0.3, 'min_context_observations': 4,
               'model_max_weight': 0.45, 'model_ood_threshold': 0.75, 'ood_confidence_penalty': 0.5, 'exploration_mode': 'thompson',
               'exploration_strength': 1.0, 'minimum_expected_reward': 0.0, 'minimum_confidence': 0.0, 'require_credible_dominance': False,
               'dominance_margin': 0.0, 'abstain_on_ood': False},
    'credit_assignment': {'mode': 'none', 'neutral_reward': 0.5},
    'latent_belief': {'enabled': True, 'max_regimes': 8, 'novelty_threshold': 0.55, 'temperature': 0.25, 'change_point_threshold': 0.55,
                      'change_point_patience': 3, 'historical_regime_weight': 0.05},
    'training': {'enabled': True, 'min_samples': 64, 'retrain_interval': 60, 'dimensions': 96, 'epochs': 100, 'learning_rate': 0.04}}
decl = {'domain_id': did, 'session_id': 'ignored', 'description': f'{DS} HTE reaction optimization: one query = one experiment; reward = yield.',
        'schema': {'entities': ['reactor', 'substrate'], 'relations': [REL], 'signals': D['context_names'] + ['yield'], 'event_types': ['state', 'feedback']},
        'hypotheses': hyps, 'learning': learning}
echo = api.post('/domains', decl)
(run / 'declaration.json').write_text(json.dumps(decl, indent=1)); (run / 'server_echo_learning.json').write_text(json.dumps(echo.get('learning'), indent=1))
trace = open(run / 'trace.jsonl', 'w', buffering=1); eps = open(run / 'episodes.jsonl', 'w', buffering=1); t0 = time.time(); overrides = gaps = 0; applied_count = 0
for e, sub in enumerate(subs):
    avail = {p for p, c in POL.items() if (sub, c) in D['yield']}
    gmax = max(D['yield'][(sub, POL[p])] for p in avail)
    ctx = dict(zip(D['context_names'], D['context'][sub]))
    tested, best, curve = set(), 0.0, []
    for step in range(BUDGET):
        q = api.post(f'/domains/{did}/query', {'session_id': 'ignored', 'question': 'Choose the next reaction condition to run.', 'relation': REL,
                     'context': {'event_type': 'state', 'values': ctx}, 'allow_exploration': True, 'top_k': 64}, mutation=True)
        sel = q.get('selection') or {}; chosen = sel.get('selected_policy'); override = None; execd = chosen
        if chosen not in avail or chosen in tested:
            override = 'already_tested' if chosen in tested else 'unavailable_or_abstained'
            execd = next((h['policy'] for h in (q.get('ranked_hypotheses') or []) if h.get('policy') in avail and h['policy'] not in tested), None) \
                or next(p for p in sorted(avail) if p not in tested)
            overrides += 1
        tested.add(execd); y = D['yield'][(sub, POL[execd])]
        fb = {'session_id': 'ignored', 'outcome': 'success', 'feedback_kind': 'execution', 'relation': REL, 'policy': execd,
              'values': {'yield': y}, 'metadata': {'episode_id': f'{DS}-o{ORDER}-e{e}', 'step': step}}
        if q.get('decision_id'): fb['decision_id'] = q['decision_id']
        try:
            r = api.post(f'/domains/{did}/feedback', fb)
        except Ambiguous:
            n = api.call('GET', f'/domains/{did}', None, mutation=False).get('event_count')
            r = {'credit_assignment': {'contextual_learning_applied': True, 'reconciled': True}} if n == applied_count + 1 else api.post(f'/domains/{did}/feedback', fb)
        applied_count += 1
        ok = (r.get('credit_assignment') or {}).get('contextual_learning_applied'); gaps += ok is not True
        best = max(best, y); curve.append(best / gmax)
        trace.write(json.dumps({'episode': e, 'step': step, 'selected': chosen, 'executed': execd, 'override': override, 'selected_by': sel.get('selected_by'),
                                'yield': y, 'gmax': gmax, 'feedback_applied': ok}) + '\n')
    trace.flush()
    eps.write(json.dumps({'episode': e, 'substrate': list(sub), 'gmax': gmax, 'curve': curve, 'calls': api.calls, 's': round(time.time() - t0)}) + '\n'); eps.flush()
    print(json.dumps({'ds': DS, 'order': ORDER, 'episode': e, 'b@5': round(curve[4], 3), 'b@10': round(curve[9], 3), 'b@end': round(curve[-1], 3),
                      'overrides': overrides, 'gaps': gaps, 'calls': api.calls, 's': round(time.time() - t0)}), flush=True)
(run / 'summary.json').write_text(json.dumps({'domain_id': did, 'dataset': DS, 'order': ORDER, 'budget': BUDGET, 'overrides': overrides,
                                             'feedback_gaps': gaps, 'api_calls': api.calls, 'api_seconds': round(api.seconds, 1)}, indent=1))
print('deleted', json.dumps(api.delete_domain(did))); api.close()
