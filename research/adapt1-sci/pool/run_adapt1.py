"""Adapt-1 contextual policy route as a pool-based experiment planner on a measured Olympus dataset (single campaign, no
retained state from other campaigns). Policies = the measured designs, with their scaled parameters as policy_features;
constant context; reward = measured value / dataset max; no design is run twice (override to the highest-ranked untested).
usage: run_adapt1.py DATASET BUDGET EXPLORATION"""
import sys, json, time
sys.path.insert(0, 'pool'); sys.path.insert(0, 'common')
import data
from client import Client, new_id
NAME, B, EXPL = sys.argv[1], int(sys.argv[2]), sys.argv[3]; INIT = int(sys.argv[4]) if len(sys.argv) > 4 else -1
X, y = data.load(NAME); ymax = max(y); POL = [f'd{i}' for i in range(len(y))]; REL = 'run_design'
def slim(path, resp):
    if path.endswith('/query') and isinstance(resp, dict):
        return {'decision_id': resp.get('decision_id'), 'selection': resp.get('selection'), 'ranked_top5': [h.get('policy') for h in (resp.get('ranked_hypotheses') or [])[:5]]}
    return resp
run = f'runs/pool/adapt1-{NAME}-{EXPL}' + (f'-init{INIT}' if INIT >= 0 else ''); __import__('os').makedirs(run, exist_ok=True)
c = Client(f'{run}/requests.jsonl', slim=slim); did = new_id(f'sci-pool-{NAME[:8]}')
hyps = [{'name': p, 'relation': REL, 'policy': p, 'policy_features': {f'x{j}': round(v, 4) for j, v in enumerate(X[i])},
         'when': [], 'predicts': ['measured objective'], 'falsified_by': [], 'weight': 1.0} for i, p in enumerate(POL)]
c.post('/domains', {'domain_id': did, 'session_id': 'ignored', 'description': f'{NAME} pool-based experiment planning', 'hypotheses': hyps,
                    'schema': {'entities': ['lab'], 'relations': [REL], 'signals': ['bias', 'reward'], 'event_types': ['state', 'feedback']},
                    'learning': {'enabled': True, 'context': {'feature_paths': ['values.bias'], 'event_types': ['state'], 'max_samples': 4096},
                                 'reward': {'aggregation': 'weighted_mean', 'components': [{'field': 'values.reward', 'goal': 'maximize', 'min': 0.0, 'max': 1.0, 'weight': 1.0, 'required': True}]},
                                 'policy': {'exploration_mode': EXPL, 'exploration_strength': 1.0, 'action_transfer_strength': 0.3, 'transfer_strength': 0.0},
                                 'credit_assignment': {'mode': 'none', 'neutral_reward': 0.5}}})
tested, best, out, t0 = set(), -1e9, open(f'{run}/trace.jsonl', 'w', buffering=1), time.time()
init = __import__('random').Random(INIT).sample(range(len(y)), 5) if INIT >= 0 else []
for n, i in enumerate(init, 1):      # the same 5 random starting experiments a GP gets, replayed as explicit-context feedback
    tested.add(i); best = max(best, y[i])
    c.post(f'/domains/{did}/feedback', {'session_id': 'ignored', 'outcome': 'success', 'feedback_kind': 'execution', 'relation': REL, 'policy': POL[i],
           'context': {'event_type': 'state', 'values': {'bias': 1.0}}, 'values': {'reward': y[i] / ymax}})
    out.write(json.dumps({'n': n, 'design': i, 'value': y[i], 'best': best, 'best_frac': best / ymax, 'override': 'random_init'}) + '\n')
for n in range(len(init) + 1, B + 1):
    q = c.post(f'/domains/{did}/query', {'session_id': 'ignored', 'question': 'Choose the next design to run.', 'relation': REL,
               'context': {'event_type': 'state', 'values': {'bias': 1.0}}, 'allow_exploration': True, 'top_k': 64}, mutation=True)
    sel = (q.get('selection') or {}).get('selected_policy'); ov = None
    if sel not in POL or int(sel[1:]) in tested:
        ov = 'already_tested'; sel = next((h['policy'] for h in (q.get('ranked_hypotheses') or []) if h.get('policy') in POL and int(h['policy'][1:]) not in tested), None) \
            or next(p for p in POL if int(p[1:]) not in tested)
    i = int(sel[1:]); tested.add(i); best = max(best, y[i])
    fb = {'session_id': 'ignored', 'outcome': 'success', 'feedback_kind': 'execution', 'relation': REL, 'policy': sel, 'values': {'reward': y[i] / ymax}}
    if q.get('decision_id'): fb['decision_id'] = q['decision_id']
    c.post(f'/domains/{did}/feedback', fb)
    out.write(json.dumps({'n': n, 'design': i, 'value': y[i], 'best': best, 'best_frac': best / ymax, 'override': ov}) + '\n')
print(json.dumps({'dataset': NAME, 'exploration': EXPL, 'best_frac@10,25,50,100': [json.loads(l)['best_frac'] for l in open(f'{run}/trace.jsonl') if json.loads(l)['n'] in (10, 25, 50, 100)], 's': round(time.time() - t0)}))
print('deleted', json.dumps(c.delete_domain(did))); c.close()
