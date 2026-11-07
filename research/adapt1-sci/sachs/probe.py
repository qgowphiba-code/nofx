import sys, csv, json, math, random
sys.path.insert(0, 'common')
from client import Client, new_id
rows = list(csv.DictReader(open('data/raw/sachs/cyto_full_data.csv')))
names = list(rows[0].keys()); safe = {n: n.replace('/', '_').lower() for n in names}
TARGET = sys.argv[1]; ORIG = next(n for n in names if safe[n] == TARGET); K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
c = Client('runs/sachs_probe.jsonl')
did = new_id(f'sci-sachs-{TARGET}')
decl = {'domain_id': did, 'session_id': 'ignored', 'description': 'Sachs 2005 single-cell signalling, Transition Discovery probe',
        'schema': {'event_types': ['observation']}, 'hypotheses': [],
        'learning': {'enabled': True, 'transition': {'enabled': True, 'event_types': ['observation'], 'input_paths': [],
                     'targets': [{'path': f'values.state.{TARGET}', 'type': 'number'}],
                     'autonomous_projection': {'enabled': True, 'discover_causal_bindings': True, 'maximum_input_paths': K},
                     'causal_graph': {'enabled': True, 'event_types': ['observation'], 'intervention_path': '', 'variables': []}}}}
r = c.post('/domains', decl); print('created; transition echo keys:', sorted((r.get('learning') or {}).get('transition', {}).keys())[:12])
random.seed(1); sample = random.sample(rows, 400)
last = None
for i in range(0, len(sample), 50):
    ops = []
    for j, row in enumerate(sample[i:i + 50]):
        v = {safe[n]: round(math.log(float(row[n]) + 1.0), 4) for n in names if safe[n] != TARGET}
        v['state'] = {TARGET: round(math.log(float(row[ORIG]) + 1.0), 4)}
        ops.append({'operation': 'event', 'ref': f'e{i + j}', 'event_type': 'observation', 'values': v, 'metadata': {'cell': i + j}})
    last = c.post(f'/domains/{did}/batch', {'session_id': 'ignored', 'operations': ops})
res = (last.get('results') or [{}])[-1] if isinstance(last, dict) else {}
print('batch response keys:', sorted(last.keys())[:10]); print('last op keys:', sorted(res.keys())[:15])
proj = json.dumps(res.get('learner_eligibility') or res, default=str)
print('learner_eligibility (trunc):', proj[:1200])
d = c.call('GET', f'/domains/{did}', None, mutation=False); t = (d.get('learning') or {}).get('transition', {})
print('stored input_paths:', t.get('input_paths'), '| projection:', json.dumps(t.get('autonomous_projection'))[:300])
print(c.delete_domain(did)); c.close()
