"""Sachs et al. 2005 (pooled single-cell data, 11 proteins; consensus network as ground truth, cdt resources).
For each protein as target, Adapt-1 Transition Discovery (empty input_paths, autonomous projection, maximum_input_paths K)
sees the log-abundances of the other 10 proteins and the target, one observation event per cell (N cells, batched by 50).
Its discovered input paths are compared with the target's neighbours in the consensus network.
usage: run.py K SEED N"""
import sys, csv, json, math, random
sys.path.insert(0, 'common')
from client import Client, new_id
K, SEED, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rows = list(csv.DictReader(open('data/raw/sachs/cyto_full_data.csv')))
names = list(rows[0].keys()); safe = {n: n.replace('/', '_').lower() for n in names}
cells = random.Random(SEED).sample(rows, N)
c = Client(f'runs/sachs/requests_k{K}_s{SEED}.jsonl')
out = open(f'runs/sachs/discovery_k{K}_s{SEED}.jsonl', 'w', buffering=1)
for orig in names:
    tgt = safe[orig]; did = new_id(f'sci-sachs-{tgt}')
    decl = {'domain_id': did, 'session_id': 'ignored', 'description': 'Sachs 2005 Transition Discovery', 'schema': {'event_types': ['observation']},
            'hypotheses': [], 'learning': {'enabled': True, 'transition': {'enabled': True, 'event_types': ['observation'], 'input_paths': [],
            'targets': [{'path': f'values.state.{tgt}', 'type': 'number'}],
            'autonomous_projection': {'enabled': True, 'discover_causal_bindings': True, 'maximum_input_paths': K},
            'causal_graph': {'enabled': True, 'event_types': ['observation'], 'intervention_path': '', 'variables': []}}}}
    c.post('/domains', decl); proj, revs = None, []
    for i in range(0, N, 50):
        ops = []
        for j, row in enumerate(cells[i:i + 50]):
            v = {safe[n]: round(math.log(float(row[n]) + 1.0), 4) for n in names if n != orig}
            v['state'] = {tgt: round(math.log(float(row[orig]) + 1.0), 4)}
            ops.append({'operation': 'event', 'ref': f'e{i + j}', 'event_type': 'observation', 'values': v, 'metadata': {'cell': i + j}})
        r = c.post(f'/domains/{did}/batch', {'session_id': 'ignored', 'operations': ops})
        p = (((r.get('results') or [{}])[-1].get('result') or {}).get('learner_eligibility') or {}).get('autonomous_projection') or {}
        if p: proj = p; revs.append((i + 50, p.get('revision'), p.get('status'), p.get('input_paths')))
    out.write(json.dumps({'target': tgt, 'K': K, 'seed': SEED, 'N': N, 'input_paths': (proj or {}).get('input_paths'),
                          'status': (proj or {}).get('status'), 'revision': (proj or {}).get('revision'),
                          'causal_variables': (proj or {}).get('causal_variables'), 'revisions': revs[-3:]}) + '\n')
    c.delete_domain(did)
c.close()
