import sys, json
sys.path.insert(0, 'rxn'); sys.path.insert(0, 'common')
from client import Client, new_id, ApiError
import decl
c = Client('runs/probe_rxn_journal.jsonl.xz')
did = new_id('sci-rxn-probe2'); c.post('/domains', decl.declaration(did))
base = {'session_id': 'ignored', 'question': 'Choose the next experiment condition.', 'relation': decl.RELATION,
        'context': {'event_type': 'state', 'values': {'d1': 0.3, 'd2': -0.5, 'd3': 0.1}}, 'allow_exploration': True, 'top_k': 48}
allowed = [decl.policy_id(k) for k in range(0, 48, 7)]
for field in ('candidate_policies', 'allowed_policies', 'policies', 'blocked_policies', 'eligible_policies', 'exclude_policies'):
    b = dict(base); b[field] = allowed if field != 'blocked_policies' and field != 'exclude_policies' else [decl.policy_id(k) for k in range(48) if decl.policy_id(k) not in allowed]
    try:
        q = c.post(f'/domains/{did}/query', b, mutation=True)
        sel = q.get('selection') or {}
        print(field, '-> selected', sel.get('selected_policy'), 'in allowed:', sel.get('selected_policy') in allowed, 'cands', sel.get('candidate_policies'), 'blocked', (sel.get('safety') or {}).get('blocked_policies'))
    except ApiError as e:
        print(field, '-> ERR', e.code, e.body[:300])
print(c.delete_domain(did)); c.close()
