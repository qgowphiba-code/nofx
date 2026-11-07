import sys, json
sys.path.insert(0, 'rxn'); sys.path.insert(0, 'common')
from client import Client, new_id, ApiError
import decl, sim
c = Client('runs/probe_rxn_journal.jsonl.xz')
did = new_id('sci-rxn-probe')
d = decl.declaration(did)
r = c.post('/domains', d)
print('create keys', list(r.keys())[:30])
print('hyp count echoed', len(r.get('hypotheses', [])), 'learning.policy', json.dumps(r.get('learning', {}).get('policy'))[:400])
q = c.post(f'/domains/{did}/query', {'session_id': 'ignored', 'question': 'Choose the next experiment condition.', 'relation': decl.RELATION,
          'context': {'event_type': 'state', 'values': {'d1': 0.3, 'd2': -0.5, 'd3': 0.1}}, 'allow_exploration': True,
          'selection_mode': 'thompson', 'top_k': 48}, mutation=True)
print('query keys', list(q.keys()))
print('selection', json.dumps(q.get('selection'))[:600])
rh = q.get('ranked_hypotheses') or []
print('ranked n', len(rh), json.dumps(rh[:2])[:700])
pol = (q.get('selection') or {}).get('selected_policy') or (q.get('selection') or {}).get('policy')
fb = c.post(f'/domains/{did}/feedback', {'session_id': 'ignored', 'outcome': 'success', 'feedback_kind': 'execution', 'decision_id': q.get('decision_id'),
            'relation': decl.RELATION, 'policy': pol, 'values': {'yield': 0.42}, 'metadata': {'episode_id': 'probe-0', 'step': 0}})
print('feedback', json.dumps(fb)[:900])
q2 = c.post(f'/domains/{did}/query', {'session_id': 'ignored', 'question': 'Choose the next experiment condition.', 'relation': decl.RELATION,
          'context': {'event_type': 'state', 'values': {'d1': 0.3, 'd2': -0.5, 'd3': 0.1}}, 'allow_exploration': True, 'top_k': 48}, mutation=True)
print('query2 selection', json.dumps(q2.get('selection'))[:400])
print('delete', c.delete_domain(did)); print('calls', c.calls, 'sec', round(c.seconds, 2)); c.close()
