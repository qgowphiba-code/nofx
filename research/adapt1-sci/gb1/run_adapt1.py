"""Adapt-1 contextual policy route designing GB1 variants position by position (4 queries per variant).
Context at position k: one-hot position + for each earlier position its chosen residue's descriptors (hydropathy, volume,
charge) and a flag. Policies: the 20 amino acids, with descriptor policy_features. After the variant is measured, each of
its 4 decisions receives the same reward (application-computed return, as the docs allow): fitness / max fitness.
No variant is measured twice: at the last position an already-measured variant is replaced by Adapt-1's highest-ranked
residue that gives a new one (override recorded). usage: run_adapt1.py EXPLORATION BUDGET"""
import sys, json, time
sys.path.insert(0, 'gb1'); sys.path.insert(0, 'common')
import land as L
from client import Client, new_id, Ambiguous
EXPL, B = sys.argv[1], int(sys.argv[2]); REL = 'choose_residue'
run_dir = f'runs/gb1/adapt1-{EXPL}'; __import__('os').makedirs(run_dir, exist_ok=True)
def slim(path, resp):
    if path.endswith('/query') and isinstance(resp, dict):
        return {'decision_id': resp.get('decision_id'), 'selection': resp.get('selection'), 'ranked_top5': [h.get('policy') for h in (resp.get('ranked_hypotheses') or [])[:5]]}
    return resp
c = Client(f'{run_dir}/requests.jsonl', slim=slim)
did = new_id(f'sci-gb1-{EXPL}')
nrm = lambda a: [(L.KD[a] + 4.5) / 9.0, (L.VOL[a] - 60) / 170, L.CHG[a]]
hyps = [{'name': f'residue {a}', 'relation': REL, 'policy': a.lower(), 'policy_features': {'hydropathy': nrm(a)[0], 'volume': nrm(a)[1], 'charge': nrm(a)[2]},
         'when': [], 'predicts': ['variant fitness'], 'falsified_by': [], 'weight': 1.0} for a in L.AA]
fields = [f'pos{k}' for k in range(4)] + [f'p{j}_{f}' for j in range(4) for f in ('set', 'hyd', 'vol', 'chg')]
decl = {'domain_id': did, 'session_id': 'ignored', 'description': 'GB1 four-site design, residue by residue', 'hypotheses': hyps,
        'schema': {'entities': ['designer', 'protein'], 'relations': [REL], 'signals': fields + ['fitness'], 'event_types': ['state', 'feedback']},
        'learning': {'enabled': True, 'context': {'feature_paths': ['values.' + f for f in fields], 'event_types': ['state'], 'max_samples': 4096},
                     'reward': {'aggregation': 'weighted_mean', 'components': [{'field': 'values.reward', 'goal': 'maximize', 'min': 0.0, 'max': 1.0, 'weight': 1.0, 'required': True}]},
                     'policy': {'exploration_mode': EXPL, 'exploration_strength': 1.0, 'action_transfer_strength': 0.3, 'transfer_strength': 0.0, 'min_context_observations': 4},
                     'credit_assignment': {'mode': 'none', 'neutral_reward': 0.5},
                     'training': {'enabled': True, 'min_samples': 64, 'retrain_interval': 64, 'dimensions': 96, 'epochs': 100, 'learning_rate': 0.04}}}
c.post('/domains', decl)
def ctx(k, prefix):
    v = {f'pos{j}': 1.0 if j == k else 0.0 for j in range(4)}
    for j in range(4):
        if j < len(prefix): h, vo, ch = nrm(prefix[j]); v.update({f'p{j}_set': 1.0, f'p{j}_hyd': h, f'p{j}_vol': vo, f'p{j}_chg': ch})
        else: v.update({f'p{j}_set': 0.0, f'p{j}_hyd': 0.0, f'p{j}_vol': 0.0, f'p{j}_chg': 0.0})
    return v
seen, best, out, t0, overrides = set(), 0.0, open(f'{run_dir}/designs.jsonl', 'w', buffering=1), time.time(), 0
for n in range(B):
    prefix, decs = '', []
    for k in range(4):
        q = c.post(f'/domains/{did}/query', {'session_id': 'ignored', 'question': 'Choose the residue at this position.', 'relation': REL,
                   'context': {'event_type': 'state', 'values': ctx(k, prefix)}, 'allow_exploration': True, 'top_k': 20}, mutation=True)
        sel = (q.get('selection') or {}).get('selected_policy') or 'a'; aa = sel.upper()
        if k == 3 and prefix + aa in seen:
            alt = [h['policy'].upper() for h in (q.get('ranked_hypotheses') or []) if prefix + h['policy'].upper() not in seen]
            if alt: aa = alt[0]; overrides += 1
        decs.append((q.get('decision_id'), ctx(k, prefix), aa.lower())); prefix += aa
    v = prefix; seen.add(v); f = L.fit(v); best = max(best, f); r = f / L.FMAX
    for dec, cx, pol in decs:
        fb = {'session_id': 'ignored', 'outcome': 'success', 'feedback_kind': 'execution', 'relation': REL, 'policy': pol, 'values': {'reward': r, 'fitness': f},
              'metadata': {'episode_id': f'v{n}', 'step': 0}}
        if dec: fb['decision_id'] = dec
        try: c.post(f'/domains/{did}/feedback', fb)
        except Ambiguous: pass
    out.write(json.dumps({'n': n + 1, 'variant': v, 'fitness': f, 'best': best, 'measured': v in L.F}) + '\n')
    if (n + 1) % 50 == 0: print(f'gb1 {EXPL} n={n + 1} best={best:.3f} overrides={overrides} calls={c.calls} {time.time() - t0:.0f}s', flush=True)
print('deleted', json.dumps(c.delete_domain(did))); c.close()
