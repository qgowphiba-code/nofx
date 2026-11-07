"""Light test: Adapt-1 sequential route on 2D HP inverse folding (terminal reward only).

usage: run_adapt1.py TRAIN_EPISODES
Episode = one target conformation; 12 decisions (H or P for residue i, in order); the only non-neutral reward is
the terminal design score p(target) in [0, 1] (exact enumeration). Context per decision (rules, known to a
designer): position, the residue's number of non-bonded contacts in the target, chain-end flag, previous choice.
Training: targets 0-19 cycled in a fixed shuffled order. Held out: targets 20-24, two frozen episodes each
(exploit only, no feedback).
"""
import sys, json, time, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'common'))
import oracle, lattice2d as L
from client import Client, new_id
NTRAIN = int(sys.argv[1]); N = L.N; REL = 'choose_residue'
run = HERE.parent / 'runs' / 'fold' / 'adapt1-seq'; run.mkdir(parents=True, exist_ok=True)
api = Client(run / 'requests.jsonl.xz')
did = new_id('sci-fold-seq')
hyps = [{'name': f'place {t}', 'relation': REL, 'policy': t.lower(), 'policy_features': {'hydrophobic': 1.0 if t == 'H' else 0.0},
         'when': [], 'predicts': ['residue type at this position'], 'falsified_by': [], 'weight': 1.0} for t in ('H', 'P')]
learning = {'enabled': True,
    'context': {'feature_paths': ['values.pos', 'values.contacts', 'values.chain_end', 'values.prev_h'], 'event_types': ['state'], 'max_samples': 4096},
    'reward': {'aggregation': 'weighted_mean', 'components': [{'field': 'values.step_reward', 'goal': 'maximize', 'min': 0.0, 'max': 1.0, 'weight': 1.0, 'required': True}]},
    'policy': {'decay_half_life_seconds': 0.0, 'transfer_strength': 0.0, 'action_transfer_strength': 0.0, 'min_context_observations': 4,
               'model_max_weight': 0.45, 'model_ood_threshold': 0.75, 'ood_confidence_penalty': 0.5, 'exploration_mode': 'thompson',
               'exploration_strength': 1.0, 'minimum_expected_reward': 0.0, 'minimum_confidence': 0.0, 'require_credible_dominance': False,
               'dominance_margin': 0.0, 'abstain_on_ood': False},
    'credit_assignment': {'mode': 'eligibility_trace', 'discount': 0.98, 'neutral_reward': 0.5, 'minimum_weight': 0.01, 'delay_path': 'values.delay_steps'},
    'sequential': {'enabled': True, 'episode_path': 'metadata.episode_id', 'step_path': 'metadata.step', 'next_context_path': 'values.next_state',
                   'reward_path': 'values.step_reward', 'terminal_path': 'values.terminal', 'discount': 0.98, 'n_step': 12,
                   'minimum_episodes': 8, 'minimum_validation_skill': 0.1, 'target_sync_interval': 10, 'conservative_weight': 0.05},
    'training': {'enabled': True, 'min_samples': 64, 'retrain_interval': 96, 'dimensions': 96, 'epochs': 100, 'learning_rate': 0.04}}
decl = {'domain_id': did, 'session_id': 'ignored', 'description': '2D HP inverse folding, residue-by-residue design, terminal reward.',
        'schema': {'entities': ['designer', 'chain'], 'relations': [REL], 'signals': ['pos', 'contacts', 'chain_end', 'prev_h', 'step_reward'],
                   'event_types': ['state', 'feedback']}, 'hypotheses': hyps, 'learning': learning}
echo = api.post('/domains', decl)
(run / 'declaration.json').write_text(json.dumps(decl, indent=1)); (run / 'server_declaration.json').write_text(json.dumps(echo, indent=1))
order = list(range(20)); random.Random(5).shuffle(order)
plan = [('train', order[e % 20], e) for e in range(NTRAIN)] + [('heldout', 20 + i // 2, NTRAIN + i) for i in range(10)]
out = open(run / 'episodes.jsonl', 'w'); trace = open(run / 'trace.jsonl', 'w'); t0 = time.time(); gaps = 0
def ctx_of(t, i, prev):
    deg = oracle.BANK[t]['degree']
    return {'pos': round(i / (N - 1), 4), 'contacts': deg[i] if i < N else 0, 'chain_end': 1 if i in (0, N - 1) else 0, 'prev_h': prev}
for phase, t, e in plan:
    seq, prev, decisions = '', 0, []
    for i in range(N):
        ctx = ctx_of(t, i, prev)
        q = api.post(f'/domains/{did}/query', {'session_id': 'ignored', 'question': 'Choose the residue type.', 'relation': REL,
                     'context': {'event_type': 'state', 'values': ctx}, 'allow_exploration': phase == 'train', 'top_k': 2,
                     **({} if phase == 'train' else {'selection_mode': 'exploit'})}, mutation=True)
        sel = q.get('selection') or {}
        pol = sel.get('selected_policy')
        if pol not in ('h', 'p'):
            ranked = [h['policy'] for h in (q.get('ranked_hypotheses') or []) if h.get('policy') in ('h', 'p')]
            pol = ranked[0] if ranked else 'p'
        seq += pol.upper(); prev = 1 if pol == 'h' else 0
        decisions.append((q.get('decision_id'), ctx, pol, sel.get('selected_by')))
    r = oracle.reward(seq, t)
    if phase == 'train':
        for i, (dec, ctx, pol, how) in enumerate(decisions):
            term = i == N - 1
            nxt = ctx_of(t, i + 1, 1 if pol == 'h' else 0) if not term else ctx
            fb = {'session_id': 'ignored', 'outcome': 'success', 'feedback_kind': 'execution', 'relation': REL, 'policy': pol,
                  'values': {'step_reward': r if term else 0.5, 'next_state': {'event_type': 'state', 'values': nxt}, 'terminal': term, 'delay_steps': 0},
                  'metadata': {'episode_id': f'ep{e}', 'step': i}}
            if dec: fb['decision_id'] = dec
            fr = api.post(f'/domains/{did}/feedback', fb)
            gaps += (fr.get('credit_assignment') or {}).get('contextual_learning_applied') is not True
    rec = {'phase': phase, 'episode': e, 'target': t, 'seq': seq, 'reward': r, 'ceiling': oracle.BANK[t]['ceiling'],
           'burial': oracle.BANK[t]['burial_heuristic'], 'selected_by': [d[3] for d in decisions], 'calls': api.calls, 's': round(time.time() - t0)}
    out.write(json.dumps(rec) + '\n'); out.flush()
    print(json.dumps({k: rec[k] for k in ('phase', 'episode', 'target', 'seq', 'reward', 'ceiling', 'calls', 's')}), flush=True)
(run / 'summary.json').write_text(json.dumps({'domain_id': did, 'train_episodes': NTRAIN, 'feedback_gaps': gaps, 'api_calls': api.calls}, indent=1))
try:
    (run / 'final_domain.json').write_text(json.dumps(api.call('GET', f'/domains/{did}', None, mutation=False), indent=1)[:3_000_000])
except Exception as ex:
    print('final read failed', ex)
print('deleted', json.dumps(api.delete_domain(did))); api.close()
