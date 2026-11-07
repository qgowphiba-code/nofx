"""Adapt-1 (production API) on Craftax-Classic-Symbolic, online: learning on, exploration on, one decision per game step.

usage: cx_run_adapt1.py SEED EPISODES OUT_DIR [JOURNAL_DIR]          (episode cap from CRAFTAX_EP_CAP, default 500)
Declaration: the docs' recommended sequential-learning declaration; context = cx_env features (inventory, stats, view block
counts, offsets to key blocks, mob proximity, facing block, achievements so far, step fraction, last action); policies =
the 17 Craftax actions (action index and a crafting/placing flag as features); reward = Craftax's own step reward (bounds
[-1, 1]); sequential learning discount 0.95, n_step 10, eligibility traces; TCP (lag 6) over inventory and stats; latent
belief on. Abstention -> random action, counted. One Domain for the run; each game is one episode.
"""
import sys, json, time, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'common'))
import numpy as np
import cx_env
from client import Client, Ambiguous, ApiError, new_id

SEED, EPISODES, OUT = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
JDIR = Path(sys.argv[4]) if len(sys.argv) > 4 else OUT
import os
BASE = os.environ.get('ADAPT1_BASE', 'https://rei-neuroadapt-api.reilabs.org/api/v1')
OUT.mkdir(parents=True, exist_ok=True); JDIR.mkdir(parents=True, exist_ok=True)
tag = f'cx-s{SEED}-cap{cx_env.EP_CAP}'
env = cx_env.CraftaxTask(SEED); FIELDS = sorted(env.reset(0).keys())
ACTIONS = cx_env.ACTIONS; POLICIES = [f'a{i}' for i in range(len(ACTIONS))]
REL = 'act'
hyps = [{'name': f'action {a}', 'relation': REL, 'policy': p, 'policy_features': {'action_index': float(i), 'is_move': float(a in ('LEFT', 'RIGHT', 'UP', 'DOWN')),
         'is_craft': float(a.startswith(('MAKE', 'PLACE')))}, 'when': [], 'predicts': ['step reward'], 'falsified_by': [], 'weight': 1.0}
        for i, (p, a) in enumerate(zip(POLICIES, ACTIONS))]
tcp_inputs = [f'inv_{n}' for n in cx_env.INV[:6]] + ['health', 'food', 'drink', 'energy']
learning = {
    'enabled': True,
    'context': {'feature_paths': [f'values.{k}' for k in FIELDS], 'event_types': ['state'], 'max_samples': 4096},
    'reward': {'aggregation': 'weighted_mean', 'components': [{'field': 'values.step_reward', 'goal': 'maximize', 'min': -1.0, 'max': 1.0, 'weight': 1.0, 'required': True}]},
    'policy': {'decay_half_life_seconds': 0.0, 'transfer_strength': 0.0, 'action_transfer_strength': 0.05, 'min_context_observations': 4,
               'model_max_weight': 0.45, 'model_ood_threshold': 0.75, 'ood_confidence_penalty': 0.5, 'exploration_mode': 'auto',
               'exploration_strength': 1.0, 'minimum_expected_reward': 0.0, 'minimum_confidence': 0.0, 'require_credible_dominance': False,
               'dominance_margin': 0.0, 'abstain_on_ood': False},
    'credit_assignment': {'mode': 'eligibility_trace', 'discount': 0.95, 'neutral_reward': 0.0, 'minimum_weight': 0.01, 'delay_path': 'values.delay_steps'},
    'latent_belief': {'enabled': True, 'max_regimes': 8, 'novelty_threshold': 0.55, 'temperature': 0.25, 'change_point_threshold': 0.55,
                      'change_point_patience': 3, 'historical_regime_weight': 0.05},
    'training': {'enabled': True, 'min_samples': 64, 'retrain_interval': 16, 'dimensions': 96, 'epochs': 100, 'learning_rate': 0.04},
    'sequential': {'enabled': True, 'episode_path': 'metadata.episode_id', 'step_path': 'metadata.step', 'next_context_path': 'values.next_state',
                   'reward_path': 'values.step_reward', 'terminal_path': 'values.terminal', 'discount': 0.95, 'n_step': 10, 'minimum_episodes': 10,
                   'minimum_validation_skill': 0.1, 'target_sync_interval': 10, 'conservative_weight': 0.05},
    'temporal_context': {'enabled': True, 'episode_path': 'metadata.episode_id', 'step_path': 'metadata.step', 'input_paths': [f'values.{k}' for k in tcp_inputs], 'maximum_lag': 6},
}
api = Client(JDIR / f'journal-{tag}.jsonl.xz', base=BASE)
did = new_id(f'cx-s{SEED}')
decl = {'domain_id': did, 'session_id': 'ignored', 'description': f'Craftax-Classic-Symbolic seed {SEED}, {cx_env.EP_CAP}-step episodes: online agent, one decision per step.',
        'schema': {'entities': ['agent', 'world'], 'relations': [REL], 'signals': FIELDS + ['step_reward'], 'event_types': ['state', 'feedback']},
        'hypotheses': hyps, 'learning': learning}
echo = api.create_domain(decl)
(OUT / f'declaration-{tag}.json').write_text(json.dumps(decl, indent=1)); (OUT / f'echo-{tag}.json').write_text(json.dumps(echo.get('learning'), indent=1))
ep_log = open(OUT / f'episodes-{tag}.jsonl', 'w', buffering=1); diag_log = open(OUT / f'diagnostics-{tag}.jsonl', 'w', buffering=1)
stats = {'steps': 0, 'fallbacks': 0, 'not_applied': 0, 'query_retries': 0}; rng = random.Random(10_000 + SEED); t0 = time.time()


def query(values, ep_id, step, diag):
    body = {'session_id': 'ignored', 'question': 'Choose the next action.', 'relation': REL,
            'context': {'event_type': 'state', 'values': values, 'metadata': {'episode_id': ep_id, 'step': step}},
            'allow_exploration': True, 'selection_mode': 'auto',
            'return_fields': (['selection', 'decision_id', 'ranked_hypotheses', 'learning_state', 'latent_belief'] if diag else ['selection', 'decision_id'])}
    for attempt in range(3):
        try:
            return api.post(f'/domains/{did}/query', body)
        except (Ambiguous, ApiError) as e:
            if isinstance(e, ApiError) and e.code < 500:
                raise
            stats['query_retries'] += 1; time.sleep(2 * (attempt + 1))
    raise SystemExit(f'query failed 3 times at {ep_id} step {step}')


try:
    for ep in range(EPISODES):
        ep_id = f'{tag}-e{ep:05d}'; values = env.reset(SEED * 100000 + ep); step = 0; rewards = []; fallbacks = 0; done = False
        while not done:
            diag = (step in (0, 250) and ep % 5 == 0)
            q = query(values, ep_id, step, diag); sel = (q.get('selection') or {}).get('selected_policy')
            if sel in POLICIES:
                a = POLICIES.index(sel)
            else:
                a = rng.randrange(len(ACTIONS)); fallbacks += 1
            nxt, r, done, info = env.step(a); rewards.append(r)
            fb = {'session_id': 'ignored', 'outcome': 'success' if r > 0 else 'neutral', 'feedback_kind': 'execution', 'relation': REL, 'policy': POLICIES[a],
                  'values': {'step_reward': float(np.clip(r, -1, 1)), 'next_state': {'values': nxt}, 'terminal': bool(done), 'delay_steps': 0}, 'metadata': {'episode_id': ep_id, 'step': step}}
            if q.get('decision_id'):
                fb['decision_id'] = q['decision_id']
            res = api.feedback(did, fb, stats)          # reconciles an ambiguous POST by event count before any resend
            if res.get('reconciled') != 'applied' and (res.get('credit_assignment') or {}).get('contextual_learning_applied') is not True:
                stats['not_applied'] += 1
            if diag:
                ls = (q.get('learning_state') or {}).get('subsystems') or {}
                diag_log.write(json.dumps({'episode': ep, 'step': step, 'selected': sel, 'selected_by': (q.get('selection') or {}).get('selected_by'),
                                           'feedback_policy': (ls.get('feedback_policy') or {}).get('sample_count'), 'sequential': ls.get('sequential') or ls.get('sequential_policy')}) + '\n')
            values = nxt; step += 1; stats['steps'] += 1
        stats['fallbacks'] += fallbacks
        ep_log.write(json.dumps({'episode': ep, 'return': float(sum(rewards)), 'achievements': env.achievements(), 'n_achievements': len(env.achievements()), 'len': step,
                                 'fallbacks': fallbacks, 'steps': stats['steps'], 'elapsed_s': round(time.time() - t0, 1), 'api_calls': api.calls}) + '\n')
        print(f'{tag} ep {ep}: return {sum(rewards):.2f} achievements {len(env.achievements())} len {step} calls {api.calls} {time.time() - t0:.0f}s', flush=True)
        if stats['steps'] > 50 and stats['not_applied'] > 0.01 * stats['steps']:
            raise SystemExit(f'abort: {stats["not_applied"]} of {stats["steps"]} feedbacks not applied')
    outcome = 'completed'
except SystemExit as e:
    outcome = f'aborted: {e}'
finally:
    deleted = api.delete_domain(did)
    summary = {'seed': SEED, 'episodes_requested': EPISODES, 'ep_cap': cx_env.EP_CAP, 'base_url': BASE, 'outcome': locals().get('outcome', 'crashed'), 'domain_id': did, 'stats': stats,
               'api_calls': api.calls, 'wall_s': round(time.time() - t0), 'delete': deleted}
    (OUT / f'summary-{tag}.json').write_text(json.dumps(summary, indent=1)); api.close(); print(json.dumps(summary))
