"""Adapt-1 (Fast-UAT) on a canonical bsuite environment, online: learning on, exploration on, every step scored.

usage: run_adapt1.py ENV SEED EPISODES OUT_DIR [JOURNAL_DIR]

Declaration: the docs' "Recommended starting declaration" for sequential learning (docs/neuroadapt/sequential-learning,
section 3), with only the task-specific fields replaced:
  - context features = the flattened bsuite observation (the same input the baseline agents get);
  - policies = the environment's discrete actions; action_transfer_strength 0 because bsuite actions have unrelated
    semantics (docs section 10: "Set both transfer strengths to 0 when actions ... have unrelated semantics");
  - reward = the raw bsuite step reward with its true bounds;
  - bandit (1-step, immediate reward): sequential off, credit_assignment none (docs: "Use mode: none when rewards are
    immediate");
  - memory5: temporal_context on over all observation fields, maximum_lag 6 (docs: temporal-context-projection).
Online loop (docs section 13): query before acting -> execute -> one decision-linked feedback per step with next_state,
terminal and step_reward. Episode IDs are zero-padded so lexicographic order is chronological (docs section 19).
If Adapt-1 abstains or returns no legal action, a uniformly random action is executed, feedback credits the executed action,
and the step is counted as a fallback (reported separately; docs section 19).
Fails loudly: any feedback with contextual_learning_applied != true is counted; the run aborts if that exceeds 1 %.
"""
import sys, json, time, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
from client import Client, Ambiguous, ApiError, new_id
import envs

ENV, SEED, EPISODES, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
JDIR = Path(sys.argv[5]) if len(sys.argv) > 5 else OUT
SPEC = envs.SPECS[ENV]
OUT.mkdir(parents=True, exist_ok=True); JDIR.mkdir(parents=True, exist_ok=True)
tag = f'{ENV}-s{SEED}'
env = envs.make_env(ENV, SEED)
n_actions = env.action_spec().num_values
fields = list(envs.obs_values(env.reset().observation))
REL = 'acts'
POLICIES = [f'a{i}' for i in range(n_actions)]

hyps = [{'name': f'action {i}', 'relation': REL, 'policy': p, 'policy_features': {'action_index': float(i)}, 'when': [],
         'predicts': ['step reward'], 'falsified_by': [], 'weight': 1.0} for i, p in enumerate(POLICIES)]
learning = {
    'enabled': True,
    'context': {'feature_paths': [f'values.{k}' for k in fields], 'event_types': ['state'], 'max_samples': 4096},
    'reward': {'aggregation': 'weighted_mean', 'components': [{'field': 'values.step_reward', 'goal': 'maximize',
               'min': SPEC['r_min'], 'max': SPEC['r_max'], 'weight': 1.0, 'required': True}]},
    'policy': {'decay_half_life_seconds': 0.0, 'transfer_strength': 0.0, 'action_transfer_strength': 0.0, 'min_context_observations': 4,
               'model_max_weight': 0.45, 'model_ood_threshold': 0.75, 'ood_confidence_penalty': 0.5, 'exploration_mode': 'auto',
               'exploration_strength': 1.0, 'minimum_expected_reward': 0.0, 'minimum_confidence': 0.0, 'require_credible_dominance': False,
               'dominance_margin': 0.0, 'abstain_on_ood': False},
    'credit_assignment': ({'mode': 'eligibility_trace', 'discount': 0.95, 'neutral_reward': 0.5, 'minimum_weight': 0.01,
                           'delay_path': 'values.delay_steps'} if SPEC['sequential'] else {'mode': 'none', 'neutral_reward': 0.5}),
    'latent_belief': {'enabled': True, 'max_regimes': 8, 'novelty_threshold': 0.55, 'temperature': 0.25, 'change_point_threshold': 0.55,
                      'change_point_patience': 3, 'historical_regime_weight': 0.05},
    'training': {'enabled': True, 'min_samples': 64, 'retrain_interval': 16, 'dimensions': 96, 'epochs': 100, 'learning_rate': 0.04},
}
if SPEC['sequential']:
    learning['sequential'] = {'enabled': True, 'episode_path': 'metadata.episode_id', 'step_path': 'metadata.step',
                              'next_context_path': 'values.next_state', 'reward_path': 'values.step_reward', 'terminal_path': 'values.terminal',
                              'discount': 0.95, 'n_step': 6, 'minimum_episodes': 10, 'minimum_validation_skill': 0.1,
                              'target_sync_interval': 10, 'conservative_weight': 0.05}
if SPEC['memory']:
    learning['temporal_context'] = {'enabled': True, 'episode_path': 'metadata.episode_id', 'step_path': 'metadata.step',
                                    'input_paths': [f'values.{k}' for k in fields], 'maximum_lag': 6}

api = Client(JDIR / f'journal-{tag}.jsonl.xz')
did = new_id(f'rl-{ENV}-s{SEED}')
decl = {'domain_id': did, 'session_id': 'ignored', 'description': f'bsuite {ENV} seed {SEED}: online agent, one decision per environment step.',
        'schema': {'entities': ['agent', 'environment'], 'relations': [REL], 'signals': fields + ['step_reward'], 'event_types': ['state', 'feedback']},
        'hypotheses': hyps, 'learning': learning}
echo = api.post('/domains', decl)
(OUT / f'declaration-{tag}.json').write_text(json.dumps(decl, indent=1))
(OUT / f'echo-{tag}.json').write_text(json.dumps(echo.get('learning'), indent=1))
ep_log = open(OUT / f'episodes-{tag}.jsonl', 'w', buffering=1)
diag_log = open(OUT / f'diagnostics-{tag}.jsonl', 'w', buffering=1)
rng = random.Random(10_000 + SEED)
stats = {'steps': 0, 'fallbacks': 0, 'not_applied': 0, 'query_retries': 0, 'feedback_reconciled': 0}
t0 = time.time()


def query(values, ep_id, step, diag):
    body = {'session_id': 'ignored', 'question': 'Choose the next action for the current state.', 'relation': REL,
            'context': {'event_type': 'state', 'values': values, 'metadata': {'episode_id': ep_id, 'step': step}},
            'allow_exploration': True, 'selection_mode': 'auto'}
    body['return_fields'] = (['selection', 'decision_id', 'ranked_hypotheses', 'learning_state', 'latent_belief'] if diag
                             else ['selection', 'decision_id'])
    for attempt in range(3):
        try:
            return api.post(f'/domains/{did}/query', body)
        except (Ambiguous, ApiError) as e:
            if isinstance(e, ApiError) and e.code < 500:
                raise
            stats['query_retries'] += 1; time.sleep(2 * (attempt + 1))
    raise SystemExit(f'query failed 3 times at {ep_id} step {step}')


def feedback(fb):
    try:
        return api.post(f'/domains/{did}/feedback', fb)
    except (Ambiguous, ApiError) as e:
        if isinstance(e, ApiError) and e.code < 500:
            raise
        # outcome unknown: resend once; the decision_id binds it, and a duplicate would show as a second feedback for one decision
        stats['feedback_reconciled'] += 1; time.sleep(2)
        return api.post(f'/domains/{did}/feedback', fb)


def compact_diag(q, ep):
    ranked = q.get('ranked_hypotheses') or []
    ls = q.get('learning_state') or {}
    subs = ls.get('subsystems') or {}
    fp = subs.get('feedback_policy') or {}
    seq = subs.get('sequential') or subs.get('sequential_policy') or {}
    return {'episode': ep, 'status': (q.get('selection') or {}).get('status'), 'selected': (q.get('selection') or {}).get('selected_policy'),
            'selected_by': (q.get('selection') or {}).get('selected_by'),
            'per_policy': {h.get('policy'): {k: (h.get('contextual_policy') or {}).get(k) for k in
                           ('expected_reward', 'selection_expected_reward', 'sequential_expected_reward', 'model_weight', 'out_of_distribution', 'effective_observations')}
                           for h in ranked if h.get('policy')},
            'feedback_policy_samples': fp.get('sample_count'), 'feedback_policy_model': (fp.get('model') or {}).get('status'),
            'sequential': {k: seq.get(k) for k in ('status', 'installed', 'validation_skill', 'current_model_type', 'reason', 'episodes', 'sample_count') if k in seq},
            'subsystem_keys': sorted(subs)}


try:
    for ep in range(EPISODES):
        ep_id = f'{tag}-e{ep:05d}'
        ts = env.reset(); step = 0; rewards = []; fallbacks = 0
        values = envs.obs_values(ts.observation)
        while not ts.last():
            diag = (step == 0 and ep % 50 == 0)
            q = query(values, ep_id, step, diag)
            sel = q.get('selection') or {}
            chosen = sel.get('selected_policy')
            if chosen in POLICIES:
                executed = chosen
            else:
                executed = POLICIES[rng.randrange(n_actions)]; fallbacks += 1
            ts = env.step(POLICIES.index(executed))
            r = float(ts.reward); rewards.append(r)
            next_values = envs.obs_values(ts.observation)
            fb = {'session_id': 'ignored', 'outcome': 'success', 'feedback_kind': 'execution', 'relation': REL, 'policy': executed,
                  'values': {'step_reward': r, 'next_state': {'values': next_values}, 'terminal': bool(ts.last()), 'delay_steps': 0},
                  'metadata': {'episode_id': ep_id, 'step': step}}
            if q.get('decision_id'):
                fb['decision_id'] = q['decision_id']
            res = feedback(fb)
            if (res.get('credit_assignment') or {}).get('contextual_learning_applied') is not True:
                stats['not_applied'] += 1
            if diag:
                d = compact_diag(q, ep); d['feedback_credit'] = res.get('credit_assignment'); diag_log.write(json.dumps(d) + '\n')
            values = next_values; step += 1; stats['steps'] += 1
        stats['fallbacks'] += fallbacks
        row = {'episode': ep, **envs.episode_metrics(ENV, rewards), 'len': len(rewards), 'fallbacks': fallbacks,
               'elapsed_s': round(time.time() - t0, 1), 'api_calls': api.calls}
        ep_log.write(json.dumps(row) + '\n')
        if stats['steps'] > 50 and stats['not_applied'] > 0.01 * stats['steps']:
            raise SystemExit(f'abort: {stats["not_applied"]} of {stats["steps"]} feedbacks not applied to contextual learning')
    outcome = 'completed'
except SystemExit as e:
    outcome = f'aborted: {e}'
finally:
    deleted = api.delete_domain(did)
    summary = {'env': ENV, 'seed': SEED, 'episodes_requested': EPISODES, 'outcome': locals().get('outcome', 'crashed'), 'domain_id': did,
               'stats': stats, 'api_calls': api.calls, 'api_seconds': round(api.seconds, 1), 'wall_s': round(time.time() - t0, 1), 'delete': deleted}
    (OUT / f'summary-{tag}.json').write_text(json.dumps(summary, indent=1))
    api.close()
    print(json.dumps(summary))
