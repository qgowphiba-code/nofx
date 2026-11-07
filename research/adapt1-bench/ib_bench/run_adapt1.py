"""Adapt-1 (Fast-UAT) on the Siemens Industrial Benchmark, online: learning on, exploration on, every step scored.

usage: run_adapt1.py SETPOINT DRIFT(0/1) SEED EPISODES OUT_DIR [JOURNAL_DIR] [--smoke]

Declaration: the docs' "Recommended starting declaration" for sequential learning (docs/neuroadapt/sequential-learning,
section 3), with the task-specific fields set as pre-registered in PREREG.md:
  - context = the 6 observed benchmark variables (setpoint, velocity, gain, shift, fatigue, consumption), fixed scaling;
  - temporal context (TCP) over those 6 inputs with maximum_lag 10: the benchmark's consumption responds to the last
    10 steerings and fatigue has hidden state, so the current observation alone is insufficient (docs: temporal-context);
  - policies = the benchmark's own 27 discrete steerings {-1,0,1}^3, policy_features = the three deltas;
    action_transfer_strength 0.05 (docs default; the features have physical meaning);
  - reward = the benchmark's step reward -(3*fatigue + consumption) with declared bounds [-800, -100];
  - sequential learning with discount 0.9 and n_step 10 (the documented 10-step delay in consumption), eligibility
    traces, latent belief on (the drifting-setpoint arm is a regime-change test).
Online loop (docs section 13): query before acting -> execute -> one decision-linked feedback per step with next_state,
terminal and step_reward. Episode IDs are zero-padded so lexicographic order is chronological (docs section 19).
If Adapt-1 abstains or returns no legal action, the benchmark's no-op steering [0,0,0] is executed, feedback credits the
executed action, and the step is counted as a fallback (reported separately).
Fails loudly: any feedback with contextual_learning_applied != true is counted; the run aborts if that exceeds 1 %.
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'common'))
import numpy as np
from client import Client, Ambiguous, ApiError, new_id
import ib_env

SP, DRIFT, SEED, EPISODES, OUT = int(sys.argv[1]), sys.argv[2] == '1', int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
JDIR = Path(sys.argv[6]) if len(sys.argv) > 6 and not sys.argv[6].startswith('--') else OUT
SMOKE = '--smoke' in sys.argv
EP_LEN = 20 if SMOKE else ib_env.EP_LEN
OUT.mkdir(parents=True, exist_ok=True); JDIR.mkdir(parents=True, exist_ok=True)
tag = f'ib-p{SP}{"d" if DRIFT else ""}-s{SEED}' + ('-smoke' if SMOKE else '')
env = ib_env.IB(SP, DRIFT, 'discrete', seed=SEED, ep_len=EP_LEN)
ACTIONS = env.env_action                      # 27 steerings, index 13 = [0,0,0]
NOOP = 13
REL = 'steer'
POLICIES = [f'a{i}' for i in range(len(ACTIONS))]
FIELDS = ib_env.OBS_NAMES

hyps = [{'name': f'steering {a}', 'relation': REL, 'policy': p,
         'policy_features': {'d_velocity': float(a[0]), 'd_gain': float(a[1]), 'd_shift': float(a[2])},
         'when': [], 'predicts': ['step reward'], 'falsified_by': [], 'weight': 1.0} for p, a in zip(POLICIES, ACTIONS)]
learning = {
    'enabled': True,
    'context': {'feature_paths': [f'values.{k}' for k in FIELDS], 'event_types': ['state'], 'max_samples': 4096},
    'reward': {'aggregation': 'weighted_mean', 'components': [{'field': 'values.step_reward', 'goal': 'maximize',
               'min': ib_env.R_MIN, 'max': ib_env.R_MAX, 'weight': 1.0, 'required': True}]},
    'policy': {'decay_half_life_seconds': 0.0, 'transfer_strength': 0.0, 'action_transfer_strength': 0.05, 'min_context_observations': 4,
               'model_max_weight': 0.45, 'model_ood_threshold': 0.75, 'ood_confidence_penalty': 0.5, 'exploration_mode': 'auto',
               'exploration_strength': 1.0, 'minimum_expected_reward': 0.0, 'minimum_confidence': 0.0, 'require_credible_dominance': False,
               'dominance_margin': 0.0, 'abstain_on_ood': False},
    'credit_assignment': {'mode': 'eligibility_trace', 'discount': 0.9, 'neutral_reward': 0.5, 'minimum_weight': 0.01,
                          'delay_path': 'values.delay_steps'},
    'latent_belief': {'enabled': True, 'max_regimes': 8, 'novelty_threshold': 0.55, 'temperature': 0.25, 'change_point_threshold': 0.55,
                      'change_point_patience': 3, 'historical_regime_weight': 0.05},
    'training': {'enabled': True, 'min_samples': 64, 'retrain_interval': 16, 'dimensions': 96, 'epochs': 100, 'learning_rate': 0.04},
    'sequential': {'enabled': True, 'episode_path': 'metadata.episode_id', 'step_path': 'metadata.step',
                   'next_context_path': 'values.next_state', 'reward_path': 'values.step_reward', 'terminal_path': 'values.terminal',
                   'discount': 0.9, 'n_step': 10, 'minimum_episodes': 10, 'minimum_validation_skill': 0.1,
                   'target_sync_interval': 10, 'conservative_weight': 0.05},
    'temporal_context': {'enabled': True, 'episode_path': 'metadata.episode_id', 'step_path': 'metadata.step',
                         'input_paths': [f'values.{k}' for k in FIELDS], 'maximum_lag': 10},
}

RESUME = '--resume' in sys.argv
START_EP, resumed_from = 0, None
if RESUME:
    # Resume on the same Domain after a process/container loss: domain_id from the saved declaration, continue from the
    # episode after the last complete one (the interrupted episode index is skipped and reported; its feedback stays in
    # the Domain's learned state).
    decl_prev = json.loads((OUT / f'declaration-{tag}.json').read_text())
    did = decl_prev['domain_id']
    done_eps = [json.loads(l) for l in open(OUT / f'episodes-{tag}.jsonl') if l.strip()]
    START_EP = (done_eps[-1]['episode'] + 2) if done_eps else 1
    resumed_from = {'completed_episodes': len(done_eps), 'skipped_episode_index': START_EP - 1, 'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    api = Client(JDIR / f'journal-{tag}-resume{START_EP}.jsonl.xz')
    assert api.call('GET', f'/domains/{did}', None, mutation=False), 'domain not found for resume'
    ep_log = open(OUT / f'episodes-{tag}.jsonl', 'a', buffering=1)
    diag_log = open(OUT / f'diagnostics-{tag}.jsonl', 'a', buffering=1)
    with open(OUT / f'resumes-{tag}.jsonl', 'a') as fh:
        fh.write(json.dumps(resumed_from) + '\n')
    print('resuming', tag, resumed_from, flush=True)
else:
    api = Client(JDIR / f'journal-{tag}.jsonl.xz')
    did = new_id(f'ib-p{SP}{"d" if DRIFT else ""}-s{SEED}')
    decl = {'domain_id': did, 'session_id': 'ignored',
            'description': f'Industrial Benchmark setpoint {SP}{" drifting" if DRIFT else ""} seed {SEED}: online agent, one decision per step.',
            'schema': {'entities': ['agent', 'plant'], 'relations': [REL], 'signals': FIELDS + ['step_reward'], 'event_types': ['state', 'feedback']},
            'hypotheses': hyps, 'learning': learning}
    echo = api.post('/domains', decl)
    (OUT / f'declaration-{tag}.json').write_text(json.dumps(decl, indent=1))
    (OUT / f'echo-{tag}.json').write_text(json.dumps(echo.get('learning'), indent=1))
    ep_log = open(OUT / f'episodes-{tag}.jsonl', 'w', buffering=1)
    diag_log = open(OUT / f'diagnostics-{tag}.jsonl', 'w', buffering=1)
stats = {'steps': 0, 'fallbacks': 0, 'not_applied': 0, 'query_retries': 0, 'feedback_reconciled': 0}
t0 = time.time()


def query(values, ep_id, step, diag):
    body = {'session_id': 'ignored', 'question': 'Choose the next steering for the current plant state.', 'relation': REL,
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
    # reconciled by the Domain's event_count before any resend (PREREG operational note 3)
    return api.feedback(did, fb, stats)


def compact_diag(q, ep, step):
    ranked = q.get('ranked_hypotheses') or []
    ls = q.get('learning_state') or {}
    subs = ls.get('subsystems') or {}
    fp = subs.get('feedback_policy') or {}
    seq = subs.get('sequential') or subs.get('sequential_policy') or {}
    return {'episode': ep, 'step': step, 'status': (q.get('selection') or {}).get('status'), 'selected': (q.get('selection') or {}).get('selected_policy'),
            'selected_by': (q.get('selection') or {}).get('selected_by'), 'latent_belief': q.get('latent_belief'),
            'per_policy': {h.get('policy'): {k: (h.get('contextual_policy') or {}).get(k) for k in
                           ('expected_reward', 'selection_expected_reward', 'sequential_expected_reward', 'model_weight', 'out_of_distribution', 'effective_observations')}
                           for h in ranked if h.get('policy')},
            'feedback_policy_samples': fp.get('sample_count'), 'feedback_policy_model': (fp.get('model') or {}),
            'sequential': {k: seq.get(k) for k in ('status', 'installed', 'validation_skill', 'current_model_type', 'reason', 'episodes', 'sample_count') if k in seq},
            'subsystem_keys': sorted(subs)}


try:
    for ep in range(START_EP, EPISODES + (START_EP - len(resumed_from and done_eps or []) if RESUME else 0)):
        ep_id = f'{tag}-e{ep:05d}'
        obs = env.reset(); step = 0; rewards = []; fallbacks = 0; setpoints = []; chosen_idx = []
        values = ib_env.scaled(obs)
        done = False
        while not done:
            diag = (step in (0, EP_LEN // 2) and (ep % 10 == 0 or SMOKE))
            q = query(values, ep_id, step, diag)
            sel = q.get('selection') or {}
            chosen = sel.get('selected_policy')
            if chosen in POLICIES:
                executed = chosen
            else:
                executed = POLICIES[NOOP]; fallbacks += 1
            a_idx = POLICIES.index(executed)
            obs, r, done, info = env.step(a_idx)
            rewards.append(r); setpoints.append(float(obs[0])); chosen_idx.append(a_idx)
            next_values = ib_env.scaled(obs)
            fb = {'session_id': 'ignored', 'outcome': 'success', 'feedback_kind': 'execution', 'relation': REL, 'policy': executed,
                  'values': {'step_reward': float(np.clip(r, ib_env.R_MIN, ib_env.R_MAX)), 'next_state': {'values': next_values},
                             'terminal': bool(done), 'delay_steps': 0},
                  'metadata': {'episode_id': ep_id, 'step': step}}
            if q.get('decision_id'):
                fb['decision_id'] = q['decision_id']
            res = feedback(fb)
            if res.get('reconciled') != 'applied' and (res.get('credit_assignment') or {}).get('contextual_learning_applied') is not True:
                stats['not_applied'] += 1
            if diag:
                d = compact_diag(q, ep, step); d['feedback_credit'] = res.get('credit_assignment'); diag_log.write(json.dumps(d) + '\n')
            values = next_values; step += 1; stats['steps'] += 1
        stats['fallbacks'] += fallbacks
        blocks = [float(np.mean(rewards[i:i + 20])) for i in range(0, len(rewards), 20)]
        row = {'episode': ep, 'mean_reward': float(np.mean(rewards)), 'blocks20': blocks, 'len': len(rewards), 'fallbacks': fallbacks,
               'setpoint_start': setpoints[0], 'setpoint_end': setpoints[-1], 'noop_frac': float(np.mean([i == NOOP for i in chosen_idx])),
               'distinct_actions': len(set(chosen_idx)), 'steps': stats['steps'], 'elapsed_s': round(time.time() - t0, 1), 'api_calls': api.calls}
        ep_log.write(json.dumps(row) + '\n')
        if stats['steps'] > 50 and stats['not_applied'] > 0.01 * stats['steps']:
            raise SystemExit(f'abort: {stats["not_applied"]} of {stats["steps"]} feedbacks not applied to contextual learning')
    outcome = 'completed'
except SystemExit as e:
    outcome = f'aborted: {e}'
except KeyboardInterrupt:
    outcome = 'interrupted'
finally:
    deleted = api.delete_domain(did)
    summary = {'setpoint': SP, 'drift': DRIFT, 'seed': SEED, 'episodes_requested': EPISODES, 'ep_len': EP_LEN,
               'outcome': locals().get('outcome', 'crashed'), 'domain_id': did, 'stats': stats, 'api_calls': api.calls,
               'api_seconds': round(api.seconds, 1), 'wall_s': round(time.time() - t0, 1), 'delete': deleted, 'resumed_from': resumed_from}
    (OUT / f'summary-{tag}.json').write_text(json.dumps(summary, indent=1))
    api.close()
    print(json.dumps(summary))
