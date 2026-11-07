"""Adapt-1 Domain declaration for experiment-level reaction optimization (contextual policy route)."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim

RELATION = 'run_condition'

def policy_id(k):
    c, T, tau = sim.CONDITIONS[k]
    return f'{sim.CATALYSTS[c]}_t{int(T)}_tau{int(tau)}'

def declaration(domain_id, context=True, exploration='thompson', action_transfer=0.3):
    hyps = []
    for k, (c, T, tau) in enumerate(sim.CONDITIONS):
        feats = {'catalyst': sim.CATALYSTS[c], 'temperature': (T - 60.0) / 30.0, 'log_residence': (math.log(tau) - math.log(8.0)) / math.log(4.0)}
        hyps.append({'name': f'run {policy_id(k)}', 'relation': RELATION, 'policy': policy_id(k), 'policy_features': feats,
                     'when': [], 'predicts': ['measured yield of the target product'], 'falsified_by': [], 'weight': 1.0})
    learning = {
        'enabled': True,
        'context': {'feature_paths': ['values.d1', 'values.d2', 'values.d3'] if context else ['values.bias'],
                    'event_types': ['state'], 'max_samples': 4096},
        'reward': {'aggregation': 'weighted_mean', 'components': [
            {'field': 'values.yield', 'goal': 'maximize', 'min': 0.0, 'max': 1.0, 'weight': 1.0, 'required': True}]},
        'policy': {'decay_half_life_seconds': 0.0, 'transfer_strength': 0.0, 'action_transfer_strength': action_transfer,
                   'min_context_observations': 4, 'model_max_weight': 0.45, 'model_ood_threshold': 0.75, 'ood_confidence_penalty': 0.5,
                   'exploration_mode': exploration, 'exploration_strength': 1.0, 'minimum_expected_reward': 0.0, 'minimum_confidence': 0.0,
                   'require_credible_dominance': False, 'dominance_margin': 0.0, 'abstain_on_ood': False},
        'credit_assignment': {'mode': 'none', 'neutral_reward': 0.5},
        'latent_belief': {'enabled': True, 'max_regimes': 8, 'novelty_threshold': 0.55, 'temperature': 0.25,
                          'change_point_threshold': 0.55, 'change_point_patience': 3, 'historical_regime_weight': 0.05},
        'training': {'enabled': True, 'min_samples': 64, 'retrain_interval': 40, 'dimensions': 96, 'epochs': 100, 'learning_rate': 0.04}}
    return {'domain_id': domain_id, 'session_id': 'ignored',
            'description': 'Reaction-condition optimization: choose one experiment condition per query; reward = measured yield.',
            'schema': {'entities': ['reactor', 'substrate'], 'relations': [RELATION], 'signals': ['d1', 'd2', 'd3', 'yield'],
                       'event_types': ['state', 'feedback']},
            'hypotheses': hyps, 'learning': learning}
