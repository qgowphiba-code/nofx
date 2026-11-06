"""Fixed settings from the plan: task encoding (2.1), fit settings (Appendix B), seed registry (Appendix A)."""

HOST = 'fast-uat'   # production rejects this environment's key (401); Fast-UAT runs af6670b (set 1's build)
RELATION = 'hp_terminal_utility'
POLICY = 'hp_state_value'
MAX_SAMPLES = 25000

# Appendix B: object-graph base (set-1 options)
OBJECT_BASE = {
    'model_type': 'object_value', 'dimensions': 32, 'object_hidden_dimensions': 32,
    'epochs': 120, 'learning_rate': 0.003, 'loss_function': 'squared', 'supervision_scope': 'all',
    'trainable_layers': 'all', 'dense_inputs': False, 'pair_geometry': True, 'peak_pooling': True,
    'quadratic_pairs': True, 'preserve_binary_inputs': True, 'validation_mode': 'episode_hash',
    'selection_objective': 'prediction', 'comparison_weight': 0,
}
# Appendix B: projected representation (A1 arm R2)
PROJECTED_BASE = {**{k: v for k, v in OBJECT_BASE.items() if k not in ('object_hidden_dimensions', 'peak_pooling')},
                  'model_type': 'mlp_v2', 'dimensions': 64, 'dense_inputs': True}
# Appendix B: comparison arms (BLG, ROB)
COMPARISON = {'comparison_weight': 1, 'comparison_loss': 'squared', 'selection_objective': 'prediction'}


def fit_options(base=OBJECT_BASE, warm_start=False, validation_episode_ids=None, **over):
    opts = {**base, 'warm_start': warm_start, **over}
    if validation_episode_ids is not None:
        opts['validation_episode_ids'] = sorted(validation_episode_ids)
    return opts


def value_config(representation='object_graph'):
    return {'relation': RELATION, 'policy': POLICY, 'representation': representation, 'max_samples': MAX_SAMPLES}


# Appendix A: seed registry (new seeds in the 80000s; set 1 used 5..77001)
SEEDS = {
    'set1_chains': 61107,          # sequences(61107, [10,11]*9): [:12] training, [12:] evaluation
    'set1_eval_states': 61108,     # grading.py: 4 random folds per evaluation chain
    'eval_s_chains': 80001,        # 18 new Eval-S chains (set-1 chains excluded)
    'eval_s_states': 80002,
    'eval_l_chains': 80101,        # [36..64 step 4]*5, panel sequences and reversals excluded
    'b1_chains': 80200,            # + lineage
    'b1_random_folds': 80300,      # + lineage
    'b1_collectors': 80400,        # + 10*lineage + generation
    'a1_chains': 80501,
    'a1_folds': 80502,
    'a3_draws': (61107, 80601, 80602),
    'validation': 80700,           # + fit index
    'twod_short': 81001,
    'twod_long': 81101,
    'game': 82001,                 # + draw
    't02_probe': 80900,            # not in Appendix A: T0.2 scratch data only
}
VALIDATION_FRACTION = 0.2
