"""Generate XLand-MiniGrid rulesets with the benchmark's own generator (scripts/ruleset_generator.py in the xland-minigrid
repository, imported from the clone), restricted to the three position-free rule/goal families: AgentHold, AgentNear,
TileNear. The released benchmarks also sample the directional variants (TileNearUp/Down/Left/Right, AgentNearUp/...),
which this study leaves out so that a macro-action agent needs no side-specific placements (PREREG section E).
Other sampling parameters follow scripts/generate_benchmarks.sh for the trivial / small / medium tiers; the encoding,
de-duplication, padding and saving mirror the generator's __main__ block.
usage: gen_rulesets.py OUT_DIR N_PER_TIER [SEED]
"""
import os, sys, random, importlib.util
import jax.numpy as jnp
SCRATCH = os.environ.get('ADAPT1_SCRATCH', '/tmp/claude-0/-home-user-nofx/7c289125-8882-5b5c-9bcd-654fe7300b9f/scratchpad')
GEN = os.path.join(SCRATCH, 'repos', 'xland-minigrid', 'scripts', 'ruleset_generator.py')
spec = importlib.util.spec_from_file_location('ruleset_generator', GEN); gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
from xminigrid.core.goals import AgentHoldGoal, AgentNearGoal, TileNearGoal
from xminigrid.core.rules import AgentHoldRule, AgentNearRule, TileNearRule
from xminigrid.benchmarks import save_bz2_pickle

OUT, N = sys.argv[1], int(sys.argv[2])
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 2026


def sample_goal():
    idx = random.randint(0, 2)
    if idx == 0:
        tile = random.choice(gen.HOLD_TILES); return AgentHoldGoal(tile=jnp.array(tile)), (tile,)
    if idx == 1:
        tile = random.choice(gen.NEAR_TILES_LHS); return AgentNearGoal(tile=jnp.array(tile)), (tile,)
    a = random.choice(gen.NEAR_TILES_LHS); b = random.choice(gen.NEAR_TILES_RHS)
    return TileNearGoal(tile_a=jnp.array(a), tile_b=jnp.array(b)), (a, b)


def sample_rule(prod_tile, used_tiles):
    idx = random.randint(0, 2)
    if idx == 0:
        tile = random.choice(gen.diff(gen.HOLD_TILES, used_tiles)); return AgentHoldRule(tile=jnp.array(tile), prod_tile=jnp.array(prod_tile)), (tile,)
    if idx == 1:
        tile = random.choice(gen.diff(gen.HOLD_TILES, used_tiles)); return AgentNearRule(tile=jnp.array(tile), prod_tile=jnp.array(prod_tile)), (tile,)
    a = random.choice(gen.diff(gen.NEAR_TILES_LHS, used_tiles)); b = random.choice(gen.diff(gen.NEAR_TILES_RHS, used_tiles))
    return TileNearRule(tile_a=jnp.array(a), tile_b=jnp.array(b), prod_tile=jnp.array(prod_tile)), (a, b)


gen.sample_goal = sample_goal
gen.sample_rule = sample_rule
pad = gen.pad_along_axis
TIERS = {  # name: dict(chain_depth, num_distractor_rules, num_distractor_objects, sample_depth, sample_distractor_rules, prune_chain, prune_prob)
    'trivial': dict(chain_depth=0, num_distractor_rules=0, num_distractor_objects=3, sample_depth=False, sample_distractor_rules=False, prune_chain=False, prune_prob=0.0),
    'small': dict(chain_depth=1, num_distractor_rules=2, num_distractor_objects=2, sample_depth=False, sample_distractor_rules=True, prune_chain=True, prune_prob=0.3),
    'medium': dict(chain_depth=2, num_distractor_rules=3, num_distractor_objects=2, sample_depth=False, sample_distractor_rules=True, prune_chain=True, prune_prob=0.1),
}
os.makedirs(OUT, exist_ok=True)
for name, kw in TIERS.items():
    random.seed(SEED + len(name))
    rulesets, seen = [], set()
    while len(rulesets) < N:
        rs = gen.sample_ruleset(**kw)
        key = gen.encode(rs)
        if key in seen:
            continue
        seen.add(key)
        rulesets.append({'goal': rs['goal'].encode(), 'rules': jnp.vstack([r.encode() for r in rs['rules']]),
                         'init_tiles': jnp.array(rs['init_tiles'], dtype=jnp.uint8), 'num_rules': jnp.asarray(rs['num_rules'], dtype=jnp.uint8)})
    max_rules = max(len(r['rules']) for r in rulesets); max_tiles = max(len(r['init_tiles']) for r in rulesets)
    concat = {'generation_config': {'tier': name, 'position_free_subset': True, **kw, 'total_rulesets': N, 'random_seed': SEED},
              'goals': jnp.vstack([r['goal'] for r in rulesets]),
              'rules': jnp.vstack([pad(r['rules'], pad_to=max_rules)[None, ...] for r in rulesets]),
              'init_tiles': jnp.vstack([pad(r['init_tiles'], pad_to=max_tiles)[None, ...] for r in rulesets]),
              'num_rules': jnp.vstack([r['num_rules'] for r in rulesets])}
    save_bz2_pickle(concat, os.path.join(OUT, f'{name}_pf'), protocol=-1)
    print(name, len(rulesets), 'rulesets; max rules', max_rules, 'max tiles', max_tiles, flush=True)
