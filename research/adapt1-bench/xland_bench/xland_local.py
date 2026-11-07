"""Local comparators for the XLand macro-action test on the same rulesets and layout seeds as the Adapt-1 runs:
random (uniform over applicable macros), explorer (systematic no-knowledge script with replay), oracle (knows goal and rules).
usage: xland_local.py TIER FIRST N SEED OUT.jsonl
"""
import sys, json, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import xland_macro as X

TIER, FIRST, N, SEED, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
bench = X.load_tier(TIER); env = X.MacroXLand(); rng = random.Random(SEED)


def applicable(e):
    present = set(e.classes_present()); pocket = e.pocket(); out = []
    for i, c in enumerate(e.slots):
        if c is None or c not in present:
            continue
        out.append(f'goto_{i}'); out.append(f'pick_{i}' if pocket[0] == 0 else f'put_{i}')
    if pocket[0] != 0:
        out.append('drop')
    return out or ['drop']


with open(OUT, 'a') as fh:
    for idx in range(FIRST, FIRST + N):
        rs = bench.get_ruleset(idx); seed = SEED * 100000 + idx; goal, rules = X.decode_ruleset(rs)
        res = {}
        res['random'] = X.play_episode(env, rs, seed, lambda e: rng.choice(applicable(e)))
        ex = None
        def mk_ex(e):
            global ex
            if ex is None: ex = X.SystematicExplorer(e)
            return ex.act()
        ex = None; res['explorer'] = X.play_episode(env, rs, seed, mk_ex, on_result=lambda a, r, ok, te: ex.observe(r, te))
        env.start(rs, seed); orc = X.Oracle(env, rs); res['oracle'] = X.play_episode(env, rs, seed, lambda e: orc.act())
        fh.write(json.dumps({'ruleset': idx, 'goal_type': goal['type'], 'n_rules': len(rules), 'n_objects': env.n_init,
                             **{f'{k}_trials': v['trial_rewards'] for k, v in res.items()}, **{f'{k}_macros': v['macros'] for k, v in res.items()}}) + '\n')
        if idx % 25 == 0:
            print(idx, {k: [round(x, 2) for x in v['trial_rewards']] for k, v in res.items()}, flush=True)
