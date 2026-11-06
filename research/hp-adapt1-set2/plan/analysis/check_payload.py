"""Offline: size of the planning/adversarial request body built exactly as stage_m.py/stage_d.py build it,
with json.dumps defaults (what safe_client sends) vs compact separators. Max over every decision of a depth-3
contact-heuristic fold of each Liu chain. argv[1] = data/liu_panel.json."""
import json, random, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / 'vendor'))
from hp import observation
from lattice_env import State, plan

ACT = lambda a: ','.join(map(str, a))
CAP = 1 << 20


def tree(s, depth):
    nodes = []; counter = [0]
    def add(st, dleft):
        nid = f'n{counter[0]}'; counter[0] += 1
        node = {'id': nid, 'maximizing': True, 'terminal_value': st.utility() if st.terminal else None, 'children': [], **observation(st)}
        nodes.append(node)
        if not st.terminal and dleft > 0:
            for a in st.legal: node['children'].append({'action': ACT(a), 'target': add(st.step(a), dleft - 1)})
        return nid
    add(s, depth); return nodes


def body(s, depth):
    nodes = tree(s, depth)
    return {'session_id': 'hpcomm-prior-20261106100739-46a61a', 'relation': 'hp_terminal_utility', 'policy': 'hp_state_value',
            'evaluator_source': 'value_model', 'expected_model_sha256': '0' * 64,
            'root': {k: v for k, v in nodes[0].items() if k != 'children'}, 'nodes': nodes,
            'options': {'max_depth': depth, 'max_evaluations': 4096, 'max_transitions': 20000, 'max_extensions': 0, 'extension_mode': 'all',
                        'branch_width': None, 'recheck_pruned_branches': False, 'cache_values': False, 'reuse_tree': False,
                        'temporal_discount': 1., 'neutral_value': .5, 'terminal_proof_bounds': None, 'exploration_probability': 0., 'timeout_seconds': 20}}, len(nodes)


panel = json.load(open(sys.argv[1]))['cases']
for c in panel:
    seq = c['sequence']; s0 = State.initial(seq); worst = {}
    while not s0.terminal:
        for depth in (3,):
            b, n = body(s0, depth)
            d = len(json.dumps(b).encode()); k = len(json.dumps(b, separators=(',', ':')).encode())
            if d > worst.get(depth, (0,))[0]: worst[depth] = (d, k, n, len(s0.chain))
        s0 = s0.step(plan(s0, 3, 'C2', random.Random(len(s0.chain)))[0])
    d, k, n, at = worst[3]
    obs_one = len(json.dumps(observation(State.initial(seq)), separators=(',', ':')).encode())
    print(f"{c['case_id']} L{len(seq)} depth3 worst body: default {d/1024:.0f} KiB ({d/CAP:.2f} of cap), compact {k/1024:.0f} KiB ({k/CAP:.2f}),"
          f" nodes {n} at placed={at}; one compact observation ~{obs_one} B; predict batch of 128 compact ~{128*obs_one/1024:.0f} KiB")
