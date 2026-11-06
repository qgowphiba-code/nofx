"""C-eval: evaluation suites (plan 2.2), built once, frozen as JSON with sha256, and scored here.

Eval-S   24 held-out short chains (L10-11) with exact oracle grading: set 1's 6 evaluation chains with
         set 1's own states (Random(61108), as vendor grading.py) + 18 new chains (Random(80002)).
Eval-L   40 long chains, sequences(80101, [36..64 step 4]*5), Liu panel sequences and reversals excluded.
Eval-Liu the 7 Liu panel chains.
2D suites (Track D) are built by the same functions with dim=2.
"""
import hashlib
import json
import random
import statistics as st
from pathlib import Path

from .hpenv import Fold
from .oracle import Oracle
from .planner import decide, run_folds
from .vendor import lattice_env, liu_panel

ROOT = Path(__file__).resolve().parent.parent
SUITES = ROOT / 'suites'
EVAL_L_LENGTHS = [36, 40, 44, 48, 52, 56, 60, 64] * 5


def set1_chains():
    allc = lattice_env.sequences(61107, [10, 11] * 9)
    return allc[:12], allc[12:]


def eval_s_chains():
    train1, eval1 = set1_chains()
    keys = {lattice_env.sequence_key(s) for s in train1 + eval1}
    new = lattice_env.sequences(80001, [10, 11] * 9, excluded=keys)
    return eval1, new


def eval_l_chains(seed=80101, lengths=EVAL_L_LENGTHS):
    panel = {lattice_env.sequence_key(s) for _, s in liu_panel()}
    return lattice_env.sequences(seed, lengths, excluded=panel)


def random_fold_chains(seq, rng, dim=3):
    f = Fold(seq, dim)
    out = [f.key()]
    while not f.terminal:
        f.push(rng.choice(f.legal()))
        out.append(f.key())
    return out


def _decisions_for_chain(ci, seq, rng, dim, n_random=4):
    orc = Oracle(seq, dim)
    paths = [random_fold_chains(seq, rng, dim) for _ in range(n_random)]
    f = Fold(seq, dim)
    opt_path = [f.key()]
    while not f.terminal:
        opt, _ = orc.optimal_moves(f.key())
        f.push(sorted(opt)[0])
        opt_path.append(f.key())
    paths.append(opt_path)
    seen, out = set(), []
    for path in paths:
        for chain in path[:-1]:
            if chain in seen:
                continue
            seen.add(chain)
            opt, kids = orc.optimal_moves(chain)
            vs, vr = orc.values(chain)
            out.append({'ci': ci, 'chain': [list(p) for p in chain], 'vstar': vs, 'vrand': vr,
                        'children': [{'action': list(a), 'vstar': v} for a, _, v in kids],
                        'opt': [list(a) for a in opt]})
    return out, orc.optimum_contacts()


def build_short_suite(groups, dim=3):
    """groups: [(label, seed, [seqs])]; each group's states come from its own Random(seed)."""
    chains, decisions = [], []
    for label, seed, seqs in groups:
        rng = random.Random(seed)
        for seq in seqs:
            ci = len(chains)
            decs, opt = _decisions_for_chain(ci, seq, rng, dim)
            chains.append({'ci': ci, 'sequence': seq, 'group': label, 'optimum_contacts': opt})
            decisions.extend(decs)
    return {'dim': dim, 'chains': chains, 'decisions': decisions}


def build_eval_s():
    eval1, new = eval_s_chains()
    return build_short_suite([('set1', 61108, eval1), ('new', 80002, new)], dim=3)


def freeze(name, obj):
    SUITES.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, sort_keys=True, separators=(',', ':'))
    (SUITES / f'{name}.json').write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    man_path = SUITES / 'MANIFEST.json'
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man[name] = {'sha256': sha, 'bytes': len(text)}
    man_path.write_text(json.dumps(man, indent=1, sort_keys=True))
    return sha


def load(name, verify=True):
    text = (SUITES / f'{name}.json').read_text()
    if verify:
        man = json.loads((SUITES / 'MANIFEST.json').read_text())
        sha = hashlib.sha256(text.encode()).hexdigest()
        if man[name]['sha256'] != sha:
            raise RuntimeError(f'suite {name} sha256 {sha} != frozen {man[name]["sha256"]}')
    return json.loads(text)


# ------------------------------------------------------------------------------------- scoring
def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = sum((a - mx) ** 2 for a in x) ** .5
    sy = sum((b - my) ** 2 for b in y) ** .5
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float('nan')


def score_short(suite, ev, depth=1, groups=None, state_values=True):
    """Best-move rate (ties averaged), regret, and correlation of the state's value with V* / V_rand.

    Returns overall metrics plus per-group (e.g. 'set1') metrics and per-decision outcomes."""
    seqs = {c['ci']: c['sequence'] for c in suite['chains']}
    group_of = {c['ci']: c['group'] for c in suite['chains']}
    decs = [d for d in suite['decisions'] if groups is None or group_of[d['ci']] in groups]
    folds = [Fold(seqs[d['ci']], suite['dim'], d['chain']) for d in decs]
    res = decide(folds, depth, ev)
    per = []
    for d, (legal, vals, _) in zip(decs, res):
        child_v = {tuple(c['action']): c['vstar'] for c in d['children']}
        opt = {tuple(a) for a in d['opt']}
        m = max(vals)
        best = [tuple(a) for a, v in zip(legal, vals) if v == m]
        informative = len(opt) < len(d['children'])
        per.append({'ci': d['ci'], 'group': group_of[d['ci']], 'informative': informative,
                    'hit': sum(a in opt for a in best) / len(best),
                    'regret': sum(d['vstar'] - child_v[a] for a in best) / len(best), 'n_best': len(best)})
    if state_values:
        sv = ev.evaluate([ev.payload(f) for f in folds])
        for p, v in zip(per, sv):
            p['value'] = v

    def summarize(rows, all_rows):
        inf = [r for r in rows if r['informative']]
        out = {'best_move_rate': st.mean(r['hit'] for r in inf), 'mean_regret': st.mean(r['regret'] for r in inf),
               'n_informative': len(inf), 'n_decisions': len(rows)}
        if state_values:
            idx = [i for i, r in enumerate(all_rows) if r in rows]
            out['corr_vstar'] = pearson([all_rows[i]['value'] for i in idx], [decs[i]['vstar'] for i in idx])
            out['corr_vrand'] = pearson([all_rows[i]['value'] for i in idx], [decs[i]['vrand'] for i in idx])
        return out
    summary = {'all': summarize(per, per)}
    for g in sorted(set(group_of.values())):
        rows = [r for r in per if r['group'] == g]
        if rows:
            summary[g] = summarize(rows, per)
    return {'depth': depth, 'summary': summary, 'per_decision': per}


def greedy_short(suite, ev, depth=1, seed=0, groups=None):
    chains = [c for c in suite['chains'] if groups is None or c['group'] in groups]
    rngs = [random.Random(seed * 1000 + c['ci']) for c in chains]
    res = run_folds([c['sequence'] for c in chains], depth, ev, rngs=rngs, dim=suite['dim'])
    return {'greedy_contacts': sum(r['contacts'] for r in res), 'optimal_contacts': sum(c['optimum_contacts'] for c in chains),
            'completed': sum(r['complete'] for r in res),
            'per_chain': [[c['ci'], r['contacts'], c['optimum_contacts']] for c, r in zip(chains, res)]}
