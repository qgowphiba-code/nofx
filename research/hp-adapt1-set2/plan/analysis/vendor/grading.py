"""Oracle grading shared by Stage A follow-ups (identical to stage_a.py's evaluation set and metrics)."""
import json, math, random
from hp import State, observation, random_fold, Oracle
from lattice_env import sequences

all_seqs = sequences(61107, [10, 11] * 9)
train_seqs, eval_seqs = all_seqs[:12], all_seqs[12:]
oracles = {ci: Oracle(seq) for ci, seq in enumerate(eval_seqs)}
erng = random.Random(61108)
eval_states = []
for ci, seq in enumerate(eval_seqs):
    orc = oracles[ci]; paths = [random_fold(seq, erng) for _ in range(4)]
    s0 = State.initial(seq); p = [s0]
    while not s0.terminal:
        opt, _ = orc.optimal_moves(s0); s0 = s0.step(sorted(opt)[0]); p.append(s0)
    paths.append(p); seen = set()
    for path in paths:
        for st in path[:-1]:
            if st.chain not in seen: seen.add(st.chain); eval_states.append((ci, st))
grade = []
for ci, st in eval_states:
    o_ = oracles[ci]; opt, vals = o_.optimal_moves(st)
    grade.append({'ci': ci, 'state': st, 'vstar': o_.vstar(st), 'vrand': o_.vrand(st), 'opt': opt, 'child_vstar': vals,
                  'children': [(a, st.step(a)) for a in st.legal]})
optimum = {ci: oracles[ci].vstar(State.initial(seq)) for ci, seq in enumerate(eval_seqs)}


def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float('nan')


def score_from(value_of_children, value_of_states=None):
    acc = []; reg = []
    child_scores = value_of_children([g['children'] for g in grade])
    for g, sc in zip(grade, child_scores):
        if len(g['opt']) == len(g['children']): continue
        m = max(sc); best = [a for (a, _), v in zip(g['children'], sc) if abs(v - m) < 1e-12]
        acc.append(sum(a in g['opt'] for a in best) / len(best))
        reg.append(sum(g['vstar'] - g['child_vstar'][a] for a in best) / len(best))
    out = {'move_accuracy': sum(acc) / len(acc), 'mean_regret': sum(reg) / len(reg), 'n_informative': len(acc)}
    if value_of_states:
        sv = value_of_states([g['state'] for g in grade])
        out['corr_vstar'] = pearson(sv, [g['vstar'] for g in grade]); out['corr_vrand'] = pearson(sv, [g['vrand'] for g in grade])
    return out


def greedy(value_of_children):
    tot = 0; opt = 0; comp = 0; per = []
    for ci, seq in enumerate(eval_seqs):
        s0 = State.initial(seq)
        while not s0.terminal:
            ch = [(a, s0.step(a)) for a in s0.legal]; sc = value_of_children([ch])[0]
            s0 = ch[max(range(len(ch)), key=lambda i: sc[i])][1]
        oc = round((optimum[ci] - .5) * 2 * s0.upper_bound) if optimum[ci] > 0 else 0
        tot += s0.contacts if s0.complete else 0; opt += oc; comp += s0.complete; per.append((s0.contacts if s0.complete else 0, oc))
    return {'greedy_contacts': tot, 'optimal_contacts': opt, 'completed': comp, 'per_chain': per}


def model_scorer(V, source):
    def predict(states):
        outv = []
        for i in range(0, len(states), 128):
            s, o = V('predict', {'inputs': [observation(t) for t in states[i:i + 128]], 'model_source': source})
            if s >= 300: raise RuntimeError(json.dumps(o)[:300])
            outv += o['predictions']
        return outv
    def kids(list_of_children):
        flat = [t for ch in list_of_children for _, t in ch]
        idx = [i for i, t in enumerate(flat) if not t.terminal]
        pv = predict([flat[i] for i in idx]); vals = [t.utility() if t.terminal else None for t in flat]
        for i, v in zip(idx, pv): vals[i] = v
        out = []; k = 0
        for ch in list_of_children: out.append(vals[k:k + len(ch)]); k += len(ch)
        return out
    return kids, predict
