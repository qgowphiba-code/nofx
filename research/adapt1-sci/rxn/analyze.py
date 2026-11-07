import json, statistics as st, glob, os
NS = (1, 3, 5, 10, 20)
BLOCKS = ((0, 10), (10, 20), (20, 40))
def summarize(curves_by_ep):
    out = {}
    for lo, hi in BLOCKS:
        eps = [e for e in curves_by_ep if lo <= e < hi]
        if not eps: continue
        out[(lo, hi)] = [st.mean(st.mean(c[n - 1] for c in curves_by_ep[e]) for e in eps) for n in NS]
    return out
rows = {}
for l in open('runs/rxn/local.jsonl'):
    d = json.loads(l); rows.setdefault(d['method'], {}).setdefault(d['episode'], []).append(d['curve'])
for arm in ('ctx', 'noctx'):
    p = f'runs/rxn/adapt1-{arm}-thompson/episodes.jsonl'
    if os.path.exists(p):
        for l in open(p):
            d = json.loads(l); rows.setdefault(f'adapt1_{arm}', {}).setdefault(d['episode'], []).append(d['curve'])
order = ['random', 'gp_episode', 'gp_transfer', 'adapt1_noctx', 'adapt1_ctx']
for lo, hi in BLOCKS:
    print(f'\n### reactions {lo}-{hi - 1}: best yield found / grid optimum, after n experiments')
    print('| method | ' + ' | '.join(f'n={n}' for n in NS) + ' |'); print('|---|' + '---:|' * len(NS))
    for m in order:
        if m not in rows: continue
        s = summarize(rows[m])
        if (lo, hi) in s:
            nep = len([e for e in rows[m] if lo <= e < hi])
            print(f'| {m} ({nep} rx) | ' + ' | '.join(f'{v:.3f}' for v in s[(lo, hi)]) + ' |')
for m in ('gp_episode', 'gp_transfer', 'adapt1_noctx'):
    for n in (5, 10):
        if 'adapt1_ctx' not in rows or m not in rows: continue
        eps = [e for e in range(20, 40) if e in rows['adapt1_ctx'] and e in rows[m]]
        diffs = [st.mean(c[n - 1] for c in rows['adapt1_ctx'][e]) - st.mean(c[n - 1] for c in rows[m][e]) for e in eps]
        if diffs:
            print(f'paired adapt1_ctx - {m} @n={n}, reactions 20-39: mean {st.mean(diffs):+.3f}, ctx better on {sum(d > 0.005 for d in diffs)}, worse on {sum(d < -0.005 for d in diffs)} of {len(diffs)}')
