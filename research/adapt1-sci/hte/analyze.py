import json, statistics as st, glob, os, sys
NS = (1, 5, 10, 20, 30)
for ds in ('aryl_amination', 'suzuki'):
    loc, seen = [], set()
    for f in sorted(glob.glob(f'runs/hte_local_{ds}*.jsonl')):
        for l in open(f):
            d = json.loads(l); key = (d['method'], d['order'], d['seed'], d['episode'])
            if key not in seen: seen.add(key); loc.append(d)
    rows = {}   # method -> (order, episode) -> list of curves
    for d in loc: rows.setdefault(d['method'], {}).setdefault((d['order'], d['episode']), []).append(d['curve'])
    for p in sorted(glob.glob(f'runs/hte/adapt1-{ds}-o*/episodes.jsonl')):
        o = int(p.split('-o')[-1].split('/')[0])
        for l in open(p):
            d = json.loads(l); rows.setdefault('adapt1', {}).setdefault((o, d['episode']), []).append(d['curve'])
    orders = sorted({o for (o, e) in rows.get('adapt1', {})})
    nsub = 15 if ds == 'aryl_amination' else 12; third = nsub // 3
    for lo, hi, lab in ((0, third, 'first third of the order'), (third, nsub, 'remaining two thirds')):
        print(f'\n### {ds}, substrates in the {lab} (orders {orders}): best yield found / best recorded, after n experiments')
        print('| method | ' + ' | '.join(f'n={n}' for n in NS) + ' |'); print('|---|' + '---:|' * len(NS))
        for m in ('random', 'gp_substrate', 'gp_transfer', 'adapt1'):
            keys = [k for k in rows.get(m, {}) if k[0] in orders and lo <= k[1] < hi]
            if not keys: continue
            vals = [st.mean(st.mean(c[n - 1] for c in rows[m][k]) for k in keys) for n in NS]
            print(f'| {m} ({len(keys)} subs) | ' + ' | '.join(f'{v:.3f}' for v in vals) + ' |')
    for m in ('gp_substrate', 'gp_transfer'):
        for n in (5, 10):
            keys = [k for k in rows.get('adapt1', {}) if k in rows.get(m, {}) and k[1] >= third]
            if not keys: continue
            diffs = [st.mean(c[n - 1] for c in rows['adapt1'][k]) - st.mean(c[n - 1] for c in rows[m][k]) for k in keys]
            print(f'paired adapt1 - {m} @n={n} (later substrates): mean {st.mean(diffs):+.3f}; adapt1 better {sum(d > 0.01 for d in diffs)}, worse {sum(d < -0.01 for d in diffs)}, of {len(diffs)}')
