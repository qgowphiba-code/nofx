import json, glob, statistics as st, sys, os
sys.path.insert(0, 'cryst')
NS = (16, 32, 64, 128, 256, 400)
ceil = {}
for p in glob.glob('runs/cryst/ceiling2_c*.json'):
    d = json.load(open(p)); ceil[d['compound']] = d['best_known']
head = {}
for p in glob.glob('runs/cryst_headroom_*.txt'):
    for l in open(p):
        if l.startswith('{'):
            d = json.loads(l.split('} ')[0] + '}' if not l.strip().endswith('}') else l); head[d['compound']] = d
local = [json.loads(l) for l in open('runs/cryst/local_baselines.jsonl')]
out = []
for c in (1, 4, 9):
    cb = ceil.get(c, head[c]['ceiling'])
    lin, cub = head[c]['linear'], head[c]['cubic']
    print(f'\n## compound {c}: linear {lin:.3f}  cubic {cub:.3f}  best-known {cb:.4f}')
    print('| method | ' + ' | '.join(f'n={n}' for n in NS) + ' | gap closed @400 |')
    print('|---|' + '---:|' * (len(NS) + 1))
    for m in ('random', 'es11', 'cmaes'):
        r = [x['curve'] for x in local if x['compound'] == c and x['method'] == m]
        vals = [st.mean(x[str(n)] for x in r) for n in NS]
        lo, hi = min(x['400'] for x in r), max(x['400'] for x in r)
        print(f'| {m} (10 seeds, mean) | ' + ' | '.join(f'{v:.3f}' for v in vals) + f' | {(vals[-1]-lin)/(cb-lin):.2f} (seeds {lo:.3f}-{hi:.3f}) |')
    for p in sorted(glob.glob(f'runs/cryst/machina-c{c}-*/attempts.jsonl')):
        rows = [json.loads(l) for l in open(p)]
        if len(rows) < 400: continue
        best = [r['best'] for r in rows]
        vals = [best[n - 1] for n in NS]
        name = os.path.basename(os.path.dirname(p)).replace(f'machina-c{c}-', 'Machina ')
        rep = sum(1 for i, r in enumerate(rows) if any(r['u'] == rows[j]['u'] for j in range(i))) / len(rows)
        print(f'| {name} | ' + ' | '.join(f'{v:.3f}' for v in vals) + f' | {(vals[-1]-lin)/(cb-lin):.2f} (exact repeats {rep:.0%}) |')
