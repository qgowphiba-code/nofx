import json, glob, statistics as st, sys
sys.path.insert(0, 'gb1'); import land as L
NS = ['24', '48', '96', '192', '384', '500']
print(f'GB1: global max {L.FMAX:.2f} (FWAA); top-0.1 % threshold {L.TOP1:.2f}; wild type 1.00')
print('| method | runs | ' + ' | '.join(f'best@{n}' for n in NS) + ' | found FWAA |'); print('|---|---:|' + '---:|' * (len(NS) + 1))
for f in sorted(glob.glob('runs/gb1_local_*.jsonl')):
    rows = [json.loads(l) for l in open(f)]
    if not rows: continue
    vals = [st.mean(r['best'][n] for r in rows) if all(n in r['best'] for r in rows) else float('nan') for n in NS]
    print(f'| {rows[0]["method"]} | {len(rows)} | ' + ' | '.join('-' if v != v else f'{v:.2f}' for v in vals) + f' | {sum(r["found_global"] for r in rows)}/{len(rows)} |')
for f in sorted(glob.glob('runs/gb1/adapt1-*/designs.jsonl')):
    rows = [json.loads(l) for l in open(f)]; b = {r['n']: r['best'] for r in rows}
    unm = sum(not r['measured'] for r in rows); uniq = len({r['variant'] for r in rows})
    print(f'| Adapt-1 {f.split("adapt1-")[1].split("/")[0]} | 1 | ' + ' | '.join(f'{b[int(n)]:.2f}' if int(n) in b else '-' for n in NS) +
          f' | {"yes" if any(r["variant"] == "FWAA" for r in rows) else "no"} |  ({len(rows)} designs, {uniq} unique, {unm} unmeasured)')
