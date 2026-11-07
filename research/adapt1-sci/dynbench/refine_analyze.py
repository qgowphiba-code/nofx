import json, glob, statistics as st, sys
sys.path.insert(0, 'common')
from stats import perm_test
ceil = [json.loads(l) for f in ('runs/dyn_ref_ceiling_a.jsonl', 'runs/dyn_ref_ceiling_b.jsonl') for l in open(f)]
C = {r['j']: r['J'] for r in ceil}; base = {r['j']: r['base_J'] for r in ceil}
cm, bm = st.mean(C.values()), st.mean(base.values())
print(f'held-out (32 unseen batches): base alone {bm:.3f}, per-batch ceiling {cm:.3f}')
print('| arm | after 0 | 5 | 10 | 20 | 30 updates | gap recovered @30 | radius @30 |')
print('|---|---:|---:|---:|---:|---:|---:|---:|')
final = {}
for d in sorted(glob.glob('runs/dyn/refine-parkram-*/')):
    h = {json.loads(l)['update']: json.loads(l) for l in open(d + 'heldout.jsonl')}
    if 30 not in h: continue
    u = [json.loads(l) for l in open(d + 'updates.jsonl')]
    rad = (u[-1]['raw'].get('update') or {}).get('next_action_search_radius')
    arm = d.rstrip('/').split('refine-parkram-')[1]; final[arm] = h[30]['per_context']
    print(f'| Machina {arm} | ' + ' | '.join(f'{h[k]["heldout_mean_J"]:.2f}' for k in (0, 5, 10, 20, 30)) + f' | {(h[30]["heldout_mean_J"] - bm) / (cm - bm):.0%} | {rad} |')
for f, lab in (('runs/dyn_ref_ctx_cmaes.jsonl', 'CMA-ES linear policy'), ('runs/dyn_ref_ctx_es11.jsonl', '(1+1)-ES linear policy')):
    rows = [json.loads(l) for l in open(f)]
    m = st.mean(r['heldout_mean_J'] for r in rows)
    print(f'| {lab} (3 seeds, 7680 executions) | | | | | {m:.2f} | {(m - bm) / (cm - bm):.0%} | |')
    final[lab] = [st.mean(vals) for vals in zip(*[r['per_context'] for r in rows])]
best = max((k for k in final if k.startswith(('linear', 'quadratic', 'shared'))), key=lambda k: st.mean(final[k]))
for other in final:
    if other == best: continue
    diffs = [a - b for a, b in zip(final[best], final[other])]
    print(f'paired over 32 held-out batches: Machina {best} - {other}: {st.mean(diffs):+.3f}, better on {sum(d > 0.01 for d in diffs)}, worse on {sum(d < -0.01 for d in diffs)}')
