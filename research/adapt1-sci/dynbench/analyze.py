import json, glob, os, statistics as st, sys
sys.path.insert(0, 'common'); sys.path.insert(0, 'dynbench')
from stats import perm_test
import problems
REF = {}
for p in glob.glob('runs/dyn_verify_*.json'):
    d = json.load(open(p)); REF[d['problem']] = (d['best_J'], d['random_J_mean'])
NS = (16, 32, 64, 128, 256, 400, 600)
for pb in ('batch', 'catmix', 'parkram'):
    ref, rmean = REF[pb]; lit = problems.PROBLEMS[pb]['J_lit']
    print(f'\n### {pb}: published optimum {lit}; 20-stage reference {ref:.5g}; random-profile mean {rmean:.5g}')
    print('| method | ' + ' | '.join(f'n={n}' for n in NS) + ' | gap closed @600 |'); print('|---|' + '---:|' * (len(NS) + 1))
    loc = [json.loads(l) for l in open(f'runs/dyn_local_{pb}.jsonl')]
    finals = {}
    for m in ('random', 'es11', 'cmaes'):
        r = [x['curve'] for x in loc if x['method'] == m]
        v = [st.mean(x[str(n)] for x in r) for n in NS]; finals[m] = [x['600'] for x in r]
        print(f'| {m} (10 seeds) | ' + ' | '.join(f'{a:.5g}' for a in v) + f' | {(v[-1] - rmean) / (ref - rmean):.2f} |')
    mach = []
    for p in sorted(glob.glob(f'runs/dyn/machina-{pb}-*/attempts.jsonl')):
        rows = [json.loads(l) for l in open(p)]
        if len(rows) < 600: continue
        b = [r['best'] for r in rows]; v = [b[n - 1] for n in NS]; mach.append(v[-1])
        print(f'| Machina {os.path.basename(os.path.dirname(p)).split(pb + "-")[1]} | ' + ' | '.join(f'{a:.5g}' for a in v) + f' | {(v[-1] - rmean) / (ref - rmean):.2f} |')
    if mach:
        for m in ('es11', 'cmaes'):
            d, pv = perm_test(mach, finals[m])
            print(f'Machina ({len(mach)} runs) - {m} (10 seeds) at 600: {d:+.4g}, permutation p = {pv:.3f}')
