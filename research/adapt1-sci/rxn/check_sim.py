import sys, statistics
sys.path.insert(0, 'rxn')
import sim
from collections import Counter
best_by_fam = {f: Counter() for f in range(4)}
rows = []
for i in range(80):
    rx = sim.make_reaction(i); g = sim.grid(rx)
    b = max(range(48), key=lambda k: g[k])
    best_by_fam[rx['family']][b] += 1
    srt = sorted(g, reverse=True)
    rows.append((rx['family'], srt[0], statistics.mean(g), srt[4]))
for f in range(4):
    r = [x for x in rows if x[0] == f]
    print('family', f, 'n', len(r), 'max %.2f mean %.2f 5th %.2f' % tuple(statistics.mean(x[k] for x in r) for k in (1, 2, 3)),
          'argmax conds', [(sim.CONDITIONS[k], n) for k, n in best_by_fam[f].most_common(4)])
