import json, statistics as st
loc = [json.loads(l) for l in open('runs/fold_local.jsonl')]
ad = [json.loads(l) for l in open('runs/fold/adapt1-seq/episodes.jsonl')]
bank = json.load(open('fold/bank.json'))['targets']
def block(rows, ph, lo, hi):
    v = [x['reward'] for x in rows if x['phase'] == ph and lo <= x['episode'] < hi]
    return st.mean(v) if v else float('nan')
print('| method | train ep 0-19 | train ep 20-39 | train ep 40-59 | train ep 60-79 | held-out (frozen) |')
print('|---|---:|---:|---:|---:|---:|')
for m in ('random', 'mc_tab', 'mc_tab_pos'):
    r = [x for x in loc if x['method'] == m]
    print(f'| {m} (10 seeds) | ' + ' | '.join(f'{block(r, "train", a, a + 20):.3f}' for a in (0, 20, 40, 60)) + f' | {block(r, "heldout", 0, 999):.3f} |')
print(f'| Adapt-1 sequential | ' + ' | '.join(f'{block(ad, "train", a, a + 20):.3f}' for a in (0, 20, 40, 60)) + f' | {block(ad, "heldout", 0, 999):.3f} |')
held = range(20, 25)
print('burial heuristic (diagnostic) held-out mean %.3f; exact ceiling held-out mean %.3f; all-target ceiling %.3f' % (
    st.mean(bank[t]['burial_heuristic'] for t in held), st.mean(bank[t]['ceiling'] for t in held), st.mean(b['ceiling'] for b in bank)))
from collections import Counter
print('Adapt-1 selected_by (train):', Counter(s for x in ad if x['phase'] == 'train' for s in x['selected_by']).most_common(5))
print('Adapt-1 held-out sequences:', [(x['target'], x['seq'], round(x['reward'], 3)) for x in ad if x['phase'] == 'heldout'])
print('Adapt-1 fraction H by contact count, last 20 train episodes:')
for d in (0, 1, 2):
    ch = [x['seq'][i] for x in ad if x['phase'] == 'train' and x['episode'] >= 60 for i in range(12) if bank[x['target']]['degree'][i] == d]
    if ch: print('  contacts', d, 'H fraction %.2f (n=%d)' % (ch.count('H') / len(ch), len(ch)))
