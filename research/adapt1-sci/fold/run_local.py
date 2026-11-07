"""Local comparators for Exp C: random sequences, tabular Monte-Carlo control (same context, same episodes)."""
import sys, json, random
sys.path.insert(0, 'fold')
import oracle, lattice2d as L
N = L.N; NTRAIN = int(sys.argv[1]); SEEDS = int(sys.argv[2])
order = list(range(20)); random.Random(5).shuffle(order)
plan = [('train', order[e % 20], e) for e in range(NTRAIN)] + [('heldout', 20 + i // 2, NTRAIN + i) for i in range(10)]
def key(t, i, prev, with_pos):
    d = oracle.BANK[t]['degree'][i]; k = (d, 1 if i in (0, N - 1) else 0, prev)
    return (i,) + k if with_pos else k
for method in ('random', 'mc_tab', 'mc_tab_pos'):
    for s in range(SEEDS):
        r = random.Random(s); Q = {}
        for phase, t, e in plan:
            seq, prev, visited = '', 0, []
            for i in range(N):
                if method == 'random':
                    a = r.choice('HP')
                else:
                    k = key(t, i, prev, method == 'mc_tab_pos')
                    qs = {a: Q.get((k, a), (0.0, 0))[0] for a in 'HP'}
                    if phase == 'train' and (r.random() < 0.1 or all(Q.get((k, a), (0, 0))[1] == 0 for a in 'HP')):
                        a = r.choice('HP')
                    else:
                        a = max('HP', key=lambda x: (qs[x], r.random()))
                    visited.append((k, a))
                seq += a; prev = 1 if a == 'H' else 0
            rew = oracle.reward(seq, t)
            if phase == 'train':
                for ka in visited:
                    m, n = Q.get(ka, (0.0, 0)); Q[ka] = (m + (rew - m) / (n + 1), n + 1)
            print(json.dumps({'method': method, 'seed': s, 'phase': phase, 'episode': e, 'target': t, 'reward': rew}))
