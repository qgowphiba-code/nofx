"""Same-budget black-box comparators for BeerFMT acquisition at the nominal context: random search, (1+1)-ES, CMA-ES over
the same 20-stage command vector (u in [-1, 1]^20, executor BEER-1). Records the best-so-far quality after every attempt.
usage: beer_local_acquire.py METHOD SEED ATTEMPTS OUT.jsonl
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'common'))
import beer_plant as B
import opt

METHOD, SEED, N, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
x0 = B.NOMINAL_X0
hist = []


def f(u):
    q = B.quality(B.run(B.profile_from_u(u), x0)['return'])
    hist.append(q)
    return q


if METHOD == 'random':
    opt.random_search(f, B.H, N, SEED)
elif METHOD == 'es11':
    opt.one_plus_one(f, B.H, N, SEED, sigma=0.3, x0=[0.0] * B.H)
elif METHOD == 'cmaes':
    opt.cmaes(f, B.H, N, SEED, sigma0=0.5, x0=[0.0] * B.H)
else:
    raise SystemExit(METHOD)
hist = hist[:N]
best = [max(hist[:i + 1]) for i in range(len(hist))]
with open(OUT, 'a') as fh:
    fh.write(json.dumps({'method': METHOD, 'seed': SEED, 'attempts': len(hist), 'best_curve': [round(b, 4) for b in best],
                         'best_q': best[-1], 'best_return': best[-1] * 600 - 400}) + '\n')
print(METHOD, SEED, 'best q', round(best[-1], 4), 'return', round(best[-1] * 600 - 400, 1))
