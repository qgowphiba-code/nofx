"""Pre-registered metrics (PREREG.md) for round 2, computed from the raw run files. Tolerates runs that are still in progress.
usage: analyze.py [RUNS_DIR] [GPEI_CSV_DIR]   -> writes results/ROUND2_RESULTS.md and results/round2_metrics.json
"""
import sys, json, glob, os, csv, statistics as st, math
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'runs')
GPEI = sys.argv[2] if len(sys.argv) > 2 else None
OUT = os.path.join(HERE, 'results'); os.makedirs(OUT, exist_ok=True)
M = {}
L = []


def say(s=''):
    L.append(s)


def rows(path):
    """JSONL file -> list of dicts; a single pretty-printed JSON object (summary-*.json, heldout-*.json) -> [obj]."""
    try:
        txt = open(path).read()
    except FileNotFoundError:
        return []
    try:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]
    except json.JSONDecodeError:
        return [json.loads(txt)] if txt.strip() else []


def fmt(x, nd=1):
    return '-' if x is None or (isinstance(x, float) and math.isnan(x)) else (f'{x:.{nd}f}' if isinstance(x, float) else str(x))


def running_mean_first_reach(per_ep_means, ep_lens, target, window=50):
    """First cumulative step at which the running mean over the last `window` episodes >= target."""
    steps = 0
    for i, (m, n) in enumerate(zip(per_ep_means, ep_lens)):
        steps += n
        if i + 1 >= window and st.mean(per_ep_means[i + 1 - window:i + 1]) >= target:
            return steps
    return None


def sign_test(a, b):
    """Two-sided exact sign test on paired lists a, b (ties dropped). Returns (better, worse, p)."""
    better = sum(1 for x, y in zip(a, b) if x > y); worse = sum(1 for x, y in zip(a, b) if x < y)
    n = better + worse
    if n == 0:
        return better, worse, 1.0
    k = min(better, worse)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return better, worse, min(1.0, p)


# ---------------------------------------------------------------- A. Industrial Benchmark
say('# Round 2 results (computed by analyze.py from the raw files)\n')
say('## A. Industrial Benchmark, sequential route\n')
M['ib'] = {}
for arm, label in (('p50', 'stationary setpoint 50'), ('p50d', 'drifting setpoint')):
    say(f'### {label}\n')
    a1 = {}
    for seed in range(3):
        eps = rows(os.path.join(RUNS, 'ib', 'adapt1', f'episodes-ib-{arm}-s{seed}.jsonl'))
        summ = rows(os.path.join(RUNS, 'ib', 'adapt1', f'summary-ib-{arm}-s{seed}.json'))
        if not eps:
            continue
        means = [e['mean_reward'] for e in eps]
        last50 = st.mean(means[-50:]) if len(eps) >= 50 else None
        diag = rows(os.path.join(RUNS, 'ib', 'adapt1', f'diagnostics-ib-{arm}-s{seed}.jsonl'))
        seq = (diag[-1].get('sequential') if diag else None) or {}
        a1[seed] = {'episodes': len(eps), 'steps': eps[-1]['steps'], 'primary_last50': last50, 'whole_run': st.mean(means),
                    'first50': st.mean(means[:50]) if len(eps) >= 50 else st.mean(means), 'fallback_rate': sum(e['fallbacks'] for e in eps) / max(1, eps[-1]['steps']),
                    'noop_frac': st.mean(e['noop_frac'] for e in eps[-50:]), 'curve_blocks10': [round(st.mean(means[i:i + 10]), 1) for i in range(0, len(means), 10)],
                    'sequential_diag': seq, 'outcome': summ[0]['outcome'] if summ else 'in progress'}
    M['ib'][arm] = {'adapt1': a1}
    say('| Adapt-1 seed | episodes | first 50 eps | last 50 eps (primary) | whole run | fallback | no-op frac | sequential model | status |')
    say('|---|---:|---:|---:|---:|---:|---:|---|---|')
    for s, v in a1.items():
        sq = v['sequential_diag']
        say(f"| {s} | {v['episodes']} | {fmt(v['first50'])} | {fmt(v['primary_last50'])} | {fmt(v['whole_run'])} | {v['fallback_rate']:.3f} | {v['noop_frac']:.2f} | {sq.get('current_model_type') or sq.get('status') or '-'} / skill {fmt(sq.get('validation_skill'), 3)} | {v['outcome']} |")
    target = st.mean(v['primary_last50'] for v in a1.values() if v['primary_last50'] is not None) if any(v['primary_last50'] is not None for v in a1.values()) else None
    say(f"\nAdapt-1 primary (mean of seeds): {fmt(target)}\n")
    say('| comparator | seed | steps run | matched window (steps 20k-30k) | asymptote (last 50 eps) | first reaches Adapt-1 primary at step | whole first 30k |')
    say('|---|---:|---:|---:|---:|---:|---:|')
    comp = {}
    for f in sorted(glob.glob(os.path.join(RUNS, 'ib', 'baselines', f'episodes-*-{arm}-s*.jsonl'))):
        name = os.path.basename(f)[len('episodes-'):-len('.jsonl')]
        algo, _, seedtag = name.split('-'); seed = int(seedtag[1:])
        eps = rows(f)
        if not eps:
            continue
        means = [e['mean_reward'] for e in eps]; lens = [e['len'] for e in eps]; cum = []
        c = 0
        for n in lens:
            c += n; cum.append(c)
        window = [m for m, cs in zip(means, cum) if 20000 < cs <= 30000]
        first30 = [m for m, cs in zip(means, cum) if cs <= 30000]
        comp[name] = {'steps': cum[-1], 'matched_20k_30k': st.mean(window) if window else None, 'asymptote_last50': st.mean(means[-50:]) if len(means) >= 50 else None,
                      'reach_step': running_mean_first_reach(means, lens, target) if target is not None else None, 'first_30k': st.mean(first30) if first30 else None,
                      'curve_blocks50': [round(st.mean(means[i:i + 50]), 1) for i in range(0, len(means), 50)]}
        say(f"| {algo} | {seed} | {cum[-1]} | {fmt(comp[name]['matched_20k_30k'])} | {fmt(comp[name]['asymptote_last50'])} | {fmt(comp[name]['reach_step'])} | {fmt(comp[name]['first_30k'])} |")
    M['ib'][arm]['comparators'] = comp
    say()

# ---------------------------------------------------------------- B. Beer
say('## B. SMPL BeerFMT, Machina\n')
say('### B1 acquisition (nominal batch): best quality after n attempts (return = 600 q - 400)\n')
NS = [16, 32, 64, 128, 256, 600]
say('| method | run | ' + ' | '.join(f'n={n}' for n in NS) + ' | best return |')
say('|---|---|' + '---:|' * (len(NS) + 1))
M['beer'] = {'acq': {}}
for d in sorted(glob.glob(os.path.join(RUNS, 'beer', 'machina-beer-*'))):
    at = rows(os.path.join(d, 'attempts.jsonl'))
    if not at:
        continue
    best = [a['best'] for a in at]
    vals = [best[min(n, len(best)) - 1] if len(best) >= 1 else None for n in NS]
    M['beer']['acq'][os.path.basename(d)] = {'curve': dict(zip(NS, vals)), 'attempts': len(at)}
    say(f"| Machina | {os.path.basename(d)[len('machina-beer-'):]} | " + ' | '.join(fmt(v, 3) for v in vals) + f" | {fmt(best[-1] * 600 - 400)} |")
loc = rows(os.path.join(RUNS, 'beer', 'local_acquire.jsonl'))
bym = defaultdict(list)
for r in loc:
    bym[r['method']].append(r)
for m, rs in bym.items():
    curves = [[r['best_curve'][min(n, len(r['best_curve'])) - 1] for n in NS] for r in rs]
    mean = [st.mean(c[i] for c in curves) for i in range(len(NS))]
    rng_ = [(min(c[i] for c in curves), max(c[i] for c in curves)) for i in range(len(NS))]
    M['beer']['acq'][m] = {'seeds': len(rs), 'mean_curve': dict(zip(NS, mean)), 'range_curve': dict(zip(NS, rng_))}
    say(f"| {m} | mean of {len(rs)} seeds | " + ' | '.join(f'{v:.3f} [{lo:.2f}-{hi:.2f}]' for v, (lo, hi) in zip(mean, rng_)) + f" | {fmt(mean[-1] * 600 - 400)} |")
say('| best constant 14.5 C | - | ' + ' | '.join(['0.780'] * len(NS)) + ' | 68 |')
say('| CMA-ES 8,000 evals (gate reference) | - | ' + ' | '.join(['-'] * len(NS)) + ' | 120 |')
say()
say('Deep-RL comparators (closed loop, SMPL protocol, one episode = one batch):\n')
say('| algo | seed | episodes | steps | running-50 mean return at 600 episodes | final running-50 mean | first episode reaching return 116 | held-out mean return (32 batches) |')
say('|---|---:|---:|---:|---:|---:|---:|---:|')
M['beer']['sb3'] = {}
for f in sorted(glob.glob(os.path.join(RUNS, 'beer', 'sb3', 'episodes-*.jsonl'))):
    name = os.path.basename(f)[len('episodes-'):-len('.jsonl')]
    eps = rows(f)
    if not eps:
        continue
    rets = [e['return'] for e in eps]
    r600 = st.mean(rets[550:600]) if len(rets) >= 600 else None
    final = st.mean(rets[-50:]) if len(rets) >= 50 else None
    reach = next((i + 1 for i in range(49, len(rets)) if st.mean(rets[i - 49:i + 1]) >= 116), None)
    ho = rows(os.path.join(RUNS, 'beer', 'sb3', f'heldout-{name}.json'))
    hov = ho[0]['heldout_mean_return'] if ho else None
    M['beer']['sb3'][name] = {'episodes': len(eps), 'steps': eps[-1]['steps'], 'r600': r600, 'final': final, 'reach116_episode': reach, 'heldout_mean_return': hov,
                             'heldout_per_context': ho[0]['heldout_returns'] if ho else None}
    say(f"| {name.split('-')[0]} | {name.split('-')[1][1:]} | {len(eps)} | {eps[-1]['steps']} | {fmt(r600)} | {fmt(final)} | {fmt(reach)} | {fmt(hov)} |")
say()
say('### B2 refinement: held-out (32 unseen batches), mean SMPL return; base alone and per-batch ceiling from the local side\n')
lr = rows(os.path.join(RUNS, 'beer', 'local_refine.jsonl'))
base = next((r for r in lr if r['what'] == 'base_on_heldout'), None)
ceil = next((r for r in lr if r['what'] == 'ceiling_on_heldout'), None)
const = next((r for r in lr if r['what'] == 'best_constant_on_heldout'), None)
M['beer']['refine'] = {'base': base, 'ceiling': ceil, 'constant': const, 'arms': {}, 'local': {}}


def gap(v):
    if base is None or ceil is None or v is None:
        return None
    return (v - base['mean_return']) / (ceil['mean_return'] - base['mean_return'])


say(f"base alone: {fmt(base['mean_return']) if base else '-'} (finished {fmt(base['finished_frac'], 2) if base else '-'}); per-batch ceiling: {fmt(ceil['mean_return']) if ceil else '-'}; best constant per batch: {fmt(const['mean_return']) if const else '-'}\n")
say('| arm / learner | executions | after 5 | 10 | 20 | 30 updates (held-out mean return) | finished frac | gap recovered | paired vs base (better/worse, p) |')
say('|---|---:|---:|---:|---:|---:|---:|---:|---|')
for d in sorted(glob.glob(os.path.join(RUNS, 'beer', 'refine-beer-*'))):
    ho = rows(os.path.join(d, 'heldout.jsonl'))
    if not ho:
        continue
    byu = {h['update']: h for h in ho}
    last = ho[-1]
    bw = sign_test(last['per_context'], base['per_context']) if base else None
    M['beer']['refine']['arms'][os.path.basename(d)] = {'heldout': {u: h['heldout_mean_return'] for u, h in byu.items()}, 'last_update': last['update'], 'finished_frac': last['finished_frac'],
                                                       'gap_recovered': gap(last['heldout_mean_return']), 'per_context': last['per_context'], 'sign_vs_base': bw}
    say(f"| Machina {os.path.basename(d)[len('refine-beer-'):]} | {last['update'] * 256} | " + ' | '.join(fmt(byu[u]['heldout_mean_return']) if u in byu else '-' for u in (5, 10, 20, 30)) +
        f" | {last['finished_frac']:.2f} | {fmt(gap(last['heldout_mean_return']), 2)} | {bw[0]}/{bw[1]}, p={bw[2]:.3f} |" if bw else ' | - |')
for r in lr:
    if r['what'] in ('contextual_cmaes', 'contextual_nes', 'robust_profile'):
        bw = sign_test(r['per_context'], base['per_context']) if base else None
        M['beer']['refine']['local'][f"{r['what']}:{r['seed']}"] = {'mean_return': r['mean_return'], 'finished_frac': r['finished_frac'], 'gap_recovered': gap(r['mean_return']), 'sign_vs_base': bw}
        say(f"| {r['what']} seed {r['seed']} | {r['executions']} | - | - | - | {fmt(r['mean_return'])} | {r['finished_frac']:.2f} | {fmt(gap(r['mean_return']), 2)} | " + (f"{bw[0]}/{bw[1]}, p={bw[2]:.3f} |" if bw else '- |'))
for name, v in M['beer']['sb3'].items():
    if v['heldout_mean_return'] is not None:
        bw = sign_test(v['heldout_per_context'], base['per_context']) if base else None
        say(f"| {name} (closed loop, trained {v['steps']} steps) | {v['steps']} | - | - | - | {fmt(v['heldout_mean_return'])} | {sum(x > 0 for x in v['heldout_per_context']) / 32:.2f} | {fmt(gap(v['heldout_mean_return']), 2)} | " + (f"{bw[0]}/{bw[1]}, p={bw[2]:.3f} |" if bw else '- |'))
# paired Machina vs the learned comparators
say('\nPaired sign tests on the 32 held-out batches, Machina linear 0.2 (final) vs each comparator (better/worse, p):\n')
arm = M['beer']['refine']['arms'].get('refine-beer-linear-r0.2')
if arm:
    for r in lr:
        if r['what'] in ('contextual_cmaes', 'contextual_nes', 'robust_profile'):
            bw = sign_test(arm['per_context'], r['per_context']); say(f"- vs {r['what']} seed {r['seed']}: {bw[0]}/{bw[1]}, p={bw[2]:.3f}")
    for name, v in M['beer']['sb3'].items():
        if v['heldout_per_context']:
            bw = sign_test(arm['per_context'], v['heldout_per_context']); say(f"- vs {name}: {bw[0]}/{bw[1]}, p={bw[2]:.3f}")
    for other in ('refine-beer-linear-r0.08', 'refine-beer-shared-r0.2'):
        o = M['beer']['refine']['arms'].get(other)
        if o:
            bw = sign_test(arm['per_context'], o['per_context']); say(f"- vs Machina {other[len('refine-beer-'):]}: {bw[0]}/{bw[1]}, p={bw[2]:.3f}")
say()

# ---------------------------------------------------------------- C. PenSim
say('## C. SMPL PenSimEnv, Machina acquisition (nominal batch seed_ref 1000); quality = (return + 100) / 4600\n')
NSP = [16, 32, 64, 128, 300]
say('| method | run | ' + ' | '.join(f'n={n}' for n in NSP) + ' | best yield (kg) | failed batches |')
say('|---|---|' + '---:|' * (len(NSP) + 2))
M['pensim'] = {'acq': {}}
for d in sorted(glob.glob(os.path.join(RUNS, 'pensim', 'machina-pensim-*'))):
    at = rows(os.path.join(d, 'attempts.jsonl'))
    if not at:
        continue
    best = [a['best'] for a in at]
    vals = [best[min(n, len(best)) - 1] for n in NSP]
    failed = sum(1 for a in at if a['steps'] < 1150)
    M['pensim']['acq'][os.path.basename(d)] = {'curve': dict(zip(NSP, vals)), 'attempts': len(at), 'failed': failed, 'best_yield': best[-1] * 4600 - 100}
    say(f"| Machina | {os.path.basename(d)[len('machina-pensim-'):]} ({len(at)} attempts) | " + ' | '.join(fmt(v, 3) for v in vals) + f" | {fmt(best[-1] * 4600 - 100, 0)} | {failed}/{len(at)} |")
lp = rows(os.path.join(RUNS, 'pensim', 'local_acquire.jsonl'))
for r in lp:
    if r['method'] == 'default_recipe':
        M['pensim']['default_recipe'] = r
        say(f"| default recipe (light control) | seeds 1-5 | " + ' | '.join(['-'] * len(NSP)) + f" | {fmt(st.mean(r['yields']), 0)} (nominal batch {fmt(r['nominal_seed_ref_yield'], 0)}) | - |")
    else:
        bc = r['best_curve']; vals = [bc[min(n, len(bc)) - 1] for n in NSP]
        M['pensim']['acq'][f"{r['method']}:{r['seed']}"] = {'curve': dict(zip(NSP, vals)), 'best_yield': r['best_q'] * 4600 - 100}
        say(f"| {r['method']} | seed {r['seed']} | " + ' | '.join(fmt(v, 3) for v in vals) + f" | {fmt(r['best_q'] * 4600 - 100, 0)} | - |")
if GPEI and os.path.isdir(GPEI):
    ys = []
    for f in glob.glob(os.path.join(GPEI, 'gpei_batch_*.csv')):
        ys.append(sum(float(row['Yield Per Step']) for row in csv.DictReader(open(f))))
    ys.sort()
    M['pensim']['gpei_dataset'] = {'n': len(ys), 'mean': st.mean(ys), 'median': st.median(ys), 'p90': ys[int(0.9 * len(ys))], 'max': ys[-1]}
    say(f"| GP-EI dataset (smpl-experiments, {len(ys)} batches, mixed seeds) | - | " + ' | '.join(['-'] * len(NSP)) + f" | mean {st.mean(ys):.0f}, p90 {ys[int(0.9 * len(ys))]:.0f}, max {ys[-1]:.0f} | - |")
say()
open(os.path.join(OUT, 'ROUND2_RESULTS.md'), 'w').write('\n'.join(L) + '\n')
json.dump(M, open(os.path.join(OUT, 'round2_metrics.json'), 'w'), indent=1, default=str)
print('\n'.join(L))
