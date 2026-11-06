"""Statistics and pre-registered decision rules (plan 2.5). Standard library only."""
import math
import random
import statistics as st


def bootstrap_mean_ci(values, n_boot=10000, alpha=0.05, seed=0):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return st.mean(values), means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]


def bootstrap_paired_rate(groups_a, groups_b, n_boot=10000, alpha=0.05, seed=0):
    """Paired difference of best-move rates, resampling chains, then decisions within chains.

    groups_x: {chain_id: [per-decision outcome, ...]} with identical keys and decision order."""
    rng = random.Random(seed)
    keys = sorted(groups_a)

    def rate(sample):
        tot = cnt = 0
        for k, idx in sample:
            for i in idx:
                tot += groups_a[k][i] - groups_b[k][i]
                cnt += 1
        return tot / cnt
    full = [(k, list(range(len(groups_a[k])))) for k in keys]
    point = rate(full)
    diffs = []
    for _ in range(n_boot):
        sample = []
        for _ in keys:
            k = keys[rng.randrange(len(keys))]
            m = len(groups_a[k])
            sample.append((k, [rng.randrange(m) for _ in range(m)]))
        diffs.append(rate(sample))
    diffs.sort()
    return point, diffs[int(alpha / 2 * n_boot)], diffs[int((1 - alpha / 2) * n_boot) - 1]


def wilcoxon_signed_rank(d):
    """Two-sided Wilcoxon signed-rank test (zeros dropped, average ranks for ties, normal approximation
    with tie correction; exact enumeration when n <= 20 and there are no ties)."""
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0:
        return {'n': 0, 'W_plus': 0.0, 'p': 1.0}
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    tie_term = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    mean = n * (n + 1) / 4
    if n <= 20 and tie_term == 0:
        # exact null distribution of W+ over 2^n sign patterns
        counts = {0: 1}
        for k in range(1, n + 1):
            nxt = dict(counts)
            for s, c in counts.items():
                nxt[s + k] = nxt.get(s + k, 0) + c
            counts = nxt
        total = 2 ** n
        lo = sum(c for s, c in counts.items() if s <= min(w_plus, n * (n + 1) / 2 - w_plus))
        p = min(1.0, 2 * lo / total)
    else:
        var = n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48
        z = (w_plus - mean - math.copysign(0.5, w_plus - mean)) / math.sqrt(var) if var > 0 else 0.0
        p = math.erfc(abs(z) / math.sqrt(2))
    return {'n': n, 'W_plus': w_plus, 'p': p}


def holm(pvalues):
    """Holm-adjusted p-values, same order as input."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def paired_long(model, reference, n_boot=10000, seed=0):
    """Per-chain deltas model - reference: mean, chain-bootstrap 95% CI, Wilcoxon, win rate."""
    d = [a - b for a, b in zip(model, reference)]
    mean, lo, hi = bootstrap_mean_ci(d, n_boot, seed=seed)
    w = wilcoxon_signed_rank(d)
    return {'mean_delta': mean, 'ci95': [lo, hi], 'wilcoxon_p': w['p'], 'n': len(d),
            'win_rate': sum(x > 0 for x in d) / len(d), 'loss_rate': sum(x < 0 for x in d) / len(d)}


def decision_rule(lineage_deltas, n_boot=10000, seed=0):
    """Plan 2.5 confirmatory criterion: average chain-level deltas across lineages; CI must exclude 0 and
    the same sign must hold in >= 2 of 3 lineages. lineage_deltas: [[per-chain delta], ...] (same chains)."""
    n_chains = len(lineage_deltas[0])
    avg = [sum(l[i] for l in lineage_deltas) / len(lineage_deltas) for i in range(n_chains)]
    res = paired_long(avg, [0.0] * n_chains, n_boot, seed)
    signs = [math.copysign(1, st.mean(l)) if st.mean(l) != 0 else 0 for l in lineage_deltas]
    sign = math.copysign(1, res['mean_delta']) if res['mean_delta'] else 0
    res['lineage_means'] = [st.mean(l) for l in lineage_deltas]
    res['same_sign_lineages'] = sum(s == sign for s in signs)
    res['ci_excludes_zero'] = res['ci95'][0] > 0 or res['ci95'][1] < 0
    res['holds'] = res['ci_excludes_zero'] and res['same_sign_lineages'] >= min(2, len(lineage_deltas))
    return res
