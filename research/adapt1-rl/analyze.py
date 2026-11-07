"""Pre-registered metrics (PREREG.md) for round 1, from the raw per-episode files. usage: analyze.py [RUNS_DIR]"""
import sys, json, glob, os, statistics as st
from collections import defaultdict
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
ENVS = ['bandit', 'catch', 'deep_sea10', 'umbrella10', 'memory5']
AGENTS = ['adapt1', 'dqn', 'boot_dqn', 'actor_critic', 'actor_critic_rnn', 'random']


def load():
    runs = defaultdict(dict)                                     # (env, agent) -> seed -> rows
    for f in glob.glob(f'{RUNS}/adapt1/episodes-*.jsonl'):
        env, seed = os.path.basename(f)[9:-6].rsplit('-s', 1)
        runs[(env, 'adapt1')][int(seed)] = [json.loads(l) for l in open(f)]
    for f in glob.glob(f'{RUNS}/baselines/episodes-*.jsonl'):
        stem = os.path.basename(f)[9:-6]
        env = next(e for e in ENVS if stem.startswith(e + '-'))
        agent, seed = stem[len(env) + 1:].rsplit('-s', 1)
        runs[(env, agent)][int(seed)] = [json.loads(l) for l in open(f)]
    return runs


def metrics(env, rows):
    last = rows[-100:]
    if env == 'bandit':
        return {'regret_all': st.mean(r['regret'] for r in rows), 'regret_last100': st.mean(r['regret'] for r in last)}
    m = {'success_all': st.mean(float(r['success']) for r in rows), 'success_last100': st.mean(float(r['success']) for r in last)}
    if env == 'deep_sea10':
        m['first_treasure'] = next((r['episode'] for r in rows if r['success']), None)
    return m


def fmt(vals, pct=False):
    vals = [v for v in vals if v is not None]
    if not vals:
        return '-'
    f = (lambda v: f'{100 * v:.0f}%') if pct else (lambda v: f'{v:.3f}')
    return f'{f(st.mean(vals))} [{f(min(vals))}-{f(max(vals))}]' if len(vals) > 1 else f(vals[0])


def main():
    runs = load(); out = []
    expected = {'bandit': 1000}
    for env in ENVS:
        n_ep = expected.get(env, 500)
        out.append(f'\n### {env}\n')
        keys = ['regret_all', 'regret_last100'] if env == 'bandit' else ['success_all', 'success_last100'] + (['first_treasure'] if env == 'deep_sea10' else [])
        out.append('| agent | seeds (complete) | ' + ' | '.join(keys) + ' |')
        out.append('|---|---|' + '---|' * len(keys))
        for agent in AGENTS:
            seeds = runs.get((env, agent), {})
            if not seeds:
                continue
            complete = {s: rows for s, rows in seeds.items() if len(rows) >= n_ep}
            ms = [metrics(env, rows[:n_ep]) for rows in complete.values()]
            cells = []
            for k in keys:
                if k == 'first_treasure':
                    v = [m[k] for m in ms]
                    never = sum(x is None for x in v)
                    cells.append((fmt([x for x in v if x is not None]) if any(x is not None for x in v) else '-') + (f' (never in {never})' if never else ''))
                else:
                    cells.append(fmt([m[k] for m in ms], pct=k.startswith('success')))
            out.append(f'| {agent} | {len(seeds)} ({len(complete)}) | ' + ' | '.join(cells) + ' |')
        # learning curve, blocks of 50 episodes, mean across seeds
        key = 'regret' if env == 'bandit' else 'success'
        out.append(f'\nBlocks of 50 episodes, mean {key} across seeds:\n')
        for agent in AGENTS:
            seeds = runs.get((env, agent), {})
            if not seeds:
                continue
            blocks = []
            for b in range(0, n_ep, 50):
                vals = [st.mean(float(r[key]) for r in rows[b:b + 50]) for rows in seeds.values() if len(rows) >= b + 50]
                blocks.append(f'{st.mean(vals):.2f}' if vals else '.')
            out.append(f'- {agent}: ' + ' '.join(blocks))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
