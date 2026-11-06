"""S0: build and freeze the evaluation suites (plan 2.2) with sha256 in suites/MANIFEST.json."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hpx import suites  # noqa: E402
from hpx.planner import Heuristic  # noqa: E402
from hpx.vendor import liu_panel  # noqa: E402


def main(which):
    if 'eval_s' in which:
        t0 = time.time()
        s = suites.build_eval_s()
        sha = suites.freeze('eval_s', s)
        inf = sum(len(d['opt']) < len(d['children']) for d in s['decisions'])
        print(f'eval_s: {len(s["chains"])} chains, {len(s["decisions"])} states, {inf} informative, '
              f'{time.time() - t0:.0f}s, sha256 {sha}', flush=True)
        for d in (1, 2, 3):
            r = suites.score_short(s, Heuristic(), d, state_values=False)['summary']
            print(f'  heuristic depth {d}: all {r["all"]["best_move_rate"]:.3f} (n={r["all"]["n_informative"]}) '
                  f'set1 {r["set1"]["best_move_rate"]:.3f} new {r["new"]["best_move_rate"]:.3f}', flush=True)
    if 'eval_l' in which:
        seqs = suites.eval_l_chains()
        sha = suites.freeze('eval_l', {'dim': 3, 'seed': 80101, 'lengths': suites.EVAL_L_LENGTHS, 'sequences': seqs})
        print(f'eval_l: {len(seqs)} chains, lengths {sorted(set(map(len, seqs)))}, sha256 {sha}', flush=True)
    if 'eval_liu' in which:
        p = liu_panel()
        sha = suites.freeze('eval_liu', {'dim': 3, 'names': [n for n, _ in p], 'sequences': [s for _, s in p]})
        print(f'eval_liu: {len(p)} chains, sha256 {sha}', flush=True)
    if 'eval_l_2d' in which:
        seqs = suites.eval_l_chains(seed=81101)
        sha = suites.freeze('eval_l_2d', {'dim': 2, 'seed': 81101, 'lengths': suites.EVAL_L_LENGTHS, 'sequences': seqs})
        print(f'eval_l_2d: {len(seqs)} chains, sha256 {sha}', flush=True)
    if 'eval_liu_2d' in which:
        p = liu_panel()
        sha = suites.freeze('eval_liu_2d', {'dim': 2, 'names': [n for n, _ in p], 'sequences': [s for _, s in p],
                                            'best_known_contacts': [suites.BEST_2D[n] for n, _ in p]})
        print(f'eval_liu_2d: {len(p)} chains, best-known total {sum(suites.BEST_2D.values())}, sha256 {sha}', flush=True)
    if 'eval_s_2d' in which:
        t0 = time.time()
        s = suites.build_eval_s_2d()
        sha = suites.freeze('eval_s_2d', s)
        inf = sum(len(d['opt']) < len(d['children']) for d in s['decisions'])
        print(f'eval_s_2d: {len(s["chains"])} chains, {len(s["decisions"])} states, {inf} informative, '
              f'{time.time() - t0:.0f}s, sha256 {sha}', flush=True)
        for d in (1, 2, 3):
            r = suites.score_short(s, Heuristic(), d, state_values=False)['summary']['all']
            g = suites.greedy_short(s, Heuristic(), d, seed=0)
            print(f'  heuristic depth {d}: best-move {r["best_move_rate"]:.3f} (n={r["n_informative"]}); greedy '
                  f'{g["greedy_contacts"]}/{g["optimal_contacts"]} ({g["completed"]}/12 complete)', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or ['eval_s', 'eval_l', 'eval_liu'])
