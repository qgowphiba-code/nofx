"""C1: compute-matched comparison (plan section 7), mostly local.

Contacts on Eval-L against leaf evaluations per decision:
  - contact heuristic at depths 1-5: contacts from the frozen baseline card (100 seeds), leaf evaluations
    per decision measured here (one fold per chain; counts barely depend on tie-breaks);
  - the best learned model of each B1 lineage at depths 1-3 and its leaf blend (from B1/B2 results).
Claim form: "at N leaf evaluations per decision, learned X vs heuristic Y".
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hpx import suites as SU  # noqa: E402
from hpx.planner import Heuristic, run_folds  # noqa: E402


def main():
    seqs = SU.load('eval_l')['sequences']
    card = SU.load('baselines_eval_l')['heuristic']
    out = {'heuristic': {}}
    for d in (1, 2, 3, 4, 5):
        res = run_folds(seqs, d, Heuristic(), rngs=[random.Random(i) for i in range(len(seqs))])
        leaves = sum(r['leaf_evaluations'] for r in res) / sum(r['decisions'] for r in res)
        out['heuristic'][d] = {'leaves_per_decision': leaves, 'mean_total': card[str(d)]['mean_total'],
                               'sd_total': card[str(d)]['sd_total']}
        print(f'heuristic depth {d}: {leaves:8.1f} leaves/decision, Eval-L total {card[str(d)]["mean_total"]}')
    out['learned'] = {}
    for p in sorted((ROOT / 'runs' / 'B1').glob('*/results_l*.json')):
        r = json.loads(p.read_text())
        for arm, a in r.get('arms', {}).items():
            fin = a.get('final')
            if not fin:
                continue
            key = f'l{r["lineage"]}-{arm}'
            out['learned'][key] = {}
            for d in (1, 2, 3):
                e = fin[f'eval_l_depth{d}']
                dec = sum(len(s) - 2 for s in seqs)
                out['learned'][key][d] = {'leaves_per_decision': sum(e['leaf_evaluations']) / dec, 'total': e['total']}
            e = fin['blend_test_eval_l']
            out['learned'][key]['blend'] = {'lam_leaf': fin['lam_leaf'], 'total': e['total'],
                                            'leaves_per_decision': sum(e['leaf_evaluations']) / sum(len(s) - 2 for s in seqs)}
            print(key, {d: (round(v['leaves_per_decision'], 1), v['total']) for d, v in out['learned'][key].items()})
    (ROOT / 'runs' / 'C1').mkdir(parents=True, exist_ok=True)
    (ROOT / 'runs' / 'C1' / 'results.json').write_text(json.dumps(out, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
