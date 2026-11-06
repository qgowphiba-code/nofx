"""Gate G1: fix r*, f*, lam*, d* and the sibling choice from A1, A2 and A3; write manifests/track_b.json.

Rules (pre-registered in the stage scripts):
  r*      A1: cheapest representation with Eval-Liu depth-3 total >= the heuristic's p5 and sibling
          agreement >= 0.95 (R0 kept if it qualifies); otherwise the best by Liu total.
  f*      A2: best Eval-S depth-1 rate; if the spread is < 0.03, the simplest (cold).
  lam*,d* A3: best BL arm by Eval-S depth-1 rate averaged over the 3 draws.
  siblings A3: BLG beats that BL arm by >= 0.03 on the same average.
  fallback A3: if no BL arm beats A0 by >= 0.03, Track B runs A0 against ROB-style targets.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from a3_targets import summarize  # noqa: E402


def latest(pattern):
    hits = sorted((ROOT / 'runs').glob(pattern))
    return hits[-1] if hits else None


def main():
    a1 = json.loads(latest('A1/*/results.json').read_text())
    a2p = latest('A2/*/results.json')
    a2 = json.loads(a2p.read_text()) if a2p else None
    draws = sorted((ROOT / 'runs' / 'A3').glob('*/results_draw*.json'))
    by_draw = {}
    for p in draws:
        by_draw[p.name] = p        # latest run wins per draw
    a3 = summarize(list(by_draw.values()))
    r_star = a1['decision']['r_star']
    rep = {'R0': ('object_graph', 'basic'), 'R1': ('object_graph', 'features'), 'R2': ('projected', 'features')}[r_star]
    f_star = a2['decision']['f_star'] if a2 else 'cold'
    overrides = {'cold': {}, 'cold_240': {'epochs': 240}}.get(f_star)
    lam, d = a3['lam_d_star'].strip('BL()').split(',')
    out = {'r_star': r_star, 'representation': rep[0], 'conditions': rep[1], 'f_star': f_star,
           'fit_overrides': overrides if overrides is not None else {}, 'lam_star': float(lam), 'd_star': int(d),
           'use_siblings': a3['use_siblings'], 'bl_beats_a0': a3['bl_beats_a0_by_0.03'],
           'a3_mean_rates_depth1': a3['mean_rate_depth1'], 'a3_draw_files': [str(p.relative_to(ROOT)) for p in by_draw.values()],
           'a1_run': str(latest('A1/*/results.json').relative_to(ROOT)), 'a2_run': str(a2p.relative_to(ROOT)) if a2p else None,
           'needs_attention': []}
    if overrides is None:
        out['needs_attention'].append(f'f* = {f_star} is a warm setting: B1 needs warm-start plumbing before it runs')
    if a3['use_siblings']:
        out['needs_attention'].append('siblings selected: B1 needs sibling rows (BLG) before it runs')
    if not a3['bl_beats_a0_by_0.03']:
        out['needs_attention'].append('no BL arm beats A0 by 0.03: plan fallback is A0 vs ROB-style targets in Track B')
    (ROOT / 'manifests').mkdir(exist_ok=True)
    (ROOT / 'manifests' / 'track_b.json').write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
