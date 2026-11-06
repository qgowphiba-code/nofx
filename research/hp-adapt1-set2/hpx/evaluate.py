"""Evaluation helpers shared by the stage scripts: bind a served model to the planner and grade it."""
import random

from .hpenv import BASIC
from .planner import Blend, Heuristic, Model, run_folds
from . import suites as SU


def model_evaluator(client, domain, source, sha, conditions=BASIC, name=None):
    """Leaf evaluator backed by value/predict on (domain, source) bound to `sha`."""
    def predict(inputs):
        return client.predict(domain, inputs, source, sha)
    return Model(predict, conditions, name=name or f'{source}:{sha[:10]}')


def grade_short(ev, suite, depths=(1,), greedy=True, greedy_depths=None):
    out = {}
    for d in depths:
        r = SU.score_short(suite, ev, d, state_values=(d == 1))
        out[f'depth{d}'] = r['summary']
        out[f'per_decision_depth{d}'] = [[p['ci'], p['hit'], round(p['regret'], 6)] + ([round(p['value'], 6)] if 'value' in p else [])
                                         for p in r['per_decision'] if p['informative']]
        if greedy and (greedy_depths is None or d in greedy_depths):
            out[f'greedy_depth{d}'] = {'all': SU.greedy_short(suite, ev, d, seed=0),
                                       'set1': SU.greedy_short(suite, ev, d, seed=0, groups=('set1',))}
    return out


def long_eval(ev, seqs, depth=3, seed=0, dim=3, eps=0.0):
    """Depth-d greedy folds of every chain (lockstep, batched leaves). Random tie-breaks, Random(seed*1000+i)."""
    rngs = [random.Random(seed * 1000 + i) for i in range(len(seqs))]
    res = run_folds(seqs, depth, ev, rngs=rngs, dim=dim, eps=eps)
    return {'total': sum(r['contacts'] for r in res), 'per_chain': [r['contacts'] for r in res],
            'complete': [r['complete'] for r in res], 'leaf_evaluations': [r['leaf_evaluations'] for r in res],
            'tie_decisions': [r['tie_decisions'] for r in res]}


__all__ = ['model_evaluator', 'grade_short', 'long_eval', 'Blend', 'Heuristic']
