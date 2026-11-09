import json
import os

import numpy as np
import pytest

from machina_finance import tasks as T
from machina_finance.references import best_known, textbook

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST = json.load(open(os.path.join(ROOT, "results", "best_known_eval.json")))
ANALYTIC = ["ac_liquidation", "ow_transient", "vwap_profile", "two_asset_transition", "three_asset_transition"]


@pytest.mark.parametrize("name", sorted(T.TASKS))
def test_naive_scores_zero_and_contexts_deterministic(name):
    t = T.get_task(name)
    C1, C2 = T.standard_contexts(t), T.standard_contexts(t)
    assert [len(C1[k]) for k in ("train", "support", "eval")] == [48, 24, 32]
    assert C1["eval"][5]["feat"] == C2["eval"][5]["feat"]
    c = C1["eval"][0]
    if name != "market_making":  # market making's naive quote is context-dependent (1/k), scored via naive_obj
        assert abs(t.score(t.naive_actions(c), c)) < 1e-12
    assert len(t.feature_names) == t.d == len(c["feat"])


@pytest.mark.parametrize("name", ANALYTIC)
def test_exact_optimum_beats_naive_and_random(name):
    t = T.get_task(name)
    rng = np.random.default_rng(0)
    for c in T.standard_contexts(t)["eval"][:4]:
        _, s, _ = best_known(t, c)
        assert s > 0
        for _ in range(20):
            assert t.score(rng.uniform(-1, 1, (t.H, t.A)), c) <= s + 1e-9


@pytest.mark.parametrize("name", ANALYTIC)
def test_stored_best_known_reproduces(name):
    t = T.get_task(name)
    for c in T.standard_contexts(t)["eval"]:
        assert abs(best_known(t, c)[1] - BEST[name]["best"][c["cid"]]) < 1e-9


def test_alpha_best_known_and_textbook():
    t = T.get_task("alpha_liquidation")
    for c in T.standard_contexts(t)["eval"][:3]:
        assert abs(best_known(t, c)[1] - BEST["alpha_liquidation"]["best"][c["cid"]]) < 1e-7
        assert abs(textbook(t, c) - BEST["alpha_liquidation"]["textbook"]["scores"][c["cid"]]) < 1e-12


def test_schedule_decoders_are_valid():
    for name in ("ac_liquidation", "ow_transient", "vwap_profile", "alpha_liquidation"):
        t = T.get_task(name)
        n = t.decode(np.random.default_rng(1).uniform(-1, 1, (t.H, 1)))
        assert abs(n.sum() - 1) < 1e-12 and (n >= 0).all()
    t = T.get_task("two_asset_transition")
    n = t.decode(np.zeros((t.H, 2)))
    assert abs(n[: t.H].sum() - 1) < 1e-12 and abs(n[t.H:].sum() + 1) < 1e-12


def test_trace_states_shape():
    t = T.get_task("ac_liquidation")
    c = t.nominal()
    st = T.trace_states(t, np.zeros((t.H, 1)), c)
    assert len(st) == t.H + 1 and st[0][0] == 1.0 and abs(st[-1][0]) < 1e-12
