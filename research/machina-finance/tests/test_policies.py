import glob
import json
import os

import numpy as np
import pytest

from machina_finance import tasks as T
from machina_finance.basis import Basis
from machina_finance.policy import FrozenPolicy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICIES = sorted(glob.glob(os.path.join(ROOT, "policies", "*.json")))
HELD = {(r["task"], r["arm"]): r for r in json.load(open(os.path.join(ROOT, "results", "policies_heldout.json")))}
BEST = json.load(open(os.path.join(ROOT, "results", "best_known_eval.json")))
SHORT = {r["task"]: r for r in json.load(open(os.path.join(ROOT, "results", "shortlist.json")))}


def test_every_kept_task_has_a_policy():
    tasks = {os.path.basename(p).split("--")[0] for p in POLICIES}
    assert tasks == set(T.TASKS)
    assert len(POLICIES) == 13


@pytest.mark.parametrize("path", POLICIES, ids=lambda p: os.path.basename(p)[:-5])
def test_policy_reproduces_prod_run_scores(path):
    pol = FrozenPolicy.load(path)
    rec = HELD[(pol.task_name, pol.meta["arm"])]
    evalc = T.standard_contexts(pol.task)["eval"]
    scores = np.array([pol.score(c) for c in evalc])
    assert np.abs(scores - np.array(rec["stored_heldout_scores"])).max() < 1e-9
    best = BEST[pol.task_name]["best"]
    share = scores.mean() / np.mean([best[c["cid"]] for c in evalc])
    assert abs(share - rec["share_of_best"]) < 1e-4
    assert abs(share - SHORT[pol.task_name][f"{pol.meta['arm']}_share"]) < 1e-9
    assert pol.meta["fit_residual"] < 1e-9


@pytest.mark.parametrize("path", POLICIES[:3], ids=lambda p: os.path.basename(p)[:-5])
def test_policy_roundtrip_and_controls(path, tmp_path):
    pol = FrozenPolicy.load(path)
    pol.save(tmp_path / "p.json")
    pol2 = FrozenPolicy.load(tmp_path / "p.json")
    c = T.standard_contexts(pol.task)["eval"][3]
    assert np.allclose(pol.actions(c["feat"]), pol2.actions(c["feat"]), atol=0)
    u = pol.controls(c)
    assert len(u) >= pol.task.H


def test_refined_beats_base_where_claimed():
    for task, r in SHORT.items():
        best_arm = r[f"{r['recommended_arm']}_share"]
        assert best_arm > r["base_share"], task
        assert best_arm > r.get("cem_ctx_lin_share", -np.inf), task


def test_linear_basis_clips_outlying_contexts():
    rng = np.random.default_rng(0)
    S = rng.standard_normal((24, 3))
    b = Basis(S, "linear")
    W = rng.standard_normal((b.P, 5))
    far = b.mu + 50 * b.sdp[0] * b.Vt[0]
    edge = b.mu + 3.0 * b.sdp[0] * b.Vt[0]
    assert np.allclose(b.phi([far]) @ W, b.phi([edge]) @ W)  # saturates at +-3 SD along the PCA axis
    inside = b.mu + 1.0 * b.sdp[1] * b.Vt[1]
    assert np.allclose(b.transform([inside]), inside)  # identity inside the box
    q = Basis(S, "quadratic")
    assert q.P == 1 + 3 + 6
    assert Basis.from_dict(q.to_dict()).phi([far]).tolist() == q.phi([far]).tolist()


def test_smoke_test_policy_matches_its_record():
    p = os.path.join(ROOT, "results", "smoke_policy_ac_liquidation.json")
    rec = json.load(open(os.path.join(ROOT, "results", "smoke_test.json")))
    if "heldout_32" not in rec:
        pytest.skip("live smoke test was skipped")
    pol = FrozenPolicy.load(p)
    evalc = T.standard_contexts(pol.task)["eval"]
    best = BEST["ac_liquidation"]["best"]
    share = np.mean([pol.score(c) for c in evalc]) / np.mean([best[c["cid"]] for c in evalc])
    assert abs(share - rec["heldout_32"]["refined_share_of_best"]) < 1e-9
    assert rec["export_check"]["exact"] and rec["eval_api_vs_policy_max_abs_diff"] < 1e-9
    assert rec["domains_left_with_prefix"] == []
