import numpy as np
import pytest

from fake_api import BASE, FakeAPI
from machina_finance import pipeline as PL
from machina_finance import tasks as T
from machina_finance.client import Client


def client(api, **kw):
    return Client(base=BASE, transport=api, sleep=lambda s: None, **kw)


def observes(api, dom):
    return [b for m, p, b, _ in api.requests if p == f"/domains/{dom}/trajectory/observe"]


def test_acquire_returns_incumbent_and_checkpoints():
    api = FakeAPI(seed=1)
    c = client(api)
    t = T.get_task("ac_liquidation")
    out = PL.acquire(c, t, "pkg-acq", t.nominal(), 12, checkpoints=(6, 12), log=lambda *a: None)
    assert len(out["curve"]) == 12 and [f["ep"] for f in out["frozen"]] == [6, 12]
    assert np.array(out["incumbent"]).shape == (t.H, t.A)
    assert out["incumbent_score"] == pytest.approx(max(out["curve"]))
    assert c.calls == 4 + 2 * 12 + 2 + 1


@pytest.mark.parametrize("task_name", ["ac_liquidation", "two_asset_transition"])
def test_mie_matches_full_evaluation_with_fewer_calls(task_name):
    t = T.get_task(task_name)
    C = T.standard_contexts(t)
    base = np.zeros((t.H, t.A))
    runs = {}
    for mie in (False, True):
        api = FakeAPI(seed=7)
        c = client(api)
        log = PL.refine(c, t, "pkg-r", base, C["train"], C["support"], 3, arm="linear", seed=21, mie=mie,
                        log=lambda *a: None)
        runs[mie] = (observes(api, "pkg-r"), c.calls, log, api)
    full, mie = runs[False], runs[True]
    for a, b in zip(full[0], mie[0]):
        sa = np.array([o["objectives"][0] for o in a["outcomes"]])
        sb = np.array([o["objectives"][0] for o in b["outcomes"]])
        assert np.abs(sa - sb).max() < 1e-12
        assert [(o["candidate_index"], o["context_id"]) for o in a["outcomes"]] == \
               [(o["candidate_index"], o["context_id"]) for o in b["outcomes"]]
    assert mie[1] < full[1] / 2.5
    assert all(r["fallbacks"] == 0 for r in mie[2])
    assert np.allclose(full[3].domains["pkg-r"]["W0"], mie[3].domains["pkg-r"]["W0"], atol=0)


@pytest.mark.parametrize("arm", ["linear", "guided"])
def test_export_recovers_the_retained_map_exactly(arm):
    t = T.get_task("ow_transient")
    C = T.standard_contexts(t)
    api = FakeAPI(seed=3)
    c = client(api)
    base = np.zeros((t.H, t.A))
    PL.refine(c, t, "pkg-r", base, C["train"], C["support"], 2, arm=arm, seed=21, log=lambda *a: None)
    b = PL.basis_for(arm, [s["feat"] for s in C["support"]])
    verify = sorted(C["eval"], key=lambda x: -b.z_max(x["feat"])[0])[:4]
    pol, check = PL.export_policy(c, t, "pkg-r", base, C["support"], arm, C["train"], verify)
    assert check["exact"] and check["fingerprint_unchanged"]
    d = api.domains["pkg-r"]
    for ctx in C["eval"]:
        assert np.abs(pol.offsets(ctx["feat"]).ravel() - d["basis"].offsets(d["W0"], [ctx["feat"]])[0]).max() < 1e-9
    if arm == "guided":
        props = [bd for m, p, bd, _ in api.requests if p.endswith("/propose")]
        assert all(isinstance(bd["context_observations"], dict) for bd in props)


def test_run_pipeline_end_to_end_deletes_everything():
    api = FakeAPI(seed=5)
    c = client(api, max_calls=2000)
    out, pol = PL.run_pipeline(c, "ac_liquidation", "pkg-test", episodes=10, updates=2, log=lambda *a: None,
                               check_eval_on_api=True)
    assert api.domains == {}
    assert sorted(out["deleted"]) == ["pkg-test-acq", "pkg-test-rlin"]
    assert out["export_check"]["exact"] and out["eval_api_vs_policy_max_abs_diff"] < 1e-9
    assert pol.base.shape == (10, 1)


def test_crn_shares_scenarios_across_candidates():
    seen = []

    def noisy(a, ctx, day):
        seen.append(day)
        return float(np.sum(a)) + day * 0.0
    f = PL.crn(noisy, 3, seed=1)
    f(np.zeros((2, 1)), {}, 4)
    first = list(seen)
    seen.clear()
    f(np.ones((2, 1)), {}, 4)
    assert seen == first and len(set(first)) == 3


def test_candidate_count_floor():
    t = T.get_task("ac_liquidation")
    with pytest.raises(ValueError):
        PL.refinement_config(t, np.zeros((10, 1)), [[0, 0, 0]], candidate_count=3)


def test_cusum_alarm():
    assert PL.cusum_alarm([0, 0, 0], [0, 0, 0], 1.0) is None
    assert PL.cusum_alarm([0] * 5 + [3] * 10, [0] * 15, 1.0) is not None
