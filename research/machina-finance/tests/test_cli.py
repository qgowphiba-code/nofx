import json
import os

import numpy as np
import pytest

import machina_finance.client as client_mod
from machina_finance import cli
from machina_finance import pipeline as PL
from machina_finance import tasks as T
from machina_finance.policy import FrozenPolicy
from machina_finance.fake import FakeAPI
from machina_finance.references import best_known

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pol_path(name):
    return os.path.join(ROOT, "policies", name)


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("MACHINA_FINANCE_FAKE", "1")
    monkeypatch.setattr(client_mod, "_FAKE", None)
    return client_mod


def run_json(capsys, argv):
    cli.main(argv)
    out = capsys.readouterr().out
    return json.loads(out[out.index("{"): out.rindex("}") + 1])


@pytest.mark.parametrize("task", sorted(T.TASKS))
def test_describe_lists_every_parameter(task, capsys):
    cli.main(["describe", "--task", task])
    out = capsys.readouterr().out
    for k in T.TASK_DOC[task]["params"]:
        assert f"  {k} " in out
    assert "controls:" in out and "score:" in out


def test_schedule_with_market_parameters(capsys):
    d = run_json(capsys, ["schedule", "--policy", pol_path("ac_liquidation--linear.json"),
                          "--params", "sigma=0.03,eta=0.005,lam=10", "--best"])
    assert d["params"] == {"sigma": 0.03, "eta": 0.005, "lam": 10.0}
    assert abs(sum(d["controls"]["fraction_per_slot"]) - 1) < 1e-4
    assert d["range_check"]["ok"] and 0 < d["share_of_best"] <= 1 + 1e-9 and d["score"] > d["base_score"]


def test_schedule_multi_leg_and_quotes_are_labelled(capsys):
    d = run_json(capsys, ["schedule", "--policy", pol_path("two_asset_transition--guided.json"), "--params", "rho=0.6"])
    assert set(d["controls"]) == {"A_sell", "B_buy"} and abs(sum(d["controls"]["B_buy"]) - 1) < 1e-4
    d = run_json(capsys, ["schedule", "--policy", pol_path("market_making--linear.json")])
    assert set(d["controls"]) == {"half_spread", "skew"} and len(d["controls"]["skew"]) == 8


def test_range_check_flags_conditions_outside_training():
    pol = FrozenPolicy.load(pol_path("ow_transient--guided.json"))
    inside = T.context_from_params(pol.task, {"halflife": 0.5})
    far = T.context_from_params(pol.task, {"halflife": 20.0, "lam": 5.0})
    assert pol.range_check(inside)["ok"]
    r = pol.range_check(far)
    assert not r["ok"] and "log_halflife" in r["outside_training_range"]


def test_context_from_params_derives_alpha_drift_and_unique_ids():
    t = T.get_task("alpha_liquidation")
    c = T.context_from_params(t, {"a": 1.0})
    assert c["p"]["mu"] == pytest.approx(2 * 1.0 * c["p"]["lam"] * c["p"]["sigma"] ** 2)
    assert c["cid"] != T.context_from_params(t, {"a": 0.5})["cid"]
    assert c["cid"] == T.context_from_params(t, {"a": 1.0})["cid"]
    with pytest.raises(KeyError):
        T.context_from_params(t, {"alpha": 1.0})


def test_verify_command_passes(capsys):
    cli.main(["verify"])
    assert "OK: every policy reproduces its prod run" in capsys.readouterr().out


def test_run_dry_run_counts_calls_per_stage(fake, capsys, tmp_path):
    d = run_json(capsys, ["run", "--task", "ac_liquidation", "--dry-run", "--quiet", "--out", str(tmp_path)])
    assert d["fake_api"] is True
    assert d["calls_by_stage"] == {"acquisition": 610, "refinement": 844, "export": 10, "cleanup": 1}
    assert d["export_check"]["exact"]
    assert FrozenPolicy.load(tmp_path / "policy.json").meta["fake_api"] is True
    assert json.load(open(tmp_path / "result.json"))["calls"] == 1466


def test_keep_then_domains_delete(fake, capsys, tmp_path):
    d = run_json(capsys, ["run", "--task", "ac_liquidation", "--episodes", "20", "--updates", "2", "--keep",
                          "--dry-run", "--quiet", "--prefix", "finm-pkg-keep", "--out", str(tmp_path)])
    assert d["kept_domain"] == "finm-pkg-keep-rlin"
    cli.main(["domains", "--prefix", "finm-pkg-keep"])
    assert "1 with prefix 'finm-pkg-keep'" in capsys.readouterr().out
    cli.main(["domains", "--prefix", "finm-pkg-keep", "--delete"])
    assert "deleted finm-pkg-keep-rlin" in capsys.readouterr().out
    cli.main(["domains"])
    assert "0 Domain(s)" in capsys.readouterr().out


def test_probe_on_fake(fake, capsys):
    cli.main(["probe"])
    assert "credits OK" in capsys.readouterr().out


def test_continue_same_learner_after_shift_and_reexport(fake):
    c = client_mod.Client()
    t = T.get_task("ac_liquidation")
    C = T.standard_contexts(t)
    out, pol = PL.run_pipeline(c, "ac_liquidation", "finm-pkg-drift", episodes=10, updates=3, keep_refinement=True,
                               log=lambda *a: None)
    dom = out["kept_domain"]

    def shifted(actions, ctx, update):  # the market after the shift: impact 50% higher
        p = dict(ctx["p"], eta=ctx["p"]["eta"] * 1.5)
        return t.score(actions, dict(ctx, p=p, cid=ctx["cid"] + "-shift"))
    rows = PL.refine(c, t, dom, pol.base, C["train"], C["support"], 6, start=3, create=False, score_fn=shifted,
                     log=lambda *a: None)
    assert [r["u"] for r in rows] == [3, 4, 5]
    pol2, check = PL.export_policy(c, t, dom, pol.base, C["support"], "linear", C["train"], C["eval"][:4],
                                   meta=dict(pol.meta))
    assert check["exact"] and pol2.meta["export_check"] is check  # the latest check replaces the old one
    c.delete_domain(dom)
    assert dom not in client_mod._FAKE.domains


class ToyTask(T.Task):
    """Minimal custom task used to test the generic best-known fallback."""
    name, H, A, d = "toy_spread", 4, 1, 1
    feature_names = ["urgency"]

    def params_from_u(self, u):
        return {"urgency": float(u[0])}

    def features(self, p):
        return [p["urgency"]]

    def decode(self, a):
        return T.schedule_weights(a[:, 0], self.H)

    def objective(self, n, ctx):
        target = np.exp(-ctx["p"]["urgency"] * np.arange(self.H))
        return float(np.sum((n - target / target.sum()) ** 2) * 1e4)


def test_best_known_falls_back_to_search_for_custom_tasks():
    t = ToyTask()
    c = t.make_contexts(1, 1, "toy")[0]
    a, s, ref = best_known(t, c)
    assert ref.get("approximate") and s >= 0 and a.shape == (4, 1)


def test_fake_state_persists_across_processes(tmp_path):
    path = str(tmp_path / "fake.pkl")
    api = FakeAPI(state_path=path)
    c = client_mod.Client(base="https://fake.invalid/api/v1", transport=api, sleep=lambda s: None)
    c.create_stage("finm-pkg-persist", {"mechanism": "episode_credit", "horizon": 2, "action_dimensions": 1})
    again = FakeAPI(state_path=path)  # as a new process would load it
    assert "finm-pkg-persist" in again.domains
