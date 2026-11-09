import json

import pytest

from fake_api import BASE, FakeAPI
from machina_finance.client import AmbiguousWrite, APIError, BudgetExceeded, Client, CreditsOut


def scripted(responses):
    """Transport returning the given (status, body) pairs in order and recording requests."""
    seen = []

    def t(method, url, headers, data, timeout):
        seen.append((method, url, headers, data))
        status, body = responses.pop(0)
        return status, json.dumps(body)
    return t, seen


def mk(t, **kw):
    return Client(base=BASE, transport=t, sleep=lambda s: None, **kw)


def test_429_is_retried_for_mutations():
    t, seen = scripted([(429, {}), (429, {}), (200, {"ok": 1})])
    c = mk(t)
    assert c.traj("d", "propose", {"x": 1}) == {"ok": 1}
    assert len(seen) == 3 and c.n429 == 2 and c.calls == 3


def test_5xx_retried_on_reads_but_ambiguous_on_mutations():
    t, _ = scripted([(503, {}), (200, {"status": "ready"})])
    assert mk(t).traj("d", "state") == {"status": "ready"}
    t, _ = scripted([(503, {})])
    with pytest.raises(AmbiguousWrite):
        mk(t).traj("d", "observe", {})


def test_402_raises_credits_out_and_probe():
    t, _ = scripted([(402, {"error": "insufficient query credits"})])
    with pytest.raises(CreditsOut):
        mk(t).traj("d", "propose", {})
    t, _ = scripted([(404, {"error": "no such domain"})])
    assert mk(t).credits_ok() is True
    t, _ = scripted([(402, {"error": "credit plan is not active"})])
    assert mk(t).credits_ok() is False


def test_other_errors_raise_api_error():
    t, _ = scripted([(422, {"error": "bad"})])
    with pytest.raises(APIError) as e:
        mk(t).traj("d", "configure", {})
    assert e.value.status == 422


def test_auth_header_from_env_and_never_journaled(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPT1_API_KEY", "sk-test-not-a-real-key")
    t, seen = scripted([(200, {})])
    c = mk(t, journal_path=str(tmp_path / "j.jsonl"))
    c.traj("d", "state")
    assert seen[0][2]["Authorization"] == "Bearer sk-test-not-a-real-key"
    assert "sk-test" not in (tmp_path / "j.jsonl").read_text()
    monkeypatch.setenv("ADAPT1_AUTH_HEADER", "X-Api-Key")
    monkeypatch.setenv("ADAPT1_AUTH_SCHEME", "")
    t, seen = scripted([(200, {})])
    mk(t).traj("d", "state")
    assert seen[0][2]["X-Api-Key"] == "sk-test-not-a-real-key" and "Authorization" not in seen[0][2]


def test_no_auth_header_without_key(monkeypatch):
    monkeypatch.delenv("ADAPT1_API_KEY", raising=False)
    t, seen = scripted([(200, {})])
    mk(t).traj("d", "state")
    assert "Authorization" not in seen[0][2]


def test_call_cap_keeps_a_cleanup_reserve():
    api = FakeAPI()
    c = Client(base=BASE, transport=api, sleep=lambda s: None, max_calls=6, cleanup_reserve=2)
    c.create_stage("dom", {"mechanism": "episode_credit", "horizon": 2, "action_dimensions": 1})  # 4 calls
    with pytest.raises(BudgetExceeded):
        c.traj("dom", "state")
    c.delete_domain("dom")  # cleanup still allowed
    assert c.calls == 5 and "dom" not in api.domains


def test_create_stage_and_ledger(tmp_path):
    api = FakeAPI()
    c = Client(base=BASE, transport=api, sleep=lambda s: None, ledger_path=str(tmp_path / "ledger.jsonl"))
    receipt, st = c.create_stage("dom", {"mechanism": "episode_credit", "horizon": 2, "action_dimensions": 1})
    assert st["status"] == "ready" and receipt["config"]["horizon"] == 2
    c.delete_domain("dom")
    c.delete_domain("dom")  # already gone: 404 tolerated
    ev = [json.loads(l)["event"] for l in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert ev == ["create_intent", "deleted", "deleted"]
