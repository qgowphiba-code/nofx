"""In-memory stand-in for the parts of the Adapt-1 trajectory API the pipeline uses (plumbing tests only).

It mimics response shapes and the contextual_execution map semantics (offsets = clip(phi(T(f)) @ W, -1, 1)
with the service-equivalent basis); it does not try to mimic how the real service learns.
"""
import hashlib
import json

import numpy as np

from machina_finance.basis import Basis

BASE = "https://fake.invalid/api/v1"


class FakeAPI:
    def __init__(self, seed=0, fail=None):
        self.domains = {}
        self.rng = np.random.default_rng(seed)
        self.requests = []
        self.fail = fail or {}  # {(method, path_suffix): [status, ...]} statuses returned before succeeding

    # transport(method, url, headers, data, timeout) -> (status, text)
    def __call__(self, method, url, headers, data, timeout):
        assert url.startswith(BASE)
        path = url[len(BASE):]
        body = json.loads(data) if data else None
        self.requests.append((method, path, body, dict(headers)))
        for (m, suffix), statuses in self.fail.items():
            if m == method and path.endswith(suffix) and statuses:
                return statuses.pop(0), json.dumps({"error": "injected"})
        status, out = self.route(method, path, body)
        return status, json.dumps(out)

    def route(self, method, path, body):
        parts = path.strip("/").split("/")
        if parts == ["domains"]:
            if method == "GET":
                return 200, {"domains": [{"domain_id": d} for d in self.domains]}
            self.domains[body["domain_id"]] = {"config": None}
            return 201, {"domain_id": body["domain_id"]}
        if parts[0] == "domains" and len(parts) == 2 and method == "DELETE":
            if parts[1] not in self.domains:
                return 404, {"error": "not found"}
            del self.domains[parts[1]]
            return 200, {"deleted": True}
        dom_id, op = parts[1], parts[3]
        if dom_id not in self.domains:
            return 404, {"error": "unknown domain"}
        d = self.domains[dom_id]
        if op == "state":
            if d["config"] is None:
                return 200, {"status": "not_configured"}
            return 200, {"status": "ready", "fingerprint": self.fingerprint(d)}
        if op == "configure":
            d["config"] = body["config"]
            if body["config"]["mechanism"] == "contextual_execution":
                c = body["config"]
                kind = "quadratic" if c.get("context_basis") == "quadratic" else "linear"
                d["basis"] = Basis(c["contexts"], kind)
                d["W0"] = np.zeros((d["basis"].P, c["horizon"] * c["action_dimensions"]))
            else:
                d["best"], d["best_actions"] = -np.inf, None
            d["pending"], d["n"] = {}, 0
            return 200, {"config": body["config"]}
        mech = d["config"]["mechanism"]
        return getattr(self, f"{mech}_{op}")(d, body)

    @staticmethod
    def fingerprint(d):
        x = d.get("W0") if "W0" in d else d.get("best_actions")
        return hashlib.sha256(json.dumps(None if x is None else np.asarray(x).tolist()).encode()).hexdigest()

    # -- episode_credit ---------------------------------------------------------------
    def episode_credit_propose(self, d, body):
        c = d["config"]
        H, A = c["horizon"], c["action_dimensions"]
        if body["mode"] == "frozen":
            a = d["best_actions"] if d["best_actions"] is not None else np.zeros((H, A))
            return 200, {"actions": np.asarray(a).tolist(), "source": "incumbent"}
        centre = d["best_actions"] if d["best_actions"] is not None else np.zeros((H, A))
        a = np.clip(np.asarray(centre) + 0.3 * self.rng.standard_normal((H, A)), -1, 1)
        d["n"] += 1
        did = f"dec-{d['n']}"
        d["pending"][did] = a
        return 200, {"decision_id": did, "actions": a.tolist(), "source": "explore"}

    def episode_credit_observe(self, d, body):
        a = d["pending"].pop(body["decision_id"])
        assert len(body["states"]) == len(body["actions"]) + 1
        improved = body["outcome"] > d["best"]
        if improved:
            d["best"], d["best_actions"] = body["outcome"], a
        return 200, {"improved_parent": bool(improved)}

    # -- contextual_execution ---------------------------------------------------------
    def contextual_execution_propose(self, d, body):
        c = d["config"]
        if c.get("response_guidance"):
            assert isinstance(body.get("context_observations"), dict)
        K = c["candidate_count"]
        Ws = [d["W0"]] + [d["W0"] + c["radius"] * self.rng.standard_normal(d["W0"].shape) / d["basis"].P
                          for _ in range(K - 1)]
        d["n"] += 1
        did = f"dec-{d['n']}"
        d["pending"][did] = Ws
        return 200, {"decision_id": did, "candidates": [{"index": k} for k in range(K)]}

    def contextual_execution_evaluate(self, d, body):
        if "candidate_index" in body:
            W = d["pending"][body["decision_id"]][body["candidate_index"]]
        else:
            W = d["W0"]
        off = d["basis"].offsets(W, [body["context"]])[0]
        c = d["config"]
        return 200, {"offsets": off.reshape(c["horizon"], c["action_dimensions"]).tolist()}

    def contextual_execution_observe(self, d, body):
        Ws = d["pending"].pop(body["decision_id"])
        K = len(Ws)
        sums, cnt = np.zeros(K), np.zeros(K)
        for o in body["outcomes"]:
            sums[o["candidate_index"]] += o["objectives"][0]
            cnt[o["candidate_index"]] += 1
        assert (cnt == cnt[0]).all() and cnt[0] > 0
        m = sums / cnt
        k = int(np.argmax(m))
        gain = float(m[k] - m[0])
        if gain > 0:
            d["W0"] = Ws[k]
        return 200, {"update": {"selected": k, "confirmation_gain": gain, "next_action_search_radius": 0.1}}
