"""Live end-to-end smoke test on the prod API (small budget): ac_liquidation, linear arm with MIE.

acquire 100 episodes at the nominal context -> 12 refinement updates -> export the frozen policy ->
compare it with the API on all 32 held-out contexts -> score it offline -> delete every Domain -> confirm
with GET /domains. Hard cap: 1,000 API calls (work stops at 970; the rest is reserved for cleanup).
Writes results/smoke_test.json and results/smoke_policy_ac_liquidation.json; the request journal goes to
runs/ (not committed).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from machina_finance import tasks as T  # noqa: E402
from machina_finance.client import Client  # noqa: E402
from machina_finance.pipeline import run_pipeline  # noqa: E402
from machina_finance.references import best_known  # noqa: E402

CAP, RESERVE = 1000, 30


def main():
    stamp = time.strftime("%m%d%H%M", time.gmtime())
    prefix = f"finm-pkg-ac-{stamp}"
    run_dir = os.path.join(ROOT, "runs", prefix)
    c = Client(journal_path=os.path.join(run_dir, "journal.jsonl"), ledger_path=os.path.join(run_dir, "ledger.jsonl"),
               max_calls=CAP, cleanup_reserve=RESERVE)
    res = {"prefix": prefix, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cap": CAP}
    if not c.credits_ok():
        res["skipped"] = "HTTP 402: no query credits"
        print(res["skipped"])
    else:
        try:
            out, pol = run_pipeline(c, "ac_liquidation", prefix, episodes=100, updates=12, arm="linear", mie=True,
                                    threads=2, check_eval_on_api=True)
            t = pol.task
            evalc = T.standard_contexts(t)["eval"]
            best = np.array([best_known(t, x)[1] for x in evalc])
            s = np.array([pol.score(x) for x in evalc])
            sb = np.array([t.score(pol.base, x) for x in evalc])
            res.update({k: out[k] for k in ("acquisition", "export_check", "eval_api_vs_policy_max_abs_diff", "deleted")})
            res["acquisition"].pop("incumbent", None)
            for f in res["acquisition"]["frozen"]:
                f.pop("actions", None)
            res["acquisition"]["nominal_best"] = float(best_known(t, t.nominal())[1])
            res["refinement_updates"] = [{k: r[k] for k in ("u", "selected", "gain", "calls", "fallbacks")}
                                         for r in out["refinement"]]
            res["heldout_32"] = {"refined_share_of_best": float(s.mean() / best.mean()),
                                 "base_share_of_best": float(sb.mean() / best.mean()),
                                 "refined_beats_base": float(np.mean(s > sb + 1e-12))}
            pol.meta["note"] = "smoke test: 100 acquisition episodes, 12 refinement updates (small budget)"
            pol.save(os.path.join(ROOT, "results", "smoke_policy_ac_liquidation.json"))
            res["secs"] = out["secs"]
        except Exception as e:  # record the failure; cleanup already ran inside run_pipeline
            res["error"] = repr(e)
            print("ERROR", repr(e))
    try:
        doms = c.list_domains()
        items = doms if isinstance(doms, list) else (doms.get("domains") or doms.get("items") or doms.get("data") or [])
        names = [(x.get("domain_id") or x.get("name") or x.get("id")) if isinstance(x, dict) else str(x) for x in items]
        res["domains_left_with_prefix"] = [n for n in names if n and n.startswith("finm-pkg")]
        res["domains_left_total"] = len(names)
    except Exception as e:
        res["domain_check_error"] = repr(e)
    res["api_calls"] = c.calls
    res["http_429"] = c.n429
    with open(os.path.join(ROOT, "results", "smoke_test.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ("heldout_32", "export_check", "eval_api_vs_policy_max_abs_diff",
                                               "api_calls", "domains_left_with_prefix", "domains_left_total",
                                               "skipped", "error")}, indent=1, default=float))


if __name__ == "__main__":
    main()
