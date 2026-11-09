"""Command line: python -m machina_finance <command> ...

Offline (no key, no network):
  list                          the shortlist and the saved policies
  describe --task T             market parameters (range, unit), features, controls and score for a task
  schedule --policy P [--params k=v,...] [--u ...] [--best]
                                controls for one market condition, with a range check
  evaluate --policy P [--fresh N]   share of best on the 32 held-out conditions or on N fresh ones
  verify                        re-score every saved policy and compare with the scores recorded on prod
Live (needs ADAPT1_API_KEY, uses credits; add --dry-run or MACHINA_FINANCE_FAKE=1 to rehearse on a fake API):
  probe                         check the key has credits
  run --task T ...              acquire -> refine -> export -> evaluate; deletes its Domains (unless --keep)
  domains [--prefix P] [--delete]   list your Domains, optionally delete those with the prefix
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from . import tasks as T
from .client import Client
from .policy import FrozenPolicy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_client(**kw):
    return Client(**kw)


def _best_eval(task_name):
    with open(os.path.join(ROOT, "results", "best_known_eval.json")) as f:
        return json.load(f)[task_name]["best"]


def _parse_params(s):
    out = {}
    for item in (s or "").split(","):
        if item.strip():
            k, _, v = item.partition("=")
            out[k.strip()] = float(v)
    return out


def cmd_list(a):
    with open(os.path.join(ROOT, "results", "shortlist.json")) as f:
        rows = json.load(f)
    print(f"{'task':24s} {'arm':7s} {'refined':>8s} {'base':>7s} {'CEM':>7s}  textbook")
    for r in rows:
        tb = f"{r['textbook']} {r['textbook_share']:.0%}" if "textbook" in r else ""
        print(f"{r['task']:24s} {r['recommended_arm']:7s} {r[r['recommended_arm'] + '_share']:8.0%} "
              f"{r['base_share']:7.0%} {r.get('cem_ctx_lin_share', float('nan')):7.0%}  {tb}")
    print("\npolicies:", ", ".join(sorted(os.listdir(os.path.join(ROOT, "policies")))))


def cmd_describe(a):
    t = T.get_task(a.task)
    doc = T.TASK_DOC[a.task]
    nom = t.nominal()["p"]
    print(f"{a.task}: {doc['title']}")
    print("market parameters (training range; nominal):")
    for k, (rng, what) in doc["params"].items():
        print(f"  {k:10s} {rng:16s} nominal {nom[k]:.6g}  {what}")
    print("features the policy sees:", ", ".join(t.feature_names), "(scaled versions of the parameters)")
    print("controls:", doc["controls"])
    print("score:", doc["score"], "(0 = naive benchmark)")


def cmd_schedule(a):
    pol = FrozenPolicy.load(a.policy)
    t = pol.task
    if a.u:
        ctx = T.context_from_u(t, [float(x) for x in a.u.split(",")])
    else:
        ctx = T.context_from_params(t, _parse_params(a.params))
    out = {"task": pol.task_name, "params": {k: round(v, 6) for k, v in ctx["p"].items() if isinstance(v, float)},
           "range_check": pol.range_check(ctx),
           "controls": T.controls_table(t, pol.controls(ctx)), "controls_meaning": T.TASK_DOC[t.name]["controls"],
           "score": round(pol.score(ctx), 6), "base_score": round(float(t.score(pol.base, ctx)), 6)}
    if a.best:
        from .references import best_known
        b = best_known(t, ctx)[1]
        out.update(best_known_score=round(b, 6), share_of_best=round(out["score"] / b, 4) if b > 0 else None)
    print(json.dumps(out, indent=1))


def cmd_evaluate(a):
    pol = FrozenPolicy.load(a.policy)
    t = pol.task
    if a.fresh:
        from .references import best_known
        ctxs = t.make_contexts(a.fresh, a.seed, "fresh")
        best = [best_known(t, c)[1] for c in ctxs]
    else:
        ctxs = T.standard_contexts(t)["eval"]
        b = _best_eval(pol.task_name)
        best = [b[c["cid"]] for c in ctxs]
    s = np.array([pol.score(c) for c in ctxs])
    sb = np.array([t.score(pol.base, c) for c in ctxs])
    print(json.dumps({"task": pol.task_name, "contexts": len(ctxs), "refined_share_of_best": float(s.mean() / np.mean(best)),
                      "base_share_of_best": float(sb.mean() / np.mean(best)),
                      "refined_beats_base": float(np.mean(s > sb + 1e-12))}, indent=1))


def cmd_verify(a):
    with open(os.path.join(ROOT, "results", "policies_heldout.json")) as f:
        rows = json.load(f)
    worst = 0.0
    print(f"{'policy':38s} {'share of best':>13s} {'max |diff| vs prod run':>23s}")
    for r in rows:
        pol = FrozenPolicy.load(os.path.join(ROOT, "policies", f"{r['task']}--{r['arm']}.json"))
        evalc = T.standard_contexts(pol.task)["eval"]
        s = np.array([pol.score(c) for c in evalc])
        best = _best_eval(r["task"])
        diff = float(np.abs(s - np.array(r["stored_heldout_scores"])).max())
        worst = max(worst, diff)
        print(f"{r['task'] + '--' + r['arm']:38s} {s.mean() / np.mean([best[c['cid']] for c in evalc]):13.1%} {diff:23.1e}")
    print("OK: every policy reproduces its prod run" if worst < 1e-9 else f"MISMATCH: worst difference {worst:.2e}")
    if worst >= 1e-9:
        sys.exit(1)


def cmd_probe(a):
    c = make_client()
    print("credits OK" if c.credits_ok() else "no credits (HTTP 402)")


def cmd_domains(a):
    c = make_client()
    d = c.list_domains()
    items = d if isinstance(d, list) else (d.get("domains") or d.get("items") or d.get("data") or [])
    names = [(x.get("domain_id") or x.get("name") or x.get("id")) if isinstance(x, dict) else str(x) for x in items]
    mine = [n for n in names if n and n.startswith(a.prefix)]
    print(f"{len(names)} Domain(s) on this key, {len(mine)} with prefix {a.prefix!r}")
    for n in mine:
        print(" ", n)
    if a.delete:
        for n in mine:
            c.delete_domain(n)
            print("  deleted", n)


def cmd_run(a):
    from .pipeline import run_pipeline
    from .references import best_known
    if a.dry_run:
        os.environ["MACHINA_FINANCE_FAKE"] = "1"
    prefix = a.prefix or f"finm-pkg-{a.task.split('_')[0][:4]}-{int(time.time()) % 100000}"
    out_dir = a.out or os.path.join(ROOT, "runs", prefix)
    os.makedirs(out_dir, exist_ok=True)
    c = make_client(journal_path=os.path.join(out_dir, "journal.jsonl"), ledger_path=os.path.join(out_dir, "ledger.jsonl"),
                    max_calls=a.max_calls)
    if not c.credits_ok():
        sys.exit("no credits (HTTP 402); nothing was created")
    out, pol = run_pipeline(c, a.task, prefix, episodes=a.episodes, updates=a.updates, arm=a.arm, mie=not a.no_mie,
                            threads=a.threads, check_eval_on_api=a.check_api, keep_refinement=a.keep,
                            log=(lambda *x: None) if a.quiet else print)
    t = pol.task
    evalc = T.standard_contexts(t)["eval"]
    try:  # stored best-known scores for the kept tasks (market making etc. would take minutes to recompute)
        stored = _best_eval(t.name)
        best = [stored[x["cid"]] for x in evalc]
    except KeyError:
        best = [best_known(t, x)[1] for x in evalc]
    out["heldout"] = {"refined_share_of_best": float(np.mean([pol.score(x) for x in evalc]) / np.mean(best)),
                      "base_share_of_best": float(np.mean([t.score(pol.base, x) for x in evalc]) / np.mean(best))}
    pol.save(os.path.join(out_dir, "policy.json"))
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    keys = ("fake_api", "heldout", "calls_by_stage", "calls", "export_check", "deleted", "kept_domain")
    print(json.dumps({k: out[k] for k in keys if k in out}, indent=1, default=float))
    print("policy and result saved in", out_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="machina_finance", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("list")
    p = sp.add_parser("describe")
    p.add_argument("--task", required=True, choices=sorted(T.TASKS))
    p = sp.add_parser("schedule")
    p.add_argument("--policy", required=True)
    p.add_argument("--params", help="market parameters in natural units, e.g. sigma=0.03,eta=0.005 (others nominal)")
    p.add_argument("--u", help="alternatively: comma-separated normalised parameters in [-1, 1]")
    p.add_argument("--best", action="store_true", help="also compute the best-known schedule's score")
    p = sp.add_parser("evaluate")
    p.add_argument("--policy", required=True)
    p.add_argument("--fresh", type=int, default=0)
    p.add_argument("--seed", type=int, default=404)
    sp.add_parser("verify")
    sp.add_parser("probe")
    p = sp.add_parser("domains")
    p.add_argument("--prefix", default="finm-pkg")
    p.add_argument("--delete", action="store_true")
    p = sp.add_parser("run")
    p.add_argument("--task", required=True, choices=sorted(T.TASKS))
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--updates", type=int, default=20)
    p.add_argument("--arm", default="linear", choices=("linear", "guided"))
    p.add_argument("--prefix")
    p.add_argument("--out")
    p.add_argument("--max-calls", type=int, default=5000)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--no-mie", action="store_true")
    p.add_argument("--check-api", action="store_true", help="also compare the policy with the API on all held-out contexts")
    p.add_argument("--keep", action="store_true", help="keep the refinement learner to continue it later")
    p.add_argument("--dry-run", action="store_true", help="rehearse on the in-memory fake API (no key, no credits)")
    p.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    {"list": cmd_list, "describe": cmd_describe, "schedule": cmd_schedule, "evaluate": cmd_evaluate,
     "verify": cmd_verify, "probe": cmd_probe, "domains": cmd_domains, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
