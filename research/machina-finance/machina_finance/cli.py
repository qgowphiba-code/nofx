"""Command line: python -m machina_finance <command> ...

  list                         shortlist with held-out shares and the saved policies
  schedule --policy P [--u ..] controls for one context (normalised parameters u in [-1, 1], default nominal)
  evaluate --policy P [--fresh N]  offline score: 32 held-out contexts (stored best-known) or N fresh ones
  probe                        check the API key has credits (needs network access)
  run --task T ...             live pipeline: acquire -> refine -> export -> evaluate; deletes its Domains
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


def _best_eval(task_name):
    with open(os.path.join(ROOT, "results", "best_known_eval.json")) as f:
        return json.load(f)[task_name]["best"]


def cmd_list(a):
    with open(os.path.join(ROOT, "results", "shortlist.json")) as f:
        rows = json.load(f)
    print(f"{'task':24s} {'arm':7s} {'refined':>8s} {'base':>7s} {'CEM':>7s}  textbook")
    for r in rows:
        tb = f"{r['textbook']} {r['textbook_share']:.0%}" if "textbook" in r else ""
        print(f"{r['task']:24s} {r['recommended_arm']:7s} {r[r['recommended_arm'] + '_share']:8.0%} "
              f"{r['base_share']:7.0%} {r.get('cem_ctx_lin_share', float('nan')):7.0%}  {tb}")
    print("\npolicies:", ", ".join(sorted(os.listdir(os.path.join(ROOT, "policies")))))


def cmd_schedule(a):
    pol = FrozenPolicy.load(a.policy)
    u = [float(x) for x in a.u.split(",")] if a.u else [0.0] * pol.task.d
    ctx = T.context_from_u(pol.task, u)
    print(json.dumps({"task": pol.task_name, "params": ctx["p"], "features": dict(zip(pol.task.feature_names, ctx["feat"])),
                      "actions": pol.actions(ctx["feat"]).round(4).tolist(),
                      "controls": np.round(pol.controls(ctx), 5).tolist(), "score": pol.score(ctx)}, indent=1))


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


def cmd_probe(a):
    c = Client()
    print("credits OK" if c.credits_ok() else "no credits (HTTP 402)")


def cmd_run(a):
    from .pipeline import run_pipeline
    prefix = a.prefix or f"finm-pkg-{T.TASKS[a.task]().name[:2]}-{int(time.time()) % 100000}"
    out_dir = a.out or os.path.join(ROOT, "runs", prefix)
    os.makedirs(out_dir, exist_ok=True)
    c = Client(journal_path=os.path.join(out_dir, "journal.jsonl"), ledger_path=os.path.join(out_dir, "ledger.jsonl"),
               max_calls=a.max_calls)
    if not c.credits_ok():
        sys.exit("no credits (HTTP 402); nothing was created")
    out, pol = run_pipeline(c, a.task, prefix, episodes=a.episodes, updates=a.updates, arm=a.arm, mie=not a.no_mie,
                            threads=a.threads, check_eval_on_api=a.check_api)
    t = pol.task
    evalc = T.standard_contexts(t)["eval"]
    from .references import best_known
    best = [best_known(t, c)[1] for c in evalc]
    out["heldout"] = {"refined_share_of_best": float(np.mean([pol.score(c) for c in evalc]) / np.mean(best)),
                      "base_share_of_best": float(np.mean([t.score(pol.base, c) for c in evalc]) / np.mean(best))}
    pol.save(os.path.join(out_dir, "policy.json"))
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("heldout", "export_check", "calls", "secs", "deleted")}, indent=1, default=float))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="machina_finance")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("list")
    p = sp.add_parser("schedule")
    p.add_argument("--policy", required=True)
    p.add_argument("--u", help="comma-separated normalised parameters in [-1, 1] (default nominal)")
    p = sp.add_parser("evaluate")
    p.add_argument("--policy", required=True)
    p.add_argument("--fresh", type=int, default=0)
    p.add_argument("--seed", type=int, default=404)
    sp.add_parser("probe")
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
    a = ap.parse_args(argv)
    {"list": cmd_list, "schedule": cmd_schedule, "evaluate": cmd_evaluate, "probe": cmd_probe, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
