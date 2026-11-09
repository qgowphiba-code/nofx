"""Build results/shortlist.{json,csv} and results/evidence.json (provenance; needs the run directory).

usage: python scripts/build_results.py --fin /path/to/fin

Held-out shares (32 contexts) for the exported policies are recomputed offline with this package;
baseline shares (same-budget contextual CEM, textbook rules, oracle-fit ceilings) are read from the runs.
Analytic tasks are also re-scored on 512 fresh contexts (exact best-known, so this is cheap).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from machina_finance import tasks as T  # noqa: E402
from machina_finance.policy import FrozenPolicy  # noqa: E402
from machina_finance.references import best_known  # noqa: E402

RUN_EVAL = {"ac_liquidation": "fx2/ac_liquidation", "ow_transient": "fx2/ow_transient",
            "two_asset_transition": "fx2/two_asset_transition", "market_making": "fx2/market_making",
            "hedge_bands": "fx2/hedge_bands", "glide_path": "fx2/glide_path", "vwap_profile": "fx2b/vwap_profile"}
RECOMMENDED = {"ac_liquidation": "linear", "ow_transient": "guided", "vwap_profile": "linear",
               "two_asset_transition": "guided", "three_asset_transition": "guided", "market_making": "linear",
               "hedge_bands": "linear", "glide_path": "linear", "alpha_liquidation": "guided"}
LABEL = {"ac_liquidation": "Liquidation, temporary + permanent impact (Almgren-Chriss)",
         "ow_transient": "Liquidation, transient impact (Obizhaeva-Wang)",
         "vwap_profile": "VWAP execution with day-type volume profile",
         "two_asset_transition": "Two-leg portfolio switch", "three_asset_transition": "Three-leg portfolio switch",
         "market_making": "Market-making quotes (spread, skew)", "hedge_bands": "Option delta-hedging bands",
         "glide_path": "Lifecycle glide path", "alpha_liquidation": "Liquidation with a pre-trade alpha signal"}
ANALYTIC_512 = {"ac_liquidation": (404, "big"), "ow_transient": (404, "big"), "vwap_profile": (404, "big"),
                "two_asset_transition": (404, "big"), "three_asset_transition": (404, "big"),
                "alpha_liquidation": (909, "fr")}
EVIDENCE_IDS = ("rs-vardecomp", "gsg", "rs-crnnoise", "rs-shift", "ep-closed-loop-alpha", "voltgt", "api-sweep")


def share(x, best):
    return float(np.mean(x) / np.mean(best))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fin", required=True)
    a = ap.parse_args()
    runs = os.path.join(a.fin, "runs")
    best_eval = json.load(open(os.path.join(ROOT, "results", "best_known_eval.json")))
    rows = []
    for task_name, arm_rec in RECOMMENDED.items():
        task = T.get_task(task_name)
        C = T.standard_contexts(task)
        evalc = C["eval"]
        best = [best_eval[task_name]["best"][c["cid"]] for c in evalc]
        row = {"task": task_name, "application": LABEL[task_name], "recommended_arm": arm_rec}
        for arm in ("linear", "guided"):
            p = os.path.join(ROOT, "policies", f"{task_name}--{arm}.json")
            if os.path.exists(p):
                pol = FrozenPolicy.load(p)
                row[f"{arm}_share"] = share([pol.score(c) for c in evalc], best)
                row["base_share"] = share([task.score(pol.base, c) for c in evalc], best)
        tb = best_eval[task_name].get("textbook") or {}
        if tb:
            row["textbook"] = tb["rule"]
            row["textbook_share"] = share([tb["scores"][c["cid"]] for c in evalc], best)
        if task_name in RUN_EVAL:
            ev = json.load(open(os.path.join(runs, RUN_EVAL[task_name], "results.json")))["eval"]
            row["cem_ctx_lin_share"] = share(ev["CEM_ctx_lin"], best)
            if "oraclefit_lin" in ev:
                row["oraclefit_lin_share"] = share(ev["oraclefit_lin"], best)
        elif task_name == "three_asset_transition":
            ev = json.load(open(os.path.join(runs, "exp", "three_asset", "state.json")))["eval"]
            row["cem_ctx_lin_share"] = share(ev["CEM_ctx_lin"], best)
            row["oraclefit_lin_share"] = share(ev["oraclefit_lin"], best)
        elif task_name == "alpha_liquidation":
            tab = json.load(open(os.path.join(runs, "exp3", "voltgt", "result.json")))["part2_alpha_liquidation"]["table"]["eval32"]
            row["cem_ctx_lin_share"] = float(tab["cem_lin_40"]["share"])
            row["cem_ctx_lin_note"] = "same-budget CEM, 40 updates"
        if task_name in ANALYTIC_512:
            seed, prefix = ANALYTIC_512[task_name]
            fresh = task.make_contexts(512, seed, prefix)
            if task_name == "alpha_liquidation":
                bf = json.load(open(os.path.join(runs, "exp3", "voltgt", "alpha_liquidation", "best_fresh.json")))
                bfresh = [bf[c["cid"]]["score"] for c in fresh]
            else:
                bfresh = [best_known(task, c)[1] for c in fresh]
            pol = FrozenPolicy.load(os.path.join(ROOT, "policies", f"{task_name}--{arm_rec}.json"))
            row["recommended_share_512_fresh"] = share([pol.score(c) for c in fresh], bfresh)
            row["base_share_512_fresh"] = share([task.score(pol.base, c) for c in fresh], bfresh)
            # outside the trained range: |u| up to 1.6 (training box is |u| <= 1), service-equivalent clipping
            rng = np.random.default_rng(606)
            far = [T.context_from_u(task, rng.uniform(-1.6, 1.6, task.d), f"far-{i}") for i in range(256)]
            bfar = [best_known(task, c)[1] for c in far]
            row["recommended_share_out_of_range"] = share([pol.score(c) for c in far], bfar)
            row["base_share_out_of_range"] = share([task.score(pol.base, c) for c in far], bfar)
        rows.append(row)
        print(task_name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                          if k not in ("task", "application")}, flush=True)
    with open(os.path.join(ROOT, "results", "shortlist.json"), "w") as f:
        json.dump(rows, f, indent=1)
    cols = ["task", "application", "recommended_arm", "base_share", "linear_share", "guided_share", "cem_ctx_lin_share",
            "textbook", "textbook_share", "oraclefit_lin_share", "recommended_share_512_fresh", "base_share_512_fresh",
            "recommended_share_out_of_range", "base_share_out_of_range"]
    with open(os.path.join(ROOT, "results", "shortlist.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    # evidence for the pipeline choices: skeptic-checked rows of the third round + earlier checks
    S = json.load(open(os.path.join(runs, "exp3", "summary.json")))
    ev = {"source": "third-round prod experiments, each re-checked by an independent reviewer",
          "experiments": [{k: r[k] for k in ("id", "title", "sound", "headline", "key_numbers", "corrections")}
                          for r in S["rows"] if r["id"] in EVIDENCE_IDS],
          "large_sample_512_main_run": {k: v["pct_of_oracle"] for k, v in
                                        json.load(open(os.path.join(runs, "fx2", "local_expand.json"))).items()
                                        if k in ("ac_liquidation", "ow_transient", "two_asset_transition")},
          "out_of_range_u_le_1_6": {k: v for k, v in json.load(open(os.path.join(runs, "fx2", "extrapolate.json"))).items()
                                    if k in ("ac_liquidation", "ow_transient", "two_asset_transition")}}
    with open(os.path.join(ROOT, "results", "evidence.json"), "w") as f:
        json.dump(ev, f, indent=1)


if __name__ == "__main__":
    main()
