"""Rebuild policies/ and results/ from the original prod run files (provenance; needs the run directory).

usage: python scripts/export_from_runs.py --fin /path/to/fin

For each kept (task, arm) it takes the retained map's offsets on the 32 held-out contexts (saved in the
run state, or read back from the request journal), fits the service-equivalent map (basis.py), checks the
fit is exact, re-scores the held-out contexts offline and compares with the scores recorded during the run.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from machina_finance import tasks as T  # noqa: E402
from machina_finance.basis import fit_map  # noqa: E402
from machina_finance.pipeline import basis_for  # noqa: E402
from machina_finance.policy import FrozenPolicy  # noqa: E402

# (task, arm, updates, how to find base / offsets / stored scores)
SOURCES = [
    ("ac_liquidation", "linear", 20, {"fx2": "fx2/ac_liquidation"}),
    ("ac_liquidation", "guided", 20, {"journal": "exp/guided/ac_liquidation", "dom": "finm-x1-ac-rgd",
                                      "base": ("fx2/ac_liquidation", "base_actions"), "scores": ("final",)}),
    ("ow_transient", "linear", 20, {"fx2": "fx2/ow_transient"}),
    ("ow_transient", "guided", 20, {"journal": "exp/guided/ow_transient", "dom": "finm-x1-ow-rgd",
                                    "base": ("fx2/ow_transient", "base_actions"), "scores": ("final",)}),
    ("vwap_profile", "linear", 20, {"journal": "exp/finish/vwap_profile-fx2b-lin", "dom": "finm-fx2b-vw-rlin",
                                    "base": ("fx2b/vwap_profile", "base_actions"), "scores": ("final",),
                                    "oracle": "fx2b/vwap_profile"}),
    ("two_asset_transition", "linear", 20, {"fx2": "fx2/two_asset_transition"}),
    ("two_asset_transition", "guided", 40, {"journal": "exp/more/guided/two_asset_transition", "dom": "finm-x2-2a-rgd",
                                            "base": ("fx2/two_asset_transition", "base_actions"), "scores": ("at40",)}),
    ("three_asset_transition", "linear", 20, {"journal": "exp/three_asset", "dom": "finm-fx3-3a-rlin",
                                              "base": ("exp/three_asset", "base"), "scores_state": "exp/three_asset/refine",
                                              "scores": ("final",), "oracle": "fx3/three_asset_transition"}),
    ("three_asset_transition", "guided", 40, {"journal": "exp/more/guided/three_asset_transition", "dom": "finm-x2-3a-rgd",
                                              "base": ("exp/three_asset", "base"), "scores": ("at40",),
                                              "oracle": "fx3/three_asset_transition"}),
    ("market_making", "linear", 20, {"journal": "exp/finish/market_making-fx2-lin", "dom": "finm-fx2-mm-rlin",
                                     "base": ("fx2/market_making", "base_actions"), "scores": ("final",)}),
    ("hedge_bands", "linear", 20, {"journal": "exp/finish/hedge_bands-fx2-lin", "dom": "finm-fx2-hb-rlin",
                                   "base": ("fx2/hedge_bands", "base_actions"), "scores": ("final",)}),
    ("glide_path", "linear", 20, {"journal": "exp/finish/glide_path-fx2-lin", "dom": "finm-fx2-gp-rlin",
                                  "base": ("fx2/glide_path", "base_actions"), "scores": ("final",)}),
    ("alpha_liquidation", "guided", 40, {"voltgt": "exp3/voltgt/alpha_liquidation/al-rgd", "snap": "40",
                                         "oracle": "exp3/voltgt/alpha_liquidation"}),
]


def load_json(*p):
    with open(os.path.join(*p)) as f:
        return json.load(f)


def base_of(runs, spec):
    d, key = spec
    p = os.path.join(runs, d, "results.json")
    return load_json(p)[key] if os.path.exists(p) else load_json(runs, d, "state.json")[key]


def journal_offsets(path, dom, evalc):
    """Last retained-evaluate offsets per held-out context for one Domain."""
    F = np.array([c["feat"] for c in evalc])
    out = {}
    with open(path) as f:
        for line in f:
            q = json.loads(line)
            if q.get("path") != f"/domains/{dom}/trajectory/evaluate" or q.get("status") != 200:
                continue
            b = q.get("body") or {}
            if set(b) != {"context"}:
                continue
            j = np.where(np.abs(F - np.asarray(b["context"], float)).max(1) < 1e-12)[0]
            if len(j):
                out[evalc[j[0]]["cid"]] = np.asarray(q["resp"]["offsets"], float).ravel()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fin", required=True)
    a = ap.parse_args()
    runs = os.path.join(a.fin, "runs")
    os.makedirs(os.path.join(ROOT, "policies"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    best_all, rows = {}, []
    for task_name, arm, updates, src in SOURCES:
        task = T.get_task(task_name)
        C = T.standard_contexts(task)
        evalc, sup = C["eval"], [c["feat"] for c in C["support"]]
        if "fx2" in src:
            res = load_json(runs, src["fx2"], "results.json")
            st = res["stages"]["R1_linear"]["out"]
            assert np.allclose(np.array(st["config_resolved"]["contexts"]), np.array(sup), atol=0, rtol=0)
            base = res["base_actions"]
            Y = {h["cid"]: np.asarray(h["offsets"], float).ravel() for h in st["held_out"]}
            stored = [h["score"] for h in st["held_out"]]
            dom, orc_dir = st["domain"], src["fx2"]
        elif "voltgt" in src:
            s = load_json(runs, src["voltgt"], "state.json")
            h = s["heldout"][src["snap"]]
            assert np.allclose(np.array(s["config_resolved"]["contexts"]), np.array(sup), atol=0, rtol=0)
            base = np.zeros((task.H, task.A)).tolist()
            Y = {c["cid"]: np.asarray(o, float).ravel() for c, o in zip(evalc, h["offsets"])}
            stored, dom, orc_dir = h["scores"], s["domain"], src["oracle"]
        else:
            base = base_of(runs, src["base"])
            Y = journal_offsets(os.path.join(runs, src["journal"], "journal.jsonl"), src["dom"], evalc)
            s = load_json(runs, src.get("scores_state", src["journal"]), "state.json")
            stored = s[src["scores"][0]]["scores"]
            dom, orc_dir = src["dom"], src.get("oracle", src["base"][0])
        assert len(Y) == len(evalc), (task_name, arm, len(Y))
        basis = basis_for(arm, sup)
        Ymat = np.array([Y[c["cid"]] for c in evalc])
        W, resid, det = fit_map(basis.phi([c["feat"] for c in evalc]), Ymat)
        orc = load_json(runs, orc_dir, "oracle.json")
        best = {c["cid"]: float(orc[c["cid"]]["score"]) for c in evalc}
        best_all.setdefault(task_name, {"best": best, "textbook": {}})
        for rk in ("as_score", "ww_score", "blind_ac_score"):
            if rk in (orc[evalc[0]["cid"]].get("ref") or {}):
                best_all[task_name]["textbook"] = {"rule": {"as_score": "Avellaneda-Stoikov", "ww_score": "Whalley-Wilmott",
                                                            "blind_ac_score": "signal-blind AC"}[rk],
                                                   "scores": {c["cid"]: float(orc[c["cid"]]["ref"][rk]) for c in evalc}}
        pol = FrozenPolicy(task_name, base, basis, W, {})
        off = np.array([pol.score(c) for c in evalc])
        diff = float(np.abs(off - np.asarray(stored, float)).max())
        bmean = np.mean(list(best.values()))
        base_scores = [float(task.score(np.asarray(base, float), c)) for c in evalc]
        pol.meta = {"arm": arm, "updates": updates, "source_domain": dom, "fit_residual": resid, "fit_determined": det,
                    "heldout_contexts": len(evalc), "heldout_share_of_best": float(off.mean() / bmean),
                    "base_share_of_best": float(np.mean(base_scores) / bmean),
                    "max_abs_diff_vs_run_scores": diff,
                    "note": "retained map exported from the prod run; offline scores reproduce the run"}
        assert det and resid < 1e-9 and diff < 1e-9, (task_name, arm, resid, diff)
        pol.save(os.path.join(ROOT, "policies", f"{task_name}--{arm}.json"))
        rows.append({"task": task_name, "arm": arm, "updates": updates, "share_of_best": round(off.mean() / bmean, 4),
                     "base_share_of_best": round(float(np.mean(base_scores) / bmean), 4), "fit_residual": resid,
                     "max_abs_diff_vs_run_scores": diff, "stored_heldout_scores": [float(x) for x in stored]})
        print(f"{task_name:24s} {arm:7s} share {off.mean() / bmean:6.1%} base {np.mean(base_scores) / bmean:6.1%} "
              f"resid {resid:.1e} diff {diff:.1e}")
    with open(os.path.join(ROOT, "results", "best_known_eval.json"), "w") as f:
        json.dump(best_all, f, indent=1)
    with open(os.path.join(ROOT, "results", "policies_heldout.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
