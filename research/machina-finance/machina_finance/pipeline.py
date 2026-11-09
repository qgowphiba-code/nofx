"""The kept Machina pipeline: acquire at the nominal context -> contextual refinement -> export a frozen policy.

  acquire()  episode_credit learner at ONE fixed (nominal) context with the documented optional profile.
             Multi-context acquisition is deliberately absent: frozen recall returns the single best record
             whatever the context, so it never beat the naive benchmark (see README, "Dropped").
  refine()   contextual_execution learner around the acquired base. arm="linear" (default) or
             arm="guided" (normalize_context + quadratic basis + response_guidance). 8 candidates per update,
             16 contexts per update. mie=True (linear arm) identifies each candidate's affine map from a few
             evaluations and spot-checks it, instead of evaluating every candidate at every context.
  export_policy()  identifies the retained map through retained `evaluate` calls and returns a FrozenPolicy
             that reproduces the service offline (checked on extra contexts).
Outcomes are computed locally from the simulator; the API never sees the base schedule or the model.
"""
import concurrent.futures as cf
import math
import time

import numpy as np

from . import tasks as T
from .basis import Basis, d_optimal_subset, fit_map
from .policy import FrozenPolicy

OPTIONAL_PROFILE = {"coherent_edits": True, "global_recall": True, "executed_prefix_mutation": True,
                    "progress_guided_mutation": False, "sequence_compaction": False,
                    "outcome_tier_exploration": True, "retain_prefix_failures": False}
IDENT_BOX = 2.5   # identification contexts lie inside +-2.5 support SD in both metrics
TOL = 1e-9


def acquisition_config(task, seed=12, profile=True):
    cfg = {"mechanism": "episode_credit", "state_dimensions": 1 + task.d, "action_dimensions": task.A,
           "goal_indices": [0], "horizon": task.H, "capacity": 256, "max_pending": 1, "seed": seed,
           "outcome_mode": "reward"}
    if profile:
        cfg.update(OPTIONAL_PROFILE)
    return cfg


def refinement_config(task, base, support_feats, arm="linear", seed=21, candidate_count=8):
    if candidate_count < 8:
        raise ValueError("fewer than 8 candidates per update stalled refinement (3 -> ~47% of best); use >= 8")
    cfg = {"mechanism": "contextual_execution", "horizon": len(base), "action_dimensions": task.A,
           "contexts": [list(map(float, f)) for f in support_feats], "objective_names": ["quality"],
           "candidate_count": candidate_count, "max_contexts": 64, "seed": seed, "radius": 0.1,
           "retained_variance": 0.99, "contextual": True, "action_trust_region": True}
    if arm == "guided":
        cfg.update(normalize_context=True, context_basis="quadratic", retained_variance=1.0, response_guidance=True)
    elif arm != "linear":
        raise ValueError("arm must be 'linear' or 'guided'")
    return cfg


def basis_for(arm, support_feats):
    return Basis(support_feats, "linear" if arm == "linear" else "quadratic")


def _rows(actions, task):
    return np.clip(np.asarray(actions, float).reshape(-1, task.A)[: task.H], -1, 1)


def compose(base, offsets):
    base = np.asarray(base, float)
    return np.clip(base + np.asarray(offsets, float).reshape(base.shape), -1, 1)


# ------------------------------------------------------------------------------------------- acquisition
def acquire(client, task, domain, ctx, episodes, seed=12, profile=True, checkpoints=(), score_fn=None,
            proposal_seed=0, log=print):
    """Learn a schedule from scratch at one fixed context. Returns curve, frozen checkpoints and incumbent."""
    score_fn = score_fn or (lambda a, c, ep: task.score(a, c))
    receipt, _ = client.create_stage(domain, acquisition_config(task, seed, profile),
                                     description=f"machina-finance acquisition {task.name}")
    out = {"domain": domain, "config": receipt.get("config"), "curve": [], "frozen": []}
    state0 = task.acq_state(ctx, 0)
    for ep in range(episodes):
        p = client.traj(domain, "propose", {"state": state0, "goal": [0.0], "mode": "explore",
                                            "request_id": f"{domain}-e{ep:05d}", "seed": proposal_seed + ep + 1})
        a = _rows(p["actions"], task)
        s = float(score_fn(a, ctx, ep))
        client.traj(domain, "observe", {"decision_id": p["decision_id"], "states": T.trace_states(task, a, ctx),
                                        "actions": a.tolist(), "outcome": s})
        out["curve"].append(s)
        if (ep + 1) in checkpoints:
            fa = frozen_actions(client, task, domain, state0, f"{domain}-frozen-{ep + 1}")
            out["frozen"].append({"ep": ep + 1, "score": float(task.score(fa, ctx)), "actions": fa.tolist()})
            log(f"  acquisition {ep + 1}/{episodes}: frozen score {out['frozen'][-1]['score']:.4f}")
    inc = frozen_actions(client, task, domain, state0, f"{domain}-final")
    out["incumbent"] = inc.tolist()
    out["incumbent_score"] = float(task.score(inc, ctx))
    return out


def frozen_actions(client, task, domain, state, request_id):
    fz = client.traj(domain, "propose", {"state": state, "goal": [0.0], "mode": "frozen", "request_id": request_id})
    return _rows(fz["actions"], task)


# -------------------------------------------------------------------------------------------- refinement
def crn(noisy_score, n_days, seed=0):
    """Common random numbers for noisy simulators: every candidate in one update is scored on the same
    n_days simulated days. noisy_score(actions, ctx, day_seed) -> score. Fresh independent noise per
    candidate kept ~0% of the gain in our runs; 100 / 10 / 1 shared days kept 91 / 73 / 37%."""
    def f(actions, ctx, update):
        return float(np.mean([noisy_score(actions, ctx, (seed * 1_000_003 + update * 10_007 + k) % 2 ** 32)
                              for k in range(n_days)]))
    return f


def draw_contexts(train, n_ctx, seed, updates):
    rng = np.random.default_rng(seed)
    return [[train[i] for i in rng.choice(len(train), n_ctx, replace=False)] for _ in range(updates)]


def refine(client, task, domain, base, train, support, updates, arm="linear", seed=21, n_ctx=16, mie=True,
           score_fn=None, threads=2, start=0, create=True, log=print):
    """Run refinement updates start..updates-1. create=False continues an existing learner (e.g. after a
    detected shift: continuing the same lineage recovered faster than a fresh learner in our runs)."""
    base = np.asarray(base, float)
    sup = [c["feat"] for c in support]
    basis = basis_for(arm, sup)
    use_mie = mie and arm == "linear"
    score_fn = score_fn or (lambda a, c, u: task.score(a, c))
    if create:
        client.create_stage(domain, refinement_config(task, base, sup, arm, seed),
                            description=f"machina-finance refinement {task.name} {arm}")
    draws = draw_contexts(train, n_ctx, seed, updates)
    log_rows = []
    for u in range(start, updates):
        ctxs = draws[u]
        cids = [c["cid"] for c in ctxs]
        body = {"state": sup[0], "goal": [0.0], "mode": "explore", "request_id": f"{domain}-u{u:03d}",
                "context_ids": cids, "discovery_context_ids": cids[::2]}
        if arm == "guided":
            body["context_observations"] = {c["cid"]: c["feat"] for c in ctxs}
        calls0 = client.calls
        p = client.traj(domain, "propose", body)
        did, K = p["decision_id"], len(p["candidates"])
        offs, info = _candidate_offsets(client, domain, did, K, ctxs, basis, use_mie, threads)
        res = [(ci, j, float(score_fn(compose(base, offs[ci, j]), c, u))) for ci in range(K) for j, c in enumerate(ctxs)]
        o = client.traj(domain, "observe", {"decision_id": did, "outcomes": [
            {"candidate_index": ci, "context_id": cids[j], "objectives": [s]} for ci, j, s in res]})
        upd = o.get("update", {}) if isinstance(o, dict) else {}
        cm = [float(np.mean([s for c2, _, s in res if c2 == ci])) for ci in range(K)]
        rec = {"u": u, "cand_means": cm, "selected": upd.get("selected"), "gain": upd.get("confirmation_gain"),
               "next_radius": upd.get("next_action_search_radius"), "calls": client.calls - calls0, **info}
        log_rows.append(rec)
        log(f"  update {u + 1}/{updates}: best candidate mean {max(cm):.4f}, calls {rec['calls']}"
            + (f", MIE fallbacks {info['fallbacks']}" if use_mie else ""))
    return log_rows


def _candidate_offsets(client, domain, did, K, ctxs, basis, use_mie, threads):
    F = np.array([c["feat"] for c in ctxs], float)
    ev = {}

    def evaluate(pairs):
        def one(pair):
            ci, j = pair
            e = client.traj(domain, "evaluate", {"context": ctxs[j]["feat"], "candidate_index": ci, "decision_id": did})
            return pair, np.asarray(e["offsets"], float).ravel()
        with cf.ThreadPoolExecutor(threads) as ex:
            for pair, off in ex.map(one, [p for p in pairs if p not in ev]):
                ev[pair] = off

    n = len(ctxs)
    if not use_mie:
        evaluate([(ci, j) for ci in range(K) for j in range(n)])
        return ev, {"mie": False, "fallbacks": 0}
    X = basis.phi(F)
    inside = [j for j in range(n) if basis.z_max(F[j])[0] <= IDENT_BOX]
    sub = d_optimal_subset(X[inside], basis.P) if len(inside) >= basis.P else None
    if sub is None:
        evaluate([(ci, j) for ci in range(K) for j in range(n)])
        return ev, {"mie": False, "fallbacks": K}
    ident = [inside[i] for i in sub]
    rest = [j for j in range(n) if j not in ident]
    spot = max(rest, key=lambda j: basis.z_max(F[j])[0]) if rest else None  # most outlying: exercises the clip
    evaluate([(ci, j) for ci in range(K) for j in ident + ([spot] if spot is not None else [])])
    out, fallbacks = {}, 0
    for ci in range(K):
        W, resid, det = fit_map(X[ident], np.array([ev[ci, j] for j in ident]))
        pred = basis.offsets(W, F)
        ok = det and resid < TOL and (spot is None or np.abs(pred[spot] - ev[ci, spot]).max() < TOL)
        if not ok:
            fallbacks += 1
            evaluate([(ci, j) for j in range(n)])
            for j in range(n):
                out[ci, j] = ev[ci, j]
            continue
        for j in range(n):
            out[ci, j] = ev[ci, j] if (ci, j) in ev else pred[j]
    return out, {"mie": True, "fallbacks": fallbacks, "ident": len(ident)}


# ---------------------------------------------------------------------------------------------- export
def export_policy(client, task, domain, base, support, arm, pool, verify, threads=2, meta=None):
    """Identify the retained map from retained evaluate calls and return (FrozenPolicy, check record).
    pool: contexts to pick identification points from (inside the support box); verify: extra contexts
    whose API offsets must match the exported policy to 1e-9 (include some outlying ones)."""
    base = np.asarray(base, float)
    basis = basis_for(arm, [c["feat"] for c in support])
    Fp = np.array([c["feat"] for c in pool], float)
    inside = [j for j in range(len(pool)) if basis.z_max(Fp[j])[0] <= IDENT_BOX]
    sub = d_optimal_subset(basis.phi(Fp[inside]), basis.P)
    if sub is None:
        raise RuntimeError("not enough well-spread identification contexts in the pool")
    ident = [pool[inside[i]] for i in sub]

    def retained(cs):
        with cf.ThreadPoolExecutor(threads) as ex:
            return list(ex.map(lambda c: np.asarray(client.traj(domain, "evaluate", {"context": c["feat"]})["offsets"],
                                                    float).ravel(), cs))
    fp0 = client.traj(domain, "state").get("fingerprint")
    Y = np.array(retained(ident))
    W, resid, det = fit_map(basis.phi([c["feat"] for c in ident]), Y)
    if not det:  # some outputs clipped at +-1: add the rest of the inside pool and refit
        extra = [pool[j] for j in inside if pool[j] not in ident]
        ident += extra
        Y = np.vstack([Y, retained(extra)])
        W, resid, det = fit_map(basis.phi([c["feat"] for c in ident]), Y)
    Yv = np.array(retained(verify)) if verify else np.zeros((0, base.size))
    fp1 = client.traj(domain, "state").get("fingerprint")
    pol = FrozenPolicy(task.name, base, basis, W, meta)
    vmax = float(np.abs(basis.offsets(W, [c["feat"] for c in verify]) - Yv).max()) if verify else 0.0
    check = {"ident_contexts": len(ident), "fit_residual": resid, "determined": bool(det), "verify_contexts": len(verify),
             "verify_max_abs_diff": vmax, "exact": bool(det and resid < TOL and vmax < TOL),
             "fingerprint_unchanged": fp0 == fp1}
    pol.meta.setdefault("export_check", check)
    return pol, check


# ---------------------------------------------------------------------------------------------- drift
def cusum_alarm(realized, expected, scale, k=0.5, h=5.0):
    """One-sided CUSUM on standardised excess cost (realized - expected) / scale. Returns the index of the
    first alarm or None. On an alarm, re-run refine(..., create=False) on the SAME learner with the new data."""
    s = 0.0
    for i, (r, e) in enumerate(zip(realized, expected)):
        s = max(0.0, s + (r - e) / scale - k)
        if s > h:
            return i
    return None


# ------------------------------------------------------------------------------------------- end to end
def run_pipeline(client, task_name, prefix, episodes=300, updates=20, arm="linear", acq_seed=12, ref_seed=21,
                 mie=True, threads=2, check_eval_on_api=False, log=print):
    """acquire (nominal) -> refine -> export -> offline held-out evaluation. Deletes every Domain it creates."""
    task = T.get_task(task_name)
    C = T.standard_contexts(task)
    t0 = time.time()
    out = {"task": task_name, "arm": arm, "episodes": episodes, "updates": updates, "domains": []}
    acq_dom, ref_dom = f"{prefix}-acq", f"{prefix}-r{arm[:3]}"
    deleted = set()

    def delete(d):
        client.delete_domain(d)
        deleted.add(d)
    try:
        out["domains"].append(acq_dom)
        cps = tuple(sorted({min(episodes, x) for x in (50, 100, 200, 300, episodes)}))
        acq = acquire(client, task, acq_dom, C["nominal"], episodes, seed=acq_seed, checkpoints=cps, log=log)
        delete(acq_dom)
        out["acquisition"] = {k: acq[k] for k in ("frozen", "incumbent", "incumbent_score")}
        base = np.asarray(acq["incumbent"], float)
        out["domains"].append(ref_dom)
        out["refinement"] = refine(client, task, ref_dom, base, C["train"], C["support"], updates, arm=arm,
                                   seed=ref_seed, mie=mie, threads=threads, log=log)
        verify = sorted(C["eval"], key=lambda c: -basis_for(arm, [s["feat"] for s in C["support"]]).z_max(c["feat"])[0])[:4]
        pol, check = export_policy(client, task, ref_dom, base, C["support"], arm, C["train"], verify, threads,
                                   meta={"source": "run_pipeline", "domain": ref_dom, "updates": updates,
                                         "episodes": episodes, "arm": arm})
        out["export_check"] = check
        if check_eval_on_api:
            api = [np.asarray(client.traj(ref_dom, "evaluate", {"context": c["feat"]})["offsets"], float).ravel()
                   for c in C["eval"]]
            out["eval_api_vs_policy_max_abs_diff"] = float(max(np.abs(pol.offsets(c["feat"]).ravel() - a).max()
                                                              for c, a in zip(C["eval"], api)))
        delete(ref_dom)
    finally:
        for d in out["domains"]:
            if d in deleted:
                continue
            try:
                delete(d)
            except Exception as e:  # report, never hide
                out.setdefault("cleanup_errors", []).append(f"{d}: {e!r}")
        out["deleted"] = sorted(deleted)
    out["policy"] = pol.to_dict()
    out["calls"] = client.calls
    out["secs"] = round(time.time() - t0, 1)
    return out, pol


def offline_report(pol, contexts, best_scores, base=None):
    task = pol.task
    s = np.array([pol.score(c) for c in contexts])
    b = np.array(best_scores, float)
    rep = {"n": len(s), "refined_mean": float(s.mean()), "refined_share": float(s.mean() / b.mean())}
    if base is not None:
        sb = np.array([task.score(np.asarray(base, float), c) for c in contexts])
        rep.update(base_mean=float(sb.mean()), base_share=float(sb.mean() / b.mean()))
    return rep


def share(scores, best):
    return float(np.mean(scores) / np.mean(best)) if len(scores) else math.nan
