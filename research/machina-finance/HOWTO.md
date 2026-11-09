# How to use machina-finance

This guide goes step by step, in the order you will need the steps. Run every command from the package folder
(`research/machina-finance`).

- **Offline steps** (sections 1–3, 5 and 6) need no key and no network.
- **Live steps** (sections 4 and 7) use Adapt-1 API credits. You can rehearse each of them first, for free,
  against an in-memory fake API by adding `--dry-run` or by setting `MACHINA_FINANCE_FAKE=1`. The fake follows
  the same flow and makes the same calls as a real run, but what it "learns" is meaningless.

Every command and Python snippet below is checked by `scripts/check_howto.py`. The checker runs the live steps
against the fake.

## 1. Install

You need Python 3.10 or newer. The only dependencies are numpy and scipy (the virtual environment comes to
about 250 MB); the API client uses only the standard library.

```bash
cd research/machina-finance
```

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -e .
```

To also run the tests:

```bash
pip install -e '.[test]'
```

## 2. Use a saved policy offline

A **policy** is a frozen schedule rule: give it a market condition and it returns the controls (for example, how
much of an order to trade in each slot). Nine tasks have saved policies in `policies/<task>--<arm>.json`. The
arm is `linear` (default refinement) or `guided` (quadratic and response-guided, see section 4).

### See what is there

```bash
python -m machina_finance list
```

For each task this prints the recommended policy's **share of best**, measured on 32 held-out market conditions
(0% = naive benchmark, 100% = best known schedule). Next to it are the schedule before refinement, same-budget
CEM search, and the textbook rule where one exists.

### What each task's inputs and outputs mean

```bash
python -m machina_finance describe --task ac_liquidation
```

| Task | Market parameters you give (training range) | Controls you get back |
|---|---|---|
| `ac_liquidation` | `sigma` volatility over the trading horizon (0.01–0.04, 0.02 = 2%); `eta` temporary impact, a slot costs eta·n²/τ (0.001–0.01); `lam` risk aversion (about 0.4–1,500) | fraction of the order to trade in each of 10 equal slots (sums to 1) |
| `ow_transient` | `halflife` of the impact decay, in units of the horizon (0.05–3); `temp_ratio` instantaneous vs transient impact (0.01–0.3); `lam` risk aversion (0.005–1) | fraction per slot, 10 slots |
| `vwap_profile` | `open` and `close` share of the day's volume in the first and last bin (0.03–0.15, 0.04–0.45); `curv` U-shape of the profile (0.3–3); `size` order as a fraction of the day's volume (0.01–0.2) | fraction per intraday bin, 13 bins |
| `two_asset_transition` | `rho` correlation (−0.3–0.9); `eta_ratio` impact of B relative to A (0.25–4); `lam` risk aversion (2–200); `sig_ratio` volatility of B relative to A (0.5–2) | per slot, amount of A sold and of B bought (8 slots, each leg sums to 1) |
| `three_asset_transition` | `rho_ab`, `rho_bc` correlations (−0.2–0.9); `eta_ratio` (0.25–4); `lam` (2–200) | per slot, A sold (total 1), B and C bought (0.5 each) |
| `market_making` | `sigma` mid-price volatility over the session, price units (1–3); `A` order-arrival intensity (70–280); `k` fill-probability decay per price unit (0.75–3); `gamma` risk aversion (0.02–0.3) | per period (8): half-spread in price units (0.15–6) and inventory skew (0–3). Bid = mid − (half_spread + skew·q), ask = mid + (half_spread − skew·q), q = inventory |
| `hedge_bands` | `vol` annualised volatility (0.15–0.5); `cost` proportional transaction cost (0.0005–0.004); `K` strike with spot 100 (90–110); `gamma` risk aversion (0.02–0.5) | per 5-day period (8): half-width of the no-trade band around the Black–Scholes delta (0–0.3) |
| `glide_path` | `gamma` relative risk aversion (2–10); `ep` equity premium a year (0.02–0.07); `vol` equity volatility a year (0.12–0.24); `w0` starting wealth in years of savings (0–10) | equity weight for each 4-year block over 40 years (0–1) |
| `alpha_liquidation` | as `ac_liquidation` plus `a`, the alpha signal as a drift-justified holding level (−0.75–1.5; below 0 sell faster, above about 0.6 hold back) | fraction per slot, 10 slots |

The learner does not see these parameters directly. It sees **features**: log-scaled, centred versions of them,
listed by `describe`. You never compute features yourself; the package does it.

### Get a schedule for one market condition

```bash
python -m machina_finance schedule --policy policies/ac_liquidation--linear.json --params sigma=0.03,eta=0.005,lam=10 --best
```

Parameters you leave out stay at their nominal (middle) value. How to read the JSON output:

- `params`: the full condition that was used.
- `range_check`: `ok: true` means the condition is inside what the policy was trained on (see section 6).
- `controls`: the schedule, labelled. `controls_meaning` says what the numbers are.
- `score`: the policy's score for this condition. It is the share of the naive benchmark's cost saved; for
  `ac_liquidation`, 0.035 means 3.5% of TWAP's risk-adjusted cost was saved. `base_score` is the same for the
  schedule before refinement.
- `best_known_score` and `share_of_best` appear only with `--best`. Omit `--best` to skip the search for the
  best schedule. That search takes seconds for the execution tasks and up to a minute for market making,
  hedging and the glide path.

The same from Python:

```python
from machina_finance import tasks as T
from machina_finance.policy import FrozenPolicy

pol = FrozenPolicy.load("policies/ac_liquidation--linear.json")
ctx = T.context_from_params(pol.task, {"sigma": 0.03, "eta": 0.005, "lam": 10})
print(pol.range_check(ctx))
print(T.controls_table(pol.task, pol.controls(ctx)))
print("share of TWAP's cost saved:", round(pol.score(ctx), 4))
```

## 3. Re-check the stored results offline

Re-score all 13 saved policies on the 32 held-out conditions and compare them with the scores recorded during
the prod runs. It should end with `OK`; differences are below 1e-9.

```bash
python -m machina_finance verify
```

Run the test suite (offline; it includes a fake API for the pipeline plumbing):

```bash
python -m pytest -q
```

Score a policy on fresh conditions drawn from the training range. This is quick for the execution tasks, whose
best-known schedule is exact; it is slow for market making, hedging and the glide path, which need a search:

```bash
python -m machina_finance evaluate --policy policies/two_asset_transition--guided.json --fresh 128
```

`scripts/export_from_runs.py` and `scripts/build_results.py` rebuilt `policies/` and `results/` from the raw prod
run folders. Those folders are not included, so these two scripts are provenance only.

## 4. Train your own on the API, end to end

### Credentials

The client reads the key from the `ADAPT1_API_KEY` environment variable and sends it as
`Authorization: Bearer <key>`, the documented format. Never paste the key into files, chat or a command line,
where it would land in shell history. Read it with hidden input instead:

<!-- check: skip (interactive) -->
```bash
read -rs ADAPT1_API_KEY
```

<!-- check: skip (interactive) -->
```bash
export ADAPT1_API_KEY
```

The key is never written to the request journal. `ADAPT1_BASE_URL`, `ADAPT1_AUTH_HEADER` and
`ADAPT1_AUTH_SCHEME` override the URL, the header name and the scheme if you ever need to. Check that the key
has credits (one call):

<!-- check: live -->
```bash
python -m machina_finance probe
```

### Rehearse first (free)

```bash
python -m machina_finance run --task ac_liquidation --dry-run --quiet
```

The dry run prints the call count of every stage. Its policy is marked `"fake_api": true`; do not use it.

### Run it

<!-- check: live -->
```bash
python -m machina_finance run --task ac_liquidation --episodes 300 --updates 20 --max-calls 2000
```

What happens, in order:

1. **Probe.** One call checks the key has credits.
2. **Acquisition.** A learner is created and learns a schedule from scratch at the **nominal** market
   condition: 300 episodes, each one a proposal plus the simulated score. Its frozen schedule becomes the
   **base**, and the learner is deleted. Acquisition is deliberately done at one condition only. Learning from
   scratch across many conditions never beat the naive benchmark (README, "Dropped").
3. **Refinement.** A second learner adjusts the base to each condition over 20 updates. Each update proposes 8
   candidate adjustments, scores each one on 16 training conditions in your simulator, and reports the scores
   back.
4. **Export.** A few calls identify the learner's final adjustment map exactly. The result is
   `runs/<prefix>/policy.json`, which needs no API at runtime.
5. **Cleanup.** The learner is deleted. The policy is then scored offline on the 32 held-out conditions
   (`heldout` in the output).

Each run writes these files to `runs/<prefix>/`:
- `result.json`: per-stage call counts, the export check and the held-out score.
- `journal.jsonl`: every request and response, without the key.
- `ledger.jsonl`: every Domain the run created and deleted.

**Budget.** Call counts per stage, from dry runs. A real run adds a few calls for automatic retries after rate
limits (HTTP 429), and 11–13 calls whenever the cheap-evaluation spot check falls back to a full evaluation (it
did not happen in the live smoke test).

| Stage | Calls | Notes |
|---|---|---|
| Probe | 1 | |
| Acquisition, 300 episodes | 610 | 2 per episode, plus create, checkpoints and delete. Market making needs `--episodes 1000`: 2,011 calls |
| Refinement, linear, 20 updates | 844 (3 features) or 1,004 (4 features) | 42 or 50 per update, thanks to MIE (cheap evaluation) |
| Refinement, linear, `--no-mie` | 2,604 | 130 per update; same results, 3× the calls |
| Refinement, guided, `--updates 40` | 5,204 | 130 per update; MIE applies to the linear arm only |
| Export | 10–11 linear, 21 guided | |
| `--check-api` | +32 | also compares the policy with the API on all 32 held-out conditions |
| Cleanup | 1 | |

Totals:
- AC liquidation, linear: 1,466 calls.
- Two-leg switch, linear: 1,627; guided with 40 updates: 5,837.
- Market making with 1,000 episodes: 3,028.

`--max-calls` is a hard cap. Work stops 20 calls before it, and those 20 are kept for deleting the Domains.

### Linear or guided?

- **Start with linear** (the default). It costs about 1,500 calls, worked on every kept task, and stayed
  sensible outside the trained range.
- **Use guided** (`--arm guided --updates 40`) when the best schedule's shape bends with conditions and you can
  spend about 5,800 calls. That was the case for multi-leg switches, transient impact and the alpha signal.
  Guided went past the best possible linear rule there (two-leg 95% vs 84%, three-leg 89% vs 82%), but each of
  those results comes from a single learner seed.
- **Avoid guided** when you have few or noisy conditions. On real-data volatility targeting it overfit (0.21
  against 0.43 for linear).
- **Keep at least 8 candidates per update** (the default). With 3, refinement stalled near 47% of best.

### Clean up and confirm nothing is left

`run` deletes every Domain it creates, even when it fails part-way. To confirm:

<!-- check: live -->
```bash
python -m machina_finance domains --prefix finm-pkg
```

To delete leftovers with that prefix:

<!-- check: live -->
```bash
python -m machina_finance domains --prefix finm-pkg --delete
```

### The 32-learner limit

Each owner can hold at most 32 configured learners. `run` holds at most one at a time: the acquisition learner
is deleted before refinement starts. If you hit the limit, creating a Domain returns HTTP 409. The client then
waits up to 20 minutes for a free slot before giving up. Learners you keep with `--keep` (section 7) count
toward the limit until you delete them.

## 5. Bring your own simulator or cost model

All the pipeline needs from you is a **task**: a simulator that scores a schedule in a given market condition.
Subclass `tasks.Task` and fill in:

| Member | What it is |
|---|---|
| `name`, `H`, `A`, `d` | task id; steps per schedule; controls per step; number of context features |
| `feature_names`, `kind` | names of the features; `"cost"` (lower is better) or `"utility"` |
| `params_from_u(u)` | maps u in [−1, 1]^d to market parameters (this defines the training range) |
| `features(p)` | the d numbers the learner sees, scaled to roughly [−1, 1] |
| `decode(a)` | actions (H × A, each in [−1, 1]) → your real controls |
| `objective(controls, ctx)` | cost (or utility) of those controls in condition `ctx` (`ctx["p"]` holds the parameters) |
| `naive_actions(ctx)` *(optional)* | your benchmark schedule; default all zeros (for `decode` above, equal slices) |
| `best_actions(ctx)` *(optional)* | your own optimiser, used for the "share of best"; otherwise a generic search is used |

The score is (benchmark cost − cost) / |benchmark cost|, so 0 is the benchmark.

**How contexts are drawn.** A Latin hypercube in u is passed through `params_from_u`, then `features`. The
standard sets are:
- 48 training conditions (seed 101), used in refinement updates;
- 24 support conditions (seed 202), which fix the learner's feature scaling and the ±3 SD clipping box;
- 32 held-out conditions (seed 303), for scoring;
- the nominal condition, u = 0.

Make sure your training range covers the conditions you will trade in.

Worked example: an order whose last slot is the closing auction, with cheaper impact there. Save this as
`my_task.py` in the package folder:

```python
# my_task.py
import numpy as np

from machina_finance import tasks as T


class CloseAuctionLiquidation(T.Task):
    """Sell one order in 6 slots; the last slot is the closing auction (cheaper impact, but you wait for it)."""
    name = "close_auction_liquidation"
    H, A, d = 6, 1, 2                      # 6 slots, 1 control per slot, 2 context features
    feature_names = ["log_sigma", "auction_discount"]
    kind = "cost"                          # lower objective is better

    def params_from_u(self, u):            # u in [-1, 1]^d -> market parameters
        return {"sigma": T.lerp_log(u[0], 0.01, 0.04), "disc": T.lerp(u[1], 0.2, 0.9)}

    def features(self, p):                 # what the learner sees, scaled to about [-1, 1]
        return [float(np.log(p["sigma"] / 0.02) / np.log(2)), (p["disc"] - 0.55) / 0.35]

    def decode(self, a):                   # actions (H x A, in [-1, 1]) -> fraction traded per slot
        return T.schedule_weights(a[:, 0], self.H)

    def objective(self, n, ctx):           # risk-adjusted cost in bps of notional
        p = ctx["p"]
        eta = np.full(self.H, 0.005)
        eta[-1] *= 1 - p["disc"]           # the auction has cheaper impact
        x = 1 - np.cumsum(n)[:-1]          # holdings left after each slot
        return float((self.H * np.sum(eta * n ** 2) + 10 * p["sigma"] ** 2 * np.sum(x ** 2)) * 1e4)


T.TASKS[CloseAuctionLiquidation.name] = CloseAuctionLiquidation

if __name__ == "__main__":
    from machina_finance.references import best_known

    t = T.get_task("close_auction_liquidation")
    C = T.standard_contexts(t)
    fixed, _, _ = best_known(t, C["nominal"])            # best single schedule, tuned at the nominal condition
    best = np.array([best_known(t, c)[1] for c in C["eval"]])
    blind = np.array([t.score(fixed, c) for c in C["eval"]])
    print(f"best schedule per condition saves {best.mean():.1%} of TWAP's cost on average")
    print(f"one fixed schedule reaches {blind.mean() / best.mean():.0%} of that")
    print(f"headroom for context-aware refinement: {1 - blind.mean() / best.mean():.0%}")
```

Running it is a cheap **fit check**: if one fixed schedule already reaches nearly 100% of the best, the context
does not matter and refinement has nothing to add. Here the headroom is about a third:

```bash
python my_task.py
```

To train it, save this as `train_close.py`:

```python
# train_close.py
import my_task  # noqa: F401  (registers the task)
from machina_finance.client import Client
from machina_finance.pipeline import run_pipeline

client = Client(journal_path="runs/close/journal.jsonl", max_calls=2000)
out, pol = run_pipeline(client, "close_auction_liquidation", "finm-pkg-close", episodes=300, updates=20)
pol.save("runs/close/policy.json")
print(out["calls_by_stage"], "| export exact:", out["export_check"]["exact"])
```

Rehearse it on the fake API first:

```bash
MACHINA_FINANCE_FAKE=1 python train_close.py
```

For the real run, set the key (section 4) and drop the `MACHINA_FINANCE_FAKE=1`.

### Noisy simulators: score candidates on shared scenarios

If your simulator draws random fills or prices, compare every candidate in an update on the **same** simulated
days (common random numbers). Shared days kept 91%, 73% and 37% of the noise-free gain with 100, 10 and 1 days;
fresh noise for each candidate kept about 0%. `pipeline.crn` wraps a noisy scorer:

```python
import numpy as np

from machina_finance import tasks as T
from machina_finance.pipeline import crn

t = T.get_task("ac_liquidation")


def noisy_score(actions, ctx, day_seed):
    """Score of one simulated day: the price path is random, impact is not."""
    p = ctx["p"]
    n = t.decode(T.as_rows(actions, t.H, t.A))
    x = 1 - np.cumsum(n)[:-1]
    tau = t.T / t.H
    shocks = np.random.default_rng(day_seed).standard_normal(t.H - 1)
    impact = p["eta"] / tau * np.sum(n ** 2)
    timing = float(np.sum(x * p["sigma"] * np.sqrt(tau) * shocks))
    cost = (impact + timing + p["lam"] * timing ** 2) * 1e4
    return (t.naive_obj(ctx) - cost) / abs(t.naive_obj(ctx))


score_fn = crn(noisy_score, n_days=10, seed=7)   # 10 shared simulated days per update
ctx = t.nominal()
front_loaded = T.weights_to_actions(np.linspace(2, 1, t.H))[:, None]
even = np.zeros((t.H, 1))
print("update 0:", round(score_fn(front_loaded, ctx, 0), 4), round(score_fn(even, ctx, 0), 4))
print("same days again:", score_fn(front_loaded, ctx, 0) == score_fn(front_loaded, ctx, 0))
```

Pass it as `refine(..., score_fn=score_fn)`. For acquisition, use one fixed scenario set for every episode:
`acquire(..., score_fn=lambda a, c, ep: score_fn(a, c, 0))`. Fixed scenarios reached 95% of best in our runs.

### Good fit, poor fit

**Good fit:**
- a few continuous controls per step and one score at the end;
- measurable pre-trade context that changes the best schedule (check the headroom as above);
- a simulator you trust;
- learning from scratch done at one nominal condition.

**Poor fit** (all tried and dropped; see the README's "Dropped and why"):
- learning from scratch across many conditions;
- one learner reacting within the episode (closed loop);
- learning from one noisy outcome per candidate;
- regime changes the learner cannot observe;
- real-data tasks without a calibrated simulator;
- very wide action spaces such as 32–48 controls, where Powell caught up.

## 6. Deploy the frozen policy

A policy file is plain JSON:

| Field | Meaning |
|---|---|
| `format` | `machina-finance-policy/1` |
| `task`, `feature_names` | which simulator and features it belongs to |
| `base` | the schedule before refinement (H × A actions in [−1, 1]) |
| `basis` | `kind` (`linear` or `quadratic`), `clip_z` (3.0), and support-set statistics `mu`, `sd`, `Vt`, `sdp` |
| `W` | the exported adjustment map (one row per basis term, one column per action) |
| `meta` | provenance and the checks run at export |

At runtime: actions = clip(base + clip(φ(clipped features) · W, −1, 1), −1, 1). The task's `decode` then turns
the actions into controls. No API call is involved.

With the package, use `FrozenPolicy.load(...).controls(ctx)` as in section 2. Without the package (for example,
to port to another language), only numpy and the JSON file are needed. This snippet reimplements the runtime
and checks it against the package:

```python
import json

import numpy as np


def load_policy(path):
    with open(path) as f:
        d = json.load(f)
    b = d["basis"]
    return {"kind": b["kind"], "clip_z": b["clip_z"], "mu": np.array(b["mu"]), "sd": np.array(b["sd"]),
            "Vt": np.array(b["Vt"]), "sdp": np.array(b["sdp"]), "W": np.array(d["W"]), "base": np.array(d["base"])}


def policy_actions(P, features):
    f = np.asarray(features, float)
    if P["kind"] == "linear":             # clip at +-3 SD in the support's PCA-whitened coordinates
        z = np.clip((f - P["mu"]) @ P["Vt"].T / P["sdp"], -P["clip_z"], P["clip_z"])
        phi = np.concatenate([[1.0], P["mu"] + (z * P["sdp"]) @ P["Vt"]])
    else:                                 # clip per-feature z-scores, then add the quadratic terms
        z = np.clip((f - P["mu"]) / P["sd"], -P["clip_z"], P["clip_z"])
        iu = np.triu_indices(len(z))
        phi = np.concatenate([[1.0], z, np.outer(z, z)[iu]])
    offsets = np.clip(phi @ P["W"], -1, 1).reshape(P["base"].shape)
    return np.clip(P["base"] + offsets, -1, 1)


P = load_policy("policies/ac_liquidation--linear.json")
a = policy_actions(P, [0.585, 0.4, -0.3])
weights = np.exp(3.0 * a[:, 0]) / np.exp(3.0 * a[:, 0]).sum()   # ac_liquidation's decoder: fraction per slot
print(weights.round(4))

from machina_finance.policy import FrozenPolicy  # noqa: E402  (only to show both give the same answer)
print("matches the package:", np.allclose(a, FrozenPolicy.load("policies/ac_liquidation--linear.json").actions([0.585, 0.4, -0.3])))
```

### Guard against conditions outside the training range

Inside the trained range, the saved policies kept their quality on 512 fresh conditions. Outside it, results
vary:
- held up well: 78% for AC, 90% for the two-leg switch;
- fell to 6% of best: the transient-impact policy.

Beyond ±3 support SD the policy also stops adapting, because the context is clipped. So check every condition
before using the policy, and decide what to do when the check fails. Here the fallback is the naive schedule
plus an alert; then re-refine with a support set that covers the new conditions:

```python
from machina_finance import tasks as T
from machina_finance.policy import FrozenPolicy

pol = FrozenPolicy.load("policies/ow_transient--guided.json")
for params in ({"halflife": 0.5}, {"halflife": 20.0}):
    ctx = T.context_from_params(pol.task, params)
    check = pol.range_check(ctx)
    if check["ok"]:
        plan = pol.controls(ctx)
    else:
        plan = pol.task.decode(pol.task.naive_actions(ctx))   # fall back to TWAP and raise an alert
        print("ALERT: outside the trained range:", check["outside_training_range"], "clipped:", check["clipped"])
    print(params, plan.round(3))
```

## 7. Monitor drift and re-refine the same learner

A frozen policy is only as good as the simulator it was trained in. Compare each day's measured cost with what
the simulator expected for the schedule you sent. Compare like with like: for example, realized implementation
shortfall against the model's expected impact cost. A one-sided CUSUM raises an alarm when costs run
persistently high:

```python
import numpy as np

from machina_finance import tasks as T
from machina_finance.pipeline import cusum_alarm
from machina_finance.policy import FrozenPolicy

pol = FrozenPolicy.load("policies/ac_liquidation--linear.json")
t = pol.task
ctx = T.context_from_params(t, {"sigma": 0.02})
n = pol.controls(ctx)
expected = ctx["p"]["eta"] / (t.T / t.H) * np.sum(n ** 2) * 1e4    # expected impact cost of this schedule, bps
rng = np.random.default_rng(0)
realized = expected + rng.normal(0, 10, 60)                          # 60 days of measured costs, noise sd 10 bps
realized[30:] += 0.4 * expected                                      # impact became 40% worse on day 30
print(f"expected {expected:.1f} bps; alarm on day {cusum_alarm(realized, [expected] * 60, scale=10)}")
```

When the alarm fires, recalibrate your simulator. Then **continue refining the same learner** rather than
starting a new one: after an unseen change in impact, continuing reached 67–69% of the new best within 20
updates, against 45–61% for a fresh learner. That needs the refinement learner to still exist, so train with
`--keep`. It then counts toward the 32-learner limit until you delete it.

<!-- check: live -->
```bash
python -m machina_finance run --task ac_liquidation --keep --prefix finm-pkg-ac-demo
```

Later, after a shift, save this as `rerefine.py`. It runs 10 more updates with the recalibrated simulator and
exports a new policy:

```python
# rerefine.py
from machina_finance import pipeline as PL
from machina_finance import tasks as T
from machina_finance.client import Client
from machina_finance.policy import FrozenPolicy

pol = FrozenPolicy.load("runs/finm-pkg-ac-demo/policy.json")    # saved by `run ... --keep`
t = pol.task
C = T.standard_contexts(t)
dom, done = pol.meta["domain"], pol.meta["updates"]


def recalibrated_score(actions, ctx, update):                    # your recalibrated simulator: impact 40% higher
    p = dict(ctx["p"], eta=ctx["p"]["eta"] * 1.4)
    return t.score(actions, dict(ctx, p=p, cid=ctx["cid"] + "-recal"))


client = Client(journal_path="runs/finm-pkg-ac-demo/journal.jsonl", max_calls=1200)
PL.refine(client, t, dom, pol.base, C["train"], C["support"], updates=done + 10, start=done,
          seed=pol.meta["ref_seed"], create=False, score_fn=recalibrated_score)
new_pol, check = PL.export_policy(client, t, dom, pol.base, C["support"], pol.meta["arm"], C["train"],
                                  C["eval"][:4], meta=dict(pol.meta, updates=done + 10))
new_pol.save("runs/finm-pkg-ac-demo/policy_after_shift.json")
print("export exact:", check["exact"], "| calls:", client.calls)
client.delete_domain(dom)                                        # or keep it for the next shift
```

<!-- check: live -->
```bash
python rerefine.py
```

## 8. Troubleshooting

| Symptom | Cause | What to do |
|---|---|---|
| `no credits (HTTP 402)`, or `CreditsOut` | The key has no query credits, or its plan is inactive | Top up, then run `probe` again. A run that stops on 402 still tries to delete its Domains. If that fails too, delete them later with `domains --prefix finm-pkg --delete` |
| Slow progress, `http_429` in results | Rate limit, about 20 requests a second per key | Nothing to do: the client waits and retries automatically, up to 60 times per request. If it is persistent, use `--threads 1` |
| HTTP 409 "trajectory learner limit reached" | 32 configured learners already exist for this owner | The client waits up to 20 minutes for a slot. Find leftovers with `domains`, delete them with `domains --delete`, and delete learners kept with `--keep` once you are done |
| HTTP 422 | The request was understood but is invalid | Read the server's message in the `resp` field of the last line of `runs/<prefix>/journal.jsonl`. Usual causes: refinement `horizon` not equal to the number of base rows; context vectors not the same length as the support set; fewer than 8 candidates; planning without a value model; a structural `command_budget` shorter than a record |
| Domains left over | A run was killed before its cleanup | `domains --prefix finm-pkg` lists them and `--delete` removes them. `runs/<prefix>/ledger.jsonl` shows what each run created and deleted |
| `AmbiguousWrite` | A network error or HTTP 5xx during a write (propose, observe, configure): it may or may not have been applied | The run stops and cleans up. If you drive the API yourself, do not blindly resend. Read `trajectory/state`, cancel a pending decision with `client.traj(domain, "cancel", {"decision_id": ...})`, then propose again |
| `BudgetExceeded` | `--max-calls` was reached | Raise the cap using the budget table in section 4; the Domains are still deleted |
| A policy with `"fake_api": true` | It came from a dry run | Rerun without `--dry-run` and without `MACHINA_FINANCE_FAKE=1` |
