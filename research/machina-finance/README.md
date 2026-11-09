# Machina finance: the shortlist worth more work

This package keeps only the finance applications where Adapt-1 Machina looked worth more work in our prod
experiments (three rounds, more than 110k API calls). It contains the pipeline that worked, 13 frozen policies
exported from the prod runs, offline tests and the evidence. The applications that did not hold up are listed under
[Dropped and why](#dropped-and-why), with no code, so nobody retries them by accident.

Everything below is **simulation with known cost models**. It shows the method can work. It does not show an
edge on live markets yet (see [Limits](#limits)).

## Shortlist

The share of best is the mean score divided by the mean best-known score on 32 held-out market conditions.
0% is the naive benchmark (TWAP, constant quotes, daily re-hedging, a fixed 60/40 mix) and 100% is the best
known schedule for each condition.

| Application | Machina, refined | Before refinement | Same-budget CEM | Textbook rule | 512 fresh / out of range |
|---|---|---|---|---|---|
| Liquidation, Almgren–Chriss impact | **83%** (linear) | 52% | 65% | | 83% / 78% |
| Liquidation, transient impact (Obizhaeva–Wang) | **87%** (guided; linear 77%) | −33% | 63% | | 87% / 6% |
| VWAP execution, day-type volume profile | **78%** (linear) | −12% | −31% | | 76% / 80% |
| Two-leg portfolio switch | **95%** (guided, 40 updates; 81% at 20; linear 52%) | 40% | 18% | | 91% / 90% |
| Three-leg portfolio switch | **89%** (guided, 40 updates; 75% at 20; linear 51%) | 31% | 33% | | 89% / 98% |
| Market-making quotes (spread, skew) | **98%** (linear) | 65% | 95% | Avellaneda–Stoikov 74% | |
| Option delta-hedging bands | **77%** (linear) | 64% | 73% | Whalley–Wilmott 58% | |
| Lifecycle glide path | **61%** (linear) | 37% | 51% | | |
| Liquidation with a pre-trade alpha signal | **71%** (guided, 40 updates) | 0% (TWAP) | 32% (40 updates) | signal-blind AC −24% | 72% / 77% |

- **Linear** is the default refinement. **Guided** adds `normalize_context`, a quadratic context basis and
  `response_guidance`; plain quadratic without guidance was worse everywhere and is dropped.
- **CEM** is cross-entropy search over linear offset maps with the same updates and contexts (20 updates unless
  noted). It is untuned.
- **512 fresh** re-scores the policy on new conditions drawn from the training range. **Out of range** uses 256
  conditions up to 60% beyond it (|u| ≤ 1.6), with the service's ±3 SD context clipping. Both are only computed
  where the best-known schedule is exact and cheap.
- Linear ceilings (oracle fit of a linear map to the best schedules): 94% AC, 92% OW, 99% VWAP, 84% two-leg,
  82% three-leg, 93% glide path. Guided refinement went past the linear ceiling on both portfolio switches.

**Why these are worth more work:**
- **Execution schedules** (the first five rows) have the strongest and broadest evidence. Machina beat its own
  starting point and black-box search on every task, the results held on 512 fresh conditions, and the robustness
  checks (seeds, noise, shifts) were run on these tasks.
- **Market making and hedging bands** clearly beat the textbook rules (+24 and +19 points), but beat same-budget
  CEM by only 3–4 points.
- **Glide path** gained 24 points over its starting point.
- **Alpha-signal liquidation** is the one case where the schedule's shape must flip with the context, and guided
  refinement handled it. It rests on one learner seed.

Full numbers are in `results/shortlist.csv`.

## The pipeline (what to run, what it costs)

1. **Acquire at one nominal condition.** Use an `episode_credit` learner with the documented optional profile.
   About 300 episodes (≈610 calls) reached 96–100% of best at the nominal condition for AC and OW
   liquidation, the two-leg switch, hedging bands and the glide path. The three-leg switch reached 62% at 300.
   Market making needs about 1,000 episodes (48% at 300, 99% at 1,000), and VWAP needed 1,000 to reach the
   best. Machina matched or beat CMA-ES and Powell at 300 evaluations.
2. **Refine across conditions.** Use `contextual_execution` around the acquired schedule, with 8 candidates and
   16 conditions per update.
   - Linear arm: 20 updates. With MIE (below) this costs 42–50 calls per update, so about 1,000 calls in total.
   - Guided arm: 40 updates at 130 calls per update, about 5,200 calls.
   - All outcomes are simulated locally. The API never sees your cost model or the base schedule.
3. **Export a frozen policy.** The retained map is an exact affine function (linear arm) or quadratic function
   (guided arm) of the clipped context features. A handful of retained `evaluate` calls identify it, and a few
   more verify it. The result is a small JSON policy that runs offline with no API calls, can be reviewed and is
   deterministic. In every run, including the live smoke test on all 32 held-out conditions, the exported map
   reproduced the API to within 1e-14.
4. **Use and monitor.** Compute schedules offline. Compare realized against expected cost (`cusum_alarm`). After
   a shift, continue refining the **same** learner (`refine(..., create=False)`).

**MIE** (map-identification evaluation, linear arm) evaluates each candidate at 1 + d well-spread conditions and
solves for its affine map. It predicts the other conditions and spot-checks the most outlying one against the
API. Any mismatch above 1e-9 falls back to full evaluation for that candidate. Outcomes are identical to full
evaluation, using about 3× fewer calls. On prod, 0 of 96 candidates needed the fallback.

**Rules that mattered** (each from a prod experiment that a separate reviewer re-checked; details in
`results/evidence.json`):
- **Use at least 8 candidates per update.** With 3, two-leg refinement stalled near 47%; with 8 it reached 74% at
  the same number of trades. On AC, halving the candidates cost about 9 points, and 60 updates reached 93.0%
  against a 93.6% ceiling.
- **With a noisy simulator, compare candidates on shared scenarios** (common random numbers, `pipeline.crn`).
  Sharing 100, 10 or 1 simulated days kept 91%, 73% and 37% of the noise-free gain. Independent noise per
  candidate kept about 0%.
- **The starting schedule matters, but refinement closes about 69% of the gap.** Two-leg starts at 0%, 25% and
  38% ended at 42%, 47% and 54% (3 seeds each). Seed-to-seed spread was about 3.5 points, and refinement beat
  contextual CEM in 9 of 9 runs.
- **After an unseen change in market impact, continue refining the same learner.** Continuing reached 67–69%
  after 20 updates against 45–61% for a fresh learner. Most of that is head start, and the difference is not
  significant across seeds (p = 0.125).

## Limits

- **All shortlisted evidence is simulated.** On real S&P 500 daily data, no schedule beat TWAP with confidence,
  and real-data volatility targeting lost to plain CEM. Real use needs an impact and fill simulator calibrated to
  your own executions (intraday), then the same pipeline.
- **The baselines are weak.** CEM and Powell are untuned, and no tuned contextual baseline has been run. Powell
  matched Machina's final level on larger baskets at 1,000 evaluations.
- **Some results rest on one learner seed, and some numbers are optimistic.**
  - The guided results (OW 87%, two-leg 95%, three-leg 89%, alpha 71%) each rest on one learner seed.
  - The recommended arm per task was picked on the 32 held-out conditions. Where cheap, the 512 fresh
    conditions confirmed the pick.
  - For market making, hedging and glide path, the best-known schedule comes from a search, not an exact
    solution, so their shares may be slightly optimistic.
- **Contexts are clipped at ±3 SD of the support set.** Beyond that the policy stops changing, and outside the
  trained range OW fell to 6%. Make the support set cover the conditions you will trade in.
- **An unseen regime change makes a frozen schedule stale.** Monitoring and re-refinement are required, not
  optional.

## Dropped and why

| Dropped | Evidence |
|---|---|
| Learning from scratch across many conditions (multi-context acquisition) | Worse than naive on every task. Frozen recall returns the single best record whatever the state, goal or condition (unkeyed in 4 of 4 probes). |
| Single-learner closed loop / short decision windows | Day score −3.62, for the same recall reason. Refining the afternoon given the signal worked (+0.035) but only tied CEM, so it is not packaged. |
| Learning from fresh noisy outcomes | Single-day fills: 36% of best, against 99% noise-free. 5-day averages: −124%. Fresh noise in refinement: about 0%. Use shared scenarios instead. |
| `step_outcomes` with `progress_guided_mutation` | Worse in 5 of 6 pairs (−40% against 62% of best on average), not significant (p = 0.16). |
| Quadratic refinement without guidance | Below linear on every simulated task, e.g. AC 60% vs 83%, OW 31% vs 77%, VWAP 22% vs 78%. |
| Structural stages (selection, deletion, ordering) on good schedules | No change on good nominal schedules; they only helped weak ones. |
| Regime flags in state/goal; `observation_validity` | A flag did not stop a stale schedule after a liquidity drop (−171% of the new best for 300 runs). Flagging 10 inflated outcome reports changed nothing (−330%). |
| Real S&P daily accumulation vs TWAP | Refined +4.9 bps against base +5.6 bps; confidence intervals include 0; gains were mostly bull-market drift. |
| Real-data volatility targeting | Linear map 0.43 of perfect tracking: not clearly ahead of the textbook rule (0.32, CI of the gap −0.05 to 0.31) and behind same-budget CEM (0.49–0.51). |
| 3 candidates per update | Stalled near 47% of best. |
| Basket transitions (4–6 assets, 32–48 controls) | Acquisition only: 98% and 95% at 1,000 runs, but Powell caught up by 1,000. Refinement untested. |

## Quickstart

**Step-by-step guide: [HOWTO.md](HOWTO.md)** covers install, using a saved policy, re-checking the results,
training your own on the API (with call budgets), bringing your own simulator, deployment, drift monitoring
and troubleshooting. Every command in it is checked by `scripts/check_howto.py`.

Offline, no key needed:

```bash
pip install -e '.[test]'
```

```bash
python -m machina_finance list
```

```bash
python -m machina_finance schedule --policy policies/ac_liquidation--linear.json --u 0.5,-0.2,0.8
```

```bash
python -m machina_finance evaluate --policy policies/two_asset_transition--guided.json --fresh 128
```

```bash
python -m pytest -q
```

Live (uses API credits; every Domain it creates is deleted at the end). The documented auth is
`Authorization: Bearer <key>` from `ADAPT1_API_KEY`. Without a key no auth header is sent, for setups where a
proxy injects it.

```bash
python -m machina_finance probe
```

```bash
python -m machina_finance run --task ac_liquidation --episodes 300 --updates 20 --max-calls 2500
```

To use your own simulator, subclass `tasks.Task`. Provide `params_from_u`, `features`, `decode`, `objective`
and, if the naive benchmark is not all zeros, `naive_actions`; then register it in `tasks.TASKS`. The pipeline
only needs `task.score(actions, context)`.

**Live smoke test** (`scripts/smoke_test.py`, AC liquidation, deliberately small: 100 acquisition episodes and
12 linear updates):
- Acquisition reached 71% of best at the nominal condition.
- On held-out conditions, refinement lifted the schedule from 36.5% to 58.1% of best and beat the base in 84%
  of conditions.
- The exported policy matched the API on all 32 held-out conditions (max difference 1e-16).
- 761 API calls, 0 retries, no MIE fallbacks, and 0 Domains left. Details in `results/smoke_test.json`.

The full budget (300 episodes, 20 updates) reached 83% in the main run.

## Next experiments worth running

1. **Linear refinement plus response guidance as the default.** It beat plain linear on vol targeting (0.48 vs
   0.43, found after the fact) but is untested on the simulated tasks. About 3.5k calls with MIE.
2. **More seeds for the guided results** (two-leg 95%, three-leg 89%, OW 87%, alpha 71%). About 8 learners each
   for ±3 points.
3. **A tuned same-budget contextual baseline.** Local only, no API calls.
4. **Recall under `outcome_mode: goal_error`.** About 200 calls. If recall keys on the goal, learning across
   conditions might work after all.
5. **One learner per regime with a router in front.** About 2–3k calls.
6. **Calibrate to real executions** (intraday fills and a fitted impact model), then rerun rows 1–5 of the shortlist.

## Files

- `machina_finance/`
  - `tasks.py`: simulators and context features, identical to the prod runs.
  - `references.py`: best-known schedules and textbook rules.
  - `basis.py`: the service-equivalent context transform.
  - `policy.py`: frozen policies.
  - `client.py`: API client with retries, 402 handling, call cap and Domain ledger.
  - `pipeline.py`: acquisition, refinement with MIE, export, CRN and drift helpers.
  - `baselines.py`: same-budget CEM, Powell and oracle fit.
  - `cli.py`: the command line (`list`, `describe`, `schedule`, `evaluate`, `verify`, `probe`, `run`,
    `domains`).
  - `fake.py`: the in-memory fake API behind `--dry-run` and the tests.
- `policies/`: 13 frozen policies exported from the prod runs, as `<task>--<arm>.json`. Each reproduces its run's
  held-out scores to below 1e-9; the tests check this.
- `results/`
  - `shortlist.{json,csv}`, `policies_heldout.json`, `best_known_eval.json`: the shortlist numbers.
  - `evidence.json`: the re-checked experiment rows behind the pipeline rules.
  - `smoke_test.json`, `smoke_policy_ac_liquidation.json`: the live smoke test.
- `scripts/`
  - `export_from_runs.py`, `build_results.py`: provenance; they rebuild `policies/` and `results/` from the
    original run folder.
  - `smoke_test.py`: the live smoke test.
  - `check_howto.py`: runs every command in HOWTO.md in a fresh copy, with live steps on the fake API.
- `tests/`: offline tests, including a fake API for pipeline plumbing.
