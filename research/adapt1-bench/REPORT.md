# Adapt-1 on established benchmarks, round 2: Industrial Benchmark and SMPL process plants (2026-11-07)

Service: Fast-UAT `https://rei-neuroadapt-fast-uat.reilabs.org/api/v1`, build `af6670b`, schema 5. Every Domain created for
these runs was deleted at the end of its run and confirmed by a 404 readback. Pre-registration: `PREREG.md` (committed
before the first counted run; its dated amendments are inside it). Tables: `results/ROUND2_RESULTS.md`, regenerated from
the raw files by `analyze.py`. Comparators ran at their own published-style budgets (the user removed the budget constraint).

*Status: section B is final except the SAC comparator; sections A and C are in progress and are filled in as their runs
complete. Anything listed as "in progress" below has not been read into a conclusion.*

## Bottom line (so far)

| Question | Answer |
|---|---|
| Machina acquisition on SMPL BeerFMT (nominal batch) | Level with CMA-ES over the same 20-stage profile: best return 100-116 in 600 batches vs CMA-ES 115 (mean of 10 seeds), (1+1)-ES 110, random 89. Closed-loop PPO reaches the 120 ceiling but needs 2,400-3,500 batches to pass Machina's 116. |
| Machina contextual refinement on SMPL BeerFMT (unseen batches) | **Below every same-data learner.** Held-out mean return: contextual linear 63.5 (radius 0.2) and 34.9 (0.08); the context-free correction 74.8; NES policies 85-95; CMA-ES policies 46-105; context-free CMA-ES profile 66-86; base alone -82.5; per-batch ceiling 116. Paired over 32 batches, Machina's contextual arm loses to each learned comparator on 24-30 of them. PPO at 1M steps is at the ceiling on all 32. |
| Adapt-1 sequential route on the Industrial Benchmark | in progress (6 runs at episode 4 of 150; about 2 s of server time per call) |
| Machina acquisition and refinement on SMPL PenSim | in progress (acquisition at attempts 150 and 100 of 300; best so far 2,990 kg vs the default recipe's 3,247 kg mean) |

## B. SMPL BeerFMT (NeurIPS 2022 Datasets & Benchmarks), Machina

Plant: SMPL's `BeerFMTEnv`, verified copy (identical returns and lengths on 120 random batches against the installed SMPL
environment). Temperature schedule, hourly steps, at most 200 h; SMPL's dense reward: -1 per hour, +200 when fermentation is
declared finished, -200 on leaving the observation box; an unfinished batch ends at 200 h with return -200. Quality for
Machina = (return + 400) / 600.

Gates (local, before the runs): best constant temperature 14.5 C finishes at 133 h (return 68); a CMA-ES 20-stage profile
finishes at 81 h (return 120); on 8 sampled batches the nominal profile fails outright on 4 while per-batch optima are
114-121. Both gates passed, so the task is neither saturated by a constant setting nor context-free.

### B1 Acquisition (nominal initial conditions, 600 batches per run)

| method | best quality after 16 / 64 / 256 / 600 attempts | best return |
|---|---|---:|
| Machina default, stream 1 | 0.677 / 0.767 / 0.810 / 0.833 | 100 |
| Machina default, stream 2 | 0.752 / 0.807 / 0.845 / 0.860 | 116 |
| Machina profile, stream 1 | 0.783 / 0.822 / 0.860 / 0.860 | 116 (reached at attempt 176) |
| Machina profile, stream 2 | 0.790 / 0.828 / 0.853 / 0.860 | 116 |
| CMA-ES, mean of 10 seeds | 0.774 / 0.815 / 0.852 / 0.858 [0.85-0.87] | 115 |
| (1+1)-ES, mean of 10 seeds | 0.816 / 0.828 / 0.844 / 0.851 [0.82-0.86] | 110 |
| random search, mean of 10 seeds | 0.791 / 0.799 / 0.808 / 0.815 [0.81-0.83] | 89 |
| best constant temperature | 0.780 | 68 |
| CMA-ES, 8,000 evaluations (reference) | | 120 |

Closed-loop PPO (SB3 defaults, SMPL protocol with initial conditions redrawn every batch), 3 seeds x 1,000,000 steps:
running 50-batch mean reaches 116 after 2,398 / 2,961 / 3,505 batches and ends at 120; held-out mean 120.0 / 120.1 / 120.1.
SAC (500,000 steps, 2 seeds): in progress.

Reading: as on the published control problems in the earlier round, Machina's acquisition is "as good as a good evolution
strategy" at the same number of batches and is beaten on data efficiency by nothing simple; PPO needs roughly 5-20x the
batches to catch up but then exceeds every open-loop profile because it reacts to the state.

### B2 Refinement (base = best acquired profile, SMPL's own batch-to-batch variation as context)

Base: Machina default stream 2's best profile (return 116 at the nominal batch). On the 32 held-out batches the base alone
finishes 38 % of them (mean return -82.5): the acquired profile is brittle. Per-batch ceiling (CMA-ES, 4,000 evaluations per
batch): 116. Best constant temperature per batch: 63.8. Refinement budget: 30 updates x 8 candidates x 32 training batches =
7,680 executions per arm; comparators use the same 7,680 executions on the same training batches.

| arm / learner | held-out mean return | finished | gap recovered | paired vs base (better/worse, p) |
|---|---:|---:|---:|---|
| Machina linear, radius 0.2 | 63.5 | 0.84 | 0.74 | 17/10, p=0.25 |
| Machina linear, radius 0.08 (docs default) | 34.9 | 0.75 | 0.59 | 12/3, p=0.035 |
| Machina shared correction (no context) | 74.8 | 0.88 | 0.79 | 18/4, p=0.004 |
| NES linear context policy, 3 seeds | 95.0 / 85.1 / 85.5 | 0.91-0.94 | 0.84-0.89 | all p <= 0.004 |
| CMA-ES linear context policy, 3 seeds | 105.0 / 46.1 / 66.1 | 0.78-0.97 | 0.65-0.94 | all p <= 0.004 |
| CMA-ES context-free profile, 3 seeds | 85.1 / 66.2 / 85.9 | 0.84-0.91 | 0.75-0.85 | all p <= 0.007 |
| PPO, closed loop, 1M steps, 3 seeds | 120.0 / 120.1 / 120.1 | 1.00 | 1.02 | 32/0 |

Paired over the 32 held-out batches, Machina linear 0.2 is worse than every learned comparator on 24-30 batches (all
p < 0.001) and worse than Machina's own context-free arm on 28 of 29; it beats the docs-default radius arm on 22 of 27.
The contextual arms plateaued after update 10 (held-out 63.3 at updates 10, 20 and 30).

Reading: on this plant the correction that helps is a robust shift of the whole profile, not a context-dependent one, and
Machina's refinement finds less of it than a plain CMA-ES or NES given the same executions. This is the opposite of the
fed-batch result in the earlier round, where the contextual stage beat the same learners by 2-8x in data. The difference is
in the plant: beer's failure mode is a hard cliff (unfinished batch = -200), and the acquired base sits on that cliff for
most batches, so the useful correction is the same for all of them.

## A. Siemens Industrial Benchmark, sequential route

In progress. Setup per `PREREG.md` section A: 6 runs (stationary and drifting setpoint, 3 seeds each), 150 episodes of 200
steps, one query and one decision-linked feedback per step. Comparators: PPO, DQN (the benchmark's 27 steerings), SAC,
do-nothing, random, at their full budgets. Measured server time per call with 8 clients on the service: about 2 s.

## C. SMPL PenSimEnv (IndPenSim), Machina

In progress. Setup per `PREREG.md` section C with its three dated amendments (fresh simulator object per batch; quality =
SMPL return shifted into [0, 1]; executor PEN-2 = adjustments around the default recipe; comparators in two start variants).
Default recipe on seeds 1-5: 3,434 / 2,973 / 3,007 / 3,476 / 3,343 kg (mean 3,247); on the nominal batch 3,481 kg. Published
GP-EI dataset shipped with smpl-experiments: 1,000 batches, mean 3,806, p90 4,066, max 4,080 kg.

## Run notes
- A job-queue helper named `queue.py` shadowed the standard library module and crashed the first launch of every Adapt-1
  run at import; the runs were relaunched after the rename (nothing had reached the server beyond Domain creation, and
  those Domains were deleted).
- PenSim: SMPL's environment object does not reset its recipe pointer, so a second batch on the same object fails at its
  first step; a fresh object is built per batch (amendment 1). Under the absolute-control executor every method's proposals
  crashed the plant; the executor was changed to adjustments around the default recipe (amendment 2), and the black-box
  comparators run both from the default recipe and from a random start (amendment 3).
- The beer per-batch ceiling is a 4,000-evaluation CMA-ES estimate; the nominal 8,000-evaluation reference (120) and PPO
  (120) show it is 3-4 return points low, which is why PPO's gap recovered reads 1.02.
