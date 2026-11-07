# Pre-registration: Adapt-1 on established benchmarks, round 2 (Industrial Benchmark, SMPL process plants)

Written and committed before any counted run. Service: Fast-UAT `https://rei-neuroadapt-fast-uat.reilabs.org/api/v1`,
build `af6670b`, schema 5 (checked at start). Smoke tests (Industrial Benchmark seed 99, 3 x 20 steps) are excluded.
Local comparators run at their own published-style budgets; comparator step counts are not a constraint here (user).

## Selection rule
A benchmark qualifies if (a) a light control (constant setting, do-nothing, default recipe) does not already saturate it,
(b) the published agents for it need far more data than Adapt-1 will get, and (c) its shape matches a documented Adapt-1
route. bsuite's basic tasks failed (a) and (b); DMC for Machina failed (c).

## A. Siemens Industrial Benchmark, sequential route (`ib_bench/`)

**Question.** Does Adapt-1's documented online route (contextual policy learning + sequential learning + temporal
context + latent belief, configured from the docs' recommended starting declaration) learn the Industrial Benchmark's
steering problem as data-efficiently as the published RL algorithms, and does it cope with a drifting setpoint?

**Environment.** `industrial_benchmark_python` 2.0 (Hein et al. 2017), Gym wrapper `IBGym`: observation = the 6 visible
variables (setpoint, velocity, gain, shift, fatigue, consumption), reward = -(3 fatigue + consumption) ("classic"),
actions = the benchmark's own 27 discrete steerings {-1, 0, +1}^3. Episodes: 200 steps from the benchmark's standard start
(v = g = h = 50), fresh noise seed per episode (the wrapper's own reset rule). Two NumPy-alias shims (`np.float`,
`np.bool8`) and a pass-through of `stationary_p` to `IDS` are the only changes (`ib_env.py`).
- Arm A1: stationary setpoint p = 50 (the benchmark's standard setting).
- Arm A2: drifting setpoint (`IDS(stationary_p=False)`), starting at 50: the regime-change test.
Local measurements before the run (p = 50, 1,000 steps): do-nothing policy mean reward -170; uniform random -258.

**Adapt-1** (`run_adapt1.py`): context = the 6 variables with fixed scaling (/100 for the first five, /500 for
consumption); TCP over the 6 inputs, `maximum_lag` 10; policies = the 27 steerings with the three deltas as
`policy_features`, `action_transfer_strength` 0.05 (docs default; physical meaning); reward component bounds [-800, -100];
`credit_assignment` eligibility trace, discount 0.9; `sequential` enabled with discount 0.9, `n_step` 10 (the benchmark's
documented 10-step consumption delay), `minimum_episodes` 10, `minimum_validation_skill` 0.1; `latent_belief` enabled;
everything else as in the docs' recommended declaration. Online loop: one query and one decision-linked feedback per step,
learning and exploration on throughout. Abstention or an illegal selection executes the no-op steering [0,0,0] and is
counted as a fallback. Seeds 0, 1, 2 per arm; 150 episodes = 30,000 steps per run (6 runs, 180,000 steps).

**Comparators** (local, `run_baselines.py`, same episode protocol and reward, online training curves logged per episode):
- PPO (SB3 defaults, VecNormalize), 1,000,000 steps, seeds 0-2, per arm.
- DQN on the 27 steerings (SB3 defaults, VecNormalize), 1,000,000 steps, seeds 0-2, per arm.
- SAC (SB3 defaults, VecNormalize), 500,000 steps, seeds 0-1, per arm (CPU-bound at ~100 steps/s; this trim is recorded).
- Do-nothing and uniform random, 30,000 steps, seed 0, per arm.

**Metrics.** Primary: Adapt-1's mean step reward over its last 50 episodes (steps 20,000-30,000), per seed. For each
comparator: (i) its mean step reward over the same step window of its own training (matched data); (ii) its mean over the
last 50 episodes of its full budget (asymptote); (iii) the step at which its running 50-episode mean first reaches Adapt-1's
primary value (data multiplier). Secondary: whole-run online mean (steps 0-30,000), fallback rate, no-op fraction,
sequential-model installation and validation skill from the diagnostics, and for A2 the mean reward split by setpoint
tercile. Learning curves: per-episode means.

**Analysis.** Report every seed's value, means and seed ranges. A difference is called only when Adapt-1's seed range does
not overlap the comparator's seed range (3 vs 3 seeds; SAC 2). Results are descriptive at this seed count.

## B. SMPL BeerFMT (NeurIPS 2022 Datasets & Benchmarks), Machina (`smpl_bench/beer_*`)

**Plant.** SMPL `BeerFMTEnvGym`: hourly steps, at most 200; one control (temperature, 9-16 C); SMPL's dense reward (-1 per
hour, +200 when fermentation is declared finished, -200 on leaving the observation box; an unfinished batch ends after
200 h). A stand-alone copy (`beer_plant.py`) matched the installed SMPL environment on 120 random batches (identical
returns and lengths; `check_against_smpl` in this file's history). Quality for Machina = (return + 400) / 600.
Batch context (SMPL's own reset variation): initial lag/dead yeast in 1.8-2.2 g/L and sugar in 117-143 g/L, measured at reset.

**Gates, run locally before this file was committed** (`beer_gate.py`, `gates/beer_gate.json`): best constant temperature
14.5 C finishes at 133 h (return 68); a CMA-ES 20-stage profile finishes at 81 h (return 120), so the light control does
not saturate the task. On 8 sampled contexts the nominal profile fails outright on 4 (return -399 or unfinished) while the
per-context optima are 114-121, so the best profile depends on the batch: refinement is meaningful. Both gates pass.

**B1 Acquisition.** Executor BEER-1: 20 stages of 10 h, u in [-1, 1] mapped linearly to [9, 16] C, outcome = quality,
state rows = SMPL observation / box maximum at stage boundaries for the executed prefix. Nominal initial conditions.
Machina `episode_credit`, arms `default` and `profile` (docs optional profile without progress-guided mutation), proposal
streams 1 and 2, 600 attempts each (4 runs).
Comparators: random search, (1+1)-ES, CMA-ES over the same u in [-1, 1]^20 at 600 attempts, 10 seeds each
(`beer_local_acquire.py`); best constant temperature (68); the 8,000-evaluation CMA-ES reference (120).
Published-RL comparators: SB3 PPO (1,000,000 steps, seeds 0-2) and SAC (500,000 steps, seeds 0-1) on SMPL's own protocol
(closed loop, hourly temperature action, initial conditions redrawn every episode) (`beer_sb3.py`).
Metrics: best quality after 16, 32, 64, 128, 256, 600 attempts (Machina and local methods; mean and range over runs);
PPO/SAC online return per episode (one episode = one batch) and the episode at which the running 50-episode mean first
reaches each Machina run's best return; held-out scores (below).

**B2 Refinement.** Base = the best acquired profile over the four Machina acquisition runs. Contexts: TRAIN 64 and HELD 32,
fixed draws (`beer_contexts.py`). Arms (`contextual_execution`): linear basis radius 0.2, linear radius 0.08 (docs
default), shared correction without context (control). 30 updates x 8 candidates x 32 training contexts = 7,680
executions per arm; frozen `evaluate` on HELD after 0, 5, 10, 20, 30 updates.
Comparators at the same 7,680 executions on TRAIN (`beer_local_refine.py`): CMA-ES-trained linear context-to-offset policy
(3 seeds), NES-trained policy (3 seeds), context-free robust profile (CMA-ES on the mean over contexts, 3 seeds); base
alone; per-context CMA-ES ceiling (4,000 evaluations per held-out batch); best constant temperature per batch; the trained
PPO/SAC policies rolled out on HELD (they observe the state, so they adapt implicitly).
Metrics: held-out mean quality, mean return, fraction of batches finished, gap recovered = (arm - base) / (ceiling - base).
A difference between two methods is called when the paired sign over the 32 held-out batches gives p < 0.05 (two-sided).

## C. SMPL PenSimEnv (IndPenSim via PenSimPy), Machina (`smpl_bench/pensim_*`), conditional

**Plant.** SMPL `PenSimEnvGym` with `fast=False` (PenSimPy's own pure-Python ODE path) and an import shim for the compiled
integrator that is not on PyPI; batch = 1,150 steps of 12 min, 6 controls (discharge, sugar feed, soybean oil, aeration,
back pressure, water), PAA from the default recipe as SMPL does; reward = penicillin yield per step, batch yield = sum.
Quality = yield / 4,500. Batch context measured at reset = PenSimPy's initial-state draw (S, DO2, X, V, Wt, pH, T, PAA);
hidden variation (growth parameters, kLa, PAA concentration, disturbance profiles) stays hidden. About 14 s per batch.
**Gate C0.** The default recipe must complete the 1,150 steps without a solver error on seeds 1-3 (`gates/pensim_check.log`,
recorded); otherwise C is dropped and reported as such.
**C1 Acquisition.** Executor PEN-1: 23 stages of 10 h, u in [-1, 1]^(23 x 6) mapped onto SMPL's action bounds.
Machina `default` and `profile`, one stream each, 300 attempts, nominal batch (seed_ref 1000).
Comparators: random, (1+1)-ES, CMA-ES at 300 attempts, 2 seeds each; the default recipe (light control; seeds 1-5); the
published GP-EI dataset shipped with smpl-experiments (1,000 batches: mean 3,806, p90 4,066, max 4,080 kg) as the
Bayesian-optimization-at-1,000-batches reference; SB3 PPO 500,000 steps (about 435 batches), seed 0.
Metrics: best quality after 16, 32, 64, 128, 300 attempts; yield of the best recipe re-run on 5 fresh seeds.
**C2 Refinement gate.** Run only if, among the top-8 acquired recipes, the best recipe differs across at least 25 % of 16
training contexts. If it passes: 20 updates x 8 candidates x 12 contexts (1,920 executions), HELD 16 contexts at updates
0, 5, 10, 20; comparators at the same executions: CMA-ES linear policy (1 seed), context-free robust recipe (1 seed), base
alone, default recipe. CPU limits may trim seeds, never below what is written here without a note in the report.

## Validity rules (all parts)
- A run counts only if it completes; an aborted or crashed run is reported, not dropped, and rerun once from scratch with
  the same seed and a new Domain; both are reported. Adapt-1 runs abort if more than 1 % of feedbacks are not applied.
- Every Domain is deleted at the end of its run and the deletion is confirmed by a 404 readback.
- Raw per-episode / per-attempt files, declarations, server echoes and diagnostics are committed; derived metrics are
  computed from them by `analyze.py` and kept separate.
- Order of work if time runs out: A and B first, C after; anything not run is listed as not run.
