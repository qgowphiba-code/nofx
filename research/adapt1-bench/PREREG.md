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
*Result (2026-11-07 14:50 UTC, before any counted C run): passed once a fresh SMPL environment object is built per batch
(the recipe's setpoint pointer is not reset by SMPL's reset(); on a reused object batches alternate ok/failed). Default
recipe yields on seeds 1-3: 1,348 / 2,992 / 2,112 kg.*
*Amendment (15:05 UTC, both Machina C1 runs restarted from attempt 0 after 1 and 3 attempts): quality is SMPL's own batch
return shifted into [0, 1], (return + 100) / 4,600, instead of yield / 4,500. A batch that fails at its first step still scores
0 and a completed batch scores within 0.3 % of yield / 4,500, but a batch that fails late keeps credit for the penicillin it
made (SMPL's -100 failure penalty is inside the return). Arbitrary recipes fail early (observed), so a flat 0 for every
failure would leave every acquisition method with no signal; the same formula applies to Machina and to all comparators.*
*Amendment 2 (15:25 UTC, both C1 runs restarted again after 27 and 26 attempts, every one a failed batch): executor PEN-2
replaces PEN-1. u = 0 is now the default recipe's stage mean (the recipe every method is handed, and the light control);
u > 0 moves linearly towards SMPL's upper action bound and u < 0 towards the lower bound. Under PEN-1 (u = 0 = mid-range of
the absolute bounds) essentially every proposal from any method crashed the plant within ~100 steps, so the task was "find
the feasible region", which is not what the published GP-EI search did either (its batches keep the default recipe's
structure and vary the setpoints within SMPL's bounds). All comparators start from u = 0 as before.*
*Amendment 3 (15:30 UTC, C1 Machina runs at attempts 80 and 40, no comparator result yet): under PEN-2, u = 0 hands the
black-box comparators the default recipe for free while Machina's first proposals carry no recipe knowledge. The
comparators therefore run in two start variants, both reported: from the default recipe (u = 0, seed < 100) and from a
uniform random u (seed >= 100). Seeds are cut to one per method and variant (CPU: one PenSim batch costs about 12 s).*
**C1 Acquisition.** Executor PEN-1: 23 stages of 10 h, u in [-1, 1]^(23 x 6) mapped onto SMPL's action bounds.
Machina `default` and `profile`, one stream each, 300 attempts, nominal batch (seed_ref 1000).
Comparators: random, (1+1)-ES, CMA-ES at 300 attempts, 2 seeds each; the default recipe (light control; seeds 1-5); the
published GP-EI dataset shipped with smpl-experiments (1,000 batches: mean 3,806, p90 4,066, max 4,080 kg) as the
Bayesian-optimization-at-1,000-batches reference; SB3 PPO 500,000 steps (about 435 batches), seed 0.
Metrics: best quality after 16, 32, 64, 128, 300 attempts; yield of the best recipe re-run on 5 fresh seeds.
**C2 Refinement gate.** Run only if, among the top-8 acquired recipes, the best recipe differs across at least 25 % of 16
training contexts. *Result (18:00 UTC): passes. On the 16 gate batches the top acquired recipe is never the best (winners
are recipes 3, 4 and 6 of the top-8), and its own yield ranges from 940 to 4,061 kg across batches. Refinement launched
with the base = the profile arm's best recipe (4,271 kg on the nominal batch; the default arm never completed a batch).* If it passes: 20 updates x 8 candidates x 12 contexts (1,920 executions), HELD 16 contexts at updates
0, 5, 10, 20; comparators at the same executions: CMA-ES linear policy (1 seed), context-free robust recipe (1 seed), base
alone, default recipe. CPU limits may trim seeds, never below what is written here without a note in the report.

## D. ChemGymRL reaction benches (Beeler et al., Digital Discovery 2024), Machina, production API (`chemgym_bench/`)

Added 2026-11-07 ~17:45 UTC, before any counted run. Service: production `f5e1ab8`. Benchmark code from the ChemGymRL-v1 clone.

**Question.** On a chemistry benchmark where the published RL agents mostly fail to beat hand-written heuristics, how fast does
Machina acquire a reaction recipe per target compared with same-budget black-box search, the paper's own heuristic, and the
paper's RL algorithms trained closed-loop at their full budgets?

**Plants.** GenWurtzReact-v1 (7 targets, 6 controls) and FictReact-v2 (4 targets, 7 controls; E is penalised for the other
targets), 20 steps, terminal reward (moles of target minus moles of the avoid material). Executor CHEM-1 (`chem_plant.py`):
open-loop 20 x ADIM recipe in [-1, 1], quality in [0, 1] as defined there. Targets pinned per batch.
**Gate (local, before this section was written).** Wurtz: the benchmark's heuristic (heat, add the target's reagents) scores
0.454 on dodecane / 5,6-dimethyldecane / 4,5-diethyloctane and 0.277-0.278 on the other three organic targets; a CMA-ES
4,000-evaluation search over the recipe reaches only 0.366-0.376 and 0.253-0.255, so the heuristic is at or near the optimum
of this bench; random recipes 0.16-0.19. NaCl is nearly saturated by random (2.04 vs heuristic 2.18 of 3). Fict: heuristic
0.94 / 0.97 / 0.97 on E / F / G and 0.18 on I (a two-stage recipe); random -0.27 to 0.52. Reading: on these benches the
heuristic is a near-optimal corner recipe for most targets, so the acquisition test measures speed to a known optimum; the
open question with headroom is Fict target I. *CMA-ES 4,000-evaluation values for Fict (18:00 UTC, `runs/chem/fict_gate.json`):
E 0.853, F 0.117, G 0.387, I 0.111 (returns), against the heuristic's 0.941 / 0.973 / 0.974 / 0.176: black-box search in
140 dimensions does not find the corner recipe within 4,000 evaluations, so the heuristic is the reference optimum for E, F,
G and the achievable value for I remains unknown.*
**Machina acquisition** (`chem_machina_acquire.py`): one Domain per target, `episode_credit`, arms `default` and `profile`,
300 attempts, state rows = the bench's thermodynamic and in-hand observation entries. 11 targets x 2 arms = 22 runs.
**Comparators** (`chem_local.py`, `chem_sb3.py`): random, (1+1)-ES, CMA-ES at 300 attempts from u = 0 (mid-range), 5 seeds
per target; the heuristic; the 4,000-evaluation CMA-ES reference; SB3 PPO, A2C, SAC and TD3 (the paper's algorithms, [32, 32]
MLP as in the paper) on the benchmark's own closed-loop protocol with the target redrawn every episode and encoded in the
observation, at the paper's 100,000 steps (3 seeds) and at 1,000,000 steps (PPO, 3 seeds), evaluated deterministically per
target. Metrics: best quality after 16 / 32 / 64 / 128 / 300 attempts per target (mean over the 2 arms vs mean over the 5
comparator seeds), attempts to reach the heuristic's value, and the deep-RL agents' per-target returns after training.
Differences are called when seed/arm ranges do not overlap (descriptive at these counts).

## E. XLand-MiniGrid (NeurIPS 2024 Datasets & Benchmarks) through macro-actions, sequential route, production API (`xland_bench/`)

Added 2026-11-07 ~17:30 UTC, before any counted run (smoke test: 2 trivial rulesets, seed 99, excluded; its Domain deletion hit
a production 502 and was completed afterwards by hand). Service: production `https://rei-neuroadapt-api.reilabs.org/api/v1`,
build `f5e1ab8`, schema 5.

**Question.** On the Alchemy-shaped task, hidden production rules and a hidden goal resampled every episode with several trials
per episode, does Adapt-1's sequential route discover and exploit the structure within an episode (improvement from trial 1
to trial K) and does retained state across episodes help, compared with no-knowledge scripts and a PPO policy trained on the
same macro decisions?

**Environment.** `xminigrid` 0.9.3, `XLand-MiniGrid-R1-9x9` (one room, 243 primitive steps per trial), rulesets generated with
the benchmark's own generator restricted to the three position-free families (AgentHold, AgentNear, TileNear; the released
benchmarks also sample directional variants), tiers trivial (chain depth 0, 3 distractor objects), small (depth 1, up to 2
distractor rules, 2 distractor objects, prune 0.3) and medium (depth 2, up to 3 distractor rules, 2 distractor objects, prune
0.1), 3,000 rulesets each (`gen_rulesets.py`, seed 2026). Episode = one ruleset for K = 6 trials on fresh layouts; trial reward
is the environment's (1 - 0.9 steps/243 at the goal, else 0). Goal and rules are never shown. Rulesets with more than 8
initial object classes are skipped and recorded.
**Macro-actions** (`xland_macro.py`): pick / goto / put for each of 10 class slots (initial classes, then products in order of
first appearance) plus drop; executed with the environment's primitive actions by breadth-first search. Inapplicable macros
are no-ops with reward 0.
**Adapt-1** (`xland_run_adapt1.py`): docs' recommended sequential declaration; context = slot present/held/known/class
codes + holding, trial and step fractions, macros in trial, last macro success and reward; policies = the 31 macros with
operation one-hot and slot index as features (`action_transfer_strength` 0.05); reward bounds [0, 1]; sequential learning
discount 0.9, `n_step` 6, eligibility traces; TCP (lag 6) over the scalar summaries; latent belief on. Abstention -> a random
applicable macro, counted. Arms: `retained` (one Domain per run, each ruleset one episode) and `tasklocal` (a fresh Domain per
ruleset, deleted after its trials). Rulesets 0-149 of each tier for small and medium, 0-49 for trivial; seed 0 (one seed per
arm and tier: 4 + 2 runs).
**Comparators** on the same rulesets and layout seeds (`xland_local.py`, `xland_sb3.py`): random applicable macro; the
systematic explorer (tries every untried goto/pick/put in a fixed order, replays the first rewarded sequence); the oracle
(knows goal and rules, shortest plan: the ceiling); SB3 PPO on the same features and macros, meta-trained on rulesets
1000-2999 of the tier for 100,000 macro decisions (about 1,500 episodes), evaluated deterministically on rulesets 0-149;
the published PPO-RNN results on the raw environment as context only (1e9 steps, not reproducible here).
**Metrics.** Mean trial reward by trial index 1..6 and overall, averaged over rulesets, per tier and arm; fraction of
rulesets solved at least once; macros per solved trial; fallback rate. The within-episode learning signal is the difference
between trial 6 and trial 1. Differences are called when a paired sign test over rulesets gives p < 0.05.

## F. MiniHack skill tasks (NeurIPS 2021 Datasets & Benchmarks), sequential route, production API (`minihack_bench/`)

Added 2026-11-07 ~18:10 UTC, before any counted run. MiniHack 1.0.2 / NLE 1.3.0 built here from source.

**Question.** On NetHack skill tasks whose solution depends on an item with a randomised appearance and a multi-key use
sequence (use verb, confirm with the fire key, answer a follow-up prompt, then act), does Adapt-1's sequential route learn
the sequence and the item's effect from reward alone, and how does its data efficiency compare with PPO on the same features?

**Tasks** (Restricted action sets, the benchmark's own): LavaCross-Levitate-Potion-Inv (11 actions), LavaCross-Levitate-
Ring-Inv (12), LavaCross (16; a random item type lies on the floor), WoD-Medium (11), WoD-Hard (11). Reward: +1 at the
goal, 0 on death, the benchmark's -0.01 per step; cap 250 steps. Local measurements before the runs (30 random episodes):
random wins 13/30 on Potion-Inv and 11/30 on Ring-Inv (random key presses do stumble on quaff-then-confirm), 0/30 on
WoD-Medium. Scripted use-then-walk sequences win rarely because the exit position varies.
**Adapt-1** (`mh_run_adapt1.py`): context = `mh_env` features (position, offsets to exit / nearest item / nearest monster,
8-neighbour lava mask, inventory class flags, inventory growth, levitation, message class, HP, step fraction, last action);
policies = the task's actions with action index and move flag as features; reward bounds [-0.1, 1]; sequential learning
discount 0.95, `n_step` 8, eligibility traces; TCP (lag 6) over position, levitation, inventory growth, HP and message flags;
latent belief on. 500 episodes per run, seeds 0 and 1, five tasks (10 runs); abstention -> random action, counted.
**Comparators** (`mh_local.py`, same seeds): random (500 episodes); the knowledgeable script (use item, confirm, walk to
the exit; WoD: pick up, zap at the minotaur) as a reference, not a baseline; SB3 PPO (MLP) on the same features at 1,000,000
environment steps, 2 seeds per task, evaluated on 200 episodes; the published IMPALA results (CNN on glyphs, 1e7-1e9 steps)
as context only.
**Metrics.** Win rate over the last 100 of 500 episodes and over all 500 (online); mean episode length; the episode at which
the running 100-episode win rate first exceeds random's; for PPO the win rate at matched environment steps (its training
curve at the step count Adapt-1 used) and after its full budget. Differences are called when seed ranges do not overlap.

*Amendment E1 (18:25 UTC; the first launch had completed 0-4 rulesets per run, all discarded): the integration now enforces
macro eligibility. A selected macro whose class is absent (or put/drop with nothing held) is treated like an abstention: a
random applicable macro is executed and the step is counted as a fallback, with the count of inapplicable selections kept
separately. In the first launch Adapt-1 spent entire episodes on inapplicable no-op macros that consume no environment
steps, so trials never ended and every episode scored 0 at the macro cap, which measures a harness property, not learning.
Production latency for query/feedback calls was about 4 s under load, so the runs are relaunched four at a time (small and
medium, both arms); trivial follows.*

## G. Craftax-Classic-Symbolic (Matthews et al. 2024), sequential route, production API (`craftax_bench/`)

Added 2026-11-07 ~18:10 UTC, before any counted run. The deliberately hostile test: 22 achievements with long, sparse
dependencies (collect wood -> table -> pickaxe -> stone -> ...), which is the documented weak spot of the sequential route.
**Environment.** `craftax` Craftax-Classic-Symbolic-v1, untouched; episodes capped at 500 steps (the benchmark's own cap
is 10,000; the cap is set by the API cost of one decision per step and recorded here). Reward = Craftax's own.
**Adapt-1** (`cx_run_adapt1.py`): context = 129 features from the environment state (inventory, stats, block counts in the
view, offsets to the nearest tree/stone/coal/iron/diamond/water/table/furnace/grass/lava, mob proximity, facing block,
achievements unlocked so far, step fraction, last action); policies = the 17 actions; reward bounds [-1, 1]; sequential
learning discount 0.95, `n_step` 10; TCP (lag 6) over inventory and stats; latent belief on. One run: seed 0, 20 episodes
(up to 10,000 decisions).
**Comparators** (`cx_local.py`, same caps): random (100 episodes); SB3 PPO (MLP) on the same features for 1,000,000 steps,
evaluated on 20 capped episodes; the published PPO-RNN results (1e9 steps, 10,000-step episodes) as context only.
**Metrics.** Achievements per episode (count and which), return per episode, trend over the 20 episodes; PPO's achievements
at matched environment steps (10,000) and after its full budget.

*Amendment F1/G1 (18:15 UTC, before any counted F or G run): a probe Domain on production measured 1.2-3.5 s per query and
3.3-12 s per feedback with four XLand and three ChemGym clients active, so one decision costs about 10 s there. Sections F
and G therefore run on Fast-UAT (`af6670b`, the round-2 server, about 3 s per call with the six Industrial Benchmark runs),
not on production, and start with the budgets the latency allows: F = three tasks first (Potion-Inv, LavaCross,
WoD-Medium), seed 0, 150 episodes each; Ring-Inv and WoD-Hard follow if time permits. G unchanged (20 capped episodes).*

## Operational note 5 (2026-11-07 ~19:15 UTC): Fast-UAT credential change; all Fast-UAT work restarts in the second session
At 19:04 UTC the Fast-UAT key saved in this session's environment started returning 401 ("Invalid bearer token") during a
credential change. All six Industrial Benchmark runs crashed at episodes 14-15 of 150, and their final Domain deletions also
failed with 401. The two PenSim refinement arms were frozen at their first update and then stopped. The key now saved in
the second environment (`nofx-uat2`) maps to the original Fast-UAT engine session (`68b3f2be...`), so note 4's premise of a
separate queue does not hold: Fast-UAT work again shares one queue. At the user's instruction everything on Fast-UAT starts
over in that second session, from scratch, on new Domains: Industrial Benchmark (A, all six runs, 150 episodes each),
PenSim refinement (C, both arms), then MiniHack (F) and Craftax (G). The crashed first attempts are kept in
`runs/ib/adapt1/crashed-1/` and `runs/pensim/crashed-1/` and reported. Their eight Domains, which exist under the same
engine session, are deleted from the second session by exact id. The Industrial Benchmark runner now also uses the
reconciled feedback path. The PenSim import shims are vendored in `vendor_shims/` so the second container can run PenSim.
Local comparators stay in this session.

## Operational note 4 (2026-11-07 ~19:00 UTC): F and G move to a second Fast-UAT key in a second session
All requests from this session's container carry this environment's saved key, because the proxy overwrites any
Authorization header; an invalid key sent by hand was accepted as the saved identity. One key is one engine session and
one server queue. The user supplied a second Fast-UAT key. Sections F and G therefore run from a second cloud session on a
separate environment (`nofx-uat2`) that holds only that key, so they use their own queue and their own CPU, while the
Industrial Benchmark and PenSim runs continue here on the first key. Same server (`/version` commit `af6670b`, schema 5),
same runners and declarations as committed on this branch, same reconciled client. Plan there: the three F tasks
(Potion-Inv, LavaCross, WoD-Medium; seed 0, 150 episodes each) run together as one experiment; then G (seed 0, 20
episodes of 500 steps); then Ring-Inv and WoD-Hard. Outputs go to the branch `claude/zen-carson-n5bvi0-uat2` and are merged
here. Notes from that session go to `runs/uat2/NOTES.md`. Local comparators for F and G stay in this session.

## Operational note 3 (2026-11-07 ~18:45 UTC): dropped connections and feedback reconciliation
Production began resetting connections mid-request (`ConnectionResetError`) at about 18:05 UTC. The XLand task-local arm
crashed on a Domain creation after 3 rulesets, and the retained arm crashed on a feedback after 2 rulesets. Two production
probes (`probes/` in the scratchpad, Domains deleted, 404 readbacks) showed that (1) a feedback resent with the same
`decision_id` is accepted and applied a second time (`feedback_policy.sample_count` 4 -> 5), so a blind resend
double-counts, and (2) `GET /domains/{id}` `event_count` rises by exactly one per feedback and not per query. The client
now reconciles an ambiguous feedback POST by re-reading `event_count` and resends only if the record did not land; a count
it cannot explain stops the run (`Client.feedback`, unit-tested against a fake server). Domain creation is reconciled by
reading the Domain back. XLand, MiniHack and Craftax use this path from now on. The Industrial Benchmark runs on Fast-UAT
were started earlier with one blind resend per ambiguous feedback; each such resend is counted in their summaries
(`feedback_reconciled`) and reported as a possible duplicate.
XLand restarts: the retained arm's two partial attempts (crash at ruleset 2; a second attempt stopped by me after 32 macros
to switch to the reconciled client) are kept in `runs/xland/crashed-retained-{1,2}/` and reported; the arm restarts from
ruleset 0 on a new Domain. The task-local arm uses a fresh Domain per ruleset, so it continues from ruleset 3; rulesets 0-2
completed with no resends. Ruleset 3 was interrupted twice without an episode row, and both of its Domains were deleted.
Scheduling (18:45 UTC, no effect on any result): the container has 4 cores and ran at load 18, which starved the PenSim
refinement rollouts. Local comparators now run at nice 15 and Adapt-1 clients at nice 0. The client waits at least the
server's Retry-After on 429/503. The proxy credential maps to one engine session per server, so concurrent runs on one server share
one queue; this is why the per-call latency grows with the number of live runs.

## Operational note 2 (2026-11-07 ~18:15 UTC): one experiment per lane
At the user's request the parallelism is cut. Live on production: the two *small*-tier XLand runs and the ChemGym Machina
queue (3 at a time). Live on Fast-UAT: the six Industrial Benchmark runs and the two PenSim refinement arms. Stopped and
rescheduled to run afterwards, in this order: XLand medium (both arms) and trivial; MiniHack (3 tasks); Craftax. Their first
minutes of output were discarded and their Domains deleted (404 readbacks). Local comparators run one queue at a time.

## Operational note (2026-11-07 ~18:00 UTC)
Launching 22 ChemGym Machina runs on top of 6 XLand runs drove the production service into 503 bursts (a Domain list
took 9.5 s). The ChemGym runs were stopped after 0-1 attempts each, their 22 Domains deleted (404 readbacks), and the
same runs re-queued three at a time; one XLand run (trivial, task-local) crashed on exhausted retries and was relaunched
from its first ruleset; the client now retries 429/503 for about six minutes. No counted result was affected.

## Validity rules (all parts)
- A run counts only if it completes; an aborted or crashed run is reported, not dropped, and rerun once from scratch with
  the same seed and a new Domain; both are reported. Adapt-1 runs abort if more than 1 % of feedbacks are not applied.
- *Amendment (15:40 UTC, Industrial Benchmark runs at episode 2 of 150, about 7 min per episode under server load): if the
  runner process is lost (container restart), an Industrial Benchmark run is resumed on the same Domain from the episode
  after the last complete one (`--resume`); the interrupted episode index is skipped, its partial feedback stays in the
  Domain's learned state, and every resume is recorded in `resumes-*.jsonl` and in the summary. Total completed episodes
  stay at 150.*
- Every Domain is deleted at the end of its run and the deletion is confirmed by a 404 readback.
- Raw per-episode / per-attempt files, declarations, server echoes and diagnostics are committed; derived metrics are
  computed from them by `analyze.py` and kept separate.
- Order of work if time runs out: A and B first, C after; anything not run is listed as not run.
