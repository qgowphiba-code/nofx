# Adapt-1 on science tasks: published benchmarks and follow-ups (2026-11-07)

Service: Fast-UAT `https://rei-neuroadapt-fast-uat.reilabs.org/api/v1`, build `af6670b`, schema 5. Machina configs report
`"experimental": true`. Every Domain created for these runs was deleted and confirmed by a 404 readback; the account had no
Domains left at the end. Each experiment has a pre-registration file written before its counted runs (`runs/PREREG*.md`).

## Bottom line

| Question | Answer |
|---|---|
| Machina acquisition on published open-loop control problems | Level with the best untuned baseline: matches (1+1)-ES, beats default CMA-ES and random, reaches 98-100 % of the gap to the published optimum in 600 runs. |
| Machina contextual refinement (Park-Ramirez with measured batch variation) | Strongest result of the day. Recovers **86 %** of the gap between the acquired base and the per-batch optimum on 32 unseen batches, better on **32 / 32** of them than a linear policy trained by CMA-ES on the same 7,680 executions (67 %). The context-free (shared) arm recovers 13 %, so the gain is context-dependent correction. |
| Adapt-1 query/feedback on published reaction datasets (Buchwald-Hartwig, Suzuki) | Carries knowledge across substrates; matches a GP that gets the same cross-substrate information on Suzuki, trails it on Buchwald-Hartwig. On the first substrates (little retained state) it is at or below random sampling. |
| Sequential route with reward only at the end (inverse folding, light) | No sign of delayed credit: random-level in training, below a tabular Monte-Carlo learner held out. |

Machina is the part of the capability set that held up; its pipeline is where it separates from generic optimizers.
Acquisition alone is "as good as a good evolution strategy"; acquisition followed by contextual refinement beat every
same-budget learner I fielded.

---

## 1. Published benchmarks

### 1a. Machina acquisition on published open-loop control problems

Formulations from the literature, each verified by reproducing its published optimum on the same 20-stage
piecewise-constant discretization before any API call:

| Problem | Source | Published optimum | Our 20-stage reference | Random-profile mean |
|---|---|---:|---:|---:|
| batch: A->B->C temperature profile | Ray 1981; Luus; APMonitor/GAMS lib | 0.6108 | 0.61045 | 0.5146 |
| catmix: catalyst mixing | Gunn & Thomas 1965; COPS #14 | 0.048056 | 0.047926 | 0.03385 |
| parkram: fed-batch secreted protein | Park & Ramirez 1988; Luus; Banga et al. | ~32.6-32.7 | 32.362 | 17.863 |

Best J after 600 runs (Machina: 4 runs = default and docs-profile arms x 2 proposal seed streams; local: 10 seeds):

| Problem | Machina (mean of 4) | (1+1)-ES | CMA-ES | Random search |
|---|---:|---:|---:|---:|
| batch | 0.61003 | 0.61044 | 0.60889 | 0.58862 |
| catmix | 0.04796 | 0.04775 | 0.04781 | 0.04333 |
| parkram | 31.85 | 32.00 | 31.16 | 27.95 |

Permutation tests at 600: vs (1+1)-ES, batch -0.0004 (p = 0.001), catmix +0.0002 (p = 0.05), parkram -0.15 (p = 0.37);
vs CMA-ES, batch +0.0011 (p = 0.11), catmix +0.00015 (p = 0.001), parkram +0.69 (p = 0.02).
Full curves (n = 16 ... 600) are in `results/A1_machina_published_control.md`. On these problems every reasonable method
closes 92-100 % of the gap by 600 runs, so they separate methods mostly on speed; Machina's default arm is often the
slowest before ~64 runs, the docs-profile arm is among the fastest on batch and parkram.

### 1b. Machina contextual refinement (the stage no ES has)

Problem: the published Park-Ramirez model with two quantities that vary from batch to batch and are measured before it
starts: feed substrate concentration (12-28; published 20) and initial glucose (3-7; published 5). This variation is our
addition. Local check first: the nominal-optimal feed profile loses 30 % at feed 12, 19 % at 15, 6 % at 18.
Pipeline: base = Machina's best acquired profile at the published conditions (J = 31.99, from 1a) -> `contextual_execution`.
Budget per arm: 30 updates x 8 candidates x 32 training batches = 7,680 executions. Evaluation: 32 unseen batches,
frozen `evaluate`.

Held-out mean J (base alone 27.69; per-batch ceiling 32.04):

| Arm | after 5 | 10 | 20 | 30 updates | Gap recovered |
|---|---:|---:|---:|---:|---:|
| Machina linear, radius 0.2 | 29.44 | 30.77 | 31.19 | **31.42** | **86 %** |
| Machina linear, radius 0.08 (docs default) | 28.38 | 30.07 | 30.81 | 31.14 | 79 % |
| Machina quadratic, radius 0.2 | 29.01 | 28.76 | 29.35 | 30.12 | 56 % |
| Machina shared correction (no context; documented control) | 28.11 | 27.95 | 28.37 | 28.25 | 13 % |
| Linear context->offset policy, CMA-ES, same 7,680 executions (3 seeds) | | | | 30.60 | 67 % |
| Same policy, (1+1)-ES (3 seeds) | | | | 27.67 | -1 % |

Paired over the 32 unseen batches, Machina linear (0.2) minus the CMA-ES policy: +0.82, better on 32 of 32.
Caveats: one run per Machina arm (the service is deterministic given seeds); the learned-policy comparators are mine and
untuned; the radius 0.2 arm departs from the docs default (the default arm also beats the CMA-ES policy).

### 1c. Adapt-1 query/feedback route on published HTE reaction datasets

Data: the EDBO repository (Shields et al., Nature 2021) copies of Buchwald-Hartwig amination (Ahneman et al., Science 2018;
15 aryl halides x 264 conditions) and Suzuki-Miyaura coupling (Perera et al., Science 2018; 12 substrate pairs x 308
conditions). Recorded yields, no simulator. One query = one experiment; 30 experiments per substrate, no repeats; one Domain
across all substrates in a fixed order (3 orders); context = substrate identity (one-hot leaving group and core/partner).
Comparators with the same information and budget: random; GP-EI per substrate (one-hot conditions, no memory of other
substrates; the one-hot form of EDBO's baseline); GP-EI with transfer (substrate + condition one-hot, all earlier substrates).
Metric: best yield found / best recorded yield for that substrate.

Later substrates (positions 6-15 / 5-12 in each order, 30 / 24 substrate visits):

| Dataset | Method | n = 5 | n = 10 | n = 30 |
|---|---|---:|---:|---:|
| Buchwald-Hartwig | random | 0.747 | 0.820 | 0.912 |
| | GP per substrate | 0.841 | 0.911 | 0.976 |
| | GP with transfer | 0.937 | 0.974 | 0.994 |
| | Adapt-1 | 0.838 | 0.893 | 0.939 |
| Suzuki | random | 0.763 | 0.836 | 0.922 |
| | GP per substrate | 0.735 | 0.801 | 0.955 |
| | GP with transfer | 0.856 | 0.899 | 0.973 |
| | Adapt-1 | 0.758 | **0.915** | 0.959 |

Paired: Suzuki n = 10, Adapt-1 - GP per substrate +0.114 (better on 15, worse on 6 of 24), Adapt-1 - GP with transfer +0.016
(7 vs 10). Buchwald-Hartwig n = 10, Adapt-1 - GP with transfer -0.081 (better on 1, worse on 24 of 30).
First substrates in each order: Adapt-1 at or below random (Buchwald-Hartwig n = 30: 0.844 vs random 0.875).
Diagnostics: every selection is labelled `core_exploration` (Thompson); overrides for already-run conditions are 7-11 %,
so the plateau is not caused by the eligibility rule. Full tables: `results/B1_hte_published_datasets.md`.

---

## 2. Our own simulators (secondary)

### 2a. Batch cooling crystallization x Machina (`results/S1_crystallization_machina.md`)
Seeded cooling crystallization with hidden kinetics. Compounds were pre-selected for headroom (cubic textbook profile at
least 0.08 below the optimum; 5 of 14 candidate compounds were rejected as forgiving). At 400 batches Machina (6 runs) led
every local method on all three compounds: vs (1+1)-ES +0.007 (p = 0.16), +0.004 (p = 0.60), +0.026 (p = 0.005); vs CMA-ES
+0.014 (p = 0.03), +0.006 (p = 0.17), +0.051 (p = 0.003). Refinement was skipped as pre-registered: in this model the best
profile does not move with batch conditions (gap <= 0.01), which is what led to the Park-Ramirez test in 1b.

### 2b. Simulated reaction-condition optimization x Adapt-1 (`results/S2_simulated_reactions.md`)
48 conditions, reactions from four recurring kinetic families, measured descriptors as context, 20 experiments per
reaction. On reactions 20-39: Adapt-1 with descriptors 0.933 of optimum after 5 experiments vs GP per reaction 0.699
(better on 19 / 20) and GP with transfer 0.981 (worse on 12 / 20). The no-context arm survived a 502 that was reconciled
through `event_count` and resumed.

### 2c. Inverse folding, 2D HP, reward only at the end (light; `results/S3_inverse_folding.md`)
25 designable 12-residue targets, exact enumeration. Adapt-1 sequential (eligibility traces, n_step 12): training reward at
random level for 60 episodes (0.000-0.002), 0.029 in the last 20; held out 0.156, carried by one target where its nearly
constant output happened to be optimal. Tabular Monte-Carlo control on the same context: 0.291 held out. The burial
heuristic equals the ceiling on held-out targets, as predicted: a near-solver control.

---

## Run notes
- A container restart interrupted Buchwald-Hartwig order 1 and Suzuki order 2 mid-episode. Their journals could not
  account for the in-flight feedback, so both Domains were deleted and both orders rerun from scratch; the interrupted
  files are kept under `runs/hte/INTERRUPTED-*` and excluded from analysis. Journals are now line-flushed plain JSONL.
- Ambiguous writes (5xx on a mutation) are reconciled by reading state (`event_count`, trajectory `version`,
  `pending_decisions`) before any resend; one such event occurred and was resolved as not applied.
- The best-known crystallization references were recomputed after Machina exceeded the first reference on compound 9.
- Local baselines are textbook defaults (no tuning); the GP uses one-hot encodings, not EDBO's DFT descriptors.


---

# Rounds 3-5 (same day): where general learners need more data, and more fields

Ground rules unchanged: pre-registration before counted runs (`runs/PREREG_round3.md`), a local headroom gate before any
refinement test (a task runs only if the nominal profile loses >= 3 % against each batch's own optimum), held-out
evaluation on unseen batches/patients, and same-data comparators with a data multiplier (1x-8x).

## Summary

| Test (field) | Machina / Adapt-1 | Best same-data comparator | Verdict |
|---|---|---|---|
| Park-Ramirez refinement, 2 measured vars (fed-batch) | 86 % of gap at 7,680 runs | CMA-ES policy 49 % at 1x, 85 % at 4x; NES 80 % at 4x | needs 4-6x less data |
| Park-Ramirez refinement, 5 measured vars incl. a distractor | 31.0 (41 % of a 14-unit gap) at 1x | CMA-ES 26.4 at 8x; NES 27.9 at 4x | > 8x less data |
| Penicillin fed-batch refinement (Cuthrell & Biegler) | 80.2 at 1x | CMA-ES 78.3 at 4x; NES 80.1 at 2x | 2x (NES) to > 4x (CMA-ES) |
| Pump drift (unmeasured 30 % shortfall), recovery | 30.55 after 5,120 runs | CMA-ES continued 29.9, restarted 27.7 | recovers best, slower in the first 5 updates |
| Insulin dosing across 10 unseen virtual patients (UVA/Padova) | 0.518 at 1x; **0.591 at 2x** | CMA-ES 0.544 at 1x, 0.563 at 8x; NES 0.561 at 2x; bolus calculator 0.510 | slower start, then ahead of all comparators up to 8x |
| RL bioprocess plant (Petsagkourakis et al.; published REINFORCE budget 35k episodes) | 1.15 in 256-600 runs | NES needs ~2,500; CMA-ES/ES 1.07/1.06 at 600 | 4-10x less than NES |
| Protein design, GB1 four-site landscape (real data) | 6.0 at 384-500 (no global max) | GP-guided design 8.5, global max in 4/5 | loses; level with MLDE |
| Sachs signalling network (real data), Discovery's chosen predictors | precision 0.39 | Lasso on the same 50 cells 0.52 | loses; above random (0.32) |
| Materials / photovoltaics experiment planning (Olympus) | crossed barrel 0.90 @50; PCE10 0.66 @50 | GP-EI 0.96 / 0.97 | between random and GP / at random |
| Park-Ramirez refinement, 10 measured vars (7 relevant + 3 distractors) | 28.3 at 1x (16 % of a 14-unit gap) | CMA-ES 25.7 at 8x; NES 26.6 at 4x (no progress) | only learner that moves; small absolute gain |
| Rejected by the gate (not run): batch reactor and catalyst mixing with batch variation | | | profile barely moves with conditions (1-2 %) |

Pattern: Machina's contextual refinement is the one component with a consistent data-efficiency edge on condition-sensitive
batch processes (three fed-batch tasks, up to 5 measured variables, with drift). On patient-to-patient
insulin dosing it started slower than a CMA-ES policy but led every comparator once given 2x the data. At 10 measured variables
all learners stall; Machina is the only one that improves at all. The query/feedback and Discovery routes trailed standard baselines on every real dataset tried.

## Details and caveats

- Data multiplier (Park-Ramirez, 2 vars): held-out % of gap at 7,680 / 15,360 / 30,720 / 61,440 runs: CMA-ES 49 / 78 / 85 / 93,
  NES 48 / 68 / 80 / 99 (2 seeds each). General learners do catch up with enough data.
- 5 vars: CMA-ES seed 0 at 1x-8x: 25.2, 24.5, 26.8, 26.4; NES 26.3, 26.5, 27.9 (more seeds were launched; see raw files).
  Shared (context-free) correction stayed below the base (24.9 vs 25.3): context is required.
- Penicillin: my per-batch ceiling (1,200 runs per batch on a stiff model) came out below Machina's result, so it is
  underestimated; values are reported in absolute terms. Our implementation reaches 86.3 vs published 87.7-88.0 (20 stages).
- Drift: right after the change the refined policy scored below the uncorrected base (24.2 vs 25.9); 5 updates later 25.4,
  10 updates later 29.0 (ahead of every comparator at that point), 20 updates later 30.6. No search-radius collapse.
- Insulin: model re-implemented from simglucose and verified (all 30 patients hold basal glucose exactly; meal responses
  physiological). Base = Machina's profile acquired on one adult; held out: 10 patients x 3 meals. Quadratic 0.473,
  shared 0.466. Longer (60 updates) and larger-radius (0.4) arms were launched to test whether step size is the limit.
- RL plant: the authors' code-default parameters allow an integration blow-up in the complex model; a coarse integrator let
  CMA-ES and random search report a spurious optimum of 36. With 80 sub-steps and non-negative states the best known value
  is ~1.155 (reached by Machina's profile arm and NES). Machina never touched the artefact region (identical values on the
  corrected simulator). Baselines were rerun on the corrected simulator; the invalid ones are kept under runs/INVALID-rl-artifact.
- GB1: a no-repeat rule bug let the first UCB arm re-measure LAAA 79 times (Adapt-1 had locked onto the prefix L-A-A); that
  run is marked invalid and was rerun with branch-exhaustion checks. Published reference (MLDE study): focused-training MLDE
  reaches the global max up to 92 % of runs at 470 variants; single-step greedy 1.2 %.
- Sachs: pooled data without intervention labels (cdt copy), so this tests predictor selection, not causal direction;
  causal variables stayed empty. Discovery committed its input paths after the first 50 cells and never revised them through
  1,000 cells; baselines therefore get the same 50 cells (full-data baselines shown for reference).
- Not run: TrpB landscape (Zenodo blocked by this environment's network policy); robotics (dropped at the user's request).

- Insulin, longer runs: radius 0.2 reached 0.529 at 40 updates and 0.591 at 60 (15,360 runs); radius 0.4 reached 0.533 at
  20 and 0.579 at 60. Per-patient ceiling 0.665, so 0.591 is 69 % of the gap.
- 5 vars, second comparator seeds: CMA-ES 25.2 / 26.0 / 26.0 / 27.4 at 1x / 2x / 4x / 8x; NES 25.5 / 26.4 / 26.4. Machina 31.0 at 1x.
- 10 vars: Machina 27.25, 27.14, 26.99, 28.31 at 5/10/20/30 updates (not monotone); shared 25.8; base 26.1, ceiling 40.1.

## Addendum: replicates of the refinement result (different service seeds and training-batch draws)

| Task | Machina linear, 30 updates (7,680 runs), per seed | Mean | Same-data comparators |
|---|---|---:|---|
| Park-Ramirez, 2 measured vars (base 27.69, ceiling 32.04) | 31.42 / 31.55 / 31.41 | 31.46 (87 % of gap) | CMA-ES 29.8 at 1x, 31.4 at 4x; NES 31.2 at 4x |
| Park-Ramirez, 5 measured vars (base 25.28, ceiling 39.27) | 31.00 / 32.07 / 31.47 | 31.51 (45 %) | CMA-ES <= 27.4 at 8x; NES <= 27.9 at 4x |
| Penicillin (base 70.27) | 80.23 / 81.44 | 80.8 | CMA-ES 78.3 at 4x; NES 80.1 at 2x |

The spread between seeds is small next to the gaps to the comparators. GB1 UCB rerun finished at 5.89 (500 unique variants,
no global maximum). At the end of the session no Domains were left on the account.
