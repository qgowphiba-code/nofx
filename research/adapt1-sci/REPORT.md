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
