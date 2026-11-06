# HP × Adapt-1, set 2: full experiment plan

**2026-11-06. Plan only: nothing has been run against the API.**
- Supersedes `hp-adapt1-next-set-plan.md` (revision 2).
- **Starts from a clean slate:** you confirmed that all set-1 domains are deleted, so no step reuses one.
- **Inputs:** the set-1 package, docs.reilabs.org, Rei's "Coupled Learning and Planning" article, and the local checks in `analysis/` (§C).

## 1. Summary

**Goal.** Establish whether Adapt-1's value route can learn a better HP evaluator than a strong hand-made heuristic, using the substrate's own documented method. Then check whether the findings carry to two neighboring areas.

**Why a second set.** Set 1 established three things:
- The label recipe fixes weak data.
- Own-policy data is the biggest lever.
- Its best long-chain model (123) still trails the contact heuristic, which scores 133.6 ± 4.6 at depth 3 over 300 tie-break seeds.

Set 1 never tried three things:
- **Blended targets.** `y = λ·u + (1−λ)·z̄` (search assessment plus observed outcome) is the documented method, and Rei used λ = 0.75 for chess.
- **Richer inputs.** Inputs that carry counts are how Rei fixed state aliasing in chess.
- **A fair baseline.** Comparisons against the heuristic's spread rather than one draw.

**Primary claim under test.** At equal search depth, learned values beat the contact heuristic on held-out long chains: either alone, or blended with the heuristic at the leaves.

**Defaults assumed.** You did not choose; override any of these:

| Decision | Default | Reason |
|---|---|---|
| Host | **Production** (`rei-neuroadapt-api.reilabs.org`), gated by a calibration replicate of set 1 (T0.3) | Your environment holds a production key and the docs target production. T0.3 tells whether set-1 numbers transfer. Fast-UAT would need its host and key added |
| Primary claim | Evaluator quality at equal depth. Search efficiency is secondary (C1) | Separates learning from search budget |
| Inputs | Richer inputs allowed (counts as conditions); the raw graph is kept as an arm in A1 | Documented practice; A1 measures the difference |
| Neighbors | 2D HP (task neighbor) and a small solved two-player game (substrate neighbor). RNA and the sequential route are optional (Track F) | Cheap, exact grading, and covers both meanings of "neighboring" |

**Program at a glance.**

| ID | Experiment | Question it answers | API cost (requests / fits) | Runs after |
|---|---|---|---|---|
| T0.1 | Local harness, test sets, baselines | — | 0 | — |
| T0.2 | Host, build, limits, throughput | Does the host behave as documented, and how fast is it? | ~50 / 1 tiny | T0.1 |
| T0.3 | Calibration: replicate set-1 Stage A | Do set-1 numbers reproduce on this host? | ~150 / 2 | T0.2 |
| T0.4 | Client-side vs native planning | Is client-side planning identical to native planning? | ~600 / 0 | T0.3 |
| A1 | Representation probe | Can the value model carry the heuristic at all? | ~3k / 3 | T0.4 |
| A2 | Fit-settings probe | Why did set-1 fits keep their starting model? | ~400 / 5 | T0.3 |
| A3 | Target construction, oracle-graded | Do blended targets beat terminal labels and the set-1 recipe? Which λ and depth? | ~2.5k / ~22 | T0.4 |
| A3b | Blended targets on own-policy data | Does the gain survive own-policy data? (Set 1's recipe didn't) | ~300 / 4 | A3 |
| B0 | Bridge: rebuild M3 *(optional; needs `m_lineage_T_rows.json`)* | Are long-chain numbers comparable with set 1? | ~1.2k / 1 | T0.4 |
| B1 | **Main experiment: replicated, length-matched lineages** | Blended vs terminal targets; learned vs heuristic at depth 3 | ~30k / 18 | Gate G1 |
| B2 | Leaf blend and depth curves (part of B1's final evaluation) | Does learning add information beyond the heuristic? | in B1 | B1 |
| B3 | Promotion audit *(optional)* | Would the documented canary adopt the better model? | ~1.5k / 2 | B1 |
| C1 | Compute-matched comparison | Contacts per leaf evaluation, learned vs heuristic | ~1k / 0 | B1 |
| C2 | Per-chain adaptation on seq5–7 *(conditional)* | Attempts-to-target vs search-only | ~4k / 12 | Gate G3 |
| D0 | 2D environment, oracle, baselines | — | 0 | — |
| D1 | 2D short chains, oracle-graded | Do the label findings hold where dead ends matter? | ~700 / ~10 | D0, G1 |
| D2 | 2D long-chain lineage *(conditional)* | Learned vs heuristic in 2D at depth 3 | ~10k / 6 | Gate GD |
| E0 | Game environment, exact solver, heuristic | — | 0 | — |
| E1 | Game value learning, solver-graded | Same label findings with opponent nodes? | ~1k / ~10 | E0, T0.4 |
| E2 | Game play strength *(conditional)* | Learned vs heuristic search over paired openings (Rei's chess protocol) | ~2.5k / 0 | Gate GE |
| E3 | Game own-play generations *(optional)* | Does own play keep improving the values? | ~3k / 4 | E2 |
| F1–F3 | Optional neighbors | — | not budgeted | — |

**Budget.**
- **Core** (T0, A, B1–B2, C1, D1, E1): about **40k requests and ~75 fits**, most of them small. Set 1 used 54k requests.
- **Conditional and optional parts:** about 25k more requests and ~30 more fits.
- **Time:** at 1–2.5 calls/s (measured in T0.2), the core is 5–12 h of API time. Plan for three sessions (§11).

All costs are ±50% until T0.2 measures throughput and fit times.

## 2. Ground rules for every experiment

### 2.1 Fixed task encoding (HP)
- **Utility:** as set 1. `U = 0.5 + 0.5·contacts/ub` for a complete fold, and `0` for a trapped one, with ub from `lattice_env.State.upper_bound`.
- **Observation:** as set 1 (`hp.observation`: object graph in the chain-tip frame).
- **Conditions:** the 4 BASIC conditions, unless an arm says otherwise. The R1/R2 arms add the 14 `features()` values (the limit is 16).
- **Blended target, computed client-side and sent as `target`.** For recorded state `s_k` in an episode with final utility `U`:
  - `z̄_k = ½ + (U − ½)·γ^(T−k)`, with **γ = 1** for HP (deterministic, single-agent).
  - `u_k = Search_d(s_k; V_θ)`: the backed-up max value at depth `d`, with exact `terminal_value` at finished leaves and model predictions elsewhere (identical to native planning; T0.4 verifies this).
  - `y_k = clip(λ·u_k + (1−λ)·z̄_k, 0, 1)`.
  - Store `u_k`, `z̄_k`, λ, d, γ, the assessor's model sha, episode and step in the run record, as the docs ask.
- **Sibling rows (optional variant, "BLG"):** one row per legal child `c` of `s_k`, with `target = Search_{d−1}(c; V_θ)` (exact if `c` is terminal) and `metadata.comparison_group = parent`. Fit with `comparison_weight 1` and squared loss.
- **Built-in consistency check:** with λ = 1 and d = 1, the blended target equals set 1's `continuation` target, so arm BL(1, 1) in A3 should reproduce set 1's A2 result (0.762).

### 2.2 Evaluation suites (built locally once, frozen, sha256 recorded)
| Suite | Content | Primary metric |
|---|---|---|
| **Eval-S** (3D short, oracle) | 24 held-out chains of length 10–11: set 1's 6 evaluation chains plus 18 new ones. States from 4 random folds plus the optimal path per chain: ~860 states, ~600 informative | Best-move rate at depth 1 (ties averaged), regret, correlation with V\* and V_rand, greedy contacts vs the optimum. Report the 6-chain subset too, for continuity with set 1 |
| **Eval-L** (3D long) | 40 generated chains of length 36–64 (5 per length in steps of 4), disjoint from all training and panel chains | Depth-3 greedy fold contacts per chain, paired against the heuristic's mean over 100 tie-break seeds |
| **Eval-Liu** | The 7 Liu panel chains | Depth-3 total, against 133.6 ± 4.6 (heuristic) and 123 (set-1 M3). Also the development split for choosing λ_leaf (§2.5) |
| **Eval-S-2D**, **Eval-L-2D**, **2D panel** | Track D: 12 short chains of length 14–16 with oracle; 40 long chains; the 7 panel sequences with 2D best-known energies 9/9/8/14/23/21/36 | As above |
| **Eval-G** | Track E: solver-labelled positions and all 2-ply openings | Best-move rate vs the solver; points over paired openings |

**Planning in evaluation:** client-side MAX search via `value/predict` (1–256 inputs per call, ≤900 KiB per body) unless T0.4 fails, in which case use native planning. Models break ties at random; ties are rare with learned values.

### 2.3 Baselines (computed locally, frozen before any API run)
For every Eval-L, Eval-Liu and 2D chain:
- the contact heuristic at depths 1–5, 100 tie-break seeds each;
- the H-core-compactness tie-break variant at depth 3;
- search without learned values (neutral leaves) at depth 3;
- best-of-k search-only curves (k = 1–128, depth 3, ε = 0.1).

Numbers already computed are in §C.

### 2.4 Run protocol
Keep set 1's recovery protocol and `safe_client.py` (write-ahead intents, receipts, operation classes, reconcile-before-retry, STOP file, ≤2 requests in flight, owner-wide fit lock). Add these:
- **Build check:** `GET /version` at the host root at the start and end of every run. If it changes mid-run, abort.
- **Payload guard:** serialize with compact separators; split any body over 900 KiB.
  - `observe`: ≤50 rows per call.
  - `predict`: ≤256 inputs per call.
- **After every fit, check:**
  - `training.status` is a finished success state and `request_id` matches;
  - **the candidate sha differs from the previous one**, the check whose absence probably produced set 1's identical O2 generations;
  - save the full report, including `selected_epoch`, `optimization_applied` and `regularization_target`.
- **Identity checks:** every `predict` and planning call carries `expected_model_sha256`, and the returned evaluator identity is checked.
- **No promotion inside experiments:**
  - Evaluate candidates directly (`model_source: candidate` / `evaluator_source: value_candidate`).
  - Install only when a step needs an installed reference: continuation rows in T0.3, A2 and A3's ROB arm. Promotion is studied separately in B3.
  - There is one candidate slot per domain, so evaluate each generation fully before the next fit replaces it.
- **Dry runs:** each pipeline first runs end to end against a local mock server (heuristic or random-feature value function).

### 2.5 Statistics and pre-registered decision rules
- **Short oracle sets:**
  - Log the outcome of every decision.
  - Compare arms by paired difference in best-move rate, with a 95% bootstrap CI that resamples chains, then decisions.
  - One decision on Eval-S is about 0.0017; the SE of a rate is about 0.016.
- **Long chains:**
  - Per chain, Δ = model contacts − heuristic mean (or arm A − arm B).
  - Report the mean Δ, a chain-bootstrap 95% CI, a Wilcoxon signed-rank test, and the win rate.
  - Aggregate lineages by averaging chain-level Δ, then report per-lineage signs.
- **Confirmatory comparisons:** only these three, Holm-corrected. Everything else is exploratory and reported with effect sizes and CIs.
  - **R1, blended vs terminal targets:** mean Δ(BL − A0) at generation 2 on Eval-L across 3 lineages. The CI must exclude 0, with the same sign in ≥2 of 3 lineages.
  - **R2, learned vs heuristic:** the best pure learned arm at generation 2, depth 3, against the heuristic's depth-3 mean on Eval-L, with the same criterion.
  - **R3, learning adds information:** the leaf blend `(1−λ_leaf)·h + λ_leaf·V` with λ_leaf ∈ {0.25, 0.5, 0.75} against λ_leaf = 0, which is the heuristic itself.
    - λ_leaf is chosen on Eval-Liu (development) and tested on Eval-L (test), so the choice can't bias the test.

### 2.6 Domain hygiene
- **Naming:** prefix `hpx2-<exp>-<arm>-<lineage>-<timestamp>-<6hex>`. Never touch domains without the prefix.
- **One value domain per (experiment, arm, lineage).** At most 4 live at a time, since dataset-worker capacity is owner-wide.
- **Deletion:** delete each domain once its outputs are saved locally, and verify with GET 404.
- **`max_samples`:** set it to cover the whole lineage, for example 25,000.

## 3. Shared components (local, built in session S0)
- **C-plan, client-side planner:** depth 1–4 MAX search with batched `predict` across parallel folds, exact terminal values, random tie-breaks and leaf blending. Unit test: with heuristic leaves it must reproduce `lattice_env.plan` exactly; `analysis/fasthp.py` already does this, reproducing 68/113/134.
- **C-target, target builder:** blended targets with provenance; sibling-row builder; record ids that encode arm, λ, d and the assessor sha prefix, so changed content never reuses an id.
- **C-eval, evaluation suites:** the suites in §2.2, with an oracle cache for short chains (Eval-S builds in about 2 min).
- **C-base, baseline cards:** the §2.3 numbers as frozen JSON.
- **C-run, runner:** the `safe_client` extensions from §2.4, a scheduler (one fit queue plus two predict workers, interleaving lineages while a fit runs), a manifest writer (build, seeds, options, model shas, data hashes), and a mock server.
- **D-env (2D) and E-env (game):** see Tracks D and E.

## 4. Track 0: setup and calibration

### T0.2 Host, build, limits and throughput
- **Do:**
  - `GET /version`.
  - On one scratch domain: configure, observe 50 small rows, `predict` 256 inputs, one oversized `predict` to confirm the 413 boundary near 1 MiB, one depth-1 planning call, and one tiny fit to time the fit queue. Then delete the domain.
  - Time 200 `predict` calls of 125 L50 inputs with 2 in flight.
- **Output:** build id; measured calls per second and fit latency, which drive the re-budget at gate G0.
- **Cost:** ~50 requests, 1 tiny fit, ~15 min.

### T0.3 Calibration replicate of set-1 Stage A
- **Do:**
  - Rebuild set 1's Stage A rows exactly from seeds: 816 terminal rows plus the s3 sibling subset with `continuation` labels.
  - Fit A0 (cold, install true), then the robust recipe (squared comparison, prediction selection, warm start).
  - Grade both on Eval-S.
- **Prediction:** if the host computes like af6670b, A0 = **0.662** and robust = **0.821** on the 6-chain subset, both bit-identical.
- **Decision:**
  - If they match, set-1 numbers transfer to this host.
  - If they don't, record the deltas, treat set 1 as directional only, and re-baseline with T0.3's numbers.
- **Cost:** ~150 requests, 2 fits, ~30 min.

### T0.4 Client-side vs native planning
- **Do:** with the T0.3 A0 model, take 100 short and 30 long states at depths 1–3. Compare native `plan.action` and `plan.value` with the client-side maximum over predicted leaves.
- **Pass:** the same action (up to exact ties) and |Δvalue| < 1e-6 in every case.
- **If it fails:** use native planning everywhere, use `plan.value` for `u`, and use compact JSON for seq7 at depth 3, which has only a 2% margin.
- **Cost:** ~600 requests, 0 fits, ~15 min.

**Gate G0:**
- Recompute the budget from the measured throughput.
- If throughput is below 1 call/s, apply the cut list (§11) before starting Track B.

## 5. Track A: HP diagnostics (cheap; fixes the settings for Track B)

### A1 Representation probe
- **Question:** can the value model represent the contact heuristic accurately enough to plan with it at depth 3 on long chains?
  - Set-1 learned models stayed below the heuristic's 5th percentile.
  - A one-line compactness tie-break adds about 5 contacts.
  - Both suggest the model misses simple aggregates.
- **Data:** ~6k states from 120 random and 60 heuristic-guided folds (depth 1, ε = 0.3) on 60 chains of length 24–60. Synthetic target = the heuristic's own partial score.
- **Arms:**
  - **R0:** object graph with set-1 options and 4 BASIC conditions.
  - **R1:** R0 plus the 14 `features()` values as conditions.
  - **R2:** the `projected` representation (counts of objects and relations) plus the 14 conditions, fitted with `mlp_v2`.
  - R0 and R1 share one object-graph domain, with distinct record ids; R2 needs its own projected domain.
- **Measures:**
  - Sibling agreement: on 1,000 held-out states, the share whose best predicted child is among the heuristic's best.
  - Error in contact units.
  - **Depth-3 planning with the probe model** on Eval-Liu and on 20 Eval-L chains, compared with the heuristic's distribution.
- **Prediction:**
  - R0 falls below the heuristic's 5th percentile (<126 on Eval-Liu), with sibling agreement below 0.9.
  - R1 and R2 land inside the heuristic's 5–95% band (126–141), with agreement of at least 0.97.
- **Decision:** r\* is the cheapest representation inside the band with agreement of at least 0.95. Keep R0 if it qualifies, since raw structure is preferred when it suffices.
- **Cost:** ~3k requests, 3 fits, ~1.5 h.

### A2 Fit-settings probe
- **Question:** why did set 1's recipe fits on own-policy data keep their incoming checkpoint (epoch 0)? Candidate causes: incumbent regularization in warm fits, prediction-error selection, or neither.
- **Data:** T0.3's Stage A rows, plus 96 own-policy folds collected by T0.3's robust model (depth 1, ε = 0.2). This mirrors set 1's O_own setup.
- **Arms:** one fit each, all using the set-1 recipe rows:
  - warm with prediction selection (set 1's setting);
  - cold initialization (installed reference kept for the continuation targets);
  - warm with `selection_objective: adaptation_prediction`;
  - warm with `function_retention_weight` 1;
  - cold with 240 epochs.
- **Measures:** Eval-S best-move rate; `selected_epoch`; `optimization_applied`.
- **Decision:** f\* = the best setting. If the differences are within noise (under 0.03), prefer the simplest, cold.
- **Cost:** ~400 requests, 5 fits, ~45 min.

### A3 Target construction, oracle-graded (the key cheap experiment)
- **Question:** do the substrate's blended targets beat terminal labels and set 1's robust recipe on HP? Which λ and which search depth?
- **Data:**
  - 3 draws of 96 random folds on set 1's 12 training chains. Draw 1 is set 1's exact folds; draws 2 and 3 use new seeds.
  - The assessor is each draw's A0 model.
- **Arms in draw 1:**
  - A0 (λ = 0);
  - ROB (set 1's recipe);
  - BL(λ, d) for λ ∈ {0.5, 0.75, 1.0} and d ∈ {1, 2};
  - BLG (the best BL plus sibling rows);
  - BL-iter (a second round with the best BL model as the assessor).
- **Arms in draws 2–3:** A0, ROB, the best two BL arms, BLG and BL-iter.
- **Measures:**
  - Primary: Eval-S best-move rate at depth 1.
  - Also: regret, correlation with V\* and V_rand, greedy contacts, and depth-3 best-move rate for the top 3 arms.
- **Predictions:**
  - BL(1, 1) ≈ 0.762, reproducing set 1's A2.
  - BL(0.75, 2) ≥ 0.83, beating ROB's 0.81, with correlation to V\* of at least 0.40 (set 1 managed 0.24–0.34).
- **Decision:** fix λ\*, d\* and whether sibling rows are used. If no BL arm beats A0 by at least 0.03 averaged over draws, Track B runs A0 against ROB-style targets instead.
- **Cost:** ~2.5k requests, ~22 small fits, ~2–2.5 h.

### A3b Blended targets on own-policy data (a mini Track B with exact grading)
- **Design:** one own-policy generation (96 folds, depth 2, ε = 0.15) for A0 and BL(λ\*, d\*), collected by each arm's own model. Refit cold and grade on Eval-S.
- **Prediction:** BL keeps at least a +0.03 lead after the own-policy generation. Set 1's recipe lost its lead at this point.
- **Cost:** ~300 requests, 4 fits, ~30 min.

**Gate G1:** fix r\*, f\*, λ\*, d\* and the sibling choice. Write them into the Track B manifest before Track B starts.

## 6. Track B: the main HP experiment

### B0 Bridge *(optional; needs `m_lineage_T_rows.json` from the original machine)*
- **Do:** rebuild M3 cold on a fresh domain (9,209 rows, 1 fit). Score Eval-Liu at depth 3, and run a leaf-blend sweep on it.
- **Prediction:** 123 if the host matches af6670b.
- **Value:** links set 1's long-chain results to this set, and gives an immediate blend result.
- **Cost:** ~1.2k requests, 1 fit.

### B1 Replicated, length-matched lineages
- **Question:** with training lengths matched to the test chains and two own-policy generations:
  - do blended targets yield better depth-3 evaluators than terminal targets (R1)?
  - does the best learned evaluator beat the contact heuristic, alone (R2) or blended at the leaves (R3)?
- **Lineages:** ℓ = 1, 2, 3, each with its own 48 training chains (lengths 24/32/40/48/56/60, 8 of each), disjoint from Eval-L, the panel and the other lineages. Seeds are in §A.
- **Arms per lineage:**
  - **A0:** λ = 0.
  - **BL:** λ\* and d\*.
  - Both use r\* and f\*.
  - Optional third arm in lineage 1: BLG, or λ = 0.5.
- **Generations:**
  - **g0:** 144 random folds (3 per chain). A0 is fitted first. BL's g0 targets come from a reconsideration pass: `u` at depth d\* from the g0 A0 model.
  - **g1 and g2:** 144 own-policy folds each, collected by the arm's own previous candidate at depth 2 with ε = 0.15.
    - Run the search at every step, before the ε coin, so `u` is recorded for every state.
    - Collector contacts per fold are logged as the learning curve.
  - **Fits:** cold (or per f\*) on all rows so far, install false. g2 uses about 17.8k IDs (the per-fit limit is 20k).
  - **Targets:** existing rows keep their original targets and provenance. A "refresh all targets" variant is optional and not budgeted.
- **Evaluation:**
  - After g0 and g1: Eval-Liu at depth 3.
  - After g2: Eval-L (40 chains) and Eval-Liu at depth 3, plus the leaf-blend curve (B2).
- **Futility gate G2:** after lineage 1's g1, if BL trails A0 by more than 8 contacts on Eval-Liu (beyond the panel's ±5 noise), drop BL from lineages 2–3 and put the budget into A0 plus the leaf blend.
- **Predictions:**
  - (i) R1: BL beats A0 by at least 1 contact per chain at g2.
  - (ii) R2: the best pure learned arm lands inside the heuristic's band, but not clearly above its mean (Δ between −1 and +1 per chain).
  - (iii) R3: a leaf blend with λ_leaf around 0.25–0.5 beats the heuristic by at least 0.5 contact per chain.
- **Scheduling:**
  - Run lineages one after another, with both arms of a lineage in parallel (2 live domains).
  - While one arm's fit runs, the other arm collects or evaluates.
- **Cost:** about 10k requests and 6 fits per lineage, so ~30k / 18 in total (+~5k / 3 for the optional arm). Roughly 3–8 h of API time.

### B2 Leaf blend and depth curves (part of B1's final evaluation)
- **Do:**
  - For each lineage's best g2 model: λ_leaf ∈ {0.25, 0.5, 0.75} on Eval-Liu (development), then the chosen λ_leaf on Eval-L (test).
  - Also run that model at depths 1 and 2 on Eval-L, for C1.
- **Cost:** included in the B1 estimate, plus ~1k requests for the extra depths.

### B3 Promotion audit *(optional)*
- **Question:** would the documented paired canary adopt the better model?
- **Do:** in lineage 1:
  - install g1 by refitting with install true;
  - stage g2 as the candidate;
  - run a paired canary of 128 cases with depth-2 completions, from held-out starts not in Eval-L;
  - compare the canary's verdict with the Eval-L verdict.
- **Cost:** ~1.5k requests, 2 fits.

**Gate G3:**
- If R2 or R3 holds, run C2.
- In every case, run C1 and write up.

## 7. Track C: search efficiency and per-chain adaptation

### C1 Compute-matched comparison (mostly local)
- **Do:** plot contacts against leaf evaluations per decision for:
  - the heuristic at depths 1–5 (local);
  - the best learned model at depths 1–3 (from B2);
  - the leaf blend.
- **Claim form:** "at N leaf evaluations per decision, learned X vs heuristic Y."
- **Context:** the heuristic gains about 8 contacts going from depth 3 to depth 5, at 25 times the leaf evaluations.
- **Cost:** ~1k requests, already counted in B2.

### C2 Per-chain adaptation on seq5–7 *(conditional on G3)*
- **Why only these chains:** search-only reaches 28.1/31, 28.1/34 and 48.9/55 by 128 attempts; the other four chains are already at their references.
- **Arms, 64 attempts per chain each:**
  - search-only (computed locally);
  - the frozen best prior, with depth-2 attempts plus ε = 0.1 and depth-3 greedy checkpoints;
  - the prior plus per-chain blended-target refits every 16 attempts.
- **Measures:** best at 16, 32 and 64 attempts; attempts needed to reach 29, 29 and 50 contacts respectively.
- **Cost:** ~4k requests, 12 fits.

## 8. Track D: neighbor 1, 2D square-lattice HP

**Why 2D.** It uses the same sequences, code path and value route, but its search behaves very differently (local check, §C):
- **More headroom:** the depth-3 contact heuristic reaches 80.8 of the 120 best-known (67%, against 78% in 3D), and 128 attempts reach only 85% (91% in 3D).
- **Dead ends matter:**
  - In 3D the heuristic never traps itself on these chains at any depth.
  - In 2D it traps 8.9% of folds at depth 1 and 0.6% at depth 3.
  - Avoiding traps is a long-horizon property, which a learned value can capture and the heuristic cannot.

### D0 Local environment (session S0)
- 2D observation in the tip frame and a 2D oracle (exact up to length ≈16).
- Eval-S-2D, Eval-L-2D and the 2D panel (best-known 9/9/8/14/23/21/36).
- Baseline cards. `lattice_env.State` already supports `dim=2`; `analysis/fasthp_nd.py` covers the heuristic.

### D1 2D short chains, oracle-graded
- **Do:** run A3's arms (A0, ROB, best BL, BLG) on 12 training chains of length 14–16 with 96 random folds, 2 data draws. Grade on Eval-S-2D.
- **Prediction:** the learned edge over the 2D heuristic at depths 1–3 is larger than the 3D edge.
- **Cost:** ~700 requests, ~10 small fits.

**Gate GD:** run D2 if D1's best arm beats the 2D heuristic at depth 1 or depth 3 on Eval-S-2D.

### D2 2D long-chain lineage *(conditional)*
- **Do:** B1's design with 1 lineage (2 if budget allows) and 2 arms, on 2D chains of length 24–64. Evaluate on the 2D panel and Eval-L-2D at depth 3, against the 2D heuristic's distribution.
- **Prediction:** the learned values beat the 2D heuristic (80.8 ± 4.3 on the panel) by at least 5 contacts. This is the case where learning should matter if it matters anywhere on HP.
- **Cost:** ~10k requests, 6 fits.

## 9. Track E: neighbor 2, a small solved two-player game

**Why.** It is the same value-plus-planning route with **opponent (MIN) nodes**, which the docs support, and it is the setting of Rei's own chess study. A small board allows exact solver grading, which chess cannot.

### E0 Local environment (session S0)
- **Game:** gravity four-in-a-row on a 5×4 board by default; use 6×5 if a bitboard solver labels it within about 1 h, or Domineering 5×5 as a fallback.
- **Components:**
  - an exact solver for position values (win 1 / draw ½ / loss 0, from the first player's view);
  - a heuristic evaluator (weighted count of open lines);
  - an object-graph encoding: cells as objects; links between like pieces along the 4 line directions; side to move and move fraction as conditions;
  - an opening suite of all 2-ply openings, both colors.
- **Planning convention:** nodes are maximizing when the first player is to move. Continuations use `maximizing` accordingly.
- **Discounting:** z̄ uses γ = 0.99, as in Rei's chess setup.

### E1 Value learning, solver-graded
- **Do:**
  - 3 draws of 600 random games.
  - Arms: A0, BL(λ\*, depth 2 with MIN nodes), BLG.
  - Grade on about 800 informative held-out positions (from random games and optimal lines): best-move rate against the solver, win/draw/loss value accuracy, regret.
  - Compare with the heuristic and with neutral-leaf search at equal depth.
- **Cost:** ~1k requests, ~10 small fits.

**Gate GE:** run E2 if the best learned arm's best-move rate at depth 2 is at least the heuristic's at depth 2.

### E2 Play strength *(conditional)*
- **Do:** run the best E1 model with **native** planning (the adversarial planner is under test) at depths 2 and 4 against:
  - heuristic search at the same depth;
  - neutral-leaf search;
  - the exact solver.
- Use paired openings: all 2-ply openings × both colors = 50 games per pairing.
- **Measures:** points rate, win/draw/loss, paired by opening; draws held against the solver.
- **Prediction:** the learned model beats heuristic search at the same depth by at least 10 points. In games the heuristic is weaker relative to search than in HP.
- **Cost:** ~2.5k requests.

### E3 Own-play generations *(optional)*
- **Do:** 2 generations of own-play data with blended targets; repeat the E1 and E2 measures.
- **Cost:** ~3k requests, 4 fits.

## 10. Track F: optional neighbors (not budgeted)
- **F1 RNA base-pair maximization:** exact O(n³) dynamic-programming oracle at any length. Tests length transfer and learned-vs-heuristic at long horizons with exact grading, which HP cannot provide above length 11.
- **F2 Sequential route, positive control:**
  - Run Sequential Discovery with its default gates (`minimum_validation_skill` 0.25) on a small task whose successors the application cannot list, e.g. a stochastic key-and-door corridor. Compare with offline tabular Q-learning.
  - This is where the Q-MLP route belongs according to the docs. Feedback writes are not idempotent, so use strict reconciliation.
- **F3 Learned-transition planning, and Machina (continuous control):** only if you want those substrate areas covered. They are far from HP.

## 11. Schedule, gates and cut list
| Session | Content | API time |
|---|---|---|
| S0 (local, about 1–2 days of development) | Shared components, test suites, baseline cards, D0, E0, mock-server dry runs | 0 |
| S1 | T0.2–T0.4 → gate G0; A1, A2, A3, A3b → gate G1; D1 → gate GD; E1 → gate GE | ~5–7 h |
| S2 | B1 lineages 1 and 2 (gate G2 after lineage 1's g1); B0 if the file is available | ~5–8 h |
| S3 | B1 lineage 3, B2, C1 → gate G3; then the conditional runs: D2, E2, C2, B3 | ~5–8 h |
| Write-up | Results tables, decision-rule outcomes, updated cheat sheet | 0 |

**Pacing:** fits are serialized per owner (about 57 small fits in S1, totaling roughly 1.5–2.5 h of fit time). Interleave fits with predict-heavy work. Set 1's time box of about 8 h per session still applies.

**Cut list, if time or throughput is short (drop from the bottom first):**

| Priority | Item |
|---|---|
| 1 | T0.2–T0.4, A1, A3 |
| 2 | B1 lineages 1–2, both arms |
| 3 | D1 |
| 4 | E1 |
| 5 | B1 lineage 3 (needed for R1 and R2 as written; with 2 lineages, report them as exploratory) |
| 6 | A2, A3b, C1 |
| 7 | Conditional runs: D2, E2, C2 |
| 8 | Optional runs: B0, B3, E3, the third B1 arm, Track F |

**Other ways to save without dropping items:**
- Cut Eval-L to 30 chains.
- Evaluate g0 and g1 on Eval-Liu only (already the plan).
- Use depth-2 collection throughout (already the plan).

## 12. Risks and mitigations
| Risk | Mitigation |
|---|---|
| The host differs from af6670b (T0.3 fails) | Re-baseline on T0.3's numbers and treat set 1 as directional |
| Production rate limits or load concerns | ≤2 requests in flight with backoff; drop to 1 if 429s appear; confirm acceptable load with Rei Labs before S2 |
| Owner-wide dataset-worker capacity | ≤4 live domains; delete after each experiment; run lineages sequentially |
| The 20,000-ID per-fit limit | 3 folds per chain; subsample sibling rows to every 4th state; drop g0 rows at g2 if needed |
| Client-side planning disagrees with native (T0.4) | Native planning everywhere; compact JSON for seq7 |
| Fits are not deterministic on the host | Add one replicate fit per key model in B1 and A3 |
| Pipeline bugs like set 1's (grading after promotion, identical generations, a process check matching itself) | Guards from §2.4, mock-server dry runs, unit tests on the shared components |
| Throughput below 1 call/s | Apply the cut list at gate G0 |
| Overfitting λ_leaf or other choices to the test set | Development/test split (Eval-Liu / Eval-L); confirmatory tests limited to R1–R3 |

## 13. Deliverables
- **Per experiment:**
  - a `manifest.json`: build, seeds, options, data hashes, model shas;
  - a `results.json` with per-decision or per-chain outcomes;
  - `log.txt`;
  - a one-line summary in a running results table.
- **Final report:**
  - a claims table (claim, evidence, decision rule, outcome);
  - figures: contacts vs depth and leaf evaluations, lineage learning curves, the leaf-blend curve, 2D vs 3D edges, and game points;
  - an updated API cheat sheet.
- **Code:** the shared components and stage scripts, with the mock server and unit tests, as in set 1.

---

## Appendix A: seed registry
New seeds are in the 80000s, so nothing collides with set 1's 5–77001 range.

| Item | Generator |
|---|---|
| Eval-S | set 1's `sequences(61107, [10,11]*9)[12:]` (6 chains), plus `sequences(80001, [10,11]*9)` with set-1 chains removed (18). States from 4 random folds `Random(80002)` plus the optimal path |
| Eval-L | `sequences(80101, [36,40,44,48,52,56,60,64]*5)`, removing panel sequences and their reversals |
| B1 training chains, lineage ℓ | `sequences(80200+ℓ, [24,32,40,48,56,60]*8)`, excluding Eval-L, the panel and other lineages |
| B1 random folds / collectors | `Random(80300+ℓ)` / `Random(80400+10ℓ+g)` |
| A1 probe data | `sequences(80501, [24..60])` with 60 chains, folds `Random(80502)` |
| A3 data draws | Draw 1: set 1's folds (`Random(61107)` on set 1's training chains). Draws 2–3: `Random(80601)`, `Random(80602)` |
| Validation episodes | 20% of episodes chosen by `Random(80700+fit_index)`, stratified by generation |
| 2D | Short `sequences(81001, [14,15,16]*8)` (12 training / 12 evaluation); Eval-L-2D `sequences(81101, …)`; panel = the 7 Liu sequences |
| Game | Data `Random(82001+draw)`; openings enumerated |

## Appendix B: fit settings
- **Object-graph base (set-1 options):**
  - model: `model_type object_value`, `dimensions 32`, `object_hidden_dimensions 32`;
  - training: `epochs 120`, `learning_rate 0.003`, `loss_function squared`, `supervision_scope all`, `trainable_layers all`;
  - representation: `dense_inputs false`, `pair_geometry true`, `peak_pooling true`, `quadratic_pairs true`, `preserve_binary_inputs true`;
  - validation and selection: `validation_mode episode_hash` with explicit `validation_episode_ids`, `selection_objective prediction`, `comparison_weight 0`.
- **Fit modes:**
  - Track B default: `warm_start false` and `install false` (cold, candidate only), unless A2 picks otherwise.
  - Continuation-row arms (T0.3, A2, A3's ROB): the first fit is cold with `install true`, to act as the target reference.
- **Projected (R2):** `model_type mlp_v2`, `dimensions 64`, `dense_inputs true`, `epochs 120`, otherwise the same.
- **Comparison arms (BLG, ROB):** `comparison_weight 1`, `comparison_loss squared`, `selection_objective prediction`, with groups present in both partitions.

## Appendix C: local baseline numbers (offline; scripts in `analysis/`, outputs in `analysis/out/`)
**3D Liu panel totals** (best-known 171, which is uncertified; the paper's own best is 158). Greedy folds, random ties:

| Evaluator | Total |
|---|---|
| Contact heuristic, depth 1 | 80.0 ± 6.9 |
| Contact heuristic, depth 2 | 117.8 ± 6.0 |
| Contact heuristic, depth 3 | 133.6 ± 4.6 (5–95%: 126–141) |
| Contact heuristic, depth 4 | 139.4 ± 4.2 |
| Contact heuristic, depth 5 | 141.5 ± 4.0 |
| Depth 3 with H-core compactness tie-break | 138.2 ± 3.5 |
| Depth 3 with neutral leaves (no learned values) | 31.9 ± 7.4 |
| Best of k attempts (depth 3, ε 0.1) | k = 1: 130.7; 8: 142.7; 32: 150.0; 128: 155.9 |

Trapped folds: 0 at every depth.

**2D, same sequences** (best-known 120):

| Evaluator | Total |
|---|---|
| Contact heuristic, depth 1 | 45.9 ± 9.6 (8.9% of folds trapped) |
| Contact heuristic, depth 2 | 69.6 ± 7.3 (2.6% trapped) |
| Contact heuristic, depth 3 | 80.8 ± 4.3 (0.6% trapped) |
| Contact heuristic, depth 4 | 89.0 ± 3.3 |
| Contact heuristic, depth 5 | 90.8 ± 3.3 |
| Depth 3 with neutral leaves | 26.8 |
| Best of k attempts (depth 3, ε 0.1) | k = 1: 79.8; 8: 90.8; 32: 98.3; 128: 102.3 |

**3D short chains** (Stage A/P oracle set, 151 decisions), contact heuristic with fair ties:
- Best-move rate 0.720 / 0.862 / 0.922 at depths 1/2/3.
- Greedy contacts 11.9 / 16.8 / 17.5 of 20.
- Set 1 reported 0.662 / 0.808 / 0.901 because ties went to the first direction.

**Payload:** the largest depth-3 planning body for seq7 is 1,152 KiB with default JSON and 1,004 KiB compact. A `predict` call with 128 L60 inputs is about 732 KiB.

**Validation:** every script reproduces set 1's numbers exactly before reporting new ones: Liu panel 68/113/134, Stage D control streams, Stage P first-tie rates.

## Appendix D: files
- `/workspace/EXPERIMENT_PLAN.md`: this plan.
- `/workspace/analysis/`: offline baselines.
  - `fasthp.py` and `fasthp_nd.py`: fast heuristic planners.
  - `check_*.py` and `out/*.out`.
  - `vendor/`: unmodified `hp.py`, `lattice_env.py` and `grading.py` from the package.
  - `data/liu_panel.json`.
  - Run with `python3 -I`; nothing uses the network.
