# HP × Adapt-1, set 2: running results

The plan is in [`plan/EXPERIMENT_PLAN.md`](plan/EXPERIMENT_PLAN.md). Each run has a `manifest.json`, `results.json`, `log.txt` and a request journal under `runs/<experiment>/<timestamp>/`.

**Host:** Fast-UAT (`rei-neuroadapt-fast-uat.reilabs.org`), build `af6670b8a9c5ce4251caa02940118110adb839ea`, schema 5. This is set 1's build. The build check was stable at the start and end of every run.

## Running results table

| ID | Status | One-line result | Requests | Run |
|---|---|---|---|---|
| T0.1 | done | Harness and suites below; the planner reproduces Liu 68/113/134 exactly; Eval-S's set-1 group equals set 1's grading set exactly | 0 | `tests/`, `suites/` |
| T0.2 | done | 2.19 predict calls/s (125 L50 inputs, 581 KiB, 2 in flight, p50 0.84 s, 0 errors); tiny fit 3.4 s; 413 between 976 and 1029 KiB ("exceeds 1 MiB"); depth-1 native plan value equals the predicted leaf max | 221 | `runs/T0.2/20261106T194844Z` |
| T0.3 | done (directional) | Eval-S depth-1 best-move rate, set-1 6-chain subset / all 24: **A0 0.735 / 0.777** (set 1: 0.662), **ROB 0.808 / 0.818** (set 1: 0.821); heuristic 0.720 / 0.695. A replicate A0 fit had a different sha but identical predictions (max \|Δ\| = 0 on 512 inputs) | 530 | `runs/T0.3/20261106T200218Z` |
| T0.4 | **PASS** | 390/390 cases (100 short + 30 long states × depths 1–3): same action, \|Δvalue\| = 0.0 exactly. Client-side planning is used for evaluation | 447 | `runs/T0.4/` |
| A1 | done, **r\* = R1** | R0 (object graph, 4 conditions): its predicted best child is among the heuristic's best on 0.995 of 1,000 held-out states (0.980 of informative ones), with MAE 0.27 contacts, but depth-3 planning falls **below the heuristic band**: Liu 124 (p5 125); Eval-L(20) 417 against a heuristic 444.1 [429–458]. R1 (+ the 14 `features()` as conditions): agreement 0.999, **in band**: Liu 125, Eval-L(20) 440. R2 (projected) was **refused** at its first upload: `422 dataset feature vocabulary exceeds the configured surface limits`. Fits took about 250 s each for 6.7k long-chain rows | ≈1.6k | `runs/A1/20261106T201620Z` |
| A3 | done: **λ\* = 1, d\* = 2, no siblings** | Mean Eval-S best-move rate over 3 draws, depths 1 / 2 / 3: heuristic 0.695 / 0.854 / 0.938; A0 0.772 / 0.888 / 0.935; ROB 0.786 / 0.891 / 0.933; BL(0.5,2) 0.782; **BL(1,2) 0.798** / 0.864 / 0.943; BL-iter 0.809 / 0.856 / 0.929; **BLG 0.811 / 0.901 / 0.952**. Draw 1 also ran BL(0.5,1) 0.765, BL(0.75,1) 0.769, BL(1,1) 0.774 (plan predicted 0.762) and BL(0.75,2) 0.767. Lookahead targets raise corr(V\*) (A0 0.38 → BL(1,2) 0.44 → BL-iter 0.57, means over draws) and lower corr(V_rand) | 5.1k | `runs/A3/20261106T204511Z` (draw 1), `runs/A3/20261106T205829Z` (draws 2–3) |
| A2 | done: **f\* = cold** | ROB rebuilt on a fresh domain scored 0.818 exactly, as in T0.3: fits are reproducible across domains. On Stage A rows plus 96 own-policy folds (ROB collector, mean 1.34 contacts), depth-1 / 2 / 3 rates were: warm_pred (set 1's setting) 0.816 / 0.900 / 0.961, selected epoch 117 with `optimization_applied: true`, so **set 1's epoch-0 behaviour did not reproduce**; warm_retain 0.830 / 0.891 / 0.947 (epoch 89); cold 0.825 / 0.898 / 0.944 (epoch 106); cold_240 0.825 / 0.913 / 0.951 (epoch 211). warm_adapt was **refused**: `adaptation_prediction requires a dense linear/mlp_v2 warm start`. The spread is 0.015, under 0.03, so the simplest setting (cold) is chosen | ≈1.4k | `runs/A2/20261106T212002Z` |

## Gate decisions

### G0 (after T0.2–T0.4)
- **Throughput:** 2.19 calls/s, above the 1 call/s cut threshold, so nothing is cut. At this rate the plan's core of about 40k requests is about 5 h of API time, the low end of the plan's 5–12 h.
- **Fit latency:** 3.4 s for 50 rows; 11 s for 816 rows (A0); 22 s for 2.2k rows with continuation targets (ROB). The plan's fit-time budget is generous.
- **Calibration (T0.3):** ROB reproduces set 1 within noise (0.808 vs 0.821; SE about 0.03 on 151 decisions). A0 is 0.07 higher than set 1's 0.662, about 2.3 SE. Following the plan, set-1 numbers are treated as **directional only**, and T0.3's numbers are the baseline.
- **Planning (T0.4):** client-side MAX search over `value/predict` leaves is bit-identical to native planning, so evaluations use the batched client-side planner (C-plan).

### G1 (Track B manifest, `manifests/track_b.json`)
| Setting | Value | From |
|---|---|---|
| r\* | R1: object graph + the 14 `features()` as conditions | A1 |
| f\* | cold (`warm_start false`), set-1 options, 120 epochs | A2 |
| λ\*, d\* | 1.0, 2 | A3 |
| Sibling rows | no | A3 |
| Fallback (A0 against ROB) | not triggered | A3 |

Substrate rules found on af6670b during Track A:
- **A fit that selects `continuation` rows cannot be installed.** It returns 422 `selected observation continuation requires install=false or application-outcome promotion`. Continuation fits therefore stay candidates, with the installed model as their reference.
- **`selection_objective: adaptation_prediction` needs a dense `linear` or `mlp_v2` warm start.** It is refused for `object_value`.
- **The `projected` representation refuses these object graphs:** `dataset feature vocabulary exceeds the configured surface limits`.

### A3 decision (input to G1)
- **λ\*, d\* = 1, 2.** BL(1,2) is the best arm in the BL(λ, d) grid, both on the mean and in every draw. In each draw its depth-1 lead over A0 was +0.041, +0.031 and +0.005, a mean of +0.026.
- **The fallback (A0 against ROB-style targets) is not triggered.** The plan's rule is "no BL arm beats A0 by at least 0.03". BLG (+0.039) and BL-iter (+0.037) clear it, while the grid arm alone (+0.026) does not. The first summary code checked only the grid. It now follows the plan's wording, and both readings are recorded here.
- **No sibling rows in Track B.** BLG − BL(1,2) = +0.013 at depth 1, below the pre-registered 0.03.
- **Exploratory:** BLG is the best arm at every depth, and the only one clearly above the heuristic at depth 3 (0.952 against 0.938). The rule excludes it, and that decision stands. Its depth-3 edge is reported as an exploratory lead.
- **Exploratory:** BL-iter, the second round of lookahead targets, improves the depth-1 rate and corr(V\*) in every draw but is worse than BL(1,2) at depths 2 and 3 in every draw. A better one-step value does not by itself make a better leaf evaluator under deeper search. Track B's later generations iterate in the same way, so this is the pattern to watch there.
- **Noise:** refitting A0 with only a different validation split moved its rate by 0.03 (T0.3: 0.777; A3 draw 1: 0.748, on the same rows). Single-draw differences under about 0.03 are within noise.

### S1 tail: A3b, D1, E1
- **A3b: the prediction fails.** On Eval-S at depth 1, BL(1,2) led A0 by **+0.036** (0.813 against 0.777) before own-policy data and trailed by **−0.012** after one own-policy generation (0.813 against 0.825). A0 gains about 0.05 from its own data and BL gains nothing. BL also collected weaker folds (1.42 against 1.72 contacts), which fits its lower depth-2 rate (0.825 against 0.879). This is set 1's failure mode, replicated with the documented blended targets.
- **D1: GD → run D2.** Eval-S-2D, mean of 2 draws, depths 1 / 2 / 3: heuristic 0.639 / 0.796 / 0.847; A0 0.653 / 0.767 / 0.827; ROB 0.755 / 0.793 / **0.855**; BL 0.768 / **0.836** / 0.843; BLG **0.796** / 0.807 / 0.828. The depth-1 edge over the heuristic is larger in 2D (+0.157) than in 3D (+0.116). At depth 3 it is near zero in both. A0's greedy depth-1 folds trap often (6–7 of 49 contacts, against the heuristic's 22).
- **E1 is not viable as specified.** Two fits were rejected (`candidate_failed_validation`):
  - The first rejection exposed a bug in `hpx/game.py`: positions rebuilt from column strings lost their winner, so every target was 0.5. It is fixed and tested. The frozen game suite is unaffected.
  - The second rejection is genuine: `validation_skill −0.036`. On this board, random-game outcomes are close to noise. The exact Bayes predictor V_rand reaches only **0.111** skill on the same validation games (0.005 in the first 6 moves, 0.315 after 12 pieces); the line heuristic reaches 0.045, and the solver value −0.43.
  - **Remedy (deviation 11):** ε-greedy (ε 0.3) depth-1 line-heuristic games, with γ 0.99 kept. Their Bayes ceiling is **0.209**. They run as `E1g` after B1 lineage 1.

### X1: native search extensions (exploratory, not in the plan)
Setup: 100 informative Eval-S decisions, exact grading. The evaluator is one learned model (A0 on A3 draw-1 data). Native planning runs on supplied depth-4 graphs.

| Native search | Best-move rate | Evaluations per decision | Extension work |
|---|---|---|---|
| Plain depth 3 (parity with client-side search: 100%) | 0.970 | 134 | 0 |
| Plain depth 4 (parity 100%) | 0.980 | 574 | 0 |
| Depth 3 + `max_extensions` 2, any mode | 0.980 | 1013 | 138 |
| Depth 3 + `max_extensions` 4, any mode | 0.980 | 1892 | 276 |
| Depth 3 + `max_extensions` 8, any mode | 0.980 | 3650 | 553 |

Client-side references on the same decisions: learned model at depth 3 / 4, 0.970 / 0.980; heuristic, 0.958 / 0.992.

- **The three modes (`all`, `competitive`, `equilibrium`) are identical**: same rates, evaluations and extension work. Single-player HP graphs have only maximizing nodes, so the opponent-oriented modes evidently have nothing to tell apart.
- **Extensions reach plain depth 4's accuracy at 1.8–6.4× its evaluations.** On these graphs they are less efficient than one more full search level.
- **Limits of the test:**
  - It is near the ceiling: depth 4 already gets 98 of 100.
  - Extensions can only expand into supplied nodes, and graphs deeper than 4 exceed 1 MiB in 3D.
- **A harder variant, X1-2D, is queued after D2.** It runs on Eval-S-2D, where the depth-3 rate is about 0.85 and branching ≤ 3 lets a depth-5 graph fit under the body limit.

## Deviations from the plan, and why

1. **Host.** The plan's default is production. This environment's key is rejected there (`401 Invalid bearer token`) and accepted on Fast-UAT, which the plan names as the alternative. Fast-UAT runs `af6670b`, the build set 1 and the docs' behavior checks used. Production runs `f5e1ab8`.
2. **T0.3 cannot be bit-identical.** Set 1's `stage_a.py` is not in the package, so the following are reconstructions:
   - Stage A rows: 96 random folds by `Random(61107)`, chain-major over set 1's 12 training chains, giving the expected 816 rows (a round-robin order also gives 816);
   - terminal rows sent as `target = U`;
   - the "s3 sibling subset": every third recorded state, one `continuation` row per child, grouped by parent;
   - validation episodes: 20% chosen by `Random(80700)`.
3. **Eval-S size.** It has 873 states but 412 informative decisions, against the plan's estimate of about 600. One decision is 0.0024 of a rate, and the SE of a rate is about 0.023.
4. **Fit identity guard.** Model shas differ between functionally identical fits, so "candidate sha differs" does not detect identical generations. Track B adds a functional check: predictions on a fixed probe set must change between generations.
5. **Candidate fits** finish with `training.status: "candidate"`, and installed fits with `"trained"`. Domain create returns 201. Both are handled in `hpx/client.py`.
6. **Seeds the plan does not name** (all in the 80000s, none colliding):
   - T0.2 scratch data: 80900;
   - T0.4 state sample: 80903;
   - A1 held-out chains and folds: 80503 and 80504;
   - A2 own-policy folds: `80800*1000 + i`;
   - baseline tie-breaks: string seeds `'<suite>-<kind><depth>-<chain>-<seed>'`.
7. **A3 grading.** Every arm is graded at depths 1–3 while it holds the candidate slot, rather than only the top three arms at depth 3. This avoids refits and costs about 2.5× the plan's A3 request estimate. A3 uses one domain per draw, with A0 installed as the assessor.
8. **A2 reference model.** The robust model is installed, making it the warm-start source and the continuation reference. This follows the plan's wording that the cold arm keeps "the installed reference for the continuation targets".
9. **A1 data.** Chain lengths cycle through 24..60. Held-out states come from 20 further chains, because the plan names no held-out set.
12. **B1 fit size.** af6670b stops training jobs after about 300 s of server time. Two failure labels appeared: `training_failed` and `interrupted`/`result_not_recorded`. Server time is not linear in row-epochs:

    | Rows × epochs | Server time |
    |---|---|
    | 5,952 × 120 | 208–217 s |
    | 11,904 × 60 | 224–239 s |
    | 11,904 × 120 | failed at 302–307 s |
    | 17,856 × 40 | cut off at about 320 s |

    B1 therefore (a) scales epochs to `min(120, 714,240 // rows)` and (b) caps rows per fit at 12,000 by dropping the oldest generation first: at g2 the g0 random-fold rows, the plan's own §12 mitigation. g2 therefore fits on the g1 and g2 own-policy rows at 60 epochs. Earlier lineage-1 attempts that hit the cap are kept under `runs/B1/` as records.
13. **G2 in lineage 1 does not fire under the final protocol.** The final protocol is run 4: 90 epochs at g0, 45 at g1–g2, at most 12k rows per fit, and fit retries. Its Eval-Liu depth-3 totals were A0 130 → 126 and BL 122 → **132** (g0 → g1). BL leads by 6, so **both arms continue in lineages 2–3**, and R1 remains confirmatory. Run 2 used the superseded 120/60-epoch protocol and was later cut off by the fit-time limit. There, G2 had fired (A0 133 → 120, BL 126 → 105). It is kept as a record only. The swing between protocols shows that single-lineage, 7-chain panel scores are noisy (±5 or more). This is why the confirmatory tests use Eval-L's 40 chains across 3 lineages.
14. **Failures under owner-shared load.** The training failures turned out not to be a fixed 300 s cap: a retried g0 fit succeeded at 325 s of server time after the first attempt failed at 305 s. During run 4 another workload on the same account (`x-fu-*` domains, not touched) was creating domains every few minutes, and fit throughput about halved. The client's retries absorb these failures.
11. **E1 data.** The plan specifies 600 *random* games per draw. Their outcomes are close to unlearnable on 5×4: the exact Bayes ceiling is a skill of 0.111, and af6670b rejects the A0 fit. E1g instead uses 600 ε-greedy line-heuristic games per draw (ε 0.3, depth 1, string seeds `e1g-<82001+draw>-<i>`), with the same arms, γ, suite and gate.
10. **A1 R2 not fitted.** The projected representation rejects these object graphs (`dataset feature vocabulary exceeds the configured surface limits`). The decision rule picks the cheapest representation that qualifies. R1 qualifies and shares the object-graph domain, so it is cheaper than R2, and r\* is unaffected. Track B uses the object graph with the 14 features as conditions.

## Baseline cards (frozen, `suites/`)

| Suite | Heuristic depth 1 / 2 / 3 / 4 / 5 (total, 100 seeds) | Depth 3 p5–p95 | H-core compactness, depth 3 | Neutral leaves, depth 3 | Best of k=1/8/32/128 |
|---|---|---|---|---|---|
| Eval-Liu (7) | 80.2 / 118.4 / **134.3 ± 5.1** / 140.2 / 141.4 | 125–142 | 137.4 | 30.8 | 131.8 / 142.7 / 150.9 / 155.0 |
| Eval-L (40) | 578.3 / 878.7 / **975.2 ± 12.8** / 1029.2 / 1048.5 | 956–994 | 1010.8 | 189.0 | 971.7 / 1039.4 / 1083.7 / 1125.4 |

| 2D panel (7) | 47.0 / 70.1 / **81.2 ± 4.4** / 89.3 / 91.0 (trapped 7.3% / 2.1% / 0.4% at depths 1–3) | 75–88 | 81.1 | 27.6 | 83.1 / 91.2 / 97.9 / 102.7 |
| Eval-L-2D (40) | 287.4 / 475.1 / **538.9 ± 16.2** / 579.4 / 605.3 (trapped 15.8% / 4.5% / 1.9%) | 508–562 | 530.2 | 167.9 | 531.9 / 601.1 / 639.4 / 669.5 |

These agree with the plan's Appendix C to within tie-break seed noise: Liu 133.6 ± 4.6 (126–141), and 80.8 ± 4.3 on the 2D panel. The 2D panel's best-known total is 120.

Other frozen suites:
- **Eval-S-2D** (12 chains, 678 states, 341 informative): the heuristic's best-move rate is 0.639 / 0.796 / 0.847 at depths 1–3.
- **game_e1** (800 informative positions on the 5×4 board): the line heuristic's best-move rate is 0.795 at depth 1 and 0.808 at depth 2, with W/D/L accuracy 0.47. Neutral-leaf search scores 0.494 and 0.616.
