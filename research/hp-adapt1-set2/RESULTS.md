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
| A3 | queued | | | |
| A2 | queued | | | |

## Gate decisions

### G0 (after T0.2–T0.4)
- **Throughput:** 2.19 calls/s, above the 1 call/s cut threshold, so nothing is cut. At this rate the plan's core of about 40k requests is about 5 h of API time, the low end of the plan's 5–12 h.
- **Fit latency:** 3.4 s for 50 rows; 11 s for 816 rows (A0); 22 s for 2.2k rows with continuation targets (ROB). The plan's fit-time budget is generous.
- **Calibration (T0.3):** ROB reproduces set 1 within noise (0.808 vs 0.821; SE about 0.03 on 151 decisions). A0 is 0.07 higher than set 1's 0.662, about 2.3 SE. Following the plan, set-1 numbers are treated as **directional only**, and T0.3's numbers are the baseline.
- **Planning (T0.4):** client-side MAX search over `value/predict` leaves is bit-identical to native planning, so evaluations use the batched client-side planner (C-plan).

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
