# Exp A pre-registration: batch cooling crystallization x Machina (written before any counted run)

Question: on a task shaped like Machina's documented use (a repeatable equipment routine: one control, a
fixed-length sequence, an outcome measured after the batch), does acquisition find good cooling profiles as
fast as general black-box optimizers given the same number of batches? Then: does contextual refinement
improve an acquired profile on unseen batch conditions?

Compound selection rule (from the local headroom check, before any API run): use the compounds whose cubic
textbook profile is >= 0.08 below the true-parameter ceiling at the nominal context -> compounds 1, 4, 9.
Compounds 5, 6, 11, 12, 13 were rejected as forgiving (cubic within 0.01 of the ceiling).

Acquisition, nominal context (seed 2 %, 100 um, T0 50 C), budget 400 batches per run:
- Machina `profile` (docs optional profile + measured per-interval fines signal): streams r1, r2
- Machina `default`: stream r1
- local, same budget, 10 seeds each: random search, (1+1)-ES (1/5 rule), CMA-ES
- diagnostics only (hand-built): linear and cubic cooling
- ceiling (labelled): best of 4 CMA-ES restarts, 4000 evaluations, true kinetics
Metric: best quality after n batches, n in {16, 32, 64, 128, 256, 400}; fraction of the linear->ceiling gap closed.
Controls: none of the local methods is a control; they are other search methods on the same budget.

Refinement (run only if the context check shows the nominal optimum loses >= 0.02 on other contexts):
contextual_execution around Machina's best acquired profile; context = measured [log seed load, seed size, T0];
compare on held-out contexts: base alone (the documented comparison), refined, a linear context->offset policy
trained by CMA-ES with the same number of batch executions, and the per-context ceiling.
