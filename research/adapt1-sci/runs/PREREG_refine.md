# Exp A'' pre-registration: Machina contextual refinement on Park-Ramirez with measured batch variation

Problem: the published Park-Ramirez fed-batch model (Exp A'), with two quantities that vary per batch and are measured
before it starts: feed substrate concentration sf in [12, 28] (published 20) and initial glucose x4(0) in [3, 7]
(published 5). Local check: the nominal-optimal feed profile loses 30 % at sf = 12, 19 % at 15, 6 % at 18 against
each batch's own optimum, so context-dependent corrections have something to recover.
Pipeline: base = Machina's best acquired profile at the published conditions (from Exp A'; J = 31.99) ->
contextual_execution refinement (structural stage skipped). Context = [(sf-20)/8, (x4(0)-5)/2]; quality = J/32.68.
Arms: shared (contextual false, the documented control), linear, quadratic, all radius 0.2 and normalize_context;
plus linear at the docs' default radius 0.08. 30 updates x 8 candidates x 32 training contexts = 7680 executions.
Evaluation: 32 held-out contexts, frozen evaluate, at updates 0, 5, 10, 20, 30.
Comparators (same 7680 executions): a linear context->offset policy trained by CMA-ES and by (1+1)-ES on minibatches
of 16 training contexts (3 seeds each). Reference: base alone; ceiling = per-context optimum (2400 evaluations each).
Metric: held-out mean J; fraction of the base->ceiling gap recovered.
