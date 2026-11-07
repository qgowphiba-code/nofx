# Round 3 pre-registration: tasks where general learners need a lot of data

Criterion (user): a task is only interesting if the comparators that solve it need much more data.
Gate: a contextual task is run only if the nominal profile loses >= 3 % on average against each batch's own optimum.
- batch3 (batch reactor, catalyst activity + initial concentration): 1.0 % -> rejected, not run
- catmix2 (catalyst mixing, rate ratio + second-reaction factor): 2.0 % -> rejected, not run
- pr5 (Park-Ramirez, 4 relevant measured variables + 1 irrelevant): 35 % -> run
Runs:
1. pr5: Machina refinement linear r0.2, quadratic r0.2, shared r0.2 (control); 30 updates (7,680 executions).
   Comparators: linear context->offset policy (120 parameters) by CMA-ES and NES at 1x, 2x, 4x, 8x the executions.
2. pr2 drift: Machina linear r0.2, 20 updates, then the feed pump delivers 0.7x the command (unmeasured), 20 more
   updates. Held-out evaluated right after the change and 5/10/20 updates later. Comparator: the CMA-ES policy continued
   after the change with the same executions, and restarted from scratch.
3. Data multiplier on pr2 (Exp A''): CMA-ES and NES policies at 1x-8x executions (minibatch 32), 2 seeds each.
4. Penicillin fed-batch (Cuthrell & Biegler) as a second published fed-batch model: only if our implementation
   reproduces the published optimum, then the same gate and protocol.
Metric throughout: held-out mean J on 32 unseen batches; gap recovered between base alone and per-batch optimum; and
executions needed to reach Machina's held-out level.
