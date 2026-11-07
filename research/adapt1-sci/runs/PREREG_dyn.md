# Exp A' pre-registration: published open-loop control benchmarks x Machina acquisition

Problems (formulations from the literature; each verified by reproducing its published optimum with a strong optimizer
on the same 20-stage piecewise-constant discretization before any API run):
- batch: A->B->C batch reactor temperature profile (Ray 1981; Luus), published optimum 0.6108; 20-stage reference 0.6105
- catmix: catalyst mixing (Gunn & Thomas 1965; COPS #14), analytic optimum 0.048056; 20-stage reference 0.04793
- parkram: Park-Ramirez fed-batch secreted protein (Park & Ramirez 1988; Luus; Banga), published optimum about 32.6-32.7
Machina: default r1, r2 and docs profile (without progress-guided mutation: no measured per-step outcome) r1, r2; 600 attempts.
Local, same budget, 10 seeds: random search, (1+1)-ES, CMA-ES. Metric: best J after n evaluations, and the fraction of the
gap between random-sampling mean and the published optimum that each method closes.
