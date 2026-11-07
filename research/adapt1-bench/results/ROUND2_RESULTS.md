# Round 2 results (computed by analyze.py from the raw files)

## A. Industrial Benchmark, sequential route

### stationary setpoint 50

| Adapt-1 seed | episodes | first 50 eps | last 50 eps (primary) | whole run | fallback | no-op frac | sequential model | status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 1 | -227.3 | - | -227.3 | 0.000 | 0.00 | - / skill - | in progress |
| 1 | 1 | -282.2 | - | -282.2 | 0.000 | 0.00 | - / skill - | in progress |
| 2 | 1 | -249.8 | - | -249.8 | 0.000 | 0.00 | - / skill - | in progress |

Adapt-1 primary (mean of seeds): -

| comparator | seed | steps run | matched window (steps 20k-30k) | asymptote (last 50 eps) | first reaches Adapt-1 primary at step | whole first 30k |
|---|---:|---:|---:|---:|---:|---:|
| ppo | 0 | 565200 | -201.3 | -151.3 | - | -219.5 |
| ppo | 1 | 571000 | -181.2 | -150.8 | - | -203.3 |

### drifting setpoint

| Adapt-1 seed | episodes | first 50 eps | last 50 eps (primary) | whole run | fallback | no-op frac | sequential model | status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 1 | -183.2 | - | -183.2 | 0.000 | 0.00 | - / skill - | in progress |
| 1 | 1 | -275.0 | - | -275.0 | 0.000 | 0.00 | - / skill - | in progress |
| 2 | 1 | -195.6 | - | -195.6 | 0.000 | 0.00 | - / skill - | in progress |

Adapt-1 primary (mean of seeds): -

| comparator | seed | steps run | matched window (steps 20k-30k) | asymptote (last 50 eps) | first reaches Adapt-1 primary at step | whole first 30k |
|---|---:|---:|---:|---:|---:|---:|

## B. SMPL BeerFMT, Machina

### B1 acquisition (nominal batch): best quality after n attempts (return = 600 q - 400)

| method | run | n=16 | n=32 | n=64 | n=128 | n=256 | n=600 | best return |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Machina | default-r1 | 0.677 | 0.735 | 0.767 | 0.775 | 0.810 | 0.833 | 100.0 |
| Machina | default-r2 | 0.752 | 0.778 | 0.807 | 0.837 | 0.845 | 0.860 | 116.0 |
| Machina | profile-r1 | 0.783 | 0.817 | 0.822 | 0.845 | 0.860 | 0.860 | 116.0 |
| Machina | profile-r2 | 0.790 | 0.793 | 0.828 | 0.843 | 0.853 | 0.860 | 116.0 |
| random | mean of 10 seeds | 0.791 [0.78-0.81] | 0.793 [0.78-0.81] | 0.799 [0.79-0.81] | 0.802 [0.79-0.81] | 0.808 [0.80-0.81] | 0.815 [0.81-0.83] | 88.7 |
| es11 | mean of 10 seeds | 0.816 [0.79-0.85] | 0.821 [0.80-0.85] | 0.828 [0.81-0.85] | 0.837 [0.81-0.85] | 0.844 [0.82-0.86] | 0.851 [0.82-0.86] | 110.4 |
| cmaes | mean of 10 seeds | 0.774 [0.75-0.80] | 0.793 [0.78-0.81] | 0.815 [0.79-0.84] | 0.837 [0.82-0.86] | 0.852 [0.83-0.86] | 0.858 [0.85-0.87] | 114.7 |
| best constant 14.5 C | - | 0.780 | 0.780 | 0.780 | 0.780 | 0.780 | 0.780 | 68 |
| CMA-ES 8,000 evals (gate reference) | - | - | - | - | - | - | - | 120 |

Deep-RL comparators (closed loop, SMPL protocol, one episode = one batch):

| algo | seed | episodes | steps | running-50 mean return at 600 episodes | final running-50 mean | first episode reaching return 116 | held-out mean return (32 batches) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0 | 6615 | 681949 | -181.4 | 120.3 | 2398 | - |
| ppo | 1 | 7229 | 693430 | 53.3 | 119.6 | 2961 | - |

### B2 refinement: held-out (32 unseen batches), mean SMPL return; base alone and per-batch ceiling from the local side

base alone: -82.5 (finished 0.38); per-batch ceiling: -; best constant per batch: 63.8

| arm / learner | executions | after 5 | 10 | 20 | 30 updates (held-out mean return) | finished frac | gap recovered | paired vs base (better/worse, p) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Machina linear-r0.08 | 0 | - | - | - | - | 0.38 | - | 0/0, p=1.000 |
| Machina linear-r0.2 | 0 | - | - | - | - | 0.38 | - | 0/0, p=1.000 |
| Machina shared-r0.2 | 0 | - | - | - | - | 0.38 | - | 0/0, p=1.000 |

Paired sign tests on the 32 held-out batches, Machina linear 0.2 (final) vs each comparator (better/worse, p):

- vs Machina linear-r0.08: 0/0, p=1.000
- vs Machina shared-r0.2: 0/0, p=1.000

## C. SMPL PenSimEnv, Machina acquisition (nominal batch seed_ref 1000); quality = (return + 100) / 4600

| method | run | n=16 | n=32 | n=64 | n=128 | n=300 | best yield (kg) | failed batches |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Machina | default-r1 (27 attempts) | 0.018 | 0.052 | 0.052 | 0.052 | 0.052 | 139 | 27/27 |
| Machina | profile-r1 (26 attempts) | 0.034 | 0.037 | 0.037 | 0.037 | 0.037 | 70 | 26/26 |
| default recipe (light control) | seeds 1-5 | - | - | - | - | - | 2163 (nominal batch 3481) | - |
| GP-EI dataset (smpl-experiments, 1000 batches, mixed seeds) | - | - | - | - | - | - | mean 3806, p90 4066, max 4080 | - |

