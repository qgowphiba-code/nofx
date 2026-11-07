# Round 2 results (computed by analyze.py from the raw files)

## A. Industrial Benchmark, sequential route

### stationary setpoint 50

| Adapt-1 seed | episodes | first 50 eps | last 50 eps (primary) | whole run | fallback | no-op frac | sequential model | status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 4 | -259.5 | - | -259.5 | 0.000 | 0.03 | - / skill - | in progress |
| 1 | 4 | -304.0 | - | -304.0 | 0.000 | 0.13 | - / skill - | in progress |
| 2 | 4 | -306.2 | - | -306.2 | 0.000 | 0.04 | - / skill - | in progress |

Adapt-1 primary (mean of seeds): -

| comparator | seed | steps run | matched window (steps 20k-30k) | asymptote (last 50 eps) | first reaches Adapt-1 primary at step | whole first 30k |
|---|---:|---:|---:|---:|---:|---:|
| ppo | 0 | 862200 | -201.3 | -149.7 | - | -219.5 |
| ppo | 1 | 868600 | -181.2 | -150.8 | - | -203.3 |

### drifting setpoint

| Adapt-1 seed | episodes | first 50 eps | last 50 eps (primary) | whole run | fallback | no-op frac | sequential model | status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 4 | -478.9 | - | -478.9 | 0.000 | 0.02 | - / skill - | in progress |
| 1 | 4 | -318.9 | - | -318.9 | 0.000 | 0.05 | - / skill - | in progress |
| 2 | 4 | -246.5 | - | -246.5 | 0.000 | 0.10 | - / skill - | in progress |

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
| ppo | 0 | 10290 | 1001410 | -181.4 | 119.9 | 2398 | 120.0 |
| ppo | 1 | 11033 | 1001408 | 53.3 | 120.5 | 2961 | 120.1 |
| ppo | 2 | 9975 | 1001458 | -115.6 | 120.4 | 3505 | 120.1 |
| sac | 0 | 497 | 65891 | - | 103.9 | - | - |
| sac | 1 | 128 | 23217 | - | -195.1 | - | - |

### B2 refinement: held-out (32 unseen batches), mean SMPL return; base alone and per-batch ceiling from the local side

base alone: -82.5 (finished 0.38); per-batch ceiling: 116.0; best constant per batch: 63.8

| arm / learner | executions | after 5 | 10 | 20 | 30 updates (held-out mean return) | finished frac | gap recovered | paired vs base (better/worse, p) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Machina linear-r0.08 | 7680 | 16.7 | 26.5 | 44.9 | 34.9 | 0.75 | 0.59 | 12/3, p=0.035 |
| Machina linear-r0.2 | 7680 | -3.7 | 63.3 | 63.3 | 63.5 | 0.84 | 0.74 | 17/10, p=0.248 |
| Machina shared-r0.2 | 7680 | 74.0 | 74.0 | 74.8 | 74.8 | 0.88 | 0.79 | 18/4, p=0.004 |
| contextual_cmaes seed 0 | 7680 | - | - | - | 105.0 | 0.97 | 0.94 | 21/4, p=0.001 |
| contextual_cmaes seed 1 | 7680 | - | - | - | 46.1 | 0.78 | 0.65 | 18/4, p=0.004 |
| contextual_cmaes seed 2 | 7680 | - | - | - | 66.1 | 0.84 | 0.75 | 17/0, p=0.000 |
| contextual_nes seed 0 | 7680 | - | - | - | 95.0 | 0.94 | 0.89 | 20/5, p=0.004 |
| contextual_nes seed 1 | 7680 | - | - | - | 85.1 | 0.91 | 0.84 | 19/4, p=0.003 |
| contextual_nes seed 2 | 7680 | - | - | - | 85.5 | 0.91 | 0.85 | 19/3, p=0.001 |
| robust_profile seed 0 | 7680 | - | - | - | 85.1 | 0.91 | 0.84 | 19/5, p=0.007 |
| robust_profile seed 1 | 7680 | - | - | - | 66.2 | 0.84 | 0.75 | 27/0, p=0.000 |
| robust_profile seed 2 | 7680 | - | - | - | 85.9 | 0.91 | 0.85 | 22/0, p=0.000 |
| ppo-s0 (closed loop, trained 1001410 steps) | 1001410 | - | - | - | 120.0 | 1.00 | 1.02 | 32/0, p=0.000 |
| ppo-s1 (closed loop, trained 1001408 steps) | 1001408 | - | - | - | 120.1 | 1.00 | 1.02 | 32/0, p=0.000 |
| ppo-s2 (closed loop, trained 1001458 steps) | 1001458 | - | - | - | 120.1 | 1.00 | 1.02 | 32/0, p=0.000 |

Paired sign tests on the 32 held-out batches, Machina linear 0.2 (final) vs each comparator (better/worse, p):

- vs contextual_cmaes seed 0: 0/29, p=0.000
- vs contextual_cmaes seed 1: 3/24, p=0.000
- vs contextual_cmaes seed 2: 0/27, p=0.000
- vs contextual_nes seed 0: 0/30, p=0.000
- vs contextual_nes seed 1: 1/26, p=0.000
- vs contextual_nes seed 2: 0/28, p=0.000
- vs robust_profile seed 0: 0/26, p=0.000
- vs robust_profile seed 1: 1/27, p=0.000
- vs robust_profile seed 2: 0/29, p=0.000
- vs ppo-s0: 0/32, p=0.000
- vs ppo-s1: 0/32, p=0.000
- vs ppo-s2: 0/32, p=0.000
- vs Machina linear-r0.08: 5/22, p=0.002
- vs Machina shared-r0.2: 1/28, p=0.000

## C. SMPL PenSimEnv, Machina acquisition (nominal batch seed_ref 1000); quality = (return + 100) / 4600

| method | run | n=16 | n=32 | n=64 | n=128 | n=300 | best yield (kg) | failed batches |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Machina | default-r1 (171 attempts) | 0.126 | 0.173 | 0.242 | 0.384 | 0.391 | 1699 | 171/171 |
| Machina | profile-r1 (115 attempts) | 0.500 | 0.568 | 0.606 | 0.675 | 0.675 | 3003 | 33/115 |
| default recipe (light control) | seeds 1-5 | - | - | - | - | - | 3247 (nominal batch 3459) | - |
| GP-EI dataset (smpl-experiments, 1000 batches, mixed seeds) | - | - | - | - | - | - | mean 3806, p90 4066, max 4080 | - |

