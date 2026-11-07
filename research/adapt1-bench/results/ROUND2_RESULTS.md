# Round 2 results (computed by analyze.py from the raw files)

## A. Industrial Benchmark, sequential route

### stationary setpoint 50

| Adapt-1 seed | episodes | first 50 eps | last 50 eps (primary) | whole run | fallback | no-op frac | sequential model | status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 13 | -262.8 | - | -262.8 | 0.000 | 0.01 | - / skill - | in progress |
| 1 | 13 | -297.3 | - | -297.3 | 0.000 | 0.48 | - / skill - | in progress |
| 2 | 13 | -408.0 | - | -408.0 | 0.000 | 0.02 | - / skill - | in progress |

Adapt-1 primary (mean of seeds): -

| comparator | seed | steps run | matched window (steps 20k-30k) | asymptote (last 50 eps) | first reaches Adapt-1 primary at step | whole first 30k |
|---|---:|---:|---:|---:|---:|---:|
| dqn | 0 | 130000 | -296.1 | -172.1 | - | -260.5 |
| ppo | 0 | 1001400 | -201.3 | -150.3 | - | -219.5 |
| ppo | 1 | 1001400 | -181.2 | -151.1 | - | -203.3 |
| ppo | 2 | 376600 | -199.6 | -152.9 | - | -218.5 |

### drifting setpoint

| Adapt-1 seed | episodes | first 50 eps | last 50 eps (primary) | whole run | fallback | no-op frac | sequential model | status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 13 | -665.0 | - | -665.0 | 0.000 | 0.01 | - / skill - | in progress |
| 1 | 13 | -272.8 | - | -272.8 | 0.000 | 0.02 | - / skill - | in progress |
| 2 | 13 | -264.9 | - | -264.9 | 0.000 | 0.33 | - / skill - | in progress |

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
| sac | 0 | 2759 | 292743 | 104.0 | 103.4 | - | - |
| sac | 1 | 2311 | 260510 | -95.8 | 106.5 | - | - |

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
| Machina | default-r1 (300 attempts) | 0.126 | 0.173 | 0.242 | 0.384 | 0.533 | 2350 | 300/300 |
| Machina | profile-r1 (300 attempts) | 0.500 | 0.568 | 0.606 | 0.675 | 0.950 | 4271 | 46/300 |
| default recipe (light control) | seeds 1-5 | - | - | - | - | - | 3247 (nominal batch 3459) | - |
| random | seed 0 | 0.418 | 0.418 | 0.418 | 0.418 | 0.598 | 2649 | - |
| es11 | seed 0 | 0.787 | 0.833 | 0.859 | 0.874 | 0.875 | 3923 | - |
| cmaes | seed 0 | 0.539 | 0.625 | 0.646 | 0.795 | 0.876 | 3928 | - |

## D. ChemGymRL reaction benches (Machina acquisition, production)

Best quality so far after N attempts (Machina: one run per arm; comparators: mean over seeds). "reach" = attempts to 99 % of
the heuristic recipe (the reference optimum where black-box search falls short), "-" = not reached within 300.

| bench / target | heuristic | CMA-ES 4,000 | method | @16 | @32 | @64 | @128 | @300 | reach |
|---|---|---|---|---|---|---|---|---|---|
| fict/E | 0.961 | 0.902 | Machina profile | 0.734 | 0.740 | 0.904 | 0.959 | 0.961 | 87 |
| | | | Machina default | 0.698 | 0.699 | 0.737 | 0.742 | 0.814 | - |
| | | | cmaes (1 seeds) | 0.754 | 0.828 | 0.842 | 0.892 | 0.945 | 0/1 seeds |
| | | | es11 (5 seeds) | 0.837 | 0.866 | 0.873 | 0.874 | 0.874 | 0/5 seeds |
| | | | random (5 seeds) | 0.761 | 0.776 | 0.788 | 0.793 | 0.799 | 0/5 seeds |
| fict/F | 0.982 | 0.411 | Machina profile | 0.215 | 0.217 | 0.337 | 0.399 | 0.774 | - |
| | | | Machina default | 0.131 | 0.134 | 0.139 | 0.168 | 0.206 | - |
| | | | cmaes (1 seeds) | 0.235 | 0.235 | 0.312 | 0.383 | 0.457 | 0/1 seeds |
| | | | es11 (5 seeds) | 0.306 | 0.366 | 0.387 | 0.394 | 0.395 | 0/5 seeds |
| | | | random (5 seeds) | 0.249 | 0.258 | 0.284 | 0.284 | 0.301 | 0/5 seeds |
| fict/G | 0.982 | 0.591 | Machina profile | 0.233 | 0.235 | 0.499 | 0.980 | 0.982 | 67 |
| | | | Machina default | 0.129 | 0.132 | 0.136 | 0.169 | 0.207 | - |
| | | | cmaes (1 seeds) | 0.249 | 0.249 | 0.347 | 0.510 | 0.627 | 0/1 seeds |
| | | | es11 (5 seeds) | 0.278 | 0.348 | 0.382 | 0.420 | 0.433 | 0/5 seeds |
| | | | random (5 seeds) | 0.252 | 0.274 | 0.284 | 0.294 | 0.294 | 0/5 seeds |
| fict/I | 0.450 | 0.408 | Machina profile | 0.210 | 0.228 | 0.330 | 0.376 | 0.429 | - |
| | | | Machina default | 0.102 | 0.106 | 0.113 | 0.155 | 0.184 | - |
| | | | es11 (5 seeds) | 0.251 | 0.320 | 0.342 | 0.346 | 0.347 | 0/5 seeds |
| | | | random (5 seeds) | 0.240 | 0.254 | 0.271 | 0.272 | 0.277 | 0/5 seeds |
| wurtz/4,5-diethyloctane | 0.454 | 0.366 | Machina profile | 0.295 | 0.296 | 0.314 | 0.432 | 0.454 | 150 |
| | | | Machina default | 0.288 | 0.288 | 0.288 | 0.290 | 0.336 | - |
| | | | cmaes (5 seeds) | 0.222 | 0.242 | 0.320 | 0.371 | 0.401 | 0/5 seeds |
| | | | es11 (5 seeds) | 0.280 | 0.354 | 0.370 | 0.370 | 0.370 | 0/5 seeds |
| | | | random (5 seeds) | 0.233 | 0.240 | 0.258 | 0.264 | 0.276 | 0/5 seeds |
| wurtz/4-ethyl-5-methylnonane | 0.278 | 0.253 | Machina profile | 0.171 | 0.174 | 0.178 | 0.263 | 0.277 | 229 |
| | | | Machina default | 0.115 | 0.115 | 0.173 | 0.173 | 0.208 | - |
| | | | cmaes (5 seeds) | 0.182 | 0.201 | 0.230 | 0.252 | 0.263 | 0/5 seeds |
| | | | es11 (5 seeds) | 0.220 | 0.236 | 0.250 | 0.251 | 0.251 | 0/5 seeds |
| | | | random (5 seeds) | 0.200 | 0.203 | 0.208 | 0.210 | 0.218 | 0/5 seeds |
| wurtz/4-ethyldecane | 0.278 | 0.255 | Machina profile | 0.192 | 0.193 | 0.194 | 0.232 | 0.278 | 153 |
| | | | Machina default | 0.111 | 0.111 | 0.181 | 0.185 | 0.192 | - |
| | | | cmaes (5 seeds) | 0.187 | 0.212 | 0.230 | 0.251 | 0.262 | 0/5 seeds |
| | | | es11 (5 seeds) | 0.223 | 0.242 | 0.248 | 0.249 | 0.249 | 0/5 seeds |
| | | | random (5 seeds) | 0.188 | 0.206 | 0.213 | 0.217 | 0.217 | 0/5 seeds |
| wurtz/5,6-dimethyldecane | 0.454 | 0.376 | Machina profile | 0.218 | 0.225 | 0.297 | 0.354 | 0.454 | 249 |
| | | | Machina default | 0.193 | 0.193 | 0.193 | 0.207 | 0.215 | - |
| | | | cmaes (5 seeds) | 0.199 | 0.238 | 0.310 | 0.367 | 0.395 | 0/5 seeds |
| | | | es11 (5 seeds) | 0.268 | 0.320 | 0.341 | 0.361 | 0.361 | 0/5 seeds |
| | | | random (5 seeds) | 0.216 | 0.229 | 0.259 | 0.276 | 0.286 | 0/5 seeds |
| wurtz/5-methylundecane | 0.277 | 0.255 | Machina profile | 0.190 | 0.190 | 0.197 | 0.266 | 0.277 | 228 |
| | | | Machina default | 0.169 | 0.169 | 0.169 | 0.177 | 0.180 | - |
| | | | cmaes (5 seeds) | 0.185 | 0.196 | 0.216 | 0.248 | 0.259 | 0/5 seeds |
| | | | es11 (5 seeds) | 0.201 | 0.235 | 0.245 | 0.245 | 0.246 | 0/5 seeds |
| | | | random (5 seeds) | 0.185 | 0.192 | 0.197 | 0.205 | 0.207 | 0/5 seeds |
| wurtz/NaCl | 0.726 | 0.725 | Machina profile | 0.694 | 0.694 | 0.694 | 0.701 | 0.726 | 229 |
| | | | Machina default | 0.694 | 0.694 | 0.694 | 0.696 | 0.700 | - |
| | | | cmaes (5 seeds) | 0.700 | 0.706 | 0.718 | 0.726 | 0.726 | 5/5 seeds |
| | | | es11 (5 seeds) | 0.714 | 0.723 | 0.726 | 0.726 | 0.726 | 5/5 seeds |
| | | | random (5 seeds) | 0.697 | 0.700 | 0.702 | 0.703 | 0.707 | 0/5 seeds |
| wurtz/dodecane | 0.454 | 0.371 | Machina profile | 0.211 | 0.211 | 0.314 | 0.346 | 0.400 | - |
| | | | Machina default | 0.165 | 0.165 | 0.205 | 0.241 | 0.284 | - |
| | | | cmaes (5 seeds) | 0.222 | 0.252 | 0.334 | 0.364 | 0.388 | 0/5 seeds |
| | | | es11 (5 seeds) | 0.274 | 0.302 | 0.331 | 0.342 | 0.343 | 0/5 seeds |
| | | | random (5 seeds) | 0.227 | 0.260 | 0.274 | 0.278 | 0.300 | 0/5 seeds |

