held-out (32 unseen batches): base alone 27.694, per-batch ceiling 32.044
| arm | after 0 | 5 | 10 | 20 | 30 updates | gap recovered @30 | radius @30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Machina linear-r0.08 | 27.69 | 28.38 | 30.07 | 30.81 | 31.14 | 79% | 0.08 |
| Machina linear-r0.2 | 27.69 | 29.44 | 30.77 | 31.19 | 31.42 | 86% | 0.11920928955078125 |
| Machina quadratic-r0.2 | 27.69 | 29.01 | 28.76 | 29.35 | 30.12 | 56% | 0.15625 |
| Machina shared-r0.2 | 27.69 | 28.11 | 27.95 | 28.37 | 28.25 | 13% | 0.022737367544323206 |
| CMA-ES linear policy (3 seeds, 7680 executions) | | | | | 30.60 | 67% | |
| (1+1)-ES linear policy (3 seeds, 7680 executions) | | | | | 27.67 | -1% | |
paired over 32 held-out batches: Machina linear-r0.2 - linear-r0.08: +0.277, better on 24, worse on 7
paired over 32 held-out batches: Machina linear-r0.2 - quadratic-r0.2: +1.298, better on 30, worse on 1
paired over 32 held-out batches: Machina linear-r0.2 - shared-r0.2: +3.165, better on 30, worse on 2
paired over 32 held-out batches: Machina linear-r0.2 - CMA-ES linear policy: +0.819, better on 32, worse on 0
paired over 32 held-out batches: Machina linear-r0.2 - (1+1)-ES linear policy: +3.749, better on 32, worse on 0
