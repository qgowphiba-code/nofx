#!/bin/bash
cd "$(dirname "$0")/.."
ds=$1
while ! grep -q '^deleted' runs/hte_${ds}_o0.out; do sleep 10; done
for o in 1 2; do python3 -I hte/run_adapt1.py $ds $o 30 > runs/hte_${ds}_o$o.out 2>&1; done
echo DONE >> runs/hte_${ds}_o2.out
