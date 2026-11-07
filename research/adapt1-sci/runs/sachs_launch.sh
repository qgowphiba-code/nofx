#!/bin/bash
cd "$(dirname "$0")/.."
for K in 3 5; do for s in 0 1 2; do python3 -I sachs/run.py $K $s 1000 >> runs/sachs/run.out 2>&1; done; done
echo DONE >> runs/sachs/run.out
