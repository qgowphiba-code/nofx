#!/bin/bash
cd "$(dirname "$0")/.."
for s in 0 1 2; do
  python3 -I pool/run_adapt1.py crossed_barrel 100 thompson $s >> runs/pool_adapt1_init.out 2>&1 &
  python3 -I pool/run_adapt1.py photo_pce10 100 thompson $s >> runs/pool_adapt1_init.out 2>&1 &
  wait
done
echo DONE >> runs/pool_adapt1_init.out
