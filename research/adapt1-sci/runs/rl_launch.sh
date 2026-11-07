#!/bin/bash
cd "$(dirname "$0")/.."
for spec in "default 1" "profile 1" "default 2" "profile 2"; do set -- $spec; python3 -I dynbench/run_machina.py rlcomplex $1 $2 600 >> runs/dyn_rlcomplex.out 2>&1; done
echo DONE >> runs/dyn_rlcomplex.out
