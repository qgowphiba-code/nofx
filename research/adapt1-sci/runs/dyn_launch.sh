#!/bin/bash
cd "$(dirname "$0")/.."
pb=$1
for spec in "default 1" "profile 1" "default 2" "profile 2"; do set -- $spec; python3 -I dynbench/run_machina.py $pb $1 $2 600 >> runs/dyn_$pb.out 2>&1; done
echo DONE >> runs/dyn_$pb.out
