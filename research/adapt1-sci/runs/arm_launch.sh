#!/bin/bash
cd "$(dirname "$0")/.."
for spec in "default 1" "profile 1"; do set -- $spec; python3 -I dynbench/run_machina.py arm $1 $2 1500 >> runs/dyn_arm.out 2>&1; done
echo DONE >> runs/dyn_arm.out
