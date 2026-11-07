#!/bin/bash
cd "$(dirname "$0")/.."
while ! grep -q DONE runs/dyn_penicillin.out; do sleep 10; done
for spec in "default 1" "profile 1" "default 2"; do set -- $spec; python3 -I dynbench/run_machina.py insulin $1 $2 600 >> runs/dyn_insulin.out 2>&1; done
echo DONE >> runs/dyn_insulin.out
