#!/bin/bash
cd "$(dirname "$0")/../.."
c=$1
while ! grep -q DONE runs/cryst/acq_c${c}.out; do sleep 5; done
for spec in "profile 3" "profile 4" "default 2"; do
  set -- $spec
  python3 -I cryst/run_machina.py $c $1 $2 400 >> runs/cryst/acq2_c${c}.out 2>&1
done
echo DONE >> runs/cryst/acq2_c${c}.out
