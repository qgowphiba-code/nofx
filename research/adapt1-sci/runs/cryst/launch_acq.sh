#!/bin/bash
cd "$(dirname "$0")/../.."
c=$1
for spec in "profile 1" "profile 2" "default 1"; do
  set -- $spec
  python3 -I cryst/run_machina.py $c $1 $2 400 >> runs/cryst/acq_c${c}.out 2>&1
done
echo DONE >> runs/cryst/acq_c${c}.out
