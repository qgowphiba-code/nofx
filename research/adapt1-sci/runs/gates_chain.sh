#!/bin/bash
# when each task's acquisition arms are DONE, run its held-out headroom gate (base = best Machina acquisition)
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
gate() {  # task donefile
  while ! grep -q DONE $2 2>/dev/null; do sleep 15; done
  for r in 0-8 8-16 16-24 24-32; do python3 -I dynbench/ctx_headroom3.py $1 $r > runs/ctx3_$1_$r.jsonl 2>&1 & done; wait
  echo GATE_DONE $1 >> runs/gates.log
}
gate arm runs/dyn_arm.out &
gate insulin runs/dyn_insulin.out &
wait
