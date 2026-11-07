#!/bin/bash
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while ! grep -q DONE runs/dyn_insulin.out 2>/dev/null; do sleep 15; done
for r in 0-8 8-16 16-24 24-32; do python3 -I dynbench/ctx_headroom3.py insulin $r > runs/ctx3_insulin_$r.jsonl 2>&1 & done; wait
echo GATE_DONE insulin >> runs/gates.log
