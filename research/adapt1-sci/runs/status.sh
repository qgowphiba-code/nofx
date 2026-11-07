#!/bin/bash
# compact status of everything running (keep output short)
cd "$(dirname "$0")/.."
for f in runs/dyn_penicillin.out runs/dyn_insulin.out runs/dyn_arm.out runs/r2_pr2drift_linear.out; do [ -f $f ] && echo "$(basename $f): $(tail -n 1 $f | cut -c1-110)"; done
for f in runs/scale2_pr5_*_1.jsonl runs/dyn_local_penicillin.jsonl runs/dyn_local_insulin.jsonl runs/dyn_local_arm.jsonl; do [ -f $f ] && echo "$(basename $f): $(wc -l < $f) lines"; done
python3 -I common/live.py
