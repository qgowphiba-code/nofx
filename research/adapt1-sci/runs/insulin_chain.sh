#!/bin/bash
cd "$(dirname "$0")/.."
while ! grep -q "GATE_DONE insulin" runs/gates.log 2>/dev/null; do sleep 20; done
loss=$(python3 -c "
import json, glob, statistics as st
rows=[json.loads(l) for f in glob.glob('runs/ctx3_insulin_*.jsonl') for l in open(f) if l.startswith('{')]
print('%.4f %d %.4f %.4f' % (st.mean((r['ceiling']-r['base'])/r['ceiling'] for r in rows), len(rows), st.mean(r['base'] for r in rows), st.mean(r['ceiling'] for r in rows)))")
echo "insulin gate: mean_loss n base ceiling = $loss" >> runs/gates.log
ok=$(python3 -c "print(1 if float('$loss'.split()[0]) >= 0.03 else 0)")
if [ "$ok" = "1" ]; then
  for a in linear quadratic shared; do nohup python3 -I dynbench/refine2.py insulin $a 30 0.2 > runs/r2_insulin_$a.out 2>&1 & done
  export OMP_NUM_THREADS=1
  for m in cmaes nes; do nohup python3 -I dynbench/scale2.py insulin $m 0 61440 > runs/scale2_insulin_${m}_0.jsonl 2> runs/scale2_insulin_${m}_0.err & done
  echo "insulin refinement launched" >> runs/gates.log
else
  echo "insulin gate failed: not run" >> runs/gates.log
fi
