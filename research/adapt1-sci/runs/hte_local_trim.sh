#!/bin/bash
# stop the original sequential comparator processes once they move past the orders now covered by separate processes
cd "$(dirname "$0")"
while true; do
  if grep -q '"order": 1' hte_local_aryl_amination.jsonl; then pkill -f "run_local.py aryl_amination 30 0,1,2" && echo killed-aryl; fi
  if grep -q '"order": 2' hte_local_suzuki.jsonl; then pkill -f "run_local.py suzuki 30 0,1,2" && echo killed-suzuki; fi
  pgrep -f "run_local.py aryl_amination 30 0,1,2" > /dev/null || pgrep -f "run_local.py suzuki 30 0,1,2" > /dev/null || break
  sleep 5
done
