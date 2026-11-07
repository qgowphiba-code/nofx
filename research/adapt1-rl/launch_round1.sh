#!/bin/bash
# Round 1 launcher. usage: launch_round1.sh PYTHON JOURNAL_DIR LOG_DIR
PY=$1; JD=$2; LD=$3; cd "$(dirname "$0")"
eps() { [ "$1" = bandit ] && echo 1000 || echo 500; }
for e in bandit catch deep_sea10 umbrella10 memory5; do for s in 0 1 2; do
  nohup "$PY" run_adapt1.py $e $s $(eps $e) runs/adapt1 "$JD" > "$LD/adapt1-$e-s$s.log" 2>&1 &
done; done
for e in bandit catch deep_sea10 umbrella10 memory5; do for a in dqn boot_dqn actor_critic actor_critic_rnn random; do for s in 0 1 2 3 4; do
  echo "$e $a $s $(eps $e)"
done; done; done | XLA_FLAGS=--xla_cpu_multi_thread_eigen=false xargs -P 3 -L 1 sh -c '"$0" run_baselines.py $1 $2 $3 $4 runs/baselines >> "'"$LD"'/baselines.log" 2>&1' "$PY"
echo BASELINES_DONE >> "$LD/baselines.log"
