#!/bin/bash
# Round 2 launcher (PREREG.md). usage: launch_round2.sh ib|beer_acq|ib_baselines|beer_baselines
# Adapt-1 runs write outputs under research/adapt1-bench/runs/ and request journals under $S/journals (kept out of git).
set -u
S=/tmp/claude-0/-home-user-nofx/7c289125-8882-5b5c-9bcd-654fe7300b9f/scratchpad
PY=$S/venv/bin/python
R=/home/user/nofx/research/adapt1-bench
export OMP_NUM_THREADS=1 ADAPT1_SCRATCH=$S
mkdir -p $S/journals $S/logs $R/runs/ib/adapt1 $R/runs/ib/baselines $R/runs/beer $R/runs/pensim
case "$1" in
  ib)   # 6 Adapt-1 runs: p50 stationary and drifting, seeds 0-2, 150 episodes of 200 steps
    for drift in 0 1; do for seed in 0 1 2; do
      nohup $PY -I $R/ib_bench/run_adapt1.py 50 $drift $seed 150 $R/runs/ib/adapt1 $S/journals > $S/logs/ib-p50$( [ $drift = 1 ] && echo d )-s$seed.log 2>&1 &
      sleep 20
    done; done ;;
  beer_acq)   # 4 Machina acquisition runs
    for arm in default profile; do for stream in 1 2; do
      nohup $PY -I $R/smpl_bench/beer_machina_acquire.py $arm $stream 600 $R/runs/beer > $S/logs/beer-acq-$arm-r$stream.log 2>&1 &
      sleep 10
    done; done ;;
  ib_baselines)   # local comparators queue (3 workers)
    J=$S/logs/ib_jobs.txt; : > $J
    for drift in 0 1; do
      for seed in 0 1 2; do echo "$PY -I $R/ib_bench/run_baselines.py ppo 50 $drift $seed 1000000 $R/runs/ib/baselines" >> $J; done
      for seed in 0 1 2; do echo "$PY -I $R/ib_bench/run_baselines.py dqn 50 $drift $seed 1000000 $R/runs/ib/baselines" >> $J; done
      echo "$PY -I $R/ib_bench/run_baselines.py zero 50 $drift 0 30000 $R/runs/ib/baselines" >> $J
      echo "$PY -I $R/ib_bench/run_baselines.py random 50 $drift 0 30000 $R/runs/ib/baselines" >> $J
      for seed in 0 1; do echo "$PY -I $R/ib_bench/run_baselines.py sac 50 $drift $seed 500000 $R/runs/ib/baselines" >> $J; done
    done
    nohup $PY -I $R/common/jobqueue.py $J 2 $S/logs/ib_jobs > $S/logs/ib_queue.log 2>&1 & ;;
  beer_baselines)   # acquisition comparators + SB3 queue (2 workers)
    J=$S/logs/beer_jobs.txt; : > $J
    for m in random es11 cmaes; do for seed in 0 1 2 3 4 5 6 7 8 9; do echo "$PY -I $R/smpl_bench/beer_local_acquire.py $m $seed 600 $R/runs/beer/local_acquire.jsonl" >> $J; done; done
    for seed in 0 1 2; do echo "$PY -I $R/smpl_bench/beer_sb3.py train ppo $seed 1000000 $R/runs/beer/sb3" >> $J; done
    for seed in 0 1; do echo "$PY -I $R/smpl_bench/beer_sb3.py train sac $seed 500000 $R/runs/beer/sb3" >> $J; done
    nohup $PY -I $R/common/jobqueue.py $J 2 $S/logs/beer_jobs > $S/logs/beer_queue.log 2>&1 & ;;
  *) echo "unknown target $1"; exit 1 ;;
esac
echo "launched $1"
