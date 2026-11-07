"""Tiny job queue for the local comparators: runs shell commands from a file with N workers, one line per job, logging
stdout/stderr per job. usage: queue.py JOBS.txt WORKERS LOG_DIR
"""
import sys, subprocess, time, os
from concurrent.futures import ThreadPoolExecutor
jobs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
W, LOG = int(sys.argv[2]), sys.argv[3]
os.makedirs(LOG, exist_ok=True)


def run(ij):
    i, cmd = ij
    t0 = time.time()
    with open(os.path.join(LOG, f'job{i:03d}.log'), 'w') as fh:
        fh.write(cmd + '\n'); fh.flush()
        rc = subprocess.call(cmd, shell=True, stdout=fh, stderr=subprocess.STDOUT)
    print(f'job {i} rc={rc} {time.time() - t0:.0f}s :: {cmd[:120]}', flush=True)
    return rc


with ThreadPoolExecutor(W) as ex:
    rcs = list(ex.map(run, enumerate(jobs)))
print('QUEUE_DONE', 'failed', sum(1 for r in rcs if r), 'of', len(rcs), flush=True)
