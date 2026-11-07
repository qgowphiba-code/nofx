"""Canonical bsuite baseline agents (bsuite.baselines.jax default_agent, untuned defaults) on the same environments,
seeds and episode budget as the Adapt-1 runs. Same loop as bsuite.baselines.experiment.run; same per-episode metrics.

usage: run_baselines.py ENV AGENT SEED EPISODES OUT_DIR
AGENT: dqn | boot_dqn | actor_critic | actor_critic_rnn | random
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import envs

ENV, AGENT, SEED, EPISODES, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
OUT.mkdir(parents=True, exist_ok=True)
env = envs.make_env(ENV, SEED)
if AGENT == 'random':
    from bsuite.baselines import random as mod
    agent = mod.default_agent(env.observation_spec(), env.action_spec(), seed=SEED)
else:
    mod = __import__(f'bsuite.baselines.jax.{AGENT}', fromlist=['default_agent'])
    agent = mod.default_agent(env.observation_spec(), env.action_spec(), seed=SEED)
t0 = time.time()
with open(OUT / f'episodes-{ENV}-{AGENT}-s{SEED}.jsonl', 'w', buffering=1) as fh:
    for ep in range(EPISODES):
        ts = env.reset(); rewards = []
        while not ts.last():
            a = agent.select_action(ts)
            new_ts = env.step(a)
            agent.update(ts, a, new_ts)
            rewards.append(float(new_ts.reward)); ts = new_ts
        fh.write(json.dumps({'episode': ep, **envs.episode_metrics(ENV, rewards), 'len': len(rewards)}) + '\n')
print(json.dumps({'env': ENV, 'agent': AGENT, 'seed': SEED, 'episodes': EPISODES, 'wall_s': round(time.time() - t0, 1)}))
