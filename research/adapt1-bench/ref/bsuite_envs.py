"""Canonical bsuite environments (google-deepmind/bsuite 0.3.6), shared by the Adapt-1 runner and the baseline runner so both
see identical environments, settings and seeds.

Settings are bsuite's own experiment environments at one sweep point each; only the episode budget is reduced (bsuite uses
10,000 episodes; API throughput allows ~1,000 here, the same for every agent).

  bandit      bandit/*: SimpleBandit, 11 arms, rewards evenly spaced in [0, 1], 1-step episodes (basic)
  catch       catch/*: 10x5 grid, 3 actions, 9-step episodes, reward +1/-1 at the end (basic, credit assignment)
  deep_sea10  deep_sea size 10: 10x10 one-hot grid, 2 actions, 10-step episodes, randomized action mapping; +1 treasure only
              by always going right, -0.001 per right move (exploration)
  umbrella10  umbrella_length chain 10 with 20 distractor bits: only the first action matters, reward +1/-1 at step 10,
              random +1/-1 rewards at every other step (credit assignment, noise)
  memory5     memory_len length 5: a 1-bit cue visible only at t=0 must be reported at the last step, +1/-1 (memory)
"""
import numpy as np
from bsuite.environments import bandit, catch, deep_sea, memory_chain, umbrella_chain

SPECS = {
    'bandit':     {'r_min': 0.0,  'r_max': 1.0, 'ep_len': 1,  'sequential': False, 'memory': False},
    'catch':      {'r_min': -1.0, 'r_max': 1.0, 'ep_len': 9,  'sequential': True,  'memory': False},
    'deep_sea10': {'r_min': -1.0, 'r_max': 1.0, 'ep_len': 10, 'sequential': True,  'memory': False},
    'umbrella10': {'r_min': -1.0, 'r_max': 1.0, 'ep_len': 10, 'sequential': True,  'memory': False},
    'memory5':    {'r_min': -1.0, 'r_max': 1.0, 'ep_len': 6,  'sequential': True,  'memory': True},
}


def make_env(name, seed):
    if name == 'bandit':
        return bandit.SimpleBandit(mapping_seed=seed)
    if name == 'catch':
        return catch.Catch(seed=seed)
    if name == 'deep_sea10':
        return deep_sea.DeepSea(size=10, seed=seed, mapping_seed=seed)
    if name == 'umbrella10':
        return umbrella_chain.UmbrellaChain(chain_length=10, n_distractor=20, seed=seed)
    if name == 'memory5':
        return memory_chain.MemoryChain(memory_length=5, num_bits=1, seed=seed)
    raise ValueError(name)


def obs_values(obs):
    """Flattened observation as named scalar fields (the same information the baseline agents receive)."""
    return {f'o{i}': round(float(v), 6) for i, v in enumerate(np.asarray(obs, dtype=float).ravel())}


def episode_metrics(name, rewards):
    """Per-episode outcome used for scoring; identical for every agent."""
    ret = float(sum(rewards)); last = float(rewards[-1])
    m = {'return': ret}
    if name == 'bandit':
        m['regret'] = 1.0 - ret                      # best arm pays 1.0
    elif name == 'catch':
        m['success'] = last > 0                      # caught the ball
    elif name == 'deep_sea10':
        m['success'] = any(r > 0.5 for r in rewards)  # reached the treasure
    elif name in ('umbrella10', 'memory5'):
        m['success'] = last > 0                      # final, decision-dependent reward was +1
    return m
