# Pre-registration: Adapt-1 on bsuite core tasks vs bsuite's canonical agents (round 1)

Written and committed before any counted run. Smoke tests used seed 99 only and are excluded.

## Question
Does Adapt-1's general online route (contextual policy learning plus sequential learning, configured with the vendor's
recommended starting declaration) learn bsuite's core tasks as data-efficiently as bsuite's own baseline agents, at matched
episode budgets?

## Environments (canonical bsuite 0.3.6 code, `envs.py`)
| Env | bsuite experiment and setting | Tests | Episodes per run |
|---|---|---|---:|
| bandit | `bandit`, 11 arms | basic learning | 1,000 |
| catch | `catch`, 10x5 | basic, credit assignment | 500 |
| deep_sea10 | `deep_sea`, size 10 (action mapping seeded per run) | deep exploration | 500 |
| umbrella10 | `umbrella_length`, chain 10, 20 distractor bits | delayed credit under reward noise | 500 |
| memory5 | `memory_len`, length 5 | memory | 500 |

bsuite's own protocol uses 10,000 episodes (1,000 for some). The budget here is reduced only because each Adapt-1 step
costs about 1.3-1.5 s of API time; every agent gets the same reduced budget. Published bsuite scores are therefore not
directly comparable; the canonical agents are re-run locally at the same budget instead.

## Agents
- **Adapt-1** (Fast-UAT, `run_adapt1.py`): the docs' "Recommended starting declaration" (sequential-learning guide, section
  3), unchanged except for task-specific fields: context = flattened observation; policies = the env's actions;
  action_transfer_strength 0 (actions have unrelated semantics, docs section 10); reward = raw bsuite step reward with its
  true bounds; bandit: sequential off and credit_assignment none (immediate reward); memory5: temporal_context over all
  observation fields, maximum_lag 6. Online loop per docs section 13: one query and one decision-linked feedback per step,
  learning and exploration on throughout. Seeds 0, 1, 2.
- **bsuite canonical agents** (`run_baselines.py`, `bsuite.baselines.jax` default_agent, untuned defaults): DQN,
  Bootstrapped DQN (20 heads), A2C (actor_critic), A2C with an RNN (actor_critic_rnn), and random. Seeds 0-4.

## Metrics (identical for every agent, from `envs.episode_metrics`)
Primary, per env:
- bandit: mean regret per episode over all 1,000 episodes (online, as bsuite scores it); also over the last 100.
- catch: success rate (ball caught) over all episodes and over the last 100.
- deep_sea10: episodes until the first treasure; fraction of treasure episodes over the last 100.
- umbrella10: success rate (final reward +1) over the last 100 and over all episodes.
- memory5: success rate over the last 100 and over all episodes.
Learning curves: means over blocks of 50 episodes.

## Analysis plan
- Report every seed's value, the mean per agent, and where Adapt-1's mean falls relative to each canonical agent's mean
  and seed range. With 3 and 5 seeds, results are descriptive; a difference is called only when Adapt-1's seed range
  does not overlap the other agent's seed range.
- Adapt-1 fallbacks (abstained or illegal selections executed as random actions) are reported per run; if any run has
  more than 1 % fallback steps, its results are also reported with those steps excluded.

## Validity rules
- A run counts only if it completes all its episodes and fewer than 1 % of its feedback calls are reported as not
  applied to contextual learning (the runner aborts above that).
- An aborted or crashed run is reported, not dropped. It is rerun once from scratch with the same seed and a new Domain;
  both are reported.
- Every Domain is deleted at the end of its run and the deletion is confirmed by a 404 readback.
- Sequential-model diagnostics (installation, validation skill, sample counts) are logged at every 50th episode and
  reported.
