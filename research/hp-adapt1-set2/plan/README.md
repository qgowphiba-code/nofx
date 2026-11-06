# HP × Adapt-1, set 2: experiment plan package (2026-11-06)

This is a handoff for the next run. It builds on `hp-adapt1-first-set` (set 1).
- It contains no API keys or credentials.
- Nothing in it has been run against the API.
- All set-1 domains have been deleted, and nothing here depends on them.

## Read in this order
1. **[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md):** the full program. It covers:
   - the tracks, each experiment, and its question, design, cost and gate;
   - the pre-registered predictions and decision rules;
   - the seeds and fit settings.
2. **`analysis/`:** the offline baselines the plan relies on (plan Appendix C).

## Contents
| Path | What it is |
|---|---|
| `EXPERIMENT_PLAN.md` | The plan |
| `analysis/fasthp.py`, `analysis/fasthp_nd.py` | Fast contact-heuristic planners (3D, and 2D/3D). They reproduce set 1's 68/113/134 exactly |
| `analysis/check_liu.py` | 3D Liu panel: heuristic tie-break distributions at depths 1–5, best-of-k search, the Stage D control streams |
| `analysis/check_short.py` | Short-chain oracle set: heuristic best-move rate with first-tie vs fair ties |
| `analysis/check_payload.py` | Planning request sizes against the 1 MiB limit |
| `analysis/check_potential.py` | Compactness tie-break variants |
| `analysis/check_neutral.py` | Search without learned values (neutral leaves) |
| `analysis/check_2d.py` | 2D vs 3D baselines and trap rates on the same seven sequences |
| `analysis/out/` | Outputs of every check |
| `analysis/data/liu_panel.json` | The 7-chain Liu panel, from set 1 |
| `analysis/vendor/` | Unmodified `hp.py`, `lattice_env.py` and `grading.py` from set 1 |

## Rerun the offline checks
Each check takes from about 20 seconds to 3.5 minutes; all of them together take about 10 minutes. None uses the network.

```bash
cd analysis
```

```bash
python3 -I check_liu.py data/liu_panel.json
```

```bash
python3 -I check_short.py
```

```bash
python3 -I check_2d.py data/liu_panel.json
```

`check_payload.py`, `check_potential.py` and `check_neutral.py` take the same `data/liu_panel.json` argument.

## Sources used for the plan
- Adapt-1 docs:
  - https://docs.reilabs.org/docs/neuroadapt/value-learning
  - https://docs.reilabs.org/docs/neuroadapt/value-observations
  - https://docs.reilabs.org/docs/neuroadapt/value-training
  - https://docs.reilabs.org/docs/neuroadapt/value-promotion
  - https://docs.reilabs.org/docs/neuroadapt/value-api
- Planning:
  - https://docs.reilabs.org/docs/planning/api
  - https://docs.reilabs.org/docs/planning/integration
  - https://docs.reilabs.org/docs/planning/options
  - https://docs.reilabs.org/docs/planning/stream-protocol
- Sequential route and operations:
  - https://docs.reilabs.org/docs/neuroadapt/sequential-learning
  - https://docs.reilabs.org/docs/neuroadapt/discovery-sequential
  - https://docs.reilabs.org/docs/neuroadapt/request-contracts
  - https://docs.reilabs.org/docs/neuroadapt/operational-behavior-and-retries
- Method article: https://reilabs.org/blog/coupled-learning-and-planning (blended targets, richer inputs, the paired-comparison protocol).
- Set 1: `hp-adapt1-first-set.zip`, especially its TLDR, API cheat sheet and code.
