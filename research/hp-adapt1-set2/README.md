# HP × Adapt-1, set 2

This directory carries out the experiment plan in [`plan/EXPERIMENT_PLAN.md`](plan/EXPERIMENT_PLAN.md) (the 2026-11-06 handoff package, copied unmodified). Results, gate decisions and deviations from the plan are in [`RESULTS.md`](RESULTS.md).

It is self-contained research code, unrelated to the trading code in the rest of this repository. It uses the Python 3.13 standard library only, with no third-party packages.

## Layout
| Path | What it is |
|---|---|
| `hpx/client.py` | The safe Adapt-1 client (plan §2.4). It keeps a write-ahead journal, applies the documented retry class per operation, and reconciles before any retry. It honors a STOP file and allows at most 2 requests in flight and one fit at a time per owner. It checks the build and payload sizes, binds model identity, touches only `hpx2-` domains and redacts the account identity |
| `hpx/hpenv.py` | An incremental 2D/3D HP fold. In 3D its observations and features are byte-identical to set 1's `hp.observation` and `State.features` |
| `hpx/planner.py` | C-plan: client-side depth-d MAX search that batches leaves across folds. Evaluators: heuristic, neutral, learned model, leaf blend |
| `hpx/targets.py` | C-target: blended targets with provenance, sibling rows, set 1's continuation sibling rows, validation splits |
| `hpx/suites.py`, `hpx/oracle.py` | C-eval: frozen suites (`suites/`, sha256 in `suites/MANIFEST.json`) and an exact 2D/3D oracle |
| `hpx/stats.py` | Bootstrap CIs, Wilcoxon signed-rank, Holm correction, the plan §2.5 decision rule |
| `hpx/game.py` | E0: 5×4 gravity four-in-a-row, with an exact solver, line heuristic, object-graph encoding and minimax planner |
| `hpx/mock.py` | A mock of the API routes used here, mirroring the response shapes recorded on af6670b, with fault injection |
| `experiments/` | One script per stage: `t02_probe`, `t03_calibration`, `t04_parity`, `a1_representation`, `a2_fit_settings`, `a3_targets`, `a3b_own_policy`, `g1_decide`, `b1_lineages`, `c1_compute`, `d1_twod_short`, `e1_game`, plus the S0 builders `s0_build_suites` and `s0_baselines` |
| `runs/<exp>/<stamp>/` | Per run: `manifest.json`, `results*.json`, `log.txt`, `journal.jsonl` and saved API responses |
| `tests/` | `python3 -m unittest discover tests` (about 3 minutes; mostly oracle builds) |

## Running
```bash
python3 -m unittest discover tests
```

Dry-run any stage against the mock. The output goes to `runs/_mock/`, which git ignores:
```bash
HPX2_MOCK=1 python3 experiments/a3_targets.py 1
```

The real API is reached through the environment's egress proxy, which supplies the key. Nothing in this directory holds a credential.
```bash
python3 experiments/a3_targets.py 1
```

To stop a run cleanly, create a `STOP` file in its run directory or in this directory. Domains created by a run are deleted at the end of that run, and deletion is verified with a 404. `runs/live_domains.json` lists any still alive.
