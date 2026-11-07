# Second-session (uat2) notes

All times UTC, 2026-11-07.

- 19:03 Key check: the Fast-UAT key on environment `nofx-uat2` maps to engine session `68b3f2be63632b2ee95a20b0` (the
  original one); `/version` commit `af6670b8a9c5ce4251caa02940118110adb839ea`, schema 5.
- 19:08 The first session reports its Fast-UAT key returns 401 since 19:04; this session becomes the only Fast-UAT
  client. User confirmed (19:13) that all Fast-UAT work (A, C, then F, G) runs here (PREREG operational note 5).
- 19:15 Merged `origin/claude/zen-carson-n5bvi0` (d9199cc).
- 19:15 Deletion of the 8 old Domains by exact id (ib-p50-s{0,1,2}, ib-p50d-s{0,1,2}, bench-pensim-ref-{linear,shared}):
  every DELETE returned 404 and every readback 404. They do not exist under this key's engine session; nothing was
  deleted by this session. (They were created under the first session's former key, which now returns 401.)
- Host: 4 CPUs, 15 GB RAM.
- 19:20 Setup: one combined venv is unresolvable (pensimpy 1.2.5 pins a numpy that does not build on Python 3.13), so
  two venvs: `venv-ap` (Industrial Benchmark, PenSim: pensimpy and industrial-benchmark-python installed `--no-deps`,
  plus numpy, scipy, pandas, scikit-learn, cma, gym 0.26.2, gymnasium 1.2.0, matplotlib, torch, tqdm as imports
  required) and `venv-fg` (MiniHack, Craftax: the pinned F/G set). SMPL cloned at c816917; shims from `vendor_shims/`.
- 19:24 Smoke tests, no API: IB episode OK; PenSim base recipe OK, 18 s per batch, quality 0.656 / 0.212 / 0.839 /
  0.318 / 0.261 on random_seed_ref 0-4 (large per-batch spread; the first attempt's held-out mean was 0.499);
  MiniHack potion / lavacross / wod_medium one random episode each OK; Craftax one 50-step episode OK.
- 19:26 Launched A (six Industrial Benchmark runs, 150 episodes) and C (PenSim refinement, linear and shared, 20
  updates, radius 0.2) together: 8 Fast-UAT clients.
