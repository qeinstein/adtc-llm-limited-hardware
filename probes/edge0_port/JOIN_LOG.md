# JOIN log (K4/16 + Q2_K deployment path)

## 2026-09-16: trace kernel launched
- `kaggle/native-sparse-edge0tracek4-v1`: v3-identical except K1K2 graph
  patch + K-param ids hook + offline top-4 + K1=4/K2=16 env. TRUE
  on-policy K4/16 trajectories (k2 affects mixtures, hence hiddens).
- Same 23 prompts + N_GEN=80 as v3 (clean K8-vs-K4 comparison).
- IQ2 capture (gates byte-identical across files; trajectory drift 2nd
  order vs the K8->K4 1st-order shift). Redo on Q2K iff marginals say so.
- Status: RUNNING (~2h). Monitor: kaggle.com/code/toheebogunade/
  jamii-native-sparse-edge0tracek4-v1

## 2026-09-16: atomic K-event semantics (JOIN-2)
- Old sims (C/Python/Track-A) were per-request sequential: within one
  (token,layer) event an early miss's admission could evict a later
  same-event hit (spurious miss). Python=C MATCH proved equivalence,
  not runtime-correctness.
- Implemented `edge0_cache_event` (C) + `cache_atomic.sim_atomic`
  (Python): score all K from pre-event state, touch hits, admit misses
  with current-event keys protected (victim walk; degenerate cap<=K:
  plain tail). Corner test: sequential 0/4 vs atomic 1/4 (C + Python).
- Randomized C==Python cross-check (5 seeds x caps/pins) in
  tests/test_cache_atomic.py; C driver cache_atomic_test doubles as the
  JOIN-3 cross-check tool. Empirical atomic-vs-seq delta reported by
  the K4 re-sweep itself.

## 2026-09-16: K4 re-sweep prepped (awaiting traces)
- `tracka_k4.py`: atomic lru/static/hyb/lfu/least-stale/pinL/belady on
  fresh ids; CORRECTED latency frame (perf_model, no stale K8/EOVH);
  writes curves + pareto + cache_config_k4.json (winning pins).
- `extract_k4.py`: npz -> ids npy with EXPLICIT quirk checks (no blind
  v3 eval0-drop copy).
