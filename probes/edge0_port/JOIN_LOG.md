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
- v1 died 12min in on curl-92 (HuggingFace HTTP/2 stream reset at 75%
  of the 10GB pull; default --retry doesn't cover 92). v2 forces
  HTTP/1.1 + --retry-all-errors (resume kept). Infra flake, not code.

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
- Test caught a REAL latent bug: ht probe loops were unbounded
  `while(key!=-1)`; tombstone saturation on small tables spins forever
  (hit at seed-1/dyn-7/event-29). Fixed with bounded scans + tomb
  fallback (behavior-identical on unsaturated tables; old MATCH stands).
  Suite green: 2 passed in 0.83s.

## 2026-09-16: K4 re-sweep prepped (awaiting traces)
- `tracka_k4.py`: atomic lru/static/hyb/lfu/least-stale/pinL/belady on
  fresh ids; CORRECTED latency frame (perf_model, no stale K8/EOVH);
  writes curves + pareto + cache_config_k4.json (winning pins).
- `extract_k4.py`: npz -> ids npy with EXPLICIT quirk checks (no blind
  v3 eval0-drop copy).

## 2026-09-18: Least-Stale correction + Q2K transfer kernel prepped
- User correction 2: old "least-stale" was decayed frequency, NOT SpecMD
  Least-Stale. Renamed to `decayed_lfu` (honest label, kept in sweep).
- TRUE least_stale ported from
  research/native_sparse_experiments/route_trace.py::_online_policy_replay:
  per-bundle EWMA reuse-interval (0.75/0.25, first observation seeds),
  victim = most distant predicted next use, ties -> most recent last-use
  then (layer, expert); atomic adaptation (pre-state scoring, per-request
  `now`, event-key protection). Wired into curves + winner selection.
- tests/test_cache_policies.py (5 tests): hand-computed EWMA victim order
  (evicts A@e7, D-over-C tiebreak@e8, C@e11), multi-key protection case,
  Belady dominance over all online sims (20 seeds, airtight single-key
  events), decayed_lfu==LFU without decay + hand-computed decay flip
  (4 vs 5 hits). Suite: 8 passed (5 new + 2 atomic + 1 gguf).
- Small-Q2K transfer kernel prepped (NOT launched; launches after the IQ2
  re-sweep picks a winner): kaggle/native-sparse-edge0tracek4q2k-v1/,
  8 paired pids [1,4,7,10,13,15,18,21] x N_GEN=80 on a fresh Q2K-experts
  transcode (IQ2 sha-verified in SCRATCH, s-kernel recipe + 120 assert,
  gates read from executed file, Q2K deleted before pull).
- Note: pytest via `| tail` backgrounded and went quiet (both files pass
  in <1s standalone); killed the stale session, no code issue.

## 2026-09-18: K4 traces landed + re-sweep done, RAM point picked
- tracek4 kernel OK in 1628s: 23x82 toks (1886), RMS 0.98-1.02, cov 0.963,
  parity 0.9987-1.0000 (all gates pass). Pull: full-output OOM-killed the
  2GB box (CLI buffers the 11GB blob); re-pulled selectively via
  --file-pattern (23 npz + result.json, 275MB).
- extract: 23 prompts kept (eval0 duplicates pid00 but diverges later ->
  KEEP per explicit quirk policy; 1.2% rows, policy-neutral).
- SWEEP (atomic, test=902 toks): winners pinL2/hyb50/pinL2/hyb25 ->
  3.0:5.07 4.0:6.37 5.0:7.40 5.5:7.93 6.0:8.33 tpsLo
  (tpsHi 5.61/7.23/8.59/9.31/9.87; tpsUp to 10.63 @6GB).
- RAM POINT: 6.0GB / hyb25 (3662 slots, 915 pins, 93.3% hit,
  10.8 miss/tok), LRU as pin-free reference (within 0.5pp).
- TRUE least_stale verdict: collapses (0.085@320; phase5e K8 also
  collapsed 0.099@826) - stale one-hit predictions treated as imminent
  clog the cache. Policy data, not a port bug (core hand-trace tested).
  Frequency family all lose to recency under train->test domain drift.
- BUG FOUND+FIXED by the sweep: belady lost to LRU at small caps
  (protected heap pops discarded, not set aside -> heap drain -> bogus
  unprotected fallback). Fixed (set-aside/restore); regression test =
  exact equality vs O(cap) scan reference (diverged 30/30 pre-fix).
  Fixed belady dominates everywhere (0.603@320 -> 0.956 ceiling).
  Note: hyb75 beats cold-start belady at 6656 (0.988 vs 0.956) - LEGIT:
  static pins are prefetch, belady starts cold. Winners unaffected.
- Atomic-vs-sequential least_stale delta documented in docstring
  (admit-then-evict-self vs protect; [7,3,3,7,1,8..] cap3: 3v2 hits).
- Transfer validation: Q2K kernel v1 PUSHED (8 paired pids); replay
  script ready (replay_q2k.py: winner+LRU on Q2K vs IQ2 + hot Jaccard),
  smoke-tested on IQ2 subset (recompute matches config to 4dp).
- Suite: 10 passed (7 policy + 2 atomic + 1 gguf).
