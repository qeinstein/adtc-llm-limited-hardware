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

## 2026-09-18: Q2K transfer validation PASSES -> lock hyb25 + 915 pins
- Q2K kernel COMPLETE: 120/120 q2k experts (12.26GB, 1534s transcode),
  8x82 toks, RMS 0.96-1.01, cov 0.963, parity >=0.9994 (all gates pass).
- Paired replay (same 8 pids, 3662 slots): winner hyb25 IQ2 0.9169/13.3
  -> Q2K 0.9229/12.3 (+0.6pp, NO degradation); LRU 0.9070/14.9 ->
  0.9148/13.6 (+0.8pp). Earlier IQ2-test-vs-Q2K "drop" was pure prompt
  mix (8-pid subset is harder than the 11-pid test split).
- Pins TRANSFER: hyb25 beats LRU by +1.0pp IQ2-paired, +0.8pp Q2K.
  DECISION: lock 6.0GB hyb25 + 915 pins (cache_config_k4.json).
- Jaccard: paired top512 0.668 (mix-confounded 0.517); mean per-pid
  top256 0.551 - real trajectory drift, yet policy holds.
- tps impact: ~12.3 miss/tok on harder mix -> ~8.2 tpsLo (vs 8.33
  projected); no Q2K-specific downgrade.
- Infra note: kaggle OAuth token expired mid-session; SDK has a 30-min
  past-expiry grace bug (uses dead token); forced refresh_access_token()
  directly. New token good to 2026-09-19 07:45 UTC.

## 2026-09-18: JOIN4 executor built + verified, final bench pushed
- Vehicle: staged (v10-style async, phase10i winner) + static pins +
  section timers + decode/prefill split + TTFT (probes/edge0_port/
  join4_phase6.h, ~1000 lines). One binary, env knobs; resident = env
  OFF (stock mmap, hooks compiled in). -lzm on/off is part of the arm
  definition (functionally required: OFF forces resident load).
- Timers: barrier-flushed per-node attribution (thread 0) into ATTN /
  GDN / MOE-rest(=router+k2) / EXPERT / SHARED / LMHEAD / MISC via
  cb-name markers (substring; names verified "cb-il"); expert nodes
  timed individually, in-node fetch_prep subtracted for compute.
  Decode/prefill by MoE ids rows; exact by construction (every node in
  exactly one bucket). TTFT = first prefill start -> first decode end.
- All 18 patch anchors verified on pin 3057bb6; FULL LOCAL BUILD GREEN
  (llama-cli links, 17.9MB) before push. K1K2 merged (disjoint anchors
  except g_state/extra_buffer, merged into single insertions).
- Pins: 80/80/1346/915 per target, preloaded at init (sim semantics),
  never evicted (violations abort). Decode-split traffic counters.
- Kernel: kaggle/native-sparse-edge0join4-v1/ (self-contained 97KB;
  template + mk injector committed; C+pins round-trip verified).
  5 arms x 23 trace-matched prompts (temp 0.7/seed 1000+pid/N_GEN=80):
  resident x3, b3/b4/b5 x2, b6 x3 + smoke gate (bit-exact outputs AND
  routes resident-vs-b6 or abort). Cold-consistent (drop+settle/run).

## 2026-09-18: JOIN4 v1 failed at smoke, v2 pushed
- v1 post-mortem (from pulled smoke logs, no guessing): (1) parse_perf
  lacked the SUMMARY fallback (CLI prints the summary box only, no
  detailed eval lines); the perf line WAS in the output. (2) Resident
  arm printed no PROFILE line: atexit report was registered only in
  phase6_init, which resident never runs. (3) Perf data: smoke
  resident = 6.17 t/s cold (77s wall incl. cold 12GB file faults).
- v2 fixes: summary fallback + eval fill from dec_graphs + prefill
  from C counters; report registration in the timer hook; cold/warm
  protocol (first run/arm cold, rest warm; headlines = warm steady
  state; per-run drops would have cost ~4h). Regression tests on the
  real failed stdout (tests/test_join4_parse.py; suite 14 pass).
- Local rebuild green; kernel rebuilt (100KB) + pushed as v2.

## 2026-09-19: JOIN4 v2 COMPLETE (bit-exact) + section bug -> join4b
- 5 arms x 23 prompts, sha-gate PASSED (1 output/pid everywhere).
  Warm: resident 5.84, b3 5.26, b4 5.11, b5 5.45, b6 4.95 t/s.
  RSS: 12.07 / 2.37 / 3.33 / 4.28 / 5.23 GB (all under target).
  Miss/tok: 65.9 / 33.8 / 18.3 / 16.9; bytes/tok 68/35/19/17MB.
- Sim validation: bench sits between sim-cold-per-prompt and
  sim-warm-continuous; gap = prefill warmup (uncaptured in traces) +
  easier full mix. Mechanism coherent, capacity gradient matches.
- b6<b5 explained: prompt-noise + disk-jitter outliers (b6-pid15
  53ms fetch on NORMAL misses) + 80-tok cold runs compress arms;
  long sessions favor b6 (10.8 vs 22.6 warm miss/tok).
- TIMER BUG: MISC=106ms (60%)! shexp matmuls run BEFORE their
  "ffn_shexp" opener (source-confirmed) -> all in MISC; TAIL markers
  never match (242 markers/graph = 6/layer exact; force_closes 40).
  Arch: 30 GDN + 10 full + 40 MoE + 40 shexp layers.
- FIX: weight-bucket timers (every matmul by loader weight name) +
  nodelist ground truth + unmatched-weight log. Local build green.
  join4b kernel (resident+b6, 12 prompts): pushed for PERF_MODEL-
  grade sections. v2 tps/RSS/traffic/fetch numbers STAND.
