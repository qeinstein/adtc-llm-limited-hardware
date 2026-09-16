# Systems notes: cache policies + fetch/overlap plumbing (outcome-independent)

Date: 2026-09-16. No quality assumption: all results hold regardless of
the Q2_K-all/Qwen3.6 gate outcomes. Tools: `cache_sim.py`,
`cache_curves.json`, `stage_overlap.c` (build via `build_phase1.sh`).

## 1. Pinned-hot + LRU beats pure LRU 5–12% (real 2016-token corpus)

Misses/token (K8, IQ2 bytes; train/test-split honest + same-split ref):

| slots | LRU | pin25+LRU | pin50+LRU | static | pinL2/L4 | Belady |
|---|---|---|---|---|---|---|
| 1024 | 148.0 | 146.9 | 144.5 | 190 | 147 | — |
| 1792 | 105.6 | 100.9 | 97.5 | 131 | 100 | — |
| 2664 | 73.0 | 70.7 | 67.8 | 84 | 70 | 34 |
| 3584 | 51.1 | 47.2 | 44.8 | 62 | 46 | — |
| 4608 | 29.1 | 28.6 | 28.3 | 42 | 27 | — |
| 5632 | 17.7 | 16.3 | 15.8 | 30 | 16 | — |

TRAIN-split ≈ ALL-split (popularity generalizes; no overfit). Static-only
loses everywhere (can't adapt); hybrid wins. Per-layer pin ≈ global pin.
**Adopt pin50+LRU** (free at boot; also more predictable). Miss burstiness:
p95 ≈ 2× mean — size the prefetch pipe for bursts, not the mean.
Policy is a 10% lever; slots and overlap are the big levers.

## 2. Real-trace staging replay validates the sims exactly

`stage_overlap sync` (noDN clock slots, real corpus slice [64:112],
cold start): 64 slots → 0.0000 hits, 256 → 0.0000, 512 → 0.2999.
Python LRU sim on the SAME slice: 0.0000 / 0.0000 / 0.3023. **Match**
(clock-vs-LRU gap 0.8%). The Pareto cache model is now validated in real
code, not just Python (fills the prior "rerun with REAL route trace"
open item). Management wall (memcpy mode): ~0.15–0.17ms/miss for 0.86MB
bundles (~5GB/s effective incl. clock scan).

## 3. Oracle-overlap bound: 96% of fetch hidden (lookahead-1 + headroom)

Direct-to-slot async prefetch, perfect corpus lookahead, 512 slots K8,
1GB/s-SSD analog (memcpy + calibrated 0.86ms/miss delay):
sync wall 419ms/tok (gemv 206 + fetch 212) → async wall **163ms/tok
(gemv 154 + stall 8)**. Hidden fraction **96.2%**. Sync-vs-async outputs
bit-identical in every run (pin protocol; no torn reads). mem mode: stall
1.8ms (fetch fully hidden, wall ≈ gemv, no thread contention).

## 4. Four measured design mandates (each found by a failure)

1. **Parallel fetch is REQUIRED**: a single serial prefetcher (0.86ms/miss)
   is producer-bound below consumer pace and hides nothing (async runs
   hung/timed-out). 4 fetch workers sustain 4× consumer pace. Matches
   Edge0's prefetch_threads=4 from first principles.
2. **Prefetch DIRECTLY into slots**: a side staging buffer costs a 0.86MB
   promote-copy per hit (measured 48–71ms/tok!) + 420MB RAM + a two-tier
   eviction pathology (82% hidden max vs 96% direct). Edge0's staged-slot
   design agrees; the copy is pure loss.
3. **Headroom rule**: prefetch needs slots ≳ 2× in-flight
   ((look+1)×320 + margin). Violations measured: 256-slot async1 → 52ms
   stall (vs 8ms at 512); async2 at 512 (960 in flight) → **251ms stall,
   WORSE than sync** (producer/consumer eviction thrash). Production
   2000–5000 slots has ample headroom for lookahead 1–2.
4. **Bounded spin (5ms)** on in-flight keys: spinning longer than a fetch
   is pure loss (100ms spin caused 8× slowdowns); shorter causes duplicate
   fetches (measured SSD read-amp ~1.3× — hidden, acceptable).

## 5. Secondary measurements

- Cold-SSD could NOT be measured locally (128MB blob stays page-resident;
  fadvise-DONTNEED ineffective in this VM; pread timed 46GB/s = RAM).
  Kept the analytic 1GB/s (conservative SATA-class) + delay-injection
  analog; target-box SSD stays a model parameter (0.5–3GB/s sensitivity).
- Sidecar (1 pread/miss) vs GGUF layout (3 preads/miss): 8% apart on SSD
  (transfer dominates seek). Sidecar preferred but NOT load-bearing.
- Overlap × Pareto: 0.9 overlap lifts the top ≤6GB config 7.71→9.64
  i5-LOW (11.6 HIGH); 0.96 → 9.80/11.9. Overlap is worth ~2 tok/s at the
  top end. Deployable prerouter < oracle — Phase 3 prices recall.
- Cold-start: first 2 tokens carry ~2× stall (16–18ms vs 8 steady) —
  prime the pipe during prefill in production.
- RSS observed: 728MB max (512 slots + pbuf-era runs); direct-slot
  production sizes are modeled, not materialized, on this 2GB box.

## 6. What this unlocks

Phase 3's prerouter plugs into a PROVEN pipe: reservation protocol,
worker pool, headroom rule, spin budget, and the 96% oracle bound are
all measured. Remaining Phase-3 question is purely ML: prediction recall
× CPU overhead — the systems side is de-risked.
