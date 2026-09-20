# S2 baseline: noDN promoted + 48-slot default (2026-09-15)

Parent frame: 215.2 ms/token, 4.65 tok/s (phase4 resident anchor: 4.650 tok/s,
215.06 ms, 10.5 GiB RSS — zero staging cost in it). Sprint goal: >15 tok/s,
<3 GiB RAM, accuracy invariant. Box here: Intel N100, 2.7 GB RAM; model is
9.9 GB so e2e is unmeasurable — projections below are harness-measured
savings applied to the adopted frame, per parent instruction.

## 1. Promotion edit (in-tree, uncommitted — parent handles git)

File: `kaggle/native-sparse-staged-q2k-v1/staged_q2k_v1.py` (v10 staged
executor, the `stage_replay` reference; 14+/8-, see `promotion.diff`).

- (a) `phase6_reserve_slot`: deleted the anonymous-staging
  `madvise(..., MADV_DONTNEED)` block (old lines 488-493). Refill keeps
  overwrite (sync `phase6_read_exact` / async `phase6_copy_read` pread into
  the same slot). New lines ~495-499 carry the noDN comment.
- (b) Slot default: old line 60-61 `CACHE_BYTES = 2_000_000_000` (opaque;
  2281 slots = 57/layer IQ2) → new `CACHE_BYTES = 48*40*876544 =
  1,682,964,480` with named constants (`STAGING_SLOTS_PER_LAYER_DEFAULT=48`,
  midpoint of the 28-64 KEEP range). Runtime yields exactly 1920 slots
  (48.0/layer) for IQ2 and ~1630 (~40.8/layer) for +17.75% Q2K slices —
  both in-range. Reservation drops 2.00 → 1.68 GB (-317 MB, helps <3 GiB).
- Deliberately untouched: zero-copy `phase6_zc_evict_range` DONTNEED (bounds
  file-backed RSS, different mechanism); `N_THREADS` (already capped stock
  `-t 4`); kernels (file stages bytes only, stock ggml GEMV); historical
  snapshots v1-v11/zerocopy (parent can cherry-pick the one-liner);
  `probes/hw_agent3_memory/stage_replay.c` (harness keeps both policies —
  it is the measurement tool, not the promoted path).

## 2. Measured on this box (real L00 IQ2 bytes + real ggml AVX2 kernels)

Rebuilt harness: `stage_replay_s2` (same source/objects/flags as
`run_stage.sh`); 4 runs at the NEW 48-slot default, seed 12346 (= parent r1
seed). Logs: `results/stage_*_s48_*_s2.txt`.

| trace | base manage_med | noDN manage_med | tok_med base→noDN | minflt/tok base→noDN | checksum base==noDN |
|---|---|---|---|---|---|
| uniform/48 | 3.490 ms | 0.797 ms (4.4x) | 6.949→4.429 ms | 1179→103 (med 1070→0) | `d49b6cbc…` PASS |
| zipf-1.1/48 | 1.413 ms | 0.379 ms (3.7x) | 5.067→4.002 ms | 505→103 (med 428→0) | `3dda4bad…` PASS |

Checksum prefixes match the parent's 52-run values → rebuild cross-validated,
and slot-count 48 is exact-output vs 16/28/64. noDN residual minflt (10279 =
cold first-touch of the 42 MB store, med 0) → zero steady-state faults.
(One noDN/uniform run shows 5 majflt — VM noise; checksum identical.)
GEMV unchanged across policies (~3.4-3.6 ms), as expected.

tlb_micro re-runs (2x, `results/tlb_micro_s2.txt`): per-miss saving
525.6 us (16.2x) and 497.5 us (14.4x) vs report 465.7 us (14.8x);
minflt 214.0/iter and pagemap 1712/1712 exact both runs. Ratios robust
(absolute wall swings 2-5x on this VM per hw_profile §4).

Sync-staging saving per layer @48: uniform 2.69 ms → x40 = **107.7 ms/tok**;
zipf 1.03 ms → x40 = **41.3 ms/tok**. Real-route LRU sim (parent,
`results/lru_sim.txt`): decode uniq-miss mean **92.1/tok** → 92.1 x
~466-526 us ≈ **43-48 ms/tok**. Compute floor: dense 47.5 + expert GEMV
63 (4T spin) = **~111 ms/tok** (hw_profile REPORT §§1,3).

## 3. S2 baseline verdict table (PROJECTED except row 1)

| # | scenario | ms/token | tok/s | status |
|---|---|---|---|---|
| 0 | Prior frame (phase4 resident anchor, Kaggle-measured) | 215.2 | 4.65 | MEASURED (other box) |
| 1 | S2 realistic: frame − real-miss saving (92.1×466us ≈ 43 ms), fully exposed sync | 172 | 5.8 | PROJECTED |
| 2 | S2 best case: compute floor (staging fully removed; uniform sync bound 108 ms would undercut it) | 111 | 9.0 | PROJECTED (floored) |
| 3 | S2 lower bound: async already hides all management | 215.2 | 4.65 | PROJECTED (no gain) |
| 4 | Staged-vehicle-relative: slow-box staged control 328 ms (3.046) − 43..108 ms | 220-285 | 3.5-4.5 | PROJECTED (mixed boxes) |

Methodology notes: (i) rows 1-2 apply N100-harness sync savings to a
Kaggle-box frame — planning bounds, not measurements; (ii) the 215.2 frame
is resident (no staging); the gain lands on the staged vehicle (row 4 is
the like-for-like view, but mixes boxes); (iii) production v10 async may
already hide the refill, leaving only the sync madvise (~28-35 us/miss ≈
2.5-3 ms/tok at 92 misses) — row 3 covers that; **measure e2e on the
10 GB box**; (iv) gap to >15 tok/s (66.7 ms) remains even at best case —
needs arch (representation/compute-removal), not host, work.

## 4. RAM / parity / config

- RAM: 48-slot window = 42,074,112 B measured (`store_bytes`); production
  reservation 1.68 GB (was 2.00 GB). Slots stay resident (bounded, tiny
  delta). <3 GiB needs e2e RSS check on the 10 GB box (dense resident + 1.68
  GB cache) — UNRESOLVED here.
- Parity: PASS — base==noDN checksums both traces, matching parent prefixes.
  E2E response/trace-hash gate must re-run on Kaggle (not runnable here).
- Config: noDN evict + 48 slots/layer + stock `-t 4` + stock ggml IQ2 kernels.
  Nothing else touched: `git diff --stat kaggle/` = 1 file (this executor).
