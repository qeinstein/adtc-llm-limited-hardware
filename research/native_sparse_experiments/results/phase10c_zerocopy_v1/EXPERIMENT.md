# Phase 10C — Zero-copy bounded execution v1 (Kaggle A/B)

## Result

Same binary, matched arms (resident / staged-v10-verbatim / zero-copy),
3 reps, 64 tokens, 2 GB cache, cold page cache per run:

| arm | tok/s mean | ms/token | peak RSS | reps |
|---|---|---|---|---|
| resident | 4.000 | 250.1 | 10,535 MiB | 4.063/4.011/3.925 |
| staged control | 2.954 | 338.6 | 3,518 MiB | 3.000/2.921/2.942 |
| zero-copy | 3.306 | 303.1 | 4,116 MiB | 3.448/3.370/3.101 |

Gain: **+11.93% tok/s (+35.5 ms/token)**. Paired per-rep ratios:
+14.9/+15.4/+5.4%. Exactness: single response hash AND single route-trace
hash across all 9 runs. Cache behavior identical (9275 misses, 6994
evictions both arms). ZC eliminated all pread (read_bytes 0 vs 8.13 GB,
rchar 0.02 vs 8.15 GB) at equal physical I/O (8.9 vs 8.7 GB).

## Verdict: EXPLOIT-conditional — speed clears the bar, RSS does not

+11.9% exceeds the 10% EXPLOIT threshold, BUT peak RSS is 4,116 MiB
(4.02 GiB): over the 4 GiB budget, and the time series shows file RSS
growing monotonically through the run (1.4→3.6 GB, still climbing at t=30s)
while staged plateaus flat at t=10s. A longer generation would blow past
further. v1 does NOT qualify as a <=4 GiB frontier point.

## Root cause: MADV_DONTNEED leaks (fault-around zombies)

- mincore: 64/64 sampled evicted slices still resident.
- madvise cost is real but fine: 0.69–0.82 s/rep (~11 ms/token).
- Leak rate ~86 MB/s ≈ 20% of admission rate. Best mechanism fitting all
  evidence: the kernel's fault-around window (64 KB default) speculatively
  maps evicted pages adjacent to admitted slices into our page tables
  (2×64 KB per slice edge / 876 KB bundle ≈ 15–20%). DONTNEED zaps correctly;
  fault-around re-seeds zombie ptes without re-admission. Readahead also
  repopulates page cache (explains mincore 64/64).
- Ruled out: wrong ranges (exactness proves read math; eviction uses
  identical math), floor growth (zc-specific code never touches llama's
  mappings), unchecked-EINVAL (ranges valid by construction).

## Secondary findings

- Within-rep drift: all arms decline rep1→rep3 (thermal/neighbor). ZC ran
  LAST in every rep → systematically disadvantaged; true effect likely
  above +11.9%. v2 alternates arm order.
- v1 fell short of the 4.1–4.3 tok/s prediction (got 3.31): the copy term
  is ~40% of the staged tax, not ~70%; residual zc-vs-resident gap (53 ms)
  includes madvise (11), LRU scans, faults/TLB. Model updated, not broken:
  copy-removal is confirmed real (+35 ms) but smaller than the warm-copy
  extrapolation suggested (staged parallel preads copy faster than the
  serial 1.33 GB/s rate used in the prediction).
- ZC variance (±5%) exceeds staged (±1.5%): TLB-shootdown/readahead
  sensitivity. v2 should stabilize via MADV_RANDOM.

## Refinement (v2): ONE-LINE fix + proof

`madvise(MADV_RANDOM)` on the file mapping at init disables readahead AND
fault-around. v2 arms: resident + staged + zc-random, order-alternated,
plus madvise error counting and exit-time smaps_rollup for RSS-compliance
proof. Expect: file RSS plateau ~3.1 GB, mincore evicted ~0, tok/s ≥ v1.
If MADV_RANDOM fails → fallback is MAP_FIXED anon-replace eviction
(guaranteed unmap). Raw v1 outputs: /tmp/zc_results (Kaggle download).
