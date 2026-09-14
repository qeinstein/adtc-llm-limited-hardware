# Phase 9B — Bounded-tax decomposition and zero-copy bound (offline + local microbench)

## Staged decomposition (committed Kaggle evidence)

Total staged cold: 312.7 ms/token (3.198 tok/s, phase8c).

- Expert GEMV core: ~87 ms (phase5a resident 210.7 vs expert-removed 123.7)
- Bounded-vs-resident tax: ~75.5 ms (staged warm 301.1 vs resident mmap
  control 225.6 = 1000/4.433, phase7b; same patched binary both arms —
  resident only flips `-lzm off` and pops `GGML_PHASE6_CACHE_BYTES` —
  so the gap is pure bounded-machinery cost)
- Removable physical I/O: ~11.6 ms (phase8c staged cold-warm)
- Remaining runtime: ~138 ms

Gate+up activation sharing is NOT automatically next: it must prove
>15.6 ms system saving or enable a larger architecture change. Phase 4
evidence (activation loads ~5.7% of fused expert cycles, ~5 ms) suggests
it does not clear the bar alone — parked pending its own bound.

## Per-miss cost structure (phase8c patch, serial `phase6_load`)

~145 misses/token x 3 pread slices (gate/up/down, ~292 KB each) = 435
pread calls moving ~127 MB/token page-cache→slot. No separate memcpy —
the copy hides inside pread. Plus per miss: O(2281) reserve scan,
`MADV_DONTNEED` on eviction, pthread create/join (staged), ready-spin.

## Local copy-rate microbench (`/tmp/zc_bound.py`, scratch, removed)

435 x 292,181 B page-cache preads = 127.1 MB in 44.3 ms (2.87 GB/s);
127 MB memcpy in 52.3 ms. Local CPU, not Kaggle — order-of-magnitude only.
Implication: the copy term is ~40-50 ms of the 75.5 ms gap, not 13-25 ms.

## Zero-copy Amdahl bound

Optimistic ceiling for removing the whole bounded tax: 75.5/312.7 = 24.1%.
Copy elimination alone: ~40-50 ms (~13-16%). Plus scheduling hygiene
(persistent workers, scan removal): +5-10 ms. Realistic prototype range:
~45-60 ms, 14-19% end-to-end → EXPLOIT territory if measured, with a
clear path to test.

Design: page-aligned contiguous expert sidecar if needed (one bundle =
876,544 B = exactly 214 x 4096 B pages); consume bytes directly from
file-backed mappings; bound working set with madvise/mincore/smaps; evict
with `MADV_DONTNEED`; optionally `mmap(MAP_FIXED)` file offsets into fixed
virtual slots instead of copying. Preserve exact K=8/routes/weights/output;
measure RSS and page residency carefully. The phase8b physical-placement
KILL does not invalidate this: that targeted range/traffic reduction, this
targets the ownership/copy/indirection tax.

## Strategic model update

Erasing expert GEMV (87) + bounded tax (75.5) + physical I/O (11.6)
completely still leaves ~138 ms/token, while 15 raw tok/s needs
66.7 ms/token. So 15 raw on Kaggle cannot come from expert/storage alone:
broad runtime/representation/multi-token redesign is required alongside.

## Decision

**FOLLOW-UP ONCE = build the zero-copy prototype as the next Kaggle
experiment** (largest single remaining measured win, 14-19% realistic).
Kill criteria: <3% measured end-to-end → KILL; 3-10% → one refinement;
>10% → EXPLOIT. Gate+up fusion stays parked pending its own >15.6 ms proof.
