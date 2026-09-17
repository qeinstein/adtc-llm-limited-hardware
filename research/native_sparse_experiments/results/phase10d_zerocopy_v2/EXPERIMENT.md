# Phase 10D — Zero-copy v2 (MADV_RANDOM): RSS fixed, speed destroyed

## Result

Same matched A/B as v1 + order alternation + smaps proof. Single change:
`MADV_RANDOM` on the file mapping (intended to kill fault-around zombies).

| arm | tok/s mean | ms/token | peak RSS | reps |
|---|---|---|---|---|
| resident | 4.096 | 244.2 | 10,538 MiB | 4.105/4.075/4.107 |
| staged control | 3.002 | 333.2 | 3,518 MiB | 2.948/3.056/3.002 |
| zero-copy | 2.409 | 415.2 | 3,693 MiB | 2.369/2.434/2.425 |

Gain: **-19.7% tok/s (-82 ms/token)**. Verdict: KILL_HARD for this variant.
Exactness still perfect (same response/route hashes as v1).

## What v2 proved (falsification with value)

1. RSS mechanism CONFIRMED: file RSS plateaus (~3.69 GB, slope ~flat at end)
   vs v1's monotonic climb to 4.02 GB. Exit smaps (post-llama-free, our
   mapping only) = 2.19 GB ≈ admitted 2.0 + 0.19 zombies. Fault-around was
   indeed the zombie vector.
2. Readahead + fault-around are LOAD-BEARING (~80 ms): without them, cold
   page faults go fully synchronous (physical reads FELL 8.9→6.1 GB from no
   over-read, yet elapsed ROSE 31→49 s: per-page sync latency dominates).
3. mincore evicted 64/64 EVEN with RANDOM: MADV_DONTNEED zaps ptes (RSS
   obeys) but does not free page-cache pages without pressure; mincore
   reports cache residency, not mapping. mincore-evicted is not a leak
   signal by itself; RSS time-series + smaps are authoritative.
4. madvise_errors = 0: DONTNEED calls "succeed"; the v1 leak was kernel
   speculation behavior, not API failure.
5. Open question (non-blocking): both zc arms show ~0.43 GB more floor-file
   RSS than staged (1.50 vs 1.07 GB). Stable, budgeted, mechanism unknown;
   does not affect the A/B or the 4 GB budget (v3 predicted ~3.5 GB total).

## Decision: MADV_RANDOM KILLED as a blunt instrument

The concept (zero-copy) is NOT killed: v1 measured +11.9% with the leak,
and v2 isolates exactly what must be preserved (speculation) and bounded
(zombie ptes). v3 uses mprotect(PROT_NONE)/PROT_READ eviction/admission:
ptes zapped (RSS tight, fault-around cannot re-map no-access ranges),
readahead + fault-around on admitted ranges fully preserved, pages stay in
page cache (minor-fault re-admission instead of DONTNEED free + major
re-fault). Predicted +12-14% at ~3.5 GB plateau. This is the one allowed
refinement of the 5-10% rule applied to the compliance fix; v3's number is
FINAL for the zero-copy line (keep winner, no v4 iteration).

Raw v2 outputs: /tmp/zc_results_v2 (Kaggle download).
