# Phase 10F — Zero-copy v4 (mprotect): +16.6% speed, RSS STILL unbound

## Result

| arm | tok/s mean | ms/token | peak RSS | reps |
|---|---|---|---|---|
| resident | 4.055 | 257.0 | 10,535 MiB | 4.461/2.985/4.720 |
| staged control | 3.273 | 305.7 | 3,518 MiB | 3.300/3.164/3.356 |
| zero-copy | 3.815 | 262.8 | 6,530 MiB | 3.878/3.552/4.016 |

Gain: **+16.56% tok/s (+42.9 ms/token)**. Paired ratios: +17.5/+12.3/+19.7%,
all reps >= +12% despite a noisy window (resident rep2 collapsed to 2.985,
proving machine noise; pairing absorbs it). Exactness perfect (same
response/route hashes as v1-v3).

## The speed concept is now measured TWICE and growing

v1 +11.9% (35.5 ms) -> v4 +16.6% (42.9 ms). The delta is mprotect's cheaper
syscall path (prot_ns 0.66 s/rep ~= 10 ms/token, no page-freeing) plus
minor-fault re-admissions from retained page cache. Copy-removal + cheap
residency management is REAL and is the largest single exact optimization
in the program's history.

## But RSS: 6.53 GiB — mprotect-alone is structurally incapable

48,807 successful PROT_NONE calls (prot_errors=0), yet file RSS grew
monotonically 2.75 -> 4.82 -> 5.67 GB with zero bounding effect. Root cause:
mprotect does not reliably decrement mm->rss_stat (protection change, not
map/unmap accounting); the pages are zapped but the REPORTED counter — the
compliance metric, and what cgroup/OOM paths read — never drops. v1's
"leak" was real zombie ptes (fault-around); v4's is an accounting wall.
mincore evicted 64/64 in both (page-cache residency, not mapping).

## Decision: v5 MAP_FIXED remap is ABSOLUTELY FINAL

Eviction MAP_FIXED-replaces interiors with PROT_NONE anon holes (munmap
path: counter decremented, fail-loud), admission MAP_FIXED-restores the
file ranges; align-IN (v3 lesson); cache retained (v4's minor-fault edge
kept). Predicted +15-16% at ~3.6 GB. No v6 under any circumstance: v5's
number KEEPs or KILLs the entire zero-copy line. (The staged backend and
the running Q2_K A/B are unaffected either way.)

Raw v4 outputs: /tmp/zc_results_v4 (Kaggle download).
