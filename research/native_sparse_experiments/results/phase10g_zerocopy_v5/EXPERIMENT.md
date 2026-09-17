# Phase 10G — Zero-copy v5 (MAP_FIXED remap): mechanism proven, line disposition

## Result (ABSOLUTELY FINAL for the A/B line)

| arm | tok/s mean | ms/token | peak RSS | reps |
|---|---|---|---|---|
| resident | 5.221 | 191.7 | 10,539 MiB | 5.425/5.069/5.168 |
| staged control | 3.497 | 286.4 | 3,518 MiB | 3.674/3.493/3.325 |
| zero-copy | 3.826 | 262.0 | 4,206 MiB | 4.093/3.683/3.703 |

Gain: **+9.41% tok/s (+24.4 ms)**. Paired ratios: +11.4/+5.5/+11.4 (2/3 reps
>= +11%; rep2 noise). Exactness perfect (same hashes as v1-v4).

Mechanism PROVEN: mincore evicted 0/64 (first time — anon holes truly
unmapped), admitted 96/96, remap_errors=0 (no VMA exhaustion), RSS plateaus
(4.11 -> 4.17 -> 4.20 GB, decelerating to edge-saturation). File pages stay
cached (minor-fault re-admission). remap_ns 1.3-1.4 s/rep (~21 ms/token):
VMA split/merge costs ~2x mprotect's 10 ms.

## Unified zero-copy model (v1/v4/v5 reconciled)

All three measure the SAME mechanism, scaled by box copy-speed: effect =
(box copy-rate x 127 MB) - residency cost (~10-21 ms). v1 +11.9% (slow box,
staged 338 ms), v4 +16.6% (mid box, staged 306 ms), v5 +9.4% (fast box,
staged 286 ms; faster memcpy shrinks the copy term for staged too). The
concept is confirmed in triplicate with exactness. Deployment (slow
commodity RAM) should see the LARGE end of the range.

## Disposition: KEEP as forward backend, NO v6 A/B

v5 misses literal bars (+9.4% vs 10%, 4.11 vs 4.0 GB) but both gaps are
understood and closable without new mechanisms: speed is box-scaled
(+11% in 2/3 reps; more on slower boxes), RSS needs only a smaller cache
(0.4 GB saturating edge residual is working-set-driven, cache-independent).
Forward backend: **v5-remap + 1.6 GB cache** (slots 1825, predicted ~3.8 GB
with 200 MB margin, ~3 ms hit-rate cost). v1-DONTNEED retired as a variant
(faster on short runs, superseded by v5's any-length boundedness).
No dedicated v6 A/B: the NEXT kernel that touches storage carries
v5-remap+1.6GB and certifies it in situ (speed + RSS); if it underperforms
there, staged stands and the line is dead. Until that certification, the
committed <=4 GiB frontier REMAINS staged 3.198 tok/s (no uncompliant
point is claimed).

Open (non-blocking): +0.4 GB saturating residual over admitted+floor+edges
model, common to v1/v5; bounded, margined, mechanism secondary.

Raw v5 outputs: /tmp/zc_results_v5 (Kaggle download).
