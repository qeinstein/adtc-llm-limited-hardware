# AGENT 3 — Memory/TLB/staging layout: REPORT (parent-run; agent stalled)

Method: the stalled agent left a complete `stage_replay` matrix (52 runs),
`tlb_micro`, and `transcode` with outputs but no verdicts. The parent
re-ran `tlb_micro` and `transcode` (reproduced to ~1%), aggregated all 52
stage runs, and wrote the verdicts below. All GEMV is real ggml AVX2
kernels on real L00 IQ2 bytes. Single-layer (L00) sync staging; traces are
synthetic (uniform = worst case, zipf-1.1 = skewed). Checksums match across
ALL policies/slots per trace -> staging policy is exact-output.

## 0. Verdicts

- **KEEP: drop `MADV_DONTNEED` from the staged evict path** (one line, zero
  risk, exact-output). It is ~80% of sync-staging management cost.
- KILL: 2 MiB huge pages / THP for staged slots (this box class).
- KEEP (config): larger slot files (28-64) + compact slot reuse.
- **KILL_HARD: transient Q8 execution cache** (12x slower than the GEMV it
  would accelerate, not hidden by overlap, outputs DIFFER, 25-51 MB).

## 1. The 12-item report (per probe)

### Probe A — MADV_DONTNEED on evict vs overwrite (noDN)

1. CPU: Intel N100, 4C/4T, KVM, THP `never` (see profiler REPORT for full topo).
2/3. Baseline: adopted 4.65 tok/s / 215.2 ms (ST single-layer harness numbers
   below are per-layer; scale x40 only for sync-staging bounds — production
   may overlap staging, see item 12).
4. Diagnosis: `tlb_micro` isolates one 876544 B bundle miss: DONTNEED path =
   **499.5 us** (27.6 madvise + 471.8 refill memcpy, **214 minflt**);
   overwrite = **33.8 us, 0 faults** — **14.8x cheaper**. Even cache/TLB-cold
   overwrite is 123.9 us (4x cheaper). Stage matrix confirms at scale, e.g.
   uniform/16 slots: manage_med 4.76-5.73 ms (base) vs 1.02-1.16 ms (noDN),
   minflt/tok 1541 vs 34. `pagemap`: 8-slot window = 1712/1712 pages, 6.69 MB.
5. Change: delete the `MADV_DONTNEED` call on slot evict (keep overwrite).
6/7. New tok/s / ms: strictly better; magnitude = currently-EXPOSED
   sync management on the 10 GB box (unmeasurable here). Upper bound from
   sync replay: uniform trace saves ~150-185 ms/token; skewed ~100 ms/token.
   (Bounds, not promises: production v10 may already overlap part of this —
   measure e2e after the one-line change.)
8. Expert ms/token: management share of the expert region falls ~5x wherever
   staging is exposed; GEMV itself unchanged (~3.3 ms/layer ST all policies).
9. RAM: slots stay resident: +14 MB @16 slots (base stride). Bounded, tiny.
10. Parity: **PASS** — FNV checksum identical for base/noDN/huge/hugeND/noslot
    within each trace (`3dda4bad` zipf / `d49b6cbc` uniform, all 52 runs).
11. **KEEP** — exact-output, one line, zero risk, strictly positive direction.
12. Stacks: **yes** — orthogonal to every arch change (pure host-cost removal).

### Probe B — 2 MiB huge pages / THP for staged slots

Huge vs base are **byte-identical on faults** (e.g. 1541/1541, 1023/1023)
and within noise on time across all 12 cells: `MADV_HUGEPAGE` formed zero
hugepages (THP=`never`, AnonHugePages=0). Stride 2 MiB wastes 2.4x bytes
(33.5 vs 14 MB @16 slots) for zero gain. **KILL** on this box class.
Revisit ONLY on a THP-enabled host, and only after noDN (faults are already
~0 without DONTNEED, so the ceiling is tiny). RAM: negative (waste).

### Probe C — slot-file sizing (16 / 28 / 64) and slot reuse

More slots -> fewer misses -> less management (uniform base: 4.76 -> 4.21 ->
2.96 ms @16/28/64; hitrate 0.10 -> 0.19 -> 0.44; skewed hr up to 0.77).
LRU sim agrees (decode uniq-miss mean 92/token over 63 groups).
Locality bonus: infinite-slot `noslot` GEMV is **4.89 ms vs ~3.3 slotted**
(224 MB span thrashes LLC) — compact slot reuse wins twice. **KEEP** as
config guidance: 28-64 slots if the per-window RAM budget allows
(56 MB @64 base-stride single-layer window). Exact-output (checksums match).

### Probe D — transient Q8 execution cache (transcode 8-16 experts)

Reproduced: transcode-8 = 40.36 ms vs IQ2-GEMV-8 = 3.35 ms (**12.03x**);
overlap test `hidden=NO` (bg 40.4 ms vs fg 3.6 ms); Q8 GEMV only 1.13x
faster than IQ2 GEMV (no prize even if transcode were free); outputs
**DIFFER** (`maxreldiff=13.01`, checksums differ) so it is not even an
exact-output change; cache costs 25.5 MB (8) / 51 MB (16).
**KILL_HARD** on all three axes (perf, parity, RAM).

## 2. Condensed stage matrix (tok_med ms, r1/r2; man=manage_med; hr=hitrate)

uniform/16: base 8.18/10.23 (man 4.76/5.73, hr .10) | noDN 4.42/5.57 (1.02/1.16)
uniform/28: base 7.50/10.06 (4.21/5.24, hr .19)    | noDN 4.17/5.42 (0.93/0.96)
uniform/64: base 6.17/8.08  (2.96/3.77, hr .44)    | noDN 3.92/5.31 (0.66/0.76)
zipf1.1/16: base 6.58/7.58  (3.25/3.60, hr .40)    | noDN 4.41/5.75 (0.73/0.80)
zipf1.1/64: base 4.36/7.32  (1.17/1.86, hr .77)    | noDN 3.51/6.31 (0.26/0.36)
huge == base in every cell (faults identical); hugeND == noDN. noslot:
gemv 4.89 (locality loss). Full 52-row table: `results/stage_*.txt`.

## 3. Suggested follow-ups (parent)

1. Apply noDN (one line) and measure e2e on the 10 GB box — the ONLY
   hardware lever in this sprint with a strictly-positive exact-output gain.
2. Rerun `stage_replay` with a REAL production route trace (replace the
   synthetic uniform/zipf driver) to size slots for the true miss curve.
3. Do NOT pursue huge pages here (THP never), Q8 transcode (dead), or more
   topology/kernel work (agents 1+2 closed those).
