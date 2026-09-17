# Phase 10E — Zero-copy v3 (mprotect): SIGSEGV, alignment root cause, v4 fix

## Result

v3 crashed: `zerocopy_rep1 exited -11` (SIGSEGV) ~10 s in (early prompt).
Resident + staged rep1 completed fine. No performance number.

## Root cause (certain)

mprotect ranges were page-aligned OUTWARD. Tensor bases are 32B-aligned, NOT
page-aligned, so every expert slice shares its first/last pages with neighbor
slices. Evicting Y set PROT_NONE on X's edge bytes while X was admitted;
GEMV then touched NONE -> SIGSEGV. The align-out overlap was harmless for
MADV_DONTNEED (transparent re-fault) but fatal for mprotect. The fail-loud
design worked as intended (loud crash, not silent corruption).

Why early: prompt prefill churns slots; the first eviction NONE'd a range
whose edge pages belonged to an admitted neighbor, touched immediately.

## Collateral: the 0.43 GB "floor excess" was bad arithmetic

Recomputed: v2 total 3.69 GB = anon 0.54 + file 3.15; file = our-mapping
2.19 (exit smaps, post-llama-free) + llama floor 0.96 ~= staged floor 1.07
within noise. No mystery, no mechanism needed. v4 budget: our 2.05 + edge
~0.1 + floor ~1.0 + anon 0.54 ~= 3.7 GB. Closed.

## v4 fix (FINAL attempt)

Round INWARD: only pages fully inside the slice are PROT_NONE/PROT_READ;
edge pages (<=6/bundle) stay READ permanently (~0.05-0.2 GB bounded worst
case). Interior NONE still blocks fault-around zombies; speculation
preserved; pages stay cached. Predicted +12-14% at ~3.7 GB plateau.
Any v4 failure KILLS the zero-copy line (staged remains the frontier).
