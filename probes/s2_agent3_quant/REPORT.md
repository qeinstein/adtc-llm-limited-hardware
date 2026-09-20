# S2 AGENT 3 — Quant format Pareto: REPORT (parent-run; agent died on network)

Method: the assigned agent wrote nothing before dying on transport errors,
so the parent ran the workstream inline with `qpareto.c` (this dir): real
L00 IQ2 expert slices dequantized to f32 (real ggml dequantizers),
requantized to Q2_K/Q3_K/Q4_K (real ggml `_ref` quantizers — the pin's x86
TU only carries Q8 quantizers), then timed with the REAL per-format AVX2
GEMV kernels (`ggml_vec_dot_{q2,q3,q4}_K_q8_K`) + noDN staging-miss refill.
ST medians, 8 experts; 4T scaling is the known ~3.5x. 6 runs (3 contended,
3 clean); ordering Q2<Q4<Q3<IQ2 held in all 6. Box N100/KVM, 2x absolute
swings — ratios are the signal, cross-checked (see below).

## 0. Verdict

- **Q2_K: PROMISING (conditional)** — 1.7x expert GEMV for +18% bytes. The
  only format that is both materially faster and plausibly RAM-fittable.
  NOT promoted yet: needs (a) perplexity/quality gate on true quant-from-
  fp16 weights, (b) slot-budget math on the 10 GB box.
- **Q4_K: KILL** — 1.6x faster but 2.02x bytes blows the bounded budget
  (3.4 GB staging cache alone at 48 slots).
- **Q3_K: KILL** — worst of both: only 1.16x faster for +54% bytes
  (3-bit unpack is ugly on AVX2; confirmed slowest K-quant decode).

## 1. Pareto numbers (medians-of-medians, clean runs; S2 realistic baseline
   172 ms / 5.8 tok/s; expert GEMV core ~63 ms at 4T spin)

| format | bytes/expert | vs IQ2 | GEMV-8 ST ms | speedup | miss refill us | maxabsdiff* | staging @48 slots |
|---|---|---|---|---|---|---|---|
| IQ2 | 876,544 | 1.00x | 5.92 | 1.00x | ~80 | 0 | 1.68 GB |
| Q2_K | 1,032,192 | 1.18x | 3.51 | 1.68x | ~93 | 0.0028 | 1.98 GB |
| Q3_K | 1,351,680 | 1.54x | 5.11 | 1.16x | ~163 | 0.0015 | 2.59 GB |
| Q4_K | 1,769,472 | 2.02x | 3.68 | 1.61x | ~172 | 0.00047 | 3.39 GB |

*maxabsdiff vs IQ2 output measures requant-from-IQ2 drift, NOT model
quality. All formats need a quality gate on true quant-from-fp16 GGUFs.

Projected end-to-end (expert GEMV 63 ms -> 63/1.68 = 37.5 ms; rest of frame
unchanged): 172 - 25.5 = **~146 ms / 6.8 tok/s** for Q2_K. Matches the old
phase10a "~2x" hint (measured 1.68x).

## 2. Why Q2_K wins the decode race (and Q3_K loses)

- IQ2 pays grid/gather scalar decode per block (~90% of block insns per the
  S1 disassembly); effective ~1.2 GB/s here.
- Q2_K is regular 2-bit + 6-bit scales: simplest K-quant decode, 2.3 GB/s
  effective — still decode-bound, just cheaper decode.
- Q3_K's 3-bit cross-byte unpacking defeats clean SIMD (slowest K kernel).
- Q4_K nibbles decode well but cost 2x bytes: loses on RAM, not speed.
- Miss refill scales ~linearly with bytes (noDN path, PTE-present).

## 3. RAM analysis (the Q2_K constraint)

Q2_K staging at the S2 48-slot default = 1.98 GB vs IQ2 1.68 GB (+305 MB).
Headroom to <3 GiB depends on dense-resident size (~1.2 GB est.): likely
OVER without a slot cut (28 slots -> 1.16 GB but +misses). Exact trade
needs the production route trace + 10 GB-box measurement. Requant one-time
cost (ref quantizers, 8 experts: Q2 2.9 s, Q3 0.65 s, Q4 3.5 s) is
irrelevant — production ships prequantized GGUF.

## 4. Cross-checks

- Independent `transcode` binary on the quiet box: iq2_gemv8 = 5.70 ms vs
  qpareto IQ2 5.92 ms. MATCH (earlier 3.3-3.4 ms values were quieter-box
  eras; ratios, not absolutes, are the finding).
- Checksums/maxabsdiff stable bitwise across all 6 runs per format
  (0.002811 / 0.001471 / 0.000469) — the requant path is deterministic.

## 5. Verdict table (8-item)

| candidate | baseline | measurement | error/quality | ms saved | tok/s | RAM | verdict | stacks |
|---|---|---|---|---|---|---|---|---|
| experts IQ2->Q2_K | 172 / 5.8 | 1.68x GEMV, +18% B | GATE OPEN (re-Q drift 0.0028; needs ppl) | ~25 | ~6.8 | +305 MB @48sl | PROMISING | yes: +noDN, +any arch win |
| experts IQ2->Q3_K | 172 / 5.8 | 1.16x GEMV, +54% B | GATE OPEN | ~9 | ~6.1 | +0.9 GB | KILL (worst trade) | — |
| experts IQ2->Q4_K | 172 / 5.8 | 1.61x GEMV, +102% B | GATE OPEN | ~24 | ~6.8 | +1.7 GB, over budget | KILL (RAM) | — |

## 6. Rebuild

```
gcc -O3 -march=native -c qpareto.c
g++ -O3 -march=native -o qpareto qpareto.o <ggml-quants.o> <ggml_mini.o> <vec.o> <shim.o> /tmp/hw1/x86_quants.o -lm -lpthread
```
(objects: see `../hw_profile/REPORT.md` section 6; needs `/tmp/agent1_raw`
L00 slices.) Raw: `results_runs.txt`.
