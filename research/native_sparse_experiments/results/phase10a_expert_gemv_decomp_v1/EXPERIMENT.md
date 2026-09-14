# Phase 10A — Expert-GEMV decomposition via exact-shape format shootout

## Method

Local AVX2 microbench (`format_shootout_ubench.c`, preserved here) calls the
REAL ggml vec-dot kernels from pinned llama.cpp 3057bb66 on exact Qwen3.5
expert shapes: gate/up rows K=2048 x 512 rows, down rows K=512 x 2048 rows.
Weights stream from a 64 MB pool (DRAM regime, >> LLC); activations stay hot
(the real regime: 280 MB weights stream/token, 2 KB activation reused).
Interleaved trials, min-of-7, FTZ/DAZ, pinned core. Machine: Intel N100
(Gracemont, AVX2+FMA) — NOT Kaggle; absolute ms approximate, relative order
robust (uop-bound kernels, all formats far from bandwidth).

Rows/token: 163,840 gate + 163,840 up @K2048, 655,360 down @K512 = 983,040.

## Results (ns/row, best of interleaved; /4-ideal = ST/4)

| format (w x act) | bpw | gate2048 | down512 | /4-ideal ms/token |
|---|---|---|---|---|
| IQ2_XXS x Q8_K (baseline) | 2.06 | 647.8 | 176.2 | 81.9 |
| Q2_K x Q8_K | 2.62 | 321.2 | 84.7 | 40.2 |
| Q4_K x Q8_K | 4.50 | 434.4 | 113.1 | 54.1 |
| Q4_0 x Q8_0 | 4.50 | 461.0 | 126.8 | 58.5 |
| MXFP4 x Q8_0 | 4.50 | 490.4 | 135.7 | 62.4 |
| Q8_0 x Q8_0 (transcode ceiling) | 8.50 | 519.2 | 131.7 | 64.1 |
| Q5_K x Q8_K | 5.50 | 538.1 | 148.5 | 68.4 |
| IQ3_XXS x Q8_K | 3.44 | 724.1 | 184.3 | 89.5 |

Validation: baseline /4-ideal 81.9 ms vs measured 87 ms expert core (phase5a)
— within 6%. Methodology confirmed.

Activation quantize (q8_K): 8.3 us/2048-row, 3.3 us/512-row → ~0.7 ms/token
total (0.2%). Activation prep/sharing upside ≈ 0. Kills the activation side
of Hypothesis B.

## Decomposition of the ~87 ms expert GEMV core

- Weight decode + integer MAC + DRAM streaming of 280 MB selected-expert
  bytes: ~81 ms (the row-dots themselves).
- Activation quantization: ~0.7 ms.
- SwiGLU/accumulate/route-weight/MUL_MAT_ID dispatch: residual ~5 ms.
- The core is DECODE-UOP-bound, not bandwidth-bound (IQ2_XXS streams at
  ~0.7 GB/s, far below DRAM). Proof: Q2_K moves 27% MORE bytes yet runs 2x
  faster — its plain shift/mask/broadcast decode beats IQ2_XXS's 16
  scalar grid/sign table lookups + 4x set_epi64x inserts per 64 values.

## Decisions

- Q2_K PROMOTED to top representation branch: projected expert core 87→~40
  ms, end-to-end ~3.7 tok/s (+15%) at +27% bytes. Full-system: 1792 slots,
  67.0% hit (vs 73.3%), 117.8 MB fresh/token (vs 74.9) on diverse corpus.
  Next Kaggle experiment after zero-copy. Needs Q2_K weights + quality gate.
- IQ3_XXS (89.5) is SLOWER than IQ2_XXS: kills "3-bit with cheaper decode"
  for the IQ family. Bits don't predict speed; decode structure does.
- Q8_0 ceiling (64.1) caps Hypothesis A at ~18 ms even at 100% L1 hits:
  key input to A's KILL (see MOONSHOT_LEDGER).
- Q4_K (54.1) parked as fallback if Q2_K quality fails. MXFP4 (62.4)
  unremarkable. Q4_0 < Q4_K: super-blocks win.
- Phase5c's "LUT slower" caution stands, but no T-MAC-style kernel was
  tested here — only shipped ggml kernels. A custom LUT kernel could still
  beat Q2_K; not needed for the next step.

## Caveats

N100 vs Kaggle Xeon absolute numbers will differ; the ~2x Q2_K/IQ2_XXS ratio
is the robust claim (same machine, same regime, uop-bound both sides).
/4-ideal scaling validated only against the 87 ms aggregate point.
Quality of Q2_K vs IQ2_XXS is NOT measured here — likelihood gate required.
