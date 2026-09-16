# Functional-Basis Oracle — Final Ceiling Experiment (L20)

**Verdict: KILL HARD.** The 256 expert functions are essentially full-rank
on the expert axis (effective rank 244/256 contextual, 231/256 proxy).
Even the unconstrained oracle with exact routing needs r≈248/256 for
small *median* routed error, mean error stays >5% until r≈248, and tails
(p99 ≈ 32%) are uncontrolled even at r=252. No meaningful shared
functional basis exists — before imposing any realizable SwiGLU
structure. Per instructions, no dictionary implementation follows.

**Branch:** `research/atom-factor-one-layer` · **Date:** 2026-09-16

---

## 1. Data provenance

**Contextual (primary, FINAL):** 2933 real L20-MoE inputs dumped from
exact model inference on Kaggle (`kaggle/native-sparse-l20hdump-v1`,
kernel v2, status COMPLETE, 16 min wall). Pinned runtime
(llama.cpp `3057bb66` + observation-only hooks), pinned weights
(`Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf`, SHA-256 verified
`2a809de3…`). 32 diverse prompts (clinical/mcqa/reasoning/general/
swahili/instruction/short, same set as phase5e), `-n 64`, seeds 1000+pid.
Hook dumps `blk.20.ffn_gate_inp` matmul input (post-attention-norm output)
per forward call; prefill AND decode rows kept (2016 decode + 917 prefill).

- Hook fired twice per call with bit-identical data (verified max diff
  0.0 on all 32 prompts); deduped to 2933 unique rows.
- Route cross-check: locally recomputed top-8 (F32 router, exact
  softmax/top8/renorm) matches hook route IDs on **2015/2016** decode
  tokens (one tie-order edge) — proves dumped x is the true MoE input
  and routing parity.
- Split by PROMPT (seed-shuffled): 22 train (2038 rows) / 10 held
  (895 rows). Near-duplicate audit: 56 held rows with held-vs-train
  cosine >0.999 (shared early-prefill states) EXCLUDED → **839 clean
  held rows**. Exceeds the ≥400/≥200 targets 5×/4×.

**Proxy (comparison):** existing calib-320/held-384 single-token inputs,
real embeds/router/experts. Labeled throughout; conclusions do not depend
on it.

Targets: exact R8 + all-expert E_e from cached dequantized L20 experts
(238/256 routed on contextual, 256/256 on proxy).

## 2. Exact layer/model

L20 of Qwen3.5-35B-A3B (middle layer, full 256-expert coverage).
Router: exact F32 GGUF weights, exact top-8 + renorm — preserved in all
reconstructions (never re-learned).

## 3. Factorization method

Correct expert-axis oracle (`fb_04_expert_axis.py`):

    M[e,(n,d)] = E_e(x_n),  [256, N·d]  (N=2038 ctx / 320 proxy)
    G = M Mᵀ (256×256 Gram, BLAS-blocked, f64) = U diag(S²) Uᵀ
    C_r = U[:, :r]                       static expert codes (256,r)
    Φ(x) = C_rᵀ E_all(x)                 oracular basis values (r,d)
    Ê_e(x) = C_r[e,:]·Φ(x);  Rhat = Σ_{e∈top8} α_e(x) Ê_e(x)

Φ(x) is deliberately oracular (solved from TRUE all-expert outputs), so
this upper-bounds any static-code shared basis; success wouldn't prove a
fast architecture, failure kills the hypothesis. Held eval is token-
blocked, routed-only accumulation (peak ~200 MB after two OOM fixes).

**Audit note:** the first attempt (`fb_01/02`, M of shape (256·N, 2048))
factorized the OUTPUT axis (rank on d=2048), not the expert axis — caught
by the r=256-exactness check before any verdict. Those numbers are kept
as a separate result: KILL on *static low-rank output-subspace*
compression (87% @ r=256/2048, flat spectrum, degenerate communities).
They do not affect this verdict.

## 4. Singular/effective-rank spectrum (expert axis, cumulative energy)

| r | 16 | 32 | 64 | 96 | 128 | 160 | 192 | 224 | 240 | 248 | 252 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ctx cumE | .096 | .190 | .351 | .484 | .609 | .718 | .829 | .925 | .968 | .985 | .993 | 1.0 |
| proxy cumE | .105 | .204 | .380 | .508 | .627 | .735 | .836 | .927 | .967 | .984 | .993 | 1.0 |

Effective rank: **243.8/256 contextual, 231.3/256 proxy** — essentially
full rank; the two distributions agree within ~3%.

## 5. Rank-vs-error table (held-out routed output, exact routing)

Contextual (held-839):

| r | median | mean | p95 | p99 | max | cos | expert |
|---|---|---|---|---|---|---|---|
| 16 | 0.947 | 0.892 | 1.017 | 1.035 | 1.061 | 0.358 | 0.942 |
| 32 | 0.838 | 0.786 | 0.995 | 1.023 | 1.039 | 0.544 | n/a |
| 64 | 0.601 | 0.608 | 0.945 | 0.993 | 1.006 | 0.742 | 0.786 |
| 96 | 0.417 | 0.463 | 0.885 | 0.959 | 1.001 | 0.839 | n/a |
| 128 | 0.345 | 0.396 | 0.862 | 0.942 | 1.001 | 0.876 | 0.556 |
| 160 | 0.266 | 0.317 | 0.783 | 0.928 | 1.002 | 0.914 | n/a |
| 192 | 0.211 | 0.275 | 0.767 | 0.923 | 1.004 | 0.925 | 0.302 |
| 224 | 0.121 | 0.177 | 0.625 | 0.913 | 0.982 | 0.959 | n/a |
| 240 | 0.011 | 0.108 | 0.439 | 0.714 | 0.942 | 0.979 | n/a |
| 248 | 0.005 | 0.027 | 0.160 | 0.339 | 0.532 | 0.998 | n/a |
| 252 | 0.003 | 0.021 | 0.136 | 0.323 | 0.504 | 0.998 | n/a |
| 256 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |

Proxy (held-384): r=64: 0.864; r=128: 0.737; r=192: 0.529; r=224: 0.415;
r=240: 0.263; r=248: med 0.032/mean 0.148; r=252: med 0.019/mean 0.093;
r=256: exact. Same verdict (see §9).

Sanity: r=256 max err **3.9e-07 contextual / 2.7e-08 proxy (PASS)** —
algebra verified. Mean-predictor floor ≈ 0.99; β dense (nnz 1.0,
participation ~17–20/256).

## 6. Routed-output error tails

The curve decays slowly then cliffs near full rank (r=224→240 median
12%→1%) — characteristic of incoherent near-full-rank structure, not
compressible signal + noise. Mean lags median everywhere (r=240: med
1.1% but mean 10.8%); p99 stays catastrophic deep into the top ranks
(r=248: 34%; r=252: 32%). Per-expert errors mirror routed errors
(r=192: 0.30). No rank offers small error AND controlled tails.

## 7. Storage/compute implications

Moot (killed), recorded for completeness. IF a rank-r static-code basis
were realizable with SwiGLU atoms: codes 256×r (f16) + r×6144 atom
params; per-token 8r compose + r atom evals. At the only viable rank
(r≈250): ~1.5M mults vs teacher 25M looks like 16× — but it compresses
256 experts into ~250 basis units (no compression), leaves p99 at 32%,
and Φ is oracular (computed from true all-expert outputs, i.e. full
teacher cost). There is no realizable win behind these numbers.

## 8. Final KEEP/KILL verdict

**KILL HARD**, on real contextual data with exact routing:

- Median <5% needs r≈240/256 (94% of full rank); mean <5% needs r≈248;
  tails uncontrolled even at r=252 (p99 32%).
- Expert-axis function space is essentially full-rank (erank 244/256).
- Route-frequency control: 238/256 experts touched; top-128 experts carry
  95.5% of router mass — the SVD cannot be rescued by reweighting toward
  frequent experts (nearly all experts route; energy↔mass logcorr ≈ 0).
- This was the unconstrained ORACLE ceiling: any realizable static-code
  basis does worse. Failure here kills the static-code shared-functional-
  basis hypothesis. No dictionary/SwiGLU-atom implementation follows.

## 9. Whether contextual data materially differs from proxy

No — same verdict, same mechanism. Spectra within ~3% (erank 244 vs 231);
both need r≈248+ for small median error with bad tails. Differences:
contextual routing is peakier (top-32 mass 65% vs 43%; 238 vs 256 experts
touched) and contextual |R8| smaller (0.65 vs 1.08); contextual mid-rank
errors are somewhat lower (r=128: 0.34 vs 0.74 — larger calib N=2038
estimates the subspace better) but converge to the same cliff. The proxy
result was directionally correct; the contextual run makes the kill final.

## 10. Next ONE experiment

None within this thesis: static shared basis is killed here (full-rank),
x-adaptive shared basis was already killed by the dense-student failure
(test ≈ floor at all widths, flat data scaling). Both sides closed. Do
not pursue atom factorization of routed experts further; if MoE
compression is revisited, it must be a structurally different hypothesis
(e.g. per-expert independent compression, which concedes no sharing).
