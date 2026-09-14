# Moonshot / Hypothesis Lab Ledger

Wall budget (staged <=4 GiB, phase8c/9b): **312.7 ms/token** =
~87 expert GEMV + ~75.5 bounded tax + ~11.6 removable I/O + ~138 remainder.
15 tok/s needs 66.7 ms/token. All optimistic savings below are against 312.7
unless noted. Status values: PROPOSED / INVESTIGATE / PROTOTYPE / EXPLOIT / KILLED.

## A — Dynamic transcode execution cache

- Hypothesis: keep the large bounded cache compressed IQ2; add a small L1 of
  recently used experts transcoded once into a compute-native form, amortized
  over temporal reuse hits.
- Closest prior art: (search pending — hot/cold code caches, JIT code caches,
  AWQ/GPTQ pre-transformed weights are static, not dynamic).
- Same/different: TBD by search; static hot/cold (phase9c) is NOT the same —
  this exploits temporal reuse, not global popularity.
- Physical mechanism: one transcode per admission; subsequent hits skip IQ2
  decode. Deletes decode/weight-traversal CPU on L1 hits.
- Required assumptions: temporal reuse exists at small L1 sizes; transcode <<
  saved decode over residency; L1 RAM fits budget.
- Optimistic saving: 18 ms absolute ceiling (Q8_0 row-dot 64.1 vs IQ2_XXS
  81.9 /4-ideal ms; exact-shape AVX2 microbench, N100) at 100% L1 hits.
- Cheap falsification: DONE. Corpus LRU: L1<320 bundles gets ~0% hits
  (sequential-scan thrashing — one token spans 320 bundles); L1=512 hits
  33.4% with 213 admissions/token; L1=1024 hits 53.8% with 148 adm/tok.
  Transcode cost (~0.6 x GEMV-row x admissions) ≈ 25–35 ms/token DWARFS the
  6–10 ms hit saving. Net ≈ −15 to −30 ms. RAM also impossible (512 Q8
  bundles = 1.8 GB over budget).
- Evidence: /tmp/zc_l1sim.py + ubench2 (scratch; method in phase10a record).
- Status: KILLED (double-kill: negative net + RAM).

## B — Top-8 as one mathematical kernel

- Hypothesis: one custom kernel consumes the shared hidden vector once and
  evaluates all 8 selected experts jointly (shared activation prep, joint
  decode/traversal, fused route weighting on down).
- Closest prior art: (search pending — fused MoE kernels: vLLM/DeepGemm grouped
  GEMM, llama.cpp MUL_MAT_ID are per-expert dispatched, not joint-decode).
- Same/different: NOT scheduling fusion (phase9a killed that); changes layout,
  decode, traversal, SIMD use.
- Physical mechanism: activation quant/LUT built once per layer instead of
  8x; single pass over x; wider SIMD groups across experts.
- Required assumptions: activation prep must be large — FALSIFIED:
  measured q8_K quantize = 8.3 us/2048-row, ~0.7 ms/token total (0.2%).
  Activation-sharing upside ≈ 0. B survives ONLY via joint decode.
- Optimistic saving: TBD by 1/2/4/8-joint microbench on exact shapes.
- Cheap falsification: microbench joint-vs-separate IQ2 row-dots, exact
  Qwen dims, AVX2.
- Evidence: none yet.
- Status: PROPOSED.

## C — MTP earlier than planned

- Hypothesis: Qwen3.5 MTP + CPU verification raises EFFECTIVE tok/s via
  multi-token acceptance, and verification batching amortizes GEMV/fetch.
- Closest prior art: Qwen3.5 MTP GGUF ecosystem + llama.cpp MTP support;
  classic speculative decoding (Leviathan et al.; Chen et al.).
- Same/different: standard mechanism, new measurement on this exact
  CPU/bounded regime. Keep RAW vs EFFECTIVE metrics separate.
- Physical mechanism: draft K tokens cheaply, verify in one target pass;
  accepted tokens share one expert-fetch/GEMV/dense/LM-head traversal.
- Required assumptions: MTP tensors available for our checkpoint; draft cheap
  on CPU; acceptance rate high enough; verification pass not ~Kx cost.
- Optimistic saving: TBD (acceptance-rate × verification-efficiency).
- Cheap falsification: (1) check checkpoint for MTP tensors; (2) existing
  llama.cpp MTP bench on Kaggle CPU, N=2/3/4.
- Evidence: none yet.
- Status: PROPOSED.

## D — IQ2 is the wrong computational representation

- Hypothesis: a compute-native low-bit format (LUT W2/W3, simple W4,
  ternary/codebook) deletes AVX2 decode cost even at somewhat larger bytes.
- Closest prior art: T-MAC (LUT-based CPU low-bit), Trellis, IQK kernels
  (inspiration, not copies).
- Same/different: our regime is unusual (2048 hidden, 512 interop, top-8,
  1-row decode, 2–4 cores, bounded cache, storage NOT dominant) — generic
  T-MAC results do not transfer directly; must microbench exact shapes.
- Physical mechanism: replace IQ2 dequant+MAC with table lookup/accumulate
  or cheaper integer arithmetic.
- Required assumptions: decode is a large share of 87 ms (phase3 says
  ~88.6% of IQP cycles are decode/materialize — supportive).
- Optimistic saving: Q2_K ≈ 40.2 vs IQ2_XXS 81.9 /4-ideal ms (exact-shape
  AVX2 microbench; baseline reproduces the measured 87 ms within 6%).
  Projected end-to-end ≈ 3.7 tok/s (+15%) at +27% bytes. Q4_K 54.1,
  Q4_0 58.5, MXFP4 62.4, Q8_0 64.1, IQ3_XXS 89.5 (worse than baseline!).
- Cheap falsification: DONE (ubench2, real ggml kernels, streaming weights,
  hot activations, interleaved min-of-7). Q2_K full-system: 1792 slots,
  67.0% hit (vs 73.3%), 117.8 MB fresh/token (vs 74.9) on diverse corpus.
- Evidence: microbench + corpus replay; needs Kaggle A/B + quality gate.
- Status: PROTOTYPE (top representation branch; next Kaggle experiment
  after zero-copy; needs Q2_K weights: requant from F16/Q8 source or
  IQ2->Q2_K transcode).

## E — Shared basis / expert residuals

- Hypothesis: within a layer, W_e ~= B + Delta_e (or U C_e, or sum alpha B_j)
  with most energy in a shared component computed once per token.
- Closest prior art: (search pending — MoE merging/compression, expert
  factorization, SVD-compressed MoE).
- Same/different: TBD.
- Physical mechanism: common GEMV once per token + tiny per-expert residual;
  attacks compute AND storage.
- Required assumptions: trained experts share strong subspace structure.
- Optimistic saving: TBD (depends on residual rank needed).
- Cheap falsification: OFFLINE weight analysis — sample early/mid/late layers,
  expert-mean residuals, pairwise similarity, shared-PCA energy. No Kaggle.
- Evidence: none yet.
- Status: PROPOSED.

## F — Why must K remain 8?

- Hypothesis: top-8 behavior compressible to top-4/2 + correction
  (renorm, shared-expert correction, low-rank surrogate, learned residual).
- Closest prior art: (search pending — MoE pruning, expert merging, K
  reduction + fine-tune).
- Same/different: TBD. Quality-gated co-design, not exact execution.
- Physical mechanism: halving routed experts halves ~87 ms core directly.
- Required assumptions: omitted routing mass small or errors structured and
  recoverable by cheap correction/distillation.
- Optimistic saving: top-4 ≈ up to ~43 ms (13.8%) if quality holds.
- Cheap falsification: calibration prompts, record router scores, measure
  omitted mass + layer/logit error for K=6/4/2. Needs inference (Kaggle).
- Evidence: none yet.
- Status: PROPOSED.

## G — Hybrid backbone deserves its own runtime

- Hypothesis: persistent recurrent-state executor for the 3x Gated DeltaNet +
  1x full-attention pattern beats generic ggml graph dispatch.
- Closest prior art: (search pending — DeltaNet CUDA kernels exist; CPU
  persistent-state decode executors?).
- Same/different: TBD.
- Physical mechanism: state stays kernel-native across tokens; fused
  state-update/gating/projection; compiled fixed decode program for 3:1.
- Required assumptions: DeltaNet surface is large in the 138 ms remainder
  (phase5a says GDN is only 5.37% of summed op work — pessimistic; must
  confirm with wall-clock bypass).
- Optimistic saving: TBD after remainder decomposition.
- Cheap falsification: wall-clock GDN-bypass bound on Kaggle.
- Evidence: none yet.
- Status: PROPOSED (gated on remainder decomposition).

## H (own) — Decode the expert ONCE per token, reuse across... nothing? No: pipeline prefetch into L2 via software prefetch

- Placeholder — to be replaced by a genuinely derived idea after SPRINT 2/3
  measurements. Not counted yet.
- Status: PROPOSED.

## I (own) — placeholder pending measurements

- Status: PROPOSED.

## J (own) — placeholder pending measurements

- Status: PROPOSED.

## Promotion rule

Any branch with >10% plausible end-to-end upside moves to the main experiment
queue immediately. 5–10%: one cheap falsification. <5% and no unlock: KILL.
