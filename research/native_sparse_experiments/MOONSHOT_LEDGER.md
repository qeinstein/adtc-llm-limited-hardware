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
- Evidence: (1) DONE — our checkpoint has 0 MTP tensors (733-tensor
  inventory). MTP weights exist separately: unsloth/Qwen3.5-35B-A3B-MTP-GGUF
  (same quant names, MTP-inclusive). Benchmark needs that file + llama.cpp
  MTP draft flags. Queued as Kaggle kernel after v2/Q2_K.
- Status: INVESTIGATE (weights located; bench pending).

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
- Optimistic saving (CORRECTED for mixed baseline): our file uses XXS for
  gate/up but faster IQ2_S for down. True mixed baseline /4-ideal = 71.1 ms;
  Q2_K = 38.9 ms (ratio 0.547). Ratio-scaled to the measured 87 ms core:
  Q2_K core ≈ 47.6 ms, saving ≈ 39 ms → ≈3.57 tok/s (+11.5%) at +27% bytes.
- Cheap falsification: DONE (ubench2 + corpus replay: 1792 slots, 67.0% hit,
  117.8 MB fresh/token). UD-Q2_K_XL download KILLED as Q2_K source: contains
  ZERO Q2_K tensors (39 layers XS/XS/IQ3_XXS + L10 higher; all slower than
  XXS per shootout; non-uniform layer 10 breaks bounded cache).
- Evidence: microbench + corpus replay + both-file tensor inventories.
- Status: PROTOTYPE (top representation branch; TRUE Q2_K via on-Kaggle
  IQ2_XXS->Q2_K transcode with llama-quantize; needs Kaggle A/B +
  likelihood quality gate; fallback: requant from Q4_K_M if gate fails).

## E — Shared basis / expert residuals

- Hypothesis: within a layer, W_e ~= B + Delta_e (or U C_e, or sum alpha B_j)
  with most energy in a shared component computed once per token.
- Closest prior art: (search pending — MoE merging/compression, expert
  factorization, SVD-compressed MoE).
- Same/different: TBD.
- Physical mechanism: common GEMV once per token + tiny per-expert residual;
  attacks compute AND storage.
- Required assumptions: trained experts share strong subspace structure.
- Optimistic saving: none — no structure found (see below).
- Cheap falsification: DONE offline. Range-fetched 96 expert slices
  (L0/10/20/30 x gate/up/down x 8 experts), dequantized via libggml,
  stacked-SVD + cosine analysis. Results (L0 all kinds, L10 gate/up):
  pairwise cosine mean ~0.000 (max 0.009); mean-residual energy 0.87
  (shared mean captures only ~13%); shared-subspace energy at rank 64
  only 0.11-0.23; rank-64 reconstruction error 0.88-0.94. Experts are
  effectively independent full-rank matrices in all families/layers.
- Evidence: /tmp/zc_svd.py + slices (scratch); consistent across 5/5 cells.
- Status: KILLED (no shared structure; B+Delta / U*C both dead).

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

## H (own) — SwiGLU gate sparsity skips up/down rows

- Hypothesis: compute gate(x) first; for intermediate dims where silu(g)~=0,
  skip the corresponding up-row and down-column (exact up to eps threshold).
- Closest prior art: DejaVu (contextual sparsity, trained predictors, 2x on
  OPT-175B/GPU); also TEAL, ReLUfication, ShadowLLM.
- Same/different: DIFFERENT substantively — predictor-free (compute gate,
  skip by threshold), MoE+SwiGLU-specific, CPU-decode regime; DejaVu needs
  trained predictors and targets GPU weight-loading I/O.
- Physical mechanism: silu kills negative pre-activations; trained gates
  often produce many negatives -> up/down GEMV on active dims only.
- Required assumptions: gate pre-activations substantially negative-side.
- Optimistic saving: 50% sparse -> ~29 ms (9%).
- Cheap falsification: ONE Kaggle measurement kernel accumulating
  |silu(g)| histograms (no math change); sparsity fraction decides.
- Evidence: none yet.
- Status: INVESTIGATE.

## I (own) — LM-head shortlist with adaptive exact fallback

- Hypothesis: cheap approximate head (low-rank/clustered) shortlists top-~1K
  of 248K vocab; score only those exactly; fall back to full head iff the
  top-1 margin is thin (adaptive exactness).
- Closest prior art: NO close hit in bounded search. Nearest: AdaptiVocab
  (static domain vocab — different), speculative decoding (different
  mechanism), hierarchical softmax (different structure). Adaptive
  exact-fallback shortlist for CPU decode appears novel in this form —
  run the experiment anyway per directive.
- Physical mechanism: logits are peaky; full 2048x248K Q4_K GEMV (25 ms) is
  overkill for argmax preservation.
- Required assumptions: shortlist contains true top-1 with high probability.
- Optimistic saving: 25 -> ~3 ms = 22 ms (7%).
- Cheap falsification: ONE Kaggle kernel capturing full logits for 64 tokens,
  then OFFLINE shortlist-hit-rate eval at various ranks/costs.
- Evidence: none yet.
- Status: INVESTIGATE.

## J (own) — Static per-pattern fused expert blobs — KILLED

- Hypothesis: top-8 SETS cluster into few frequent patterns/layer; precompute
  fused interleaved blobs per pattern (layout-only, same bytes).
- Cheap falsification: DONE offline. Diverse corpus: 1714 distinct
  sets/layer (of 2016), top-50 covers 13.9%. Single prompt: top-10 covers
  17.7%. Needs 70%+ -> structurally dead.
- Status: KILLED (same-day falsification).

## J2 (own) — Uniform-Q2 system: dense trunk to Q2_K as well

- Hypothesis: Q2_K's 2x decode win applies to the 73 ms dense GEMV
  (attention proj + LM head) too; quality may survive since Q2_K keeps
  imatrix-optimized scales.
- Closest prior art: uniform low-bit LLMs (same/different TBD).
- Same/different: TBD by quality gate, not by speed reasoning.
- Physical mechanism: same decode-uop win as experts, on resident dense.
- Required assumptions: quality gate passes (risky; dense is sensitive).
- Optimistic saving: 73 x 0.4 + 39 = ~68 ms (+22%) combined with Q2_K experts.
- Cheap falsification: FREE — one extra arm (all-Q2_K) in the already-planned
  Q2_K transcode kernel + likelihood gate. Zero extra Kaggle cost.
- Evidence: none yet.
- Status: PROTOTYPE-QUEUED (rides the Q2_K kernel).

## Promotion rule

Any branch with >10% plausible end-to-end upside moves to the main experiment
queue immediately. 5–10%: one cheap falsification. <5% and no unlock: KILL.
