# Phase 3.5 — Training-free K4 normalization (paper verification log)

Source read in full: arXiv:2609.04575v1 (Chen & Yao, submitted 4 Sep 2026),
"Training-Free Halving of Activated Experts in Fine-Grained MoE Models"
(PDF fetched + text-extracted locally; no code link in the paper).
Status: formula VERIFIED, implementation mapped to our pin. No guessing.

## Exact formula (paper Eq. 2)

Router: p = softmax(W_g x) over ALL E experts (E=256, flat: top-1 mass
0.046, top-8 mass 0.182, top-32 mass 0.380 on WikiText).

  w_i = p_i / SUM_{j in T_k2} p_j,   for i in T_k1        (T_k1 ⊆ T_k2)

  y = SUM_{i in T_k1} w_i E_i(x) + g_sh E_sh(x)

- k1 = executed experts (compute). k2 = normalization reference (gain).
- k2=k1 → standard renormalization. k2=E → none (catastrophic: -27.45 MMLU).
- k2=k (trained 8) → trained gain. k2>k → BELOW trained gain, per-token
  adaptive (weights sum to m_k1/m_k2 < 1, NOT to one).
- Shared expert UNCHANGED (separate gate g_sh). No weights modified.
- Only k1 FFNs evaluated; router already computes all E probs → k2 costs
  ~nothing (a wider top-k; we MEASURE it rather than assume).

## Reported numbers (Qwen3.6-35B-A3B, MMLU 2000q 5-shot A-D logits, McNemar)

native 81.65% | (6,6) -2.60 | (6,8) -1.10 | (6,16) +0.75 (!!) |
(4,4) -4.65 | (4,8) -3.10 | (4,16) -0.35 (p=0.66, indistinguishable).
GSM8K 0-shot greedy: (4,4) -6.60 trunc 10.4% | (4,16) -0.60 trunc 1.4%.
Perplexity-optimal k2=8, MMLU-optimal k2=16 (-1.10 at k2=8, p=0.021):
DO NOT select k2 on perplexity. Paper recommends scanning {k1,k,2k}.
397B replicates (k=10→k1=5,k2=10: -0.55, p=0.24).

## Mapping to our pin (llama.cpp 3057bb6)

- Router+norm live in llm_graph_context::build_moe_ffn (src/llama-graph.cpp):
  probs=softmax over E (gating SOFTMAX, qwen35moe passes norm_w=true);
  selected=argsort_top_k(selection_probs, n_expert_used); weights =
  get_rows(probs, selected) / sum (the `if (norm_w)` block); mixture loop
  sums hparams.n_expert_used(il) views (MUST override too — uniform arch).
- Patch (all env-gated, native when unset): GGML_MOE_K1 (default 8) shadows
  n_expert_used for selection+dims+sum-loop; GGML_MOE_K2 (default = native
  k) builds a second top-k2 mass (argsort+get_rows+sum_rows+clamp+div);
  k2==k1 keeps the NATIVE norm lines verbatim (control bit-parity).
- moe_out (routed mixture, pre-shared-add) named edge0_moe_out-<il> and
  captured POST-barrier in the CPU node loop (fusion only touches
  RMS_NORM+MUL, verified safe). cb() does NOT name tensors (delegate!).

## Ours vs paper (interpretation notes)

- Our MMLU is matched-likelihood (mmlu-test.bin, 200 tasks), NOT 5-shot
  A-D prompting; absolute scores differ (~42 vs ~82). The TEST is the
  DELTA PATTERN (naive ≈ -4..5 scaled vs k2=16 ≈ 0), not absolute values.
- Our n=200 has ~3.1pp paired MDD (vs paper's 0.98 at n=2000): architecture
  sanity gate per the brief, not an equivalence proof. Follow-up MMLU-1000
  if the pilot lands in the ambiguous zone.
- Paper disabled reasoning via chat template (transformers); our logprob
  MMLU is thinking-independent, so no thinking control needed for the gate.
  Generations (sanity set) WILL think; inspected as-is.
