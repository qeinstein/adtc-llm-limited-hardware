# Phase 0 — Edge0 facts verified from primary sources

Date: 2026-09-16. All claims below were read directly from the primary
sources listed in §12 (repo clone, checkpoint `config.json`, HF model
card, paper PDF). Where our previous summary
(`docs/research/edge0-architecture.md`, written from `Edge0-AI/Edge0 @
0700e65`) disagrees or is now stale, the source wins — see §11.

## 1. Released model / checkpoint

- Repo: `https://github.com/Edge0-AI/Edge0` (Apache-2.0), framework +
  paper (`paper/main.pdf`, "The Other Half of the Memory Wall", AutoArk,
  Sept 2026).
- 35B tier checkpoint: `Edge0/Edge0-35B-A3B-preview` (HF + ModelScope).
  Single model directory = ready-to-run unit:
  `config.json`, `model-0000{1..4}-of-00004.safetensors`, tokenizer,
  `lora_edge0_35b.safetensors`, `prerouter_edge0_35b.safetensors`.
- Base model (card front-matter `base_model`): **`Qwen/Qwen3.6-35B-A3B`**.
  NOTE: this is Qwen**3.6**, not 3.5 — but `config.json` still uses
  `model_type: qwen3_5_moe`, arch `Qwen3_5MoeForConditionalGeneration`.
  Our path is Qwen3.5-35B-A3B; arch-level compat must be checked
  tensor-by-tensor in Phase 2/3, not assumed.
- Disk size: 19.5 GB int4 base + 0.2 GB adapters (paper Table 1).
  README quick-start says ~23 GB download (likely incl. vision weights;
  `strip_vision_weights.py` exists in `scripts/`).
- License: Apache-2.0 incl. vendored code (NOTICE: mlx-lm MIT, Ling MLX).

## 2. Architecture facts (from checkpoint `config.json`)

- hidden 2048, 40 layers, **256 routed experts + 1 shared expert**,
  `moe_intermediate_size` = `shared_expert_intermediate_size` = 512.
- **Native `num_experts_per_tok = 8`** — the base model is K=8; K=4 is a
  deployment width imposed by the runtime + retrained adapters (§4).
- Routing: softmax → top-k → renormalize (`norm_topk_prob=True`).
- Hybrid attention: 30× linear-attention + 10× full-attention
  (`full_attention_interval: 4`), GQA 16q/2kv, head_dim 256,
  `attn_output_gate: True`. Vocab 248320. MTP block present, unused.
- Routed-expert params: 40×256×3×(512×2048) ≈ 32.2B; per-layer expert
  weights 453 MB at int4 ≈ 1.77 MB/expert; routed total ≈ 18 GB.

## 3. Quant format

- Global: **4-bit affine, group 64** (MLX `mlx-lm` quantize format,
  consumed by `mx.gather_qmm`, bf16-internal; measured residual
  rel-L2 ≈ 0.24%).
- Tensor layout: `switch_mlp.<proj>.weight` packed u32 `[E, out, in/8]`;
  `scales`/`biases` bf16 per group `[E, out, in/64]`.
  `WeightLayout.SEPARATE`: gate/up/down stacked as separate tensors
  under `language_model.model.layers.{N}.mlp.switch_mlp`.
- Exception: router gates (`mlp.gate`, `mlp.shared_expert_gate`, all 40
  layers = 80 overrides in `config.json:quantization_config`) are **8-bit,
  group 64**.
- Math contract: every execution path verified element-wise vs the
  dequantized reference (rel-L2 < 1%); routing fns extracted verbatim
  from vendored models, parity-tested bit-identical.

## 4. Deployed routed K

- **K=4** (`MoESpec top_k=4`, `LayerOptions.staged_k4()`: `staged_n=4`,
  `staged_trigger=4`, `top_k=4` override, `prerouter_top_k=4`).
- K=8→K=4 is done by retraining adapters for the new width, not by
  truncating: "retraining a tier for a different routing width means
  swapping adapter files and nothing else" (paper §4). Same-session A/B:
  K8→K4 nearly doubles decode (3.3→6.4 tok/s, M2 16 GB) at one cache
  budget with the same weights at both widths.

## 5. Adapter / Recover-LoRA details

- `r=16, alpha=32` (scale 2.0), fp16, **unmerged parallel delta**:
  `y = W4bit(x) + scale·((x@A.T)@B.T)` (`adapters/lora.py`).
- Targets: **attention, linear-attention, and shared-expert
  projections — NOT routed experts** (they stream and must stay
  replaceable). Keys `<module>.lora_A [r,in]` / `<module>.lora_B [out,r]`.
- Size ≈ 42 MB, "no measurable decode time".
- Why unmerged: merging + requantizing to 4-bit erases the delta (LoRA
  RMS 1e-3 < 4-bit group step): 34% survives on attention proj, 2% on
  dense proj, 18% at logits level (paper §3.3).
- Training: frozen int4 base, distilled from the fp16 teacher **under
  the student (prerouter-routed) path**, so it jointly compensates
  quantization + routing replacement. 3-phase recipe (§10).

## 6. Prerouter architecture

- 33 heads, owners = layers 6..38, `start_layer=7`, hidden 512, fp16.
  Consumed at layers 7..38 (32 staged layers); the layer-38 head
  predicts layer-39 routing which is never staged (ships unconsumed).
- Per-head input (dim = dmodel + 2E = 2048 + 512 = **2560**):
  `concat[hidden, this-token executed top-k one-hot, prev-token top-k
  one-hot]` (`feature_topk="executed"`).
- Head math: `fc1(2560→512) → exact-erf GELU → fc2(512→256) +
  linear_init(2560→256)` on the same features. `linear_init` is
  warm-started from the next layer's router weight (default zero), so
  training starts from "apply next router to this hidden state" + MLP
  correction.
- Head logits → expert set via the **same** softmax-topk math as the
  base router (`prerouter/heads.py`, `moe/routing.py`).
- File: `prerouter_edge0_35b.safetensors`, keys `layers.<N>.fc1.weight`
  etc. (33 heads × (2560×512 + 512×256 + 2560×256) fp16 ≈ 33×4.2 MB
  ≈ 140 MB — fits the 0.2 GB adapter budget together with LoRA.)

## 7. How routing is predicted (double shift)

- Head owned by layer N runs at token t on **layer N's MoE-input
  hidden state (post-attention norm output)** and predicts **layer N+1's
  routing** (layer shift). Layer N+1 consumes the prediction made at
  token **t−1** (token shift). `CrossTokenStager` commits predictions
  at step boundaries; one flush per step predicts all 32 staged layers.
- Lead must be a full token, not one layer: same-token pre-gating
  needs per-layer sync + head eval that drains the GPU pipeline
  (30–100 ms/step, worse than the load it hides; every same-token
  variant fell below a plain LRU baseline). Paper §3.2.
- Feature drift is real and accepted: at training the one-hot feature
  is the base router's selection; at decode it is the head's own prior
  prediction. Not retrained for; priced into the quality numbers.

## 8. Prediction DETERMINES execution (not prefetch-only)

- **Prediction-as-routing**: at decode, MoE blocks route via the
  prerouter logits (`patch_call=True` installs a class-level `__call__`
  patch on the vendored qwen blocks). Staged set == routed set exactly,
  **zero drop by construction** (`staged_k4()` keeps
  `staged_replace=False` because the router itself is already the
  prediction; the slot table maps without drops).
- The prediction *additionally* drives the streaming tier: it is the
  staged-fill source (loads overlap the forward pass), gets `pin_bonus`
  in hot-pin selection, and feeds `history_prefetch`. But its primary
  role is choosing the executed set.
- Paper: "the coverage-vs-quality trade-off that pre-gated systems
  handle at runtime is eliminated by construction. What the
  approximation costs is transferred to training, where it can be paid
  once."

## 9. Streaming / staging design

- Weights live on SSD as per-layer stacked safetensors; `SafetensorsMmap`
  byte-range mmaps; OS page cache carries hot data.
- `SharedExpertCache`: one global cross-layer LRU, `cache_slots=64`
  bundles (~81 MiB on 8B tier; 35B bundles are bigger). Deliberately too
  small to hold a token's working set — the design wins on staging, not
  residency.
- Per-layer `StreamingSwitchGLU`, decode workhorse = staged path:
  4 fixed slots + overflow zero slot; slot table + `take` keeps indices
  on GPU (zero per-layer host sync); `asm_cache` reuses graph nodes per
  expert set; `incr_stack` replaces 9 `mx.stack` nodes with in-place
  `put_along_axis` row writes (+34% decode in A/B); `incr_writeback`
  dedups staged bundles out of the LRU; `warm_willneed` issues
  `madvise(WILLNEED)` readahead over predicted ranges (staging wall
  162→27 ms/step on M2).
- Prefill: 35B uses on-demand expert path (`full_layer_prefill=False`,
  chunk 2048, `prefill_hot=32`); whole-layer bulk prefill is the 8B
  profile. Decode-only prerouter (prefill hits every expert, nothing
  to predict).
- Serving: **single-stream, one request at a time, FIFO-serialized**
  (paper §6; `edge0 serve` OpenAI-compat `/v1/chat/completions`).
- Threads: `load_threads=8`, `prefetch_threads=4`, `prefetch_cap=48`.

## 10. Training recipe (paper §4 — disclosed, contrary to our old note)

All phases run on the **dequantized bf16 reconstruction of the 4-bit
deployment checkpoint** ("train on what is served"), base frozen:

1. **Distill the heads**: only head params trained; loss imitates the
   next layer's true router (predict routing, don't fit text).
2. **SFT on the student path**: LoRA attached (attn/linear-attn/shared
   only), full forward with prerouter routing active, ~2M rows of
   teacher-generated text. "Distillation-only checkpoints with student
   routing produce repetitive, collapsed text"; SFT is what makes the
   approximation usable. Order heads→SFT→distill was "not negotiable"
   (SFT signal drowns tiny head gradients otherwise).
3. **On-policy distillation**: Phase-2 checkpoint generates; fp16 base
   scores; reverse-KL (mode-seeking) on teacher top-k + tail term;
   ~200k rows (1/10 of SFT corpus).
- Objective guidance: router agreement is the wrong target
  (cross-token prediction is information-limited); what matters is text
  quality under student routing.

## 11. Deltas vs our previous summary

1. Base is **Qwen3.6**-35B-A3B per the release card (old note assumed
   Qwen3.5 stock config). `model_type` is still `qwen3_5_moe`, so the
   arch family matches; weight-level compat with our Qwen3.5 checkpoint
   is unverified and must be tested, not assumed.
2. Training recipe is **now disclosed** (paper §4, in-repo); old note
   §M said it was undisclosed. The 3-phase recipe + "train on dequant
   bf16" + "LoRA skips routed experts" + phase-ordering constraint are
   the actionable new facts for a Phase 4.
3. Old note §H computed ~270 MB/token working set at K=4 and called
   15 tok/s "physically impossible" without hits; paper Table 4
   reconciles this: actual **disk reads** are 58.9 MiB/step at K=4
   (page cache absorbs the rest), and the win is moving reads off the
   critical path (blocked-load time 244→102 ms/step), not deleting
   bytes. The old note's conclusion (design lives or dies on
   hit/predict rates) still stands.
4. Throughput numbers moved: old note quoted README 14.9–17.7 tok/s;
   paper Table 1 (latest paired) reports **20.4 tok/s** on the same
   M4 Pro 24 GB. Old GATE A1 KILL was under the single-GGUF profiler
   contract; this sprint's contract (custom native-sparse N100 engine,
   single-stream decode, <3 GiB) is different, so that KILL does not
   apply here — recorded for honesty, not as a stop signal.

## 12. Published hardware / throughput numbers

- M4 Pro 24 GB (Table 1, latest paired, think-mode on): **20.4 tok/s**
  decode, prefill 113/140 cold/warm (3.1k prompt), peak 2.9 GiB.
  No-prerouter arm on same box: 19.9 tok/s (hot-cache regime: prerouter
  gain shrinks when weights fit page cache — §6 limitation).
- M2 16 GB A/B, 18.4 GiB checkpoint that does NOT fit (Table 3/4):
  on-demand vs prerouter: K=2: 4.8→8.6 (+80%); **K=4: 3.5→6.4 (+82%)**;
  K=8: 1.8→3.3 (+84%). Blocked-load time/step: 244→102 ms at K=4.
  Decode gains grow with storage latency, model size, K.
- Resident baseline (vanilla mlx-lm, all 19.5 GB resident): 3.9 tok/s
  at 18.2 GiB — the "weights don't fit" regime is why streaming wins.
- Quality (OpenCompass, identical settings): −3.9 avg vs fp16 base
  (AIME 86.6/92.7, HumanEval 90.9/95.1, GPQA-D 79.8/81.8, MMLU-Pro
  81.0/84.6, IFBench 57.9/61.7). Loss concentrates in long-chain
  reasoning (AIME −6.1).
- 8B tier (Ling 3.0 tiny, K=8, sigmoid-group): 28.0 tok/s, 1.5 GiB,
  −2.8 avg. Not our target; listed for completeness.

## 13. Sources (exact)

- `Edge0-AI/Edge0` main @ clone 2026-09-16: `README.md`,
  `docs/architecture.md`, `docs/moe.md`, `docs/prerouter.md`,
  `docs/streaming.md`, `docs/models/edge0-35b.md`, `paper/main.pdf`,
  `src/edge0/{moe,spec/routing,prerouter/{spec,heads,install,stager},
  streaming/options,adapters/lora,models/edge0_35b,backends/mlx/quant,
  engine/{base,qwen},server}`.
- `Edge0/Edge0-35B-A3B-preview`: `README.md` (model card),
  `config.json` (`num_experts_per_tok=8`, quant overrides incl. 8-bit
  router gates), file list (4 base shards + 2 adapter files).
- Full paper text extracted to session scratch for audit
  (not committed): key sections §§3–6 + Tables 1–4 + Appendices A/B.

## 14. Implications for our N100 path (carry-forward, not verdicts)

- The mechanism that matters for us is **K4 execution + prediction-as-
  routing + unmerged LoRA recovery**, not MLX/Metal/SSD specifics. Our
  analog of the SSD tier is our bounded/staged expert path; our analog
  of `gather_qmm` is an AVX2 group-int4 GEMV kernel.
- Two facts favor testing: (a) K8→K4 nearly doubled their decode at
  fixed weights; (b) their int4 group-64 is hardware-friendly vs our
  IQ2 (simpler dequant, higher bytes — the exact trade Phase 1 prices).
- Two facts urge caution: (a) their base is Qwen3.6, ours Qwen3.5 —
  released adapters may not transfer (Phase 2/3 must test, and Phase 4
  exists for exactly this); (b) their decode floor is 44 ms/step of
  graph building on M4 — our N100 floor will be shaped by AVX2 GEMV +
  staging, which Phase 1 measures directly.
