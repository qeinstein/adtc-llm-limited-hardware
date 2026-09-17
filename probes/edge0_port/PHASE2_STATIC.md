# Phase 2 — Released-model quality (STATIC PORTION; gate results pending)

Status: static analysis COMPLETE (local, verified 2026-09-16). Gate kernels
RUNNING on Kaggle (see §6); quality verdict lands when they complete.

## 1. Static compatibility: shapes YES, weights NO

Compared `Edge0/Edge0-35B-A3B-preview` (Qwen3.6-int4) vs `Qwen/Qwen3.5-35B-A3B`
from primary sources (both `config.json`, both `tokenizer.json`, both
safetensors index maps, both adapter headers — all fetched, KB-scale):

- Architecture: IDENTICAL on every structural key (hidden 2048, 40 layers,
  256 experts, native K8, moe/shared inter 512, GQA 16q/2kv, head_dim 256,
  linear-attn 16/32/128 dims, full-attn interval 4, layer_types 40/40,
  rope (mrope 11/11/10, theta 1e7, prf 0.25), vocab 248320, GQA/MTP flags).
  Diffs are metadata-only (bos/pad keys, mlp_only_layers []).
- Tokenizer: core vocab dicts IDENTICAL (248044 + same IDs; probed EN
  tokens match). Edge0 adds 7 audio/TTS special tokens; all shared IDs
  unchanged. Tokenizer-compatible (their superset).
- Tensor maps: identical except MTP expert layout (3.6 fused gate_up/down
  vs 3.5 per-expert; MTP unused by either runtime — irrelevant).
- Adapter shapes fit our dims exactly: LoRA r16 on shared gate/up/down
  (40×3), linear_attn in_proj a/b/qkv/z + out (30×5), self_attn q/k/v/o
  (10×4), 620 tensors F16 ≈42MB, zero routed-expert targets; prerouter
  33 heads owners 6..38, fc1 [512,2560] + fc2 [256,512] +
  linear_init [256,2560] F16 ≈138MB (2560 = 2048+2×256 ✓ our dims).
- LoRA metadata: K=4, r=16, alpha=32, converted 2026-09-08 from
  `.../qwen35-v7-deploy/lora_qwen35_v7_round9.npz` (training lineage
  mentions "qwen35" — internal codename, not evidence of 3.5 weights;
  not relied upon).

## 2. Weight identity: Qwen3.6 ≠ Qwen3.5 (adapters do NOT transfer)

bf16 slice comparison, same tensors (L20 E0 gate_up 4MB + L20 router gate
1MB, range-fetched from both base repos):
- expert gate_up: cos 0.9498, relL2 0.314 → DIFFERENT pretrain.
- router gate: cos 0.9919, relL2 0.130 → closer but still different.
Both far beyond any numeric noise (bf16 sources). Conclusion: their LoRA
(correction for 3.6-int4 error) and their prerouter (3.6 routing from 3.6
hidden states) do NOT transfer to our 3.5-IQ2/Q2K weights. Double block:
(1) base mismatch (3.6 vs 3.5), (2) quant mismatch (int4 vs IQ2/Q2_K).
Direct adapter reuse is KILLED with precise cause. Any K4+prerouter path
needs OUR OWN adaptation (Phase 4). The MECHANISM (dims, math, staging,
3-phase recipe) ports cleanly — shapes all match.

## 3. Why "as intended" execution is not available here

- Released runtime is MLX-only (Apple Silicon); no Mac in this environment.
  No CUDA/CPU backend exists upstream (`backends/cuda/` is a reserved stub).
- Conversion paths and their fidelity gaps:
  (a) MLX-int4 → GGUF + llama.cpp K8: drops K4 AND LoRA AND prerouter —
      tests their base+quant only, not their product. (b) +GGUF metadata
      K=4: naive top-4 truncation, not predicted routing — the paper's
      collapse case (SFT-on-student-path is load-bearing). (c) +LoRA:
      llama.cpp LoRA has no mapping for their linear_attn in_proj_* (30
      layers of non-standard targets) — partial at best; and the LoRA was
      trained for predicted-routing, mismatched under naive-K4.
  A converted franken-model represents the released pipeline on ZERO axes;
  building it would cost hours and prove nothing. NOT pursued (documented
  instead, per protocol).
- Executable proxy (faithful on its own axis): matched-quant 3.5-vs-3.6
  (both unsloth UD-IQ2_XXS) on OUR gates — isolates the pretrain delta and
  establishes control baselines Phase 4 needs. Kernel running (§6).

## 4. Published quality (their measurement, our axes uncovered)

Their OpenCompass (identical settings, edge0-35b int4+K4+adapters+prerouter
vs Qwen3.6 fp16): avg −3.9 (AIME 86.6/92.7, HumanEval 90.9/95.1, GPQA-D
79.8/81.8, MMLU-Pro 81.0/84.6, IFBench 57.9/61.7). Loss concentrates in
long-chain reasoning (AIME −6.1). NOT covered by them: Kiswahili, medical
safety/disposition, ADTC task, agentic behavior (they flag agents as weak).
Our gates cover exactly those — results pending.

## 5. Structural verdict (independent of pending gates)

The released MODEL cannot be submitted for our task: (a) no CPU/Ubuntu
runtime exists; (b) adapters don't transfer to our checkpoint (§2).
Per the phase brief, this is NOT an abandon-the-mechanism signal: Phase 1
was STRONG (K4+Q2_K expert path 39.6ms), so we proceed to Phase 3/4 with
"need our own adaptation." Remaining Phase-2 question for the kernels:
is the 3.6 BASE itself viable on our gates (informs Phase-4 base choice:
adapt 3.5, or switch to 3.6)?

## 6. Pending kernels (both RUNNING as of 16:05 UTC)

- `toheebogunade/jamii-native-sparse-q2k-quality-v1` v1: matched MMLU-100
  control vs q2k-experts vs q2k-all (Q2_K likelihood gate; reject if >2pp).
- `toheebogunade/jamii-native-sparse-edge0phase2-v1` v1: 3.5-vs-3.6 on
  MMLU-100 + 18 swahili (keyword-scored) + 24 heldout (captured for
  adjudicated grading) + tp_001/002; disk-safe sequential; deterministic
  (temp 0, seed 42); full outputs + timings saved.
Retrieve: `kaggle kernels output <slug> -p <dir>` when complete (~2–4h).
On arrival: adjudicate heldout outputs, fill §7 (gate table + deltas),
finalize verdict, commit, proceed to Phase 3.
