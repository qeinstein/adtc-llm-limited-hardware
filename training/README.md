# Jamii Afya post-training

Production LoRA post-training for African-context medical usefulness, safety
and calibration — on top of the FROZEN sparse deployment (K4/16 + Q2_K
routed experts + bounded executor, `configs/final_runtime.json`). Training
must not alter the deployment architecture; the tuned weights merge back
into BF16 and flow through the exact GGUF transformation (§15 of the
program brief).

**STATUS: scaffold + data pipeline implemented; training NOT executed —
no training compute is available in this environment (see
"Compute boundary" below). Nothing here has produced a tuned checkpoint.**

## Method (locked unless measurements say otherwise)

- Base: `Qwen/Qwen3.6-35B-A3B` @ `995ad96e` (Apache-2.0, qwen3_5_moe arch).
- BF16 LoRA (NOT QLoRA: unusually high quantization deltas on this
  family per Qwen3.5-MoE guidance). Pilot r=8, alpha=32; r=16 only on
  clear underfit. 1-epoch class, conservative LR, early stopping on
  held-out composite (never train loss alone).
- Completion-only loss (`assistant_only_loss=True`): train FINAL
  RESPONSES, never synthetic chain-of-thought. Native thinking preserved;
  post-training tests run thinking enabled AND disabled.
- Frozen: embeddings, LM head (initially), router (`mlp.gate`),
  `shared_expert_gate`, norms, visual tower, MTP layers. MoE aux loss per
  framework recommendation; load-balance behavior monitored, not retuned.
- DPO: OPTIONAL, gated on genuinely validated preference pairs; else SKIP.

## LoRA targets (exact, from the base `model.safetensors.index.json`)

1045 tensors, 71.9 GB BF16. Language layers
(`model.language_model.layers.{0..39}`):

| group | modules (PEFT `target_modules` substrings) | layers |
|---|---|---|
| full attention | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 10 (`self_attn.*`) |
| GDN projections | `in_proj_qkv`, `in_proj_a`, `in_proj_b`, `in_proj_z`, `out_proj` | 30 (`linear_attn.*`) |
| routed experts (fused!) | `gate_up_proj`, `down_proj` under `mlp.experts` | 40 |
| shared expert | `gate_proj`, `up_proj`, `down_proj` under `mlp.shared_expert` | 40 |

Excluded: `mlp.gate` (router), `mlp.shared_expert_gate`, `lm_head`,
`embed_tokens`, all norms, `linear_attn.conv1d/A_log/dt_bias` (not
linear), `model.visual.*`, `mtp.*`.

CAUTION: experts are fused `gate_up_proj` single tensors, not
gate/up `nn.Linear` pairs — verify the framework constructs LoRA on them
(Unsloth Qwen3.5-MoE or MS-SWIFT recommended; stock PEFT needs MoE-aware
wrapping). `train.py` asserts the adapted module set at startup and
records it; anything unexpected aborts before step 1.

## Layout

- `configs/lora_pilot_r8.yaml` — pilot (~1–2k examples, short run).
- `configs/lora_full_r8.yaml` — full mixture (same rank; scaled schedule).
- `build_dataset.py` — licensed pipeline: download → filter → dedup →
  quarantine → contamination screen → manifests + `LICENSE_LEDGER.json`.
- `train.py` — TRL SFTTrainer driver (assistant-only loss, early stop).
- `evaluate.py` — baseline/tuned/BF16/sparse 3-stage harness entrypoint.
- `merge.py` — merge adapter → BF16 candidate (+SHA).
- `export_final.sh` — BF16 → GGUF → Q2_K-experts quant → K4/16 verify.

## Data (see `data/DATA_CARD.md`)

Train: AfriMed-QA v2 train slice (saq+mcq WITH responses only, ≈1k —
consumer queries lack gold answers, eval-only), MedQA train, MedMCQA
train (stratified), PubMedQA `pqa_labeled`, OASST1 replay (vetted).
Eval-only: AfriMed test (internal `split` column, hashed/quarantined),
MedQA/MedMCQA/PubMedQA heldouts, MedSafetyBench, NigBench per terms.

## Compute boundary

BF16 LoRA on 35B needs ≈4×40GB (MS-SWIFT reference) or 2×80GB.
This environment has no GPU (N100 CPU, 2.6 GB RAM); Kaggle free GPUs
(16 GB) cannot hold 72 GB of BF16 weights. Training stops here until
provisioned GPU time is authorized — see TRAINING.md §Compute for the
exact configuration, VRAM math, wall-time and cost estimate.
