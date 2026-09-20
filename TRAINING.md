# Jamii Afya post-training (TRAINING.md)

STATUS: scaffold + data pipeline implemented and executed; training NOT
executed (no training compute — see Compute boundary). No tuned checkpoint
exists. This file records the locked method so a provisioned run is mechanical.

## Base

- `Qwen/Qwen3.6-35B-A3B` @ `995ad96e` (Apache-2.0, `qwen3_5_moe` arch tag).
- 1045 tensors, 71,903,645,408 bytes BF16 (from `model.safetensors.index.json`).
- Language layers `model.language_model.layers.{0..39}`; visual tower and
  `mtp.*` layers exist in the checkpoint and are EXCLUDED (frozen, never
  adapted, absent from the deployment artifact).

## Method

> Final word: the one-shot preflight (`training/PREFLIGHT.md`) switched the
> plan to 4-bit QLoRA/FSDP2 and then returned NO-GO at the load gate
> (22.0 GB/rank vs 14.56 GB usable). Nothing below was executed.
> Preserved for a provisioned run.

- Originally BF16 LoRA (PEFT/TRL on Transformers v5, or Unsloth /
  MS-SWIFT Qwen3.5-MoE path); preflight later specified 4-bit QLoRA.
- Pilot r=8/alpha=32/dropout=0.05; r=16 only on clear pilot underfit.
- Completion-only loss (`assistant_only_loss=True`): final responses only,
  never synthetic CoT. Native thinking preserved; post-training tests run
  thinking enabled AND disabled.
- 1 epoch, conservative LR (pilot 1e-4, full 5e-5), cosine, grad-accum to
  eff. batch 16/32, grad checkpointing, max seq 2048, seeds data=7/train=11.
- Early stopping on held-out composite (medical + safety + general replay),
  eval every 50/100 steps, patience 2. Never train-loss alone.
- Frozen: embeddings, LM head (initially), router (`mlp.gate`),
  `shared_expert_gate`, norms, conv/A_log/dt_bias, visual, MTP.
- Framework default MoE aux loss; expert utilization monitored.
- DPO: gated (validated pairs only) else SKIP.

## LoRA targets (exact, verified against the weight index)

`q_proj k_proj v_proj o_proj` (10 full-attn layers) ·
`in_proj_qkv in_proj_a in_proj_b in_proj_z out_proj` (30 GDN layers) ·
`gate_up_proj down_proj` under `mlp.experts` (fused! 40 layers) ·
`gate_proj up_proj down_proj` under `mlp.shared_expert` (40 layers).

`train.py` asserts the adapted set at startup and aborts if any forbidden
module (router/head/embeddings/visual/MTP/norms) would receive adapters.

## Data

Mixture: `data/manifests/full_mixture.jsonl` (weights afrimed×2, medqa×1,
medmcqa×1, oasst×0.5); pilot: `data/manifests/pilot_mixture.jsonl` (1465).
Built by `training/build_dataset.py` (pinned revisions, quarantine,
contamination screen). See `data/DATA_CARD.md` + `data/LICENSE_LEDGER.json`.

## Compute boundary (training stops here until provisioned)

Measured/estimated VRAM for BF16 LoRA r=8, seq 2048, batch 1, checkpointing:

| component | bytes |
|---|---|
| BF16 weights (frozen) | 71.9 GB (measured) |
| LoRA params + grads + AdamW states (~0.5B trainable) | ~5–8 GB |
| activations (checkpointed) | ~25–45 GB |
| CUDA/framework overhead | ~5 GB |
| **total** | **~110–125 GB** |

Fits: 4×A100-40GB (160 GB, MS-SWIFT reference) or 2×80GB (A100/H100).
Does NOT fit: 1×80GB (weights+activations exceed it), any 24GB card, or
Kaggle free GPUs (2×T4-16GB = 32 GB; weights alone are 2.25× that).

- Wall time (est.): pilot ~1–2 h; full 1-epoch (~800 steps) ~8–16 h on 4×A100.
- Cost (est., on-demand 2026): 4×A100-40GB ≈ $6–8/h → ~$100–150/full run;
  2×H100-80GB similar. Spot/preemptible roughly half (checkpoint tolerant).
- Kaggle note: 6 h GPU quota is available on this account but the hardware
  (≤2×16 GB) cannot hold the weights; TPU (20 h) has no supported
  PEFT/TRL path for this stack.
- Launch note: at provision time `train.py` must be wrapped for multi-GPU
  (`torchrun` + FSDP/DeepSpeed sharding); the checked-in driver is the
  single-process reference and asserts the same module set either way.

## Free-tier QLoRA canary: attempted, FAILED at load (STOP per failure rule)

Axolotl FSDP2 + 4-bit QLoRA on 2×T4-16GB (Kaggle free), r8/α16, q/k/v/o
only, fp16/sdpa, seq 2048, microbatch 1, offload on. Kernel:
`kaggle/native-sparse-qlora-canary-v1` (v1–v7, 2026-09-20).

Proven working: env install (~5 min, all pins exact), bitsandbytes NF4 on
sm75, 72 GB snapshot download, `axolotl preprocess` (1465 rows tokenized,
chat template + completion-only masking verified), 2-worker launch.

Failure: CUDA OOM during weight loading on BOTH ranks (~46% of tensors,
14.21 GB in use of 14.56 usable per T4). The FSDP2-QLoRA path materializes
the full quantized model per rank at init (~19 GB: 35B×0.5B + NF4 overhead),
before FSDP wrapping/sharding or CPU offload can engage. Gap ≈ 30%.

Why tuning cannot fix it: seq len, rank, microbatch, and grad accumulation
affect step memory only; load memory is init-path-determined. The one
relevant knob (`cpu_ram_efficient_loading`) is explicitly forbidden with
4-bit by Axolotl's validator (deadlocks the scatter). Excluding vision/MTP
would save ~1.5 GB at most — still above 14.56 GB.

Verdict: free-tier T4 QLoRA of this 35B model is structurally blocked at
load. No paid-GPU fallback per user directive. The UNTUNED Q2_K+K4/16
artifact (A) remains the deployment; closeout continues with system
prompt + safety guard + guidance layer + harness + UI/HF/docs.

## Production preflight: NO-GO (2026-09-20, see training/PREFLIGHT.md)

Full A–Z preflight executed. Load gate RED by exact proof (22.00 GB per
rank vs 14.56 GB usable, +51%; Axolotl forces per-rank-full load and its
validator rejects the low-memory flag with FSDP2+4-bit; FSDP1 fallback
needs ~82 GB host RAM). Production NOT launched — the one shot is
preserved. Ready for provisioned hardware: `training/kaggle_train.yaml`,
`training/preflight.py`, `training/environment.lock.txt`,
`training/data_audit.json` (24,066 deduped rows, seq 2048, mask 100/100).

## After a winner (gated, §15–17 of the program brief)

1. `merge.py` → BF16 candidate (+SHA manifest).
2. `export_final.sh` → GGUF → Q2_K-experts-only quant (script refuses to
   bless output until the per-tensor override pass matches
   `configs/final_runtime.json`).
3. K4/16 + bounded-executor verification, tensor/type/hash checks.
4. 3-stage eval (native / tuned-BF16 / tuned-sparse) via `evaluate.py`.
5. Memory/speed regression (RSS, tok/s, TTFT, routes) on the deployment
   artifact. Ship only on clear product-objective win with no safety or
   general-capability regression; else KEEP THE UNTUNED MODEL.
