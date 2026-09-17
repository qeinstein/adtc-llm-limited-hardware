# Technical Report — Jamii Afya Falcon Submission

**Team ID:** jamii-afya
**Domain:** healthcare_medical
**Active model:** `Falcon-H1-1.5B-Deep-JamiiAfya-Q4_K_M.gguf`
**Base:** `tiiuae/Falcon-H1-1.5B-Deep-Instruct`
**Base revision:** `b6648636ddc906688974282de6e7a243395f5423`
**Runtime:** llama.cpp / GGUF, CPU-only at deployment
**Languages:** English and Kiswahili

## Submission decision

The active submission is Falcon-H1-1.5B-Deep-Instruct at the exact pinned
revision above. The earlier 16-step P100/sm60 adapter is retained only as an
emergency fallback record; it is not part of the production trajectory.

Final training is run on Kaggle T4 x2, or another NVIDIA accelerator with
compute capability sm75 or newer. Startup checks fail before model loading when
the capability or the optimized `mamba-ssm` plus `causal-conv1d` path is absent.
The P100 naive-Mamba path is not an accepted training environment.

## Training and export

`scripts/build_falcon_submission_sft.py` constructs the fixed audited corpus
from project safety data, accepted normal clinical data, general/conversational
data, and public MCQA train splits. MCQA is ordinary SFT consisting of a user
question with choices and a concise correct answer; no chain-of-thought or
listwise multi-forward loss is used. Safety rows have two renderings with the
same answer: one with the compact Jamii Afya system prompt and one without it.
The builder measures assistant-token mass and adds Kiswahili rows until its
assistant-token share is at least 10%.

The single adapter trajectory is:

| Stage | Steps | Learning rate | Supervised-token mixture |
|---|---:|---:|---|
| Domain lock-in | 96 | 3e-5, 5% warmup, cosine | 40% safety / 30% clinical / 15% general / 10% MCQA / 5% Kiswahili boost |
| Safety polish | 24 | 1e-5, no warmup | 60% safety / 20% clinical / 10% general / 5% MCQA / 5% Kiswahili boost |
| Capability replay | 16 | 5e-6, no warmup | 10% safety / 20% clinical / 35% general / 30% MCQA / 5% Kiswahili boost |

Training is fp16 LoRA (`r=16`, `alpha=32`, dropout `0.05`, bias `none`) over
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj`, `gate_proj`, `up_proj`, and
`down_proj`. `out_proj` and `conv1d` are forbidden targets. The loop uses
512-token packed sequences, batch 2 per device, gradient accumulation 4,
`ddp_find_unused_parameters=false`, no gradient checkpointing, and AdamW with
betas `(0.9, 0.95)`, zero weight decay, and max gradient norm 1.0.

Only Stage1-step96, Stage2-final, and Stage3-final can be selected. The frozen
clinical/safety gate is applied in Stage3 → Stage2 → Stage1 order. If none
passes, stock Falcon-H1 at the pinned revision is exported as the emergency
fallback. No DPO, ORPO, KTO, PPO, GRPO, ablation, or checkpoint search is part
of this sprint.

The selected adapter is merged with `merge_and_unload()`, checked for forbidden
targets, and compared with adapter-on deterministic generation before GGUF
conversion. The sole deployment quantization is Q4_K_M.

## Validation and profiling

Fast development generations run only at Stage1 steps 48 and 96, Stage2 final,
and Stage3 final, with `max_new_tokens=96`. A critical development violation is
recorded as invalid but does not alter the predetermined later stages.

The final validator evaluates the exact Q4_K_M GGUF for the frozen
clinical/safety battery with and without the system prompt, general reasoning,
MCQA likelihood, English, Kiswahili, and repetition/EOS behavior. The exact
GGUF is also profiled three times using:

```text
llama-bench -m Falcon-H1-1.5B-Deep-JamiiAfya-Q4_K_M.gguf -p 512 -n 128 -ngl 0
```

The export and validation manifests record the final file's SHA256 and exact
byte size, along with the base revision, adapter commit, and training-config
hash. The active download script verifies the SHA256 before accepting the
artifact.

Historical Qwen experiments and the earlier P100 Falcon run remain research
records; they are not active submission artifacts.

*Medical content is for clinical decision support only and is not a substitute
for assessment by a qualified clinician.*
