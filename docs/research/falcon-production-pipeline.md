# Falcon production post-training pipeline

Status: implementation/audit gate; no production training run launched.

## Canonical inputs

- model: `tiiuae/Falcon-H1-1.5B-Deep-Instruct`
- pinned model/tokenizer revision: `b6648636ddc906688974282de6e7a243395f5423`
- config: [`configs/falcon-production-v1.json`](../../configs/falcon-production-v1.json)
- dataset builder: [`scripts/build_falcon_dataset.py`](../../scripts/build_falcon_dataset.py)
- trainer: [`scripts/train_falcon_production.py`](../../scripts/train_falcon_production.py)
- streamed runner: [`scripts/run_streamed.py`](../../scripts/run_streamed.py)
- Kaggle entrypoint: [`kaggle/phase04-falcon-production/phase04_falcon_production.ipynb`](../../kaggle/phase04-falcon-production/phase04_falcon_production.ipynb)

## Correctness decisions

The production trainer uses completion-only SFT labels: every prompt token is
`-100`, every assistant target token is trained, and EOS is appended to every
target. Prompts are left-truncated only after verifying the complete answer fits
within `max_length`; MCQA rows are rejected instead of partially truncating a
choice. MCQA ranking uses the same character-normalized continuation score as
the ADTC `acc_norm` objective, with a small gold-token NLL auxiliary term. The
sampler has a separate per-stage token-share target. Each row receives its
objective share divided by that objective's total loss-token mass, so the
expected sampled loss-token exposure matches the configured share rather than
silently becoming row-balanced or favoring short rows.

Falcon-H1 is hybrid attention/SSM. The LoRA target list therefore includes its
attention projections (`q/k/v/o`), Mamba projections (`in_proj/out_proj`), and
MLP projections (`gate/up/down`). The trainer validates that every configured
target exists as a linear module before training.

## Stages

1. `stage_a_capability_preserving`: low-rate SFT with light MCQA replay.
2. `stage_b_clinical_safety`: clinical/refusal/disposition specialization.
3. `stage_c_mcqa_replay`: controlled MCQA replay while retaining SFT mass.

Stage B/C initialize from the previous stage's adapter but start fresh
optimizer/scheduler state; `--resume-from-checkpoint` is reserved for resuming
the same stage. The stages are separate checkpointed runs. Selection must use validation and
the frozen final batteries, not the last checkpoint or training loss alone.

## Observability and resume

The trainer writes timestamped `events.jsonl`, `training_metrics.jsonl`,
`heartbeat.jsonl`, streamed persistence logs, `environment.json`,
`run_manifest.json`, checkpoint manifests, and `final_summary.json`. Training
metrics include component losses, optimizer step, learning rate, gradient norm,
loss-token/example throughput, GPU memory, elapsed time, and ETA. A background
heartbeat reports even when a single forward/backward step is slow. Hugging Face Trainer checkpoints retain
adapter weights, optimizer, scheduler, scaler/RNG state, and global step; the
`latest` resume path passes the checkpoint to `trainer.train(resume_from_checkpoint=...)`.

Long runs refuse to start unless `FALCON_CHECKPOINT_DATASET` names an existing
private Kaggle Dataset. Every configured persistence interval uploads the
checkpoint state through `scripts/persist_checkpoint.py`. `--allow-ephemeral`
is reserved for the tiny resume test. The reserved private destination is
`toheebogunade/jamii-afya-falcon-production-checkpoints`. The infrastructure
gate passed on 2026-09-12: a two-step checkpoint resumed at step 2, completed
through step 4, uploaded to the Dataset, and was retrieved with optimizer,
scheduler, RNG, scaler, adapter, and Trainer state intact. The uploader and
verifier wait for Kaggle's asynchronous Dataset version creation, so an upload
cannot be mistaken for durable storage before the new files are visible.

## Current gate

The tracked raw-data audit found 118 clean project SFT rows and no exact or
high-similarity leakage against the 63 frozen final prompts. The clean-worker
build currently yields 13,542 train and 1,470 dev rows after the capped
train-only MCQA sources, with the exact token/facet manifest written by the
builder. Training now uses a deterministic 64-row fast-dev subset during
stage runs; larger frozen batteries remain separate final evaluations. The
P100 resume gate measured roughly 8--13 loss tokens/sec and about 145 seconds
per optimizer step at effective batch 16, so one full epoch is not an
acceptable production schedule. A bounded pilot must establish the stage
step budget and quality curve before any long run.
