# Falcon production post-training pipeline

Status: implementation/audit gate; the bounded pilot is archived as invalid
because it used the pre-audit chat target and its generation evaluation exposed
that mismatch. No production checkpoint has been accepted.

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
`-100`, and every assistant target token is trained. The target is derived from
the tokenizer's complete Falcon chat rendering, including the assistant-turn
`<|im_end|>` boundary; generic `<|end_of_text|>` is not substituted. Prompts
are left-truncated only after verifying the complete answer fits within
`max_length`; MCQA answer continuations are never truncated or partially
trained. To keep P100 activation memory bounded, unusually long MCQA contexts
use a recorded 384-token head+tail window during training and fast-dev
scoring. Final deployment MCQA evaluation remains a separate full-context gate;
the data manifest records every context reduced by the bounded window.
MCQA ranking uses the same character-normalized continuation score as
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

Stage B/C initialize from the previous stage's selected adapter/checkpoint but
start fresh optimizer/scheduler state; `--resume-from-checkpoint` is reserved
for resuming the same stage. The stages are separate checkpointed runs. Each
stage writes `checkpoint_selection.json` from complete checkpoints and then
tests up to the best loss-ranked candidates with the frozen generation gate.
All tested candidates are retained in `quality_selection.json`; among those
that pass the hard safety veto, the candidate with the highest frozen-battery
pass rate wins, with eval loss as a tie-breaker. A stage with no evaluation or
no quality-passing candidate is not promotable.

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
checkpoint state through `scripts/persist_checkpoint.py`; the Trainer callback
requests an additional save at the persistence boundary when ordinary
`save_steps` would be slower. `--allow-ephemeral`
is reserved for the tiny resume test. The reserved private destination is
`toheebogunade/jamii-afya-falcon-production-checkpoints`. The infrastructure
gate passed on 2026-09-12: a two-step checkpoint resumed at step 2, completed
through step 4, uploaded to the Dataset, and was retrieved with optimizer,
scheduler, RNG, scaler, adapter, and Trainer state intact. The uploader and
verifier wait for Kaggle's asynchronous Dataset version creation, so an upload
cannot be mistaken for durable storage before the new files are visible.

HF generation uses the same contract as training: the stop set is built from
the configured EOS IDs plus `<|im_end|>` only when that token is present in the
tokenizer vocabulary. It never uses `convert_tokens_to_ids` as an existence
test, so PAD/unknown IDs cannot terminate generation. The first remote smoke
run exposed this exact failure and is not accepted as evidence; a corrected
smoke must pass before training resumes.

## Current gate

The tracked raw-data audit found 142 clean project SFT rows and no exact or
high-similarity leakage against the 63 frozen final prompts. The clean-worker
build now yields `15,036` accepted rows (`13,564` train and `1,472` dev;
`30,051` SFT tokens and `7,426,390` MCQA tokens before loss masking) after the
capped train-only sources. The exact token/facet manifest is written by the
builder. Training now uses a deterministic 64-row fast-dev subset during
stage runs; larger frozen batteries remain separate final evaluations. The
P100 resume gate measured roughly 8--13 loss tokens/sec and about 145 seconds
per optimizer step at effective batch 16, so one full epoch is not an
acceptable production schedule. A bounded pilot must establish the stage
step budget and quality curve before any long run.

## Archived bounded pilot

Kaggle kernel version 4 completed eight Stage-A optimizer steps on
2026-09-12 (`falcon-production-v1-pilot-20260912.json`). It persisted a
technically resumable adapter checkpoint, but it is not a candidate: the run
was made before the shared exact-chat formatter was added, and its generation
battery emitted only 1--3 visible characters for almost every prompt. Its
`52.38%` fast-dev `acc_norm` is therefore a training smoke result only, not a
quality claim. The checkpoint remains in Kaggle history and must not be used
for export or submission.

Kaggle version 7 reached the smoke stage after the complete data build but
failed before model loading because the standalone smoke script lacked the
repository root on `sys.path` (`ModuleNotFoundError: scripts`). It is recorded
in `falcon-generation-smoke-v7-20260913.json`; this is an infrastructure
failure, not a model result.

Version 8 fixed that import boundary and loaded the model, but its generation
trace still decoded to `As` with raw IDs `[4638, 0, 0, 0, 0, 0, 0, 0]` despite
the correct stop set `[11, 228]`. It is recorded in
`falcon-generation-smoke-v8-20260913.json` and is not a valid baseline. The
next smoke records generation score steps so padding after an early stop can
be distinguished from the model actually selecting PAD.

Version 9 identified the distinction: score step 1 was finite, then all later
scores were `NaN`; argmax over those non-finite logits produced ID 0. This is a
P100/sm_60 FP16 numerical failure, not a tokenizer stop failure. P100 HF
Falcon loading now uses FP32 for smoke and autoregressive evaluation;
sm_70+ retains the faster supported low-precision path.

The attempted FP32 training resume gate then OOMed on the 16-GB P100 during
the first backward pass. The production config therefore separates the paths:
P100 training uses explicit plain FP16 to fit memory, with finite-loss fail-fast;
P100 HF autoregressive evaluation remains FP32. The OOM is recorded in
`falcon-resume-gate-v13-20260913.json`; the next gate tests whether FP16
teacher-forcing training is numerically stable.

Version 15 answered that question: both FP16 optimizer steps completed with
finite losses (`3.62` final loss, grad norm `2.07`) at about `158 s/step`.
The run then OOMed during dev evaluation because four MCQA items created a
19.5-GiB Falcon-H1 Mamba intermediate. The evaluation batch is now forced to
one; this is recorded in `falcon-resume-gate-v15-20260913.json`.

Version 10 reran the same smoke in FP32 and produced eight finite non-PAD
tokens (`As Jamii Afya, I ur...`). The numeric gate now passes; a 64-token
smoke is being used to inspect completion behavior before the notebook returns
to audit-only mode.

Version 11 (64-token cap) confirms the stock baseline’s response-quality
problem: FP32 remains finite, but no stop token is emitted, the answer is cut
mid-sentence, it includes a questionable `cardiac arrest` differential, and it
does not reach referral disposition. This is recorded in
`falcon-generation-smoke-v11-20260913.json`; it is precisely the behavior the
held-out post-training suite must improve without harming general capability.

Version 16 passes the infrastructure gate: FP16 P100 training is finite at
about `149 s/optimizer step`, batch-1 dev evaluation avoids OOM (`231.8 s` for
64 rows), global step resumes `2 -> 4` with optimizer/scheduler/RNG state,
step 4 is selected by lower eval loss (`2.0814`), and that checkpoint is
uploaded to and retrieved from the private Kaggle Dataset. Full details are in
`falcon-resume-gate-v16-20260913.json`.

## Archived controlled pilot v18

Pilot `falcon-production-v1-pilot-20260913T141934Z` completed eight Stage-A
steps on a P100 and persisted a complete step-8 checkpoint. It is **not a
candidate**. Fast-dev `acc_norm` was `53.9683%`, but the frozen generation
battery showed widespread truncation, incoherent Kiswahili-like continuations,
missing dispositions, an invented WHO protocol, a guessed malaria dose, and
unsafe deworming guidance. The full record is
`docs/research/experiments/falcon-pilot-v18-20260913.json`. Stage B/C and export
are explicitly vetoed for this adapter; the next experiment must compare safer
lower-rate/target-module settings with evaluations at multiple optimizer steps.

## Archived stock-vs-adapter comparison v23

Kernel v23 performed the required apples-to-apples comparison using the same
P100 FP32 evaluator, 64-row objective-stratified fast-dev set (14 SFT and 50
MCQA), production system prompt, and 64-token generation cap. Stock scored
`52.0%` MCQA / `56.0%` acc_norm with SFT loss `3.588034`; the durable step-8
adapter scored `52.0%` / `56.0%` with SFT loss `3.585137`. Twenty-three of 24
held-out raw generations were byte-identical; only h01 differed. The adapter
therefore has no measured quality gain and is not promotable. The complete
record is `docs/research/experiments/falcon-pilot-v23-compare-20260913.json`.

The frozen generation battery is now backed by
`docs/research/falcon_generation_rubric.json` and
`scripts/score_falcon_battery.py`. It is a conservative machine veto layer,
not a substitute for clinical review: missing output, critical safety failure,
or fabricated-protocol acceptance rejects promotion while preserving every raw
generation. The Kaggle notebook runs it after persisted evaluations and before
selected-stage checkpoint persistence, so a low dev loss cannot silently
promote an unsafe candidate.
