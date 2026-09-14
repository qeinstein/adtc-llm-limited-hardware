# Falcon production post-training audit

This document is the audit gate for the insurance submission built from
`tiiuae/Falcon-H1-1.5B-Deep-Instruct`. It supersedes the probe recipe; no
expensive production run is authorized by this document alone.

## Aborted probe

The deleted Kaggle v15 job is recorded in
[`falcon-probe-v15-aborted.json`](experiments/falcon-probe-v15-aborted.json).
It produced no valid checkpoint or score. Its main infrastructure failure was
buffered child-process output: the wrapper captured all training stdout and
printed it only after `trainer.train()` returned.

## Findings that must be fixed

1. The old loader masked prompts only indirectly through a custom loss. It did
   not expose labels, answer-token counts, or a fail-closed truncation policy.
2. Chat and MCQA truncation could construct sequences longer than the declared
   maximum when the prompt itself was too long. Production data building must
   reject or explicitly report these rows and preserve the answer/disposition.
3. `save_steps` created local Trainer checkpoints, but no complete resume path
   restored optimizer, scheduler, scaler, RNG, sampler position, and global
   step. `--resume_adapter` is not equivalent to a Trainer resume.
4. The old objective mixed MCQA ranking and generation loss without token-share
   accounting or independently logged component losses.
5. The old dataset had no frozen development manifest, quantitative provenance
   manifest, or cross-source near-duplicate/holdout contamination gate.
6. The notebook had no live heartbeat, structured JSONL metrics, durable
   mid-run checkpoint upload, or measured step-rate preflight.
7. The initial production-v1 source list had no generation-oriented general
   replay. MCQA ranking rows do not substitute for ordinary instruction
   following, so the corrected source list includes a small bilingual,
   project-authored set with explicit provenance rather than silently relying
   on MCQA to preserve generation behavior.

## Replacement gate

The replacement pipeline is split into deterministic dataset construction,
audit, staged training, checkpoint/resume verification, candidate evaluation,
merge/export, and deployment-model evaluation. Every stage writes a manifest
and is runnable from a clean checkout. A production run must not start until
the tiny resume test and short throughput benchmark pass.

The first corrected generation-smoke rerun (Kaggle v7) completed the exact
data audit/build but failed at the script import boundary before loading model
weights. That failure is preserved as an aborted infrastructure experiment;
the launch-path fix is included in the next version.

## Trainability proof

Trainability proof v1 is preserved as
[`falcon-trainability-proof-v1-20260913.json`](experiments/falcon-trainability-proof-v1-20260913.json).
It failed before model loading because the standalone script did not expose
the repository root to Python imports; it produced no optimizer state or model
evidence.

Proof v2 is preserved as
[`falcon-trainability-proof-v2-20260913.json`](experiments/falcon-trainability-proof-v2-20260913.json).
On a P100/sm_60 in FP16, 32 deterministic rows (16 SFT, 16 MCQA) over 64
steps changed all 528 trainable LoRA tensors, produced a logit delta RMS of
9.35 (max 57.0), reduced both objective losses materially, and had exact
adapter-off/stock and save/reload agreement. The adapter changed all four
generation checks. This is a **PASS for trainability only**, not a quality
candidate: the deliberately overfit generations became repetitive referral
text. No real-data training is authorized until prompt and small-data quality
probes provide evidence.

Proof v3 is the current implementation confirmation, recorded in
[`falcon-trainability-proof-v3-20260913.json`](experiments/falcon-trainability-proof-v3-20260913.json).
It repeats the same pass with the instrumented loss summary and all 16 SFT
fixture generations: 16/16 generations changed, SFT loss fell 61.15%, and
MCQA loss fell 94.27%. The quality rejection remains unchanged.

## Bounded real-data follow-up

The first real-data pilot is preserved as
[`falcon-small-real-probe-v1-20260913.json`](experiments/falcon-small-real-probe-v1-20260913.json).
It was rejected because 8 optimizer steps at 5e-6 changed neither MCQA nor
any of the 16 development/validation generations.

The stronger canonical follow-up ran on Kaggle kernel v25 and is preserved as
[`falcon-small-real-followup-v1-20260914.json`](experiments/falcon-small-real-followup-v1-20260914.json).
It completed 16 steps on a P100 in 38m30s at 7.27 tokens/s, persisted steps
8 and 16, and selected step 16 by fast development loss (2.9065). It passed
the responsiveness test: 12/16 development/validation generations changed,
fast-dev MCQA moved from 40% to 60% (acc_norm 80% to 100%), and SFT loss moved
from 3.5880 to 3.4842. It failed the quality test: the development battery
remained 0/8 and validation 2/8 under the embedded rubric, with critical
failures on emergency/safety and fabricated-protocol cases. It is rejected
for promotion and export; no frozen final holdout was read.

The bounded attention-plus-Mamba micro-probe ran on Kaggle kernel v2/1 and is
preserved as
[`falcon-small-real-micro-probe-v2-20260914.json`](experiments/falcon-small-real-micro-probe-v2-20260914.json).
It used the corrected 382-row mixture (346 train / 36 dev), with 80.84% of
loss tokens from SFT and 19.16% from MCQA, and completed 16 steps at 6.91
tokens/s on a P100. The adapter changed 6/16 development/validation
generations, but did not move the 25-row MCQA slice (48% raw, 60% acc_norm)
and did not improve the quality gate: development remained 0/8 and validation
2/8, with critical failures on d01/d04/d05 and v01/v03/v04. It is rejected
for promotion and export. The frozen final holdout was not touched, and only
small JSON/log summaries were retrieved locally; no model or adapter files
were downloaded.

## Data review and current authorization

The exact v25 mixture contained 142 SFT and 32 MCQA rows (30,051 and 11,474
all-sequence tokens respectively), but the loss-token share was 97.38% SFT
versus 2.62% MCQA. The source audit now flags 15 MCQA-shaped rows stored as
SFT, 25 authority/protocol claims, 5 numeric medication examples, 8
toxin/disinfectant examples, 5 invasive-procedure examples, and 74 repeated
disclaimer rows for manual review. These are review flags, not automatic
deletions.

The executable production data policy now excludes only the 15
`mcqa_shaped_sft` rows from `project_clinical_generation`; the public MCQA
train-only source remains available as the explicit MCQA objective. Each
excluded row is retained in the source file and recorded as
`quality_excluded` in the next `data_manifest.json` rather than disappearing
silently.

Kaggle audit v26 demonstrated that the previous 2,000-record-per-dataset
default produced 14,894 MCQA rows and only 127 SFT rows: 93.64% of loss tokens
were MCQA and 99.64% of packed tokens were MCQA. That configuration is
rejected in [`falcon-data-audit-v26-20260914.json`](experiments/falcon-data-audit-v26-20260914.json).

The corrected v27 audit uses a 250-record-per-dataset cap and two letter-order
permutations. It produced 1,962 MCQA and 127 SFT rows, with 64.58% MCQA and
35.42% SFT by loss tokens; packed-token share remains 97.16% MCQA because each
MCQA choice reuses the context. The result and hashes are recorded in
[`falcon-data-audit-v27-20260914.json`](experiments/falcon-data-audit-v27-20260914.json).
The trainer therefore samples by configured loss-token exposure and logs the
raw packed and objective-token views separately.

The audit notebook now invokes the trainer's tokenizer-only preflight. Kaggle
v28 passed it before model construction: 1,773 MCQA and 116 SFT training items,
64.12% MCQA / 35.88% SFT by exact loss tokens, 100 explicitly bounded MCQA
context truncations, and no weight load. See
[`falcon-tokenized-preflight-v28-20260914.json`](experiments/falcon-tokenized-preflight-v28-20260914.json).

The independent trainability gate and durable resume/persistence gate pass;
the real-data quality gate and system-prompt selection gate do not. Therefore
another long training run is **not authorized**. The next experiment must use
the audited data mixture and quality reports, remain development/validation
only, and demonstrate a genuine safety/usefulness improvement before scaling.
