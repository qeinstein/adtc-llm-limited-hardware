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
