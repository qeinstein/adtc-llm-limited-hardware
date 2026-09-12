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

## Replacement gate

The replacement pipeline is split into deterministic dataset construction,
audit, staged training, checkpoint/resume verification, candidate evaluation,
merge/export, and deployment-model evaluation. Every stage writes a manifest
and is runnable from a clean checkout. A production run must not start until
the tiny resume test and short throughput benchmark pass.
