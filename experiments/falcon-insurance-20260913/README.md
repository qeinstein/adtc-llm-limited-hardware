# Falcon insurance experiments — 2026-09-13

This directory is metadata-only. Model weights, adapters, GGUFs, and Kaggle
worker outputs stay off the local filesystem.

## Gates

| Experiment | Status | Decision |
|---|---|---|
| deployment baseline v1 | aborted: missing `/usr/bin/time` | follow-up |
| deployment baseline v2 | aborted: post-exit `psutil` race | follow-up |
| trainability proof v1 | aborted: import boundary | reject |
| trainability proof v2/v3 | movement proof passed; quality intentionally rejected | follow-up |
| system-prompt search | Kaggle evaluation in progress | pending |

The trainability proof authorizes only a bounded real-data probe after prompt
selection and frozen-gate measurement. It does not authorize production-scale
training. A candidate is never promoted without the frozen clinical/safety
veto, usefulness/regression checks, and deployment-model evaluation.

## Artifact policy

Small machine-readable experiment records live in
`docs/research/experiments/`. Raw generations and large checkpoints remain in
Kaggle outputs or the private checkpoint dataset and are referenced by kernel,
version, SHA, and manifest rather than committed to Git.
