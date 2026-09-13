# Phase 7C selective Q4 projection challenger v1 — quantizer blockers

## Hypothesis

Requantizing only the measured Q5_K attention/Gated-DeltaNet projection
families to Q4_K might reduce dense decode cost enough to move the resident
throughput frontier.  The router, native K=8 routing, expert weights, and all
non-selected tensor types were intended to remain unchanged.

## Configuration

- Model: Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf, SHA-256
  `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- llama.cpp: `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Target regex: `attn_(qkv|gate|q|k|v|output)\\.weight=q4_k`
- Intended benchmark: CPU-only, 4 threads, AVX2, 64 generated tokens,
  three repetitions, resident mmap, poll 0, no runtime repack.
- Research harness source: initial attempt based on `ff24614`; corrected
  retry source is preserved in the current Kaggle harness.

## Result

The first two attempts below produced no valid performance result. The
corrected third attempt completed the planned resident A/B:

| arm | samples tok/s | mean tok/s | ms/token | peak RSS MiB | file size |
|---|---|---:|---:|---:|---:|
| exact IQ2_XXS control | 4.2, 4.2, 4.2 | 4.200 | 238.095 | 10,430.4 | 10,656,955,008 B |
| selective attention/GDN Q4_K | 4.5, 4.5, 4.5 | 4.500 | 222.222 | 10,307.8 | 10,528,504,448 B |

The candidate is 7.1429% faster in this three-repeat run, uses 122.6 MiB
less peak RSS, and is 1.2053% smaller on disk. The candidate model SHA-256
is `86f27f69e0cea7f8cd0420452f1b600c57cedce6589f60c2cfee06a8c0395d39`.
This is a meaningful but not yet accepted frontier movement because the
candidate changes selected projection weights.

Attempt 1 stopped in the harness because its source patch anchor did not
match the pinned quantizer.  Attempt 2 fixed that anchor, built the pinned
runtime, downloaded and verified the model, then stopped during quantization:

```text
ERROR: this quantization requires an importance matrix!
- offending tensor: blk.0.ffn_down_exps.weight
- target type: iq2_s
```

The failure is caused by the selective-only patch returning each unmatched
tensor's existing type while the quantizer still performs its preflight
importance-matrix requirement check for that very-low-bit type.  It is not
evidence that Q4_K execution is slower or that the model's Q5_K projections
are numerically unsuitable.

## Corrective action

The harness now also suppresses the imatrix requirement when the target type
equals the source type in the selective-only mode. This preserves unchanged
tensors by copy and permits only the explicit Q5_K-to-Q4_K overrides to be
requantized. No model binary is stored in this repository.

## Decision

`QUALITY GATE RUNNING`: the corrected A/B exceeds the 3% keep threshold, but
the selective-Q4 branch is not accepted until the held-out clinical/safety,
English/Kiswahili instruction, and basic MCQ gate is complete. If that gate
shows a material capability or safety regression, kill the branch despite the
speed/RSS win. The quality gate is a small initial screen, not a broad model
quality claim.

## Artifacts

- Raw attempt 1: `raw/attempt_v1_harness_anchor_failure/`
- Raw attempt 2: `raw/attempt_v2_source_imatrix_failure/`
- Valid third attempt: `raw/attempt_v3_valid/`
- Corrected harness: `kaggle/native-sparse-selective-q4-v1/`
- Quality gate harness: `kaggle/native-sparse-selective-q4-quality-v1/`
