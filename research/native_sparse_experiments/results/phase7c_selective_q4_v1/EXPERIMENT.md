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

No valid performance or quality result was produced.

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
equals the source type in the selective-only mode.  This preserves unchanged
tensors by copy and permits only the explicit Q5_K-to-Q4_K overrides to be
requantized.  A third submission is required before this branch can be
classified.  No model binary is stored in this repository.

## Decision

`FOLLOW-UP ONCE`: the implementation blocker is narrow and mechanically
correctable.  Do not treat either failed attempt as a quantization result.
If the corrected A/B produces less than 3% whole-runtime improvement, kill
the selective-Q4 branch under the experiment value rule; if it wins, run the
held-out capability gate before keeping it.

## Artifacts

- Raw attempt 1: `raw/attempt_v1_harness_anchor_failure/`
- Raw attempt 2: `raw/attempt_v2_source_imatrix_failure/`
- Corrected harness: `kaggle/native-sparse-selective-q4-v1/`
