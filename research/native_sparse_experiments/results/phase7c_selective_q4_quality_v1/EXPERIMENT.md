# Phase 7C selective-Q4 initial quality gate v1

## Purpose

This was the first held-out quality screen for the selective-Q4 challenger
that was 7.14% faster on resident decode. It compared the exact IQ2_XXS
control and the candidate on eight clinical/safety/instruction probes and six
basic MCQs, with identical runtime settings and deterministic seed.

## Result and validity

Both models ran successfully and preserved the same short deterministic
control response where applicable. However, `-n 48` was insufficient for this
Qwen configuration with automatic reasoning enabled: most outputs ended in
the thinking block before an answer. The rubric therefore scored both arms
very low and the MCQ parser found no completed final answers. Aggregate
scores were identical (clinical 2/8, safety 1/7, MCQ 0/6), but this is an
inconclusive measurement, not evidence of equivalent broad quality.

The raw outputs are retained under `raw/attempt_v1_inconclusive/`. The result
is explicitly superseded as a quality decision by the reasoning-disabled v2
gate, which keeps the same frozen prompts and adds an answer-completion
condition.

## Decision

`INCONCLUSIVE / FOLLOW-UP ONCE`: rerun the same gate with `--reasoning off`
and a parser that removes the prompt/banner when no thinking marker exists.
Do not accept or reject the Q4 representation based on this run.

## Configuration

- Model/checkpoint: control SHA-256
  `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b` and
  selective-Q4 candidate SHA-256
  `86f27f69e0cea7f8cd0420452f1b600c57cedce6589f60c2cfee06a8c0395d39`
- llama.cpp: `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Kaggle: 4-vCPU Intel Xeon AVX2, four threads, CPU-only, mmap, poll 0
- Prompt settings: context 768, 48 generated tokens, temperature 0, seed
  1234, no runtime repack
- Harness: `kaggle/native-sparse-selective-q4-quality-v1/`
- Corrected follow-up: `kaggle/native-sparse-selective-q4-quality-v2/`
