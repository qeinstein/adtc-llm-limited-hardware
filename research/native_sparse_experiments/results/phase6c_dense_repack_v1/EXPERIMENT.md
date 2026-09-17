# Phase 6C dense Q5_K repack v1 — harness failure

## Hypothesis

The standard mainline CPU weight-repacking path may improve the Q5_K attention
and Gated-DeltaNet projection workload used by Qwen3.5 decode.

## Result

The runtime built successfully and both CLI smoke commands completed, but the
benchmark arm was invalid. The pinned `llama-bench` help does not expose
`--repack` or `--no-repack`; both benchmark invocations returned the usage
page and `error: invalid parameter for argument`. No tok/s result was produced.

The CLI outputs were not accepted as correctness results because v1’s response
parser included the performance line, yielding hashes different from the
established control. This is a harness defect, not evidence of numerical
divergence. The raw outputs show successful model execution.

## Corrective action

Version 2 uses `llama-cli`, which accepts the repack flags, for three measured
64-token repetitions per arm and uses the control response parser. The model,
Q5_K projection family, prompt, threads, affinity, and exact native routing
remain unchanged.

## Configuration evidence

- Model: exact Qwen3.5-35B-A3B IQ2_XXS control, SHA-256 recorded in raw result
- llama.cpp: `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Hardware: 4-vCPU AVX2 Xeon Kaggle allocation, 4 threads
- Intended arms: `--no-repack` vs `--repack`
- Build: Release, CPU-only, `GGML_NATIVE=ON`

## Artifacts

Raw outputs are preserved in `raw/`. Corrective harness:
`kaggle/native-sparse-dense-repack-v2/`.
