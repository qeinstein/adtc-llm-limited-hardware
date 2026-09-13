# Phase 6A ik_llama.cpp v1 — harness false-negative

## Hypothesis

The official `ik_llama.cpp` CPU implementation may provide a materially
stronger Qwen3.5 low-bit runtime baseline or useful donor path on the same
4-vCPU AVX2 Kaggle allocation.

## Result

The upstream checkout at
`3bb386eb68ffee0a5dc7db21da0735d594929eeb` configured and built both
`llama-cli` and `llama-bench`. Its `llama-cli --help` printed a complete usage
page but returned exit status 1. The v1 harness treated that conventional help
exit as fatal before downloading the model or running the smoke/benchmark.

Consequently this run has no valid model performance, RSS, output, or route
measurement and must not be interpreted as an ik_llama incompatibility.

## Corrective action

The failure is preserved as a harness defect. Version 2 accepts a nonzero help
status when usage text is present and continues with the current 64-token,
three-repeat configuration. It retains the same immutable upstream commit and
exact model identity.

## Configuration evidence

- Hardware: 4-vCPU Intel Xeon, AVX2, CPUs 0–3
- Build: CPU-only, `GGML_NATIVE=ON`, Release, `-j4`
- Upstream: `ikawrakow/ik_llama.cpp`, commit above
- v1 smoke configuration: 4 threads, context 512, fixed clinical prompt
- v1 benchmark configuration recorded `bench_gen=32` because it was submitted
  before the decode-length correction; v2 is the apples-to-apples run
- No model, weight, routing, or K change

## Artifacts

All raw Kaggle output is under `raw/`, including the successful build logs,
the help page, result JSON, and kernel log. Corrective harness:
`kaggle/native-sparse-ik-llama-v2/`.
