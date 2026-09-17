# Phase 3 v1 — single-row IQ2_XXS AVX2 selected-expert path

## Hypothesis

Calling the existing exact IQ2_XXS AVX2 8-output-row GEMV directly for each
one-row selected expert would remove the existing IQ panel's four-lane tail
duplication and raise the resident decode ceiling.

## Reproducibility

- Kaggle kernel: `toheebogunade/jamii-native-sparse-single-row-avx2`, version 1
- Research source: commit `26ca4b9`
- Runtime source: llama.cpp `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Checkpoint: `Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf`, 10,656,955,008 bytes,
  SHA-256 `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- Model repository commit: `bc014a17be43adabd7066b7a86075ff935c6a4e2`
- Hardware: Kaggle, 4-vCPU / 2-core Intel Xeon @ 2.20 GHz, AVX2/FMA,
  one socket, 55 MiB L3
- Build: Release, `GGML_NATIVE=ON`, CPU-only, static, four build jobs
- Benchmark: resident mmap, lazy off, CPU-only, four threads, poll 0,
  prompt 0, 64 decode tokens, three repetitions
- Logical selected payload: 280,494,080 bytes/token; fresh scheduled traffic:
  0 bytes/token

## Kernel design

The opt-in arm adds a `cne1 == 1` branch to `MUL_MAT_ID`. It decodes the
selected expert's eight output rows into the existing IQP panel representation
and calls the existing `iqp_gemv_8x8_q8_K` once for the one activation row.
The native router, K=8, selected IDs, weights, IQ2_XXS values, and graph are
unchanged. The generic vec-dot path remains the control.

## Results

| arm | samples tok/s | mean tok/s | ms/token | peak RSS |
|---|---:|---:|---:|---:|
| generic control | 4.917, 4.996, 5.005 | **4.973** | 201.095 | 10,527.7 MiB |
| single-row IQP | 3.012, 3.034, 3.015 | **3.020** | 331.084 | 10,532.3 MiB |
| single-row IQP + profiling | 2.880, 2.934, 2.881 | 2.898 | 345.017 | 10,526.3 MiB |

The optimized arm is 39.26% slower than the generic control. The profile arm
is intentionally not used as the throughput result because its TSC reads and
atomic counters add overhead.

The profile recorded 23,715,840 decode and GEMV calls over the 192 benchmark
tokens. Aggregate TSC cycles were:

- panel decode: 290,594,682,999 cycles, 88.58% of instrumented IQP cycles
- GEMV: 37,467,195,558 cycles, 11.42%
- per call: approximately 12,253 decode cycles and 1,580 GEMV cycles

Thus the attempted optimization removed the wrong cost: decoding eight rows
dominates the direct GEMV and is substantially more expensive than the
generic selected-row dot path.

## Correctness

Both deterministic 24-token smoke runs completed successfully. Generated
payloads are byte-identical and hash to
`a3c3c73a067cbb5fab9e68bd7d06c03d3c488e74467d33c40fae339836006c8c`, matching
Phase 0. Native K=8, 40 layers, routing, weights, and quantization were not
changed; no expert was dropped or substituted.

## RAM-floor attempt

The lazy mmap arms logged RSS before the first routed-expert event and after
short decode:

| arm | pre-first-route RSS | pre-first-route anon/file | peak RSS |
|---|---:|---:|---:|
| lazy, context 64, 1 token | 629.2 MiB | 446.2 / 274.8 MiB | 4,847.4 MiB |
| lazy, context 512, 1 token | 635.5 MiB | 456.2 / 277.9 MiB | 4,868.9 MiB |
| lazy, context 512, 64 tokens | 642.8 MiB | 456.2 / 281.7 MiB | 6,538.9 MiB |

The pre-first-route value is an early loader/context boundary, not yet the
true non-routed resident floor: mmap-backed non-routed trunk pages are touched
later. The next version therefore adds a measurement-only `SKIP_MOE` arm that
executes the graph while zeroing `MUL_MAT_ID` outputs before routed expert
access. Logical model accounting is 8,975,810,560 routed expert bytes and
1,681,144,448 remaining model bytes; resident attribution still requires the
skip arm.

## Conclusion and next experiment

This exact single-row dispatch is rejected as a speed optimization. The
dominant measured term is one-row IQ2_XXS dequantization/panel construction,
not GEMV. The next bottleneck-driven experiment is a true one-row decoder that
reads one raw IQ2_XXS row and fuses its dequantization with the dot product,
followed by the `SKIP_MOE` RAM-floor measurement.

Raw Kaggle output is in `native-sparse-single-row-avx2-results/`.
