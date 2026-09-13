# Phase 5A-v2 — split non-MoE matrix work and LM-head Amdahl bound

## Hypothesis

The first Amdahl profile grouped a large amount of non-routed matrix work as
`other_matmul`.  Split it by weight family and remove only the LM head in a
measurement-only arm to identify the next system-level CPU target.

## Reproducibility

- Kaggle kernel: `toheebogunade/jamii-native-sparse-end-to-end-amdahl-profile-v2`
- Runtime: llama.cpp `3057bb66c86c46d5781e50e85462a760ba7d1feb` with the
  existing exact native-routing control and measurement hooks.
- Model: Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf, SHA-256
  `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`.
- Kaggle 4-vCPU / 2-core AVX2 Xeon, four threads, `-lzm off`, `--poll 0`,
  `-n 64`, three benchmark repetitions per arm.

## Wall-clock Amdahl arms

| Arm | Mean tok/s | Samples tok/s | Mean ms/token | Peak RSS MiB |
|---|---:|---|---:|---:|
| Exact resident generic control | 4.733 | 4.815 / 4.864 / 4.519 | 211.299 | 10,528.1 |
| Routed experts removed | 8.174 | 8.216 / 8.096 / 8.210 | 122.338 | 10,525.0 |
| LM head removed | 5.385 | 5.409 / 5.376 / 5.370 | 185.703 | 10,530.8 |
| Instrumented exact control | 4.887 | 4.914 / 4.877 / 4.870 | 204.623 | 10,534.9 |

The routed-expert removal saves 88.961 ms/token, 42.10% of control wall time,
and gives an expert-infinite-speed ceiling of 8.174 tok/s.  The LM-head
removal saves 25.596 ms/token, 12.11%, and gives an LM-head-infinite ceiling
of 5.385 tok/s.  Assuming those two savings were additive, their combined
upper bound is about 10.34 tok/s; this is a theoretical bound, not a measured
combined runtime.

## Summed operator work

TSC intervals are summed across worker threads and are therefore work
attribution rather than one wall-clock cycle stream.

| Category | Share |
|---|---:|
| Routed MoE | 47.44% |
| Attention projection matmuls | 24.76% |
| Other matrix multiplication | 6.71% |
| Gated DeltaNet | 5.11% |
| Shared expert | 3.81% |
| Router | 2.25% |
| Other operations | 7.41% |
| Full attention op | 0.27% |
| Normalization | 0.08% |

The next compute target is attention projection matmul, not another isolated
expert-dispatch experiment.  The output projection is worth addressing later
because its wall-time bound is measurable, but it cannot reach either target
alone.

## Correctness

The exact control produced the established hash
`a3c3c73a067cbb5fab9e68bd7d06c03d3c488e74467d33c40fae339836006c8c`.
Native K=8, 40 layers, routes, and weights were unchanged.  The two removed
arms are explicitly invalid-output measurement bounds.

Raw Kaggle logs, source patch, stdout/stderr, and result JSON are preserved in
this directory.
