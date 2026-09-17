# Phase 5A — end-to-end resident decode/Amdahl profile

## Hypothesis

Routed expert work may be too small a fraction of complete token latency for
another expert-only kernel to reach the throughput target. Measure the full
resident token by operator category and bound the expert contribution with a
measurement-only expert-removed run.

## Reproducibility

- Kaggle kernel: `toheebogunade/jamii-native-sparse-end-to-end-amdahl-profile-v1`
- Research source: `cd2be78`
- Runtime: llama.cpp `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Model: Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf, SHA-256
  `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- Hardware: Kaggle 4-vCPU / 2-core Intel Xeon @ 2.20 GHz, AVX2/FMA
- Compiler: GCC/G++ 11.4, `GGML_NATIVE=ON`, CPU-only
- Four threads, poll 0, resident mmap, `-lzm off`, `-n 64`, three repeats
- Prompt: oral rehydration solution clinical smoke prompt used in Phase 4

## Wall-clock Amdahl bound

| Arm | Mean tok/s | Mean ms/token | Peak RSS |
|---|---:|---:|---:|
| Exact resident generic control | 4.745 | 210.73 | 10,529.8 MiB |
| Routed `MUL_MAT_ID` removed, measurement only | 8.083 | 123.72 | 10,531.1 MiB |
| Instrumented generic profile | 4.861 | 205.71 | 10,533.4 MiB |

Removing routed expert execution saves 87.01 ms/token, or 41.29% of the
control wall time. The resulting expert-infinite-speed ceiling is **8.083
tok/s**, a 1.70x maximum expert-only speedup. Therefore routed-expert
optimization alone cannot reach 10 tok/s, even if it became free.

The removed arm is not a valid model output and was excluded from correctness.

## Operator work profile

The profile sums TSC intervals across worker threads; it is a work attribution,
not a single wall-clock cycle stream.

| Category | Share of summed operator work |
|---|---:|
| Routed MoE | 46.98% |
| Other matrix multiplication | 33.64% |
| Gated DeltaNet | 5.37% |
| Shared expert | 3.82% |
| Router | 2.27% |
| Other operations | 7.57% |
| Full attention | 0.28% |
| Normalization | 0.08% |

The next system-level compute target is non-MoE matrix execution, with Gated
DeltaNet the next named operator. Expert layout work remains useful for RAM and
secondary latency, but the Amdahl gate rejects an expert-only strategy as the
complete throughput solution.

## Correctness

The exact control smoke run succeeded with the established response hash
`a3c3c73a067cbb5fab9e68bd7d06c03d3c488e74467d33c40fae339836006c8c`.
Native K=8 and 40 layers were preserved. The expert-removed arm was explicitly
measurement-only.

## RAM

The non-MoE skip arm reproduced the earlier operational floor at **1,613.2
MiB RSS**. The lazy steady arm reached 6,539.2 MiB in this harness; the
historical Phase 0 lazy result remains 5,155.3 MiB under its original process
conditions. The resident control remains approximately 10.53 GiB.

Raw Kaggle logs, source patch, stdout/stderr, route traces, and `result.json`
are stored beside this report.
