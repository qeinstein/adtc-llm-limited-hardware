# Phase 4 v1 — fused packed-IQ2 selected-expert execution

## Hypothesis

A purpose-built AVX2 path that shares activation q8 loads across output rows
and consumes packed IQ2_XXS weights directly should beat the generic selected
expert path by removing decoded-panel materialization and repeated activation
loads.

## Implementation

The opt-in `GGML_FUSED_IQ2=1` path handles `cne1 == 1` routed operations. It
groups four contiguous output rows and calls a new `ggml_vec_dot_iq2_xxs_q8_K_4x1`
kernel. For each 256-element IQ2 block, the kernel loads the two activation q8
vectors once, then performs each row's IQ2 grid lookup, sign handling, scale
handling, integer dot, and accumulation directly from the packed weight row.
No decoded expert matrix or IQP panel scratch is created. Partial row groups
fall back to the exact existing dot function.

This is a narrow AVX2 experiment, not a replacement of the general runtime.
The generic path and the prior raw-row path remain available as controls.

## Reproducibility

- Kaggle kernel: `toheebogunade/jamii-native-sparse-fused-iq2-avx2`, version 1
- Research source: commit `5e56d95`
- Runtime source: llama.cpp `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Runtime patch SHA-256: `9680a8e8747d09bf16b8a83a3b31a5385be98607b3a6910883cb3fc6f9591509`
- Checkpoint: `Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf`, 10,656,955,008 bytes,
  SHA-256 `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- Hardware: Kaggle 4-vCPU / 2-core Intel Xeon @ 2.20 GHz, AVX2/FMA
- Compiler: GCC/G++ 11.4.0; `GGML_NATIVE=ON`; CPU-only
- Configuration: resident mmap, `-lzm off`, four threads, poll 0, no affinity
  override, 64 decode tokens, three repetitions per throughput arm
- Prompt: `Give one concise reason oral rehydration solution helps a child with watery diarrhoea.`

## Throughput

Each throughput row contains the three `llama-bench` samples, not a cherry-picked
best result.

| arm | samples tok/s | mean tok/s | mean ms/token | peak RSS |
|---|---:|---:|---:|---:|
| generic control | 4.501, 4.869, 4.825 | **4.731** | **211.35** | 10,533.5 MiB |
| raw single-row IQ2 | 4.919, 4.915, 4.822 | **4.885** | **204.70** | 10,527.4 MiB |
| fused IQ2 4x1 | 4.772, 4.732, 4.831 | **4.778** | **209.28** | 10,526.4 MiB |

The fused path is +0.99% versus this run's generic control but -2.19% versus
the same-run raw-row reference. It does not improve the historical resident
ceiling of approximately 5.2 tok/s. The profile arm is excluded from the
throughput comparison because TSC instrumentation reduces it to 2.996 tok/s.

## Correctness

The generic control, fused path, and raw-row path all produced response
SHA-256:

`a3c3c73a067cbb5fab9e68bd7d06c03d3c488e74467d33c40fae339836006c8c`

The compared runs used native K=8 and 40 layers. Router behavior, selected
expert IDs, weights, quantization, and graph topology were unchanged. The
standalone randomized four-row kernel smoke test also matched the existing
per-row IQ2 dot function.

## Hot-path profile

The profile process emitted the same aggregate counter four times because the
first version's exit-report registration was racy across worker threads; the
counter values themselves were identical and are recorded once here:

| counter | aggregate TSC intervals | per fused call |
|---|---:|---:|
| fused function total | 278,752,737,540 | 17,630.8 cycles |
| activation q8 loads | 15,995,103,654 | 1,011.7 cycles |
| IQ2 decode + integer MAC region | 87,801,949,064 | 5,553.4 cycles |

The profile process recorded 15,810,560 fused calls, eight IQ2 blocks per
call, and 126,484,480 block groups. Normalized over its 256 benchmark decode
tokens, the summed worker intervals are approximately 1.089 billion total,
62.5 million activation-load, and 343.0 million decode/MAC TSC cycles per
token. These are summed per-thread intervals, not wall-clock cycles; they are
useful for phase attribution, not for claiming a single-core cycle rate.

The measured load fraction is 5.74% of the fused-function interval and the
decode/MAC region is 31.50%. The remaining interval includes q2 pointer/table
handling, accumulator management, block-loop overhead, and output reduction.
The result indicates that activation-load duplication was not the dominant
limitation, and that the four-row accumulator layout likely incurred enough
register/spill and loop overhead to erase the intended gain.

## RAM and traffic

Resident throughput RSS remained approximately 10.53 GiB because this is the
resident control pool with zero fresh expert bytes/token. The RAM-floor arms
reproduced the prior measurement:

| arm | peak RSS | peak anon/file |
|---|---:|---:|
| lazy, context 512, 64 tokens | 6,539.0 MiB | 539.8 / 5,999.4 MiB |
| non-MoE skip, context 512, 64 tokens | **1,613.0 MiB** | **539.8 / 1,073.3 MiB** |

The non-MoE floor remains approximately 1.61 GiB. The bounded-cache design
and exact route-replay estimates are in [CACHE_BUDGET.md](CACHE_BUDGET.md).

## Conclusion and follow-up

True packed-IQ2 fusion is correct and removes large temporary materialization,
but this first four-row AVX2 form is not a speedup. The next measurement-driven
optimization is a two-row fused kernel with fewer live vector accumulators and
the same shared activation-load experiment, plus fixed profile registration.
If that also remains neutral, the evidence will favor a deeper packed-layout
or decode-table redesign rather than further dispatch changes. Bounded expert
storage remains necessary for RAM, but the resident compute ceiling has not
yet justified making storage the primary throughput attack.
