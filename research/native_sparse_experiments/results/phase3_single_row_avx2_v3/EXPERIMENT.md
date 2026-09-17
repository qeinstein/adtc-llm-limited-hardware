# Phase 3 v3 — raw-row IQ2 isolation and non-MoE RAM floor

## Hypotheses

1. The generic selected-expert ceiling may be limited by `MUL_MAT_ID`
   dispatch/scatter overhead rather than the IQ2_XXS raw row-dot kernel.
2. Executing the same graph while preventing routed expert access can measure
   the non-routed trunk/runtime RSS floor.

## Reproducibility

- Kaggle kernel: `toheebogunade/jamii-native-sparse-single-row-avx2`, version 3
- Research source: commit `d427af8`
- Runtime source: llama.cpp `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Checkpoint: `Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf`, 10,656,955,008 bytes,
  SHA-256 `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- Hardware: Kaggle 4-vCPU / 2-core Intel Xeon @ 2.20 GHz, AVX2/FMA
- Configuration: resident mmap, lazy off for throughput arms, lazy on for RAM
  arms, CPU-only, four threads, poll 0, 64 decode tokens, three repetitions

## Raw-row path

For `cne1 == 1`, the opt-in raw arm bypasses IQP panel scratch and directly
invokes the existing AVX2 `ggml_vec_dot_iq2_xxs_q8_K` for each output row of
the selected expert. It does not change native routing, K=8, weights,
quantization, or graph topology. This is an isolation probe for dispatch and
panel overhead, not a new multi-row kernel.

| arm | samples tok/s | mean tok/s | ms/token | peak RSS |
|---|---:|---:|---:|---:|
| generic control | 4.152, 4.198, 4.153 | **4.168** | 239.940 | 10,532.3 MiB |
| panel-backed single-row IQP | 2.444, 2.530, 2.519 | **2.498** | 400.360 | 10,530.2 MiB |
| raw-row IQ2 | 4.179, 4.199, 4.175 | **4.184** | 238.991 | 10,525.2 MiB |

Raw-row dispatch is only +0.40% versus its same-run generic control, within
the run-to-run noise seen across Kaggle workers. The v1 panel-backed path was
39.26% below its control. Together these results reject panel construction as
an optimization strategy and show that simply isolating dispatch does not
raise the ceiling. A new fused multi-row raw IQ2 kernel would need to reduce
the dequant arithmetic itself, not just change the call boundary.

## Correctness

The generic, panel-backed, and raw-row deterministic 24-token outputs were
all byte-identical with response SHA-256
`a3c3c73a067cbb5fab9e68bd7d06c03d3c488e74467d33c40fae339836006c8c`.
All arms used native K=8 and 40 layers; no expert was dropped or substituted.

## Resident RAM floor

The lazy arms logged process RSS before the first routed-expert event and
after decode. The measurement-only `GGML_PHASE3_SKIP_MOE=1` arm retains the
graph, trunk operations, context shape, and runtime buffers but zeros
`MUL_MAT_ID` outputs before any routed expert weight access. Its numerical
output is intentionally invalid, so it is a memory-attribution arm only.

| arm | pre-first-route RSS | peak RSS | peak anon/file |
|---|---:|---:|---:|
| lazy, context 64, 1 token | 632.7 MiB | 4,847.5 MiB | 518.4 / 4,329.3 MiB |
| lazy, context 512, 1 token | 639.1 MiB | 4,866.1 MiB | 536.9 / 4,329.4 MiB |
| lazy, context 512, 64 tokens | 630.6 MiB | 6,538.9 MiB | 539.8 / 5,999.1 MiB |
| non-MoE skip, context 512, 64 tokens | 646.1 MiB | **1,612.9 MiB** | **539.7 / 1,073.2 MiB** |

Under the same short-context shape, the measured non-routed operational floor
is therefore approximately 1.61 GiB RSS, with approximately 0.54 GiB of
anonymous runtime/state and 1.07 GiB of file-backed trunk/other model pages.
The corresponding lazy inference run is 4.93 GiB higher at peak, attributable
primarily to routed expert pages plus associated file-backed residency. This
is an execution-floor measurement, not a claim that all non-routed logical
weights must be resident simultaneously.

Logical file accounting independently gives 8,975,810,560 routed-expert bytes
and 1,681,144,448 remaining model bytes. The resident measurement shows that
the `<4 GiB` target is plausible for runtime-managed expert residency, but
the `<=2–2.5 GiB` target still requires roughly 0.5–1.1 GiB of additional
runtime/representation work even before bounded routed storage is added.

## Conclusion and next move

The immediate compute ceiling remains approximately 4.2–5.0 tok/s on this
4-vCPU Kaggle Xeon. The panel-backed specialized path is rejected; raw-row
dispatch is neutral. The next justified work is a true fused raw IQ2_XXS
multi-row kernel (eight weight rows processed directly from the on-disk
representation with shared activation loads), followed by a bounded explicit
expert cache sized against the measured 1.61 GiB floor. A clean-sheet runtime
or new packed representation is now justified as an option, but should be
driven by the fused-kernel/cache measurements rather than introduced
speculatively.

Raw Kaggle output is in `native-sparse-single-row-avx2-results/`.
