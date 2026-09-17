# Phase 4 v2 — two-row fused packed-IQ2 follow-up

## Hypothesis

The v1 four-row fused kernel likely kept too many AVX2 accumulators live. A
two-row group should reduce register pressure while preserving direct packed
IQ2_XXS consumption and shared activation q8 loads.

## Reproducibility

- Kaggle kernel: `toheebogunade/jamii-native-sparse-fused-iq2-avx2-v2`, version 1
- Research source: commit `f650021`
- Runtime source: llama.cpp `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Runtime patch SHA-256: `8c4981b242b48946440c10d09ab3da32dae2d35171187a698b204b5f8e1e84d2`
- Checkpoint: `Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf`, SHA-256
  `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- Hardware: Kaggle 4-vCPU / 2-core Intel Xeon @ 2.20 GHz, AVX2/FMA
- Compiler: GCC/G++ 11.4.0; `GGML_NATIVE=ON`; CPU-only
- Configuration: resident mmap, `-lzm off`, four threads, poll 0, 64 decode
  tokens, three repetitions per throughput arm, default one benchmark warmup

## Throughput

| arm | samples tok/s | mean tok/s | mean ms/token | peak RSS |
|---|---:|---:|---:|---:|
| generic control | 4.574, 4.150, 4.675 | **4.466** | **223.92** | 10,524.0 MiB |
| raw single-row IQ2 | 4.619, 4.681, 4.650 | **4.650** | **215.06** | 10,524.2 MiB |
| fused IQ2 2x1 | 4.307, 4.476, 4.541 | **4.441** | **225.16** | 10,528.2 MiB |

The two-row fused path is -0.55% versus its generic control and -4.49% versus
the same-run raw-row reference. It is also below the v1 four-row fused mean of
4.778 tok/s. Lowering the group width did not recover the resident ceiling.

## Correctness

All three deterministic CLI arms produced the same response SHA-256:

`a3c3c73a067cbb5fab9e68bd7d06c03d3c488e74467d33c40fae339836006c8c`

Native K=8, router behavior, selected experts, weights, quantization, and
graph topology were unchanged. The path is therefore correct despite its
negative performance result.

## Profile

The fixed-registration profile emitted one aggregate line:

`total_cycles=504498346409 load_cycles=25744645369 decode_mac_cycles=94970727645 calls=31621120 blocks=252968960`

That is 15,954.5 total TSC cycles per two-row fused call, 814.2 cycles for the
shared activation loads, and 3,003.4 cycles for the decode/integer-MAC region.
The corresponding fractions of the fused-function interval are 5.10% load,
18.82% decode/MAC, and 76.07% other loop/accumulator/dispatch work. The
two-row function doubles the number of fused calls per token, so the reduced
per-call footprint does not translate into a lower end-to-end time.

The profile arm itself is instrumentation-distorted (2.095 tok/s) and is not
used as a throughput result. Its counters are summed worker TSC intervals;
the `llama-bench` process includes one warmup plus three 64-token repetitions.

## RAM and conclusion

The RAM-floor arms reproduced the non-MoE floor at **1,613.1 MiB RSS** with
approximately **539.8 MiB anonymous / 1,073.4 MiB file-backed** pages. The
resident control remains approximately 10.53 GiB and scheduled fresh expert
traffic is zero in these compute arms.

This follow-up rejects accumulator group width as the missing optimization.
The exact direct fused IQ2 path is now measured as correct but insufficient:
the dominant unaddressed cost is the packed representation's decode/table/
weight handling and surrounding per-row execution, not activation-load
duplication alone. A custom packed expert layout or a specialized executor
that changes the representation/access pattern is justified as the next
compute experiment. The bounded-cache design remains required for the RAM
objective and is recorded in the v1 Phase 4 artifact.
