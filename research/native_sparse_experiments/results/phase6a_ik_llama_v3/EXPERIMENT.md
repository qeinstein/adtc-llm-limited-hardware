# Phase 6A ik_llama.cpp v3 — external CPU baseline

## Hypothesis

The upstream `ik_llama.cpp` CPU implementation, including its custom low-bit
execution support and Qwen3.5-MoE support, may provide a materially stronger
resident decode baseline for the exact IQ2_XXS checkpoint.

## Configuration

- Upstream: `https://github.com/ikawrakow/ik_llama.cpp.git`, commit
  `3bb386eb68ffee0a5dc7db21da0735d594929eeb`
- Research harness source: `e5bb66e`
- Model: Qwen3.5-35B-A3B-UD-IQ2_XXS, SHA-256
  `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- Kaggle: 4-vCPU Intel Xeon, AVX2, 4 threads, CPU-only, allowed CPUs 0–3
- Benchmark: `llama-bench`, prompt 0, generation 64, 3 repetitions,
  `-t 4 -ngl 0`, mmap enabled, default ik fused-MoE path
- The optional `--run-time-repack` arm was not included; this is the default
  upstream CPU baseline, not a tuned ik configuration.

## Result

| runtime | generation samples tok/s | mean tok/s | ms/token | peak RSS |
|---|---|---:|---:|---:|
| ik_llama default CPU | 2.5711, 2.5202, 2.4926 | 2.5280 | 395.58 | 10,328.3 MiB |
| mainline same-workload reference | 4.1, 4.2, 4.2 | 4.1667 | 240.00 | 10,430.7 MiB |

The ik baseline is 39.3% slower than the matched mainline no-repack
measurement.  Its three-run standard deviation is 0.0399 tok/s.  The seeded
CLI smoke tests completed successfully and were deterministic within ik:
the cold and warm response hashes both equal
`d291847873c6cfc96855bebf16434eeada90ba45f6a4d9fc67de48e37b8db479`.

The mainline/ik route equality was not measured because this probe had no
route-observation hook.  The checkpoint loaded as Qwen3.5-MoE with native
fused-MoE scheduling and no configured expert substitution, but this result
is not used as an exact cross-runtime correctness claim.

## Conclusion

ik_llama is a useful donor/reference and demonstrates relevant custom CPU
work, but this unmodified default configuration is not a runtime pivot for
this exact workload.  It is substantially below the current mainline
resident control and therefore does not justify adopting it wholesale.  A
future donor comparison can isolate `--run-time-repack` or individual kernels,
but the immediate highest-information work remains the exact bounded storage
path and the larger mainline attention/dense runtime terms.

## Artifacts

- Raw Kaggle output, logs, benchmark JSON, and process samples: `raw/`
- Harness: `kaggle/native-sparse-ik-llama-v3/`
- Kaggle kernel: `toheebogunade/jamii-native-sparse-ik-llama-v3`
