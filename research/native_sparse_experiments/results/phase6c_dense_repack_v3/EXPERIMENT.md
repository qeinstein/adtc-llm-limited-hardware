# Phase 6C dense Q5_K repack v3

## Hypothesis

The standard llama.cpp CPU Q5_K repacking path can improve the dominant Qwen
attention/Gated-DeltaNet projection decode workload without changing model
semantics.

## Configuration

- Model: exact Qwen3.5-35B-A3B IQ2_XXS, SHA-256
  `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- llama.cpp: `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Research source: `b090403`
- Kaggle: 4-vCPU Intel Xeon, AVX2, 4 threads, CPU-only, inherited affinity
- Workload: same clinical prompt, 64 CLI decode tokens, 3 repetitions,
  context 512, temperature 0, seed 1234
- Arms: `--no-repack` and `--repack`
- Attention/GDN projection tensors are Q5_K in the committed inventory.

## Result

| arm | repetitions tok/s | mean tok/s | ms/token | peak RSS |
|---|---|---:|---:|---:|
| no repack | 4.1, 4.2, 4.2 | 4.1667 | 240.00 | 10,430.7 MiB |
| standard repack | 4.2, 4.2, 4.2 | 4.2000 | 238.10 | 10,538.2 MiB |

The standard repack changes throughput by only +0.8% while increasing peak
RSS by about 107.5 MiB. Both CLI smoke arms return the established exact
control hash `a3c3c73a067cbb5fab9e68bd7d06c03d3c488e74467d33c40fae339836006c8c`.
No route, K, or weight change was made.

## Conclusion

This is a measured dense donor attempt, and it is effectively neutral for the
target workload. The generic repack path is not a runtime pivot: its small
speed change is within the observed run spread and costs memory. A future
dense optimization must use a materially different schedule/layout or target
the broader non-MoE graph, not repeat this standard repack toggle.

## Artifacts

- Raw Kaggle output and process logs: `raw/`
- Harness: `kaggle/native-sparse-dense-repack-v3/`
- Kaggle kernel: `toheebogunade/jamii-native-sparse-dense-repack-v3`
