# Phase 8C — Critical-path storage measurement (v10 bounded executor)

## Hypothesis

v10 improves prompt batching, while exact single-token decode has little
same-layer overlap. A warm page-cache oracle measures the removable
unhidden storage wall time at the 2,281-slot / <=4 GiB configuration.

## Configuration

- Model: `Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf`
- Model SHA-256: `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- Native routing: top-8, unchanged weights
- Runtime commit: `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Research harness source: `cae43e5` (`kaggle/native-sparse-critical-path-v1/`)
- Kaggle kernel: `toheebogunade/jamii-native-sparse-critical-path-v1`
- Kaggle: Linux, 4-vCPU Intel Xeon @ 2.20 GHz, AVX2, four compute threads
- Cache: 2,000,000,000-byte capacity; 876,544-byte slots; 2,281 slots
- Decode: 64 tokens, three repetitions per arm, temperature 0, seed 1234
- Arms: serial cold, serial warm page-cache oracle, staged cold, staged warm
  page-cache oracle. Warm arms skip cache drops and are measurement-only I/O
  oracles, not deployable <=4 GiB points.

## Results

| arm | decode tok/s (mean) | ms/token | peak RSS (max) | physical reads/token |
|---|---|---:|---:|---:|
| serial cold | 3.114066 | 321.223 | 3517.36 MiB | 136.583 MB |
| serial warm oracle | 3.347492 | 298.793 | 3520.45 MiB | ~0 |
| staged cold | 3.198323 | 312.668 | 3517.80 MiB | 136.596 MB |
| staged warm oracle | 3.322881 | 301.054 | 3520.45 MiB | 0 |

Logical fresh expert traffic: 127.0304 decimal MB/token (unchanged in all
arms; warm oracles serve it from page cache).

Fully removing physical storage I/O recovers:

- serial: 22.430 ms/token, ~7.50% optimistic end-to-end ceiling
- staged: 11.613 ms/token, ~3.71% optimistic end-to-end ceiling

## Correctness

All twelve runs produced response SHA-256
`5804c970f14ce9902178660a5262689f2ffb3518619336403244933d85cafca1`.
Native K=8, weights unchanged.

## Decision

**STORAGE IS NOT THE MAIN THROUGHPUT BOTTLENECK — FRONTIER as a bound.**
This invalidates earlier reasoning that treated all fresh expert bytes as
blocking disk cost. Do not estimate prefetch opportunity from
bytes/bandwidth alone; all storage/prefetch proposals must be evaluated
against this measured wall-clock stall (7.5% serial, 3.7% staged). The
research priority moves to compute, graph execution, runtime overhead,
representation, and multi-token amortization.

Raw Kaggle artifacts are in `raw/`.
