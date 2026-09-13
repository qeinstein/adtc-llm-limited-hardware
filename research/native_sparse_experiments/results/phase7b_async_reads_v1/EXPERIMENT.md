# Phase 7B asynchronous bounded expert reads v1 — concurrent miss reads

## Hypothesis

Submitting the three plane reads for all missing selected experts concurrently
can reduce unhidden storage wait in the exact 2 GB bounded cache without
changing expert bytes, routing, or arithmetic.

## Configuration

- Model: Qwen3.5-35B-A3B-UD-IQ2_XXS, SHA-256
  `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- llama.cpp: `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Research source: `ff24614`
- Kaggle: 4-vCPU Intel Xeon, AVX2, 4 threads, CPU-only
- Workload: fixed clinical prompt, 64 decode tokens, 3 repetitions,
  temperature 0, seed 1234, context 512
- Cache: 2,000,000,000 bytes, 2,281 slots, global LRU
- Candidate: one pthread per missing expert; each worker performs the same
  three `pread` calls as the serial baseline and all workers join before
  exact expert compute. This is concurrent I/O, not read/compute pipelining.

## Result

| arm | samples tok/s | mean tok/s | ms/token | peak RSS |
|---|---:|---:|---:|---:|
| resident control | 8.0, 7.9, 8.1 | 8.000 | 125.00 | 10,533.9 MiB |
| bounded concurrent reads | 4.7, 4.8, 4.7 | 4.733 | 211.29 | 3,517.6 MiB |

The resident control was an unusually fast Kaggle allocation, so the
throughput comparison to prior kernels is not a valid hardware-normalized
comparison. The important scheduling counters are internally comparable to
the candidate's own wall time. A matched serial-vs-concurrent rerun is
required before attributing an end-to-end gain to this arm.

## I/O result

Each bounded repetition had 9,275 missing experts and transferred exactly
8,129,945,600 bytes per 64-token repetition (8,129.95 decimal MB, or
7,753.32 MiB total), equivalent to 127.0304 decimal MB/token (121.1456
MiB/token).
The read-call count remained 27,825, i.e. three plane reads per miss.

Aggregate worker read intervals were 32.003, 27.002, and 27.271 seconds,
while the actual all-worker join intervals were 6.919, 6.619, and 6.800
seconds. Thus the candidate created substantial concurrency among reads, but
the join remained before expert execution and no read/compute overlap was
measured. The 6.62–6.92 second interval is approximately half of the
212 ms/token bounded decode wall time.

## Correctness

All six response hashes were
`5804c970f14ce9902178660a5262689f2ffb3518619336403244933d85cafca1`; all
control and bounded route files were byte-identical. Native K=8, router
semantics, expert IDs, weights, and IQP arithmetic were unchanged.

## Decision

`FOLLOW-UP ONCE`: preserve the concurrent-I/O primitive, but do not yet call
it a frontier movement. It has enough internal overlap to justify one
matched serial-vs-concurrent A/B and, if positive, a true read/compute
pipeline. Do not make more contiguous-layout variants.

## Artifacts

- Raw Kaggle output, source patch, process samples, and routes: `raw/`
- Harness: `kaggle/native-sparse-bounded-executor-v8/`
- Kaggle kernel: `toheebogunade/jamii-native-sparse-bounded-executor-v8`
