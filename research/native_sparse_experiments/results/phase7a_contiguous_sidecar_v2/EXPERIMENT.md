# Phase 7A contiguous exact sidecar v2 — corrected cold-cache result

## Hypothesis

Replacing three plane reads with one fixed-size read from a contiguous
`(layer, expert)` sidecar should reduce bounded-cache I/O stall without
changing exact IQP semantics.

## Configuration

- Model: Qwen3.5-35B-A3B-UD-IQ2_XXS, SHA-256
  `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- llama.cpp: `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Research source: `5cecdc4`
- Kaggle: 4-vCPU Intel Xeon, AVX2, 4 threads, CPU-only
- Workload: fixed clinical prompt, 64 requested decode tokens, 3 repetitions,
  temperature 0, seed 1234, context 512
- Cache: 2,000,000,000 bytes, 2,281 slots of 876,544 bytes, global LRU
- Candidate: 40×256 contiguous records, 8,975,810,560-byte sidecar copied
  byte-for-byte from the original GGUF planes
- The model and sidecar file pages were explicitly dropped before each
  process; the first process also dropped sidecar pages after construction.

## Result

| arm | samples tok/s | mean tok/s | ms/token | peak RSS |
|---|---|---:|---:|---:|
| resident control | 4.9, 4.9, 4.9 | 4.900 | 204.08 | 10,537.1 MiB |
| one-read sidecar | 3.2, 3.0, 3.3 | 3.167 | 315.79 | 3,517.0 MiB |

The sidecar path remained exact: control and bounded route files were
byte-identical for all repetitions, and all response hashes matched at
`5804c970f14ce9902178660a5262689f2ffb3518619336403244933d85cafca1`.

## I/O result

The candidate reduced read calls from the v3 three-plane count of 27,825 to
9,275 per repetition while transferring the same 8,129,945,600 logical
bytes.  This did not improve the actual storage path.  Cold sidecar read
intervals were 11.764, 19.057, and 14.433 s, mean 15.085 s, or about
539 MB/s.  Reads occupied an estimated 74.6% of bounded wall time, compared
with about 44.3% for the original three-plane v3 arm.

The one-time sidecar build took 38.831 s and created 8.976 GB of additional
storage.  It is recorded separately from decode time but is a further
deployment cost.

## Decision

`KILL` the simple contiguous-sidecar/layout branch.  It is exact and reduces
syscall count, but its physical read behavior is substantially worse on this
Kaggle filesystem and it regresses end-to-end bounded throughput.  No further
padding, ordering, or one-read sidecar variants are justified.

The next storage experiment must attack unhidden stall through asynchronous
submission/overlap, or storage should remain on the existing three-plane
global-LRU baseline while compute work proceeds.

## Artifacts

- Raw Kaggle output, source patch, routes, and process samples: `raw/`
- Harness: `kaggle/native-sparse-bounded-executor-v7/`
- Kaggle kernel: `toheebogunade/jamii-native-sparse-bounded-executor-v7`
