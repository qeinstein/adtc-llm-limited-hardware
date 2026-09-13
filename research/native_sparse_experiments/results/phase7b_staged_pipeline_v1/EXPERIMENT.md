# Phase 7B staged bounded expert pipeline v1

## Hypothesis

Publishing each asynchronously read gate/up/down plane as it becomes ready
should let the exact selected-expert executor consume resident or completed
planes while later storage reads proceed, reducing the serialized bounded
storage penalty.

## Configuration

- Model: Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf, SHA-256
  `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- llama.cpp: `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Research source: `69464e6b35113baa387b5140547aa1b5ff7780df`
- Kaggle: 4-vCPU Intel Xeon, AVX2, CPU-only, four threads, inherited affinity
- Runtime: `--poll 0`, resident control or `-lzm on` plus explicit 2,000,000,000-byte
  bounded cache (2,281 slots of 876,544 bytes)
- Workload: fixed clinical prompt, 64 generated tokens, temperature 0, seed
  1234, context 512, three repetitions per arm
- Serial arm: three sequential plane reads per missing routed bundle, then
  exact IQP execution.
- Staged arm: one worker per missing bundle, plane readiness masks, compute
  waits only for the selected plane, and joins workers when the graph advances
  to the next layer or at exit.

## Result

| arm | decode samples tok/s | mean process s | median process s | peak RSS MiB |
|---|---|---:|---:|---:|
| resident control | 4.4, 4.5, 4.4 | 27.893 | 27.535 | 10,537.6 |
| bounded serial | 3.2, 3.2, 3.2 | 35.086 | 34.995 | 3,517.3 |
| bounded staged | 3.2, 3.2, 3.2 | 32.329 | 32.305 | 3,517.5 |

The CLI rounds decode speed to 0.1 tok/s, so it shows no throughput delta.
The unrounded process elapsed time is consistently lower for staged execution:
mean 32.329 s versus 35.086 s, a 7.857% reduction; per-repetition reductions
were 8.49%, 6.79%, and 8.29%. This is a storage-schedule improvement, not a
claim that the decode kernel reached a new 0.1-tok/s bucket.

## I/O and overlap

Both bounded arms had 9,275 misses, 89.611% operational hit rate, and exactly
8,129,945,600 logical bytes per 64-token repetition: 8,129.9456 decimal MB
total, 7,753.3203 MiB total, or 127.0304 decimal MB/token (121.1456 MiB/token).
The logged `read_calls` is 83,475 for serial and 27,825 for staged; the former
is the existing three-plane counter convention and the latter is one worker
record per bundle. `/proc/<pid>/io` sampled approximately 8.74 GB of physical
read bytes per bounded repetition in both schedules, retaining the distinction
between logical payload and observed kernel I/O.

The serial measured read interval averaged 10.602 s per 64 tokens. In staged
execution, aggregate worker read intervals are not wall time because they run
concurrently; the explicit final/advance join wait averaged 1.073 s. Consumers
still accumulated 6.161 s of readiness waits across 294,600 wait events, or
96.269 ms/token as an aggregate diagnostic. This confirms that the pipeline
exposes overlap but remains storage/availability constrained; readiness waits
must not be added as wall time because they occur in concurrent executor work.

## Correctness

All control, serial, and staged response hashes equal
`5804c970f14ce9902178660a5262689f2ffb3518619336403244933d85cafca1`.
Routes are preserved byte-for-byte in the raw route traces. Native K=8,
router semantics, expert IDs, expert bytes, and IQP arithmetic are unchanged.

## Decision

`KEEP / EXPLOIT ONCE`: the staged schedule gives a stable 6.79–8.49% process
elapsed improvement and reduces the final join from the prior approximately
8.5 s scale to 1.073 s, but it does not yet move the rounded decode metric.
The individual-thread-per-miss design is only a probe. One follow-up should
replace it with a bounded worker pool or equivalent queue, preserving plane
readiness and deterministic cache ownership. If that does not move decoded
throughput or materially reduce unhidden readiness wait, kill further basic
I/O scheduling work and shift the main effort to whole-runtime compute.

## Artifacts

- Raw Kaggle result, process samples, routes, patch, and kernel log: `raw/`
- Harness: `kaggle/native-sparse-bounded-executor-v10/`
- Kaggle kernel: `toheebogunade/jamii-native-sparse-bounded-executor-v10`
