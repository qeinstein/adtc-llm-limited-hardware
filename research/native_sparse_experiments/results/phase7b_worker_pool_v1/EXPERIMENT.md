# Phase 7B — Fixed worker-pool bounded executor

## Hypothesis

A fixed four-worker read pool would preserve the Phase 7B staged plane-read
overlap while avoiding one pthread creation per missing expert bundle.  The
experiment keeps the exact IQ2_XXS model and the existing two-gigabyte global
byte-bounded cache; only the read scheduling mechanism changes.

## Configuration

- Model: `Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf`
- Model SHA-256: `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- Native routing: top-8, unchanged
- Runtime commit: `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Research harness source: `1cee728`
- Kaggle: Linux, 4-vCPU Intel Xeon, AVX2, four compute threads
- Cache: 2,000,000,000-byte capacity; 876,544-byte slots; 2,281 slots
- Prompt: `Give one concise reason oral rehydration solution helps a child with watery diarrhoea.`
- Decode: 64 tokens, three repetitions per arm, temperature 0, seed 1234
- Arms: resident control, bounded serial reads, bounded staged reads with a
  fixed four-worker pool

## Results

| arm | decode tok/s (repetitions) | mean | process elapsed mean | peak RSS |
|---|---:|---:|---:|---:|
| resident control | 4.4, 4.4, 4.3 | 4.433 | 27.660 s | 10,534.9 MiB |
| bounded serial | 3.2, 3.2, 3.1 | 3.167 | 35.290 s | 3,517.4 MiB |
| bounded staged worker pool | 3.1, 3.2, 3.1 | 3.133 | 32.918 s | 3,517.6 MiB |

The staged worker-pool process elapsed time was 6.722% below serial bounded
execution, but rounded decode throughput was 1.053% lower.  The result is not
a decoded-throughput improvement over the existing v10 staged executor.

The worker pool generated the same logical traffic as serial execution:

- 9,275 misses per 64-token repetition
- 27,825 physical plane reads in the staged arm versus 83,475 serial reads
- 8,129,945,600 logical read bytes per repetition
- 127.0304 decimal MB/token, or 121.1456 MiB/token

The fixed pool did not improve readiness.  Aggregate ready-wait time was
17.319 s per 64-token repetition, or 270.616 ms/token, versus 6.161 s and
96.269 ms/token in v10.  Four read workers competed with the four compute
threads and reduced the useful overlap.  Final async join wait was 0.567 s
per repetition.

## Correctness

All three arms produced response SHA-256
`5804c970f14ce9902178660a5262689f2ffb3518619336403244933d85cafca1` and the
route SHA-256 was
`400d85562944ec574faa1d0a629528237dd8c1f7638b164dda567607a3de3aa2`.
Native top-8 routing and expert semantics were unchanged.

## Decision

**KILL as a throughput optimization.**  The fixed pool is technically exact
and lowers process elapsed time relative to serial reads, but it does not
raise decoded throughput and worsens readiness wait relative to v10.  Keep
the v10 staged executor as the storage scheduling control and stop adding
basic worker-pool variants.  The next storage work, if revisited, must remove
the four-reader/four-compute contention or replace the execution schedule at
a larger architectural level.

Raw Kaggle artifacts are in `raw/`.
