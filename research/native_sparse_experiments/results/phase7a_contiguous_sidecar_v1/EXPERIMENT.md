# Phase 7A contiguous exact sidecar v1 — preliminary result

## Hypothesis

Packing each `(layer, expert)` gate/up/down bundle into one contiguous
byte-identical sidecar record should reduce bounded-cache read overhead by
replacing three plane reads with one fixed-size read.

## Configuration

- Model: Qwen3.5-35B-A3B-UD-IQ2_XXS, SHA-256
  `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- llama.cpp: `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Research source: `c054761`
- Kaggle: 4-vCPU Intel Xeon, AVX2, 4 threads, CPU-only
- Workload: fixed clinical prompt, 64 requested decode tokens, 3 repetitions,
  temperature 0, seed 1234, context 512
- Cache: 2,000,000,000 bytes, 2,281 slots of 876,544 bytes, global LRU
- Candidate: 40×256 contiguous records, 8,975,810,560-byte sidecar copied
  from the original GGUF tensor planes

## Result

| arm | samples tok/s | mean tok/s | peak RSS |
|---|---|---:|---:|
| resident control | 4.3, 4.1, 4.3 | 4.233 | 10,535.9 MiB |
| one-read sidecar | 2.7, 3.1, 3.4 | 3.067 | 3,517.2 MiB |

The sidecar run remained exact: control and bounded route files were
byte-identical for all repetitions, and all response hashes matched at
`5804c970f14ce9902178660a5262689f2ffb3518619336403244933d85cafca1`.

The first sidecar process built the 8.976-GB file in 46.760 seconds.  That
one-time setup is not included in the CLI decode timing, but is recorded as a
deployment cost.

## I/O result and validity limitation

The candidate issued 9,275 read calls per repetition instead of the v3
three-plane count of 27,825, while transferring the same 8,129,945,600
logical bytes.  Reported read intervals were 17.809, 9.840, and 6.051 s.
However, the sidecar was created immediately before the first bounded run and
the harness did not evict the sidecar's filesystem pages between repetitions.
Repetitions 2–3 therefore benefited from a warmer page cache than the
three-plane control.  The apparent steady-state improvement is not an
apples-to-apples storage result.

## Conclusion

The exact contiguous representation and one-read path compile and preserve
native semantics, but this v1 measurement is classified `FOLLOW-UP ONCE`,
not a frontier movement.  The next version explicitly drops sidecar pages
before every bounded process and drops them after construction before the
first decode.  Only that corrected result will decide whether the sidecar
reduces unhidden I/O stall enough to keep.

## Artifacts

- Raw Kaggle output, source patch, routes, and process samples: `raw/`
- Harness: `kaggle/native-sparse-bounded-executor-v6/`
- Kaggle kernel: `toheebogunade/jamii-native-sparse-bounded-executor-v6`
