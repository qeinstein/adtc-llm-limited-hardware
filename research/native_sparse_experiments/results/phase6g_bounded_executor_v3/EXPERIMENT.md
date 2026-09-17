# Phase 6G bounded expert executor v3 — exact-IQP bounded storage

## Hypothesis

An explicit `pread`-backed global-LRU cache can replace routed-expert mmap
residency in the real Qwen path while preserving the exact IQP execution,
native routes, and deterministic output.

## Configuration

- Model: Qwen3.5-35B-A3B-UD-IQ2_XXS, 10,656,955,008 bytes,
  SHA-256 `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- llama.cpp: `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Research source: `b912ab0` plus the v3 harness patch
- Kaggle: 4-vCPU Intel Xeon, AVX2, 4 threads, CPU-only, inherited affinity
- Workload: fixed clinical prompt, 64 requested decode tokens, 3 repetitions,
  temperature 0, seed 1234, context 512
- Cache: 2,000,000,000-byte anonymous fixed storage, 2,281 slots of
  876,544 bytes, explicit three-plane `pread`, global LRU
- Control: same patched model path, mmap-resident routed tensors
- Bounded arm: exact IQP kernel enabled; its routed source pointers are
  redirected to the corresponding fixed cache slots

## Result

| arm | repetitions tok/s | mean tok/s | ms/token | peak RSS |
|---|---|---:|---:|---:|
| resident control | 4.1, 4.1, 4.0 | 4.067 | 245.90 | 10,537.1 MiB |
| explicit exact-IQP cache | 3.0, 3.0, 3.0 | 3.000 | 333.33 | 3,517.5 MiB |

The bounded arm is a real exact-runtime result below 4 GiB: 3.435 GiB peak
RSS, with the same generated response hash as the same-run controls and
identical route traces for all three repetitions.  It is 26.2% slower than
the same-run resident control, so bounded ownership solves the RAM constraint
but not the throughput target.

The common deterministic response hash in this run is
`5804c970f14ce9902178660a5262689f2ffb3518619336403244933d85cafca1`; this is
a same-prompt/same-binary control-vs-cache equality check, not a replacement
for the established cross-experiment control hash.

## Cache and I/O measurements

Each repetition reported:

```text
slot_bytes=876544 slots=2281 reserved_bytes=1999396864
requests=89280 hits=80005 misses=9275 evictions=6994
read_bytes=8129945600 read_ns=9.391–9.492 s
```

The operational hit rate is 89.611%.  The cache counters cover the three
routed matrix nodes and therefore are not directly comparable to the route
corpus's one-bundle request definition.  The explicit reads supply
8,129,945,600 bytes per 64-token repetition (8,129.95 decimal MB, or
7,753.32 MiB total), equivalent to 127.0304 decimal MB/token (121.1456
MiB/token).  The mean instrumented read interval is 9.448 s
per repetition, equivalent to about 860.5 MB/s (820.6 MiB/s) while reading.
That interval is 44.3% of the bounded arm's estimated 64-token wall time;
the remainder includes the exact CPU execution and cache/dispatch overhead.

The logged `read_calls=83475` counts three plane reads per logical cache
request, so it corresponds to 27,825 inferred `pread` syscalls per
repetition.  Read-size percentiles were not instrumented; each routed plane
read is fixed by the existing gate/up/down plane layout.

## Correctness

- Native K=8 and 40 layers were unchanged.
- The exact IQP decoder consumed cache-slot pointers rather than mmap
  pointers; no weight bytes were transformed.
- Control and bounded route files are byte-identical for repetitions 1–3.
- All six arms are deterministic within the run and share the response hash
  above.
- No expert dropping, routing approximation, K reduction, or weight change.

This validates the bounded storage path for the tested deterministic smoke
workload.  It is a runtime correctness check, not a broad quality benchmark.

## Conclusion

This version closes the v2 exactness failure: explicit bounded reads can feed
the existing native IQP path without route drift or output drift while
keeping process RSS at 3.435 GiB.  The measured 3 tok/s path is below the
resident control because the current implementation serializes explicit
reads and exact expert compute.  The next storage step is a 3 GiB cache arm,
but throughput work must target the whole runtime: routed experts are only
about 42% of wall time and attention projections another 22.9% by direct
wall-clock Amdahl measurement.

## Artifacts

- Raw Kaggle output, route traces, process samples, and runtime patch: `raw/`
- Harness: `kaggle/native-sparse-bounded-executor-v3/`
- Kaggle kernel: `toheebogunade/jamii-native-sparse-bounded-executor-v3`
