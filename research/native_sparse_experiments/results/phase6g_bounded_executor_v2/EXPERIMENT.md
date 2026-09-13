# Phase 6G bounded expert executor v2 — real RAM/I/O result, arithmetic invalid

## Hypothesis

An explicit pread-backed global-LRU cache can replace routed-expert mmap
residency in the real Qwen path under a 4 GiB process budget.

## Configuration

- Model: Qwen3.5-35B-A3B-UD-IQ2_XXS, 10,656,955,008 bytes,
  SHA-256 `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- llama.cpp: `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Research source: `8e66767`
- Kaggle: 4-vCPU Intel Xeon, AVX2, 4 threads, inherited affinity, CPU-only
- Workload: fixed clinical prompt, 64 requested decode tokens, 3 repetitions,
  temperature 0, seed 1234, context 512
- Cache: 2,000,000,000-byte anonymous fixed storage, 2,281 slots of
  876,544 bytes, explicit three-plane `pread`, global LRU
- Control: same patched model path, mmap resident routed tensors, no explicit
  cache

## Result

| arm | generation tok/s (mean; range) | ms/token from mean | peak RSS |
|---|---:|---:|---:|
| resident control | 4.000 (3.9–4.1) | 250.0 | 10,539.6 MiB |
| explicit cache arm | 2.933 (2.9–3.0) | 340.91 | 3,517.5 MiB |

The cache arm is a real <=4 GiB process run at 3.435 GiB RSS. It is not yet a
valid exact-runtime result: v2 intentionally disabled the IQP panel to keep
the storage experiment isolated, and the resulting numerical drift changes
later routes. The route traces match through event 224 and first differ at
event 225 (`blk.35.ffn_gate_exps.weight`). All three cache/control pairs were
deterministic within the arm, but their common response hash
`5804c970f14ce9902178660a5262689f2ffb3518619336403244933d85cafca1` is not
the established exact-control hash.

## Cache and I/O measurements

Per repetition, the cache reported:

```text
slot_bytes=876544 slots=2281 reserved_bytes=1999396864
requests=89280 hits=79981 misses=9299 evictions=7018
read_bytes=8150982656 read_ns=10.965–11.421 s
```

The operational request hit rate is 89.584%; requests count the three routed
matrix nodes and is not directly the route-corpus request definition. Fresh
cache bytes are 8,150,982,656 per process, or 127.36 MB per 64 requested
decode tokens (123.50 MB per 66 route-event token equivalents including the
observed initial/prompt work). The explicit reads supply about 733 MB/s while
inside the instrumented read interval. The read interval is approximately
11.12 s per repetition, about 51% of the 21.82 s 64-token wall estimate, so
this first serialized cache path is both I/O- and compute-expensive.

The prototype’s `read_calls` counter increments by three for each plane read;
the logged 83,691 therefore corresponds to 27,897 inferred actual `pread`
syscalls. The cache reads fixed 270,336-byte gate/up planes and 335,872-byte
down planes. Per-read latency percentiles were not instrumented in v2.

## Conclusion and next correction

The storage ownership concept works and demonstrates the structural RAM result:
the routed expert residency can be replaced by an explicitly bounded cache
without approaching 4 GiB. The current performance is 26.7% below the mmap
control, with roughly half of token time in serialized reads. However, v2 is
not an exact model result because of the deliberate IQP disable. Version 3
keeps IQP enabled and redirects its source plane to the same cache slot. That
run is the exact storage gate; only its output/routes can validate the final
bounded path.

## Artifacts

- Raw Kaggle output, route traces, process samples, and runtime patch: `raw/`
- Corrective exact-IQP harness: `kaggle/native-sparse-bounded-executor-v3/`
- Kaggle kernel: `toheebogunade/jamii-native-sparse-bounded-executor-v2`
