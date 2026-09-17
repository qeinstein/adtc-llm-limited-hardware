# Phase 6G bounded expert executor v4 — exact-IQP 3 GiB design point

## Hypothesis

The exact-IQP explicit global-LRU executor remains correct when its fixed
expert cache is reduced to the committed 3 GiB budget point, and its real RSS
can remain below 3 GiB without changing native routing or weights.

## Configuration

- Model: Qwen3.5-35B-A3B-UD-IQ2_XXS, 10,656,955,008 bytes,
  SHA-256 `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- llama.cpp: `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Research source: `75d0cc7`
- Kaggle: 4-vCPU Intel Xeon, AVX2, 4 threads, CPU-only, inherited affinity
- Workload: fixed clinical prompt, 64 requested decode tokens, 3 repetitions,
  temperature 0, seed 1234, context 512
- Cache: 1,261,346,816 bytes, exactly 1,439 slots of 876,544 bytes, explicit
  three-plane `pread`, global LRU
- Control: same patched model path, mmap-resident routed tensors
- Bounded arm: exact IQP kernel enabled and fed from fixed cache slots

## Result

| arm | repetitions tok/s | mean tok/s | ms/token | peak RSS |
|---|---|---:|---:|---:|
| resident control | 4.8, 4.7, 4.7 | 4.733 | 211.27 | 10,536.8 MiB |
| explicit exact-IQP cache | 3.1, 3.0, 3.0 | 3.033 | 329.67 | 2,813.8 MiB |

The bounded arm is a real exact-runtime result below the 3 GiB process
target: peak RSS was 2.748 GiB, with 1,742.7 MiB anonymous and 1,070.8 MiB
file-backed RSS.  It is 35.9% slower than the same-run resident control.

The cache and control route files are byte-identical for all three
repetitions.  All six runs share the deterministic response hash
`5804c970f14ce9902178660a5262689f2ffb3518619336403244933d85cafca1`.
This is a same-prompt/same-binary equality result; broad quality remains
unmeasured.

## Cache and I/O measurements

Each bounded repetition reported:

```text
slot_bytes=876544 slots=1439 reserved_bytes=1261346816
requests=89280 hits=76808 misses=12472 evictions=11033
read_calls=112248 read_bytes=10932256768
read_ns=11.456–11.808 s
```

The operational hit rate is 86.030%.  The logged read-call count represents
three plane reads per logical cache request, or 37,416 inferred `pread`
syscalls per repetition.  Fresh bytes are 10,932,256,768 per 64-token
repetition (10,932.26 decimal MB, or 10,425.81 MiB total), equivalent to
170.8165 decimal MB/token (162.9033 MiB/token).  Mean
instrumented read time is 11.638 s, with about 939.4 MB/s (895.8 MiB/s)
throughput.  Reads occupy an estimated 55.2% of bounded 64-token wall time;
the remainder is exact CPU execution, cache lookup, and runtime overhead.

Read-size percentiles were not instrumented; each read is a fixed gate/up or
down plane slice in the current GGUF-derived layout.

## Correctness

- Native K=8 and 40 layers were unchanged.
- The exact IQP decoder consumed fixed cache-slot pointers; no weight bytes
  were transformed.
- Control and bounded route files are byte-identical for repetitions 1–3.
- All repetitions are deterministic and output hashes match within the run.
- No expert dropping, routing approximation, K reduction, or weight change.

This validates the 3 GiB bounded storage point for the tested deterministic
smoke workload.  It is not a broad capability or clinical-quality result.

## Conclusion

The explicit exact-IQP cache now demonstrates both a real `<4 GiB` run
(v3, 3.435 GiB) and a real `<3 GiB` run (v4, 2.748 GiB).  Reducing the cache
from 2,281 to 1,439 slots increases fresh reads and lowers throughput from
3.000 to 3.033 tok/s only within run quantization/noise; the bounded system
is I/O-dominated at this point.  A 2.5 GiB arm is technically plausible but
will increase traffic again; it is useful as a RAM-frontier measurement, not
as a route to the throughput target.

The strategic result is now clear: exact bounded storage is demonstrated,
while >10 tok/s requires replacing more than the routed expert path.  The
measured routed-expert and attention-projection Amdahl arms leave a
conservative independent-work ceiling around 13.5 tok/s before optimizing
other terms, so a deeper whole-runtime CPU plan remains justified.

## Artifacts

- Raw Kaggle output, route traces, process samples, and runtime patch: `raw/`
- Harness: `kaggle/native-sparse-bounded-executor-v4/`
- Kaggle kernel: `toheebogunade/jamii-native-sparse-bounded-executor-v4`
