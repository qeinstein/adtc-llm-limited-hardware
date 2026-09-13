# Phase 6G bounded expert executor v5 — exact-IQP 2.5 GiB frontier

## Hypothesis

The exact-IQP explicit global-LRU executor can operate at the committed
2.5 GiB stretch budget with unchanged native routing, and the experiment will
quantify the extra traffic imposed by the smaller cache.

## Configuration

- Model: Qwen3.5-35B-A3B-UD-IQ2_XXS, 10,656,955,008 bytes,
  SHA-256 `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- llama.cpp: `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Research source: `83b300b`
- Kaggle: 4-vCPU Intel Xeon, AVX2, 4 threads, CPU-only, inherited affinity
- Workload: fixed clinical prompt, 64 requested decode tokens, 3 repetitions,
  temperature 0, seed 1234, context 512
- Cache: 724,025,344 bytes, exactly 826 slots of 876,544 bytes, explicit
  three-plane `pread`, global LRU
- Control: same patched model path, mmap-resident routed tensors
- Bounded arm: exact IQP kernel enabled and fed from fixed cache slots

## Result

| arm | repetitions tok/s | mean tok/s | ms/token | peak RSS |
|---|---|---:|---:|---:|
| resident control | 4.8, 4.8, 4.8 | 4.800 | 208.33 | 10,536.4 MiB |
| explicit exact-IQP cache | 2.9, 2.9, 2.8 | 2.867 | 348.84 | 2,301.2 MiB |

The bounded arm is a real exact-runtime result below the 2.5 GiB stretch
target: peak RSS was 2.247 GiB, with 1,230.3 MiB anonymous and 1,070.9 MiB
file-backed RSS.  It is 40.3% slower than the same-run resident control and
5.5% slower than the 3 GiB bounded arm.

The cache and control route files are byte-identical for all three
repetitions.  All six runs share the deterministic response hash
`5804c970f14ce9902178660a5262689f2ffb3518619336403244933d85cafca1`.
This is a same-prompt/same-binary equality result; broad quality remains
unmeasured.

## Cache and I/O measurements

Each bounded repetition reported:

```text
slot_bytes=876544 slots=826 reserved_bytes=724025344
requests=89280 hits=74193 misses=15087 evictions=14261
read_calls=135783 read_bytes=13224419328
read_ns=12.542–12.827 s
```

The operational hit rate is 83.101%.  The logged read-call count represents
three plane reads per logical cache request, or 45,261 inferred `pread`
syscalls per repetition.  Fresh bytes are 13,224,419,328 per 64-token
repetition (13,224.42 decimal MB, or 12,611.79 MiB total), equivalent to
206.6316 decimal MB/token (197.0592 MiB/token).  Mean
instrumented read time is 12.683 s, with about 1,042.7 MB/s (994.4 MiB/s)
throughput.  Reads occupy an estimated 56.8% of bounded 64-token wall time.

Read-size percentiles were not instrumented; each read is a fixed gate/up or
down plane slice in the current GGUF-derived layout.

## Correctness

- Native K=8 and 40 layers were unchanged.
- The exact IQP decoder consumed fixed cache-slot pointers; no weight bytes
  were transformed.
- Control and bounded route files are byte-identical for repetitions 1–3.
- All repetitions are deterministic and output hashes match within the run.
- No expert dropping, routing approximation, K reduction, or weight change.

This validates the 2.5 GiB bounded storage point for the tested deterministic
smoke workload.  It is not a broad capability or clinical-quality result.

## Conclusion

The exact bounded executor now reaches the intended RAM frontier in measured
model execution: 3.435 GiB at the 4 GiB cache design point, 2.748 GiB at the
3 GiB design point, and 2.247 GiB at the 2.5 GiB design point.  The smaller
cache raises fresh logical reads from 170.82 to 206.63 MB per 64 generated
tokens and leaves the storage path clearly dominant in this serialized
prototype.

This establishes that `<4 GiB`, `2.5–3 GiB`, and even the measured 2.5 GiB
frontier are structurally reachable without changing the exact model.  It
does not establish the throughput target: the best bounded result is 3.033
tok/s, and the best resident controls remain about 4.7–5.2 tok/s.  Further
progress requires a whole-runtime CPU execution plan, not more cache-policy
tuning on this small prompt.

## Artifacts

- Raw Kaggle output, route traces, process samples, and runtime patch: `raw/`
- Harness: `kaggle/native-sparse-bounded-executor-v5/`
- Kaggle kernel: `toheebogunade/jamii-native-sparse-bounded-executor-v5`
