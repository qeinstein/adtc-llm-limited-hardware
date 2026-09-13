# Phase 7B asynchronous bounded expert reads v2 — matched A/B

## Hypothesis

Concurrent reads for all missing routed bundles can reduce bounded exact
decode latency relative to the matched serialized three-plane `pread`
baseline.  This isolates I/O scheduling; both arms use the same cache,
weights, IQP decoder, and CPU execution path.

## Configuration

- Model: Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf, SHA-256
  `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- llama.cpp: `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Research source: `a0ccb3e` (v9 harness); traffic-unit correction is
  recorded separately in `ce5c119`.
- Kaggle: 4-vCPU Intel Xeon, AVX2, CPU-only, four threads, inherited
  affinity, `--poll 0`
- Workload: fixed clinical prompt, 64 generated tokens, temperature 0,
  seed 1234, context 512, three repetitions per arm
- Cache: 2,000,000,000 bytes, 2,281 fixed 876,544-byte global-LRU slots
- Serial arm: existing three sequential plane reads per missing bundle.
- Async arm: one worker per missing bundle, same three reads, all workers
  joined before expert computation.  It does not overlap reads with compute.

## Result

The CLI reports generation throughput to only one decimal place in this
build, so both the rounded decode values and the higher-variance wall-clock
measurements are preserved:

| arm | samples tok/s | mean tok/s | median tok/s | min–max tok/s | mean process s | median process s | peak RSS MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| resident control | 4.8, 4.9, 4.8 | 4.833 | 4.8 | 4.8–4.9 | 25.267 | 24.509 | 10,537.1 |
| bounded serial | 3.3, 3.3, 2.0 | 2.867 | 3.3 | 2.0–3.3 | 37.009 | 33.227 | 3,517.8 |
| bounded async | 3.3, 3.2, 3.1 | 3.200 | 3.2 | 3.1–3.3 | 31.839 | 31.249 | 3,517.7 |

The async arm is 11.6% higher than serial on the rounded mean decode metric,
but 3.0% lower on the rounded median; the serial third repetition had a
large filesystem stall.  Total process elapsed time improved 14.0% by the
mean and 6.0% by the median.  This is a positive scheduling signal, not a
stable final throughput claim.

## I/O result

All bounded repetitions had 9,275 misses and transferred exactly
8,129,945,600 bytes per 64-token repetition (8,129.95 decimal MB, or
7,753.32 MiB total), equivalent to 127.0304 decimal MB/token (121.1456
MiB/token).  The operational hit rate was 89.611% in both arms.

Serial instrumented read intervals were 9.913, 9.577, and 13.623 s (mean
11.038 s).  Async all-worker join intervals were 8.006, 7.986, and 9.529 s
(mean 8.507 s), a 22.9% lower measured read wait.  The async summed worker
intervals are not wall time and are retained in `ANALYSIS.json` and raw
stderr only as a concurrency diagnostic.

The actual read operation count is 27,825 `pread` calls per repetition: the
runtime's `read_calls` counter multiplies this by three in the serial and
async reporting paths and is therefore labeled as a logged counter in the
raw evidence.  Logical and physical kernel read counters are preserved in
the process samples.

## Correctness

- All nine response hashes equal
  `5804c970f14ce9902178660a5262689f2ffb3518619336403244933d85cafca1`.
- All nine route files are byte-identical with SHA-256
  `400d85562944ec574faa1d0a629528237dd8c1f7638b164dda567607a3de3aa2`.
- Native K=8, router semantics, expert IDs, expert bytes, and IQP arithmetic
  are unchanged.

## Decision

`FOLLOW-UP ONCE`.  The matched result justifies one true staged
read/compute-pipeline attempt because v2 reduces read wait but still blocks
all expert compute behind the join.  Do not make more sidecar/layout or
cache-policy variants.  If staged execution fails to produce at least 3%
end-to-end improvement with exact output/routes, kill this storage-schedule
branch and return the bounded baseline to storage-control status.

## Artifacts

- Raw Kaggle output and process samples: `raw/`
- Harness: `kaggle/native-sparse-bounded-executor-v9/`
- Kaggle kernel: `toheebogunade/jamii-native-sparse-bounded-executor-v9`
