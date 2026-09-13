# Phase 7D selective-Q4 plus bounded executor v1

## Hypothesis

The validated selective attention/GDN Q4_K challenger should retain its
resident decode win when routed experts are managed by the real explicit
2-GB byte-bounded cache.

## Configuration

- Control: Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf, SHA-256
  `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- Candidate: selective attention/GDN Q4_K GGUF, SHA-256
  `86f27f69e0cea7f8cd0420452f1b600c57cedce6589f60c2cfee06a8c0395d39`
- llama.cpp: `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Kaggle: 4-vCPU Intel Xeon AVX2, four threads, CPU-only, inherited affinity
- Cache: 2,000,000,000 bytes, 2,281 fixed 876,544-byte global-LRU slots
- Workload: fixed clinical prompt, 64 generated tokens, temperature 0, seed
  1234, context 512, three repetitions, `-lzm on`, serial three-plane reads
- The runtime patch is the exact bounded owner used in Phase 7B; only the
  selected dense projection representation differs in the candidate.

## Result

| arm | decode tok/s | mean | median | process elapsed mean/median s | peak RSS MiB |
|---|---|---:|---:|---:|---:|
| bounded IQ2 control | 1.5, 2.5, 2.7 | 2.233 | 2.5 | 57.648 / 47.328 | 3,517.3 |
| bounded selective Q4 | 2.6, 2.8, 2.8 | 2.733 | 2.8 | 42.653 / 43.190 | 3,947.6 |

The first control repetition had a large filesystem stall, so the matched
repetitions 2–3 are also reported: control averaged 2.6 tok/s and 46.248 s,
while Q4 averaged 2.8 tok/s and 41.727 s. The corresponding rounded decode
delta is +7.692%, and process elapsed is 9.774% lower. The candidate remains
under the 4-GiB limit, but its 3,947.6 MiB peak leaves only 148.4 MiB of
headroom and is 430.3 MiB above the control. It therefore does not carry the
2.5-GiB point forward.

## Traffic and route behavior

The control transferred 8,129,945,600 logical bytes per 64-token repetition,
127.0304 decimal MB/token (121.1456 MiB/token), with 9,275 misses and an
89.611% hit rate. The Q4 candidate transferred 8,014,241,792 logical bytes
per repetition, 125.2225 decimal MB/token (119.4215 MiB/token), with 9,143
misses and an 89.759% hit rate. This small traffic reduction is not an exact
storage win: the Q4 route trace SHA-256 is
`8a6d5640cd79c0f3ae689f99249297f593d3d5e3660f236b4e6b40e3dcf0a7cd`, versus
the control `400d85562944ec574faa1d0a629528237dd8c1f7638b164dda567607a3de3aa2`.
The candidate response hash nevertheless stayed equal to the bounded control
hash `5804c970f14ce9902178660a5262689f2ffb3518619336403244933d85cafca1`.

The increased anonymous RSS is the main unresolved result: control peak
anonymous/file RSS was approximately 2,446.6/1,070.8 MiB, while candidate was
2,997.8/949.8 MiB. The candidate's 2-GB explicit cache is unchanged; the
additional anonymous residency must be diagnosed before using this challenger
at tighter budgets.

## Decision

`KEEP AS <=4-GiB CHALLENGER / DO NOT PROMOTE YET`: the candidate is faster in
the stable matched repetitions and still fits under 4 GiB, but this run is
not an exact route-preserving control and the RSS increase eliminates the
2.5-GiB target. First complete the quality v2 gate. If quality passes, inspect
why the quantized file causes approximately 551 MiB more non-cache anonymous
RSS, then rerun with a staged/pool owner only if that accounting issue is
resolved. Do not call the candidate an exact 2.5-GiB point.

## Artifacts

- Raw Kaggle output, route traces, process samples, patches, and kernel log:
  `raw/`
- Harness: `kaggle/native-sparse-q4-bounded-v1/`
- Kaggle kernel: `toheebogunade/jamii-native-sparse-q4-bounded-executor-v1`
