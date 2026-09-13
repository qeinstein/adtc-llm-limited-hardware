# Native sparse route-trace analysis

This directory contains a model-free analysis pass for routed Qwen3.5 traces.
It does not load a checkpoint or implement a sparse runtime.

## Trace contract

Input may be a JSON array or JSONL. Each token record must contain `layers`,
with exactly 40 entries. Each entry is either an eight-element expert-ID list
or `{ "experts": [...] }`. IDs must be distinct integers in `[0, 256)`.

Example:

```json
{"token": 0, "layers": [[0,1,2,3,4,5,6,7], "... 39 more layers ..."]}
```

The analyzer treats each `(layer, expert_id)` as a cache bundle. The default
bundle size is 1,700,000 bytes, the rough Qwen3.5 4-bit estimate in the
research notes; pass the measured packed size instead when available. For
non-uniform packed layers, `--bundle-by-layer-file` accepts either a JSON list
of 40 byte sizes or a JSON object keyed by layer number.

## Outputs

`route_trace.py` reports:

- total and per-token route requests, unique layer/expert bundles, and global
  and per-layer popularity;
- adjacent-token Jaccard/shared-request overlap and same-layer retention;
- uncached bytes/token;
- global-LRU hit/miss rates, fresh requests/token, and fresh bytes/token for
  each requested capacity;
- an equally partitioned per-layer LRU (`partitioned_lru`), where
  `floor(total/40)` bundles go to every layer and the remainder goes to the
  lowest layer indices;
- a static per-layer popularity cache (`static_popularity_oracle`) that picks
  the most-requested layer/expert bundles from the complete trace. This is a
  hindsight upper-control, explicitly marked non-deployable, not a prediction
  method.

Run on Kaggle (after checkout, with no model download required):

```bash
python research/native_sparse_experiments/route_trace.py \
  --trace research/native_sparse_experiments/tests/fixtures/valid_one_token.json \
  --capacities 0,64,128,256,512,1024 \
  --output /kaggle/working/route_report.json
# Optional exact per-layer byte accounting:
# --bundle-by-layer-file /kaggle/working/qwen35_bundle_bytes.json
python -m pytest research/native_sparse_experiments/tests -q
```

For a real trace, replace the fixture path. The unit tests exercise shape
validation, duplicate/range rejection, JSON array input, overlap, popularity,
global/per-layer LRU behavior, static-oracle selection, and byte accounting.

## Measured execution status

The exact Qwen3.5-35B-A3B IQ2_XXS control and follow-up experiments are
preserved under the results directory:

- phase0_kaggle_v4: exact native top-8 output/route equality; 4.8 tok/s at
  10,528.7 MiB resident and 2.7 tok/s at 5,155.3 MiB with routed-expert lazy
  mmap.
- phase1_cpu_sweep_v2: resident CPU ceiling. Four threads with poll 0 reached
  4.716 tok/s across three repeats; thread, poll, and affinity tuning changed
  throughput by only about 2%.
- phase2_iqp_decode_v1: compute-kernel intervention. Forcing the existing
  batch-oriented IQ panel at one row per expert regressed 43.42% to 2.940
  tok/s versus a 5.196 tok/s same-binary generic control, with deterministic
  generated-payload equality and unchanged residency/traffic.
- phase3_single_row_avx2_v1: a purpose-built `cne1 == 1` path using the
  existing IQP decoder plus one-row AVX2 GEMV remained exact but regressed
  39.26% to 3.020 tok/s versus a 4.973 tok/s control. Profiling attributed
  88.58% of the instrumented IQP cycles to decoding/materializing the eight
  weight rows.
- phase3_single_row_avx2_v3: direct raw IQ2_XXS row-dot dispatch was neutral
  at 4.184 tok/s versus a 4.168 tok/s same-run control (+0.40%). The
  measurement-only non-MoE arm measured a 1,612.9 MiB operational resident
  floor (539.7 MiB anonymous, 1,073.2 MiB file-backed) against 6,538.9 MiB
  for the lazy 64-token inference arm.
- phase4_fused_iq2_v1: a true four-row fused packed-IQ2_XXS AVX2 path shared
  activation loads and avoided decoded panels, but measured 4.778 tok/s
  versus 4.885 tok/s for the same-run raw-row reference. It remained exact;
  the profile attributed 5.74% of summed fused-function cycles to activation
  loads and 31.50% to the decode/integer-MAC region, with the remainder in
  loop, accumulator, and surrounding handling.
- phase4_fused_iq2_v2: a two-row fused follow-up reduced per-call profile
  cost but doubled fused call count and measured 4.441 tok/s versus 4.650
  tok/s raw-row. It also remained exact. Both phase 4 reports include raw
  Kaggle bundles and the v1 cache budget design.

Current classification: resident CPU/DRAM-kernel execution is the larger term
in the lazy path (about 212 ms/token resident plus about 158 ms/token added
lazy-storage stall at the measured Phase 0 point). The single-row experiments
show that panel-backed specialization and dispatch isolation do not improve
the ceiling; raw IQ2 dequantization remains dominant. Storage remains a major
secondary bottleneck, but storage optimization alone cannot reach 10 tok/s.
The direct fused IQ2 path is now a correct but insufficient optimization:
activation-load sharing and row-group width do not clear the ceiling. A
custom packed expert layout/decode-table access pattern or a specialized
executor is justified as the next compute experiment. In parallel, bounded
expert storage should be implemented against the measured 1.61 GiB floor;
the committed Phase 4 cache design budgets 2,664 / 1,439 / 826 bundles for
4 / 3 / 2.5 GiB total RSS and replays 97.53 / 110.52 / 141.66 MB fresh
logical bytes per token respectively.

## Phase 5 decision measurements

- `phase5a_amdahl_v1`: the exact resident control averaged 4.745 tok/s
  (210.73 ms/token).  A measurement-only routed-MoE removal reached 8.083
  tok/s (123.72 ms/token), so routed experts account for 41.29% of wall time
  and cannot alone reach 10 tok/s.  Summed operator work was 46.98% routed
  MoE, 33.64% other matrix multiplication, 5.37% Gated DeltaNet, and 3.82%
  shared expert.
- `phase5b_lowbit_ceiling_v1/v2` and `phase5c_iq2_representation_v1`: an
  exact offline-resolved IQ2 layout was slower than the control at both 74
  bytes/block (+12.12% storage) and 70 bytes/block (+6.06%).  A simple
  unsigned W2 LUT/gather reference was slower too.  Metadata resolution and
  record padding are therefore rejected as standalone optimizations.
- `phase5e_route_corpus_v1`: 32 prompts yielded 2,016 exact route tokens and
  9,577 unique bundles.  On the diverse corpus, global LRU fresh logical
  traffic is 63.96 / 106.87 / 143.63 MB/token at 4 / 3 / 2.5 GiB budgets.
  Layer partitioning changes this by about one percent; the tested online
  frequency/staleness policies are worse.  Belady is retained only as an
  offline headroom oracle.
- `phase5g_bounded_cache_v1`: a model-free explicit `pread` store plus fixed
  aligned byte-bounded LRU cache passes unit tests.  It is the selected
  transitional storage architecture; model integration follows after the
  current compute target is identified.
- `phase5h_floor_decomp_v1`: loader inventory finds 733 unique tensors, with
  1.5555 GiB logical non-routed payload (0.2664 GiB each for input embeddings
  and LM head, 0.8578 GiB attention/DeltaNet trunk, 0.0861 GiB shared expert,
  0.0781 GiB router).  The skip-MoE process measured 1,609.96 MiB RSS.

- `phase5a_amdahl_v2`: splitting that matrix bucket and removing only the LM
  head gives 5.385 tok/s versus the 4.733 tok/s control, a 12.11% wall-time
  saving.  Routed-expert removal reaches 8.174 tok/s.  Treating both savings
  as additive gives only a theoretical 10.34 tok/s ceiling.  Attention
  projection matmuls are 24.76% of summed operator work and are now the next
  compute target.

The exact control and native routes remain unchanged; no model
representation change is justified yet.

## Phase 6 runtime and bounded-storage measurements

- `phase6b_attention_amdahl_v1`: the exact resident control averaged 4.365
  tok/s (229.08 ms/token).  Measurement-only attention projection bypass
  reached 5.662 tok/s (176.62 ms/token), saving 52.46 ms/token or 22.90% of
  wall time.  The dominant profiled Q5_K families were `attn_qkv` (48.3% of
  projection TSC), `attn_gate` (25.3%), and full-attention `attn_q` (16.0%).
  Even a free attention-projection path therefore remains below 10 tok/s.
- `phase6c_dense_repack_v3`: the standard mainline Q5_K repack was a matched
  donor/kernel A/B at 4.167 versus 4.200 tok/s (+0.8%) and added 107.5 MiB
  RSS.  It is neutral within run spread and is not a runtime pivot.
- `phase6a_ik_llama_v3`: the exact IQ2_XXS checkpoint loaded and ran on the
  upstream ik_llama CPU path, but default `llama-bench` generation averaged
  2.528 tok/s across three 64-token samples versus 4.167 tok/s for the
  matched mainline reference.  Its deterministic smoke was internally
  stable, but no cross-runtime route hook was present.  ik_llama remains a
  code donor/reference, not the replacement baseline.
- `phase6g_bounded_executor_v3`: exact-IQP explicit pread into 2,281 fixed
  global-LRU slots reached 3.000 tok/s at 3,517.5 MiB RSS (3.435 GiB), versus
  4.067 tok/s at 10,537.1 MiB resident.  Route files were byte-identical and
  output hashes matched across control/cache repetitions.  The cache had an
  89.61% operational hit rate and supplied 127.03 MB fresh logical bytes per
  64 generated tokens; instrumented reads occupied about 44.3% of wall time.
- `phase6g_bounded_executor_v4`: the 1,439-slot / 1,261,346,816-byte exact
  cache reached 3.033 tok/s at 2,813.8 MiB RSS (2.748 GiB), with 86.03% hit
  rate, 170.82 MB fresh logical bytes per 64 tokens, and reads occupying
  about 55.2% of wall time.  Routes and deterministic outputs remained equal.
- `phase6g_bounded_executor_v5`: the 826-slot / 724,025,344-byte exact cache
  reached 2.867 tok/s at 2,301.2 MiB RSS (2.247 GiB), with 83.10% hit rate,
  206.63 MB fresh logical bytes per 64 tokens, and reads occupying about
  56.8% of wall time.  Routes and deterministic outputs again remained
  equal.  This demonstrates the measured 2.5 GiB RAM frontier, but it is not
  a throughput solution.

The phase-6 storage arms use a short deterministic prompt, so their measured
cache hit rates and fresh bytes are workload-specific.  The larger 2,016-token
corpus remains the cache-policy planning source: global-LRU replay predicts
63.96 / 106.87 / 143.63 MB fresh logical bytes per token at 4 / 3 / 2.5 GiB
budgets respectively.  The real executor confirms the important causal
result: explicit bounded ownership works, while serialized I/O plus CPU
execution is too slow for the target.

The combined independent wall fractions measured so far are approximately
42.1% routed experts and 22.9% attention projections.  A fraction-based
expert-plus-attention-free guide is only about 13.5 tok/s, before accounting
for interactions and all remaining runtime work; routed-expert optimization
alone has an 8.17 tok/s measured ceiling.  The non-MoE floor remains
1,609.96 MiB RSS, decomposed logically as 0.2664 GiB input embeddings,
0.2664 GiB LM head, 0.8578 GiB attention/DeltaNet trunk, 0.0861 GiB shared
expert, 0.0781 GiB router, plus runtime/state.  The exact bounded storage
path is therefore selected for the RAM track; further cache-policy tuning is
not the next throughput move.  The next high-information compute experiment
is a materially different dense/runtime execution schedule targeting the
Q5_K attention/Gated-DeltaNet families and the remaining non-MoE graph, with
the exact resident and exact bounded paths retained as controls.  No K,
routing, expert-topology, or model-weight change is justified yet.
