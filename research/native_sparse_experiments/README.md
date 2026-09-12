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

Current classification: resident CPU/DRAM-kernel execution is the larger term
in the lazy path (about 212 ms/token resident plus about 158 ms/token added
lazy-storage stall at the measured Phase 0 point). The single-row experiments
show that panel-backed specialization and dispatch isolation do not improve
the ceiling; raw IQ2 dequantization remains dominant. Storage remains a major
secondary bottleneck, but storage optimization alone cannot reach 10 tok/s.
The next compute experiment should implement a true fused multi-row raw
IQ2_XXS kernel with shared activation loads, then move to a bounded expert
cache sized against the measured 1.61 GiB non-routed floor.
