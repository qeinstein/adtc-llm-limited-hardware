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
research notes; pass the measured packed size instead when available.

## Outputs

`route_trace.py` reports:

- total and per-token route requests, unique layer/expert bundles, and global
  and per-layer popularity;
- adjacent-token Jaccard/shared-request overlap and same-layer retention;
- uncached bytes/token;
- global-LRU hit/miss rates, fresh requests/token, and fresh bytes/token for
  each requested capacity.

Run on Kaggle (after checkout, with no model download required):

```bash
python research/native_sparse_experiments/route_trace.py \
  --trace research/native_sparse_experiments/tests/fixtures/valid_one_token.json \
  --capacities 0,64,128,256,512,1024 \
  --output /kaggle/working/route_report.json
python -m pytest research/native_sparse_experiments/tests -q
```

For a real trace, replace the fixture path. The unit tests exercise shape
validation, duplicate/range rejection, JSON array input, overlap, popularity,
and LRU byte accounting.

