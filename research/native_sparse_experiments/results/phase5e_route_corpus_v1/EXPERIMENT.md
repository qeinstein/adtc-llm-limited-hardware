# Phase 5E/F — diverse exact route corpus and cache-policy replay

## Corpus

The original 23-token trace was too small for cache policy selection.  This
Kaggle collection decoded 32 independent prompts with the unchanged
Qwen3.5-35B-A3B IQ2_XXS runtime and recorded the exact native selected IDs at
every routed layer.  It produced 2,016 tokens, 645,120 route requests, and
9,577 unique `(layer, expert)` bundles.  The corpus spans clinical, MCQA,
reasoning, general, instruction, short, and Kiswahili prompts.

- Runtime: llama.cpp `3057bb66c86c46d5781e50e85462a760ba7d1feb` plus an
  observation-only route hook.
- Model SHA-256:
  `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`.
- Kaggle: 4-vCPU AVX2 Xeon, four threads, `-n 64`, context 512, deterministic
  per-prompt seeds.
- Corpus SHA-256:
  `17c2861f95cd56eeb8f79d6a6ab5462b809f46e10e5d02f53558a760357fd0d0`.

Adjacent tokens share 100.27 of 320 layer/expert requests on average
(Jaccard 0.1956).  Same-layer retention averages 31.33%, with a range of
7.18%–39.89% across layers.  Exact LRU reuse distances have median 842 and
p95 5,310 distinct bundles, confirming that the old 23-token trace was not a
safe proxy for a long interactive workload.

## Cache replay

The analyzer uses the measured 876,544-byte bundle and a conservative budget
of 1,612.9 MiB floor plus 256 MiB reserve.  Thus the control slots are 2,664,
1,439, and 826 at total RSS targets of 4, 3, and 2.5 GiB respectively.

| Policy | 4 GiB hit / fresh MB/token | 3 GiB hit / fresh MB/token | 2.5 GiB hit / fresh MB/token |
|---|---:|---:|---:|
| Global LRU | 77.20% / 63.96 | 61.90% / 106.87 | 48.79% / 143.63 |
| Layer-partitioned LRU | 76.32% / 66.41 | 62.09% / 106.33 | 49.12% / 142.71 |
| Online frequency+recency | 64.96% / 98.28 | 46.25% / 150.78 | 33.28% / 187.14 |
| Online predicted-next-use | 30.49% / 194.96 | 16.96% / 232.93 | 9.87% / 252.82 |
| Static popularity hindsight control | 73.97% / 73.01 | 56.20% / 122.85 | 42.27% / 161.94 |
| Belady next-use hindsight oracle | 89.37% / 29.80 | 79.78% / 56.73 | 69.95% / 84.29 |

All fresh-byte figures are logical packed bytes supplied to the kernel, not
physical NVMe traffic.  Policy metadata is negligible at this scale under a
compact 16-byte entry estimate: 41.6 KiB at 2,664 entries, 22.5 KiB at 1,439,
and 12.9 KiB at 826.  The cache report also records p50/p95 miss size (both
876,544 bytes here) and per-layer counts.

## Decision

Global LRU is the practical baseline.  Partitioning helps only slightly at
the smaller budgets and hurts at 4 GiB, so it is not selected without a
layer-aware executor reason.  The two simple online policies tested are
clearly worse on this mixed corpus; the large Belady gap (29.80 vs 63.96 MB
fresh/token at 4 GiB) shows that future-aware route prediction or workload
segmentation may still offer headroom, but Belady itself is not deployable.

The next storage implementation should therefore start with a byte-bounded
global LRU and measure actual pread traffic.  Cache policy complexity is not
justified yet.  The full raw corpus and replay JSON are stored beside this
report.
