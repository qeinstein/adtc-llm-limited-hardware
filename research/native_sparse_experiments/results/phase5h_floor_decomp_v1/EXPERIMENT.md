# Phase 5H — non-MoE resident-floor decomposition

## Hypothesis

The earlier 1.61 GiB non-MoE floor should be separated into logical model
tensors and actual anonymous/file-backed RSS.  This determines how much of a
strict 2.5–3 GiB budget is available to an explicit expert cache.

## Reproducibility

- Kaggle kernel: `toheebogunade/jamii-native-sparse-non-moe-floor-decomposition-v1`
- Runtime: llama.cpp `3057bb66c86c46d5781e50e85462a760ba7d1feb`, with only the
  established routed-expert lazy flags plus measurement hooks.
- Model: Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf, SHA-256
  `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`.
- Kaggle 4-vCPU AVX2 Xeon, four CPU threads, `-c 512`, `-n 64`, `-lzm on`,
  `--poll 0`.
- The routed `MUL_MAT_ID` output was zeroed before expert access.  This is a
  RAM measurement-only arm and is not a valid model output.

## Measured RSS

| Quantity | MiB |
|---|---:|
| Peak process RSS | 1,609.965 |
| Anonymous RSS | 536.902 |
| File-backed RSS | 1,073.066 |
| Major faults | 68 |
| Minor faults | 194,309 |

This reproduces the prior approximately 1.61 GiB operational floor.  It is
slightly below the earlier 1,612.9 MiB sample because the Kaggle process and
sampling schedule differ; the difference is not treated as a model change.

## Unique tensor inventory

The loader emitted each tensor twice through its normal metadata paths.  The
following table deduplicates by tensor name and sums logical `ggml_nbytes`:

| Category | Tensors | Logical bytes | GiB |
|---|---:|---:|---:|
| Input embeddings | 1 | 286,064,640 | 0.2664 |
| LM head/output | 1 | 286,064,640 | 0.2664 |
| Attention/DeltaNet trunk | 330 | 921,070,080 | 0.8578 |
| Norms | 81 | 663,552 | 0.0006 |
| Router | 40 | 83,886,080 | 0.0781 |
| Shared expert | 160 | 92,405,760 | 0.0861 |
| **Non-routed total** | **613** | **1,670,154,752** | **1.5555** |
| Routed expert tensors (lazy) | 120 | 8,975,810,560 | 8.3594 |

The inventory has 733 unique tensors in total.  The non-routed logical total
is larger than active non-routed file-backed RSS because mmap pages are
resident on demand; anonymous runtime state accounts for much of the rest.
The operational floor, not the logical sum alone, is the current cache-budget
baseline.

## Decision

An exactly bounded routed cache remains structurally compatible with `<4 GiB`:
the measured floor leaves roughly 2.23 GiB after a conservative 256 MiB
reserve.  The 2.5 GiB target leaves about 691 MiB for fixed expert slots under
the same reserve, so staging and scratch must be measured rather than added
implicitly.  The input/output embedding pair alone is about 546 MiB logical;
lazy/streaming or a more compact output path is the most promising floor
reduction target if 2.5 GiB proves too tight.

Raw inventory, stdout/stderr, runtime patch, and result JSON are preserved in
this directory.
