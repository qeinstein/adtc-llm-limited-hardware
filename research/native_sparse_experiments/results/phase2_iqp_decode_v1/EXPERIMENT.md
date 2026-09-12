# Phase 2 IQ2_XXS single-token expert-kernel A/B

- Experiment ID: `phase2_iqp_decode_v1`
- Hypothesis: allowing the existing IQ panel `MUL_MAT_ID` kernel at one
  routed row per expert may improve exact single-token IQ2_XXS decode.
- Research source commit: `2b1a3ad`
- Kaggle kernel: `toheebogunade/jamii-native-sparse-iqp-decode-sweep`,
  version 1
- Checkpoint: `Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf`, 10,656,955,008 bytes,
  SHA-256 `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`,
  repo commit `bc014a17be43adabd7066b7a86075ff935c6a4e2`
- Runtime: llama.cpp `3057bb66c86c46d5781e50e85462a760ba7d1feb`,
  patched only to make `GGML_IQP_MIN_BATCH_ID` process-selectable
- Hardware: Kaggle CPU, 4 logical / 2 physical Intel Xeon AVX2 cores
- Configuration: resident mmap, lazy off, CPU-only, 4 threads, poll 0,
  64-token decode, three repetitions

Kernel finding:

The pinned runtime supports IQ2_XXS in its IQ panel `MUL_MAT_ID` path, but
the upstream per-expert minimum batch is eight. Exact single-token native
top-8 routing selects eight distinct experts, leaving one activation row per
selected expert. Upstream therefore uses the generic vec-dot path. The patch
preserved threshold 8 as the default and allowed threshold 1 only by
environment variable. `GGML_NO_IQ_PANEL=1` provided a same-binary generic
control.

| arm | mean tok/s | samples tok/s | peak RSS |
|---|---:|---|---:|
| upstream threshold 8 | 5.168 | 5.127, 5.148, 5.228 | 10,534.8 MiB |
| forced IQ panel threshold 1 | 2.940 | 2.930, 2.948, 2.941 | 10,535.0 MiB |
| threshold 1 + IQ panel disabled | **5.196** | 5.168, 5.213, 5.207 | 10,534.8 MiB |

Forcing the panel path reduced throughput by 43.42% against the same-binary
disabled control and changed RSS by only +0.21 MiB. The likely cause is the
existing panel's shape: it decodes eight weight rows and its short-tail tile
duplicates one activation into four lanes, doing work designed for a larger
per-expert batch. Merely lowering the guard is therefore rejected.

Correctness:

The raw `result.json` says `invalid_result` because the first version of the
response extractor hashed CLI loading-spinner bytes before the generated
text. Direct byte extraction from `[Start thinking]` to the performance
footer proves that generic and forced-IQP generated payloads are identical.
Both hash to
`a3c3c73a067cbb5fab9e68bd7d06c03d3c488e74467d33c40fae339836006c8c`,
which also matches Phase 0. This is deterministic 24-token smoke equality,
not a broad quality evaluation. The original machine result is retained
unchanged; `ANALYSIS.json` records the corrected interpretation.

Routing and traffic:

The patch does not alter the model graph, router, native K=8, selected IDs,
weights, loading mode, or expert residency. No expert is dropped or
substituted. The resident arms schedule zero fresh expert bytes per decoded
token; logical selected payload remains 280,494,080 bytes/token and is
unchanged by the optimization.

Conclusion:

The bottleneck classification survives: the exact resident CPU path is the
hard ceiling, and the existing batch-oriented IQ panel cannot be repurposed
for single-row experts by changing a threshold. The next runtime experiment
should implement or isolate a dedicated IQ2_XXS `MUL_MAT_ID` 8-output-row by
1-activation-row kernel, avoiding four-lane duplication and minimizing
per-expert dispatch. Storage/cache work remains necessary afterward, but it
cannot raise throughput above this compute ceiling.
