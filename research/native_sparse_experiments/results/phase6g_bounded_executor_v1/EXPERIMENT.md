# Phase 6G bounded expert executor v1 — descriptor-lifetime failure

## Hypothesis

An explicit `pread`-backed fixed global-LRU cache can replace routed-expert
`mmap` residency in the exact Qwen3.5-35B-A3B path while preserving native
top-8 routing and output behavior.

## Configuration

- Model: `Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf`, 10,656,955,008 bytes,
  SHA-256 `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- Model source: `unsloth/Qwen3.5-35B-A3B-GGUF`, commit `bc014a17be43adabd7066b7a86075ff935c6a4e2`
- llama.cpp: `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Research base: `2c9e965`
- Kaggle: CPU-only, 4 threads, inherited affinity, `-ngl 0`, `--poll 0`
- Decode: 64 tokens, 3 planned repetitions, fixed clinical prompt,
  temperature 0, seed 1234, context 512
- Cache arm: 2,000,000,000-byte anonymous fixed storage, full per-expert
  gate/up/down bundle, explicit `pread`, global LRU, `-lzm on`
- Control arm: same patched runtime and prompt, `-lzm off`

## Result

The control arm completed one repetition and produced the established-looking
summary of approximately 4.8 generation tok/s. Its route trace contains 7,920
`MUL_MAT_ID` events, covering the full 40-layer 64-token decode (`3` routed
matrix nodes per layer/token).

The bounded arm entered the real model path and reached cache preparation, but
aborted before generation with:

```text
PHASE6_BOUNDED_CACHE_ERROR pread failed
```

No bounded throughput, RSS, or traffic number is valid from this run. The
partial process trace and all stdout/stderr/build artifacts are preserved in
`raw/`.

## Diagnosis and conclusion

The loader registration passed the original `llama_file::file_id()` into a
process-global C cache. `llama_model_loader` owns that descriptor and is a
temporary object during model construction; after construction the descriptor
can be closed before decode. The failure is therefore descriptor ownership,
not evidence against explicit caching, the cache capacity, or the route
corpus prediction.

Version 2 duplicates every registered descriptor and owns the duplicate until
process exit. It is the direct corrective rerun and does not change model
bytes, routes, cache policy, cache capacity, or arithmetic.

## Artifacts

- Raw Kaggle output: `raw/`
- Corrective harness: `kaggle/native-sparse-bounded-executor-v2/`
- Kaggle kernel: `toheebogunade/jamii-native-sparse-bounded-executor-v1`
