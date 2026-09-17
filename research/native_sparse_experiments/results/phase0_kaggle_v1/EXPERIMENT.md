# Phase 0 instrumentation calibration (failed closed)

- Hypothesis: an observation-only route hook and lazy-policy arms can execute
  the exact native Qwen3.5 route on Kaggle CPU.
- Checkpoint: `Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf`, immutable Hugging Face repo
  commit `bc014a17be43adabd7066b7a86075ff935c6a4e2`.
- Runtime: llama.cpp `3057bb66c86c46d5781e50e85462a760ba7d1feb`.
- Hardware: Kaggle CPU, 4 logical AVX2 CPUs, Intel Xeon 2.20 GHz, 32.87 GB host RAM.
- Outcome: model build, download, hash, native route trace, and one 24-token
  eager inference completed. The harness then deliberately failed because its
  legacy llama timing parser did not recognize current llama-cli's response
  summary format.
- Partial measurement: eager output reported 4.9 generation tok/s and produced
  23 complete decode route records, each containing 40 layers and native top-8
  expert IDs. These partial numbers are calibration evidence, not the formal
  baseline.
- Conclusion: keep the inference path; fix measurement parsing. No model or
  runtime conclusion is drawn from this failed experiment.

The raw Kaggle log, route trace, process samples, source patch, commands, and
hardware record are preserved below this directory.
