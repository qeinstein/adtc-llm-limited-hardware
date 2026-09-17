# Phase 1 CPU sweep v1 — pre-measurement failure

- Experiment ID: `phase1_cpu_sweep_v1`
- Kaggle kernel: `toheebogunade/jamii-native-sparse-cpu-thread-sweep`, version 1
- Research base commit: `58594d2d8f53ad7628ece93c17b636fb153e812f`
- Submitted hardening commit: `3c175c7`
- Runtime target: llama.cpp `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Hardware: Kaggle CPU, 4 logical / 2 physical AVX2 Xeon cores
- Status: failed before model download or benchmark execution

The llama.cpp build completed successfully. The pinned `llama-bench` binary
does not support `--version`; it printed usage and returned exit code 1. The
script correctly failed closed, wrote `result.json`, and produced no throughput
measurement. Version 2 replaces this unsupported probe with
`git rev-parse HEAD`. This run is preserved as harness evidence and must not be
interpreted as a model or runtime result.
