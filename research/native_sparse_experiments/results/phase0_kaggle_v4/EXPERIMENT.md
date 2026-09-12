# Phase 0 exact native-sparse baseline

- Hypothesis: upstream lazy mmap can reduce routed-expert residency without
  changing Qwen3.5's native top-8 route or generated output.
- Research source commit: `aa6848a` (the executed script is also preserved in
  `native-sparse-phase0-results/script.py`).
- Kaggle kernel: `toheebogunade/jamii-native-sparse-phase-0-exact`, successful
  version retrieved 2026-09-12.
- Checkpoint: `Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf`, 10,656,955,008 bytes, SHA-256
  `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`,
  immutable repo commit `bc014a17be43adabd7066b7a86075ff935c6a4e2`.
- Runtime: llama.cpp `3057bb66c86c46d5781e50e85462a760ba7d1feb`, native AVX2 CPU build,
  plus an observation-only route/tensor-size hook and lazy flags on routed
  tensors. Patch SHA-256 is recorded in `result.json`.
- Hardware: Kaggle CPU, 4 logical / 2 physical Intel Xeon cores at 2.20 GHz,
  AVX2, 32.87 GB host RAM, Linux 6.12.90+.
- Commands/config: full argv for every arm is stored in `result.json`; all use
  native K=8, 40 layers, temperature 0, seed 1234, 512 context, and 24 requested
  generation tokens.

Measured results:

| arm | loading | peak RSS | decode | p50 / p95 inter-token | observed decode read counter |
|---|---:|---:|---:|---:|---:|
| eager | mmap, lazy off | 10,528.7 MiB | 4.8 tok/s | 205.5 / 210.1 ms | 1.1 KB/token |
| routed flags + auto | mmap, lazy auto | 10,536.9 MiB | 4.7 tok/s | 210.0 / 223.4 ms | 0 B/token |
| `global_lazy` raw arm name | expert-flagged mmap, lazy on | 5,155.3 MiB | 2.7 tok/s | 350.6 / 511.1 ms | 30.95 MB/token |

The independent resident `llama-bench` decode ceiling was 4.945 tok/s for 32
tokens. The response payload and all 7,360 routed expert references were
identical across arms. Every one of 23 complete decode records had 40 layers,
8 distinct IDs per layer, and IDs in `[0, 256)`. No K reduction, expert drop,
or route approximation occurred. This is deterministic smoke equivalence, not
a broad capability evaluation.

The exact packed routed payload is 876,544 bytes per `(layer, expert)` bundle,
or 280,494,080 logical selected bytes per uncached token. Global LRU replay on
the real trace gave:

| cache bundles | approximate payload capacity | hit rate | logical fresh bytes/token |
|---:|---:|---:|---:|
| 0--256 | 0--214 MiB | 0% | 280.49 MB |
| 512 | 428 MiB | 36.66% | 177.67 MB |
| 1,024 | 856 MiB | 53.06% | 131.67 MB |
| 2,048 | 1,712 MiB | 64.65% | 99.16 MB |

`/proc/<pid>/io` is preserved as an observed kernel counter, not asserted to
be exact physical NVMe traffic for mmap. The tensor-size route replay is exact
logical traffic under its stated cache policy. `posix_fadvise(DONTNEED)` was
requested before each arm but is not claimed as a proof of fully cold cache.

Despite the historical raw arm name, `-lzm on` means that the loader accepts
the tensors explicitly marked `TENSOR_READ_LAZY`; in this patch those are the
routed experts. It is therefore the true expert-lazy arm, not a request to make
every tensor lazy. `auto` rejects these tensors because each is below its 4-GiB
heuristic threshold, leaving the whole-file mapping populated.

Conclusion: the exact native CPU path works. Current resident performance is
CPU/memory-kernel limited below 5 tok/s. Expert-lazy mmap cuts roughly half the
RSS but still misses the 4 GiB target and adds a large page-fault/storage stall.
The `auto` expert-flag attempt had no material effect and is rejected. The next
experiments are a CPU thread/affinity sweep and true expert-only bounded loading;
the measured global-LRU cliff at 320 requests/token also motivates testing a
layer-partitioned cache policy before any model approximation.

`result.json` is the canonical machine-readable result. Raw outputs, route
traces, process samples, build logs, source patch, hardware, and exact script
are preserved beside it.
