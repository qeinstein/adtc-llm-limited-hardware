# Phase 1 exact resident CPU ceiling

- Experiment ID: `phase1_cpu_sweep_v2`
- Hypothesis: Phase 0's approximately 4.9 tok/s resident result may be
  recoverable through thread count, polling, or strict CPU affinity.
- Research commit: `6c175c7`
- Kaggle kernel: `toheebogunade/jamii-native-sparse-cpu-thread-sweep`,
  successful version 2
- Checkpoint: `Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf`, 10,656,955,008 bytes,
  SHA-256 `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`,
  repo commit `bc014a17be43adabd7066b7a86075ff935c6a4e2`
- Runtime: unmodified llama.cpp
  `3057bb66c86c46d5781e50e85462a760ba7d1feb`, native AVX2 CPU build
- Hardware: Kaggle CPU, 4 logical / 2 physical Intel Xeon cores at 2.20 GHz,
  AVX2, 32.87 GB host RAM
- Configuration: resident mmap, lazy off, CPU-only, decode 64 tokens, three
  repetitions per arm

Measured results:

| arm | mean tok/s | samples tok/s | peak RSS |
|---|---:|---|---:|
| 1 thread, poll 50 | 2.329 | 2.363, 2.316, 2.307 | 10,535.1 MiB |
| 2 threads, poll 50 | 3.588 | 3.685, 3.037, 4.042 | 10,535.3 MiB |
| 3 threads, poll 50 | 3.999 | 4.010, 3.995, 3.990 | 10,535.5 MiB |
| 4 threads, poll 50 | 4.614 | 4.543, 4.641, 4.657 | 10,535.3 MiB |
| 4 threads, poll 0 | **4.716** | 4.709, 4.747, 4.693 | 10,535.3 MiB |
| 4 threads, poll 100 | 4.655 | 4.705, 4.530, 4.729 | 10,535.3 MiB |
| 4 threads, poll 0xF strict affinity | 4.691 | 4.691, 4.702, 4.681 | 10,535.3 MiB |

All arms use the same unmodified native router/top-K runtime; there is no
expert drop, substitution, K change, or routing approximation. This benchmark
does not generate text, so its correctness statement is architectural exactness
rather than a new payload-equality test. Phase 0 already established
deterministic route and output equality for the same resident runtime/model.

The strongest tuning gain is only 2.22% over the default four-thread arm.
Strict affinity gains 1.68%. Four logical threads are still best despite only
two physical cores. The prior one-repeat Phase 0 `llama-bench` result of
4.945 tok/s is 4.85% above the new best, consistent with run-to-run host
variance rather than a hidden thread configuration that can reach the target.

At 4.716 tok/s, the resident floor is about 212 ms/token. Phase 0 expert-lazy
decode was about 370 ms/token (2.7 tok/s), leaving about 158 ms/token of
additional page-fault/storage-path latency. In the current lazy system this is
approximately 57% resident CPU/DRAM-kernel time and 43% added lazy-storage
stall. Both matter, but resident CPU execution is the larger term and hard-caps
throughput below 5 tok/s even with zero decode-time expert transfers.

Conclusion: thread, polling, and ordinary affinity tuning do not rescue the
target. The dominant immediate bottleneck is the resident CPU compute/DRAM
kernel path; storage is a major secondary bottleneck. The next experiment
must target the single-token selected-expert IQ2_XXS CPU path before investing
in prefetch or cache engineering alone.
