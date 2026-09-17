# Phase 7D — Selective-Q4 bounded v2 (no-repack matched)

## Hypothesis

Disabling the accidental Q4_K CPU repack removes the v1 anonymous-RSS
penalty while preserving the selective-Q4 bounded speed gain. Both arms use
`--no-repack`.

## Configuration

- Control: `Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf`
  (`2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`)
- Challenger: selective attention/GDN Q4_K
  (`86f27f69e0cea7f8cd0420452f1b600c57cedce6589f60c2cfee06a8c0395d39`)
- Native routing: top-8, router and experts unchanged
- Runtime commit: `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Research harness source: `cae43e5` (`kaggle/native-sparse-q4-bounded-v2/`)
- Kaggle kernel: `toheebogunade/jamii-native-sparse-selective-q4-bounded-v2`
- Kaggle: Linux, 4-vCPU Intel Xeon @ 2.20 GHz, AVX2, four compute threads
- Cache: 2,000,000,000-byte capacity; 876,544-byte slots; 2,281 slots
- Decode: 64 tokens, three repetitions per arm, temperature 0, seed 1234

## Results

| arm | decode tok/s (repetitions) | mean | peak RSS (max) |
|---|---|---:|---:|
| exact IQ2 control | 2.539814, 2.594522, 3.150681 | 2.761672 | 3514.62 MiB |
| selective-Q4 | 2.869955, 2.727493, 2.602488 | 2.733312 | 3392.98 MiB |

Selective-Q4 is ~1.03% slower on the matched bounded mean and reduces peak
RSS by ~121.6 MiB. The previous ~430 MiB RSS regression was an automatic
Q4_K CPU-repack confound, now removed.

## Correctness

Control responses match the established smoke hash
`5804c970f14ce9902178660a5262689f2ffb3518619336403244933d85cafca1`.
The challenger is a representation change, so its response hash differs by
design; router and expert weights are unchanged. Broad quality is gated
separately by the likelihood-based quality v3 harness.

## Decision

**KILL as a throughput path.** Do not preserve a slower representation
merely because work was invested in it — unless a later radically different
execution schedule makes the Q4 representation uniquely useful.

Raw Kaggle artifacts are in `raw/`.
