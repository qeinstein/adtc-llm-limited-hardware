# Phase 5B — external W2 LUT and exact IQ2-derived layout ceiling

## Hypothesis

If a different low-bit access pattern is substantially faster on Qwen's actual
expert geometries, the current IQ2 metadata path—not 2-bit arithmetic alone—is
the dominant representation problem.

## Reproducibility

- Kaggle kernel: `toheebogunade/jamii-native-sparse-external-w2-lut-ceiling-v1`
- Research source: `cd2be78` plus the uncommitted harness at submission time
- Runtime source: llama.cpp `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Hardware: Kaggle CPU, AVX2/FMA; compiler flags `-O3 -DNDEBUG -mavx2 -mfma -march=haswell -std=c++17`
- Eight sequential expert matrices, 80 repeats, shapes `512 x 2048` and
  `2048 x 512`; these are the two projection geometries used by gate/up/down.
  This is a kernel ceiling microbenchmark, not a fused three-stage expert
  execution benchmark.

The external method is a T-MAC-style W2A8 lookup experiment, not the T-MAC
implementation itself and not an IQ2 quality comparison. T-MAC's primary
lookup-table approach is described in its [official repository](https://github.com/microsoft/T-MAC/)
and [paper](https://arxiv.org/abs/2407.00088).

## Representations

- Existing IQ2_XXS: 66 bytes per 256-weight block, 2.0625 bpw.
- Exact custom `resolved_code_sign_index`: 74 bytes per block, 2.3125 bpw,
  +12.121%. It combines each 8-bit grid ID with its 7-bit sign ID and stores
  the two scale nibbles directly. A 32,768-entry `(sign, grid)` table resolves
  the same byte values and sign masks.
- External unsigned W2A8: 2.0 bpw, prepacked low/high bit planes. This changes
  the numerical representation and is a performance ceiling only.

## Results

| Shape | IQ2 AVX2 GMAC/s | Exact custom GMAC/s | W2 scalar LUT GMAC/s | W2 AVX2 gather GMAC/s |
|---|---:|---:|---:|---:|
| 512 x 2048 | 7.208 | 5.297 | 1.247 | 1.685 |
| 2048 x 512 | 6.427 | 5.515 | 1.348 | 1.863 |

The exact custom layout preserved the dot result exactly in both shapes but was
26.5% slower for `512 x 2048` and 14.2% slower for `2048 x 512` in this run.
The simple external LUT/gather kernels were also slower than the current IQ2
AVX2 kernel, so this particular LUT construction is not a competent replacement
for the current path. It does not invalidate T-MAC's published kernels; it
shows that merely adding scalar/gather lookups is insufficient for this shape.

## Conclusion

The tested exact layout is rejected as a direct optimization. The result still
supports changing the on-disk representation, but the next candidate must
change the arithmetic/access schedule more deeply—e.g. a vectorized table
layout or a custom executor that fuses the three expert projections. No model
weights, routes, K, or capability representation were changed.

Raw benchmark source, compiler log, stdout/stderr, and `result.json` are stored
beside this report.
