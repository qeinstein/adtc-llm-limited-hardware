# Probe 11b tooling: exact weight recovery + stacked SVD

Reusable for future weight-structure analyses (distillation, sparsity).

## Files

- `dequant_tool.c` — exact ggml dequant of one GGUF tensor slice to raw F32.
  Build against a llama.cpp checkout at the pinned commit:
  `gcc -O2 -o /tmp/dequant dequant_tool.c ggml/src/ggml-quants.c
  -Iggml/include -Iggml/src -lm`
  (has stub `ggml_abort/row_size/type_size/type_name`; dequant path never
  calls them). Usage:
  `dequant in.bin out.f32 <iq2_xxs|iq2_s> ne0 nrows`
- `sv_gate64.npy` — 2048 stacked singular values, layer-10 gate, 64 experts
  (32768x2048, exact IQ2_XXS dequant). Rank-512 keeps 38.9% energy.
- `stacked_svd.py` — Gram-route SVD (eig of W'W; direct SVD OOMs small boxes).

## GGUF slicing notes (learned the hard way)

- Data section starts at `align32(tensor_infos_end)` = 10989695 -> **10989696**
  ( NOT +32; off-by-32 misalignment decodes to NaN scales — validate with
  `isfinite` + weight stats mean~0 std~0.01 ).
- Expert `e` of `_exps` tensors is contiguous at `e * expert_bytes`:
  gate/up IQ2_XXS: 512 rows x 8 blocks x 66 B = 270336 B;
  down IQ2_S: 2048 rows x 2 blocks x 82 B = 335872 B.
