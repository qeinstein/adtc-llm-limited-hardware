# Phase 5B-v2 — packed exact IQ2-derived layout follow-up

## Hypothesis

The Phase 5B-v1 exact resolved layout may have lost performance to its
74-byte naturally padded record.  Removing that padding should reduce the
cache footprint and recover the IQ2_XXS AVX2 ceiling without changing values.

## Conditions

- Kaggle 4-vCPU AVX2/FMA Xeon, 80 repeats, 8 sequential matrices.
- Shapes `512 x 2048` and `2048 x 512`, matching Qwen gate/up/down geometry.
- `-O3 -DNDEBUG -mavx2 -mfma -march=haswell -std=c++17`.
- Same llama.cpp IQ2 decoder, random deterministic blocks, and activation for
  the control and candidate.

The candidate packs each resolved group as eight 16-bit combined grid/sign
indices plus one byte of scale nibbles: 70 bytes per 256-weight block,
2.1875 bpw, versus 66 bytes and 2.0625 bpw for IQ2_XXS.  The dot checksum
matched the control before timing.

## Results

| Shape | IQ2 AVX2 GMAC/s | Packed exact GMAC/s | Candidate delta | Control bytes | Candidate bytes |
|---|---:|---:|---:|---:|---:|
| 512 x 2048 | 6.570 | 4.407 | -32.9% | 2,162,688 | 2,293,760 |
| 2048 x 512 | 6.211 | 4.229 | -31.9% | 2,162,688 | 2,293,760 |

The 70-byte candidate is slower than the 74-byte v1 result as well as the
IQ2 control in this run.  Packing recovered storage (+6.06% rather than
+12.12%) but did not recover execution speed; unaligned field access and the
same per-code lookup/accumulator schedule remain expensive.

## Decision

Padding-only and metadata-resolution-only layouts are rejected as direct
compute optimizations.  The exact IQ2 control remains the representation
control.  A useful replacement now needs a different arithmetic/access
schedule—such as a genuinely LUT-native tile kernel—or must target the larger
non-MoE matrix share identified by the Phase 5A Amdahl profile.  The packed
format remains a possible storage format only if a future executor consumes it
without the measured lookup penalty.

Raw Kaggle output, compiler log, source result, and stdout are preserved in
this directory.
