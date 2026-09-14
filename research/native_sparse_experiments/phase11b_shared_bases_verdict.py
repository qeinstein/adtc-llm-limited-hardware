"""Phase 11b / PROBE 2 (shared bases, SPEEDUP-ARCH #2): KILL HARD.

1. HYPOTHESIS: per-layer expert matrices share rank-256..512 structure
   (Wg_e approx Ag_e B etc.) justifying distillation (6.4x doc math).
2. IMPLEMENTATION: range-fetched 64 experts x gate (layer 10, IQ2_XXS,
   17.3MB), exact ggml dequant (local C tool), stacked SVD via Gram eig
   (32768x2048; direct SVD OOMs 2.7GB box). Bar stated upfront: rank-512
   rel-Frob >15% => KILL.
3. BASELINE: full-rank gate stack (256 experts/layer, 512x2048 each).
4. RESULT: spectrum essentially WHITE past SV#1 (5.9, 3.7, 3.6, 3.6...).
   rank-64: 96.2% err | 128: 93.3% | 256: 88.1% | 384: 83.1%
   | 512: 78.2% (38.9% energy) | 768: 68.6% | 1024: 58.9%.
   No low-rank structure whatsoever. STOPPED per brief (no up/down, no
   kernels): 78% >> 15% bar; subset-best argument makes 256-expert only worse.
5. MS/TOKEN: no path (reconstruction fails; distillation moot).
6. TOK/S: no path.
7. PARITY: N/A (representation discarded before numerics matter).
8. MEMORY: moot; note rank-512 shared B alone = 1M params/layer (no savings
   even if it worked).
9. VERDICT: KILL HARD.
10. NEXT: none for shared bases. Tooling kept (probe11_shared_bases/:
    dequant_tool.c, svd script, sv_gate64.npy) for future weight analyses.

Evidence: /tmp/sv_gate64.npy (2048 SVs), gate64.f32 (256MB, NOT committed).
"""

PROBE2_SHARED_BASES = {
    "verdict": "KILL_HARD",
    "stack": "layer10 gate, 64 experts, 32768x2048, exact IQ2_XXS dequant",
    "rel_err": {64: 0.9617, 128: 0.9329, 256: 0.8805, 384: 0.8306,
                512: 0.7819, 768: 0.6856, 1024: 0.5885},
    "top_sv": [5.9, 3.7, 3.6, 3.6, 3.6],
    "bar": "rank-512 rel-err > 0.15 => KILL; measured 0.78",
}
