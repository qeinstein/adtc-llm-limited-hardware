"""Phase 11a / PROBE 1 (fused MoE, SPEEDUP-ARCH #1+#3+#4): KILL (ceiling ~2%).

1. HYPOTHESIS: joint top-8 execution + fused gate/up + direct weighted
   down-accumulation materially cuts MoE cost (87ms core, staged box).
2. IMPLEMENTATION: source inspection at pinned 3057bb66 (no prototype: ceiling
   decisive). MUL_MAT_ID: ggml-cpu.c:1542; MoE graph: llama-graph.cpp:1993;
   qwen35moe FFN: qwen35moe.cpp:494.
3. BASELINE: per layer decode: gate MMID + up MMID + swiglu_split + down MMID
   + mul(weights) + 7 chained adds = 12 nodes; 480 MoE nodes/token; 87ms core.
4. RESULT:
   - #1 joint top-8: ALREADY IMPLEMENTED. MUL_MAT_ID groups rows by expert and
     dispatches all 8 jointly with dynamic chunking. Gain: 0.
   - #3 fused gate/up: runtime ALREADY SUPPORTS merged gate_up path
     (llama-graph.cpp:2169; qwen35moe passes ffn_gate_up_exps, qwen35moe:509);
     our file just lacks the tensor. Transcoder-only. Saves 1 dispatch +
     1 x F32->Q8 conversion/layer ~ 10us x 40 = ~0.4ms (~0.5% core).
   - #4 direct accumulation: saves ~360KB/layer L2-resident traffic (64KB
     down-dst + 128KB mul + 168KB adds = ~0.2ms) + 8 tiny dispatches (~1ms
     generous) = ~1.2ms (~1.5% core).
   - Stacked ceiling ~1.6-2.1ms (~2% core, ~0.6% e2e). NO weight-byte
     reduction possible (same 8 experts must be read).
5. MS/TOKEN: 87 -> ~85 projected (core); e2e staged 312.7 -> ~311.
6. TOK/S: 3.20 -> ~3.22 projected (within noise).
7. PARITY: moot (no prototype). #4 would be exact mod FP order.
8. MEMORY: -15MB/token L2-resident temp traffic; zero DRAM/working-set change.
9. VERDICT: KILL (<3% bar; even generous dispatch math stays ~2% core).
10. NEXT: none for MoE fusion. (If ever revisited: #3 is transcoder-only.)

Evidence: /tmp/ggml-cpu.c, /tmp/llama-graph.cpp, /tmp/qwen35moe.cpp @3057bb66.
"""

PROBE1_FUSED_MOE = {
    "verdict": "KILL",
    "ceiling_ms": 2.0, "ceiling_pct_core": 0.02, "ceiling_pct_e2e": 0.006,
    "finding_1": "joint top-8 dispatch already implemented (MUL_MAT_ID)",
    "finding_3": "gate_up runtime path exists; transcoder-only; ~0.4ms",
    "finding_4": "~360KB/layer L2 traffic + 8 dispatches; ~1.2ms",
}
