"""Phase 11c / PROBE 3 (exact backbone fusion, SPEEDUP-ARCH #5/#6/#7): KILL.

3A. RESIDUAL+RMSNorm (#5/#6):
1-4. Per norm (d=2048 F32): residual 24KB + rmsnorm 32KB = 56KB; #6 saves the
   8KB var-read (640KB/token over 80 norms); #5 (fold g into W) saves ~8KB
   more but REQUIRES REQUANTIZATION (approximate; violates exactness).
   Stacked ceiling ~0.06-0.1ms (~0.03% e2e). 9. KILL HARD. 10. None.

3B. DELTANET STATE FUSION (#7):
1-4. Inspected ACTUAL implementation @3057bb66: per-token update+query is
   ALREADY FUSED in ggml_compute_forward_gated_delta_net_one_chunk
   (ops.cpp:10895: memcpy state in, decay/delta-rule/query in scratch,
   K=1 writes straight to output slot). Decode uses K=1
   (delta-net-base.cpp:402) + one ggml_cpy/new_state persist per layer
   (delta-net-base.cpp:553). State = 128x128x32 F32 = 2MB/layer x 30.
   Current: 8MB/layer (memcpy R+W + cpy R+W) = 240MB/token. Ideal in-place:
   4MB/layer = 120MB. Savings 120MB ~ 5-8ms ~ 2-2.5% staged e2e, via DEEP
   recurrent-memory surgery (kernel writes persistent slots; rollback/MTP
   protocol impact). #7-as-specified (1R+1W IN the op) already done.
5. MS/TOKEN: 312.7 -> ~306 projected (staged). 6. TOK/S: 3.20 -> ~3.27.
7. PARITY: in-place is exactly equivalent (same flops/order); moot (no proto).
8. MEMORY: -120MB/token DRAM state traffic (projected).
9. VERDICT: KILL (<3% bar + high effort/risk). Conditional note: if GDN ever
   dominates, the copy-back fusion is the precise target.
10. NEXT: none now. Bypass kernel measured GDN wall 8.4ms (phase11d).

OVERALL: KILL (3A hard on arithmetic; 3B on threshold+effort).
"""

PROBE3_BACKBONE = {
    "verdict": "KILL",
    "a_resid_norm": {"verdict": "KILL_HARD", "ceiling_ms": 0.1,
                     "note": "#5 needs requant (approximate)"},
    "b_deltanet": {"verdict": "KILL", "ceiling_ms": 8, "ceiling_pct_e2e": 0.025,
                   "finding": "K=1 op already fused 1R+1W; rest is memory-module copy-back",
                   "state_mb_per_layer": 2.0, "gdn_layers": 30},
}
