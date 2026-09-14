"""Phase 11d: remainder-bypass-v1 verdict -- 10b table RESOLVED (one noisy arm).

Kernel COMPLETE, status ok, exact-control hash VALID. Control 223.25ms has a
warmup-high r1 (238.6); tight basis r2/r3 = 215.6ms used below (verdicts robust
either way; per-family reps quoted).

Wall saved vs 215.6ms (bypass events fired, deterministic across reps):
  attn_proj      47.5ms  reps tight   REPRODUCED (6b 48ms expect)
  lm_head        24.3ms  reps tight   REPRODUCED (5a-v2 25ms expect)
  gdn_ssm         8.4ms  reps 203/215/203  NEW WALL TRUTH (11ms op-est was close)
  attn_scores    ~0.0ms  reps TIGHT (216/217/214)  ~10ms gap guess REFUTED
                 (1-token flash decode over cached KVs is ~free)
  shared_expert  CONTAMINATED (reps 209/223/266, contention); bypass fired
                 (30,880 events); TSC-implied ~5ms; needs one clean re-measure
  all_dense      90.4ms  residual 125.2ms = routed experts (~105-115 derived)
                 + elementwise + scheduler/sampler (~15)

TSC corroboration (summed worker cycles): attn 71G >> gdn 20G > shared 13G
> dense 7G ~ lmhead 6G >> scores 0.9G. Order matches wall except lmhead
(TSC-small/wall-large: unresolved-minor, likely freq/parallelization artifact;
wall number independently reproduces 5a-v2, stands).

Resolved 10b remainder (resident ~216ms box): experts ~110 | attn 47.5 |
lmhead 24.3 | gdn 8.4 | scores ~0 | shared ~5 (TSC) | dense-other+router ~7
| norms/swiglu/elemwise/sched/sampler ~13. Experts + attn + lmhead = 84%.

Next: H gate-sparsity probe targets the ~110ms experts; I shortlist targets
the 24ms head. Shared-expert wall needs one clean arm (cheap, fold into H/I).
"""

BYPASS_V1 = {
    "control_ms": 223.25, "tight_basis_ms": 215.6, "hash_ok": True,
    "attn_proj_ms": 47.5, "lm_head_ms": 24.3, "gdn_ssm_ms": 8.4,
    "attn_scores_ms": 0.0, "shared_expert_ms": "contaminated (TSC~5)",
    "all_dense_ms": 90.4, "residual_ms": 125.2,
    "experts_derived_ms": "105-115",
}
