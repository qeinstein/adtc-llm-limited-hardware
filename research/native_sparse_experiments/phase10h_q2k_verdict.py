"""Phase 10h: Q2_K ablation VERDICT — refinement hypothesis TESTED and CONFIRMED-ish.

Question (from 10g): was the -9.9% Q2_K expert slowdown kernel or traffic?
Test: J2 -- Q2_K for experts+DENSE (dense Q2_K requant has ~zero traffic delta,
resident in cache) isolates the Q2_K KERNEL delta cleanly. If J2 >> J1, the
expert slowdown is traffic (staged path copies +34% more bytes through a SLOW
memcpy box), NOT kernel - and the kernel is a KEEP under zero-copy.

VERDICT: **CONFIRMED. Q2_K kernel is a KEEP; expert traffic is the killer.**

  staged-IQ2   (control):    5.09 tok/s  (196.5 ms/tok)   [3 reps, box-fast]
  staged-Q2K-experts (J1):  4.58 tok/s  (218.3 ms/tok)   -10.0%  *** REPRODUCED
  staged-Q2K-ALL (J2):      5.06 tok/s  (197.5 ms/tok)   -0.5%   *** CONFIRMED
  resident-IQ2   (ceiling):  7.95 tok/s  (125.8 ms/tok)   +56%

Decomposition (measured, staged path, this box):
  J2-vs-J1 = DENSE Q2_K delta = +21 ms (+9.6%): kernel WIN, ~zero traffic.
    => Q2_K vec_dot kernel is FASTER than Q4_K/Q5_K/Q6_K paths (matches ubench
       -44% on 1-row experts and -63% on dense rows).
  J1-vs-control = EXPERT net = -22 ms: kernel-win(K) MINUS traffic-penalty(T).
    => With K ~ 25-39 ms (ubench-scaled), T ~ +47-61 ms for +2.74 GB expert
       traffic through ~0.7-0.9 GB/s effective staged copies (async workqueues
       + slow box memcpy: 16 threads x ~3 s each per run).
  => Under zero-copy (traffic ELIMINATED for experts via v5 remap, see 10e),
     the expert Q2_K delta should approach the pure kernel win: +25-39 ms,
     i.e. zc-Q2K-experts ~ +35-45% over staged-IQ2 control, near the ceiling.

Path audit (route_trace, rep 2, n=320 sample rows):
  decode cne1: ALL arms = 1 (single expert per row) => generic vec_dot path in
    BOTH IQ2 and Q2K decode. Fair kernel+traffic comparison - verdict stands.
  prompt cne1 medians differ (iq2: 8,5,5,12,13,14,10; q2k: 1,1,2,1,1,1,1) =>
    IQ2 prompt hits IQP batch panels (cne1>=8), Q2K prompt stays generic
    (Q2_K has no dequantize_row_q2k path). So PREFILL compares paths too;
    decode numbers above are the clean verdict. (Prompt decode-only bench:
    no prompt tok/s logged in this kernel; J1/J2 decode rates are valid.)
  routes overlap 100% (4096/4096), id-hash identical => requant preserves
    routing; sample outputs byte-identical (iq2 vs j1).

J2 generation-behavior note: Q2K-ALL answered DIRECTLY (27 tokens, EOS, no
thinking trace, correct+concise) vs IQ2's 64-token thinking ramble (hit -n).
Same 29-token prompt; requests 53760 vs 89280 explained EXACTLY by fewer
decode tokens (120 nodes x 8 x (29+G)). NO corruption. Behavior delta is a
QUALITY-GATE input, not a speed confound (per-token rates are valid).

Rep-1 writeback storm (method fix for follow-ups): rep 1 of EVERY arm shows
read_ns thread-sum 500-680 s vs ~100-200 s in reps 2-3 (39 s elapsed!) from
dirty transcode-output writeback competing with staged reads on shared disk.
Reps 2-3 (clean): q2k -4.5%, q2kall +5.3% (refine-territory even staged!).
Follow-up kernel adds sync + settle + cache-drop before EACH arm.

REFINEMENT (promoted to main queue): combination kernel
  native-sparse-zc-q2k-combo-v1: sequential experts+all transcodes (sync+settle
  +delete between), arms: resident_iq2 x2, staged_iq2 x3 [control], zc_iq2 x3
  [v5 cert], zc_q2k x3 [combo], zc_q2kall x3 [J2-combo], resident_q2k x1
  [kernel-win isolation, OOM-guarded].
  Predictions: resident_q2k ~ 10 tok/s (K isolated); zc_q2k +35-45% over
  staged control (~6.5-7.5 tok/s this box); zc_q2kall similar+.
  Quality gate (likelihood MMLU, Q4-v3 pattern) follows ONLY if speed wins.

Files: kaggle/native-sparse-q2k-v1/ (kernel), this record.
Evidence: /tmp/zc_results_q2k/native-sparse-q2k-v1-results/{result.json,
stdout.txt} (Kaggle output; also committed session-side on push).
"""

Q2K_VERDICT = {
    "control_staged_iq2": {"tps": 5.09, "ms_per_tok": 196.5},
    "j1_staged_q2k_experts": {"tps": 4.58, "ms_per_tok": 218.3, "delta_vs_control": -0.100},
    "j2_staged_q2k_all": {"tps": 5.06, "ms_per_tok": 197.5, "delta_vs_control": -0.005},
    "ceiling_resident_iq2": {"tps": 7.95, "ms_per_tok": 125.8, "delta_vs_control": +0.56},
    "dense_q2k_kernel_win_ms": 21,   # J2 - J1, ~zero traffic delta
    "expert_kernel_win_ubench_ms": (25, 39),
    "expert_traffic_penalty_ms": (47, 61),
    "reps2_3_clean": {"q2k": -0.045, "q2kall": +0.053},
    "mechanism": "KEEP kernel (fast vec_dot), KILL staged traffic (+34% bytes through slow copies)",
    "refinement": "zc-q2k-combo-v1: Q2_K experts x v5-remap zero-copy + resident_q2k K-isolation + settle-fix",
}
