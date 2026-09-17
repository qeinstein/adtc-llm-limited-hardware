"""Phase 10j: moonshot-C MTP probe v1 verdict -- INCONCLUSIVE (method failure).

mtp-probe-v1 COMPLETE (wall 46min, 32GB box, vanilla @3057bb66, MTP file pinned):

  base    3.30 tok/s   reps [4.2, 2.4]  <-- BASE CONTAMINATED (+/-27%)
  mtp_n1  3.90 tok/s   reps [3.9, 3.9]  nominal +18% (UNSOUND, see below)
  mtp_n2  3.70 tok/s   reps [3.7, 3.7]  nominal +12% (UNSOUND)
  mtp_n3  3.40 tok/s   reps [3.4, 3.4]  nominal +3%
  mtp_n4  2.95 tok/s   reps [2.5, 3.4]  nominal -11%

Why the nominal EXPLOIT verdicts are UNSOUND:
1. Base reps split 4.2 vs 2.4 on IDENTICAL tokens (same response hash) with
   IDENTICAL prefill (6.0 vs 6.1 t/s) => rep-2 decode-only slowdown 1.75x,
   consistent with mid-run neighbor contention on a shared box. Base CV 27%
   DWARFS the effects. Nominal n1 gain is -7% vs base-r1, +62% vs base-r2.
2. 1-decimal quantization: no precise-timing patch, no eval-time line in
   this CLI mode, stderr carries only GPU warnings. +/-1.5% noise.
3. No acceptance telemetry (no --verbose, no n_decode printed).

SOLID signals (within-probe, shared conditions):
a. EXACTNESS HOLDS: all 10 arms byte-identical response hash (4e9f39c9).
   MTP greedy is exact => NO quality gate needed, ever, for this path.
b. WITHIN-MTP MONOTONIC DECLINE n1 > n2 > n3 > n4: extra draft positions
   are pure overhead => single-head acceptance too low for N>1 to help.
   The ONLY open question is n1-vs-base magnitude.
c. RSS 11.6-11.9 GB (+250MB MTP ctx); invocation verified working
   (no -md, mem-shared single load).

REFINE-ONCE (method fix, cheap ~25min, no transcode):
  mtp-probe-v2: precise %.6f timing patch; INTERLEAVED base/n1 x4 pairs +
    n2 x2 (contamination-resistant); method gate (base CV must be <10%
    else INCONCLUSIVE); one trailing n1 --verbose telemetry arm (excluded
    from means) to capture acceptance stats if printed.
"""

MTP_V1 = {
    "verdict": "INCONCLUSIVE_METHOD_FAILURE",
    "base_tps": 3.30, "base_reps": [4.2, 2.4],
    "mtp_n1_tps": 3.90, "mtp_n2_tps": 3.70,
    "mtp_n3_tps": 3.40, "mtp_n4_tps": 2.95,
    "exactness": "all arms byte-identical (no quality gate needed)",
    "solid_signal": "n1>n2>n3>n4 within-probe: N>1 pure overhead",
    "refinement": "mtp-probe-v2: precise timing + interleaved pairs + method gate + verbose telemetry",
}
