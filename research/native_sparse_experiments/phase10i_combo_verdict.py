"""Phase 10i: COMBO verdict -- zc path KILLED on fast-I/O box, kernel win PROVED.

Combo kernel (native-sparse-zc-q2k-combo-v1, COMPLETE, wall 1.3h, 32GB box):

  resident_iq2   4.955 tok/s   (201.8 ms)   ceiling (this box CPU ~1.6x slower)
  staged_iq2     3.602 tok/s   (277.7 ms)   control, 2GB staged
  zc_iq2         2.917 tok/s   (342.8 ms)   -19.0%  KILL_HARD (v5-cert FAILED)
  zc_q2k         2.689 tok/s   (clean 3.12) -13.3% clean  KILL_HARD (combo FAILED)
  zc_q2kall      2.581 tok/s   (clean 2.97) -17.6% clean  KILL_HARD
  resident_q2k   6.529 tok/s   (153.2 ms)   +31.8% vs resident  KERNEL WIN PROVED

Findings:

1. resident_q2k +31.8% (-48.6 ms traffic-free) ISOLATES the Q2_K expert kernel
   win, LARGER than the ubench prediction (25-39 ms). The representation
   thesis is CONFIRMED; the delivery vehicle (bounded/zc backends) is the
   problem. (12 GB RSS: mechanism proof, not a frontier point.)

2. v5 does NOT replicate here (-19% vs +9.41%). Telemetry closes the
   mechanism: zc eliminated ALL userspace copies (rchar 8.2 -> 0.0 GB,
   read_calls 0) but physical I/O is IDENTICAL (6132 vs 6142 MB: fast
   disk/pagecache serves both). The +65 ms loss = remap/fault churn:
   28.5K madvise calls, 8.46 GB re-mapped over 64 tokens (132 MB/token),
   i.e. ~33K pages/token re-faulted (~33 ms) + 8.8 ms madvise + TLB
   shootdowns. Staged faults ONCE per persistent slot (amortized); zc
   re-faults EVERY miss. v5's win is I/O-box-specific: on slow-I/O boxes
   traffic savings dominate; on fast-I/O boxes fault churn dominates.
   Follow-up adds /proc/PID/stat minflt/majflt sampling to PROVE the
   fault counts (this kernel only sampled RSS/IO).

3. OWN-GOAL: combo zc arms used 1.6 GB cache; v5 used 2 GB (zerocopy_v5.py:52
   CACHE_BYTES=2GB; the "1.6GB" was a plan note, not v5's actual). Combo zc
   misses 11347 vs staged 9275 (+22%). Explains only ~+6 ms of the +65 ms;
   the rest is (2). Follow-up runs 2 GB everywhere (v5-verbatim re-cert).

4. Rep-1 outliers persist (zc_q2k r1 1.827, elapsed 72s; identical requests/
   madvise across reps => pure writeback contention). The 30+10s settle did
   NOT drain ~12 GB dirty transcode output. Follow-up: 90s post-transcode
   settle. Reps 2-3 are the signal (above).

5. Route "collapse" (zc_q2k 0.656, zc_q2kall 0.19 overlap) is FULLY explained
   by GENERATION divergence, not backend nondeterminism: response hashes
   show zc_q2k == resident_q2k (2d02d34a, self-consistent across backends),
   zc_q2kall == J2 behavior (7f5a2f42, early-EOS direct answer, 53760
   requests), all-IQ2 arms identical (5804c970, zc_iq2 routes 1.0 exact).
   NOTE: q2k-experts generation DIVERGED from IQ2 here but was byte-identical
   in the q2k kernel => the representation sits ON an argmax boundary for
   some token; box/FP-order sensitive. Quality gate still required.

6. File sizes: experts 12.25 GB (+1.59 file, +2.74 rchar traffic over 10.66);
   all 11.47 GB (+0.81; dense shrank). q2kall file SMALLER than q2k-experts.

PROMOTED follow-up (single kernel settles three questions):
  native-sparse-staged-q2k-v1: resident_iq2 x1, staged_iq2 x3 (control 2GB),
    staged_q2k x3 (THESIS), zc_iq2 x2 (v5-verbatim re-cert, 2GB),
    zc_q2k x2 (combo retest, 2GB hedge); experts transcode only; 90s
    post-transcode settle; minflt/majflt telemetry.
  Thesis: staged async FULLY hides I/O here (async_wait_ns=1, 0 wait) =>
    staged_q2k's +34% traffic bytes are free, kernel wins -40ms-class =>
    predicts +8-15% over staged control on fast-copy boxes. Box-lottery
    risk acknowledged (slow-copy box => <=0); zc arms hedge the other way.
  Quality gate STILL HELD: fires iff a bounded arm wins >10%.

Per-threshold verdicts: zc_iq2/zc_q2k/zc_q2kall KILL_HARD as composed;
staged_q2k is a NEW arm (not a refinement of a killed arm) justified by the
resident +31.8% isolation + async-overlap telemetry.
"""

COMBO_VERDICT = {
    "resident_iq2": {"tps": 4.955, "ms": 201.8},
    "staged_iq2": {"tps": 3.602, "ms": 277.7},
    "zc_iq2": {"tps": 2.917, "ms": 342.8, "delta": -0.190, "verdict": "KILL_HARD"},
    "zc_q2k_clean": {"tps": 3.12, "delta": -0.133, "verdict": "KILL_HARD"},
    "zc_q2kall_clean": {"tps": 2.97, "delta": -0.176, "verdict": "KILL_HARD"},
    "resident_q2k": {"tps": 6.529, "ms": 153.2, "delta_vs_resident": +0.318,
                     "verdict": "KERNEL_WIN_ISOLATED"},
    "mechanism": "zc re-faults every miss; staged amortizes; v5 win is I/O-box-specific",
    "own_goal": "combo zc used 1.6GB cache, v5 used 2GB (+22% misses)",
    "followup": "staged-q2k-v1: staged_q2k thesis + v5-verbatim recert + combo retest + minflt",
}
