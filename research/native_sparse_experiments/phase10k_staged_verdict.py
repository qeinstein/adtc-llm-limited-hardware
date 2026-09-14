"""Phase 10k: STAGED-Q2K verdict -- Q2K line KILLED on all vehicles.

staged-q2k-v1 COMPLETE (wall 54min, 32GB/slow-CPU box, 2GB verbatim):

  resident_iq2   4.110 tok/s   anchor
  staged_iq2     3.046 tok/s   control (reps flat: 3.036/3.064/3.037)
  staged_q2k     2.720 tok/s   -10.7%  KILL_HARD (reps 2.432/2.587/3.141 rising)
  zc_iq2         2.553 tok/s   -16.2%  KILL_HARD (v5 re-cert FAILED again)
  zc_q2k         2.584 tok/s   -15.2%  KILL_HARD (combo retest FAILED)

1. staged_q2k rising reps dissected: rchar/cache-stats/minflt IDENTICAL across
   reps; readMB 10781->9505->9439 and elapsed 51->38->33s => rep-1/2 still
   storm-hit (90s did NOT drain 12GB dirty + 32min transcode), rep-3 (+3.1%
   single rep) is storm-free but 12%-warmer (DONTNEED partially ineffective).
   True cold-clean estimate ~2.9-3.0 (-2 to -4%). The +3.1% is ONE warm rep,
   not a 3-10% measurement => KILL stands (robust across adjustments).
   Traffic pays TWICE: +17.75% bytes/slot AND +13.5% misses (1937 vs 2281
   slots) => +34% rchar >= kernel win under memcpy. J1 (-4.5% clean) +
   this (-10.7%) => staged vehicle DEAD on 2/2 boxes.

2. minflt/majflt PROVES the zc fault mechanism (and refines it):
     staged: 2.22M minor + 769 major | zc: 0.42M minor + 1.16M MAJOR (1500x!)
   Remap converts staged's sequential, readahead-friendly, async-hidden I/O
   into 1.16M SYNCHRONOUS RANDOM MAJOR FAULTS (~27K/s). Same ~6.1GB physical
   bytes, but zc pays per-page fault latency while staged pays batched async
   throughput. v5 now 1/3 boxes (re-cert failed at verbatim 2GB: cache
   own-goal EXCLUDED as the cause). zc vehicle DEAD (box-specific win).

3. Determinism notes: Q2K response hash 2d02d34a == combo (self-consistent);
   route overlap 0.6564827967171717 IDENTICAL to combo to 16 decimals (same
   diverged generation => same routes, cross-kernel deterministic). zc_iq2
   routes 1.0 exact (backend exactness holds for IQ2).

Q2K LINE VERDICT: KILL_HARD on all bounded vehicles. Kernel PROVED fast
(+31.8% resident, phase10i) but undeliverable under budget: staged copies
the +34% bytes through slow memcpy, zc major-faults them. Only resident
(12GB RSS) wins -- ineligible. Quality gate STANDS DOWN permanently for Q2K
(no winner; kernel file kept as reusable pattern).

Frontier state: NO bounded challenger has beaten staged control on >=2
boxes. Best exact <=4GiB remains staged control itself (3.0-3.6 tok/s,
box-dependent). Live branches: MTP-v2 (re-pushed, resilient download),
remainder-bypass / H/I sparsity (untouched), moonshot lab (peer lane).
"""

STAGED_Q2K_VERDICT = {
    "resident_iq2": {"tps": 4.110},
    "staged_iq2": {"tps": 3.046},
    "staged_q2k": {"tps": 2.720, "delta": -0.107, "verdict": "KILL_HARD",
                   "clean_estimate": "-2 to -4% (rep3 +3.1% is one warm rep)"},
    "zc_iq2": {"tps": 2.553, "delta": -0.162, "verdict": "KILL_HARD",
               "note": "v5-verbatim 2GB re-cert FAILED; v5 now 1/3 boxes"},
    "zc_q2k": {"tps": 2.584, "delta": -0.152, "verdict": "KILL_HARD"},
    "faults": {"staged_minflt": 2218993, "staged_majflt": 783,
               "zc_minflt": 418082, "zc_majflt": 1157849,
               "mechanism": "remap -> 1.16M synchronous random major faults"},
    "q2k_line": "KILL_HARD all bounded vehicles; quality gate stood down",
}
