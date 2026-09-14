"""Phase 10l: moonshot-C MTP probe v2 verdict -- KILL_HARD (both N).

mtp-probe-v2 COMPLETE (wall 32min, v1 method fixes held):

  base    4.5498 tok/s   reps [4.616, 4.588, 4.627, 4.368], CV 2.3% (gate SOUND)
  mtp_n1  4.4322 tok/s   reps [4.491, 4.454, 4.331, 4.454]   -2.58%  KILL_HARD
  mtp_n2  4.1572 tok/s   reps [4.161, 4.154]                 -8.63%  KILL_HARD

Pairwise (interleaved): n1 lost 3/4 pairs; its only win was pair 4 where the
BASE rep itself dipped (prompt 6.88 vs ~7.4: mild contention). All 11 arms
byte-identical responses (exactness holds; no quality question ever).

Conclusion: single-head MTP draft+verify overhead exceeds acceptance benefit
on CPU decode; N=2 strictly worse (-8.6%). Moonshot C KILLED as a CPU
throughput path. No further MTP work (verbose acceptance telemetry not
retrieved: verdict decisive without it).
"""

MTP_V2 = {
    "verdict": "KILL_HARD",
    "base_tps": 4.5498, "base_cv": 0.0232, "method_sound": True,
    "mtp_n1_tps": 4.4322, "mtp_n1_delta": -0.0258,
    "mtp_n2_tps": 4.1572, "mtp_n2_delta": -0.0863,
    "exactness": "all arms byte-identical",
}
