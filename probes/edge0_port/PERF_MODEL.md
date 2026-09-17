# Corrected performance model (source of truth for all projections)

Date: 2026-09-16. Architecture: Qwen3.6 + real router + K4/16 + Q2_K
routed experts + static-hot/LRU cache + bounded reactive fetch + existing
CPU kernels. Rerunnable: `python3 probes/edge0_port/perf_model.py`.
Supersedes the K8-based Track-A floor (§A explains what was stale).

## 0. Equation

T_total = T_expert + T_nonexp + T_mgmt + T_fetch, where

- T_expert = 89.0 x 0.555 x 0.49 = **24.2 ms** (Kaggle-class)
- T_nonexp = 48 + 8 + 11 + 10 + 17 + 25.6 = **119.6 ms**
- T_mgmt = **5 ms** (adopted)
- T_fetch = fresh_MB/ssd_bw + miss x 0.25 (reactive, overlap=0)
- i5: cpu terms /1.4 (lo) or /1.7 (hi); fetch unscaled (disk-bound)

K4/16 normalization costs nothing measurable vs K4 (same router GEMV,
same 4 executed MLPs; only the renormalization denominator widens).

## 1. Component provenance

| term | ms | source | hw | K/fmt | measured? | conf |
|---|---|---|---|---|---|---|
| expert K8+IQ2 anchor 89.0 | 89.0 | phase5a-v2 routed-removed wall (88.961) | Kaggle 4-vCPU Xeon | K8/IQ2 | wall YES | HIGH |
| Q2_K ratio 0.555 | x0.555 | phase1 ubench k8_q2k ST (101.9/183.6) | N100, real ggml | K8 | YES | HIGH |
| K-scale 0.49 | x0.49 | phase1 ubench K pairs 0.484-0.491 | N100 | K4 | YES | HIGH |
| attn proj 48 | 48.0 | phase6b wall bypass 22.9% x 211.3 | Kaggle | - | wall YES | MED-HIGH |
| shared 8 | 8.0 | phase5a op-fraction 3.81% | Kaggle | - | fraction | MED |
| GDN 11 | 11.0 | phase5a op-fraction ~5.3% | Kaggle | - | fraction | LOW |
| scores 10 | 10.0 | phase10b gap (unattributed) | Kaggle | - | NO | LOW |
| gaps 17 | 17.0 | phase10b gaps (norms/router 12 + sched 5) | Kaggle | - | NO | LOW |
| LM head 25.6 | 25.6 | phase5a-v2 head-removed wall 12.11% | Kaggle | - | wall YES | HIGH |
| mgmt 5 | 5.0 | adopted (noDN cache mgmt) | - | - | NO | LOW |
| i5 /1.4 /1.7 | - | adopted conservative (clocks-only .. +IPC) | - | - | NO | MED |
| K8 hit curves | - | tracka_curves.json, test-split sims | - | K8 | sim | HIGH (for K8) |
| K4 miss x0.5 | - | INTERIM scaling (fresh K4 traces at JOIN) | - | K4 | NO | MED-LOW |
| ssd 1GB/s, t_miss 0.25 | - | adopted cold-SSD; 2.87 page-cache measured (phase9b local), 1.33 Kaggle warm | - | - | partial | MED |
| fixed RSS 2.22GB | - | dense 1.67 (GGUF inv, 3.5-file proxy!) + KV 0.25 + OVH 0.30 | - | - | bytes YES | MED |

Cross-validation: the 89-model predicts resident K4/K8 = 211.3/(122.3+89x0.49)
= **1.27x**, vs independently measured MMLU-wall **1.26x** (2112->1672s,
user-reported interim, kernel still RUNNING, pull pending). The old
110-model cannot reproduce this ratio. Linear K-scaling + ~89ms expert
share are therefore VALIDATED on full-model runs.

## 2A. Autopsy of the "139.5ms floor" (STALE, do not use)

139.49 = 195.285 / 1.4, with 195.285 = 48.285 [K8-Q2K core: CORE 87 x
FMT_RATIO 0.555] + 23.0 [EOVH_FIX 8 + EOVH_K8 15] + 124.0 [ATTN 48 +
SHARED 8 + GDN 11 + SCORES 10 + GAPS 17 + LMHEAD 25 + MGMT 5].

Two independent staleness bugs, both fixed in §0:

1. K8, not K4: the 48.3ms core is K8 expert compute. K4 halves it.
2. Phantom EOVH: 8+15 has NO source in committed evidence (searched
   research/ + probes/) and CONTRADICTS the direct wall measurement:
   phase5a-v2 routed-removed saves 88.961ms, i.e. the ENTIRE K8
   expert path is ~89ms, of which 10a attributes ~82 to row-dots +
   ~5 swiglu/accum/dispatch + 0.7 actquant. There is no room for a
   further 23ms. The old frame's N100 resident total (234ms) overshoots
   the measured 211.3 by exactly ~23ms. EOVH is dropped; the
   conservative case carries +10ms uncertainty pending any source.

It was also mislabeled "dense": 36% of it (71.3/195.3) was expert-path.
Conversely the challenger's "~105ms non-expert" undercounts: the measured
phase5a-v2 non-expert remainder is 122.3ms (attributed 119.6 + 2.7
unattributed). Corrected perfect-cache caps: **9.4 (lo) / 11.4 (hi)**
tok/s, not ~7.

## 2B. Reconciliation of the five statements

- Phase-1 7.7/8.9 @5.7GB: assumed K4+Q2K + **dense-Q2K 0.60x (attn/shared,
  UNGATED, excluded from credible per decision log)** + 4GB LRU cache +
  EOVH-frame + i5lo/hi. Restated on current rules (dense as-is, no EOVH,
  winning policies): ~7.5/~8.7 @6GB. The delta is assumptions, not error.
- "~7 perfect-cache cap": K8 floor + phantom EOVH (§A). Withdrawn.
- MMLU 1.26x: validates the 89-model (predicts 1.27x). Strongest
  independent check we have.
- 39.6ms "expert path": model projection 24.1 + 15.5(EOVH, phantom).
  Corrected: **24.2ms** (89 x 0.555 x 0.49, all three inputs measured).

## 2C. RSS Pareto, current K4/16 architecture (INTERIM: K8 traces x0.5)

From `perf_model.py` (slots = (T-2.22)GB/1032192B; policy pinL4@3,
hyb25@4-5, hyb50@5.5-6.5):

| GB | case | slots | hit% | miss/tok | MB/tok | fetch | expC | nonExp | total | t/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 3.0 | cons/lo/hi/up | 755 | 37/48/48/53 | 100/84/84/75 | 104/86/86/78 | 129/107/107/46 | 24/17/14 | 89/73 | 242/214/195/133 | 4.1/4.7/5.1/7.5 |
| 4.0 | cons/lo/hi/up | 1724 | 62/68/68/71 | 61/51/51/46 | 63/53/53/48 | 79/66/66/28 | 24/17/14 | 89/73 | 192/172/153/116 | 5.2/5.8/6.5/8.7 |
| 5.0 | cons/lo/hi/up | 2693 | 75/79/79/81 | 40/34/34/30 | 42/35/35/31 | 52/43/43/19 | 24/17/14 | 89/73 | 165/149/131/106 | 6.1/6.7/7.7/9.4 |
| 5.5 | cons/lo/hi/up | 3177 | 80/83/83/85 | 33/27/27/24 | 34/28/28/25 | 42/35/35/15 | 24/17/14 | 89/73 | 155/141/122/102 | 6.5/7.1/8.2/9.8 |
| 6.0 | cons/lo/hi/up | 3662 | 84/87/87/88 | 26/22/22/20 | 27/22/22/20 | 33/28/28/12 | 24/17/14 | 89/73 | 147/134/115/99 | 6.8/7.5/8.7/10.1 |
| 6.5 | cons/lo/hi/up | 4146 | 87/89/89/90 | 21/17/17/16 | 22/18/18/16 | 27/22/22/10 | 24/17/14 | 89/73 | 140/129/110/97 | 7.1/7.8/9.1/10.3 |

Cases: cons = miss x0.60, i5lo, cold disk, +10ms EOVH-uncertainty;
credible lo/hi = miss x0.50, i5lo/hi, cold disk; upside = miss x0.45,
i5hi, page-cache disk 2.87GB/s. Measured: expert 24.2 (anchored),
attn/head (walls), K8 curves. Projected: K4 locality, i5 factors,
disk behavior, gaps/GDN/mgmt. Replace K4 column at JOIN with fresh
K4/16 trace re-sweep. 3.5-file dense proxy must be verified vs 3.6.

## 2D. Cases and verdicts

- CONSERVATIVE (only strongly measured): 4.1 @3GB .. 7.1 @6.5GB.
- CREDIBLE (measured + supported projections): 4.7-5.1 @3GB ..
  6.7-7.7 @5GB .. 7.5-8.7 @6GB .. 7.8-9.1 @6.5GB.
- UPSIDE (profiler-driven, no dead ideas): 7.5 @3GB .. 9.4 @5GB ..
  10.1 @6GB .. 10.3 @6.5GB (needs max RAM + page-cache disk +
  hi clocks + good K4 locality, all at once).

Verdicts:
- **10 tok/s**: upside-only at >=6GB (10.1/10.3); <5GB upside = 9.4,
  short. NOT credible. Marginal, everything-must-land.
- **12 tok/s** (83.3ms): upside@6.5 = 97ms; needs unproven LM-shortlist
  (-13) AND gap shrinkage on top. NOT supported.
- **15 / 18 tok/s**: no path in any case. Needs ~2x dense GEMV (dense
  requant ungated/risky, fusion unproven) or MTP-class gains (KILLED).
  NOT supported. Note S_perf (fixed profiler ref) saturates at 15
  tok/s anyway: 18 buys zero score unless scoring is relative.

Score tension (fixed ref: 1GB must buy >=1.43 t/s): our marginals are
~0.6-0.9 t/s/GB, so under fixed-ref scoring FEWER GB wins (3GB: 20.8
pts vs 6.5GB: 17.0 on perf+eff). Speed-first goal says 6.5GB;
fixed-score says ~3-4GB. Stated sprint goal is speed-first; revisit
if scoring interpretation changes.

## 2E. Remaining floor (5GB credible-lo = 149.4ms)

| component | Kaggle-ms | i5lo-ms | reducibility this sprint |
|---|---|---|---|
| fetch (reactive) | 43.1 | 43.1 | HIGH: RAM 5->6.5GB halves it; page-cache disk /2.87. Planned mechanisms. |
| attention proj | 48.0 | 34.3 | LOW: dense requant ungated; fusion unproven. Confirm wall on profiler first. |
| gaps (scores 10 + rest 17) | 27.0 | 19.3 | UNKNOWN: unattributed; profiler resolves; some may evaporate on i5. |
| LM head | 25.6 | 18.3 | MED-LOW: shortlist unproven (same-arch, allowed as upside). -13ms if lands. |
| expert compute (SOLVED) | 24.2 | 17.3 | LOW: K4+Q2K done; further = fusion. Only 11% of total. |
| GDN | 11.0 | 7.9 | LOW (op-frac; may be smaller in wall). |
| shared expert | 8.0 | 5.7 | LOW. |
| mgmt | 5.0 | 3.6 | LOW (adopted; immaterial). |

At 5GB: fetch+attn = 77/149 (52%). At 6.5GB fetch drops to 22 and
attn (34) dominates. JOIN step 6 order: (1) spend RAM to 6-6.5GB +
verify page-cache disk behavior (fetch -20ms, +~1.1 t/s, no new
science); (2) profile integrated run, resolve gaps, confirm attn/head
walls; (3) attack largest CONFIRMED component (likely attn or
fetch-residual). Do not optimize the expert path (solved) or resurrect
prerouting/dense-Q2K/MTP/zero-copy.

## 2F. Locked path + pending

Qwen3.6 + real router + K4/16 + Q2_K experts + static-hot/LRU + reactive
fetch + existing kernels. Thinking stays (UI surfaces reasoning; NOT a
blocker; post-training may shorten). KEEP/KILL per decision log (2B:
dense-Q2K stays out of credible until/unless gated).

Pending: explicit K8 patched-control; rest of 3.5 sanity/layer/speed
checks; user-reported K4/16 MMLU (41.0, 1672s) VERIFICATION via output
pull (kernel RUNNING at audit time); fresh K4/16 traces; K4 cache
re-sweep (replaces §2C interim); 3.6 dense-inventory check; integrated
profiler; final tok/s.
