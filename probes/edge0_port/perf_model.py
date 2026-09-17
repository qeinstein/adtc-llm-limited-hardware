#!/usr/bin/env python3
"""Corrected performance model for the final sprint architecture.

Qwen3.6 + real router + K4/16 + Q2_K routed experts + static-hot/LRU cache
+ bounded reactive fetch + existing CPU kernels.

SOURCE OF TRUTH for all tok/s projections. Replaces the stale K8-based
Track-A floor (139.5ms, which bundled K8 expert compute + unprovenanced
expert overhead). See probes/edge0_port/PERF_MODEL.md for the full audit.

Equation (all terms ms/token; Kaggle-class unless noted):
  T_total = T_expert(K4,Q2K) + T_nonexp + T_mgmt + T_fetch
  T_expert  = 89.0 * 0.555 * 0.49          (wall-anchored, ratios measured)
  T_nonexp  = 48 + 8 + 11 + 10 + 17 + 25.6 (phase5a-v2/6b/10b walls+fractions)
  T_fetch   = fresh_MB/ssd_bw + miss*0.25  (reactive, overlap=0)
  i5 scaling: cpu terms /1.4 (lo) or /1.7 (hi); fetch unscaled (disk-bound)

K4 miss interim: K8 test-split miss x K4_MISS_SCALE (fresh K4 traces at JOIN
replace this). No dead ideas: no prerouting overlap, no dense-Q2K, no MTP.
"""
import json
from pathlib import Path

EDGE0 = Path("probes/edge0_port")

# ---- wall-anchored constants (provenance in PERF_MODEL.md) ----
EXPERT_K8_IQ2 = 89.0      # phase5a-v2: routed-removed wall saving, Kaggle
FMT_Q2K = 0.555           # phase1 ubench k8_q2k ST ratio (N100, real kernels)
K_SCALE = 0.49            # phase1 ubench K-scaling 0.484-0.491 (N100)
ATTN = 48.0               # phase6b wall bypass (22.9% of 211.3)
SHARED = 8.0              # phase5a op-fraction 3.81%
GDN = 11.0                # phase5a op-fraction ~5.3% (LOW conf, needs wall)
SCORES = 10.0             # phase10b gap, unattributed (LOW conf)
GAPS = 17.0               # phase10b gaps norms/router 12 + sampler 5 (LOW)
LMHEAD = 25.6             # phase5a-v2 wall bypass 12.11%
MGMT = 5.0                # adopted cache-mgmt (LOW conf, ~3 i5-ms)
I5_LO, I5_HI = 1.4, 1.7   # conservative i5 factors (clocks-only .. +IPC)
BUNDLE = 1032192          # Q2_K expert bundle bytes (exact)
FIXED_GB = 2.22           # dense 1.67 + KV 0.25 + OVH 0.30
T_MISS = 0.25             # adopted per-miss ms (LOW conf)
POLICY = {3.0: "pinL4", 4.0: "hyb25", 5.0: "hyb25", 5.5: "hyb50",
          6.0: "hyb50", 6.5: "hyb50"}

CASES = {
    # name: (k4_miss_scale, i5_div, ssd_bw, eovh_uncertainty_kagms)
    "conservative": (0.60, I5_LO, 1.0, 10.0),
    "credible-lo": (0.50, I5_LO, 1.0, 0.0),
    "credible-hi": (0.50, I5_HI, 1.0, 0.0),
    "upside": (0.45, I5_HI, 2.87, 0.0),
}

TARGETS = [3.0, 4.0, 5.0, 5.5, 6.0, 6.5]


def k8_miss_at(slots, policy):
    curves = json.load(open(EDGE0 / "tracka_curves.json"))
    ms = []
    for cname in curves:
        cs = sorted(int(c) for c in curves[cname])
        get = lambda c: curves[cname][str(c)][policy][1]
        if slots <= cs[0]:
            ms.append(get(cs[0]))
            continue
        for a, b in zip(cs, cs[1:]):
            if slots <= b:
                ma, mb = get(a), get(b)
                ms.append(mb + (slots - b) / (b - a) * (mb - ma))
                break
        else:
            ma, mb = get(cs[-2]), get(cs[-1])
            sl = (mb - ma) / (cs[-1] - cs[-2])
            ms.append(max(4.0, mb + sl * (slots - cs[-1])))
    return sum(ms) / len(ms)


def main():
    expert = EXPERT_K8_IQ2 * FMT_Q2K * K_SCALE
    nonexp = ATTN + SHARED + GDN + SCORES + GAPS + LMHEAD
    print(f"expert(K4,Q2K)={expert:.1f} nonexpert={nonexp:.1f} "
          f"(attn {ATTN} shared {SHARED} gdn {GDN} scores {SCORES} "
          f"gaps {GAPS} head {LMHEAD}) mgmt={MGMT}")
    print("TargGB case         slots expGB hit% missK4  B/tok fetch "
          "expC nonExp  total tok/s")
    for T in TARGETS:
        exp_gb = T - FIXED_GB
        slots = int(exp_gb * 1e9 / BUNDLE)
        mk8 = k8_miss_at(slots, POLICY[T])
        for case, (ks, div, bw, eu) in CASES.items():
            miss = mk8 * ks
            fresh = miss * BUNDLE / 1e6
            fetch = fresh / bw + miss * T_MISS
            expc = (expert + eu) / div
            ne = (nonexp + MGMT) / div
            tot = expc + ne + fetch
            hit = 1 - miss / 160.0
            print(f"{T:5.1f} {case:13s} {slots:5d} {exp_gb:5.2f} "
                  f"{100*hit:5.1f} {miss:6.1f} {fresh:6.1f}M {fetch:5.1f} "
                  f"{expc:4.1f} {ne:5.1f} {tot:6.1f} {1000/tot:5.2f}")


if __name__ == "__main__":
    main()
