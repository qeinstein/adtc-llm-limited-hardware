#!/usr/bin/env python3
"""Export deployment cache config: winning policy pins per RSS target.

Pins derived from trace-v3 TRAIN (p00-p11, eval0 dropped) = deployment base
3.6, same methodology as tracka_pareto.py. JOIN re-derives on K4 traces.
Also stores the python-measured hit% at EXACT slot counts so the C replay
(cache_replay.c) can cross-check bit-for-bit policy equivalence.
Writes probes/edge0_port/cache_config.json + /tmp stream .bin for C.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "probes/edge0_port")
import tracka_pareto as T

EDGE0 = Path("probes/edge0_port")
TARGETS = {3.0: ("pinL4", None), 4.0: ("hyb25", 0.25), 5.0: ("hyb25", 0.25),
           5.5: ("hyb50", 0.50), 6.0: ("hyb50", 0.50)}


def main():
    pareto = {r["target"]: r for r in
              json.load(open(EDGE0 / "tracka_pareto.json"))}
    tr, te = T.load_trv3()
    pop = Counter(k for t in tr for r in t for k in r)
    pop_layer = {}
    for t in tr:
        for li, reqs in enumerate(t):
            pop_layer.setdefault(li, Counter()).update(reqs)
    seq = T.flatten(te)
    ntok = len(te)
    cfg = {"bundle": "q2k", "bundle_bytes": T.BUNDLE, "experts": 10240,
           "source": "tracev3-train-p00-p11-eval0dropped",
           "note": "JOIN re-derives pins on fresh K4 traces",
           "targets": {}}
    for target, (pol, frac) in TARGETS.items():
        slots = pareto[target]["slots"]
        if pol == "pinL4":
            pins = sorted(kk for li in range(40)
                          for kk, _ in pop_layer[li].most_common(4))
            per = {str(li): [kk for kk, _ in pop_layer[li].most_common(4)]
                   for li in range(40)}
        else:
            pins = sorted(k for k, _ in pop.most_common(int(slots * frac)))
            per = {}
        h, m = T.sim_hybrid(seq, ntok, slots, set(pins))
        cfg["targets"][str(target)] = {
            "slots": slots, "policy": pol, "pin_frac": frac,
            "npins": len(pins), "dyn_slots": slots - len(pins),
            "expected_hit": round(h, 6), "expected_miss_per_tok": round(m, 3),
            "pins_global": pins, "pins_per_layer": per}
        print(f"{target}GB {pol} slots={slots} pins={len(pins)} "
              f"hit={h:.4f} miss/t={m:.1f}", flush=True)
    json.dump(cfg, open(EDGE0 / "cache_config.json", "w"))
    import numpy as np
    np.array(seq, dtype=np.uint16).tofile("/tmp/edge0_phase1/trv3_test_stream.u16")
    print("wrote cache_config.json + trv3_test_stream.u16 "
          f"({len(seq)} reqs)")


if __name__ == "__main__":
    main()
