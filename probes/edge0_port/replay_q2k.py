#!/usr/bin/env python3
"""JOIN-3c: replay the IQ2-derived winning cache policy on Q2K routes.

Q2K transfer validation: does the 6.0GB winner (hyb25 + pins from
cache_config_k4.json) hold its hit rate on TRUE Q2K trajectories, or do
the IQ2-derived pins mis-transfer (decode-trajectory drift)? Pure LRU at
the same cap is the pin-free reference: if pins hurt on Q2K, lock LRU.

Usage: replay_q2k.py <Q2K_IDS_DIR>
  (extracted via: extract_k4.py traces/tracek4q2k traces/tracek4q2kroutes)

Reports, at the 6.0GB RAM point (3662 slots):
  - winner + LRU miss/tok on Q2K vs IQ2-test (same code path)
  - PAIRED same-pid IQ2 vs Q2K (isolates trajectory drift from prompt mix)
  - hot-set Jaccard (top-512 freq sets: IQ2-test vs Q2K, + paired)
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_atomic as A

EDGE0 = Path("probes/edge0_port")
IQ2_IDS = Path("traces/tracek4routes")
RAM_POINT = "6.0"


def load_ids(d):
    toks = []
    for f in sorted(Path(d).glob("ids_p*.npy")):
        I = np.load(f)
        assert I.ndim == 3 and I.shape[1] == 40 and I.shape[2] == 4, f
        toks.extend([[[(li << 8) | int(e) for e in I[t, li, :]]
                      for li in range(40)] for t in range(I.shape[0])])
    assert toks, d
    return toks


def hotset(toks, n=512):
    c = Counter(k for t in toks for lay in t for k in lay)
    return set(k for k, _ in c.most_common(n))


def main():
    qdir = sys.argv[1] if len(sys.argv) > 1 else \
        (_ for _ in ()).throw(SystemExit("usage: replay_q2k.py <Q2K_IDS_DIR>"))
    cfg = json.loads((EDGE0 / "cache_config_k4.json").read_text())[RAM_POINT]
    cap, pins = cfg["slots"], set(cfg["pins_global"])
    print(f"RAM point {RAM_POINT}GB: policy={cfg['policy']} slots={cap} "
          f"npins={len(pins)} iq2-expected-hit={cfg['expected_hit']}")
    q2k = load_ids(qdir)
    print(f"Q2K routes: {len(q2k)} toks from {qdir}")
    qpids = sorted(int(re.search(r"p(\d+)", f.name).group(1))
                   for f in Path(qdir).glob("ids_p*.npy"))
    paired = []
    for pid in qpids:
        I = np.load(IQ2_IDS / f"ids_p{pid:02d}.npy")
        paired.extend([[[(li << 8) | int(e) for e in I[t, li, :]]
                        for li in range(40)] for t in range(I.shape[0])])
    print(f"IQ2 paired routes (same {len(qpids)} pids): {len(paired)} toks")
    iq2te = []
    for f in sorted(IQ2_IDS.glob("ids_p*.npy")):
        if int(re.search(r"p(\d+)", f.name).group(1)) >= 12:
            I = np.load(f)
            iq2te.extend([[[(li << 8) | int(e) for e in I[t, li, :]]
                           for li in range(40)] for t in range(I.shape[0])])
    print(f"IQ2 test routes: {len(iq2te)} toks")
    for name, toks in (("IQ2-test", iq2te), ("IQ2-paired", paired),
                         ("Q2K", q2k)):
        te = [[r for r in t] for t in toks]
        h_w, m_w = A.sim_atomic(te, cap, pins)[:2]
        te = [[r for r in t] for t in toks]
        h_l, m_l = A.sim_atomic(te, cap)[:2]
        print(f"  {name:8s} winner({cfg['policy']},{len(pins)}pins): "
              f"hit={h_w:.4f} miss/tok={m_w:.1f} | "
              f"lru: hit={h_l:.4f} miss/tok={m_l:.1f}")
    a, b = hotset(iq2te), hotset(q2k)
    print(f"hot-set Jaccard(top512 IQ2-test vs Q2K): "
          f"{len(a & b) / len(a | b):.4f} "
          f"(overlap {len(a & b)}/512)")
    ap, bp = hotset(paired), hotset(q2k)
    print(f"hot-set Jaccard(top512 IQ2-paired vs Q2K): "
          f"{len(ap & bp) / len(ap | bp):.4f} "
          f"(overlap {len(ap & bp)}/512)")
    def tophot(I, n=256):
        c = Counter()
        for t in range(I.shape[0]):
            for li in range(40):
                for e in I[t, li, :]:
                    c[(li << 8) | int(e)] += 1
        return set(k for k, _ in c.most_common(n))

    per = []
    for pid in qpids:
        A1 = np.load(IQ2_IDS / f"ids_p{pid:02d}.npy")
        A2 = np.load(Path(qdir) / f"ids_p{pid:02d}.npy")
        s1, s2 = tophot(A1), tophot(A2)
        per.append(len(s1 & s2) / len(s1 | s2))
    print(f"mean per-pid Jaccard(top256, {len(qpids)} paired pids): "
          f"{sum(per) / len(per):.4f}")


if __name__ == "__main__":
    main()
