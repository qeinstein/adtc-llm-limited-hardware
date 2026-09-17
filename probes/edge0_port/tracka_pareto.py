#!/usr/bin/env python3
"""Track A: RAM/cache Pareto on REAL traces (deployment-honest test split).

Policies: pure LRU, static top-freq pins, static-hot+LRU hybrid (pin frac
sweep), online LFU, per-layer pins + LRU, Belady (upper bound only).
All policies evaluated on the TEST stream from empty; pins from TRAIN.
Corpora: (1) phase5e route corpus (3.5, 2016 toks, half/half split);
(2) trace-v3 3.6 routes (23 prompts, 12/11 prompt split, eval0 dropped).

Latency/scoring frame adopted from phase1_pareto.py (same constants):
Q2_K experts, K8, dense as-is (NO dense-Q2K: killed), overlap=0
(reactive fetch; 96% was the oracle bound, dead with prerouting NO-GO),
ssd=1GB/s, t_miss=0.25ms, i5 factors 1.4/1.7.
Score: S=0.3*S_perf+0.2*S_eff (S_acc held fixed) => 1 GB must buy >=1.43
tok/s to be score-neutral under fixed S_perf (15 tps ref).

Writes: stdout tables + tracka_curves.json + tracka_pareto.json
"""
import heapq
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

sys.path.insert(0, "probes/edge0_port")
import phase1_pareto as P

EDGE0 = Path("probes/edge0_port")
TRV3 = Path("/tmp/edge0_phase1/traces/tracev3routes")

CAPS = [320, 512, 826, 1024, 1439, 1792, 2300, 2664, 3200, 3584,
        4096, 4608, 5120, 5632, 6656]
TARGETS = [3.0, 4.0, 5.0, 5.5, 6.0]
BUNDLE = P.BUNDLE["q2k"]  # 1032192 exact


def load_phase5e():
    toks = P.load_corpus()
    n = len(toks)
    return toks[:n // 2], toks[n // 2:]


def load_trv3():
    import numpy as np
    import re
    tr, te = [], []
    for f in sorted(TRV3.glob("ids_p*.npy")):
        pid = int(re.search(r"p(\d+)", f.name).group(1))
        I = np.load(f)[1:]  # drop eval0 (cross-prompt duplicate quirk)
        per_tok = []
        for t in range(I.shape[0]):
            per_tok.append([[(li << 8) | int(e) for e in I[t, li, :]]
                            for li in range(40)])
        (tr if pid < 12 else te).extend(per_tok)
    return tr, te


def flatten(toks):
    return [k for layers in toks for reqs in layers for k in reqs]


def sim_lru(seq, ntok, cap):
    c = OrderedDict()
    h = 0
    for k in seq:
        if k in c:
            h += 1
            c.move_to_end(k)
        else:
            c[k] = None
            if len(c) > cap:
                c.popitem(last=False)
    return h / len(seq), (len(seq) - h) / ntok


def sim_static(seq, ntok, pins):
    h = sum(1 for k in seq if k in pins)
    return h / len(seq), (len(seq) - h) / ntok


def sim_hybrid(seq, ntok, cap, pins):
    dyn = max(0, cap - len(pins))
    c = OrderedDict()
    h = 0
    for k in seq:
        if k in pins or k in c:
            h += 1
            if k in c:
                c.move_to_end(k)
        else:
            if dyn > 0:
                c[k] = None
                if len(c) > dyn:
                    c.popitem(last=False)
    return h / len(seq), (len(seq) - h) / ntok


def sim_lfu(seq, ntok, cap):
    freq, use, cache, heap = {}, {}, set(), []
    tick, h = 0, 0
    for k in seq:
        tick += 1
        if k in cache:
            h += 1
            freq[k] += 1
            use[k] = tick
            heapq.heappush(heap, (freq[k], tick, k))
        else:
            if len(cache) >= cap:
                while True:
                    f, u, ev = heapq.heappop(heap)
                    if ev in cache and freq[ev] == f and use[ev] == u:
                        cache.discard(ev)
                        del freq[ev]
                        del use[ev]
                        break
            cache.add(k)
            freq[k] = 1
            use[k] = tick
            heapq.heappush(heap, (1, tick, k))
    return h / len(seq), (len(seq) - h) / ntok


def per_layer_stats(train):
    pop = {}
    for t in train:
        for li, reqs in enumerate(t):
            pop.setdefault(li, Counter()).update(reqs)
    for li in sorted(pop)[:3]:
        c = pop[li]
        tot = sum(c.values())
        top8 = sum(v for _, v in c.most_common(8)) / tot
        print(f"  layer {li}: top8-mass={top8:.3f} nunique={len(c)}")
    return pop


def main():
    sys.path.insert(0, ".")
    from src.score import s_eff, s_perf_fixed
    print("== dense inventory (3.5-file proxy for 3.6; same dense arch) ==")
    groups = P.dense_inventory()
    tot = sum(groups.values())
    dense_gb = (tot - groups["routed-experts"]) / 1e9
    print(f"  dense as-is {dense_gb:.3f} GB (total {tot/1e9:.3f})")
    fixed = dense_gb + P.KV_GB + P.OVH_GB
    print(f"  fixed non-expert RSS {fixed:.3f} GB")

    corpora = {}
    print("== per-layer concentration (train) ==")
    tr5, te5 = load_phase5e()
    print(f"phase5e: train {len(tr5)} test {len(te5)} toks")
    pop5 = per_layer_stats(tr5)
    corpora["phase5e"] = (tr5, te5, pop5)
    tr3, te3 = load_trv3()
    print(f"tracev3-3.6: train {len(tr3)} test {len(te3)} toks")
    pop3 = per_layer_stats(tr3)
    corpora["tracev3"] = (tr3, te3, pop3)

    curves = {}
    for cname, (train, test, pop) in corpora.items():
        seq = flatten(test)
        ntok = len(test)
        print(f"== {cname}: test {len(seq)} reqs ==")
        pop_all = Counter(k for t in train for r in t for k in r)
        for cap in CAPS:
            row = {}
            h, m = sim_lru(seq, ntok, cap)
            row["lru"] = [h, m]
            top = set(k for k, _ in pop_all.most_common(cap))
            h, m = sim_static(seq, ntok, top)
            row["static"] = [h, m]
            for frac in (0.25, 0.5, 0.75):
                pn = int(cap * frac)
                pins = set(k for k, _ in pop_all.most_common(pn))
                h, m = sim_hybrid(seq, ntok, cap, pins)
                row[f"hyb{int(frac*100)}"] = [h, m]
            h, m = sim_lfu(seq, ntok, cap)
            row["lfu"] = [h, m]
            for k in (2, 4):
                pl = set(kk for li in range(40)
                         for kk, _ in pop[li].most_common(k))
                h, m = sim_hybrid(seq, ntok, cap, pl)
                row[f"pinL{k}"] = [h, m]
            h, m = P.belady_sim(test, cap)
            row["belady"] = [h, m]
            curves.setdefault(cname, {})[cap] = row
            print(f"  cap {cap:5d}: " + " ".join(
                f"{k}={v[0]:.3f}" for k, v in row.items()), flush=True)
    json.dump(curves, open(EDGE0 / "tracka_curves.json", "w"))

    def cpu_ms():
        core = P.CORE * P.FMT_RATIO["q2k"]
        eovh = P.EOVH_FIX + P.EOVH_K8
        return core + eovh + P.ATTN + P.SHARED + P.GDN + P.SCORES + \
            P.GAPS + P.LMHEAD + P.MGMT

    cpu = cpu_ms()
    print(f"\n== CPU bucket (q2k K8, dense as-is): {cpu:.1f} ms ==")
    print("TargGB policy    slots expGB  hit% miss/t  B/tok fetch tpsLo "
          "tpsHi dT/dG dS/dG sEff")
    pareto, prev = [], None
    for T in TARGETS:
        exp_gb = T - fixed
        slots = int(exp_gb * 1e9 / BUNDLE)
        best = None
        for pol in ("lru", "static", "hyb25", "hyb50", "hyb75", "lfu",
                    "pinL2", "pinL4"):
            ms = []
            for cname in corpora:
                cs = sorted(curves[cname])
                if slots <= cs[0]:
                    ms.append(curves[cname][cs[0]][pol][1])
                    continue
                for a, b in zip(cs, cs[1:]):
                    if slots <= b:
                        ma = curves[cname][a][pol][1]
                        mb = curves[cname][b][pol][1]
                        ms.append(mb + (slots - b) / (b - a) * (mb - ma))
                        break
                else:
                    ms.append(curves[cname][cs[-1]][pol][1])
            miss = sum(ms) / len(ms)
            fresh = miss * BUNDLE / 1e6
            fetch = fresh + miss * 0.25
            lo, hi = 1000 / (cpu / 1.4 + fetch), 1000 / (cpu / 1.7 + fetch)
            if best is None or lo > best[0]:
                best = (lo, hi, pol, miss, fresh, fetch)
        lo, hi, pol, miss, fresh, fetch = best
        hit = 1 - miss / 320.0
        seff = s_eff(T)
        if prev is None:
            dtg, dsg = 0.0, 0.0
        else:
            pt, plo, pse = prev
            dtg = (lo - plo) / (T - pt)
            ds = (0.3 * (s_perf_fixed(lo) - s_perf_fixed(plo)) +
                  0.2 * (seff - pse)) / (T - pt)
            dsg = ds
        prev = (T, lo, seff)
        pareto.append({"target": T, "policy": pol, "slots": slots,
                       "exp_gb": round(exp_gb, 3), "hit": round(hit, 4),
                       "miss": round(miss, 1),
                       "bytes_per_tok": int(miss * BUNDLE),
                       "fetch_ms": round(fetch, 1),
                       "total_ms_lo": round(cpu / 1.4 + fetch, 1),
                       "tps_lo": round(lo, 2), "tps_hi": round(hi, 2),
                       "marginal_tps_per_gb": round(dtg, 2),
                       "marginal_score_per_gb": round(dsg, 2),
                       "s_eff": round(seff, 1)})
        print(f"{T:5.1f} {pol:8s} {slots:5d} {exp_gb:5.2f} {100*hit:5.1f} "
              f"{miss:6.1f} {miss*BUNDLE/1e6:6.1f}M {fetch:5.1f} "
              f"{lo:5.2f} {hi:5.2f} {dtg:+5.2f} {dsg:+5.2f} {seff:4.1f}")
    json.dump(pareto, open(EDGE0 / "tracka_pareto.json", "w"), indent=1)
    print("wrote tracka_curves.json + tracka_pareto.json")


if __name__ == "__main__":
    main()