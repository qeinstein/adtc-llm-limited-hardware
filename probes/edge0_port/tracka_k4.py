#!/usr/bin/env python3
"""JOIN-3: cache/RSS Pareto re-sweep on FRESH K4/16 traces (atomic events).

Input: traces/tracek4routes/ids_p<NN>.npy  [T,40,4] int (extractor:
route TRACEs from trace_p<NN>.npz topk_ids; see extract_k4.py).
Split mirrors v3: pid<12 train (pins), else test (all sims on test).

Policies (ALL atomic-event): lru, static, hyb25/50/75, lfu,
least-stale (decayed frequency), pinL2/pinL4, belady (protected oracle).
Latency frame: CORRECTED model (perf_model), NOT phase1_pareto's stale
K8/EOVH constants. K4 misses are MEASURED here (no x0.5 interim).

Writes: tracka_k4_curves.json, tracka_k4_pareto.json, cache_config_k4.json
(winning pins per RSS target for integration).
"""
import heapq
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

sys.path.insert(0, "probes/edge0_port")
import cache_atomic as A
import perf_model as M

EDGE0 = Path("probes/edge0_port")
TRK4 = Path("traces/tracek4routes")
CAPS = [320, 512, 826, 1024, 1439, 1792, 2300, 2664, 3200, 3584,
        4096, 4608, 5120, 5632, 6656]
TARGETS = [3.0, 4.0, 5.0, 5.5, 6.0]
BUNDLE = M.BUNDLE
HALVE_EVERY = 20000  # least-stale decay interval (requests)


def load_split():
    import numpy as np
    tr, te = [], []
    for f in sorted(TRK4.glob("ids_p*.npy")):
        pid = int(re.search(r"p(\d+)", f.name).group(1))
        I = np.load(f)
        assert I.ndim == 3 and I.shape[1] == 40 and I.shape[2] == 4, f.shape
        toks = [[[(li << 8) | int(e) for e in I[t, li, :]]
                 for li in range(40)] for t in range(I.shape[0])]
        (tr if pid < 12 else te).extend(toks)
    return tr, te


def sim_lfu_atomic(toks, cap):
    freq, use, cache, heap = {}, {}, set(), []
    tick, h, tot = 0, 0, 0
    for layers in toks:
        for reqs in layers:
            tot += len(reqs)
            pre = [k in cache for k in reqs]
            h += sum(pre)
            for k in reqs:
                tick += 1
                if k in cache:
                    freq[k] += 1
                    use[k] = tick
                    heapq.heappush(heap, (freq[k], tick, k))
            for k in reqs:  # admit misses (protect: never evict event keys)
                if k in cache:
                    continue
                evset = set(reqs)
                skipped = []
                while len(cache) >= cap and heap:
                    f, u, evk = heapq.heappop(heap)
                    if not (evk in cache and freq[evk] == f
                            and use[evk] == u):
                        continue
                    if evk in evset:  # protected: set aside, try next
                        skipped.append((f, u, evk))
                        continue
                    cache.discard(evk)
                    del freq[evk]
                    del use[evk]
                    break
                for e in skipped:  # restore protected entries
                    heapq.heappush(heap, e)
                if len(cache) >= cap:  # degenerate: all cached protected
                    evk = min(cache, key=lambda kk: (freq[kk], use[kk]))
                    cache.discard(evk)
                    del freq[evk]
                    del use[evk]
                cache.add(k)
                freq[k] = 1
                use[k] = tick
                heapq.heappush(heap, (1, tick, k))
    return h / tot, (tot - h) / len(toks)


def sim_stale_atomic(toks, cap):
    """Least-stale: decayed-frequency eviction (halve counts periodically)."""
    cnt, use, cache = {}, {}, set()
    tick, nreq, h, tot = 0, 0, 0, 0
    for layers in toks:
        for reqs in layers:
            tot += len(reqs)
            pre = [k in cache for k in reqs]
            h += sum(pre)
            for k in reqs:
                tick += 1
                nreq += 1
                if nreq % HALVE_EVERY == 0:
                    for kk in cnt:
                        cnt[kk] *= 0.5
                if k in cache:
                    cnt[k] = cnt.get(k, 0) + 1
                    use[k] = tick
            for k in reqs:
                if k in cache:
                    continue
                evset = set(reqs)
                if len(cache) >= cap:
                    cands = [kk for kk in cache if kk not in evset] or \
                        list(cache)
                    evk = min(cands, key=lambda kk: (cnt.get(kk, 0),
                                                    use.get(kk, 0)))
                    cache.discard(evk)
                    cnt.pop(evk, None)
                    use.pop(evk, None)
                cache.add(k)
                cnt[k] = 1
                use[k] = tick
    return h / tot, (tot - h) / len(toks)


def sim_belady_atomic(toks, cap):
    """Optimal offline with same-event protection (upper bound)."""
    seq = [k for layers in toks for reqs in layers for k in reqs]
    bounds = []  # (start, end) per event in seq
    o = 0
    for layers in toks:
        for reqs in layers:
            bounds.append((o, o + len(reqs)))
            o += len(reqs)
    future = [0] * len(seq)
    last = {}
    for i in range(len(seq) - 1, -1, -1):
        future[i] = last.get(seq[i], len(seq) + 1)
        last[seq[i]] = i
    cache, heap = {}, []
    h = 0
    for a, b in bounds:
        ev = seq[a:b]
        evset = set(ev)
        for i in range(a, b):
            if seq[i] in cache:
                h += 1
        for i in range(a, b):
            k = seq[i]
            if k in cache:
                cache[k] = future[i]
                heapq.heappush(heap, (-future[i], k))
                continue
            while len(cache) >= cap and heap:
                negu, evk = heapq.heappop(heap)
                if evk in cache and cache[evk] == -negu and \
                   evk not in evset:
                    del cache[evk]
                    break
            if len(cache) >= cap:  # degenerate: all cached are event keys
                evk = max(cache, key=lambda kk: cache[kk])
                del cache[evk]
            cache[k] = future[i]
            heapq.heappush(heap, (-future[i], k))
    return h / len(seq), (len(seq) - h) / len(toks)


def main():
    sys.path.insert(0, ".")
    from src.score import s_eff, s_perf_fixed
    train, test = load_split()
    print(f"K4 traces: train {len(train)} test {len(test)} toks", flush=True)
    pop = Counter(k for t in train for r in t for k in r)
    popl = {}
    for t in train:
        for li, reqs in enumerate(t):
            popl.setdefault(li, Counter()).update(reqs)
    curves = {}
    for cap in CAPS:
        row = {}
        te = [[r for r in t] for t in test]
        row["lru"] = A.sim_atomic(te, cap)[:2]
        top = set(k for k, _ in pop.most_common(cap))
        h = sum(1 for t in test for r in t for k in r if k in top)
        tot = sum(len(r) for t in test for r in t)
        row["static"] = (h / tot, (tot - h) / len(test))
        for frac in (0.25, 0.5, 0.75):
            pins = set(k for k, _ in pop.most_common(int(cap * frac)))
            row[f"hyb{int(frac*100)}"] = A.sim_atomic(te, cap, pins)[:2]
        row["lfu"] = sim_lfu_atomic(te, cap)
        row["stale"] = sim_stale_atomic(te, cap)
        for k in (2, 4):
            pl = set(kk for li in range(40)
                     for kk, _ in popl[li].most_common(k))
            h2, m2 = A.sim_atomic(te, cap, pl)[:2]
            row[f"pinL{k}"] = (h2, m2)
        row["belady"] = sim_belady_atomic(te, cap)
        curves[cap] = {k: [round(v[0], 6), round(v[1], 3)]
                       for k, v in row.items()}
        print(f"  cap {cap:5d}: " + " ".join(
            f"{k}={v[0]:.3f}" for k, v in row.items()), flush=True)
    json.dump(curves, open(EDGE0 / "tracka_k4_curves.json", "w"))

    expert = M.EXPERT_K8_IQ2 * M.FMT_Q2K * M.K_SCALE
    nonexp = M.ATTN + M.SHARED + M.GDN + M.SCORES + M.GAPS + M.LMHEAD
    cpu = expert + nonexp + M.MGMT
    print(f"\n== CPU (K4/Q2K, corrected): {cpu:.1f} ms ==")
    print("TargGB policy    slots expGB  hit% miss/t  B/tok fetch "
          "tpsLo tpsHi tpsUp dT/dG ms/G")
    pareto, prev, cfg = [], None, {}
    for T in TARGETS:
        exp_gb = T - M.FIXED_GB
        slots = int(exp_gb * 1e9 / BUNDLE)
        best = None
        for pol in ("lru", "static", "hyb25", "hyb50", "hyb75", "lfu",
                    "stale", "pinL2", "pinL4"):
            cs = sorted(curves)
            if slots <= cs[0]:
                miss = curves[cs[0]][pol][1]
            else:
                for a, b in zip(cs, cs[1:]):
                    if slots <= b:
                        ma = curves[a][pol][1]
                        mb = curves[b][pol][1]
                        miss = mb + (slots - b) / (b - a) * (mb - ma)
                        break
                else:
                    miss = curves[cs[-1]][pol][1]
            fresh = miss * BUNDLE / 1e6
            fetch = fresh + miss * M.T_MISS
            lo = 1000 / (cpu / M.I5_LO + fetch)
            if best is None or lo > best[0]:
                best = (lo, pol, miss, fresh, fetch)
        lo, pol, miss, fresh, fetch = best
        hi = 1000 / (cpu / M.I5_HI + fetch)
        up = 1000 / (cpu / M.I5_HI + fresh / 2.87 + miss * M.T_MISS)
        hit = 1 - miss / 160.0
        seff = s_eff(T)
        if prev is None:
            dtg, msg = 0.0, 0.0
        else:
            pt, plo, pms = prev
            dtg = (lo - plo) / (T - pt)
            msg = (pms - 1000 / lo) / (T - pt)  # ms saved per GB
        prev = (T, lo, 1000 / lo)
        # winning pins for integration export
        if pol.startswith("hyb"):
            frac = int(pol[3:]) / 100
            pins = sorted(k for k, _ in pop.most_common(int(slots * frac)))
        elif pol.startswith("pinL"):
            kk = int(pol[4:])
            pins = sorted(kk2 for li in range(40)
                          for kk2, _ in popl[li].most_common(kk))
        elif pol == "static":
            pins = sorted(k for k, _ in pop.most_common(slots))
        else:
            pins = []
        cfg[str(T)] = {"policy": pol, "slots": slots, "npins": len(pins),
                       "pins_global": pins,
                       "expected_hit": round(hit, 6)}
        pareto.append({"target": T, "policy": pol, "slots": slots,
                       "exp_gb": round(exp_gb, 3), "hit": round(hit, 4),
                       "miss": round(miss, 1),
                       "bytes_per_tok": int(miss * BUNDLE),
                       "fetch_ms": round(fetch, 1),
                       "tps_lo": round(lo, 2), "tps_hi": round(hi, 2),
                       "tps_up": round(up, 2),
                       "marginal_tps_per_gb": round(dtg, 2),
                       "marginal_ms_saved_per_gb": round(msg, 1),
                       "s_eff": round(seff, 1)})
        print(f"{T:5.1f} {pol:8s} {slots:5d} {exp_gb:5.2f} {100*hit:5.1f} "
              f"{miss:6.1f} {fresh:6.1f}M {fetch:5.1f} "
              f"{lo:5.2f} {hi:5.2f} {up:5.2f} {dtg:+5.2f} {msg:+5.1f}")
    json.dump(pareto, open(EDGE0 / "tracka_k4_pareto.json", "w"), indent=1)
    json.dump(cfg, open(EDGE0 / "cache_config_k4.json", "w"))
    print("wrote tracka_k4_curves.json + tracka_k4_pareto.json + "
          "cache_config_k4.json")


if __name__ == "__main__":
    main()
