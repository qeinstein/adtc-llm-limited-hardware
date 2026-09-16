#!/usr/bin/env python3
"""Phase-1 Pareto model v2: LRU+Belady sims on the real route corpus, dense
bytes from the GGUF inventory, phase10b wall-measured buckets, N11-proxy ms
and i5-projected tok/s. Finds fastest config under 4/5/5.5/6 GB RSS.

Reads (in-repo): phase5e route_corpus.jsonl, agent1_expert/gguf_inventory.json
"""
import json
from collections import OrderedDict

CORPUS = ("research/native_sparse_experiments/results/"
          "phase5e_route_corpus_v1/route_corpus.jsonl")
INV = "probes/agent1_expert/gguf_inventory.json"
BPW = {"F32": 32.0, "Q5_K": 5.5, "Q6_K": 6.5625, "Q8_0": 8.5,
       "IQ2_XXS": 2.0625, "IQ2_S": 2.5, "Q4_K": 4.5, "Q2_K": 2.625}

# ---- measured inputs (this sprint, N100 real weights/kernels) ----
FMT_RATIO = {"iq2": 1.0, "q2k": 0.555, "q3k": 0.890, "q40": 0.833,
             "q4k": 0.635}          # ST expert-core ratio vs K8+IQ2
BUNDLE = {"iq2": 876544, "q2k": 1032192, "q3k": 1351680, "q40": 1769472,
          "q4k": 1769472}           # measured/exact bytes per expert bundle
# ---- adopted frame (phase5a/9b/10b wall measurements, "4-core x86" class)
CORE = 87.0                          # expert GEMV core, K8
EOVH_FIX, EOVH_K8 = 8.0, 15.0        # expert overhead: fixed + K-scaled
ATTN, GDN, SHARED = 48.0, 11.0, 8.0  # dense GEMV-ish (wall/op-frac)
SCORES, GAPS = 10.0, 17.0            # scores/softmax/AV; norms/elem/sampler
LMHEAD = 25.0                        # full 248k vocab Q4_K GEMV
MGMT = 5.0                           # slot/cache management (noDN path)
DENSE_Q2K = 0.60                     # dense-GEMV Q2_K win (J2; needs gate)
I5_LOW, I5_HIGH = 1.4, 1.7           # conservative i5 factors
KV_GB, OVH_GB = 0.25, 0.30


def load_corpus():
    toks = []
    with open(CORPUS) as f:
        for line in f:
            d = json.loads(line)
            toks.append([[(li << 8) | e for e in exps]
                         for li, exps in enumerate(d["layers"])])
    return toks


def lru_sim(toks, cap):
    cache = OrderedDict()
    hits = tot = 0
    for layers in toks:
        for reqs in layers:
            for k in reqs:
                tot += 1
                if k in cache:
                    hits += 1
                    cache.move_to_end(k)
                else:
                    cache[k] = None
                    if len(cache) > cap:
                        cache.popitem(last=False)
    return hits / tot, (tot - hits) / len(toks)


def belady_sim(toks, cap):
    # optimal offline: evict the key with farthest next use (heap, lazy del)
    import heapq
    seq = [k for layers in toks for reqs in layers for k in reqs]
    future = [0] * len(seq)
    last = {}
    for i in range(len(seq) - 1, -1, -1):
        k = seq[i]
        future[i] = last.get(k, len(seq) + 1)
        last[k] = i
    cache = {}   # k -> next use
    heap = []    # (-next use, k); stale entries skipped lazily
    hits = 0
    for i, k in enumerate(seq):
        if k in cache:
            hits += 1
        else:
            if len(cache) >= cap:
                while True:
                    negu, ev = heapq.heappop(heap)
                    if ev in cache and cache[ev] == -negu:
                        del cache[ev]
                        break
        cache[k] = future[i]
        heapq.heappush(heap, (-future[i], k))
    return hits / len(seq), (len(seq) - hits) / len(toks)


def dense_inventory():
    inv = json.load(open(INV))
    groups = {}
    for t in inv["tensors"]:
        n = t["name"]
        ne = 1
        for d in t["shape"]:
            ne *= d
        b = ne * BPW[t["type"]] / 8
        if "exps" in n:
            g = "routed-experts"
        elif "output.weight" in n:
            g = "lm-head"
        elif "token_embd" in n:
            g = "embeddings"
        elif "attn_" in n:
            g = "attention"
        elif "shexp" in n:
            g = "shared-expert"
        elif "ffn_gate_inp" in n or "ffn_gate_exps" in n or "norm" in n:
            g = "norms+router"
        else:
            g = "other-dense"
        groups[g] = groups.get(g, 0) + b
    return groups


def main():
    toks = load_corpus()
    print(f"corpus: {len(toks)} tokens x 40 layers x K8")
    for cap, ref in ((826, 0.4879), (1439, 0.6190), (2664, 0.7720)):
        h, m = lru_sim(toks, cap)
        ok = "MATCH" if abs(h - ref) < 0.002 else "MISMATCH!"
        print(f"  LRU@{cap}: hit={h:.4f} ref={ref} miss/tok={m:.1f} {ok}")
    for cap, ref in ((826, 0.6995), (1439, 0.7978), (2664, 0.8937)):
        h, m = belady_sim(toks, cap)
        ok = "MATCH" if abs(h - ref) < 0.002 else "MISMATCH!"
        print(f"  BEL@{cap}: hit={h:.4f} ref={ref} miss/tok={m:.1f} {ok}")

    groups = dense_inventory()
    tot = sum(groups.values())
    print("\nGGUF inventory bytes:")
    for g, b in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"  {g:16s} {b/1e9:7.3f} GB")
    print(f"  {'TOTAL':16s} {tot/1e9:7.3f} GB")
    dense_gb = (tot - groups["routed-experts"]) / 1e9
    # dense-q2k variant: Q4/Q5/Q6 dense GEMV tensors -> 2.625bpw (keys below)
    inv = json.load(open(INV))
    dq_bytes = 0
    for t in inv["tensors"]:
        n = t["name"]
        if "exps" in n:
            continue
        ne = 1
        for d in t["shape"]:
            ne *= d
        if t["type"] in ("Q4_K", "Q5_K", "Q6_K") and "token_embd" not in n \
           and "output.weight" not in n:
            dq_bytes += ne * BPW["Q2_K"] / 8
        else:
            dq_bytes += ne * BPW[t["type"]] / 8
    dense_q2k_gb = dq_bytes / 1e9
    print(f"  dense as-is {dense_gb:.3f} GB; dense-q2k {dense_q2k_gb:.3f} GB "
          f"(attn/shared only; head/embeddings stay)")

    caps = [320, 512, 826, 1024, 1439, 1792, 2300, 2664, 3200, 3584,
            4096, 4608, 5120, 5632, 6656]
    lru, bel = {}, {}
    for cap in caps:
        lru[cap] = lru_sim(toks, cap)
        bel[cap] = belady_sim(toks, cap)
    print("\nslots   LRU hit% miss/t | BEL hit% miss/t")
    for cap in caps:
        print(f"{cap:5d}  {100*lru[cap][0]:5.1f} {lru[cap][1]:6.1f} | "
              f"{100*bel[cap][0]:5.1f} {bel[cap][1]:6.1f}")

    def misses_at(curve, slots):
        cs = sorted(curve)
        if slots <= cs[0]:
            return curve[cs[0]][1]
        for a, b in zip(cs, cs[1:]):
            if slots <= b:
                ma, mb = curve[a][1], curve[b][1]
                return mb + (slots - b) / (b - a) * (mb - ma)
        ma, mb = curve[cs[-2]][1], curve[cs[-1]][1]
        sl = (mb - ma) / (cs[-1] - cs[-2]) * 0.3
        return max(6.0, mb + sl * (slots - cs[-1]))

    def ev(fmt, K, cache_gb, policy="lru", dense_q2k=False,
           lm_short=False, ssd_bw=1.0, t_miss=0.25):
        slots = int(cache_gb * 1e9 / BUNDLE[fmt])
        curve = bel if policy == "belady" else lru
        miss = misses_at(curve, slots) * (K / 8.0)
        fresh_mb = miss * BUNDLE[fmt] / 1e6
        core = CORE * FMT_RATIO[fmt] * (K / 8.0)
        eovh = EOVH_FIX + EOVH_K8 * (K / 8.0)
        gemv = (ATTN + SHARED) * (DENSE_Q2K if dense_q2k else 1.0)
        lm = 3.0 if lm_short else LMHEAD
        fetch = fresh_mb / (ssd_bw * 1000.0) * 1000.0 + miss * t_miss
        cpu = core + eovh + gemv + GDN + SCORES + GAPS + lm + MGMT
        n100 = cpu + fetch
        dg = dense_q2k_gb if dense_q2k else dense_gb
        rss = dg + cache_gb + KV_GB + OVH_GB
        return {"slots": slots, "miss": miss, "fresh_mb": fresh_mb,
                "core": core, "cpu": cpu, "fetch": fetch, "n100": n100,
                "i5lo": cpu / I5_LOW + fetch, "i5hi": cpu / I5_HIGH + fetch,
                "rss": rss}

    print("\n== fastest credible per RSS cap (LRU, ssd=1GB/s, t_miss=0.25ms) ==")
    print("cap    t/s  config")
    for cap in (4.0, 5.0, 5.5, 6.0):
        best = None
        for fmt in ("iq2", "q2k", "q4k", "q40"):
            for K in (8, 4):
                for cg in (1.0, 1.5, 2.0, 2.4, 3.0, 3.5, 4.0, 4.4, 4.8):
                    for dq in (False, True):
                        r = ev(fmt, K, cg, dense_q2k=dq)
                        if r["rss"] > cap:
                            continue
                        tps = 1000 / r["i5lo"]
                        if best is None or tps > best[0]:
                            best = (tps, fmt, K, dq, cg, r)
        tps, fmt, K, dq, cg, r = best
        print(f"<={cap}GB {tps:5.2f} {fmt} K={K} dq={int(dq)} cache={cg} "
              f"slots={r['slots']} miss={r['miss']:.0f} fresh={r['fresh_mb']:.0f}MB "
              f"core={r['core']:.1f} cpu={r['cpu']:.1f} fetch={r['fetch']:.1f} "
              f"n100={r['n100']:.1f} i5hi={1000/r['i5hi']:.1f}t/s rss={r['rss']:.2f}")

    print("\n== key rows (tok/s | peak RSS | quality note | mechanism) ==")
    key = [("iq2", 8, 1.0, "lru", False, False, "current bounded path"),
            ("iq2", 8, 2.4, "lru", False, False, "current + big cache"),
            ("q2k", 8, 2.4, "lru", False, False, "Q2_K experts, same tier?"),
            ("q2k", 8, 4.0, "lru", True, False, "+dense-Q2K (gate TBD)"),
            ("q2k", 4, 4.0, "lru", True, False, "+K4 (needs Phase-4 recovery)"),
            ("q4k", 4, 4.0, "lru", True, False, "K4+Q4K (better weights)"),
            ("q2k", 4, 4.0, "lru", True, True, "+LM-shortlist (unproven)"),
            ("q2k", 4, 4.0, "belady", True, False, "perfect-prefetch bound")]
    for fmt, K, cg, pol, dq, lms, note in key:
        r = ev(fmt, K, cg, policy=pol, dense_q2k=dq, lm_short=lms)
        print(f"  {1000/r['i5lo']:5.2f}t/s | {r['rss']:.2f}GB | {note} | "
              f"{fmt} K={K} {pol} slots={r['slots']} miss={r['miss']:.0f} "
              f"fresh={r['fresh_mb']:.0f} core={r['core']:.1f} "
              f"cpu={r['cpu']:.1f} fetch={r['fetch']:.1f} n100={r['n100']:.0f} "
              f"i5hi={1000/r['i5hi']:.1f}")

    print("\n== disk sensitivity (q2k K4 dq cache4.0) ==")
    for bw in (0.5, 1.0, 3.0):
        row = []
        for tm in (0.1, 0.5):
            r = ev("q2k", 4, 4.0, dense_q2k=True, ssd_bw=bw, t_miss=tm)
            row.append(f"tm={tm}:{1000/r['i5lo']:.1f}")
        print(f"  bw={bw}: " + " ".join(row))


if __name__ == "__main__":
    main()
