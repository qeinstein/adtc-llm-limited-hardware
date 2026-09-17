#!/usr/bin/env python3
"""Pinned-hot + dynamic-cache policy sims on the real route corpus.

Outcome-independent staging/cache infrastructure (no quality assumption).
Compares, at fixed slot budgets: global LRU, static-pin-only, pinned-hot +
LRU-remainder (honest train/test split AND same-split reference), and
per-layer pinned + global LRU. Reports hit/miss/fresh-MB + miss distribution
(for prefetch-pipe sizing).

Reads: research/.../phase5e_route_corpus_v1/route_corpus.jsonl (in-repo).
Writes: stdout tables + cache_curves.json (slot -> misses/token per policy).
"""
import json
from collections import Counter, OrderedDict

CORPUS = ("research/native_sparse_experiments/results/"
          "phase5e_route_corpus_v1/route_corpus.jsonl")
BUNDLE_IQ2 = 876544


def load():
    toks = []
    with open(CORPUS) as f:
        for line in f:
            d = json.loads(line)
            toks.append([[(li << 8) | e for e in exps]
                         for li, exps in enumerate(d["layers"])])
    return toks


def run_lru(toks, cap, pinned=frozenset()):
    dyn_cap = max(0, cap - len(pinned))
    cache = OrderedDict()
    hits = tot = 0
    per_tok = []
    for layers in toks:
        m = 0
        for reqs in layers:
            for k in reqs:
                tot += 1
                if k in pinned or k in cache:
                    hits += 1
                    if k in cache:
                        cache.move_to_end(k)
                else:
                    m += 1
                    if dyn_cap > 0:
                        cache[k] = None
                        if len(cache) > dyn_cap:
                            cache.popitem(last=False)
        per_tok.append(m)
    return hits / tot, (tot - hits) / len(toks), per_tok


def topn(counter, n):
    return frozenset(k for k, _ in counter.most_common(n))


def main():
    toks = load()
    n = len(toks)
    train, test = toks[:n // 2], toks[n // 2:]
    pop_train = Counter(k for t in train for r in t for k in r)
    pop_all = Counter(k for t in toks for r in t for k in r)
    # per-layer popularity (train)
    pop_layer = {}
    for t in train:
        for li, reqs in enumerate(t):
            c = pop_layer.setdefault(li, Counter())
            c.update(reqs)

    import statistics as st
    print(f"corpus {n} tokens; train/test {len(train)}/{len(test)}")
    print("policy            slots  pinned eval   hit%  miss/t  p50  p95  freshMB(tok,iq2)")
    out = {"policies": {}}
    cfgs = []
    for cap in (1024, 1792, 2664, 3584, 4608, 5632):
        cfgs.append((f"lru", cap, frozenset(), toks))
        cfgs.append((f"static-all", cap, topn(pop_all, cap), toks))
        for frac in (0.25, 0.5):
            pn = int(cap * frac)
            cfgs.append((f"pin{int(frac*100)}+lru_TRAIN", cap,
                         topn(pop_train, pn), test))
            cfgs.append((f"pin{int(frac*100)}+lru_ALL", cap,
                         topn(pop_all, pn), toks))
        # per-layer pin: top-2/layer (80) + top-4/layer (160)
        for k in (2, 4):
            pl = frozenset(kk for li in range(40)
                            for kk, _ in pop_layer[li].most_common(k))
            cfgs.append((f"pinL{k}+lru_TRAIN", cap, pl, test))
    for name, cap, pinned, ev in cfgs:
        h, m, pt = run_lru(ev, cap, pinned)
        p50 = st.median(pt)
        p95 = sorted(pt)[int(0.95 * len(pt))]
        fresh = m * BUNDLE_IQ2 / 1e6
        key = f"{name}@{cap}"
        out["policies"][key] = {"hit": h, "miss_per_tok": m, "p50": p50,
                                "p95": p95, "ntok": len(ev),
                                "pinned": len(pinned)}
        print(f"{name:16s} {cap:5d} {len(pinned):6d} {len(ev):4d} "
              f"{100*h:5.1f} {m:6.1f} {p50:4.0f} {p95:4.0f} {fresh:7.1f}")
    json.dump(out, open("probes/edge0_port/cache_curves.json", "w"), indent=1)
    print("wrote probes/edge0_port/cache_curves.json")


if __name__ == "__main__":
    main()
