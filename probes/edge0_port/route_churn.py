#!/usr/bin/env python3
"""Quant-induced route churn: does Q2_K routing invalidate IQ2-derived cache pins?

Gate-free systems analysis (no quality assumption). Compares staged_iq2 vs
staged_q2k route traces from the COMPLETE staged-q2k kernel (n_gen=64, reps=3):
  1. event-level zip overlap/exact (reproduces kernel's route_agreement as check)
  2. per-layer mean overlap (where churn concentrates)
  3. hot-set stability: Jaccard(top-N IQ2, top-N Q2K) at Pareto slot counts
  4. miss impact: decode-phase token rows, LRU + IQ2 pins on IQ2-test vs Q2K-test

Reads: $EDGE0_TRACES/*.routes.jsonl (default /tmp/edge0_phase1/traces/staged-q2k/...)
Writes: stdout tables.
"""
import json
import os
import re
import sys
from collections import Counter, OrderedDict

TRACES = os.environ.get("EDGE0_TRACES", "/tmp/edge0_phase1/traces/staged-q2k")


def events(prefix, rep):
    p = os.path.join(TRACES, f"{prefix}_rep{rep}.routes.jsonl")
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            m = re.match(r"blk\.(\d+)\.(ffn_\w+_exps)", d.get("weight", ""))
            if not m:
                continue
            out.append((int(m.group(1)), m.group(2), d.get("shape", [8, 0]),
                        d.get("ids", [])))
    return out


def ov_exact(a, b):
    sa, sb = set(a), set(b)
    if not sa or len(sa) != len(sb):
        return None
    return len(sa & sb) / len(sa), sa == sb


def n_layers(evs):
    return max((li for li, w, s, ids in evs if w == "ffn_gate_exps"), default=-1) + 1


def decode_token_rows(evs, nl):
    """Group decode-phase (ntok==1) gate events into tokens via layer-0 boundaries."""
    toks, cur = [], {}
    for li, w, shape, ids in evs:
        if w != "ffn_gate_exps" or len(shape) != 2 or shape[1] != 1:
            continue
        if li == 0 and cur:
            toks.append(cur)
            cur = {}
        cur[li] = ids[:8]
    if cur:
        toks.append(cur)
    return [[t.get(li, []) for li in range(nl)] for t in toks if len(t) == nl]


def run_lru(toks, cap, pinned=frozenset()):
    dyn_cap = max(0, cap - len(pinned))
    cache = OrderedDict()
    hits = tot = 0
    for layers in toks:
        for li, reqs in enumerate(layers):
            for e in reqs:
                k = (li << 8) | e
                tot += 1
                if k in pinned or k in cache:
                    hits += 1
                    if k in cache:
                        cache.move_to_end(k)
                elif dyn_cap > 0:
                    cache[k] = None
                    if len(cache) > dyn_cap:
                        cache.popitem(last=False)
    return (tot - hits) / len(toks) if toks else 0.0


def main():
    reps = (1, 2, 3)
    # 1. event-level zip per rep (kernel-identical: all events incl. 3 weights)
    print("rep  events  mean_overlap  exact_rate")
    for r in reps:
        a = events("staged_iq2", r)
        b = events("staged_q2k", r)
        n = min(len(a), len(b))
        if len(a) != len(b):
            print(f"  rep{r}: LENGTH MISMATCH iq2={len(a)} q2k={len(b)} (diverged gen?)",
                  file=sys.stderr)
        o = e = c = 0
        pr = [0.0, 0]
        dc = [0.0, 0]
        for (la, wa, sa, ia), (lb, wb, sb, ib) in zip(a[:n], b[:n]):
            if (la, wa) != (lb, wb):
                continue
            rr = ov_exact(ia, ib)
            if rr is None:
                continue
            oi, ei = rr
            o += oi
            e += ei
            c += 1
            tgt = dc if sb[1] == 1 else pr
            tgt[0] += oi
            tgt[1] += 1
        print(f"  {r}  {c:6d}  {o/c:.4f}        {e/c:.4f}   "
              f"prompt={pr[0]/pr[1]:.4f}(n={pr[1]}) decode={dc[0]/dc[1]:.4f}(n={dc[1]})")
    # 2. per-layer overlap, gate weight, all reps pooled
    _nl = n_layers(events("staged_iq2", 1))
    print(f"MoE layers in trace: {_nl}")
    per_layer = {li: [0.0, 0] for li in range(_nl)}
    for r in reps:
        ga = [(li, ids) for li, w, s, ids in events("staged_iq2", r) if w == "ffn_gate_exps"]
        gb = [(li, ids) for li, w, s, ids in events("staged_q2k", r) if w == "ffn_gate_exps"]
        for (la, ia), (lb, ib) in zip(ga, gb):
            if la != lb:
                continue
            rr = ov_exact(ia, ib)
            if rr is None:
                continue
            per_layer[la][0] += rr[0]
            per_layer[la][1] += 1
    lavg = sorted(((v[0] / v[1] if v[1] else float("nan"), li) for li, v in per_layer.items()))
    print("per-layer mean overlap worst-8:", [(li, f"{o:.3f}") for o, li in lavg[:8]])
    print("per-layer mean overlap best-3:", [(li, f"{o:.3f}") for o, li in lavg[-3:]])
    ncov = sum(1 for _, li in lavg for v in [per_layer[li]] if v[1])
    print(f"layers with aligned gate events: {ncov}/{_nl}")
    # 3. hot-set Jaccard over gate events
    pop_i = Counter()
    pop_q = Counter()
    for r in reps:
        for li, w, s, ids in events("staged_iq2", r):
            if w == "ffn_gate_exps":
                pop_i.update((li << 8) | e for e in ids)
        for li, w, s, ids in events("staged_q2k", r):
            if w == "ffn_gate_exps":
                pop_q.update((li << 8) | e for e in ids)
    print("slots  jaccard(topN iq2, topN q2k)")
    for slots in (320, 512, 826, 1024, 1439, 1792, 2664):
        si = set(k for k, _ in pop_i.most_common(slots))
        sq = set(k for k, _ in pop_q.most_common(slots))
        print(f"{slots:5d}  {len(si & sq)/len(si | sq):.4f}")
    # 4. decode-token LRU: pins from iq2-train, eval iq2-test vs q2k-test
    ti = [t for r in reps for t in decode_token_rows(events("staged_iq2", r), _nl)]
    tq = [t for r in reps for t in decode_token_rows(events("staged_q2k", r), _nl)]
    print(f"decode tokens: iq2={len(ti)} q2k={len(tq)}")
    train, test_i = ti[:len(ti) // 2], ti[len(ti) // 2:]
    test_q = tq[len(tq) // 2:]
    pop_tr = Counter((li << 8) | e for t in train for li, reqs in enumerate(t) for e in reqs)
    print("cap   pinned  miss/t(iq2-test)  miss/t(q2k-test)  churn-delta")
    for cap in (826, 1439, 2664):
        pn = cap // 2
        pinned = frozenset(k for k, _ in pop_tr.most_common(pn))
        mi = run_lru(test_i, cap, pinned)
        mq = run_lru(test_q, cap, pinned)
        print(f"{cap:5d} {pn:6d}  {mi:14.1f}  {mq:14.1f}  {mq-mi:+.1f}")


if __name__ == "__main__":
    main()
