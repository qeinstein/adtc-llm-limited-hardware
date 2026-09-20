#!/usr/bin/env python3
"""Atomic K-event cache sims (JOIN-2 runtime semantics) + sequential baseline.

Event = one (token, layer) request set (K=4 deployed, any K supported).
Atomic: all K hits/misses scored from the SAME pre-event state; hits
touched in request order; misses admitted in request order with
current-event keys protected from eviction (victim = LRU tail walking
upward past event keys; degenerate cap<=K: plain tail). Mirrors
edge0_cache_event() in cache_policy.h EXACTLY (cross-checked by
tests/test_cache_atomic.py, including randomized streams).

Toks shape (same as tracka/cache_sim): toks[tok][layer] = [keys...].
"""
from collections import OrderedDict

KEY_MOD = 10240  # (layer<<8)|expert key space; folds above like the C bitset


def _norm(k):
    return k % KEY_MOD


def sim_atomic(toks, cap, pins=frozenset()):
    """Atomic-event hybrid (pins + LRU). Returns (hit_rate, miss_per_tok)."""
    pins = frozenset(_norm(k) for k in pins)  # pin bitset folds mod 10240
    dyn = max(0, cap - len(pins))
    c = OrderedDict()  # tail=LRU (left) .. head=MRU (right)
    hits = tot = 0
    for layers in toks:
        for reqs in layers:
            ks = [int(k) for k in reqs]  # LRU keys raw, exactly like C
            tot += len(ks)
            if dyn <= 0:
                hits += sum(1 for k in ks if k in pins)
                continue
            pre = [(k in pins or k in c) for k in ks]
            hits += sum(pre)
            ev = set(ks)
            for k, h in zip(ks, pre):  # touch pre-event hits
                if h and k in c:
                    c.move_to_end(k)
            for k, h in zip(ks, pre):  # admit misses
                if h or k in c or k in pins:
                    continue
                if len(c) >= dyn:  # victim: tail walk past event keys
                    v = None
                    for cand in c:  # OrderedDict iterates tail(LRU)->head
                        if cand not in ev:
                            v = cand
                            break
                    if v is None:  # degenerate: cap <= K
                        v = next(iter(c))
                    del c[v]
                c[k] = None
    n = len(toks)
    return hits / tot, (tot - hits) / n


def sim_sequential(toks, cap, pins=frozenset()):
    """Legacy per-request interleaved sim (Track-A semantics)."""
    pins = frozenset(_norm(k) for k in pins)
    dyn = max(0, cap - len(pins))
    c = OrderedDict()
    hits = tot = 0
    for layers in toks:
        for reqs in layers:
            for k in (int(x) for x in reqs):
                tot += 1
                if k in pins or k in c:
                    hits += 1
                    if k in c:
                        c.move_to_end(k)
                elif dyn > 0:
                    c[k] = None
                    if len(c) > dyn:
                        c.popitem(last=False)
    n = len(toks)
    return hits / tot, (tot - hits) / n
