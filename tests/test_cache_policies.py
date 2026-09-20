"""JOIN-3: unit tests for tracka_k4 cache-policy sims.

Covers the TRUE SpecMD/phase5e least_stale port (EWMA reuse-interval,
hand-computed victim order incl. tiebreaks), atomic event protection,
Belady dominance over all online policies (single-key + multikey),
heap-Belady equality vs an obviously-correct scan reference, and
decayed_lfu (LFU equivalence without decay + a hand-computed decay flip).
"""
import importlib.util
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses need the host module registered
    spec.loader.exec_module(mod)
    return mod


T = _load("tracka_k4", "probes/edge0_port/tracka_k4.py")
A = _load("cache_atomic", "probes/edge0_port/cache_atomic.py")


def toks_of(stream):
    """Single-key events: toks[t][layer][reqs]."""
    return [[[k]] for k in stream]


def cum_hits(sim, stream, cap):
    """Cumulative hits after each prefix (pins down hit pattern)."""
    out = []
    for n in range(1, len(stream) + 1):
        toks = toks_of(stream[:n])
        rate, _ = sim(toks, cap)
        out.append(int(round(rate * n)))
    return out


def ref_lfu(stream, cap):
    """Independent classic LFU (freq, then LRU tiebreak)."""
    freq, use, cache = {}, {}, set()
    h = 0
    for t, k in enumerate(stream):
        if k in cache:
            h += 1
            freq[k] += 1
            use[k] = t
        else:
            if len(cache) >= cap:
                evk = min(cache, key=lambda x: (freq[x], use[x]))
                cache.discard(evk)
                del freq[evk]
                del use[evk]
            cache.add(k)
            freq[k] = 1
            use[k] = t
    return h


def test_leaststale_ewma_arithmetic():
    # Hand-traced against phase5e semantics (seed-first-interval,
    # 0.75/0.25 EWMA, victim = max last+max(ei,1), ties -> most
    # recent last-use): cap=3 stream below evicts A@e7, D@e8 (tie
    # over C via larger last), C@e11; pattern [0,0,0,1,1,1,1,0,0,1,1,0].
    A_, B, C, D = 1, 2, 3, 4
    stream = [A_, B, C, A_, B, C, A_, D, A_, B, C, D]
    assert cum_hits(T.sim_leaststale_atomic, stream, 3) == \
        [0, 0, 0, 1, 2, 3, 4, 4, 4, 5, 6, 6]


def test_leaststale_event_protection():
    # Multi-key events: e1=[C,D] must evict A (not the protected C,
    # which has the max predicted next use) -> C hits, A misses after.
    A_, B, C, D = 1, 2, 3, 4
    toks = [[[A_, B]], [[C, D]], [[C]], [[A_]]]
    h, n = 0, 0
    cum = []
    for i in range(1, 5):
        rate, _ = T.sim_leaststale_atomic(toks[:i], 2)
        n += len(toks[i - 1][0])
        h = int(round(rate * n))
        cum.append(h)
    assert cum == [0, 0, 1, 1]


def test_belady_dominates_online():
    # Single-key events: protection never binds, so the protected
    # oracle is the classic optimal -> hits >= every online policy.
    online = [lambda t, c: A.sim_atomic(t, c)[:2],
              T.sim_lfu_atomic, T.sim_decayed_lfu,
              T.sim_leaststale_atomic]
    for seed in range(20):
        rng = random.Random(1000 + seed)
        stream = rng.choices(range(12), k=200)
        toks = toks_of(stream)
        b, _ = T.sim_belady_atomic(toks, 5)
        bh = int(round(b * 200))
        for sim in online:
            r, _ = sim(toks, 5)
            assert bh >= int(round(r * 200)), (seed, sim)


def test_decayed_lfu_matches_lfu_without_decay(monkeypatch):
    monkeypatch.setattr(T, "HALVE_EVERY", 10 ** 12)
    streams = [[1, 1, 1, 1, 2, 2, 3, 1],
               [3, 1, 2, 1, 3, 3, 2, 1, 4, 4, 4, 2],
               random.Random(7).choices(range(6), k=120)]
    for s in streams:
        rate, _ = T.sim_decayed_lfu(toks_of(s), 2)
        assert int(round(rate * len(s))) == ref_lfu(s, 2), s[:8]


def test_decayed_lfu_decay_flips_decision(monkeypatch):
    # [A,A,A,A,B,B,C,A] cap=2: pure LFU evicts B@C (4v2) -> A hits (5).
    # HALVE=5 halves A to 2.0@req5 -> tie 2v2 -> LRU evicts A -> 4 hits.
    s = [1, 1, 1, 1, 2, 2, 3, 1]
    monkeypatch.setattr(T, "HALVE_EVERY", 10 ** 12)
    r_pure, _ = T.sim_decayed_lfu(toks_of(s), 2)
    monkeypatch.setattr(T, "HALVE_EVERY", 5)
    r_dec, _ = T.sim_decayed_lfu(toks_of(s), 2)
    assert abs(r_pure - 5 / 8) < 1e-9
    assert abs(r_dec - 4 / 8) < 1e-9


def test_belady_dominates_multikey():
    # Per-layer-shaped 4-key events (like real traces): the protected
    # oracle must still dominate every online policy.
    online = [lambda t, c: A.sim_atomic(t, c)[:2],
              T.sim_lfu_atomic, T.sim_decayed_lfu,
              T.sim_leaststale_atomic]
    for seed in range(10):
        rng = random.Random(2000 + seed)
        toks = [[rng.sample(range(20), 4) for _ in range(2)]
                for _ in range(40)]
        tot = 40 * 2 * 4
        for cap in (3, 6, 10):
            b, _ = T.sim_belady_atomic(toks, cap)
            bh = int(round(b * tot))
            for sim in online:
                r, _ = sim(toks, cap)
                assert bh >= int(round(r * tot)), (seed, cap, sim)


def ref_belady(toks, cap):
    """Obviously-correct protected Belady (O(cap) scan, no heap)."""
    seq = [k for t in toks for lay in t for k in lay]
    bounds, o = [], 0
    for t in toks:
        for lay in t:
            bounds.append((o, o + len(lay)))
            o += len(lay)
    future = [0] * len(seq)
    last = {}
    for i in range(len(seq) - 1, -1, -1):
        future[i] = last.get(seq[i], len(seq) + 1)
        last[seq[i]] = i
    cache, h = {}, 0
    for a, b in bounds:
        evset = set(seq[a:b])
        for i in range(a, b):
            if seq[i] in cache:
                h += 1
        for i in range(a, b):
            k = seq[i]
            if k in cache:
                cache[k] = future[i]
                continue
            if len(cache) >= cap:
                cands = [kk for kk in cache if kk not in evset] or \
                    list(cache)
                del cache[max(cands, key=lambda kk: cache[kk])]
            cache[k] = future[i]
    return h


def test_belady_matches_scan_reference():
    # REGRESSION: the first heap port discarded protected entries
    # instead of setting them aside, draining the heap and falling back
    # to unprotected max (belady lost to LRU at small caps on K4 traces).
    # Uniform + locality-structured streams, exact hit equality.
    for seed in range(10):
        rng = random.Random(2000 + seed)
        uni = [[rng.sample(range(20), 4) for _ in range(2)]
               for _ in range(40)]
        hot = rng.sample(range(40), 8)
        loc = [[[rng.choice(hot) if rng.random() < 0.7
                 else rng.randrange(40) for _ in range(4)]
                for _ in range(2)] for _ in range(40)]
        for toks, cap in ((uni, 3), (uni, 6), (uni, 10),
                          (loc, 6), (loc, 10), (loc, 16)):
            tot = sum(len(lay) for t in toks for lay in t)
            r, _ = T.sim_belady_atomic(toks, cap)
            assert int(round(r * tot)) == ref_belady(toks, cap), \
                (seed, cap)
