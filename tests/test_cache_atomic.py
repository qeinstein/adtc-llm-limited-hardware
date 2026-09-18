"""JOIN-2: atomic K-event cache semantics + C/Python equivalence.

Corner proof: with dyn_cap=2 holding {X=LRU-tail, Y}, event [M(new), X]
must score X as a HIT (pre-event state) and protect it while admitting
M. Legacy sequential processing self-evicts X (0/2); atomic scores 1/2.

Randomized: C edge0_cache_event == Python sim_atomic EXACTLY (and C
sequential == Python sequential) across seeds/caps/pins.
"""
import random
import struct
import subprocess
import sys
from pathlib import Path

EDGE0 = Path(__file__).resolve().parents[1] / "probes" / "edge0_port"
sys.path.insert(0, str(EDGE0.parent))

from edge0_port.cache_atomic import sim_atomic, sim_sequential


def build_driver(tmp_path):
    exe = tmp_path / "cache_atomic_test"
    r = subprocess.run(["gcc", "-O2", "-o", str(exe),
                        str(EDGE0 / "cache_atomic_test.c")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return exe


def run_driver(exe, dyn, events, pins, mode, tmp_path, tag):
    k = len(events[0])
    sf = tmp_path / f"s_{tag}.bin"
    with open(sf, "wb") as f:
        f.write(struct.pack("<II", len(events), k))
        for e in events:
            f.write(struct.pack("<%dH" % k, *e))
    pf = tmp_path / f"p_{tag}.bin"
    with open(pf, "wb") as f:
        f.write(struct.pack("<I", len(pins)))
        for p in sorted(pins):
            f.write(struct.pack("<H", p))
    r = subprocess.run([str(exe), str(dyn), str(sf), mode, str(pf)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    h, rr = r.stdout.split()[1], r.stdout.split()[3]
    return int(h), int(rr)


def test_self_eviction_corner(tmp_path):
    X, Y, M = 1, 2, 3
    # e1 fills {X=tail,Y}; e2=[M,X]: sequential evicts X for M (0/2),
    # atomic scores X a hit from pre-event state and evicts Y (1/2).
    toks = [[[X, Y]], [[M, X]]]  # 2 toks x 1 layer x K=2
    hs, _ = sim_sequential(toks, 2)
    ha, _ = sim_atomic(toks, 2)
    assert hs == 0.0 and ha == 0.25, (hs, ha)  # 0/4 vs 1/4
    exe = build_driver(tmp_path)
    events = [[X, Y], [M, X]]
    got, reqs = run_driver(exe, 2, events, set(), "atomic", tmp_path, "c")
    assert reqs == 4 and got == 1, (got, reqs)
    got_s, _ = run_driver(exe, 2, events, set(), "seq", tmp_path, "cs")
    assert got_s == 0, got_s


def test_randomized_c_python_agreement(tmp_path):
    exe = build_driver(tmp_path)
    for seed in range(1, 6):
        rng = random.Random(1000 + seed)
        for dyn, npins in ((0, 10), (7, 5), (64, 20)):
            pins = set(rng.sample(range(200), npins))
            events = []
            for _ in range(300):
                e = []
                for _ in range(4):
                    e.append(rng.randrange(40) if rng.random() < 0.7
                             else rng.randrange(1024))
                events.append(e)
            toks = [[e] for e in events]
            cap = dyn + len(pins)
            ha, _ = sim_atomic(toks, cap, pins)
            hs, _ = sim_sequential(toks, cap, pins)
            ga, ra = run_driver(exe, dyn, events, pins, "atomic",
                                tmp_path, f"a{seed}_{dyn}")
            gs, rs = run_driver(exe, dyn, events, pins, "seq",
                                tmp_path, f"s{seed}_{dyn}")
            assert ra == rs == 1200
            assert ga == round(ha * 1200), (seed, dyn, ga, ha)
            assert gs == round(hs * 1200), (seed, dyn, gs, hs)
