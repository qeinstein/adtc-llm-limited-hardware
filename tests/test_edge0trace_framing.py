"""Framing tests for the edge0trace trainset kernel (seq-order ids matching).

Regression: trace v1 aborted with `ids 79 != tok 82` because the first 3
hidden-eval sweeps carried no runtime ids (llama.cpp batching quirk; the
hidden states themselves are RMS-valid). Ids must match hidden-evals by
global seq order and tolerate missing sweeps — never by positional zip.
"""
import importlib.util
from pathlib import Path

import pytest

_KERNEL = (Path(__file__).resolve().parent.parent / "kaggle" /
           "native-sparse-edge0trace-v1" / "edge0trace_v1.py")
_spec = importlib.util.spec_from_file_location("edge0trace_v1", _KERNEL)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
match_ids_to_sweeps = _mod.match_ids_to_sweeps

NL = 4  # small layer count keeps the synthetic streams readable


def _h(t, layer, seq):
    return {"layer": layer, "seq": seq, "kind": 0}


def _i(t, layer, seq):
    return {"layer": layer, "seq": seq, "kind": 2, "vals": [t] * 8}


def _sweeps(nt, ids_fn):
    """Build h/ids streams: hidden in layer runs; ids per ids_fn(t, layer)."""
    h, ids, seq = [], [], 0
    for t in range(nt):
        for layer in range(NL):
            h.append((_h(t, layer, seq)))
            seq += 1
            for extra in ids_fn(t, layer):
                ids.append(extra(seq))
                seq += 1
    return h, ids


def test_v1_pattern_prefix_sweeps_lack_ids():
    # v1 forensics: sweeps 0..2 hidden-only, then per-layer (up, down).
    def ids_fn(t, layer):
        if t < 3:
            return []
        return [lambda s, t=t, layer=layer: _i(t, layer, s)]
    h, ids = _sweeps(6, ids_fn)
    m = match_ids_to_sweeps(h, ids, 6, n_layers=NL)
    assert set(m) == set(range(NL))
    for layer in range(NL):
        got = sorted(t for t, _ in m[layer])
        assert got == [3, 4, 5], (layer, got)


def test_full_interleave_maps_each_ids_to_own_sweep():
    def ids_fn(t, layer):
        return [lambda s, t=t, layer=layer: _i(t, layer, s)]
    h, ids = _sweeps(3, ids_fn)
    m = match_ids_to_sweeps(h, ids, 3, n_layers=NL)
    for layer in range(NL):
        assert sorted(t for t, _ in m[layer]) == [0, 1, 2]


def test_no_ids_at_all_is_tolerated():
    h, _ = _sweeps(2, lambda t, layer: [])
    assert match_ids_to_sweeps(h, [], 2, n_layers=NL) == {}


def test_duplicate_layer_eval_rejected():
    h, ids = _sweeps(2, lambda t, layer: [])
    # two ids records inside sweep 1's seq window for the same layer
    s0 = h[NL]["seq"]
    ids = [_i(1, 0, s0 + 1), _i(1, 0, s0 + 2)]
    with pytest.raises(AssertionError):
        match_ids_to_sweeps(h, ids, 2, n_layers=NL)


def test_ids_before_first_sweep_rejected():
    h, _ = _sweeps(2, lambda t, layer: [])
    with pytest.raises(AssertionError):
        match_ids_to_sweeps(h, [_i(0, 0, -1)], 2, n_layers=NL)


check_parity = _mod.check_parity


def test_parity_gate_accepts_v2_measured_noise():
    # v2 pids 0/1/2: parity 0.9994/0.9991/0.9987 at coverage 0.9634 —
    # small-integer fp16 boundary flips, provably benign.
    assert check_parity(0.9987, 0.9634, 2) == (0.9987, 0.9634)


def test_parity_gate_rejects_low_parity():
    with pytest.raises(RuntimeError):
        check_parity(0.50, 0.96, 0)


def test_parity_gate_rejects_low_coverage():
    with pytest.raises(RuntimeError):
        check_parity(1.0, 0.10, 0)
