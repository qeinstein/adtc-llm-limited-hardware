"""Regression: GGUF KV parser must skip ALL scalar metadata types.

Phase 3.5 kernel died 4.3h in with `RuntimeError: kv type 5` (I32) inside
transcode_q2k, losing the Q2K/layer/sanity stages. The shipped parser only
handled STR(8)/ARR(9). This builds a synthetic GGUF exercising every scalar
KV type and asserts the shipped parser walks the header.
"""
import struct
import sys
from pathlib import Path

KDIR = Path(__file__).resolve().parents[1] / "kaggle" / \
    "native-sparse-edge0phase35-v1"
BDIR = Path(__file__).resolve().parents[1] / "kaggle" / \
    "native-sparse-edge0base-v1"
sys.path.insert(0, str(KDIR))
sys.path.insert(0, str(BDIR))

import edge0phase35_v1 as k35
import edge0base_v1 as kbase


def _str(b):
    return struct.pack("<Q", len(b)) + b


def _kv(key, typ, payload):
    return _str(key.encode()) + struct.pack("<I", typ) + payload


def _tensor(name, typ):
    return (_str(name.encode()) + struct.pack("<I", 2) +
            struct.pack("<QQ", 4, 8) + struct.pack("<I", typ) +
            struct.pack("<Q", 0))


def build_gguf(path):
    kvs = [
        _kv("general.name", 8, _str(b"synth")),
        _kv("general.tags", 9, struct.pack("<IQ", 8, 2) + _str(b"x") +
            _str(b"yy")),
        # I32 first among scalars: pre-fix code raises "kv type 5" here,
        # exactly the field failure.
        _kv("some.i32", 5, struct.pack("<i", -7)),
        _kv("t.u8", 0, b"\x01"),
        _kv("t.i8", 1, b"\xff"),
        _kv("t.u16", 2, struct.pack("<H", 9)),
        _kv("t.i16", 3, struct.pack("<h", -9)),
        _kv("t.u32", 4, struct.pack("<I", 10)),
        _kv("t.f32", 6, struct.pack("<f", 0.5)),
        _kv("t.bool", 7, b"\x01"),
        _kv("t.u64", 10, struct.pack("<Q", 11)),
        _kv("t.i64", 11, struct.pack("<q", -11)),
        _kv("t.f64", 12, struct.pack("<d", 0.25)),
        _kv("t.arr_u32", 9, struct.pack("<IQ", 4, 3) + b"\x01\x00\x00\x00" * 3),
    ]
    tensors = [_tensor("blk.0.ffn_gate_exps", 16),
               _tensor("output.weight", 8)]
    hdr = (b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) +
           b"".join(kvs) + b"".join(tensors))
    Path(path).write_bytes(hdr + bytes(64))


def test_kv_skip_all_scalar_types(tmp_path):
    f = tmp_path / "synth.gguf"
    build_gguf(f)
    assert k35.gguf_tensor_table(f) == [
        ("blk.0.ffn_gate_exps", "iq2_xxs"), ("output.weight", "q8_0")]


def test_kv_skip_edge0base_parser(tmp_path):
    # edge0base duplicated the walker with wrong header offsets and no
    # vtype-9 (array) support; v3 died in transcode_q2k with "kv walk".
    # Same synthetic file must walk identically there.
    f = tmp_path / "synth.gguf"
    build_gguf(f)
    assert kbase.gguf_tensor_table(f) == [
        ("blk.0.ffn_gate_exps", "iq2_xxs"), ("output.weight", "q8_0")]
