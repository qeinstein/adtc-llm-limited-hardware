"""Agent1 probe 00: parse GGUF tensor table from fetched metadata prefix.

Reads /tmp/gguf_meta.bin (12 MiB prefix of the 9.9 GiB UD-IQ2_XXS file;
tensor infos end at 10989695 per probe11 README). Prints the full tensor
inventory grouped by family + quant type, and writes machine-readable JSON.

Usage: python3 00_gguf_inventory.py
"""
import json
import struct
import sys

META = "/tmp/gguf_meta.bin"
OUT = "/home/fluxx/Workspace/adtc-llm-native-sparse/probes/agent1_expert/gguf_inventory.json"

GGML_TYPES = {
    0: ("F32", 4, 1), 1: ("F16", 2, 1), 2: ("Q4_0", 18, 32), 3: ("Q4_1", 20, 32),
    6: ("Q5_0", 22, 32), 7: ("Q5_1", 24, 32), 8: ("Q8_0", 34, 32),
    10: ("Q2_K", 84, 256), 11: ("Q3_K", 110, 256), 12: ("Q4_K", 144, 256),
    13: ("Q5_K", 176, 256), 14: ("Q6_K", 210, 256), 15: ("Q8_K", 292, 256),
    16: ("IQ2_XXS", 66, 256), 17: ("IQ2_XS", 66, 256), 18: ("IQ3_XXS", 110, 256),
    19: ("IQ1_S", 50, 256), 20: ("IQ4_NL", 36, 32), 21: ("IQ3_S", 82, 256),
    22: ("IQ2_S", 82, 256), 23: ("IQ4_XS", 72, 256), 24: ("I8", 1, 1),
    25: ("I16", 2, 1), 26: ("I32", 4, 1), 27: ("I64", 8, 1),
    28: ("F64", 8, 1), 29: ("IQ1_M", 56, 256), 30: ("BF16", 2, 1),
    31: ("Q4_0_4_4", 18, 32), 32: ("Q4_0_4_8", 18, 32), 33: ("Q4_0_8_8", 18, 32),
}


def main():
    with open(META, "rb") as f:
        d = f.read()
    assert d[:4] == b"GGUF", d[:4]
    off = 4
    ver = struct.unpack_from("<i", d, off)[0]; off += 4
    nt = struct.unpack_from("<q", d, off)[0]; off += 8
    nkv = struct.unpack_from("<q", d, off)[0]; off += 8

    def read_str(o):
        (n,) = struct.unpack_from("<q", d, o); o += 8
        s = d[o:o + n].decode("utf-8", "replace"); o += n
        return s, o

    def skip_val(t, o):
        # returns new offset
        if t in (0,):  # u8
            return o + 1
        if t in (1,):  # i8
            return o + 1
        if t in (2, 6):  # u16, f16
            return o + 2
        if t in (3, 7):  # u32, f32
            return o + 4
        if t in (10, 11):  # u64, i64, f64
            return o + 8
        if t in (4, 5, 9):  # bool, str handled below... actually 4=i8? fix below
            raise AssertionError(t)
        raise AssertionError(t)

    # GGUF value types: 0=U8 1=I8 2=U16 3=I16 4=U32 5=I32 6=F32 7=BOOL
    # 8=STR 9=ARR 10=U64 11=I64 12=F64
    SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    for _ in range(nkv):
        _, off = read_str(off)
        t = struct.unpack_from("<i", d, off)[0]; off += 4
        if t == 8:  # string
            _, off = read_str(off)
        elif t == 9:  # array
            at = struct.unpack_from("<i", d, off)[0]; off += 4
            (n,) = struct.unpack_from("<q", d, off); off += 8
            if at == 8:
                for _ in range(n):
                    _, off = read_str(off)
            else:
                off += SZ[at] * n
        else:
            off += SZ[t]
    kv_end = off
    print(f"version={ver} ntensors={nt} nkv={nkv} kv_end={kv_end}")

    tensors = []
    for _ in range(nt):
        name, off = read_str(off)
        (nd,) = struct.unpack_from("<i", d, off); off += 4
        shape = []
        for _ in range(nd):
            (v,) = struct.unpack_from("<q", d, off); off += 8
            shape.append(v)
        (ty,) = struct.unpack_from("<i", d, off); off += 4
        (t_off,) = struct.unpack_from("<q", d, off); off += 8
        tensors.append({"name": name, "shape": shape, "type_id": ty,
                        "type": GGML_TYPES.get(ty, (f"UNK{ty}", -1, -1))[0],
                        "offset": t_off})
    print(f"parsed {len(tensors)} tensors, header_bytes_used={off}")
    assert off == 10989695, f"tensor infos end moved: {off}"
    data_start = (off + 31) // 32 * 32
    print(f"data_start={data_start}")
    for t in tensors:
        t["file_offset"] = data_start + t["offset"]

    # size accounting
    def nbytes(t):
        tname, bs, k = GGML_TYPES.get(t["type_id"], ("?", 0, 1))
        n = 1
        for v in t["shape"]:
            n *= v
        return n // k * bs if k else 0

    fam = {}
    for t in tensors:
        parts = t["name"].split(".")
        key = ".".join(parts[2:]) if parts[0] == "blk" else ".".join(parts[:1])
        if parts[0] == "blk":
            key = parts[2]
        else:
            key = parts[0]
        e = fam.setdefault((key, t["type"]), {"n": 0, "bytes": 0, "ex": t["name"], "shape_ex": t["shape"]})
        e["n"] += 1
        e["bytes"] += nbytes(t)
    print(f"{'family':28s} {'type':10s} {'count':>5s} {'MiB':>10s}  example")
    for (key, ty), e in sorted(fam.items(), key=lambda kv: -kv[1]["bytes"]):
        print(f"{key:28s} {ty:10s} {e['n']:5d} {e['bytes']/2**20:10.1f}  {e['ex']} {e['shape_ex']}")

    with open(OUT, "w") as f:
        json.dump({"data_start": data_start, "tensors": tensors}, f)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
