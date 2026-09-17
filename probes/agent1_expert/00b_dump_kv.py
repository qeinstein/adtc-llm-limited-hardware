"""Dump GGUF key-value metadata (hyperparams) from /tmp/gguf_meta.bin."""
import struct

d = open("/tmp/gguf_meta.bin", "rb").read()
off = 4
ver = struct.unpack_from("<i", d, off)[0]; off += 4
nt = struct.unpack_from("<q", d, off)[0]; off += 8
nkv = struct.unpack_from("<q", d, off)[0]; off += 8

def read_str(o):
    (n,) = struct.unpack_from("<q", d, o); o += 8
    return d[o:o + n].decode("utf-8", "replace"), o + n

SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
       10: "<Q", 11: "<q", 12: "<d"}
for _ in range(nkv):
    k, off = read_str(off)
    t = struct.unpack_from("<i", d, off)[0]; off += 4
    if t == 8:
        v, off = read_str(off)
    elif t == 9:
        at = struct.unpack_from("<i", d, off)[0]; off += 4
        (n,) = struct.unpack_from("<q", d, off); off += 8
        if at == 8:
            vs = []
            for _ in range(n):
                s, off = read_str(off); vs.append(s)
            v = vs if len(vs) > 4 else vs
            if isinstance(v, list) and len(v) > 4:
                v = f"[{len(v)} strings]"
        else:
            vs = [struct.unpack_from(FMT[at], d, off + i * SZ[at])[0] for i in range(min(n, 8))]
            off += SZ[at] * n
            v = vs if n <= 8 else f"{vs}... (n={n})"
    else:
        v = struct.unpack_from(FMT[t], d, off)[0]; off += SZ[t]
    if "tokenizer" not in k and "chat" not in k:
        print(f"{k} = {v}")
