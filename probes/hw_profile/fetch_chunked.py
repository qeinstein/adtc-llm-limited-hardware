"""Chunked resumable fetch for large single tensors (e.g. output.weight 286MB).
Splits into 8MB parts, fetches missing parts, assembles. Safe to re-run.
Usage: fetch_chunked.py <tensor_name> <nchunks>
Uses inventory offsets. Writes parts to /tmp/hwprof_dense/.parts/.
"""
import json
import os
import subprocess
import sys

BASE = "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/bc014a17be43adabd7066b7a86075ff935c6a4e2/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
OUTDIR = "/tmp/hwprof_dense"
PARTS = os.path.join(OUTDIR, ".parts")
os.makedirs(PARTS, exist_ok=True)
INV = json.load(open("/home/fluxx/Workspace/adtc-llm-native-sparse/probes/agent1_expert/gguf_inventory.json"))
T = {t["name"]: t for t in INV["tensors"]}
names = sorted(T, key=lambda n: T[n]["file_offset"])
SIZE = {}
for i, n in enumerate(names):
    if i + 1 < len(names):
        SIZE[n] = T[names[i + 1]]["file_offset"] - T[n]["file_offset"]

CHUNK = 8 << 20


def fetch_range(a, b, out):
    n = b - a + 1
    if os.path.exists(out) and os.path.getsize(out) == n:
        return True
    tmp = out + ".part"
    for attempt in range(4):
        r = subprocess.run(["curl", "-sL", "--fail", "--retry", "2", "--max-time", "120",
                            "-r", f"{a}-{b}", "-o", tmp, BASE],
                           capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) == n:
            os.rename(tmp, out)
            return True
    return False


def main(name, nchunks=None):
    t = T[name]
    total = SIZE[name]
    base = t["file_offset"]
    n = nchunks or (total + CHUNK - 1) // CHUNK
    step = (total + n - 1) // n
    tag = name.replace(".", "_")
    ok = True
    for i in range(n):
        a = base + i * step
        b = min(base + (i + 1) * step, base + total) - 1
        if a > b:
            continue
        out = os.path.join(PARTS, f"{tag}.p{i:03d}")
        if fetch_range(a, b, out):
            print(f"part {i}/{n} ok ({b-a+1} B)", flush=True)
        else:
            print(f"part {i}/{n} FAILED", flush=True)
            ok = False
    if not ok:
        raise SystemExit("parts missing, re-run")
    dest = os.path.join(OUTDIR, tag + ".bin")
    with open(dest, "wb") as f:
        for i in range(n):
            out = os.path.join(PARTS, f"{tag}.p{i:03d}")
            if not os.path.exists(out):
                continue
            with open(out, "rb") as g:
                while True:
                    blk = g.read(1 << 20)
                    if not blk:
                        break
                    f.write(blk)
    got = os.path.getsize(dest)
    assert got == total, f"{got} vs {total}"
    print(f"assembled {dest} ({got} B)")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
