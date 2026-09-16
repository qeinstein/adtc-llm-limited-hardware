#!/usr/bin/env python3
"""Fetch bf16 source weights for 2 experts (L20 E0,E1, gate_up+down) from
Qwen/Qwen3.5-35B-A3B safetensors via range requests, for Phase-1 quant-error
analysis. Total ~12 MB. Verifies sizes; prints tensor metadata."""
import json
import os
import struct
import subprocess
import sys

IDX = "/tmp/qwen35_idx.json"
OUT = "/tmp/edge0_phase1/bf16"
BASE = "https://huggingface.co/Qwen/Qwen3.5-35B-A3B/resolve/main"

os.makedirs(OUT, exist_ok=True)
d = json.load(open(IDX))
wm = d["weight_map"]
names = ["model.language_model.layers.20.mlp.experts.gate_up_proj",
         "model.language_model.layers.20.mlp.experts.down_proj"]
def shard_header(shard):
    r = subprocess.run(["curl", "-sL", "--fail", "-r", "0-7",
                        f"{BASE}/{shard}"], capture_output=True, check=True)
    (hlen,) = struct.unpack("<Q", r.stdout)
    r = subprocess.run(["curl", "-sL", "--fail", "-r", f"8-{8+hlen-1}",
                        f"{BASE}/{shard}"], capture_output=True, check=True)
    return 8 + hlen, json.loads(r.stdout)

hdrs = {}
for n in names:
    sh = wm[n]
    if sh not in hdrs:
        hdrs[sh] = shard_header(sh)
        print("shard:", sh, flush=True)
    base, hdr = hdrs[sh]
    t = hdr[n]
    print(n, "shape=", t["shape"], "dtype=", t["dtype"],
          "off=", t["data_offsets"], flush=True)

# expert slices (row-major, expert axis first)
# gate_up [256,1024,2048] bf16: expert e -> 1024*2048 u16
# down    [256,2048,512]  bf16: expert e -> 2048*512 u16
specs = {"gate_up_proj": 1024 * 2048 * 2, "down_proj": 2048 * 512 * 2}
for n in names:
    shard = wm[n]
    base, hdr = hdrs[shard]
    short = n.split(".")[-1]
    esz = specs[short]
    start = base + hdr[n]["data_offsets"][0]
    for e in (0, 1):
        a = start + e * esz
        b = a + esz - 1
        out = f"{OUT}/L20_E{e:03d}_{short}.bf16"
        if os.path.exists(out) and os.path.getsize(out) == esz:
            print("have", out, flush=True)
            continue
        tmp = out + ".part"
        r = subprocess.run(["curl", "-sL", "--fail", "--retry", "3",
                            "-r", f"{a}-{b}", "-o", tmp, f"{BASE}/{shard}"])
        if r.returncode != 0:
            sys.exit(f"fetch {a}-{b} failed")
        os.rename(tmp, out)
        print("got", out, esz, flush=True)
print("OK", flush=True)
