"""S2-Agent2 08: fetch union experts (raw) + shared expert for layers.

Generalizes s2_02 to any layer list. Union-of-top8 from THIS dir's
routing_Lxx.npz; raw slices to /tmp/agent1_raw (size-validated, skipped if
complete); NO bulk dequant (s2_09 dequants streaming per expert and deletes
f32 after use, to bound /tmp). Shared expert bf16 from HF shard 14
(index-verified: ALL layers' shared experts live in shard 14).
Usage: python3 s2_08_fetch_layer.py 5,20,30
"""
import json
import os
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import torch

WS = "/home/fluxx/Workspace/adtc-llm-native-sparse"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "weights")
os.makedirs(OUT, exist_ok=True)
RAW = "/tmp/agent1_raw"
BASE_GGUF = ("https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/"
             "bc014a17be43adabd7066b7a86075ff935c6a4e2/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf")
BASE_HF = "https://huggingface.co/Qwen/Qwen3.5-35B-A3B/resolve/main"
INV = json.load(open(os.path.join(WS, "probes/agent1_expert/gguf_inventory.json")))
T = {t["name"]: t for t in INV["tensors"]}
GATE_EXPERT_BYTES = 270336
DOWN_EXPERT_BYTES = 335872


def fetch_range_gguf(a, b, out):
    if os.path.exists(out) and os.path.getsize(out) == b - a + 1:
        return False
    r = subprocess.run(["curl", "-sL", "--fail", "--retry", "3", "-r", f"{a}-{b}",
                        "-o", out, BASE_GGUF], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"fetch {a}-{b} failed: {r.stderr[:300]}")
    assert os.path.getsize(out) == b - a + 1
    return True


def fetch_range_hf(url, start, end, timeout=180):
    h = {"Range": f"bytes={start}-{end}"}
    r = requests.get(url, headers=h, timeout=timeout)
    r.raise_for_status()
    assert r.status_code == 206
    return r.content


def main():
    layers = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [5, 20, 30]
    jobs = []
    for L in layers:
        r = np.load(os.path.join(OUT, f"routing_L{L:02d}.npz"))
        need = sorted(set(r["top8"].ravel().tolist()))
        print(f"L{L:02d} union need={len(need)}", flush=True)
        tg = T[f"blk.{L}.ffn_gate_exps.weight"]
        tu = T[f"blk.{L}.ffn_up_exps.weight"]
        td = T[f"blk.{L}.ffn_down_exps.weight"]
        for e in need:
            for tinfo, nb, kind, ext in ((tg, GATE_EXPERT_BYTES, "gate", "iq2xxs"),
                                        (tu, GATE_EXPERT_BYTES, "up", "iq2xxs"),
                                        (td, DOWN_EXPERT_BYTES, "down", "iq2s")):
                src = f"{RAW}/L{L:02d}_E{e:03d}_{kind}.{ext}"
                a = tinfo["file_offset"] + e * nb
                if not (os.path.exists(src) and os.path.getsize(src) == nb):
                    jobs.append((a, a + nb - 1, src))
    print(f"fetch jobs: {len(jobs)} (~{sum(j[1]-j[0]+1 for j in jobs)/1e6:.0f} MB)", flush=True)
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(lambda j: fetch_range_gguf(*j), jobs))
    print("raw slices complete", flush=True)

    url14 = f"{BASE_HF}/model.safetensors-00014-of-00014.safetensors"
    n = struct.unpack("<Q", fetch_range_hf(url14, 0, 7))[0]
    h14 = json.loads(fetch_range_hf(url14, 8, 8 + n - 1))
    h14.pop("__metadata__", None)
    d14 = 8 + n

    def fetch_hf_tensor(name):
        e = h14[name]
        s, t = e["data_offsets"]
        raw = fetch_range_hf(url14, d14 + s, d14 + t - 1)
        shape = tuple(e["shape"])
        dt = e["dtype"]
        if dt == "BF16":
            return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape)
        elif dt == "F32":
            return torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape)
        raise ValueError(dt)

    for L in layers:
        pre = f"model.language_model.layers.{L}.mlp"
        for nm, fn in (("shared_expert.gate_proj.weight", f"L{L:02d}_shexp_gate.pt"),
                       ("shared_expert.up_proj.weight", f"L{L:02d}_shexp_up.pt"),
                       ("shared_expert.down_proj.weight", f"L{L:02d}_shexp_down.pt"),
                       ("shared_expert_gate.weight", f"L{L:02d}_shexp_inp.pt")):
            p = os.path.join(OUT, fn)
            if not os.path.exists(p):
                t = fetch_hf_tensor(f"{pre}.{nm}")
                torch.save(t, p)
                print(f"saved {fn} {tuple(t.shape)} {t.dtype}", flush=True)
    print("OK s2_08", flush=True)


if __name__ == "__main__":
    main()
