"""S2-Agent2 02: fetch missing L00 union experts + L00 shared expert.

- Union-of-top8 experts for L00 (224) from pinned GGUF via Range;
  skips slices already in /tmp/agent1_raw (validates size), dequants to
  /tmp/agent1_f32 with /tmp/dequant (iq2_xxs / iq2_s).
- Shared expert (gate/up/down bf16 + gate vector f32) for L00 from HF
  shard 14 -> s2 weights dir (NOT agent3 dir).
Output: weights/L00_shexp_{gate,up,down,inp}.pt
"""
import json
import os
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor

import requests
import torch

WS = "/home/fluxx/Workspace/adtc-llm-native-sparse"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "weights")
os.makedirs(OUT, exist_ok=True)
A3 = os.path.join(WS, "probes/agent3_funcmoe/weights")
RAW = "/tmp/agent1_raw"
DEQ = "/tmp/agent1_f32"
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


def dequant(src, dst, ty, ne0, nr):
    if os.path.exists(dst):
        return
    r = subprocess.run(["/tmp/dequant", src, dst, ty, str(ne0), str(nr)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"dequant {src}: {r.stderr[:200]}")


def main():
    import numpy as np
    r = np.load(os.path.join(A3, "routing_L00.npz"))
    need = sorted(set(r["top8"].ravel().tolist()))
    print(f"L00 union need={len(need)}", flush=True)
    L = 0
    tg = T[f"blk.{L}.ffn_gate_exps.weight"]
    tu = T[f"blk.{L}.ffn_up_exps.weight"]
    td = T[f"blk.{L}.ffn_down_exps.weight"]
    jobs = []
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
    n_deq = 0
    for e in need:
        for kind, ext in (("gate", "iq2xxs"), ("up", "iq2xxs"), ("down", "iq2s")):
            src = f"{RAW}/L{L:02d}_E{e:03d}_{kind}.{ext}"
            assert os.path.exists(src), src
            dst = f"{DEQ}/L{L:02d}_E{e:03d}_{kind}.f32"
            if not os.path.exists(dst):
                ty = "iq2_xxs" if "xxs" in ext else "iq2_s"
                ne0, nr = (2048, 512) if kind != "down" else (512, 2048)
                dequant(src, dst, ty, ne0, nr)
                n_deq += 1
    print(f"dequanted {n_deq} new slices", flush=True)

    # shared expert bf16 from HF shard 14
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

    pre = "model.language_model.layers.0.mlp"
    for nm, fn in (("shared_expert.gate_proj.weight", "L00_shexp_gate.pt"),
                   ("shared_expert.up_proj.weight", "L00_shexp_up.pt"),
                   ("shared_expert.down_proj.weight", "L00_shexp_down.pt"),
                   ("shared_expert_gate.weight", "L00_shexp_inp.pt")):
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            t = fetch_hf_tensor(f"{pre}.{nm}")
            torch.save(t, p)
            print(f"saved {fn} {tuple(t.shape)} {t.dtype}", flush=True)
    print("OK s2_02", flush=True)


if __name__ == "__main__":
    main()
