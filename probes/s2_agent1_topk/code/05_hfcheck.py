"""S2 Agent1 script 05: HF bf16 cross-check of L00 gate/up zero-rows.

Question: are the ~42% all-zero gate/up rows in L00 GGUF-dequant REAL
(structural sparsity in bf16 source) or a GGUF/quant/fetch artifact?

Method: fetch expert-0 slice of fused experts.gate_up_proj [256,1024,2048]
and experts.down_proj (expert 0) in bf16 from Qwen/Qwen3.5-35B-A3B
safetensors via HTTPS Range, for L0 (+L10 control). Compare zero-row
patterns and values vs /tmp/agent1_f32 GGUF dequants. Gate/up half order
([gate;up] vs [up;gate]) is resolved by value match on dense L10.

Writes results/hf_crosscheck.json + stdout verdict.
"""
import json
import os
import struct

import numpy as np
import requests
import torch

WS = "/home/fluxx/Workspace/adtc-llm-native-sparse"
RES = os.path.join(WS, "probes/s2_agent1_topk/results")
DEQ = "/tmp/agent1_f32"
BASE = "https://huggingface.co/Qwen/Qwen3.5-35B-A3B/resolve/main"


def fetch_range(url, start, end, timeout=240):
    h = {"Range": f"bytes={start}-{end}"}
    r = requests.get(url, headers=h, timeout=timeout)
    r.raise_for_status()
    assert r.status_code == 206, f"{r.status_code} for {url}"
    return r.content


def main():
    os.makedirs(RES, exist_ok=True)
    idx = json.loads(requests.get(f"{BASE}/model.safetensors.index.json",
                                  timeout=120).content)
    wm = idx["weight_map"]
    out = {}
    hdrs = {}

    def shard_entry(key):
        fn = wm[key]
        if fn not in hdrs:
            url = f"{BASE}/{fn}"
            n = struct.unpack("<Q", fetch_range(url, 0, 7))[0]
            h = json.loads(fetch_range(url, 8, 8 + n - 1))
            h.pop("__metadata__", None)
            hdrs[fn] = (url, 8 + n, h)
        url, dstart, h = hdrs[fn]
        return url, dstart, h[key]

    def expert0(key):
        url, dstart, e = shard_entry(key)
        shape, dt = tuple(e["shape"]), e["dtype"]
        s, t = e["data_offsets"]
        assert dt == "BF16", (key, dt)
        assert len(shape) == 3 and shape[0] == 256, (key, shape)
        per = (t - s) // 256
        raw = fetch_range(url, dstart + s, dstart + s + per - 1)
        W = torch.frombuffer(bytearray(raw),
                             dtype=torch.bfloat16).float().numpy()
        return W.reshape(shape[1], shape[2]), shape

    def cmp(kind_hf, H, L, kind_gguf):
        sh = (2048, 512) if kind_gguf == "down" else (512, 2048)
        assert H.shape == sh, (kind_hf, H.shape)
        G = np.fromfile(f"{DEQ}/L{L:02d}_E000_{kind_gguf}.f32",
                        dtype=np.float32).reshape(sh)
        zr = (np.linalg.norm(H, axis=1) == 0)
        zr_g = (np.linalg.norm(G, axis=1) == 0)
        agree = float((zr == zr_g).mean())
        nz = (~zr) & (~zr_g)
        rel = float(np.linalg.norm(H[nz] - G[nz]) /
                    np.linalg.norm(H[nz])) if nz.sum() else -1.0
        key = f"L{L}_{kind_hf}_vs_{kind_gguf}"
        out[key] = {"hf_zero_rows": int(zr.sum()),
                    "gguf_zero_rows": int(zr_g.sum()),
                    "zero_pattern_agree": agree,
                    "nonzero_rel_diff": rel}
        print(f"  {key}: HF_zero={zr.sum()} GGUF_zero={zr_g.sum()} "
              f"agree={agree:.4f} nz_rel={rel:.4f}", flush=True)

    for L in (0, 10):
        GU, sh = expert0(f"model.language_model.layers.{L}.mlp.experts."
                         f"gate_up_proj")
        print(f"L{L} gate_up_proj expert0 {GU.shape} (fused {sh})",
              flush=True)
        assert GU.shape == (1024, 2048), GU.shape
        cmp("first512", GU[:512], L, "gate")
        cmp("last512", GU[512:], L, "up")
        cmp("first512", GU[:512], L, "up")
        cmp("last512", GU[512:], L, "gate")
        D, sh = expert0(f"model.language_model.layers.{L}.mlp.experts."
                        f"down_proj")
        print(f"L{L} down_proj expert0 {D.shape} (fused {sh})", flush=True)
        cmp("down", D, L, "down")
    json.dump(out, open(os.path.join(RES, "hf_crosscheck.json"), "w"), indent=1)
    print("OK 05", flush=True)


if __name__ == "__main__":
    main()
