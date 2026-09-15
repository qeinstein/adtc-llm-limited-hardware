"""Agent3 probe 02: proxy routing + fetch union experts + shared expert.

Inputs: REAL bf16 embeds (01) -> proxy MoE inputs x = RMSNorm(emb) * g_L
  (REAL embedding directions, REAL post-attention-norm scales; direction is a
  proxy for true hidden states -- same methodology as agent2, validated there).
Routing: REAL F32 GGUF routers, exact llama.cpp semantics:
  probs = softmax(x @ W.T) over 256; top-8 by prob; w = renorm.
Fetches union-of-top8 experts/layer from pinned GGUF + dequant to f32
  (reuse agent1 layout), and shared-expert bf16 (gate/up/down + gate vector)
  from HF shard 14. Saves per-layer routing + proxy inputs for probe 03.

Validates proxy routing against the REAL route corpus (selection-frequency
  rank correlation) and reports top-K mass concentration.

Usage: python3 02_fetch_experts.py [0,10,20,30]
"""
import json
import os
import struct
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WS = "/home/fluxx/Workspace/adtc-llm-native-sparse"
OUT = os.path.join(HERE, "weights")
os.makedirs(OUT, exist_ok=True)
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
    layers = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 10, 20, 30]
    emb = torch.load(os.path.join(OUT, "calib_embed.pt"), map_location="cpu",
                     weights_only=True).to(torch.float32).numpy()
    N = emb.shape[0]
    print(f"N={N} tokens", flush=True)
    # RMSNorm(emb) once (eps 1e-6); per-layer scale applied later
    rms = np.sqrt((emb.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + 1e-6)
    emb_n = (emb / rms).astype(np.float64)

    # REAL corpus selection frequencies for validation
    corpus = os.path.join(WS, "research/native_sparse_experiments/results/"
                              "phase5e_route_corpus_v1/route_corpus.jsonl")
    recs = [json.loads(l) for l in open(corpus)]
    selfreq = []
    for L in range(40):
        c = Counter()
        for r in recs:
            c.update(r["layers"][L])
        tot = sum(c.values())
        selfreq.append(np.array([c.get(e, 0) / tot for e in range(256)]))
    print(f"corpus: {len(recs)} real tokens", flush=True)

    # HF shard14 header (shared experts + router gate check)
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
            ten = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape)
        elif dt == "F32":
            ten = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape)
        else:
            raise ValueError(dt)
        return ten

    for L in layers:
        W = np.fromfile(f"{RAW}/router_L{L:02d}.f32", dtype=np.float32).reshape(256, 2048)
        g = np.fromfile(f"{RAW}/L{L:02d}_post_attention_norm.f32", dtype=np.float32)
        X = emb_n * g[None, :].astype(np.float64)  # (N,2048) proxy inputs
        np.save(os.path.join(OUT, f"inputs_L{L:02d}.npy"), X.astype(np.float32))
        logits = X @ W.T.astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        top8 = np.argsort(P, axis=1)[:, ::-1][:, :8]
        pw = np.take_along_axis(P, top8, axis=1)
        pw = pw / pw.sum(axis=1, keepdims=True)
        np.savez_compressed(os.path.join(OUT, f"routing_L{L:02d}.npz"),
                            top8=top8, weights=pw.astype(np.float32),
                            logits=logits.astype(np.float32))
        # concentration + corpus agreement
        Ps = np.sort(P, axis=1)[:, ::-1]
        cum = Ps.cumsum(axis=1).mean(axis=0)
        pf = np.zeros(256)
        for row in top8:
            for e in row:
                pf[e] += 1
        pf /= pf.sum()
        # spearman-ish: pearson on log freqs over union support
        m = (pf > 0) | (selfreq[L] > 0)
        corr = float(np.corrcoef(np.log(pf[m] + 1e-9),
                                 np.log(selfreq[L][m] + 1e-9))[0, 1])
        nunion = len(set(top8.ravel().tolist()))
        print(f"L{L:02d}: rms(g)={float(np.sqrt((g**2).mean())):.3f} "
              f"top1/2/4/8 mass={cum[0]:.3f}/{cum[1]:.3f}/{cum[3]:.3f}/{cum[7]:.3f} "
              f"union={nunion} corpus_logcorr={corr:.3f}", flush=True)

        # fetch + dequant union experts (parallel curl, 12 workers)
        tg = T[f"blk.{L}.ffn_gate_exps.weight"]
        tu = T[f"blk.{L}.ffn_up_exps.weight"]
        td = T[f"blk.{L}.ffn_down_exps.weight"]
        need = sorted(set(top8.ravel().tolist()))
        jobs = []
        for e in need:
            for tinfo, nb, kind, ext in ((tg, GATE_EXPERT_BYTES, "gate", "iq2xxs"),
                                         (tu, GATE_EXPERT_BYTES, "up", "iq2xxs"),
                                         (td, DOWN_EXPERT_BYTES, "down", "iq2s")):
                src = f"{RAW}/L{L:02d}_E{e:03d}_{kind}.{ext}"
                a = tinfo["file_offset"] + e * nb
                jobs.append((a, a + nb - 1, src))
        # drop already-complete (validates size; refetch partials from kill)
        jobs = [j for j in jobs
                if not (os.path.exists(j[2]) and os.path.getsize(j[2]) == j[1] - j[0] + 1)]
        with ThreadPoolExecutor(max_workers=12) as ex:
            list(ex.map(lambda j: fetch_range_gguf(*j), jobs))
        for e in need:
            for kind, ext in (("gate", "iq2xxs"), ("up", "iq2xxs"), ("down", "iq2s")):
                src = f"{RAW}/L{L:02d}_E{e:03d}_{kind}.{ext}"
                assert os.path.exists(src), src
                dst = f"{DEQ}/L{L:02d}_E{e:03d}_{kind}.f32"
                ty = "iq2_xxs" if "xxs" in ext else "iq2_s"
                ne0, nr = (2048, 512) if kind != "down" else (512, 2048)
                dequant(src, dst, ty, ne0, nr)
        print(f"  L{L:02d}: fetched {len(jobs)} slices, {len(need)} experts ready",
              flush=True)

        # shared expert bf16 from HF
        pre = f"model.language_model.layers.{L}.mlp"
        for nm, fn in (("shared_expert.gate_proj.weight", f"L{L:02d}_shexp_gate.pt"),
                       ("shared_expert.up_proj.weight", f"L{L:02d}_shexp_up.pt"),
                       ("shared_expert.down_proj.weight", f"L{L:02d}_shexp_down.pt"),
                       ("shared_expert_gate.weight", f"L{L:02d}_shexp_inp.pt")):
            p = os.path.join(OUT, fn)
            if not os.path.exists(p):
                t = fetch_hf_tensor(f"{pre}.{nm}")
                torch.save(t, p)
                print(f"  saved {fn} {tuple(t.shape)} {t.dtype}", flush=True)
    print("OK probe02", flush=True)


if __name__ == "__main__":
    main()
