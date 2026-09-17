"""Atom factorization one-layer experiment -- 00: inputs + routing for L20.

Layer choice: L20 (middle layer, most representative; full 256/256 experts
already cached in /tmp/agent1_f32 as dequantized f32).

Inputs: proxy-REAL methodology (same as probes/agent2_attn + agent3_funcmoe,
validated there against the real route corpus):
    x = RMSNorm(real_embed_row) * real_post_attention_norm_scale
  Direction comes from REAL bf16 embedding rows; scales are REAL norm weights.
  TRUE hidden states would require a full 35B forward pass (out of scope for
  a cheap falsification); proxy routing is validated vs the real route corpus.

Routing: REAL F32 GGUF router, exact llama.cpp semantics
    (softmax over 256, top-8 by prob, renorm).

Split (by token ID, no leakage):
  - calib:   existing 320 ids (probes/agent3_funcmoe/weights/calib_ids.json)
  - heldout: newly mined unique ids from data/*.json (target +384)

Outputs (OUT = probes/atom_factorization_one_layer/assets/):
  - calib_ids.json, held_ids.json
  - held_embed.pt (bf16 rows for new ids)
  - inputs_L20.npy        (N,2048) f32, calib first then heldout
  - routing_L20.npz       top8 (N,8), weights (N,8) f32, logits (N,256) f32
  - split.json            {n_calib, n_held, ...}
  - routing_valid.json    corpus agreement + concentration stats
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)
WS = "/home/fluxx/Workspace/adtc-llm-native-sparse"
A3 = os.path.join(WS, "probes/agent3_funcmoe/weights")
RAW = "/tmp/agent1_raw"
L = 20
N_HELD_TARGET = 384

REPO = "Qwen/Qwen3.5-35B-A3B"
BASE = f"https://huggingface.co/{REPO}/resolve/main"


def fetch_range(url, start, end, timeout=180):
    # curl with retries (proven path from agent1; requests hit transient DNS
    # failures without retry). Returns exact byte range.
    with tempfile.NamedTemporaryFile(delete=False) as f:
        tmp = f.name
    try:
        r = subprocess.run(
            ["curl", "-sL", "--fail", "--retry", "5", "--retry-delay", "3",
             "--max-time", str(timeout), "-r", f"{start}-{end}",
             "-o", tmp, url],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"fetch {start}-{end} failed: {r.stderr[:300]}")
        with open(tmp, "rb") as f:
            data = f.read()
        assert len(data) == end - start + 1, \
            f"short fetch: {len(data)} vs {end - start + 1}"
        return data
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def mine_held_ids(cal_ids):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(
        WS, "probes/agent2_attn/weights/tokenizer.json"))
    texts = []
    for fn in ("data/medical_guidelines.json", "data/swahili_eval_set.json",
               "data/falcon_probe_sft.json", "data/medical_lora_dataset.json"):
        p = os.path.join(WS, fn)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        rows = d if isinstance(d, list) else d.get("rows", d.get("data", []))
        for r in rows:
            if isinstance(r, dict):
                for k in ("text", "query", "prompt", "completion", "answer",
                          "instruction", "input", "output"):
                    if isinstance(r.get(k), str) and len(r[k]) > 40:
                        texts.append(r[k])
            elif isinstance(r, str) and len(r) > 40:
                texts.append(r)
    seen = set(cal_ids)
    held = []
    for t in texts:
        for i in tok.encode(t).ids:
            if i not in seen:
                seen.add(i)
                held.append(i)
                if len(held) >= N_HELD_TARGET:
                    break
        if len(held) >= N_HELD_TARGET:
            break
    print(f"mined {len(held)} new held-out ids from {len(texts)} texts",
          flush=True)
    assert len(held) >= 256, "not enough fresh tokens for held-out"
    return held


def fetch_held_embeds(held_ids):
    url9 = f"{BASE}/model.safetensors-00009-of-00014.safetensors"
    n = struct.unpack("<Q", fetch_range(url9, 0, 7))[0]
    hdr = json.loads(fetch_range(url9, 8, 8 + n - 1))
    hdr.pop("__metadata__", None)
    data_start = 8 + n
    e = hdr["model.language_model.embed_tokens.weight"]
    assert e["dtype"] == "BF16"
    hidden = e["shape"][1]
    row_bytes = hidden * 2
    base, _ = e["data_offsets"]
    order = sorted(held_ids)
    groups, g = [], [order[0]]
    for i in order[1:]:
        if i == g[-1] + 1:
            g.append(i)
        else:
            groups.append(g)
            g = [i]
    groups.append(g)
    print(f"{len(groups)} range groups for {len(order)} rows", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    have = {}

    def one(g):
        raw = fetch_range(url9, data_start + base + g[0] * row_bytes,
                          data_start + base + (g[-1] + 1) * row_bytes - 1)
        m = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(
            len(g), hidden)
        return [(tid, m[k]) for k, tid in enumerate(g)]

    done = [0]
    with ThreadPoolExecutor(max_workers=12) as ex:
        for rows in ex.map(one, groups):
            for tid, row in rows:
                have[tid] = row
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == len(groups):
                print(f"  group {done[0]}/{len(groups)}", flush=True)
    emb = torch.stack([have[i] for i in held_ids])
    torch.save(emb, os.path.join(OUT, "held_embed.pt"))
    return emb


def main():
    cal = json.load(open(os.path.join(A3, "calib_ids.json")))
    cal_ids = cal["ids"]
    n_calib = len(cal_ids)
    print(f"calib ids: {n_calib}", flush=True)
    cal_emb = torch.load(os.path.join(A3, "calib_embed.pt"),
                         map_location="cpu", weights_only=True)

    held_path = os.path.join(OUT, "held_ids.json")
    if os.path.exists(held_path) and os.path.exists(
            os.path.join(OUT, "held_embed.pt")):
        held_ids = json.load(open(held_path))["ids"]
        held_emb = torch.load(os.path.join(OUT, "held_embed.pt"),
                              map_location="cpu", weights_only=True)
        print(f"reusing {len(held_ids)} held-out ids", flush=True)
    else:
        held_ids = mine_held_ids(cal_ids)
        held_emb = fetch_held_embeds(held_ids)
        json.dump({"ids": held_ids, "n": len(held_ids)},
                  open(held_path, "w"))
    n_held = len(held_ids)
    assert set(cal_ids).isdisjoint(set(held_ids)), "calib/held ID overlap!"

    emb = torch.cat([cal_emb, held_emb], dim=0).to(torch.float32).numpy()
    N = emb.shape[0]
    print(f"N={N} (calib {n_calib} + held {n_held})", flush=True)

    g = np.fromfile(f"{RAW}/L{L:02d}_post_attention_norm.f32",
                    dtype=np.float32)
    rms = np.sqrt((emb.astype(np.float64) ** 2).mean(axis=1, keepdims=True)
                  + 1e-6)
    X = (emb / rms * g[None, :].astype(np.float64)).astype(np.float32)
    np.save(os.path.join(OUT, "inputs_L20.npy"), X)
    del emb, cal_emb, held_emb

    W = np.fromfile(f"{RAW}/router_L{L:02d}.f32",
                    dtype=np.float32).reshape(256, 2048)
    logits = X.astype(np.float64) @ W.T.astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    top8 = np.argsort(P, axis=1)[:, ::-1][:, :8]
    pw = np.take_along_axis(P, top8, axis=1)
    pw = pw / pw.sum(axis=1, keepdims=True)
    np.savez_compressed(os.path.join(OUT, "routing_L20.npz"),
                        top8=top8, weights=pw.astype(np.float32),
                        logits=logits.astype(np.float32))
    del P, logits

    Ps = np.sort(
        np.exp(np.load(os.path.join(OUT, "routing_L20.npz"))["logits"]
               .astype(np.float64)), axis=1)[:, ::-1]
    Ps /= Ps.sum(axis=1, keepdims=True)
    cum = Ps.cumsum(axis=1).mean(axis=0)

    # validate proxy routing vs REAL route corpus (selection-freq logcorr)
    corpus = os.path.join(
        WS, "research/native_sparse_experiments/results/"
        "phase5e_route_corpus_v1/route_corpus.jsonl")
    recs = [json.loads(l) for l in open(corpus)]
    c = Counter()
    for r in recs:
        lay = r["layers"][L] if isinstance(r["layers"][L], list) \
            else r["layers"][str(L)]
        c.update(lay)
    tot = sum(c.values())
    selfreq = np.array([c.get(e, 0) / tot for e in range(256)])
    pf = np.zeros(256)
    for row in top8:
        for e in row:
            pf[e] += 1
    pf /= pf.sum()
    m = (pf > 0) | (selfreq > 0)
    corr = float(np.corrcoef(np.log(pf[m] + 1e-9),
                             np.log(selfreq[m] + 1e-9))[0, 1])
    nunion = len(set(top8.ravel().tolist()))
    nunion_cal = len(set(top8[:n_calib].ravel().tolist()))
    nunion_held = len(set(top8[n_calib:].ravel().tolist()))
    print(f"L{L:02d}: top1/2/4/8 mass={cum[0]:.3f}/{cum[1]:.3f}/"
          f"{cum[3]:.3f}/{cum[7]:.3f} union={nunion} "
          f"(cal {nunion_cal} / held {nunion_held}) "
          f"corpus_logcorr={corr:.3f} ({len(recs)} real tokens)", flush=True)

    json.dump({
        "layer": L, "n_calib": n_calib, "n_held": n_held, "N": N,
    }, open(os.path.join(OUT, "split.json"), "w"), indent=1)
    json.dump({
        "top_mass": [float(cum[0]), float(cum[1]), float(cum[3]),
                     float(cum[7])],
        "union": nunion, "union_calib": nunion_cal,
        "union_held": nunion_held,
        "corpus_logcorr": corr, "corpus_tokens": len(recs),
    }, open(os.path.join(OUT, "routing_valid.json"), "w"), indent=1)
    print("OK 00", flush=True)


if __name__ == "__main__":
    sys.exit(main())
