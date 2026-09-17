"""10: expand the proxy pool for a fair student ceiling (+2560 tokens).

Small-N result (h=512: train 0.011 / test 1.27) proves N=256 is the
overfit regime -- a capacity verdict needs more training points.
Mines fresh token IDs (disjoint from the existing 704), fetches their
REAL bf16 embeds, and builds inputs + exact routing + exact R8 + S + B
for L20. Original calib-320/held-384 files are untouched; held-384 stays
the FIXED test set for comparability.

Outputs (assets/): *_X.* (X = extension pool), split_big.json
  train = calib-320 + 2304 new; val = 256 new; test = held-384 (orig idx).
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
WS = "/home/fluxx/Workspace/adtc-llm-native-sparse"
RAW = "/tmp/agent1_raw"
DEQ = "/tmp/agent1_f32"
L = 20
N_NEW = 2560
BASE = "https://huggingface.co/Qwen/Qwen3.5-35B-A3B/resolve/main"


def fetch_range(url, start, end, timeout=180):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        tmp = f.name
    try:
        r = subprocess.run(
            ["curl", "-sL", "--fail", "--retry", "5", "--retry-delay", "3",
             "--max-time", str(timeout), "-r", f"{start}-{end}",
             "-o", tmp, url], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"fetch failed: {r.stderr[:200]}")
        with open(tmp, "rb") as f:
            data = f.read()
        assert len(data) == end - start + 1
        return data
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def main():
    cal_ids = json.load(open(os.path.join(
        WS, "probes/agent3_funcmoe/weights/calib_ids.json")))["ids"]
    held_ids = json.load(open(os.path.join(OUT, "held_ids.json")))["ids"]
    seen = set(cal_ids) | set(held_ids)
    xp = os.path.join(OUT, "ext_ids.json")
    ep = os.path.join(OUT, "ext_embed.pt")
    if os.path.exists(xp) and os.path.exists(ep):
        new_ids = json.load(open(xp))["ids"]
        emb = torch.load(ep, map_location="cpu", weights_only=True)
        print(f"reusing {len(new_ids)} ext ids", flush=True)
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(
            WS, "probes/agent2_attn/weights/tokenizer.json"))
        texts = []
        for fn in ("data/medical_guidelines.json", "data/swahili_eval_set.json",
                   "data/falcon_probe_sft.json", "data/medical_lora_dataset.json"):
            p = os.path.join(WS, fn)
            d = json.load(open(p))
            rows = d if isinstance(d, list) else d.get("rows", d.get("data", []))
            for r in rows:
                if isinstance(r, dict):
                    for k in ("text", "query", "prompt", "completion",
                              "answer", "instruction", "input", "output"):
                        if isinstance(r.get(k), str) and len(r[k]) > 20:
                            texts.append(r[k])
                elif isinstance(r, str) and len(r) > 20:
                    texts.append(r)
        # also slice long texts into windows to harvest more ids
        new_ids = []
        for t in texts:
            ids = tok.encode(t).ids
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    new_ids.append(i)
                    if len(new_ids) >= N_NEW:
                        break
            if len(new_ids) >= N_NEW:
                break
        print(f"mined {len(new_ids)} new ids from {len(texts)} texts",
              flush=True)
        assert len(new_ids) >= 2000, "not enough fresh tokens"
        url9 = f"{BASE}/model.safetensors-00009-of-00014.safetensors"
        n = struct.unpack("<Q", fetch_range(url9, 0, 7))[0]
        hdr = json.loads(fetch_range(url9, 8, 8 + n - 1))
        hdr.pop("__metadata__", None)
        d0 = 8 + n
        e = hdr["model.language_model.embed_tokens.weight"]
        hidden, row_bytes = e["shape"][1], e["shape"][1] * 2
        base, _ = e["data_offsets"]
        order = sorted(new_ids)
        groups, g = [], [order[0]]
        for i in order[1:]:
            if i == g[-1] + 1:
                g.append(i)
            else:
                groups.append(g)
                g = [i]
        groups.append(g)
        print(f"{len(groups)} range groups", flush=True)
        have = {}

        def one(g):
            raw = fetch_range(url9, d0 + base + g[0] * row_bytes,
                              d0 + base + (g[-1] + 1) * row_bytes - 1)
            m = torch.frombuffer(bytearray(raw),
                                 dtype=torch.bfloat16).reshape(len(g), hidden)
            return [(tid, m[k]) for k, tid in enumerate(g)]

        done = [0]
        with ThreadPoolExecutor(max_workers=12) as ex:
            for rows in ex.map(one, groups):
                for tid, row in rows:
                    have[tid] = row
                done[0] += 1
                if done[0] % 100 == 0 or done[0] == len(groups):
                    print(f"  group {done[0]}/{len(groups)}", flush=True)
        emb = torch.stack([have[i] for i in new_ids])
        torch.save(emb, ep)
        json.dump({"ids": new_ids, "n": len(new_ids)}, open(xp, "w"))
    Nn = len(new_ids)

    # inputs + routing
    g = np.fromfile(f"{RAW}/L{L:02d}_post_attention_norm.f32", dtype=np.float32)
    E = emb.to(torch.float32).numpy()
    del emb
    rms = np.sqrt((E.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + 1e-6)
    X = (E / rms * g[None, :].astype(np.float64)).astype(np.float32)
    del E
    np.save(os.path.join(OUT, "inputs_L20_X.npy"), X)
    W = np.fromfile(f"{RAW}/router_L{L:02d}.f32",
                    dtype=np.float32).reshape(256, 2048)
    logits = X.astype(np.float64) @ W.T.astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    top8 = np.argsort(P, axis=1)[:, ::-1][:, :8]
    pw = np.take_along_axis(P, top8, axis=1)
    pw = pw / pw.sum(axis=1, keepdims=True)
    np.savez_compressed(os.path.join(OUT, "routing_L20_X.npz"), top8=top8,
                        weights=pw.astype(np.float32))
    del P, logits
    print(f"X: N={Nn} union={len(set(top8.ravel().tolist()))}", flush=True)

    # exact R8 for new tokens (stream union experts)
    w8 = pw.astype(np.float64)
    use = defaultdict(list)
    for t in range(Nn):
        for s in range(8):
            use[int(top8[t, s])].append((t, s))
    R8 = np.zeros((Nn, 2048), np.float64)
    for i, e in enumerate(sorted(use)):
        G = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_gate.f32",
                        dtype=np.float32).reshape(512, 2048)
        U = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_up.f32",
                        dtype=np.float32).reshape(512, 2048)
        D = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_down.f32",
                        dtype=np.float32).reshape(2048, 512)
        gv = X @ G.T
        uv = X @ U.T
        Y = (((gv / (1.0 + np.exp(-gv))) * uv) @ D.T).astype(np.float64)
        for (t, s) in use[e]:
            R8[t] += w8[t, s] * Y[t]
        del G, U, D, gv, uv, Y
        if (i + 1) % 60 == 0:
            print(f"  R8 {i+1}/{len(use)}", flush=True)
    np.savez_compressed(os.path.join(OUT, "teacher_L20_X.npz"),
                        R8=R8.astype(np.float32))

    # shared S + B
    sh = np.load(os.path.join(OUT, "shexp_L20.npz"))
    Xd = X.astype(np.float64)
    gv = Xd @ sh["G"].T.astype(np.float64)
    uv = Xd @ sh["U"].T.astype(np.float64)
    S = (((gv / (1.0 + np.exp(-gv))) * uv) @ sh["D"].T.astype(np.float64))
    sg = 1.0 / (1.0 + np.exp(-(Xd @ sh["w"].T.astype(np.float64).ravel())))
    S = S * sg[:, None]
    B = (R8 + S).astype(np.float32)
    np.savez_compressed(os.path.join(OUT, "targets_L20_X.npz"), B=B,
                        S=S.astype(np.float32))
    print(f"|R8|={np.linalg.norm(R8,axis=1).mean():.3f} "
          f"|B|={np.linalg.norm(B,axis=1).mean():.3f}", flush=True)

    # big split: train = calib-320 + 2304 new; val = 256 new; test = held-384
    split = json.load(open(os.path.join(OUT, "split.json")))
    n_calib = split["n_calib"]
    rng = np.random.default_rng(11)
    perm = rng.permutation(Nn)
    va_new = perm[:256]
    tr_new = perm[256:]
    json.dump({
        "n_calib": n_calib, "n_held": split["N"] - n_calib, "n_new": Nn,
        "train_calib": list(range(n_calib)),
        "train_new": [int(i) for i in tr_new],
        "val_new": [int(i) for i in va_new],
    }, open(os.path.join(OUT, "split_big.json"), "w"))
    print(f"train={n_calib}+{len(tr_new)} val={len(va_new)} "
          f"test={split['N']-n_calib}(held, fixed)", flush=True)
    print("OK 10", flush=True)


if __name__ == "__main__":
    sys.exit(main())
