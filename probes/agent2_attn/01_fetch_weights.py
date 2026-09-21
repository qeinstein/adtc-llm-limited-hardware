"""Agent2 probe 01: range-fetch REAL Qwen3.5-35B-A3B full-attention weights.

Fetches (bf16) for full-attention layers {3, 19, 39} (early/mid/late):
  shard13: self_attn.q_proj (8192x2048 fused Q+gate), o_proj (2048x4096)
  shard14: k_proj, v_proj (512x2048), q_norm, k_norm (256), input_layernorm (2048)
  shard9:  embed_tokens rows for calibration token IDs (real embeddings)
Tokenizer: tokenizer.json from the same repo (tokenizers lib).

Output: probes/agent2_attn/weights/*.pt + manifest.json
"""
import io
import json
import os
import struct
import sys

import requests
import torch

REPO = "Qwen/Qwen3.5-35B-A3B"
BASE = f"https://huggingface.co/{REPO}/resolve/main"
LAYERS = [3, 19, 39]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
os.makedirs(OUT, exist_ok=True)

PROMPTS = [
    "Give one concise reason oral rehydration solution helps a child with watery diarrhoea.",
    "A two-year-old child has watery diarrhoea six times today and sunken eyes. What should I do?",
    "A pregnant woman at 30 weeks presents with severe headache, blurred vision, and swelling of the face. What danger signs should I check for and what should I do in a rural clinic without a doctor on site?",
    "The child has a fever of 39 degrees and a dry cough. What is the correct dose of amoxicillin for a 12 kg child?",
]


def fetch_range(url, start, end, timeout=120):
    h = {"Range": f"bytes={start}-{end}"}
    r = requests.get(url, headers=h, timeout=timeout)
    r.raise_for_status()
    assert r.status_code == 206, f"no range support: {r.status_code} {url}"
    return r.content


def shard_header(shard):
    url = f"{BASE}/{shard}"
    n = struct.unpack("<Q", fetch_range(url, 0, 7))[0]
    raw = fetch_range(url, 8, 8 + n - 1)
    hdr = json.loads(raw)
    hdr.pop("__metadata__", None)
    data_start = 8 + n
    return url, hdr, data_start


def fetch_tensor(url, hdr, data_start, name):
    e = hdr[name]
    s, t = e["data_offsets"]
    raw = fetch_range(url, data_start + s, data_start + t - 1)
    shape = tuple(e["shape"])
    dt = e["dtype"]
    if dt == "BF16":
        ten = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape)
    elif dt == "F32":
        ten = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape)
    elif dt == "F16":
        ten = torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(shape)
    else:
        raise ValueError(f"dtype {dt} for {name}")
    return ten


def main():
    manifest = {"repo": REPO, "layers": LAYERS, "tensors": {}}
    # Tokenizer -> calibration token IDs
    r = requests.get(f"{BASE}/tokenizer.json", timeout=120)
    r.raise_for_status()
    open(os.path.join(OUT, "tokenizer.json"), "wb").write(r.content)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(OUT, "tokenizer.json"))
    ids = []
    for p in PROMPTS:
        ids.extend(tok.encode(p).ids)
    # keep first 128 unique-ish (preserve order, dedupe)
    seen, cal_ids = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            cal_ids.append(i)
    cal_ids = cal_ids[:128]
    manifest["calibration_ids"] = cal_ids
    manifest["n_calibration"] = len(cal_ids)
    print(f"calibration tokens: {len(cal_ids)}", flush=True)

    # Shard 13: q_proj + o_proj
    url13, h13, d13 = shard_header("model.safetensors-00013-of-00014.safetensors")
    print(f"shard13 tensors: {len(h13)}", flush=True)
    for L in LAYERS:
        for nm in ("q_proj", "o_proj"):
            name = f"model.language_model.layers.{L}.self_attn.{nm}.weight"
            t = fetch_tensor(url13, h13, d13, name)
            fn = f"L{L}_{nm}.pt"
            torch.save(t, os.path.join(OUT, fn))
            manifest["tensors"][name] = {"file": fn, "shape": list(t.shape), "dtype": str(t.dtype)}
            print(f"saved {fn} {tuple(t.shape)}", flush=True)

    # Shard 14: k/v/norms
    url14, h14, d14 = shard_header("model.safetensors-00014-of-00014.safetensors")
    for L in LAYERS:
        for nm in ("k_proj", "v_proj"):
            name = f"model.language_model.layers.{L}.self_attn.{nm}.weight"
            t = fetch_tensor(url14, h14, d14, name)
            fn = f"L{L}_{nm}.pt"
            torch.save(t, os.path.join(OUT, fn))
            manifest["tensors"][name] = {"file": fn, "shape": list(t.shape), "dtype": str(t.dtype)}
            print(f"saved {fn} {tuple(t.shape)}", flush=True)
        for nm in ("q_norm", "k_norm"):
            name = f"model.language_model.layers.{L}.self_attn.{nm}.weight"
            t = fetch_tensor(url14, h14, d14, name)
            fn = f"L{L}_{nm}.pt"
            torch.save(t, os.path.join(OUT, fn))
            manifest["tensors"][name] = {"file": fn, "shape": list(t.shape), "dtype": str(t.dtype)}
        name = f"model.language_model.layers.{L}.input_layernorm.weight"
        t = fetch_tensor(url14, h14, d14, name)
        fn = f"L{L}_input_ln.pt"
        torch.save(t, os.path.join(OUT, fn))
        manifest["tensors"][name] = {"file": fn, "shape": list(t.shape), "dtype": str(t.dtype)}
    print("shard14 done", flush=True)

    # Shard 9: embedding rows for calibration IDs
    url9, h9, d9 = shard_header("model.safetensors-00009-of-00014.safetensors")
    e = h9["model.language_model.embed_tokens.weight"]
    print(f"embed: shape={e['shape']} dtype={e['dtype']}", flush=True)
    assert e["dtype"] == "BF16"
    hidden = e["shape"][1]
    row_bytes = hidden * 2
    base, _ = e["data_offsets"]
    rows = []
    # group contiguous IDs into single range requests
    order = sorted(cal_ids)
    groups, g = [], [order[0]]
    for i in order[1:]:
        if i == g[-1] + 1:
            g.append(i)
        else:
            groups.append(g)
            g = [i]
    groups.append(g)
    by_id = {}
    for g in groups:
        raw = fetch_range(url9, d9 + base + g[0] * row_bytes,
                          d9 + base + (g[-1] + 1) * row_bytes - 1)
        m = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(len(g), hidden)
        for k, tid in enumerate(g):
            by_id[tid] = m[k]
    emb = torch.stack([by_id[i] for i in cal_ids])
    torch.save(emb, os.path.join(OUT, "calib_embed.pt"))
    manifest["tensors"]["model.language_model.embed_tokens.weight[calib]"] = {
        "file": "calib_embed.pt", "shape": list(emb.shape), "dtype": str(emb.dtype)}
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print("OK: all weights fetched", flush=True)


if __name__ == "__main__":
    sys.exit(main())
