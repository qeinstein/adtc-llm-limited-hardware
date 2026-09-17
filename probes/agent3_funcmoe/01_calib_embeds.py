"""Agent3 probe 01: extended calibration token set + REAL embedding rows.

Reuses agent2's 117 calibration IDs (clinical EN/SW prompts) first, then extends
to N=320 with more real clinical/swahili text from data/*.json, tokenized with
the REAL Qwen3.5 tokenizer. Fetches REAL bf16 embed_tokens rows from HF
shard 9 via grouped HTTPS Range (same method as probes/agent2_attn/01).

Output: probes/agent3_funcmoe/weights/calib_ids.json, calib_embed.pt
"""
import json
import os
import struct

import requests
import torch

REPO = "Qwen/Qwen3.5-35B-A3B"
BASE = f"https://huggingface.co/{REPO}/resolve/main"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "weights")
os.makedirs(OUT, exist_ok=True)
WS = "/home/fluxx/Workspace/adtc-llm-native-sparse"
N_TARGET = 320


def fetch_range(url, start, end, timeout=180):
    h = {"Range": f"bytes={start}-{end}"}
    r = requests.get(url, headers=h, timeout=timeout)
    r.raise_for_status()
    assert r.status_code == 206, f"no range support: {r.status_code}"
    return r.content


def main():
    a2tok = os.path.join(WS, "probes/agent2_attn/weights/tokenizer.json")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a2tok)
    a2man = json.load(open(os.path.join(WS, "probes/agent2_attn/weights/manifest.json")))
    base_ids = list(a2man["calibration_ids"])
    print(f"agent2 base ids: {len(base_ids)}", flush=True)

    texts = []
    for fn in ("data/medical_guidelines.json", "data/swahili_eval_set.json",
               "data/falcon_probe_sft.json"):
        p = os.path.join(WS, fn)
        d = json.load(open(p))
        rows = d if isinstance(d, list) else d.get("rows", d.get("data", []))
        for r in rows:
            if isinstance(r, dict):
                for k in ("text", "query", "prompt", "completion", "answer"):
                    if isinstance(r.get(k), str) and len(r[k]) > 40:
                        texts.append(r[k])
            elif isinstance(r, str) and len(r) > 40:
                texts.append(r)
    print(f"extra texts: {len(texts)}", flush=True)

    seen, cal_ids = set(base_ids), list(base_ids)
    for t in texts:
        for i in tok.encode(t).ids:
            if i not in seen:
                seen.add(i)
                cal_ids.append(i)
                if len(cal_ids) >= N_TARGET:
                    break
        if len(cal_ids) >= N_TARGET:
            break
    print(f"total calib ids: {len(cal_ids)}", flush=True)
    assert len(cal_ids) >= 300, "not enough unique tokens for rank-256 measurements"

    # shard 9 header
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

    # which ids do we already have from agent2?
    have = {}
    a2emb = torch.load(os.path.join(WS, "probes/agent2_attn/weights/calib_embed.pt"),
                       map_location="cpu", weights_only=True)
    for tid, row in zip(base_ids, a2emb):
        have[tid] = row
    need = [i for i in cal_ids if i not in have]
    print(f"have {len(have)}, need {len(need)}", flush=True)
    order = sorted(need)
    groups, g = [], [order[0]]
    for i in order[1:]:
        if i == g[-1] + 1:
            g.append(i)
        else:
            groups.append(g)
            g = [i]
    groups.append(g)
    print(f"{len(groups)} range groups", flush=True)
    for gi, g in enumerate(groups):
        raw = fetch_range(url9, data_start + base + g[0] * row_bytes,
                          data_start + base + (g[-1] + 1) * row_bytes - 1)
        m = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(len(g), hidden)
        for k, tid in enumerate(g):
            have[tid] = m[k]
        if gi % 20 == 0:
            print(f"  group {gi}/{len(groups)}", flush=True)
    emb = torch.stack([have[i] for i in cal_ids])
    torch.save(emb, os.path.join(OUT, "calib_embed.pt"))
    json.dump({"ids": cal_ids, "n": len(cal_ids),
               "base117": base_ids}, open(os.path.join(OUT, "calib_ids.json"), "w"))
    print(f"OK: calib_embed {tuple(emb.shape)}", flush=True)


if __name__ == "__main__":
    main()
