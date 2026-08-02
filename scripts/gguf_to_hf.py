#!/usr/bin/env python3
"""Recover HF-format weights from our shipped Q4_0 GGUF.

WHY THIS EXISTS: the RunPod pod (and with it output/jamii-lora, the LoRA adapter
from the 8-hour training run) is gone. All that survives of that run is the
quantized GGUF we published to Hugging Face. We need HF-format weights to do a
follow-up LoRA pass, because the real remaining problem is generation quality —
and the 80.0 arc_easy ranking ability lives in these weights. Retraining from
Qwen3-0.6B-Base on a laptop would throw that away (we can afford ~23k item-passes
overnight on an M4 vs the original run's ~186k), and since the automated accuracy
score is 50% of the total, losing it to gain prettier prose is a net loss.

So: dequantize the GGUF back to fp16, LoRA-SFT on top of that, requantize. The
base weights are already on the Q4_0 grid, so re-quantizing them is close to
lossless; only the merged LoRA delta gets newly quantized.

Approach: rather than translate GGUF names to HF names and hope the shapes line
up, we load the real Qwen3-0.6B-Base skeleton and walk ITS state_dict, pulling
the matching GGUF tensor for each parameter and asserting the element count
matches. A naming or transpose mistake then fails loudly instead of silently
producing a scrambled model.

    python scripts/gguf_to_hf.py --gguf model/Qwen3-0.6B-Q4_0.gguf \
        --out output/jamii-hf-recovered
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def hf_to_gguf_name(hf: str) -> str | None:
    """Map an HF parameter name to its GGUF counterpart."""
    if hf == "model.embed_tokens.weight":
        return "token_embd.weight"
    if hf == "model.norm.weight":
        return "output_norm.weight"
    if hf == "lm_head.weight":
        return None  # tied to embed_tokens in Qwen3-0.6B; no separate GGUF tensor

    m = re.match(r"model\.layers\.(\d+)\.(.+)", hf)
    if not m:
        return None
    i, rest = m.group(1), m.group(2)
    table = {
        "self_attn.q_proj.weight": f"blk.{i}.attn_q.weight",
        "self_attn.k_proj.weight": f"blk.{i}.attn_k.weight",
        "self_attn.v_proj.weight": f"blk.{i}.attn_v.weight",
        "self_attn.o_proj.weight": f"blk.{i}.attn_output.weight",
        "self_attn.q_norm.weight": f"blk.{i}.attn_q_norm.weight",
        "self_attn.k_norm.weight": f"blk.{i}.attn_k_norm.weight",
        "mlp.gate_proj.weight": f"blk.{i}.ffn_gate.weight",
        "mlp.up_proj.weight": f"blk.{i}.ffn_up.weight",
        "mlp.down_proj.weight": f"blk.{i}.ffn_down.weight",
        "input_layernorm.weight": f"blk.{i}.attn_norm.weight",
        "post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
    }
    return table.get(rest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=str(ROOT / "model" / "Qwen3-0.6B-Q4_0.gguf"))
    ap.add_argument("--base-model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--out", default=str(ROOT / "output" / "jamii-hf-recovered"))
    args = ap.parse_args()

    import numpy as np
    import torch
    from gguf import GGUFReader
    from gguf.quants import dequantize
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Reading {args.gguf} ...")
    reader = GGUFReader(args.gguf)
    gg = {t.name: t for t in reader.tensors}
    print(f"  {len(gg)} GGUF tensors")

    print(f"Loading HF skeleton {args.base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float32, trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    sd = model.state_dict()
    replaced, skipped, missing = 0, [], []

    for hf_name, param in sd.items():
        gname = hf_to_gguf_name(hf_name)
        if gname is None:
            skipped.append(hf_name)
            continue
        t = gg.get(gname)
        if t is None:
            missing.append((hf_name, gname))
            continue

        arr = dequantize(t.data, t.tensor_type).astype(np.float32)
        if arr.size != param.numel():
            raise SystemExit(
                f"FATAL size mismatch {hf_name} <- {gname}: "
                f"gguf {arr.size} vs hf {param.numel()} (shape {tuple(param.shape)})"
            )
        # GGUF stores dims fastest-first (reverse of HF), so reshape to the HF
        # parameter's own shape rather than trusting the GGUF dim order.
        sd[hf_name] = torch.from_numpy(arr.reshape(tuple(param.shape)))
        replaced += 1

    if missing:
        raise SystemExit(f"FATAL: no GGUF tensor for {missing[:5]} ({len(missing)} total)")

    model.load_state_dict(sd, strict=True)
    model = model.to(torch.float16)

    print(f"  replaced {replaced} tensors; skipped (tied/none): {skipped}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"Saved recovered HF model -> {args.out}")

    # Sanity check: the recovered model must produce coherent text. A transpose or
    # naming error yields fluent-looking garbage or pure noise, so eyeball it.
    print("\n--- sanity generation ---")
    model.eval()
    ids = tok("Question: What is the capital of France?\nAnswer:", return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=12, do_sample=False)
    print(tok.decode(out[0], skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
