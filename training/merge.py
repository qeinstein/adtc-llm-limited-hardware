#!/usr/bin/env python3
"""Merge a LoRA adapter into BF16 base weights. UNEXECUTED (no checkpoint).

  python training/merge.py --adapter training/runs/full_r8/checkpoint-BEST \
      --out models/jamii-qwen36-sft-bf16

Verifies: base revision, adapter config hash, merged weight count vs base
index (1045 tensors), output SHA manifest. Never touches the deployment
GGUF (see export_final.sh for the separate, gated transform).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(8 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ad = Path(args.adapter)
    cfg = json.loads((ad / "adapter_config.json").read_text())
    base, rev = cfg["base_model_name_or_path"], None
    print("base:", base, flush=True)
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                             trust_remote_code=True),
        str(ad))
    merged = model.merge_and_unload()
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    tok.save_pretrained(out)
    manifest = {"adapter": str(ad),
                "adapter_config_sha256": sha256(ad / "adapter_config.json"),
                "files": {p.name: sha256(p) for p in sorted(out.glob("*.safetensors"))}}
    (out / "merge_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("merged ->", out, flush=True)


if __name__ == "__main__":
    main()
