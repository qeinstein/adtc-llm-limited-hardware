#!/usr/bin/env python3
"""One-prompt Falcon-H1 generation smoke test on the Kaggle worker.

This deliberately avoids the full evaluation battery. It validates that the
official chat template, Falcon stop-token set, and HF generation loop produce
more than the invalid one-to-three-character outputs seen in the archived
pre-fix pilot. Model weights are loaded only on the remote worker.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoTokenizer
    from transformers.models.falcon_h1.modeling_falcon_h1 import FalconH1ForCausalLM

    prompt = "A child has fast breathing and chest indrawing. What should a community health worker do next?"
    print(json.dumps({"timestamp_utc": stamp(), "event": "smoke_model_load_start", "model": args.model}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FalconH1ForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(json.dumps({"timestamp_utc": stamp(), "event": "smoke_model_load_complete", "device": str(device)}), flush=True)
    messages = [
        {"role": "system", "content": "You are Jamii Afya, an offline medical decision-support assistant. Always surface danger signs and when to refer."},
        {"role": "user", "content": prompt},
    ]
    try:
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
    except TypeError:
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if isinstance(encoded, dict):
        inputs = {key: value.to(device) for key, value in encoded.items()}
        prompt_len = int(inputs["input_ids"].shape[-1])
    else:
        inputs = {"input_ids": encoded.to(device), "attention_mask": torch.ones_like(encoded).to(device)}
        prompt_len = int(encoded.shape[-1])
    configured = getattr(model.generation_config, "eos_token_id", [])
    stop_ids = [configured] if isinstance(configured, int) else list(configured or [])
    if tokenizer.eos_token_id is not None:
        stop_ids.append(int(tokenizer.eos_token_id))
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        stop_ids.append(im_end)
    stop_ids = sorted(set(stop_ids))
    print(json.dumps({"timestamp_utc": stamp(), "event": "smoke_generation_start", "prompt_tokens": prompt_len, "stop_ids": stop_ids}), flush=True)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max(1, args.max_new_tokens),
            do_sample=False,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
    ids = [int(x) for x in output[0][prompt_len:].detach().cpu().tolist()]
    text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    result = {"timestamp_utc": stamp(), "event": "smoke_generation_complete", "prompt": prompt, "new_tokens": len(ids), "token_ids": ids, "visible_chars": len(text), "text": text}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
