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
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
    from scripts.falcon_format import generation_stop_ids

    prompt = "A child has fast breathing and chest indrawing. What should a community health worker do next?"
    print(json.dumps({"timestamp_utc": stamp(), "event": "smoke_model_load_start", "model": args.model}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    dtype = torch.float32 if capability is not None and capability < (7, 0) else (torch.float16 if torch.cuda.is_available() else torch.float32)
    model = FalconH1ForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(json.dumps({"timestamp_utc": stamp(), "event": "smoke_model_load_complete", "device": str(device), "cuda_capability": capability, "dtype": str(dtype)}), flush=True)
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
    stop_ids = generation_stop_ids(tokenizer, model.generation_config)
    print(json.dumps({"timestamp_utc": stamp(), "event": "smoke_generation_start", "prompt_tokens": prompt_len, "stop_ids": stop_ids, "tokenizer_eos_token_id": tokenizer.eos_token_id, "tokenizer_pad_token_id": tokenizer.pad_token_id, "model_eos_token_id": getattr(model.generation_config, "eos_token_id", None), "model_pad_token_id": getattr(model.generation_config, "pad_token_id", None)}), flush=True)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max(1, args.max_new_tokens),
            do_sample=False,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    raw_ids = [int(x) for x in output.sequences[0][prompt_len:].detach().cpu().tolist()]
    score_steps = len(output.scores)
    ids = raw_ids[:score_steps] if score_steps else raw_ids
    topk = []
    for scores in output.scores:
        values, indices = scores[0].float().topk(min(5, scores.shape[-1]))
        topk.append({"ids": [int(x) for x in indices.cpu().tolist()], "scores": [round(float(x), 4) for x in values.cpu().tolist()]})
    text = tokenizer.decode(torch.tensor(ids, device=output.sequences.device), skip_special_tokens=True)
    stopped_on = ids[-1] if ids and ids[-1] in stop_ids else None
    result = {"timestamp_utc": stamp(), "event": "smoke_generation_complete", "prompt": prompt, "new_tokens": len(ids), "raw_new_tokens": len(raw_ids), "score_steps": score_steps, "token_ids": ids, "raw_token_ids": raw_ids, "stopped_on": stopped_on, "topk_each_step": topk, "visible_chars": len(text), "text": text}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
