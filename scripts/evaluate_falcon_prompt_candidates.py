#!/usr/bin/env python3
"""Evaluate versioned Falcon system prompts on development/validation sets.

The frozen final battery is deliberately not accepted as an input here.  This
script loads stock Falcon once, evaluates every prompt candidate with the same
deterministic decoding settings, preserves raw generations, and reports
prompt-token and response-length costs alongside conservative task checks.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_falcon_battery import rule_result


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prompts(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    values = payload.get("prompts", []) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError(f"{path}: expected prompt list")
    return [item for item in values if isinstance(item, dict)]


def candidates(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    values = payload.get("candidates", []) if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path}: expected non-empty candidates list")
    return [item for item in values if isinstance(item, dict) and item.get("id") and item.get("text")]


def apply_chat(tokenizer: Any, messages: list[dict[str, str]], device: Any) -> Any:
    try:
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
    except TypeError:
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    # Fast tokenizers return a tensor; some versions return BatchEncoding,
    # whose ``to`` method does not make it a tensor and has no ``shape``.
    if isinstance(encoded, dict) or hasattr(encoded, "keys"):
        return encoded["input_ids"].to(device)
    return encoded.to(device)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="tiiuae/Falcon-H1-1.5B-Deep-Instruct")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--development", required=True, type=Path)
    ap.add_argument("--validation", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    candidate_values = candidates(args.candidates.resolve())
    splits = {"development": prompts(args.development.resolve()), "validation": prompts(args.validation.resolve())}

    import torch
    from scripts.train_lora import patch_peft_transformers_compat

    patch_peft_transformers_compat()
    from peft import PeftModel
    from transformers import AutoTokenizer
    from transformers.models.falcon_h1.modeling_falcon_h1 import FalconH1ForCausalLM
    from scripts.falcon_format import generation_stop_ids

    tokenizer_kwargs = {"trust_remote_code": True}
    model_kwargs = {"trust_remote_code": True}
    if args.revision:
        tokenizer_kwargs["revision"] = args.revision
        model_kwargs["revision"] = args.revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    capability = torch.cuda.get_device_capability(0) if device.type == "cuda" else None
    dtype = torch.float32 if capability is not None and capability < (7, 0) else (torch.float16 if device.type == "cuda" else torch.float32)
    model_kwargs["torch_dtype"] = dtype
    model = FalconH1ForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    stop_ids = generation_stop_ids(tokenizer, model.generation_config)
    results: dict[str, Any] = {
        "schema_version": "1.0.0",
        "experiment_id": "falcon-system-prompt-search",
        "model": args.model,
        "revision": args.revision,
        "adapter": args.adapter,
        "candidates_file": str(args.candidates.resolve()),
        "candidates_sha256": sha256_file(args.candidates.resolve()),
        "development_file": str(args.development.resolve()),
        "validation_file": str(args.validation.resolve()),
        "device": str(device),
        "dtype": str(dtype),
        "seed": args.seed,
        "stop_ids": stop_ids,
        "candidates": {},
    }
    for candidate in candidate_values:
        candidate_id = str(candidate["id"])
        candidate_root = out / candidate_id
        candidate_root.mkdir(parents=True, exist_ok=True)
        candidate_result: dict[str, Any] = {"id": candidate_id, "family": candidate.get("family", "unknown"), "text": candidate["text"], "splits": {}}
        for split_name, items in splits.items():
            split_root = candidate_root / split_name
            split_root.mkdir(parents=True, exist_ok=True)
            scored: list[dict[str, Any]] = []
            for item in items:
                prompt_id = str(item["id"])
                user_text = str(item.get("text") or item.get("query") or item.get("instruction") or "")
                if not user_text:
                    raise ValueError(f"{split_name}/{prompt_id}: empty prompt")
                inputs = apply_chat(tokenizer, [{"role": "system", "content": candidate["text"]}, {"role": "user", "content": user_text}], device)
                prompt_len = int(inputs.shape[-1])
                started = time.monotonic()
                with torch.no_grad():
                    generated = model.generate(inputs, max_new_tokens=int(item.get("max_tokens", 96)), do_sample=False, temperature=0.0, eos_token_id=stop_ids or None, pad_token_id=tokenizer.pad_token_id)
                elapsed = time.monotonic() - started
                generated_ids = generated[0, prompt_len:].detach().cpu().tolist()
                text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                (split_root / f"{prompt_id}.txt").write_text(text, encoding="utf-8")
                quality = dict(item.get("quality", {}))
                result = rule_result(prompt_id, text, quality, 1)
                result.update({"prompt_tokens": prompt_len, "new_tokens": len(generated_ids), "elapsed_seconds": round(elapsed, 4), "stopped_on": generated_ids[-1] if generated_ids and generated_ids[-1] in stop_ids else None, "section": item.get("section", "unknown"), "critical": bool(quality.get("critical")), "source_text": user_text})
                scored.append(result)
            passed = sum(int(item["passed"]) for item in scored)
            critical_failures = [item["id"] for item in scored if not item["passed"] and item["critical"]]
            candidate_result["splits"][split_name] = {
                "prompt_count": len(scored),
                "passed_count": passed,
                "failed_count": len(scored) - passed,
                "pass_rate_percent": round(100 * passed / max(1, len(scored)), 3),
                "critical_failures": critical_failures,
                "mean_prompt_tokens": round(sum(item["prompt_tokens"] for item in scored) / max(1, len(scored)), 3),
                "mean_new_tokens": round(sum(item["new_tokens"] for item in scored) / max(1, len(scored)), 3),
                "cap_rate_percent": round(100 * sum(item["stopped_on"] is None for item in scored) / max(1, len(scored)), 3),
                "mean_latency_seconds": round(sum(item["elapsed_seconds"] for item in scored) / max(1, len(scored)), 3),
                "raw_dir": str(split_root),
                "items": scored,
            }
        validation = candidate_result["splits"]["validation"]
        candidate_result["validation_selection_eligible"] = not validation["critical_failures"] and validation["pass_rate_percent"] >= 50.0
        (candidate_root / "candidate_summary.json").write_text(json.dumps(candidate_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        results["candidates"][candidate_id] = candidate_result
        print(json.dumps({"timestamp_utc": stamp(), "event": "prompt_candidate_complete", "candidate": candidate_id, "development_pass_rate": candidate_result["splits"]["development"]["pass_rate_percent"], "validation_pass_rate": validation["pass_rate_percent"], "validation_critical_failures": validation["critical_failures"]}, ensure_ascii=False), flush=True)
    eligible = [value for value in results["candidates"].values() if value["validation_selection_eligible"]]
    results["selected_candidate"] = max(eligible, key=lambda value: (value["splits"]["validation"]["pass_rate_percent"], -value["splits"]["validation"]["mean_prompt_tokens"]))["id"] if eligible else None
    results["completed_utc"] = stamp()
    (out / "prompt_search_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"timestamp_utc": stamp(), "event": "prompt_search_complete", "candidate_count": len(candidate_values), "selected_candidate": results["selected_candidate"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
