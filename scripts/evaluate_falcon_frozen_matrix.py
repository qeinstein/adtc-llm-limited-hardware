#!/usr/bin/env python3
"""Evaluate prompt/weight combinations on a frozen Falcon battery.

Prompt selection must happen on the separate development/validation batteries
first. This command has no selection logic: it measures every supplied prompt
for one weight variant and preserves raw responses, timing, token cost, and
the conservative frozen-battery rubric result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    values = payload.get("candidates", []) if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path}: expected a non-empty candidate list")
    return [item for item in values if isinstance(item, dict) and item.get("id") and item.get("text")]


def load_battery(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    values = payload.get("prompts", []) if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path}: expected a non-empty prompt list")
    return [item for item in values if isinstance(item, dict) and item.get("id")]


def apply_chat(tokenizer: Any, messages: list[dict[str, str]], device: Any) -> Any:
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
    if isinstance(encoded, dict):
        return {key: value.to(device) for key, value in encoded.items()}
    return {"input_ids": encoded.to(device), "attention_mask": encoded.new_ones(encoded.shape)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="tiiuae/Falcon-H1-1.5B-Deep-Instruct")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--battery", required=True, type=Path)
    ap.add_argument("--rubric", default=ROOT / "docs/research/falcon_generation_rubric.json", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    candidates = load_candidates(args.candidates.resolve())
    battery = load_battery(args.battery.resolve())
    rubric = load_json(args.rubric.resolve())
    rules = rubric.get("rules", {})
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    events = out / "matrix_events.jsonl"

    def emit(name: str, **fields: Any) -> None:
        record = {"timestamp_utc": stamp(), "event": name, **fields}
        print(json.dumps(record, ensure_ascii=False), flush=True)
        with events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    emit("matrix_start", model=args.model, adapter=args.adapter, candidates=len(candidates), battery=len(battery))
    import torch
    from scripts.train_lora import patch_peft_transformers_compat

    patch_peft_transformers_compat()
    from peft import PeftModel
    from transformers import AutoTokenizer
    from transformers.models.falcon_h1.modeling_falcon_h1 import FalconH1ForCausalLM
    from scripts.falcon_format import generation_stop_ids
    from scripts.score_falcon_battery import rule_result

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
    emit("model_load_start", device=str(device), dtype=str(dtype), capability=capability)
    model = FalconH1ForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(Path(args.adapter).resolve()), is_trainable=False)
    model.eval()
    stop_ids = generation_stop_ids(tokenizer, model.generation_config)
    emit("model_load_complete", stop_ids=stop_ids)

    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "experiment_id": "falcon-frozen-prompt-weight-matrix",
        "variant": "adapter" if args.adapter else "stock",
        "model": args.model,
        "revision": args.revision,
        "adapter": args.adapter,
        "device": str(device),
        "dtype": str(dtype),
        "seed": args.seed,
        "candidates_file": str(args.candidates.resolve()),
        "candidates_sha256": sha256_file(args.candidates.resolve()),
        "battery_file": str(args.battery.resolve()),
        "battery_sha256": sha256_file(args.battery.resolve()),
        "rubric_file": str(args.rubric.resolve()),
        "stop_ids": stop_ids,
        "candidates": {},
    }
    minimum_chars = int(rubric.get("minimum_nonempty_chars", 1))
    for candidate in candidates:
        candidate_id = str(candidate["id"])
        candidate_dir = out / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        scored: list[dict[str, Any]] = []
        for index, item in enumerate(battery, 1):
            prompt_id = str(item["id"])
            text = str(item.get("text") or item.get("query") or item.get("instruction") or "")
            if not text:
                raise ValueError(f"{args.battery}/{prompt_id}: empty prompt")
            inputs = apply_chat(tokenizer, [{"role": "system", "content": str(candidate["text"])}, {"role": "user", "content": text}], device)
            prompt_len = int(inputs["input_ids"].shape[-1])
            started = time.monotonic()
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=int(item.get("max_tokens", 256)),
                    do_sample=False,
                    eos_token_id=stop_ids or None,
                    pad_token_id=tokenizer.pad_token_id,
                )
            elapsed = time.monotonic() - started
            generated_ids = generated[0, prompt_len:].detach().cpu().tolist()
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            (candidate_dir / f"{prompt_id}.txt").write_text(output_text, encoding="utf-8")
            quality = rule_result(prompt_id, output_text, rules.get(prompt_id, {}), minimum_chars)
            quality.update({
                "prompt_tokens": prompt_len,
                "new_tokens": len(generated_ids),
                "elapsed_seconds": round(elapsed, 4),
                "stopped_on": generated_ids[-1] if generated_ids and generated_ids[-1] in stop_ids else None,
                "section": item.get("section", "unknown"),
                "check": item.get("check", ""),
                "source_text": text,
            })
            scored.append(quality)
            emit("matrix_item", candidate=candidate_id, id=prompt_id, index=index, total=len(battery), passed=quality["passed"], new_tokens=len(generated_ids))
        critical_sections = set(str(value) for value in rubric.get("critical_sections", []))
        critical_failures = [item["id"] for item in scored if not item["passed"] and item.get("section") in critical_sections]
        passed = sum(int(item["passed"]) for item in scored)
        summary = {
            "id": candidate_id,
            "family": candidate.get("family", "unknown"),
            "text": candidate["text"],
            "prompt_tokens": len(tokenizer(str(candidate["text"]), add_special_tokens=False)["input_ids"]),
            "prompt_sha256": hashlib.sha256(str(candidate["text"]).encode("utf-8")).hexdigest(),
            "prompt_count": len(scored),
            "passed_count": passed,
            "failed_count": len(scored) - passed,
            "pass_rate_percent": round(100 * passed / max(1, len(scored)), 3),
            "critical_failures": critical_failures,
            "promotion_eligible": not critical_failures and len(scored) == len(battery),
            "mean_prompt_tokens_in_chat": round(sum(item["prompt_tokens"] for item in scored) / max(1, len(scored)), 3),
            "mean_new_tokens": round(sum(item["new_tokens"] for item in scored) / max(1, len(scored)), 3),
            "mean_latency_seconds": round(sum(item["elapsed_seconds"] for item in scored) / max(1, len(scored)), 3),
            "cap_rate_percent": round(100 * sum(item["stopped_on"] is None for item in scored) / max(1, len(scored)), 3),
            "raw_dir": str(candidate_dir),
            "items": scored,
        }
        result["candidates"][candidate_id] = summary
        emit("candidate_complete", candidate=candidate_id, pass_rate_percent=summary["pass_rate_percent"], critical_failures=critical_failures)
    result["completed_utc"] = stamp()
    (out / "frozen_matrix.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    emit("matrix_complete", output=str(out / "frozen_matrix.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
