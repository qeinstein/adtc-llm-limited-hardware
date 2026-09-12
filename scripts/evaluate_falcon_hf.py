#!/usr/bin/env python3
"""Evaluate a Falcon HF base+adapter checkpoint on the normalized dev set.

This is the fast between-stage evaluator. It intentionally evaluates the same
tokenized SFT/MCQA representation used by the production trainer and writes raw
generation outputs for the frozen batteries. The final deployment gate still
uses ``evaluate_falcon_candidate.py`` on the merged/quantized GGUF.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit(path: Path, event: str, **fields: Any) -> None:
    record = {"timestamp_utc": stamp(), "event": event, **fields}
    print(json.dumps(record, ensure_ascii=False), flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/falcon-production-v1.json"))
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory; omit to evaluate stock base")
    ap.add_argument("--max-dev", type=int, default=0)
    ap.add_argument("--battery", action="append", default=[])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    events = out_dir / "eval_events.jsonl"
    random.seed(args.seed)

    rows = load_rows(data_dir / "dev.jsonl")
    if args.max_dev:
        rows = rows[: args.max_dev]
    emit(events, "evaluation_start", rows=len(rows), adapter=args.adapter, model=config["model"])

    import torch
    from scripts.train_lora import patch_peft_transformers_compat

    patch_peft_transformers_compat()
    from peft import PeftModel
    from transformers import AutoTokenizer
    from transformers.models.falcon_h1.modeling_falcon_h1 import FalconH1ForCausalLM

    model_id = config["model"]["id"]
    revision = config["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=config["model"].get("tokenizer_revision", revision), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    emit(events, "model_load_start", device=str(device), dtype=str(dtype))
    model = FalconH1ForCausalLM.from_pretrained(model_id, revision=revision, trust_remote_code=True, torch_dtype=dtype)
    if args.adapter:
        adapter = Path(args.adapter).resolve()
        if not adapter.is_dir():
            raise FileNotFoundError(adapter)
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.to(device)
    model.eval()
    emit(events, "model_load_complete")

    max_len = int(config["data"]["max_length"])
    system = config["data"]["system_prompt"]
    sft_losses: list[float] = []
    mcqa_correct = 0
    mcqa_norm_correct = 0
    mcqa_total = 0

    def sft_item(row: dict[str, Any]) -> float:
        from scripts.train_falcon_production import render_sft

        prompt, target = render_sft(tokenizer, row, system)
        if len(target) > max_len:
            raise ValueError(f"{row.get('example_id')}: target exceeds max_length")
        prompt = prompt[-max(0, max_len - len(target)):]
        ids = prompt + target
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention = torch.ones_like(input_ids)
        labels = torch.tensor([[-100] * len(prompt) + target], dtype=torch.long, device=device)
        with torch.no_grad():
            return float(model(input_ids=input_ids, attention_mask=attention, labels=labels).loss.detach().cpu())

    def mcqa_item(row: dict[str, Any]) -> tuple[bool, bool]:
        context = str(row["context"])
        ctx = list(tokenizer(context, add_special_tokens=False)["input_ids"])
        scores: list[float] = []
        for choice in row["choices"]:
            continuation = list(tokenizer(" " + str(choice), add_special_tokens=False)["input_ids"])
            ids = ctx + continuation
            if not continuation or len(ids) > max_len:
                raise ValueError(f"{row.get('example_id')}: invalid MCQA continuation")
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits[0].float()
            logp = logits[:-1].log_softmax(dim=-1)
            score = sum(float(logp[len(ctx) + i - 1, token].detach().cpu()) for i, token in enumerate(continuation))
            scores.append(score)
        normalized = [score / max(1, len(str(choice))) for score, choice in zip(scores, row["choices"])]
        gold = int(row["gold"])
        return max(range(len(scores)), key=scores.__getitem__) == gold, max(range(len(normalized)), key=normalized.__getitem__) == gold

    emit(events, "dev_start", rows=len(rows))
    for index, row in enumerate(rows, 1):
        if row.get("format") == "sft":
            sft_losses.append(sft_item(row))
        elif row.get("format") == "mcqa":
            correct, normalized = mcqa_item(row)
            mcqa_correct += int(correct)
            mcqa_norm_correct += int(normalized)
            mcqa_total += 1
        else:
            raise ValueError(f"unsupported normalized row format: {row.get('format')!r}")
        if index % 25 == 0 or index == len(rows):
            emit(events, "dev_progress", completed=index, total=len(rows))

    metrics: dict[str, Any] = {
        "sft_dev_loss": round(sum(sft_losses) / max(1, len(sft_losses)), 6),
        "sft_dev_n": len(sft_losses),
        "mcqa_n": mcqa_total,
        "mcqa_acc": round(100 * mcqa_correct / max(1, mcqa_total), 4),
        "mcqa_acc_norm": round(100 * mcqa_norm_correct / max(1, mcqa_total), 4),
    }
    emit(events, "dev_complete", **metrics)

    for battery_path in args.battery:
        battery = Path(battery_path).resolve()
        payload = json.loads(battery.read_text(encoding="utf-8"))
        prompts = payload.get("prompts", []) if isinstance(payload, dict) else payload
        destination = out_dir / battery.stem
        destination.mkdir(parents=True, exist_ok=True)
        emit(events, "generation_start", battery=str(battery), count=len(prompts))
        index: list[dict[str, Any]] = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": str(prompt["text"])},
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
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=min(args.max_new_tokens, int(prompt.get("max_tokens", args.max_new_tokens))), do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
            (destination / f"{prompt['id']}.txt").write_text(text, encoding="utf-8")
            index.append({"id": prompt["id"], "chars": len(text), "words": len(text.split()), "section": prompt.get("section", ""), "check": prompt.get("check", "")})
            emit(events, "generation_item", battery=battery.stem, id=prompt["id"], chars=len(text))
        (destination / "_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        emit(events, "generation_complete", battery=battery.stem, count=len(index))

    metrics["adapter"] = args.adapter
    metrics["rows"] = len(rows)
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    emit(events, "evaluation_complete", metrics=str(out_dir / "eval_metrics.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
