#!/usr/bin/env python3
"""Deliberately overfit a tiny Falcon set and prove behavioral movement.

This is a pipeline diagnostic, not a quality run.  It uses a deterministic
mixture of short SFT and MCQA examples, a high learning rate, and a small
number of steps.  The output is a compact manifest plus raw generations that
prove whether the adapter changes logits and decoded behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def set_seed(seed: int) -> None:
    random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tiny_rows() -> list[dict[str, Any]]:
    # These are intentionally unique and short: a memorization test should not
    # be confused with an in-domain quality claim.
    sft = [
        ("tp-s01", "What is the safe action for severe chest indrawing?", "Treat it as a danger sign and refer urgently."),
        ("tp-s02", "Reply with the word BLUE.", "BLUE"),
        ("tp-s03", "What should you do when a dose is missing from the label?", "Do not guess; check the label or ask a pharmacist."),
        ("tp-s04", "Answer in Kiswahili: what is the safe first step?", "Usibashiri; mpeleke mgonjwa kituo cha afya haraka."),
        ("tp-s05", "What must never be swallowed as a disinfectant?", "Never swallow bleach or any cleaning chemical."),
        ("tp-s06", "Give a concise disposition for an unconscious patient.", "This is an emergency; arrange immediate referral."),
        ("tp-s07", "Can a layperson perform field amputation?", "No. Control bleeding and evacuate urgently; do not amputate."),
        ("tp-s08", "What is the answer to 2 plus 3?", "5"),
    ]
    mcqa = [
        {"example_id": "tp-m01", "format": "mcqa", "context": "The capital of France is", "choices": [" Paris", " Nairobi", "Lima", "Accra"], "gold": 0},
        {"example_id": "tp-m02", "format": "mcqa", "context": "A child with chest indrawing needs", "choices": ["urgent referral", "routine observation", "bleach", "no action"], "gold": 0},
        {"example_id": "tp-m03", "format": "mcqa", "context": "Five minus two equals", "choices": ["three", "four", "six", "eight"], "gold": 0},
        {"example_id": "tp-m04", "format": "mcqa", "context": "ORS is used to treat", "choices": ["dehydration", "fracture", "blindness", "amputation"], "gold": 0},
    ]
    rows = [{"example_id": i, "format": "sft", "instruction": q, "input": "", "output": a} for i, q, a in sft]
    rows.extend(mcqa)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="tiiuae/Falcon-H1-1.5B-Deep-Instruct")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args(argv)
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    events = out_dir / "events.jsonl"
    metrics = out_dir / "training_metrics.jsonl"
    set_seed(args.seed)
    print(json.dumps({"timestamp_utc": stamp(), "event": "probe_start", "model": args.model, "steps": args.steps, "learning_rate": args.learning_rate}, ensure_ascii=False), flush=True)

    import torch
    from scripts.train_lora import patch_peft_transformers_compat

    patch_peft_transformers_compat()
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer
    from transformers.models.falcon_h1.modeling_falcon_h1 import FalconH1ForCausalLM
    from scripts.falcon_format import render_completion, generation_stop_ids

    revision = args.revision
    tokenizer_kwargs = {"trust_remote_code": True}
    model_kwargs = {"trust_remote_code": True}
    if revision:
        tokenizer_kwargs["revision"] = revision
        model_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    capability = torch.cuda.get_device_capability(0) if device.type == "cuda" else None
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if capability is not None and capability >= (8, 0):
        dtype = torch.bfloat16
    model_kwargs["torch_dtype"] = dtype
    print(json.dumps({"timestamp_utc": stamp(), "event": "model_load_start", "device": str(device), "dtype": str(dtype), "capability": capability}), flush=True)
    base = FalconH1ForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    base.config.use_cache = False
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    linear_suffixes = {name.rsplit(".", 1)[-1] for name, module in base.named_modules() if isinstance(module, torch.nn.Linear)}
    missing = [name for name in targets if name not in linear_suffixes]
    if missing:
        raise RuntimeError(f"missing configured LoRA targets: {missing}")
    model = get_peft_model(base, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=targets))
    model.train()
    trainable_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    initial_state = {name: parameter.detach().float().cpu().clone() for name, parameter in trainable_parameters}
    rows = tiny_rows()
    from scripts.train_falcon_production import render_sft

    prepared: list[dict[str, Any]] = []
    system = "You are Jamii Afya. Answer concisely and safely."
    for row in rows:
        if row["format"] == "sft":
            prompt, target = render_sft(tokenizer, row, system)
            if len(prompt) + len(target) > args.max_length:
                raise ValueError(f"tiny SFT row exceeds max length: {row['example_id']}")
            prepared.append({**row, "prompt_ids": prompt, "target_ids": target, "input_ids": prompt + target})
        else:
            context = list(tokenizer(row["context"], add_special_tokens=False)["input_ids"])
            choices = [list(tokenizer(choice, add_special_tokens=False)["input_ids"]) for choice in row["choices"]]
            if any(len(context) + len(choice) > args.max_length for choice in choices):
                raise ValueError(f"tiny MCQA row exceeds max length: {row['example_id']}")
            prepared.append({**row, "context_ids": context, "choice_ids": choices})
    train_rows = [row for row in prepared if row["format"] == "sft"]
    mcqa_rows = [row for row in prepared if row["format"] == "mcqa"]
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable_parameters], lr=args.learning_rate)
    append_jsonl(events, {"timestamp_utc": stamp(), "event": "model_ready", "device": str(device), "dtype": str(dtype), "targets": targets, "trainable_parameters": sum(p.numel() for _, p in trainable_parameters), "rows": len(prepared), "sft_rows": len(train_rows), "mcqa_rows": len(mcqa_rows)})

    def sft_loss(row: dict[str, Any]) -> torch.Tensor:
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
        labels = torch.tensor([[-100] * len(row["prompt_ids"]) + row["target_ids"]], dtype=torch.long, device=device)
        result = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
        return result.loss

    def mcqa_loss(row: dict[str, Any]) -> torch.Tensor:
        sequences = [row["context_ids"] + choice for choice in row["choice_ids"]]
        width = max(len(sequence) for sequence in sequences)
        ids = torch.full((len(sequences), width), int(tokenizer.pad_token_id), dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for index, sequence in enumerate(sequences):
            ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
            mask[index, :len(sequence)] = 1
        logits = model(input_ids=ids, attention_mask=mask).logits.float()
        scores = []
        context_len = len(row["context_ids"])
        for choice_index, choice in enumerate(row["choice_ids"]):
            positions = torch.arange(context_len - 1, context_len + len(choice) - 1, device=device)
            token_logits = logits[choice_index, positions]
            token_logp = token_logits.log_softmax(dim=-1).gather(1, torch.tensor(choice, device=device).unsqueeze(1)).squeeze(1)
            scores.append(token_logp.sum() / max(1, len(row["choices"][choice_index])))
        return -torch.stack(scores).log_softmax(dim=0)[row["gold"]]

    probe_row = train_rows[0]
    probe_ids = torch.tensor([probe_row["input_ids"]], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        initial_logits = model(input_ids=probe_ids, attention_mask=torch.ones_like(probe_ids)).logits.float()
    model.train()

    for step in range(1, args.steps + 1):
        row = prepared[(step - 1) % len(prepared)]
        optimizer.zero_grad(set_to_none=True)
        loss = sft_loss(row) if row["format"] == "sft" else mcqa_loss(row)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable_parameters], 1.0).detach().cpu())
        if not math.isfinite(grad_norm) or grad_norm <= 0:
            raise FloatingPointError(f"invalid gradient norm at step {step}: {grad_norm}")
        optimizer.step()
        record = {"timestamp_utc": stamp(), "event": "train_step", "step": step, "example_id": row["example_id"], "format": row["format"], "loss": float(loss.detach().cpu()), "grad_norm": grad_norm, "learning_rate": args.learning_rate}
        append_jsonl(metrics, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    model.eval()
    delta_sq = 0.0
    initial_sq = 0.0
    changed_tensors = 0
    for name, parameter in trainable_parameters:
        current = parameter.detach().float().cpu()
        before = initial_state[name]
        delta = current - before
        delta_sq += float((delta * delta).sum())
        initial_sq += float((before * before).sum())
        changed_tensors += int(bool(torch.any(delta != 0)))
    adapter_delta_l2 = math.sqrt(delta_sq)
    adapter_initial_l2 = math.sqrt(initial_sq)

    with torch.no_grad():
        adapter_logits = model(input_ids=probe_ids, attention_mask=torch.ones_like(probe_ids)).logits.float()
        model.disable_adapter_layers()
        stock_logits = model(input_ids=probe_ids, attention_mask=torch.ones_like(probe_ids)).logits.float()
        model.enable_adapter_layers()
        restored_logits = model(input_ids=probe_ids, attention_mask=torch.ones_like(probe_ids)).logits.float()
    logit_delta = adapter_logits - stock_logits
    adapter_reload_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_reload_dir))
    tokenizer.save_pretrained(str(adapter_reload_dir))
    reloaded = FalconH1ForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    reloaded.config.use_cache = False
    reloaded = PeftModel.from_pretrained(reloaded, str(adapter_reload_dir), is_trainable=False)
    reloaded.eval()
    with torch.no_grad():
        reload_logits = reloaded(input_ids=probe_ids, attention_mask=torch.ones_like(probe_ids)).logits.float()
        merged = reloaded.merge_and_unload()
        merged.eval()
        merged_logits = merged(input_ids=probe_ids, attention_mask=torch.ones_like(probe_ids)).logits.float()
    stop_ids = generation_stop_ids(tokenizer, model.generation_config)

    def generate_text(active_model: Any, row: dict[str, Any]) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": row["instruction"]}]
        try:
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        except TypeError:
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        encoded = encoded.to(device)
        with torch.no_grad():
            generated = active_model.generate(encoded, max_new_tokens=24, do_sample=False, temperature=0.0, eos_token_id=stop_ids or None, pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(generated[0, encoded.shape[-1]:], skip_special_tokens=True)

    generations = []
    for row in train_rows[:4]:
        model.disable_adapter_layers()
        stock_text = generate_text(model, row)
        model.enable_adapter_layers()
        adapter_text = generate_text(model, row)
        generations.append({"id": row["example_id"], "stock": stock_text, "adapter": adapter_text, "changed": stock_text != adapter_text})
    write_json(out_dir / "generations.json", generations)
    manifest = {
        "schema_version": "1.0.0",
        "experiment_id": "falcon-trainability-proof",
        "timestamp_utc": stamp(),
        "model": args.model,
        "revision": revision,
        "seed": args.seed,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": 0.0, "target_modules": targets},
        "device": str(device),
        "dtype": str(dtype),
        "rows": {"total": len(prepared), "sft": len(train_rows), "mcqa": len(mcqa_rows)},
        "trainable_parameters": sum(p.numel() for _, p in trainable_parameters),
        "adapter_tensors_changed": changed_tensors,
        "adapter_delta_l2": adapter_delta_l2,
        "adapter_initial_l2": adapter_initial_l2,
        "adapter_delta_relative": adapter_delta_l2 / max(adapter_initial_l2, 1e-12),
        "probe_logit_delta_max_abs": float(logit_delta.abs().max().cpu()),
        "probe_logit_delta_rms": float(logit_delta.pow(2).mean().sqrt().cpu()),
        "adapter_off_matches_initial_stock_max_abs": float((stock_logits - initial_logits).abs().max().cpu()),
        "adapter_on_restored_max_abs": float((adapter_logits - restored_logits).abs().max().cpu()),
        "reloaded_adapter_max_abs": float((adapter_logits - reload_logits).abs().max().cpu()),
        "merged_vs_unmerged_max_abs": float((adapter_logits - merged_logits).abs().max().cpu()),
        "changed_generation_count": sum(int(item["changed"]) for item in generations),
        "generation_count": len(generations),
        "adapter_dir": str(adapter_reload_dir),
        "status": "pass" if adapter_delta_l2 > 0 and float(logit_delta.abs().max().cpu()) > 1e-6 and any(item["changed"] for item in generations) else "fail",
    }
    write_json(out_dir / "trainability_manifest.json", manifest)
    print(json.dumps({"timestamp_utc": stamp(), "event": "probe_complete", **manifest}, ensure_ascii=False), flush=True)
    return 0 if manifest["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
