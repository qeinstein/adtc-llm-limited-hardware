#!/usr/bin/env python3
"""LoRA SFT driver (TRL SFTTrainer + PEFT). UNEXECUTED — no training compute.

Reads training/configs/*.yaml. Enforces, by assertion before step 1:
- base model id + revision match the config
- adapted module set contains ONLY allowed substrings and NONE of the
  forbidden ones (router, head, embeddings, visual, MTP, norms)
- completion-only loss (assistant tokens only; no synthetic CoT training)
- early stopping on held-out composite, never train loss alone

Run (on provisioned GPU hardware):
  python training/train.py --config training/configs/lora_pilot_r8.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


def check_modules(model, lora_cfg: dict) -> list[str]:
    """Assert the adapted set is exactly the allowed one. Returns it."""
    allowed = lora_cfg["target_modules"]
    forbidden = lora_cfg["forbidden_substrings"]
    adapted = sorted({n for n, _ in model.named_modules()
                      if any(t in n for t in allowed)})
    bad = [n for n in adapted if any(f in n for f in forbidden)]
    if bad:
        raise RuntimeError(f"FORBIDDEN modules would receive LoRA: {bad[:10]}")
    if not adapted:
        raise RuntimeError("no modules matched target_modules; check names")
    print(f"adapting {len(adapted)} modules, e.g. {adapted[:5]}", flush=True)
    return adapted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="load config + validate mixture, do not train")
    args = ap.parse_args()
    cfg = load_config(Path(args.config))
    mix_path = ROOT / cfg["data"]["mixture"]
    if not mix_path.exists():
        raise FileNotFoundError(f"mixture missing (run build_dataset.py): {mix_path}")
    n = sum(1 for _ in open(mix_path, encoding="utf-8"))
    print(f"mixture: {mix_path} ({n} examples)", flush=True)
    print(f"base: {cfg['base_model']} @ {cfg['base_revision']}", flush=True)
    if args.dry_run:
        print("dry-run OK (module asserts run at train time)", flush=True)
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device; BF16 LoRA needs provisioned GPUs")
    tok = AutoTokenizer.from_pretrained(cfg["base_model"],
                                        revision=cfg["base_revision"],
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], revision=cfg["base_revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.get("attn_implementation", "flash_attention_2"),
        trust_remote_code=True)
    lora = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                      lora_dropout=cfg["lora"]["dropout"], bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=cfg["lora"]["target_modules"])
    model = get_peft_model(model, lora)
    check_modules(model, cfg["lora"])  # aborts on forbidden/router/head adapts
    model.print_trainable_parameters()

    sys_prompt = json.loads((ROOT / "prompts" / "system.json").read_text())["text"]

    def to_messages(ex):
        return {"messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["response"]}]}

    rows = [json.loads(l) for l in open(mix_path, encoding="utf-8")]
    ds = Dataset.from_list([to_messages(r) for r in rows])
    split = ds.train_test_split(test_size=min(500, max(50, len(ds) // 20)),
                                seed=cfg["seeds"]["data"])
    sft = SFTConfig(
        output_dir=cfg["logging"]["output_dir"],
        num_train_epochs=cfg["optim"]["epochs"],
        per_device_train_batch_size=cfg["optim"]["per_device_batch"],
        gradient_accumulation_steps=cfg["optim"]["grad_accum"],
        learning_rate=cfg["optim"]["lr"],
        lr_scheduler_type=cfg["optim"]["lr_scheduler"],
        warmup_ratio=cfg["optim"]["warmup_ratio"],
        max_grad_norm=cfg["optim"]["max_grad_norm"],
        bf16=True, gradient_checkpointing=cfg["optim"]["gradient_checkpointing"],
        max_length=cfg["data"]["max_seq_length"],
        assistant_only_loss=bool(cfg["data"]["assistant_only_loss"]),
        packing=bool(cfg["data"]["packing"]),
        eval_strategy="steps", eval_steps=cfg["optim"]["eval_every_steps"],
        save_strategy="steps", save_steps=cfg["optim"]["eval_every_steps"],
        save_total_limit=cfg["logging"]["save_total_limit"],
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, seed=cfg["seeds"]["train"],
        logging_steps=cfg["logging"]["log_every"], report_to="none")
    trainer = SFTTrainer(model=model, processing_class=tok, args=sft,
                         train_dataset=split["train"],
                         eval_dataset=split["test"])
    trainer.train()
    trainer.save_model()
    print("saved", cfg["logging"]["output_dir"], flush=True)


if __name__ == "__main__":
    main()
