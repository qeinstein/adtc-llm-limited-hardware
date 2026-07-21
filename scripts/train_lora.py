#!/usr/bin/env python3
"""QLoRA fine-tune for Jamii Afya (default base: Qwen3-1.7B, Apache-2.0).

Two-part SFT, unified as {prompt, completion} with COMPLETION-ONLY loss:

1. Accuracy (the 50% lever): public MCQA TRAIN splits in the EXACT lm-eval
   completion prompt shapes (output/accuracy_sft.jsonl from build_accuracy_sft.py).
   The grader scores MCQ via loglikelihood on the raw GGUF with NO chat template,
   so we train template-free completion format to match it. Legit: train splits
   only, never test/validation (see the ADTC rules — fine-tuning is encouraged and
   there is no anti-contamination clause).

2. Product/judges: our bilingual EN/SW clinical data (data/medical_lora_dataset.json)
   rendered through the model's CHAT template (thinking disabled), so the shipped
   model is also a good interactive advisor for the qualitative judging.

Runs on a single modest GPU (Udutech). Then merge + quantize with export_gguf.sh.

    python scripts/build_accuracy_sft.py
    python scripts/prepare_dataset.py
    python scripts/train_lora.py --base_model Qwen/Qwen3-1.7B --output_dir output/jamii-lora
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tune (accuracy MCQA + clinical chat)")
    # BASE (not instruct): the profiler scores loglikelihood MCQ template-free,
    # where base models beat instruct. Our MCQ-completion SFT then trains the exact
    # scored task on public train splits. (Size may switch to 0.6B/4B after the
    # scalar-tps + accuracy sweep — see REPORT.)
    p.add_argument("--base_model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--accuracy_file", default=str(ROOT / "output" / "accuracy_sft.jsonl"))
    p.add_argument("--clinical_file", default=str(ROOT / "data" / "medical_lora_dataset.json"))
    p.add_argument("--output_dir", default=str(ROOT / "output" / "jamii-lora"))
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)  # LoRA adapters use a higher LR
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--clinical_repeat", type=int, default=4,
                   help="Upsample the (small) clinical set so it isn't drowned by MCQA")
    return p.parse_args()


SYSTEM = (
    "You are Jamii Afya, an offline medical decision-support assistant for community "
    "health workers in rural African clinics. Answer in the question's language "
    "(English or Kiswahili). Always surface danger signs and when to refer."
)


def build_rows(tokenizer, accuracy_file: str, clinical_file: str, clinical_repeat: int) -> list[dict]:
    rows: list[dict] = []

    # 1. MCQA completion rows (bare, template-free — matches lm-eval scoring).
    ap = Path(accuracy_file)
    if ap.exists():
        for line in ap.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append({"prompt": r["prompt"], "completion": r["completion"]})
    mcqa_n = len(rows)

    # 2. Clinical chat rows rendered through the model's chat template.
    cp = Path(clinical_file)
    clin = 0
    if cp.exists():
        for item in json.loads(cp.read_text(encoding="utf-8")):
            instr = (item.get("instruction") or "").strip()
            out = (item.get("output") or "").strip()
            if not instr or not out:
                continue
            inp = (item.get("input") or "").strip()
            user = f"{instr}\n\n{inp}" if inp else instr
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for _ in range(max(1, clinical_repeat)):
                rows.append({"prompt": prompt, "completion": out})
                clin += 1
    print(f"Dataset: {mcqa_n} MCQA(completion) + {clin} clinical(chat) = {len(rows)} rows")
    return rows


def main() -> int:
    args = parse_args()
    print("=" * 70)
    print(f"  Jamii Afya QLoRA — base {args.base_model}")
    print("=" * 70)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    print(f"CUDA: {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = build_rows(tokenizer, args.accuracy_file, args.clinical_file, args.clinical_repeat)
    if not rows:
        print("ERROR: no training rows. Run build_accuracy_sft.py and/or check clinical_file.")
        return 1
    dataset = Dataset.from_list(rows).shuffle(seed=3407)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    sft = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="epoch",
        bf16=True,
        max_length=args.max_len,
        gradient_checkpointing=True,
        completion_only_loss=True,   # mask the prompt; learn only the answer tokens
        report_to="none",
    )

    # NOTE: TRL API drifts across versions (processing_class/max_length are recent).
    trainer = SFTTrainer(
        model=model, args=sft, train_dataset=dataset,
        peft_config=peft_config, processing_class=tokenizer,
    )
    print(f"Training {len(dataset)} rows for {args.epochs} epochs...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nAdapter -> {args.output_dir}\nNext: bash scripts/export_gguf.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
