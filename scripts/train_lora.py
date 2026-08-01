#!/usr/bin/env python3
"""QLoRA fine-tune for Jamii Afya — Qwen3-0.6B-Base, listwise MCQ ranking + clinical SFT.

WHY a custom trainer instead of plain next-token SFT: the ADTC profiler scores
multiple-choice questions by RANKING answer choices via loglikelihood (never
generation), template-free. Standard fine-tuning only pushes UP the gold answer's
probability and never pushes DOWN the wrong choices — a 2026 study benchmarking
exactly Qwen3-0.6B/1.7B/4B/8B found this "gold-only" objective is the *worst* of
the options tested, and that a listwise/contrastive objective (rank all choices,
train the correct one to win) is measurably better — specifically at sub-3B scale
(the advantage vanishes above 3B). See PROGRESS.md for the full citation trail.

Two data types are trained together in one run:
  1. MCQA choice-list rows (output/accuracy_sft.jsonl, from build_accuracy_sft.py):
     scored with a LENGTH-NORMALIZED listwise ranking loss — normalizing by the
     CHARACTER length of each choice string, exactly mirroring the profiler's
     `acc_norm` metric, plus a small auxiliary NLL term on the gold choice alone
     (keeps the model well-calibrated as an LM, avoids degenerate shortcuts).
     No chat template — the profiler scores raw completions.
  2. Clinical chat rows (data/medical_lora_dataset.json): standard completion-only
     next-token loss, rendered through the model's OWN chat template with
     thinking disabled — this is what judges see when they load the GGUF into
     LM Studio/Ollama, so the shipped template must behave well stand-alone.
  3. Optional Track-B healthcare corpus free text (output/healthcare_corpus.jsonl):
     plain causal-LM continuation loss over document chunks.

Runs on a single modest GPU (RunPod). Then merge + quantize with export_gguf.sh.

    python scripts/build_accuracy_sft.py --max-per-dataset 20000
    python scripts/prepare_dataset.py
    python scripts/train_lora.py --base_model Qwen/Qwen3-0.6B-Base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYSTEM = (
    "You are Jamii Afya, an offline medical decision-support assistant for community "
    "health workers in rural African clinics. Answer in the question's language "
    "(English or Kiswahili). Always surface danger signs and when to refer."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tune (listwise MCQ ranking + clinical chat)")
    p.add_argument("--base_model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--accuracy_file", default=str(ROOT / "output" / "accuracy_sft.jsonl"))
    p.add_argument("--clinical_file", nargs="+",
                   default=[str(ROOT / "data" / "medical_lora_dataset.json")],
                   help="One or more Alpaca-style JSON files (e.g. add "
                        "output/synthetic_clinical_chat.json after running "
                        "generate_synthetic_data.py)")
    p.add_argument("--healthcare_corpus_file", default=str(ROOT / "output" / "healthcare_corpus.jsonl"),
                   help="Optional Track-B free-text corpus from build_healthcare_corpus.py")
    p.add_argument("--output_dir", default=str(ROOT / "output" / "jamii-lora"))
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Number of ITEMS per step (each MCQA item unrolls to its own choice count)")
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Trade ~20-40%% speed for lower VRAM. OFF by default — a 0.6B model "
                        "doesn't need it on a 16GB+ GPU; enable only if you hit an OOM error.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--max_len", type=int, default=768)
    p.add_argument("--clinical_repeat", type=int, default=3,
                   help="Upsample the (small) clinical set so it isn't drowned by MCQA")
    p.add_argument("--aux_nll_weight", type=float, default=0.2,
                   help="Weight of the plain gold-NLL term mixed with the ranking loss")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Dataset: unifies MCQA choice-list rows and clinical chat rows into one schema
# --------------------------------------------------------------------------- #
class UnifiedDataset:
    def __init__(self, tokenizer, accuracy_file: str, clinical_files: list[str],
                 healthcare_corpus_file: str, clinical_repeat: int, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len
        self.items: list[dict] = []
        self._load_mcqa(accuracy_file)
        n_mcqa = len(self.items)
        for cf in clinical_files:
            self._load_clinical(cf, clinical_repeat)
        n_clinical = len(self.items) - n_mcqa
        self._load_healthcare_corpus(healthcare_corpus_file)
        n_corpus = len(self.items) - n_mcqa - n_clinical
        print(f"Dataset: {n_mcqa} MCQA(ranking) + {n_clinical} clinical(chat) + "
              f"{n_corpus} healthcare-corpus(causal-LM) = {len(self.items)} items")

    def _encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _load_mcqa(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        n_bad = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                ctx_ids = self._encode(r["context"])
                choice_ids, choice_charlens = [], []
                for c in r["choices"]:
                    ids = self._encode(" " + c)
                    if len(ctx_ids) + len(ids) > self.max_len:
                        ids = ids[: max(1, self.max_len - len(ctx_ids))]
                    choice_ids.append(ids)
                    choice_charlens.append(max(1, len(c)))
                self.items.append({
                    "kind": "mcqa",
                    "ctx_ids": ctx_ids,
                    "choice_ids": choice_ids,
                    "choice_charlens": choice_charlens,
                    "gold": r["gold"],
                })
            except (json.JSONDecodeError, KeyError, TypeError, IndexError) as e:
                n_bad += 1
                if n_bad <= 3:  # show the first few so real corruption is still visible
                    print(f"  [warn] skipping malformed MCQA row in {path}: {type(e).__name__}: {e}")
        if n_bad:
            print(f"  [warn] skipped {n_bad} malformed row(s) in {path} (out of otherwise-loaded data)")

    def _load_clinical(self, path: str, repeat: int) -> None:
        p = Path(path)
        if not p.exists():
            return
        for item in json.loads(p.read_text(encoding="utf-8")):
            instr = (item.get("instruction") or "").strip()
            out = (item.get("output") or "").strip()
            if not instr or not out:
                continue
            inp = (item.get("input") or "").strip()
            user = f"{instr}\n\n{inp}" if inp else instr
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
            prompt = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            ctx_ids = self._encode(prompt)
            tgt_ids = self._encode(out) + [self.tok.eos_token_id]
            if len(ctx_ids) + len(tgt_ids) > self.max_len:
                ctx_ids = ctx_ids[-(self.max_len - len(tgt_ids)):]
            for _ in range(max(1, repeat)):
                self.items.append({
                    "kind": "chat",
                    "ctx_ids": ctx_ids,
                    "choice_ids": [tgt_ids],
                    "choice_charlens": [max(1, len(out))],
                    "gold": 0,
                })

    def _load_healthcare_corpus(self, path: str) -> None:
        """Track B: plain causal-LM continuation over free-text documents (no Q/A
        split — the whole chunk is 'context', modeled with standard LM loss)."""
        p = Path(path)
        if not p.exists():
            return
        n_bad = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                n_bad += 1
                if n_bad <= 3:
                    print(f"  [warn] skipping malformed corpus row in {path}: {e}")
                continue
            text = r.get("text", "").strip()
            if not text:
                continue
            ids = self._encode(text)
            if len(ids) < 8:
                continue
            ids = ids[: self.max_len]
            split = max(1, len(ids) // 4)  # first quarter = "context" (rest modeled as target)
            self.items.append({
                "kind": "corpus",
                "ctx_ids": ids[:split],
                "choice_ids": [ids[split:]],
                "choice_charlens": [max(1, len(text))],
                "gold": 0,
            })
        if n_bad:
            print(f"  [warn] skipped {n_bad} malformed row(s) in {path} (out of otherwise-loaded data)")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def identity_collate(batch: list[dict]) -> list[dict]:
    return batch


def main() -> int:
    args = parse_args()
    print("=" * 70)
    print(f"  Jamii Afya QLoRA — base {args.base_model}")
    print("  Objective: listwise MCQ ranking (char-length-normalized) + clinical SFT")
    print("=" * 70)

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    print(f"CUDA: {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = UnifiedDataset(
        tokenizer, args.accuracy_file, args.clinical_file, args.healthcare_corpus_file,
        args.clinical_repeat, args.max_len,
    )
    if len(dataset) == 0:
        print("ERROR: no training items. Run build_accuracy_sft.py and/or check clinical_file.")
        return 1

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
    # NOTE: this helper defaults to use_gradient_checkpointing=True internally,
    # independent of the TrainingArguments flag below — must be passed explicitly
    # or it silently re-enables checkpointing regardless of --gradient_checkpointing.
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing
    )

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id

    def choice_logprobs(ctx_ids: list[int], choices: list[list[int]]):
        """One BATCHED forward pass for all K choices of an item (right-padded to
        the batch's max length), instead of K separate forward calls — roughly a
        Kx reduction in forward passes for MCQA rows, which are most of the data.
        Returns (summed_logprob, n_tokens) per choice."""
        import torch as _torch

        seqs = [ctx_ids + c for c in choices]
        max_len = max(len(s) for s in seqs)
        input_ids = _torch.full((len(seqs), max_len), pad_id, dtype=_torch.long, device=device)
        attn_mask = _torch.zeros((len(seqs), max_len), dtype=_torch.long, device=device)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = _torch.tensor(s, device=device)
            attn_mask[i, : len(s)] = 1

        out = model(input_ids=input_ids, attention_mask=attn_mask)
        logprobs = F.log_softmax(out.logits.float(), dim=-1)  # (K, L, V)

        start = len(ctx_ids) - 1  # same context for every choice in this item
        results = []
        for i, c_ids in enumerate(choices):
            total = _torch.zeros((), device=device, dtype=_torch.float32)
            for j, tok in enumerate(c_ids):
                total = total + logprobs[i, start + j, tok]
            results.append((total, len(c_ids)))
        return results

    class RankingTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            item_losses = []
            for item in inputs:
                results = choice_logprobs(item["ctx_ids"], item["choice_ids"])
                per_choice_sum = [r[0] for r in results]
                per_choice_ntok = [r[1] for r in results]
                per_choice_norm = [
                    s / cl for s, cl in zip(per_choice_sum, item["choice_charlens"])
                ]  # char-length norm == acc_norm

                gold = item["gold"]
                aux_nll = -(per_choice_sum[gold] / max(1, per_choice_ntok[gold]))

                if len(per_choice_norm) > 1:
                    scores = torch.stack(per_choice_norm)
                    ranking_loss = -F.log_softmax(scores, dim=0)[gold]
                    item_loss = ranking_loss + args.aux_nll_weight * aux_nll
                else:
                    item_loss = aux_nll  # chat / corpus rows: plain NLL

                item_losses.append(item_loss)

            loss = torch.stack(item_losses).mean()
            return (loss, None) if return_outputs else loss

    targs = TrainingArguments(
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
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = RankingTrainer(
        model=model, args=targs, train_dataset=dataset, data_collator=identity_collate,
    )
    print(f"Training {len(dataset)} items for {args.epochs} epochs "
          f"(batch={args.batch_size} items x grad_accum={args.grad_accum})...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nAdapter -> {args.output_dir}\nNext: bash scripts/export_gguf.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
