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
import os
import sys
from pathlib import Path

# Must be set before torch is imported/initializes CUDA. Reduces OOM risk from
# allocator fragmentation on long training runs (PyTorch's own suggestion in the
# OOM error message).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
                   help="Number of ITEMS per step, all fused into ONE forward pass "
                        "(each item unrolls to its own choice count as extra rows). "
                        "Kept conservative after a real OOM at batch_size=8 — this isn't "
                        "just about the loss computation, the model's own internal "
                        "per-layer activations (MLP intermediate states, LoRA adapter "
                        "matmuls, etc.) scale with total rows too. Lower if you still hit OOM.")
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True,
                   help="Trade ~20-40%% speed for much lower VRAM. ON by default after "
                        "real OOMs at batch_size=8/16 without it — a fused multi-row "
                        "batch needs this, unlike a single-sequence forward pass. Pass "
                        "--no-gradient_checkpointing to disable if you have VRAM to spare.")
    p.add_argument("--no-gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--max_len", type=int, default=256,
                   help="Every row in a fused micro-batch pads to the LONGEST row in "
                        "it, and that full (rows x length x ~152k vocab) tensor is "
                        "materialized by the model's forward pass regardless of the "
                        "loss-computation fix — keep this modest to avoid OOM.")
    p.add_argument("--clinical_repeat", type=int, default=3,
                   help="Upsample the (small) clinical set so it isn't drowned by MCQA")
    p.add_argument("--aux_nll_weight", type=float, default=0.2,
                   help="Weight of the plain gold-NLL term mixed with the ranking loss")
    p.add_argument("--resume_adapter", default=None,
                   help="Path to an existing LoRA adapter dir (e.g. output/jamii-lora) to "
                        "CONTINUE training from, instead of initializing a fresh adapter. "
                        "For a short, targeted follow-up pass (e.g. more Swahili data) "
                        "without repeating the full multi-hour run. Use a low --lr "
                        "(e.g. 2e-5) to avoid catastrophically overwriting what the "
                        "first run already learned.")
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
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
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

    if args.resume_adapter:
        print(f"Continuing training from existing adapter: {args.resume_adapter}")
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id

    class RankingTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # BRUTAL OPTIMIZATION: flatten every item's every choice in this whole
            # micro-batch into ONE padded tensor and do a SINGLE forward pass, instead
            # of one forward call per item (which was still one call per item even
            # after the earlier per-item choice-batching fix). For a tiny 0.6B model,
            # Python/kernel-launch overhead per call dominates over raw FLOPs, so
            # cutting N calls -> 1 call per step is where the real wall-clock win is.
            all_seqs: list[list[int]] = []
            bounds: list[tuple[int, int, int]] = []  # (row_start, row_end, ctx_len) per item
            for item in inputs:
                ctx_ids = item["ctx_ids"]
                start = len(all_seqs)
                for c_ids in item["choice_ids"]:
                    all_seqs.append(ctx_ids + c_ids)
                bounds.append((start, len(all_seqs), len(ctx_ids)))

            max_len = max(len(s) for s in all_seqs)
            input_ids = torch.full((len(all_seqs), max_len), pad_id, dtype=torch.long, device=device)
            attn_mask = torch.zeros((len(all_seqs), max_len), dtype=torch.long, device=device)
            for i, s in enumerate(all_seqs):
                input_ids[i, : len(s)] = torch.tensor(s, device=device)
                attn_mask[i, : len(s)] = 1

            out = model(input_ids=input_ids, attention_mask=attn_mask)
            raw_logits = out.logits  # (N_total_rows, L, V) — V=151936 for Qwen3.
            # CRITICAL: do NOT run log_softmax/float() over this whole tensor — with a
            # ~152k vocab, (rows x length x vocab) blows past 24GB instantly (this is
            # what actually OOM'd). We only ever need a handful of positions per row
            # (the choice tokens), so slice those out FIRST and normalize only that
            # tiny slice — cuts this computation's memory by orders of magnitude.
            #
            # SPEED: gather every (row, position, target-token) triple needed across the
            # WHOLE micro-batch into flat lists, then do ONE slice + ONE log_softmax +
            # ONE gather for all of them at once, instead of a Python loop summing one
            # token at a time. This matters a lot for long targets (Track-B corpus rows
            # can be ~150+ tokens) — was previously ~150 individual GPU ops in a Python
            # loop per such row. Total elements processed is identical (no memory
            # regression), verified bit-identical to the loop version with a synthetic
            # test before landing this.
            flat_rows, flat_positions, flat_targets = [], [], []
            choice_spans = []  # (item_idx, flat_start, flat_end) in item/choice order
            for item_idx, ((row_start, row_end, ctx_len), item) in enumerate(zip(bounds, inputs)):
                pos_start = ctx_len - 1
                for row, c_ids in zip(range(row_start, row_end), item["choice_ids"]):
                    fs = len(flat_rows)
                    for j, tok in enumerate(c_ids):
                        flat_rows.append(row)
                        flat_positions.append(pos_start + j)
                        flat_targets.append(tok)
                    choice_spans.append((item_idx, fs, len(flat_rows)))

            row_idx = torch.tensor(flat_rows, device=device)
            pos_idx = torch.tensor(flat_positions, device=device)
            tgt_idx = torch.tensor(flat_targets, device=device)
            needed_logits = raw_logits[row_idx, pos_idx, :].float()  # (T, V) — T is small
            needed_logprobs = F.log_softmax(needed_logits, dim=-1)
            token_logprobs = needed_logprobs.gather(1, tgt_idx.unsqueeze(1)).squeeze(1)  # (T,)

            per_item_sums: list[list[torch.Tensor]] = [[] for _ in inputs]
            per_item_ntok: list[list[int]] = [[] for _ in inputs]
            for item_idx, fs, fe in choice_spans:
                per_item_sums[item_idx].append(token_logprobs[fs:fe].sum())
                per_item_ntok[item_idx].append(fe - fs)

            item_losses = []
            for item_idx, item in enumerate(inputs):
                per_choice_sum = per_item_sums[item_idx]
                per_choice_ntok = per_item_ntok[item_idx]
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
