#!/usr/bin/env python3
"""Robust MCQ accuracy on a GGUF — the exact metric the ADTC profiler's lm-eval uses.

Computes per-choice continuation loglikelihood via llama-cpp-python
`create_completion(echo=True, logprobs=1)` and reports:
  acc      = argmax over choices of sum(token_logprobs)          == gold
  acc_norm = argmax over choices of sum(token_logprobs)/len(str) == gold   (char-normalized)

This matches lm-evaluation-harness scoring (which prefers acc_norm) without depending
on its fragile GGUF-server backend. Accuracy is hardware-independent, so these numbers
transfer to the audit VM. Prompts use the exact eval templates (see adtc-accuracy-sft).

    python scripts/mcq_eval.py --model model.gguf --task arc_easy --limit 100
"""

from __future__ import annotations

import argparse
import sys

LETTERS = ["A", "B", "C", "D", "E"]


def fmt_fulltext(q, choices, gold_idx):
    return f"Question: {q.strip()}\nAnswer:", [f" {c.strip()}" for c in choices], gold_idx


def fmt_letter(q, options, gold_idx):
    body = "\n".join(f"{LETTERS[i]}. {o.strip()}" for i, o in enumerate(options))
    return f"{q.strip()}\n{body}\nAnswer:", [f" {LETTERS[i]}" for i in range(len(options))], gold_idx


def load_task(task, limit):
    """Return list of (context, continuations, gold_idx). TEST/VALIDATION splits only
    (we EVALUATE here; training uses the train splits — never mixed)."""
    from datasets import load_dataset
    items = []
    if task in ("arc_easy", "arc_challenge"):
        cfg = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
        for r in load_dataset("allenai/ai2_arc", cfg, split="test"):
            labels, texts = r["choices"]["label"], r["choices"]["text"]
            if r["answerKey"] not in labels:
                continue
            items.append(fmt_fulltext(r["question"], texts, labels.index(r["answerKey"])))
            if len(items) >= limit:
                break
    elif task == "openbookqa":
        for r in load_dataset("allenai/openbookqa", "main", split="test"):
            labels, texts = r["choices"]["label"], r["choices"]["text"]
            if r["answerKey"] not in labels:
                continue
            items.append(fmt_fulltext(r["question_stem"], texts, labels.index(r["answerKey"])))
            if len(items) >= limit:
                break
    elif task == "medmcqa":
        for r in load_dataset("openlifescienceai/medmcqa", split="validation"):
            opts = [r["opa"], r["opb"], r["opc"], r["opd"]]
            if 0 <= r["cop"] < 4 and all(opts):
                items.append(fmt_letter(r["question"], opts, r["cop"]))
            if len(items) >= limit:
                break
    elif task == "pubmedqa":
        for r in load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train"):
            ctx = " ".join(r["context"]["contexts"])[:1200]
            dec = (r.get("final_decision") or "").lower()
            choices = ["yes", "no", "maybe"]
            if dec in choices:
                c, conts, _ = fmt_fulltext("", choices, choices.index(dec))
                items.append((f"Abstract: {ctx}\nQuestion: {r['question'].strip()}\nAnswer:",
                              [f" {x}" for x in choices], choices.index(dec)))
            if len(items) >= limit:
                break
    else:
        raise SystemExit(f"unknown task {task}")
    return items


def loglik(llm, ctx, cont):
    """Sum of continuation token logprobs given context (echo + logprobs)."""
    full = ctx + cont
    out = llm.create_completion(prompt=full, max_tokens=1, logprobs=1, echo=True, temperature=0.0)
    lp = out["choices"][0]["logprobs"]
    offs, tlps = lp["text_offset"], lp["token_logprobs"]
    total, n = 0.0, 0
    for off, tl in zip(offs, tlps):
        if tl is None:
            continue
        if len(ctx) <= off < len(full):  # continuation prompt tokens only
            total += tl
            n += 1
    return total, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="arc_easy")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    from llama_cpp import Llama
    llm = Llama(model_path=args.model, n_ctx=args.n_ctx, n_gpu_layers=0,
                n_threads=args.threads, logits_all=True, verbose=False)

    items = load_task(args.task, args.limit)
    acc = acc_norm = 0
    for ctx, conts, gold in items:
        scores, norms = [], []
        for c in conts:
            ll, _ = loglik(llm, ctx, c)
            scores.append(ll)
            norms.append(ll / max(len(c), 1))
        if max(range(len(scores)), key=lambda i: scores[i]) == gold:
            acc += 1
        if max(range(len(norms)), key=lambda i: norms[i]) == gold:
            acc_norm += 1
    n = len(items)
    print(f"{args.task}: n={n}  acc={100*acc/n:.1f}  acc_norm={100*acc_norm/n:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
