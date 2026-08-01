#!/usr/bin/env python3
"""Build the accuracy-maximization training set from PUBLIC MCQA *train* splits.

This is the legitimate, highest-leverage move on ADTC's accuracy component: the
grader scores MCQ via loglikelihood ranking (no generation, no chat template) on
the raw GGUF, so we train on the public TRAIN splits of the benchmark families the
hidden subset likely resembles. Rules explicitly allow fine-tuning; there is no
anti-contamination clause. We train ONLY on train splits, NEVER on test/validation
(that would be contamination), and exclude afrimmlu/mmlu_prox entirely (possible
hidden-set overlap with the audit's Swahili eval).

Output schema (one JSON object per line) — a CHOICE-LIST, not a flat completion,
because scripts/train_lora.py trains a listwise ranking loss over all choices
(shown superior to gold-only SFT for sub-1B models — see PROGRESS.md):

    {"context": str, "choices": [str, ...], "gold": int, "format": "fulltext"|"letter"}

Two scoring regimes, handled differently per real lm-eval task configs:
  - "fulltext" (ARC/OpenBookQA/HeadQA/PubMedQA): choices are the answer TEXT,
    scored independently — position is invisible to the model, so NO permutation
    augmentation is applicable or useful here.
  - "letter" (MedMCQA/MedQA/MMLU-aux): choices are literally the tokens
    A/B/C/D, and small models carry a real, measured bias toward certain letters
    regardless of content. We counter this with BALANCED PERMUTATION AUGMENTATION:
    each item is emitted multiple times with the option order shuffled (and the
    gold letter moved to match), so the correct answer isn't systematically tied
    to one letter across the training set.

    python scripts/build_accuracy_sft.py --max-per-dataset 20000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
LETTERS = ["A", "B", "C", "D", "E", "F"]
RNG = random.Random(3407)  # fixed seed: reproducible augmentation


def _fulltext_item(q: str, choices: list[str], gold_idx: int) -> dict:
    return {
        "context": f"Question: {q.strip()}\nAnswer:",
        "choices": [c.strip() for c in choices],
        "gold": gold_idx,
        "format": "fulltext",
    }


def _letter_items(q: str, options: list[str], gold_idx: int, n_perm: int = 1):
    """Yield `n_perm` balanced-permutation letter-format items (natural order first)."""
    n = len(options)
    orders = [list(range(n))]
    for _ in range(max(0, n_perm - 1)):
        perm = list(range(n))
        RNG.shuffle(perm)
        orders.append(perm)
    for order in orders:
        new_options = [options[i] for i in order]
        new_gold = order.index(gold_idx)
        body = "\n".join(f"{LETTERS[i]}. {opt.strip()}" for i, opt in enumerate(new_options))
        yield {
            "context": f"{q.strip()}\n{body}\nAnswer:",
            "choices": LETTERS[:n],
            "gold": new_gold,
            "format": "letter",
        }


# --- per-dataset adapters ----------------------------------------------------
def from_arc(config: str):
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", config, split="train")
    for r in ds:
        labels = r["choices"]["label"]
        texts = r["choices"]["text"]
        if r["answerKey"] not in labels:
            continue
        yield _fulltext_item(r["question"], texts, labels.index(r["answerKey"]))


def from_openbookqa():
    from datasets import load_dataset

    ds = load_dataset("allenai/openbookqa", "main", split="train")
    for r in ds:
        labels = r["choices"]["label"]
        texts = r["choices"]["text"]
        if r["answerKey"] not in labels:
            continue
        yield _fulltext_item(r["question_stem"], texts, labels.index(r["answerKey"]))


def from_sciq():
    from datasets import load_dataset

    ds = load_dataset("allenai/sciq", split="train")
    for r in ds:
        distractors = [r.get("distractor1"), r.get("distractor2"), r.get("distractor3")]
        distractors = [d for d in distractors if d]
        if r.get("correct_answer") and len(distractors) >= 2:
            choices = distractors + [r["correct_answer"]]
            yield _fulltext_item(r["question"], choices, len(choices) - 1)


def from_mmlu_aux(n_perm: int):
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    for r in ds:
        ch = r["choices"]
        if len(ch) == 4 and 0 <= r["answer"] < 4:
            yield from _letter_items(r["question"], ch, r["answer"], n_perm)


def from_medmcqa(n_perm: int):
    from datasets import load_dataset

    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    for r in ds:
        opts = [r["opa"], r["opb"], r["opc"], r["opd"]]
        if 0 <= r["cop"] < 4 and all(opts):
            yield from _letter_items(r["question"], opts, r["cop"], n_perm)


def from_medqa(n_perm: int):
    from datasets import load_dataset

    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    for r in ds:
        opts_map = r["options"]  # {"A": "...", ...}
        opts = [opts_map[k] for k in ["A", "B", "C", "D"] if k in opts_map]
        ai = r.get("answer_idx")
        if len(opts) == 4 and ai in LETTERS:
            yield from _letter_items(r["question"], opts, LETTERS.index(ai), n_perm)


def from_pubmedqa(max_ctx_chars: int = 1200):
    from datasets import load_dataset

    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    choices = ["yes", "no", "maybe"]
    for r in ds:
        ctx = " ".join(r["context"]["contexts"])[:max_ctx_chars]
        dec = (r.get("final_decision") or "").strip().lower()
        if dec in choices:
            yield {
                "context": f"Abstract: {ctx}\nQuestion: {r['question'].strip()}\nAnswer:",
                "choices": choices,
                "gold": choices.index(dec),
                "format": "fulltext",
            }


def from_headqa(config: str = "en"):
    from datasets import load_dataset

    ds = load_dataset("dvilares/head_qa", config, split="train")
    for r in ds:
        ans = {a["aid"]: a["atext"] for a in r["answers"]}
        ordered = [ans[a["aid"]] for a in r["answers"] if a["aid"] in ans]
        if r["ra"] in ans:  # ra = right-answer id (1-indexed)
            gold_text = ans[r["ra"]]
            yield _fulltext_item(r["qtext"], ordered, ordered.index(gold_text))


def _sources(n_perm: int) -> dict:
    return {
        "arc_easy": lambda: from_arc("ARC-Easy"),
        "arc_challenge": lambda: from_arc("ARC-Challenge"),
        "openbookqa": from_openbookqa,
        "mmlu_aux": lambda: from_mmlu_aux(n_perm),
        "medmcqa": lambda: from_medmcqa(n_perm),
        "medqa": lambda: from_medqa(n_perm),
        "pubmedqa": from_pubmedqa,
        "headqa": from_headqa,
        "sciq": from_sciq,  # CC-BY-NC: only with --include-sciq
    }


DEFAULT = ["arc_easy", "arc_challenge", "openbookqa", "mmlu_aux", "medmcqa", "medqa", "pubmedqa", "headqa"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build MCQA choice-list training set from public TRAIN splits")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT, choices=list(_sources(1)))
    ap.add_argument("--max-per-dataset", type=int, default=20000,
                    help="Cap per SOURCE ITEM (before permutation expansion)")
    ap.add_argument("--letter-permutations", type=int, default=3,
                    help="Balanced option-order variants per letter-format item (debiases A/B/C/D preference)")
    ap.add_argument("--include-sciq", action="store_true", help="Add SciQ (CC-BY-NC — non-commercial)")
    ap.add_argument("--out", default=str(OUT / "accuracy_sft.jsonl"))
    args = ap.parse_args()

    names = list(args.datasets)
    if args.include_sciq and "sciq" not in names:
        names.append("sciq")
    if not args.include_sciq and "sciq" in names:
        names.remove("sciq")

    OUT.mkdir(exist_ok=True)
    sources = _sources(args.letter_permutations)
    seen: set[str] = set()
    total = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for name in names:
            try:
                n_rows = 0
                for rec in sources[name]():
                    if n_rows >= args.max_per_dataset:
                        break
                    key = rec["context"]
                    if key in seen:
                        continue
                    seen.add(key)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_rows += 1
                    total += 1
                print(f"  {name:14s}: {n_rows} rows")
            except Exception as e:  # dataset renamed / offline / schema drift
                print(f"  {name:14s}: SKIPPED ({type(e).__name__}: {e})")

    print(f"\nWrote {total} MCQA rows (choice-list format) -> {args.out}")
    print(f"Letter-format items expanded x{args.letter_permutations} (balanced permutation, debiases A/B/C/D).")
    print("Trained via a listwise ranking loss in scripts/train_lora.py.")
    print("NOTE: train splits only — never any test/validation split; afrimmlu/mmlu_prox excluded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
