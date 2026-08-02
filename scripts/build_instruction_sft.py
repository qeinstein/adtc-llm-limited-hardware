#!/usr/bin/env python3
"""Build a REAL instruction-following SFT set — the fix for broken generation.

THE DIAGNOSIS (measured, not guessed). Our first training run used:
    ~93,000 MCQA rows  (listwise ranking loss — teaches the model to RANK)
        80 chat rows   (data/medical_lora_dataset.json, upsampled 3x = 240)

So the model saw ~93k examples of ranking a fixed set of choices and 240 examples
of actually WRITING an answer. It learned exactly that: arc_easy acc_norm 80.0
(excellent ranking) while free generation restated the system prompt, invented
citations, looped, and drifted into Chinese. We fine-tuned from Qwen3-0.6B-**Base**,
which has no instruction post-training of its own, so those 240 examples were the
ONLY thing teaching it to answer or to emit EOS. That is the whole bug.

THE FIX: thousands of genuine question -> answer pairs, in the Alpaca schema
train_lora.py already consumes via --clinical_file (which appends EOS to every
target, so this also teaches the model to STOP).

Sources are all open and commercially usable (we deliberately kept the whole
pipeline Apache-2.0-compatible — no CC-BY-NC datasets, which rules out Alpaca
and no_robots):
  - MedQuAD (lavita/MedQuAD)      NIH/NLM-derived consumer health Q&A, public domain.
                                  Already downloaded for Track B, but flattened to
                                  raw text there — the Q/A structure was thrown away.
  - PubMedQA (pqa_labeled)        research questions + long-form expert answers.
  - medalpaca flashcards          already instruction-shaped medical Q&A.
  - data/medical_lora_dataset.json our own hand-written bilingual clinical set.

Answer-length filtering matters: NIH pages run to thousands of characters, and
training on rambling answers teaches rambling. We keep answers in a band that
looks like the concise, safety-framed replies the product should give.

    python scripts/build_instruction_sft.py --max-per-source 12000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"

MIN_ANSWER_CHARS = 60
MAX_ANSWER_CHARS = 900   # keeps answers concise enough to teach stopping
MIN_QUESTION_CHARS = 12


def _clean(text: str) -> str:
    """Strip web artifacts that ride along with NIH/wiki-derived answers."""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)          # collapse hard wraps
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"^\s*(Key Points|Summary)\s*[:\-]\s*", "", text, flags=re.I)
    return text.strip()


def _acceptable(q: str, a: str) -> bool:
    if len(q) < MIN_QUESTION_CHARS or not (MIN_ANSWER_CHARS <= len(a) <= MAX_ANSWER_CHARS):
        return False
    # Reject answers that are mostly a list of links/section headers, or that
    # trail off mid-sentence — both teach bad generation habits.
    if a.count("|") > 3 or a.count("- ") > 8:
        return False
    return a[-1] in ".!?)"


def from_medquad(limit: int):
    from datasets import load_dataset

    ds = load_dataset("lavita/MedQuAD", split="train")
    n = 0
    for r in ds:
        q, a = _clean(r.get("question") or ""), _clean(r.get("answer") or "")
        if _acceptable(q, a):
            yield {"instruction": q, "input": "", "output": a}
            n += 1
            if n >= limit:
                return


def from_pubmedqa(limit: int):
    from datasets import load_dataset

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    n = 0
    for r in ds:
        q = _clean(r.get("question") or "")
        a = _clean(r.get("long_answer") or "")
        if _acceptable(q, a):
            yield {"instruction": q, "input": "", "output": a}
            n += 1
            if n >= limit:
                return


def from_flashcards(limit: int):
    from datasets import load_dataset

    ds = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
    n = 0
    for r in ds:
        q = _clean(r.get("input") or r.get("instruction") or "")
        a = _clean(r.get("output") or "")
        if _acceptable(q, a):
            yield {"instruction": q, "input": "", "output": a}
            n += 1
            if n >= limit:
                return


def from_ours(limit: int):
    """Our own bilingual clinical set — the answer STYLE we actually want."""
    p = DATA / "medical_lora_dataset.json"
    if not p.exists():
        return
    for item in json.loads(p.read_text(encoding="utf-8"))[:limit]:
        instr = (item.get("instruction") or "").strip()
        out = (item.get("output") or "").strip()
        if instr and out:
            yield {"instruction": instr, "input": (item.get("input") or "").strip(), "output": out}


SOURCES = {
    "medquad": from_medquad,
    "pubmedqa": from_pubmedqa,
    "flashcards": from_flashcards,
    "ours": from_ours,
}
DEFAULT = ["medquad", "flashcards", "pubmedqa", "ours"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build instruction-following SFT data")
    ap.add_argument("--sources", nargs="+", default=DEFAULT, choices=list(SOURCES))
    ap.add_argument("--max-per-source", type=int, default=12000)
    ap.add_argument("--our-repeat", type=int, default=25,
                    help="Upsample our own bilingual clinical set so its answer style "
                         "is not drowned out by the much larger public sources.")
    ap.add_argument("--out", default=str(OUT / "instruction_sft.json"))
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    rows: list[dict] = []
    for name in args.sources:
        try:
            got = list(SOURCES[name](args.max_per_source))
        except Exception as e:
            print(f"  [warn] {name} failed ({type(e).__name__}: {e}) — skipping")
            continue
        if name == "ours":
            got = got * max(1, args.our_repeat)
        rows.extend(got)
        print(f"  {name:12s} -> {len(got):6d} rows")

    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    lens = [len(r["output"]) for r in rows] or [0]
    print(f"\nTotal: {len(rows)} instruction rows -> {args.out}")
    print(f"Answer length: min={min(lens)} mean={sum(lens)//len(lens)} max={max(lens)} chars")
    print("\nThis replaces the 80-row chat set that caused the generation failure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
