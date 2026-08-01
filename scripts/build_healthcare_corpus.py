#!/usr/bin/env python3
"""Track B: build a broad open-healthcare free-text corpus for continued training.

Track A (build_accuracy_sft.py) targets the exact automated scoring FORMAT.
Track B targets genuine depth: judges download the raw GGUF and run it standalone
in LM Studio/Ollama with no RAG in that loop, so the model's own knowledge is what
gets read. This corpus is modeled with plain causal-LM loss (not completion-only),
i.e. continued/domain-adaptive pretraining on real clinical text.

Sources (all public, permissively-licensed-or-research-use HF datasets):
  - MedQuAD:        NIH-derived consumer health Q&A (medical_qa -> flattened text)
  - EPFL Guidelines: curated clinical practice guideline text (the closest thing to
                     WHO/IMCI-style reference material at scale)
  - Medical Meadow:  medical flashcards + WikiDoc articles (medAlpaca project)

Each source is optional and independently skip-on-failure (dataset schemas drift).
Output: output/healthcare_corpus.jsonl of {"text": str} chunks, consumed by
scripts/train_lora.py's `_load_healthcare_corpus`.

    python scripts/build_healthcare_corpus.py --max-per-dataset 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"


def from_medquad():
    from datasets import load_dataset

    ds = load_dataset("lavita/MedQuAD", split="train")
    for r in ds:
        q = (r.get("question") or "").strip()
        a = (r.get("answer") or "").strip()
        if q and a and len(a) > 40:
            yield {"text": f"{q}\n{a}"}


def from_guidelines(max_chars: int = 2000):
    from datasets import load_dataset

    ds = load_dataset("epfl-llm/guidelines", split="train")
    for r in ds:
        text = (r.get("clean_text") or r.get("text") or "").strip()
        if text and len(text) > 100:
            yield {"text": text[:max_chars]}


def from_medical_meadow_flashcards():
    from datasets import load_dataset

    ds = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
    for r in ds:
        instr = (r.get("instruction") or "").strip()
        out = (r.get("output") or "").strip()
        if instr and out:
            yield {"text": f"{instr}\n{out}"}


def from_medical_meadow_wikidoc(max_chars: int = 2000):
    from datasets import load_dataset

    ds = load_dataset("medalpaca/medical_meadow_wikidoc", split="train")
    for r in ds:
        instr = (r.get("instruction") or "").strip()
        out = (r.get("output") or "").strip()
        if instr and out:
            yield {"text": f"{instr}\n{out}"[:max_chars]}


def from_pubmedqa_abstracts(max_chars: int = 1500):
    """Raw PubMedQA abstracts as plain text (distinct from Track A's Q&A use of
    this same dataset) — 273k available, pure biomedical prose for depth."""
    from datasets import load_dataset

    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    for r in ds:
        ctx = " ".join(r["context"]["contexts"]).strip()
        if len(ctx) > 200:
            yield {"text": ctx[:max_chars]}


SOURCES = {
    "medquad": from_medquad,
    "guidelines": from_guidelines,
    "flashcards": from_medical_meadow_flashcards,
    "wikidoc": from_medical_meadow_wikidoc,
    "pubmed_abstracts": from_pubmedqa_abstracts,
}
DEFAULT = ["medquad", "guidelines", "flashcards", "wikidoc", "pubmed_abstracts"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the Track-B broad healthcare corpus")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT, choices=list(SOURCES))
    ap.add_argument("--max-per-dataset", type=int, default=15000,
                    help="Each source has 10k-270k rows available; 5k was too thin (see PROGRESS.md)")
    ap.add_argument("--out", default=str(OUT / "healthcare_corpus.jsonl"))
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    seen: set[str] = set()
    total = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for name in args.datasets:
            try:
                n = 0
                for rec in SOURCES[name]():
                    key = rec["text"][:200]
                    if key in seen:
                        continue
                    seen.add(key)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                    total += 1
                    if n >= args.max_per_dataset:
                        break
                print(f"  {name:12s}: {n} chunks")
            except Exception as e:  # dataset renamed / offline / schema drift
                print(f"  {name:12s}: SKIPPED ({type(e).__name__}: {e})")

    print(f"\nWrote {total} healthcare corpus chunks -> {args.out}")
    print("Trained via plain causal-LM loss in scripts/train_lora.py (Track B).")
    if total == 0:
        print("WARNING: corpus is empty — train_lora.py will just skip Track B, which is fine "
              "(Track A + clinical SFT still run), but re-check dataset availability if unintended.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
