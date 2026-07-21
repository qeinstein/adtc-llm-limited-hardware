#!/usr/bin/env python3
"""Prepare fine-tuning splits and the imatrix calibration corpus.

Inputs (committed, authored by clinicians/curated from WHO-IMCI):
    data/medical_lora_dataset.json   Alpaca-format {instruction, input, output}

Outputs (git-ignored intermediates under output/):
    output/train.jsonl               chat-format rows for SFT
    output/val.jsonl                 held-out rows
    output/calibration_corpus.txt    EN+SW medical text for llama-imatrix (§quant)

The calibration corpus is the domain (English+Kiswahili medical) text used by
llama-imatrix so quantization puts precision where our use case needs it — this is
what makes an in-domain Q4_K_M better than a generic one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"


def load_alpaca(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    clean = []
    for r in rows:
        instr = (r.get("instruction") or "").strip()
        out = (r.get("output") or "").strip()
        if not instr or not out:
            continue
        inp = (r.get("input") or "").strip()
        user = f"{instr}\n\n{inp}" if inp else instr
        clean.append({"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": out},
        ]})
    return clean


def deterministic_split(rows: list[dict], val_every: int = 8) -> tuple[list, list]:
    """Every Nth row -> validation. Deterministic, no RNG (reproducible)."""
    train = [r for i, r in enumerate(rows) if i % val_every != 0]
    val = [r for i, r in enumerate(rows) if i % val_every == 0]
    return train, val


def write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_calibration_corpus(dataset: list[dict]) -> str:
    parts: list[str] = []
    # Clinical knowledge base (dense EN+SW terminology).
    guidelines = DATA / "medical_guidelines.json"
    if guidelines.exists():
        for d in json.loads(guidelines.read_text(encoding="utf-8")):
            parts.append(f"{d.get('title', '')}. {d.get('text', '')}")
    # Instruction/answer text (task + language distribution we care about).
    for row in dataset:
        for m in row["messages"]:
            parts.append(m["content"])
    return "\n\n".join(p.strip() for p in parts if p.strip())


def main() -> int:
    src = DATA / "medical_lora_dataset.json"
    if not src.exists():
        print(f"ERROR: {src} not found.")
        return 1

    rows = load_alpaca(src)
    if not rows:
        print("ERROR: dataset is empty after cleaning.")
        return 1

    OUT.mkdir(exist_ok=True)
    train, val = deterministic_split(rows)
    write_jsonl(train, OUT / "train.jsonl")
    write_jsonl(val, OUT / "val.jsonl")

    corpus = build_calibration_corpus(rows)
    (OUT / "calibration_corpus.txt").write_text(corpus, encoding="utf-8")

    print(f"Prepared {len(rows)} rows -> {len(train)} train / {len(val)} val")
    print(f"  output/train.jsonl, output/val.jsonl")
    print(f"  output/calibration_corpus.txt ({len(corpus.split())} words for imatrix)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
