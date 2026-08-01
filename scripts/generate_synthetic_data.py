#!/usr/bin/env python3
"""Generate bilingual (EN + Kiswahili) training data via teacher-model distillation,
GROUNDED STRICTLY in our own verified clinical guidelines (data/medical_guidelines.json).

WHY THIS EXISTS: after Track A (public benchmark MCQA) + Track B (open healthcare
corpora), every training source we have is effectively English-only — there is no
Swahili medical MCQA dataset and none of the Track-B sources are multilingual. But
Swahili competence claims a real scoring bonus (see PROGRESS.md), so this is our
biggest remaining data gap, and there is no public dataset that fills it.

The fix: use a strong "teacher" model to EXPAND our own 32 hand-curated, medically
reviewed guidelines into many diverse bilingual examples. Grounding on OUR OWN
verified source text (not open-ended generation) is what makes this safe for a
medical application — the teacher is instructed to only elaborate on given facts,
not invent new clinical claims. This is the same "distillation" technique shown in
research to be one of the highest-leverage ways to punch above a small model's
weight class (see PROGRESS.md / memory for citations).

Produces THREE outputs, each in the exact schema the other scripts already expect,
so they can be concatenated straight in:
  - output/synthetic_clinical_chat.json   (Alpaca-style, same shape as
    data/medical_lora_dataset.json — merge by passing as an extra --clinical_file
    to train_lora.py, or concatenate the JSON arrays)
  - output/synthetic_mcqa.jsonl           (choice-list rows, same shape as
    build_accuracy_sft.py's output — `cat` onto output/accuracy_sft.jsonl)
  - output/synthetic_corpus.jsonl         (free-text chunks, same shape as
    build_healthcare_corpus.py's output — `cat` onto output/healthcare_corpus.jsonl)

REQUIRES an OpenAI-compatible API key (OpenAI, or any compatible provider via
--base-url). This is the one external dependency in the whole pipeline that needs
a human to supply credentials — there is no way around that for teacher-model
distillation.

    export OPENAI_API_KEY=sk-...
    python scripts/generate_synthetic_data.py --model gpt-4o-mini --per-guideline 6
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
LETTERS = ["A", "B", "C", "D"]

SYSTEM_PROMPT = (
    "You are a bilingual (English/Kiswahili) medical-education content generator for "
    "an offline clinical decision-support tool used by community health workers in "
    "rural African clinics. You will be given ONE verified clinical guideline. "
    "Generate content that ONLY elaborates on the facts already present in that "
    "guideline — do not invent new clinical claims, doses, or thresholds not stated "
    "or directly implied by the source text. Always keep danger-sign/referral framing "
    "when relevant. Output STRICT JSON matching the requested schema, nothing else — "
    "no markdown fences, no commentary."
)

TASK_PROMPT = """Source guideline (title + text):
TITLE: {title}
TEXT: {text}

Generate exactly this JSON object (no markdown, no extra text):
{{
  "chat_en": [ {{"instruction": "<a realistic CHW/patient question in English>", "output": "<clear, safety-framed answer using ONLY facts from the source text>"}} , ... {n} items ],
  "chat_sw": [ {{"instruction": "<the same style of question, in natural Kiswahili>", "output": "<the answer in Kiswahili, same safety framing>"}} , ... {n} items ],
  "mcq_en": [ {{"question": "<English clinical question testing a fact from the source>", "options": ["<opt A>","<opt B>","<opt C>","<opt D>"], "correct_index": <0-3>}} , ... {n} items ],
  "mcq_sw": [ {{"question": "<the same in Kiswahili>", "options": ["<A>","<B>","<C>","<D>"], "correct_index": <0-3>}} , ... {n} items ]
}}

Vary phrasing, patient ages/context, and question style across items. For MCQ
distractors, use plausible-but-wrong options (common misconceptions or adjacent
conditions), not absurd ones."""


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start : end + 1])


def call_teacher(client, model: str, title: str, text: str, n: int) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": TASK_PROMPT.format(title=title, text=text, n=n)},
        ],
        temperature=0.7,
    )
    return _extract_json(resp.choices[0].message.content)


def main() -> int:
    ap = argparse.ArgumentParser(description="Distill bilingual training data from our verified guidelines")
    ap.add_argument("--model", default="gpt-4o-mini", help="Teacher model name")
    ap.add_argument("--base-url", default=None, help="Override for OpenAI-compatible endpoints")
    ap.add_argument("--per-guideline", type=int, default=6, help="Items per language per category, per guideline")
    ap.add_argument("--guidelines-file", default=str(DATA / "medical_guidelines.json"))
    args = ap.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai")
        return 1

    client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()

    guidelines = json.loads(Path(args.guidelines_file).read_text(encoding="utf-8"))
    print(f"Distilling from {len(guidelines)} guidelines x {args.per_guideline} items/category/language "
          f"= up to {len(guidelines) * args.per_guideline * 4} new examples...")

    OUT.mkdir(exist_ok=True)
    chat_rows: list[dict] = []
    mcqa_f = open(OUT / "synthetic_mcqa.jsonl", "w", encoding="utf-8")
    corpus_f = open(OUT / "synthetic_corpus.jsonl", "w", encoding="utf-8")

    n_ok, n_fail = 0, 0
    for g in guidelines:
        title, text = g.get("title", ""), g.get("text", "")
        if not text:
            continue
        try:
            result = call_teacher(client, args.model, title, text, args.per_guideline)

            for item in result.get("chat_en", []) + result.get("chat_sw", []):
                if item.get("instruction") and item.get("output"):
                    chat_rows.append({"instruction": item["instruction"], "input": "", "output": item["output"]})

            for key in ("mcq_en", "mcq_sw"):
                for item in result.get(key, []):
                    opts = item.get("options", [])
                    ci = item.get("correct_index")
                    if len(opts) == 4 and isinstance(ci, int) and 0 <= ci < 4:
                        body = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
                        row = {
                            "context": f"{item['question']}\n{body}\nAnswer:",
                            "choices": LETTERS,
                            "gold": ci,
                            "format": "letter",
                        }
                        mcqa_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            # also bank the raw Q+A pairs as free text for Track B depth
            for item in result.get("chat_en", []) + result.get("chat_sw", []):
                if item.get("instruction") and item.get("output"):
                    corpus_f.write(json.dumps(
                        {"text": f"{item['instruction']}\n{item['output']}"}, ensure_ascii=False
                    ) + "\n")

            n_ok += 1
            print(f"  [{n_ok+n_fail}/{len(guidelines)}] {title[:50]:50s} OK")
        except Exception as e:
            n_fail += 1
            print(f"  [{n_ok+n_fail}/{len(guidelines)}] {title[:50]:50s} FAILED ({type(e).__name__}: {e})")

    mcqa_f.close()
    corpus_f.close()
    (OUT / "synthetic_clinical_chat.json").write_text(
        json.dumps(chat_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nDone: {n_ok} guidelines succeeded, {n_fail} failed.")
    print(f"  {len(chat_rows)} chat rows      -> output/synthetic_clinical_chat.json")
    print("  MCQA rows              -> output/synthetic_mcqa.jsonl")
    print("  free-text corpus rows  -> output/synthetic_corpus.jsonl")
    print("\nMerge before training:")
    print("  cat output/synthetic_mcqa.jsonl >> output/accuracy_sft.jsonl")
    print("  cat output/synthetic_corpus.jsonl >> output/healthcare_corpus.jsonl")
    print("  python -c \"import json; a=json.load(open('data/medical_lora_dataset.json')); "
          "b=json.load(open('output/synthetic_clinical_chat.json')); "
          "json.dump(a+b, open('output/clinical_combined.json','w'), ensure_ascii=False, indent=2)\"")
    print("  then: python scripts/train_lora.py --clinical_file output/clinical_combined.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
