#!/usr/bin/env python3
"""Targeted follow-up data generation for the ONE gap the qualitative stress test
found: the model reliably loops/degenerates on LONGER, complex Kiswahili answers
and doesn't properly update its recommendation across multi-turn danger-sign
escalation. generate_synthetic_data.py's chat_sw items were short single-turn
Q&A -- exactly the shape that was UNDER-represented, which is the likely real
cause (not a sampling issue -- raising repeat_penalty to 1.4 didn't fix it,
see PROGRESS.md).

This generates, grounded strictly in our own verified guidelines (same safety
approach as generate_synthetic_data.py):
  1. Longer, multi-clause Kiswahili answers (3+ sentences covering assessment,
     action, AND danger signs in one response -- forces the model to sustain
     coherent Kiswahili past the point where it was breaking down).
  2. Multi-turn danger-escalation dialogues, encoded as a SINGLE training row
     (prior turn folded into the instruction as context, output = the updated
     turn-2 answer) -- our training pipeline only takes instruction/output
     pairs, so a multi-turn conversation is represented as one example that
     teaches "given this context, the new information changes the answer to
     THIS", which is exactly the failure mode observed (turn 2 was identical
     to turn 1, ignoring new danger signs).

Output: output/hard_swahili_chat.json (same Alpaca-style schema as
synthetic_clinical_chat.json) -- pass it as an extra --clinical_file to
train_lora.py's --resume_adapter continued-training pass.

    export OPENROUTER_API_KEY=sk-or-...
    python scripts/generate_hard_swahili.py --per-guideline 8
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

SYSTEM_PROMPT = (
    "You are a bilingual medical-education content generator for an offline "
    "clinical decision-support tool used by community health workers in rural "
    "African clinics. You will be given ONE verified clinical guideline. "
    "Generate content that ONLY elaborates on facts already present in that "
    "guideline -- do not invent new clinical claims, doses, or thresholds. "
    "Always keep danger-sign/referral framing when relevant. Output STRICT "
    "JSON matching the requested schema, nothing else -- no markdown fences."
)

TASK_PROMPT = """Source guideline (title + text):
TITLE: {title}
TEXT: {text}

Generate exactly this JSON object (no markdown, no extra text):
{{
  "long_answers_sw": [
    {{"question": "<a realistic, somewhat complex CHW question in natural Kiswahili, "
      "involving 2+ symptoms or a specific scenario>",
      "answer": "<a THOROUGH answer in natural, medically sound Kiswahili, "
      "at least 3 full sentences: (1) what to assess, (2) what action to take, "
      "(3) explicit danger signs and when to refer. Do not repeat phrases -- "
      "vary sentence structure naturally like a real clinician would write.>"}}
    , ... {n} items
  ],
  "escalation_dialogues_sw": [
    {{"turn1_question": "<an initial CHW question in Kiswahili describing a mild-looking "
      "presentation>",
      "turn1_answer": "<a short, appropriately calm initial answer in Kiswahili>",
      "turn2_question": "<a REALISTIC Kiswahili follow-up adding ONE OR MORE NEW danger "
      "signs from the guideline that should clearly escalate the recommendation>",
      "turn2_answer": "<the UPDATED answer in Kiswahili that explicitly acknowledges the "
      "new information and changes the recommendation accordingly -- e.g. now urgent "
      "referral -- do not just repeat turn1_answer>"}}
    , ... {n} items
  ]
}}

Vary patient ages, context, and phrasing across items. Keep language natural and
medically precise, not just grammatically correct Kiswahili."""


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


def _build_client(base_url: str | None, model: str):
    import os

    from openai import OpenAI

    if base_url:
        return OpenAI(base_url=base_url), model

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
            default_headers={
                "HTTP-Referer": "https://github.com/qeinstein/adtc-llm-limited-hardware",
                "X-Title": "Jamii Afya hard Swahili data generation",
            },
        )
        if model == "gpt-4o-mini":
            model = "google/gemini-2.5-flash"
        return client, model

    return OpenAI(), model


ESCALATION_CONTEXT_TEMPLATE = (
    "Mazungumzo ya awali:\n"
    "S1: {q1}\n"
    "J1: {a1}\n\n"
    "Swali jipya (ongezeko la taarifa): {q2}"
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate hard/long Kiswahili training examples")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--per-guideline", type=int, default=8)
    ap.add_argument("--guidelines-file", default=str(DATA / "medical_guidelines.json"))
    args = ap.parse_args()

    try:
        import openai  # noqa: F401
    except ImportError:
        print("ERROR: pip install openai")
        return 1

    client, model = _build_client(args.base_url, args.model)
    args.model = model

    guidelines = json.loads(Path(args.guidelines_file).read_text(encoding="utf-8"))
    print(f"Generating hard Kiswahili data from {len(guidelines)} guidelines "
          f"x {args.per_guideline} items/category...")

    OUT.mkdir(exist_ok=True)
    chat_rows: list[dict] = []
    n_ok, n_fail = 0, 0

    for g in guidelines:
        title, text = g.get("title", ""), g.get("text", "")
        if not text:
            continue
        try:
            result = call_teacher(client, args.model, title, text, args.per_guideline)

            for item in result.get("long_answers_sw", []):
                q, a = item.get("question"), item.get("answer")
                if q and a and len(a.split()) >= 15:  # enforce genuinely "long"
                    chat_rows.append({"instruction": q, "input": "", "output": a})

            for item in result.get("escalation_dialogues_sw", []):
                q1, a1 = item.get("turn1_question"), item.get("turn1_answer")
                q2, a2 = item.get("turn2_question"), item.get("turn2_answer")
                if q1 and a1 and q2 and a2:
                    instruction = ESCALATION_CONTEXT_TEMPLATE.format(q1=q1, a1=a1, q2=q2)
                    chat_rows.append({"instruction": instruction, "input": "", "output": a2})

            n_ok += 1
            print(f"  [{n_ok+n_fail}/{len(guidelines)}] {title[:50]:50s} OK "
                  f"(+{len(result.get('long_answers_sw', []))} long, "
                  f"+{len(result.get('escalation_dialogues_sw', []))} escalation)")
        except Exception as e:
            n_fail += 1
            print(f"  [{n_ok+n_fail}/{len(guidelines)}] {title[:50]:50s} FAILED "
                  f"({type(e).__name__}: {e})")

    (OUT / "hard_swahili_chat.json").write_text(
        json.dumps(chat_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nDone: {n_ok} guidelines OK, {n_fail} failed.")
    print(f"  {len(chat_rows)} hard Kiswahili rows -> output/hard_swahili_chat.json")
    print("\nNext: continue training from the existing adapter with a low LR, e.g.:")
    print("  PYTHONPATH=. python scripts/train_lora.py \\")
    print("    --resume_adapter output/jamii-lora \\")
    print("    --clinical_file output/hard_swahili_chat.json \\")
    print("    --healthcare_corpus_file /dev/null \\")
    print("    --accuracy_file /dev/null \\")
    print("    --max_len 384 --epochs 4 --lr 2e-5 --clinical_repeat 1 \\")
    print("    --output_dir output/jamii-lora-swahili-v2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
