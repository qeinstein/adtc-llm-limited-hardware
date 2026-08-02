#!/usr/bin/env python3
"""Generate the general-conversation half of the SFT set (EN + Kiswahili).

WHY: the model could only do clinical Q&A. A bare "hi" produced the full ORS/zinc
diarrhoea answer, and "asante" produced invented malaria doses — because every
training example it had ever seen was a clinical question, so it had no notion
that some inputs are not clinical. Judges will absolutely type "hello" and "what
can you do?" before they type a triage scenario.

Four categories, all bilingual:
  identity     — who/what it is, what it can and cannot do. Grounded in FACTS we
                 supply below, not left to the teacher's imagination, so the model
                 does not learn to claim it is a doctor or that it works online.
  social       — greetings, thanks, goodbye. Short answers that redirect to the
                 clinical task instead of free-associating.
  general      — ordinary non-medical questions. The model must degrade gracefully
                 rather than answer everything with danger signs.
  literacy     — plain-language health explainers ("what is malaria?"), which sit
                 between small talk and triage and are what a CHW actually asks.

Answers are deliberately SHORT. Long targets are what taught it to ramble.

    export OPENROUTER_API_KEY=...   # or put it in .env
    python scripts/build_general_sft.py --per-category 60
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

# Ground truth about the product. The teacher must not invent these.
FACTS = """
- Name: Jamii Afya. An offline clinical decision-support assistant.
- Users: community health workers and nurses in rural African clinics.
- Runs fully offline on an ordinary laptop, CPU only. No internet, no cloud, no
  patient data ever leaves the device.
- Languages: English and Kiswahili. It replies in whichever the user used.
- It is DECISION SUPPORT, not a doctor and not a diagnosis. It cannot examine,
  prescribe, or replace a clinician.
- Its clinical answers come from WHO/IMCI-based primary-care guidelines stored
  on the device.
- It is a small model. It should say plainly when it does not know something
  rather than guess, and should never invent doses.
"""

SYSTEM = (
    "You generate training data for a bilingual (English/Kiswahili) offline medical "
    "assistant. Output STRICT JSON only — no markdown fences, no commentary. "
    "Kiswahili must be natural and idiomatic, not word-for-word translated English."
)

PROMPTS = {
    "identity": """Facts about the assistant (treat as ground truth, do not contradict):
{facts}

Generate {n} question/answer pairs where a user asks about the assistant itself —
who it is, what it can do, whether it is a doctor, whether it works without
internet, what languages it speaks, whether their data is private, what it should
not be used for. Roughly half in English, half in natural Kiswahili.

Answers: 1-3 sentences, warm but plain. Accurate to the facts above. Where relevant
make clear it supports rather than replaces a clinician.

JSON: {{"pairs":[{{"q":"...","a":"..."}}]}}""",

    "social": """Generate {n} short social exchanges for the assistant described here:
{facts}

Greetings, thanks, goodbyes, "how are you", small talk. Roughly half English, half
natural Kiswahili (hujambo, habari, asante, karibu, kwaheri, etc.).

Answers must be ONE short sentence, friendly, and gently invite the health worker
to describe the patient or ask a clinical question. Do NOT give medical advice in
these — that is the exact failure being trained out.

JSON: {{"pairs":[{{"q":"...","a":"..."}}]}}""",

    "general": """Generate {n} ordinary NON-medical questions a person might type, with
good answers, for the assistant described here:
{facts}

Mix: simple factual questions (capital cities, basic arithmetic, what a word means,
days of the week), practical questions (how to store something, how to boil water
safely), and a few questions clearly outside its scope (write me a poem, who will
win the election, what is the weather today).

Roughly half English, half natural Kiswahili. Answer the answerable ones briefly
and correctly. For out-of-scope or unknowable ones, say plainly that it cannot help
with that and steer back to what it is for. 1-3 sentences. Never refuse a simple
factual question it can obviously answer.

JSON: {{"pairs":[{{"q":"...","a":"..."}}]}}""",

    "literacy": """Generate {n} plain-language health-literacy questions and answers for
the assistant described here:
{facts}

Things like: what is malaria, how does a mosquito net help, why finish antibiotics,
what does dehydration mean, why are vaccines given, what is a danger sign. General
explanation — NOT diagnosis or dosing for a specific patient.

Roughly half English, half natural Kiswahili. Answers 2-4 sentences, simple enough
for a community health worker to repeat to a patient. Include no specific drug doses.

JSON: {{"pairs":[{{"q":"...","a":"..."}}]}}""",
}


def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b == -1:
        raise ValueError("no JSON object in teacher output")
    return json.loads(text[a : b + 1])


def build_client(model: str):
    import os

    from openai import OpenAI

    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            default_headers={
                "HTTP-Referer": "https://github.com/qeinstein/adtc-llm-limited-hardware",
                "X-Title": "Jamii Afya general SFT",
            },
        )
        if model == "gpt-4o-mini":
            model = "google/gemini-2.5-flash"
        return client, model
    return OpenAI(), model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--per-category", type=int, default=60)
    ap.add_argument("--batch", type=int, default=20, help="pairs requested per API call")
    ap.add_argument("--out", default=str(OUT / "general_sft.json"))
    args = ap.parse_args()

    client, model = build_client(args.model)
    OUT.mkdir(exist_ok=True)
    rows: list[dict] = []

    for cat, template in PROMPTS.items():
        got = 0
        while got < args.per_category:
            n = min(args.batch, args.per_category - got)
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user",
                         "content": template.format(facts=FACTS, n=n)},
                    ],
                    temperature=0.9,   # variety matters more than precision here
                )
                pairs = _extract_json(resp.choices[0].message.content).get("pairs", [])
            except Exception as e:
                print(f"  [warn] {cat}: {type(e).__name__}: {e}")
                break
            new = 0
            for p in pairs:
                q, a = (p.get("q") or "").strip(), (p.get("a") or "").strip()
                if q and a and len(a) <= 600:
                    rows.append({"instruction": q, "input": "", "output": a})
                    new += 1
            got += new
            print(f"  {cat:9s} +{new:3d}  (total {got}/{args.per_category})", flush=True)
            if new == 0:
                break

    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    lens = [len(r["output"]) for r in rows] or [0]
    print(f"\n{len(rows)} general rows -> {args.out}")
    print(f"answer length: min={min(lens)} mean={sum(lens)//len(lens)} max={max(lens)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
