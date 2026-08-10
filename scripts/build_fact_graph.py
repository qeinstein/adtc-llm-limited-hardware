#!/usr/bin/env python3
"""Decompose the 32 verified guidelines into atomic, bilingual, source-traceable
facts -- the offline half of a zero-training answer architecture.

WHY: everything tried today (repetition guards, refusal training, retrieval
thresholds) made hallucination LESS LIKELY. None made it IMPOSSIBLE. The model
invented "Zaptomycin" doses and degenerated into invented Kiswahili words because
it was asked to freely author content -- exactly the one thing a 0.6B model
cannot be trusted to do reliably (measured today: 79% accurate at RANKING fixed
choices, unreliable at GENERATING novel text).

This script does the one-time, offline, unlimited-compute work of turning our 32
guidelines into a small database of atomic facts (assess / action / dose /
danger_sign / refer), each one a DIRECT EXTRACT of the source text -- the teacher
model is instructed to copy, not paraphrase, and every fact is checked against
the source guideline text before being kept. At runtime, the model never writes
free text for content: it only SELECTS which pre-verified facts are relevant
(the ranking task it is actually good at), and a fixed template renders them.
A fact that cannot be found verbatim in its source guideline is not a fact this
system can ever say.

    python scripts/build_fact_graph.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"

SYSTEM = (
    "You extract atomic clinical facts from ONE verified WHO/IMCI guideline. "
    "You must COPY exact phrases from the source text -- do not paraphrase, do not "
    "add information, do not invent numbers or drug names not present in the source. "
    "If the source has no dose, leave dose facts out entirely rather than estimate one. "
    "Output STRICT JSON only, no markdown fences."
)

TASK = """Source guideline:
TITLE: {title}
TEXT: {text}

Extract every atomic clinical fact as one of these types:
  assess      - a sign or symptom to check for
  action      - a non-drug action to take (e.g. give ORS, isolate the child)
  dose        - a SPECIFIC drug + dose stated in the source (skip if none is given)
  danger_sign - a sign that means urgent referral is needed
  refer       - a referral instruction

Each fact's "text_en" and "text_sw" must be near-verbatim extracts from the source
(the source already contains both languages inline -- pull the matching Kiswahili
term/phrase that appears next to the English one; if no Kiswahili equivalent is
present in the source for a given fact, leave text_sw empty rather than translate).

JSON: {{"facts": [{{"type": "assess|action|dose|danger_sign|refer",
                    "text_en": "...", "text_sw": "..."}}]}}"""


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
        c = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                   default_headers={"X-Title": "Jamii Afya fact graph"})
        if model == "gpt-4o-mini":
            model = "google/gemini-2.5-flash"
        return c, model
    return OpenAI(), model


# A fact "counts as verified" only if a normalized version of its extracted text is a
# substring of the normalized source. This is the hard gate: an extraction that
# doesn't pass this check is a paraphrase or an invention, and is dropped, not kept.
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _verified(fact_text: str, source: str) -> bool:
    if not fact_text or len(fact_text) < 6:
        return False
    ft, src = _norm(fact_text), _norm(source)
    return ft in src


def main() -> int:
    guidelines = json.loads((DATA / "medical_guidelines.json").read_text())
    client, model = build_client("gpt-4o-mini")

    graph: list[dict] = []
    kept, dropped = 0, 0

    for g in guidelines:
        gid, title, text = g["id"], g["title"], g["text"]
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.0,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": TASK.format(title=title, text=text)}],
            )
            facts = _extract_json(resp.choices[0].message.content).get("facts", [])
        except Exception as e:
            print(f"  [warn] {gid}: {type(e).__name__}: {e}", flush=True)
            continue

        g_kept = 0
        for f in facts:
            en = (f.get("text_en") or "").strip()
            sw = (f.get("text_sw") or "").strip()
            ftype = (f.get("type") or "").strip()
            if ftype not in ("assess", "action", "dose", "danger_sign", "refer"):
                continue
            en_ok = _verified(en, text)
            sw_ok = (not sw) or _verified(sw, text)
            if en_ok and sw_ok:
                graph.append({
                    "guideline_id": gid, "guideline_title": title,
                    "type": ftype, "text_en": en, "text_sw": sw,
                })
                kept += 1
                g_kept += 1
            else:
                dropped += 1
        print(f"  {gid}  {title[:44]:44s} +{g_kept:2d} facts (kept)", flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / "fact_graph.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False))
    print(f"\n{kept} verified facts kept, {dropped} rejected (not found in source text)")
    print(f"-> {OUT / 'fact_graph.json'}")
    by_type: dict[str, int] = {}
    for f in graph:
        by_type[f["type"]] = by_type.get(f["type"], 0) + 1
    print("by type:", by_type)
    return 0


if __name__ == "__main__":
    sys.exit(main())
