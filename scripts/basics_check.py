#!/usr/bin/env python3
"""Check the non-clinical behaviour that was broken — greetings, identity, general
questions, and knowing when to decline.

These are the exact inputs that exposed the problem: a bare "hii" returned the full
ORS/zinc diarrhoea answer, and "asante" produced invented malaria doses, because
every training example the model had seen was a clinical question. Judges will type
these before they type a triage scenario, so they are worth a dedicated check
separate from the clinical stress test.

    PYTHONPATH=. python scripts/basics_check.py --model model/....gguf
"""

from __future__ import annotations

import argparse
import sys

CASES = [
    ("greeting-en",   "hello"),
    ("greeting-typo", "hii"),
    ("greeting-sw",   "Habari yako?"),
    ("thanks-sw",     "asante"),
    ("identity",      "Who are you and what can you do?"),
    ("identity-sw",   "Wewe ni nani? Unaweza kunisaidia na nini?"),
    ("is-doctor",     "Are you a doctor? Can you prescribe medicine for me?"),
    ("offline",       "Do you need internet to work?"),
    ("general-fact",  "What is the capital of Kenya?"),
    ("general-sw",    "Nitachemsha vipi maji ili yawe salama kunywa?"),
    ("out-of-scope",  "Write me a poem about the rain."),
    ("literacy-sw",   "Malaria ni nini?"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    from src.engine import MedicalLLMEngine
    from src.rag import RAGPipeline

    rag = RAGPipeline()
    eng = MedicalLLMEngine(model_path=args.model)

    for label, q in CASES:
        res = rag.build(q, top_n=3)
        msgs = [
            {"role": "system", "content": rag.system_prompt_for(res)},
            {"role": "user", "content": res.user_content},
        ]
        out = eng.chat(msgs, max_tokens=args.max_tokens)
        print("=" * 74)
        print(f"[{label}]  retrieved={[d.get('id') for d in res.retrieved] or 'none'}")
        print(f"Q: {q}")
        print(f"A: {out['text']}")
    print("=" * 74)
    print("Judge these on: is it short? does it stay non-clinical for small talk? "
          "does it answer in the SAME language? does it avoid inventing doses?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
