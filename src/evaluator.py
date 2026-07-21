"""Offline bilingual clinical concept-recall evaluation.

Our own domain metric: does the model's answer surface the clinical concepts a
correct answer must contain (in English or Kiswahili)? This is deterministic,
offline, and needs no judge — useful for regression-testing the RAG stack and
for comparing candidate models on our own EN+SW eval set. It is a proxy, not the
official Sacc (the profiler runs lm-eval on the raw model; see src/accuracy.py).

Matching is robust: multi-word gold phrases are matched as normalized substrings;
single tokens use prefix-aware matching (so "referral" matches "refer", and
Swahili inflections match) instead of brittle exact equality.

The evaluator takes an ``answer_fn(query) -> str`` so it can be unit-tested with a
stub and run for real against the LLM engine — no weights required to test.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

_NORM_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_MIN_PREFIX = 4


def normalize(text: str) -> str:
    return _NORM_RE.sub(" ", text.lower())


def _prefix_hit(keyword: str, answer_tokens: set[str]) -> bool:
    if keyword in answer_tokens:
        return True
    if len(keyword) < _MIN_PREFIX:
        return False
    for tok in answer_tokens:
        if len(tok) < _MIN_PREFIX:
            continue
        n = min(len(keyword), len(tok))
        i = 0
        while i < n and keyword[i] == tok[i]:
            i += 1
        if i >= _MIN_PREFIX:
            return True
    return False


def keyword_matches(keyword: str, answer: str) -> bool:
    norm_answer = normalize(answer)
    kw = normalize(keyword).strip()
    if not kw:
        return False
    if " " in kw:  # multi-word phrase -> substring match on normalized text
        return kw in norm_answer
    return _prefix_hit(kw, set(_WORD_RE.findall(norm_answer)))


def score_answer(answer: str, gold_keywords: list[str]) -> tuple[float, list[str], list[str]]:
    if not gold_keywords:
        return 1.0, [], []
    matched = [kw for kw in gold_keywords if keyword_matches(kw, answer)]
    missed = [kw for kw in gold_keywords if kw not in matched]
    return len(matched) / len(gold_keywords), matched, missed


class ConceptRecallEvaluator:
    def __init__(self, eval_cases: list[dict[str, Any]]):
        self.eval_cases = eval_cases

    @classmethod
    def from_json(cls, path: Path | str) -> "ConceptRecallEvaluator":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def evaluate(
        self, answer_fn: Callable[[str], str], verbose: bool = False
    ) -> dict[str, Any]:
        results = []
        total = 0.0
        for idx, case in enumerate(self.eval_cases):
            query = case["query"]
            gold = case.get("gold_keywords", [])
            answer = answer_fn(query)
            acc, matched, missed = score_answer(answer, gold)
            total += acc
            results.append(
                {
                    "description": case.get("description", f"case {idx + 1}"),
                    "query": query,
                    "accuracy": round(acc, 3),
                    "matched": matched,
                    "missed": missed,
                }
            )
            if verbose:
                print(f"[{acc * 100:5.1f}%] {case.get('description', query[:50])}")
                if missed:
                    print(f"         missed: {missed}")
        mean = total / len(self.eval_cases) if self.eval_cases else 0.0
        return {"mean_accuracy": round(mean, 4), "n_cases": len(self.eval_cases), "results": results}
