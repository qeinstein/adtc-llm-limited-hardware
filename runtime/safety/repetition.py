"""Repetition / looping safeguard (Kiswahili stability + general).

Detects pathological looping via repeated n-grams plus completion/length
checks. Pure Python, no model call. Used by output_lint; a failure
triggers regen-with-concise-instruction, then the safe fallback.
"""
from __future__ import annotations

from collections import Counter

MAX_THINKING_CHARS = 3000
MAX_ANSWER_CHARS = 4000
NGRAM_N = 10
NGRAM_MAX_REPEATS = 3


def ngram_repeats(text: str, n: int = NGRAM_N) -> int:
    """Max repeat count of any word n-gram (0/1 = healthy)."""
    toks = text.split()
    if len(toks) < n + 1:
        return 0
    grams = Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return max(grams.values()) if grams else 0


def trailing_loop(text: str, min_tail: int = 200) -> bool:
    """True if the ending is a verbatim repeat of an earlier span."""
    toks = text.split()
    if len(toks) < 60:
        return False
    tail = " ".join(toks[-30:])
    head = " ".join(toks[:-30])
    return tail in head and len(tail) >= min_tail


def check_repetition(text: str, thinking: str = "") -> dict:
    """Return {"ok", "failures": [...], "stats": {...}}."""
    failures = []
    text = text or ""
    thinking = thinking or ""
    stats = {"answer_chars": len(text),
             "thinking_chars": len(thinking),
             "answer_ngram_repeats": ngram_repeats(text),
             "thinking_ngram_repeats": ngram_repeats(thinking)}
    if stats["answer_ngram_repeats"] >= NGRAM_MAX_REPEATS:
        failures.append("repetition-loop-answer")
    if stats["thinking_ngram_repeats"] >= NGRAM_MAX_REPEATS:
        failures.append("repetition-loop-thinking")
    if trailing_loop(text):
        failures.append("repetition-trailing-loop")
    if len(text) > MAX_ANSWER_CHARS:
        failures.append("answer-overlong")
    if len(thinking) > MAX_THINKING_CHARS:
        failures.append("thinking-overlong")
    return {"ok": not failures, "failures": failures, "stats": stats}
