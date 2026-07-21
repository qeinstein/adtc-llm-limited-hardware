"""Query-focused extractive context compression.

Retrieved guideline text is squeezed to a small, high-signal set of sentences
before it enters the prompt. This keeps prefill short, which on CPU is the main
cost and also bounds KV-cache RAM. Pure standard library.

Scoring: each sentence is scored by the query terms it contains, weighted by how
*rare* each term is across the candidate sentences (a local IDF, so generic words
like "the"/"na" don't dominate), and lightly biased toward earlier sentences
(definitions/summaries tend to come first in guideline text). Selection is greedy
under a word budget; selected sentences are then restored to their original order
so the passage still reads coherently.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable

from src.retriever import STOPWORDS

_WS_RE = re.compile(r"\s+")
_MD_RE = re.compile(r"[#*_`>]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\w+", re.UNICODE)


_MIN_PREFIX = 4  # min shared-prefix length to treat two tokens as the same concept


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _match_weight(q: str, sent_terms: set[str]) -> float:
    """1.0 for an exact hit, 0.7 for a morphological (shared-prefix) hit, else 0.

    Language-agnostic: connects diagnose/diagnosed/diagnosis, treat/treatment,
    dehydrate/dehydration, and Swahili inflections, without an English-only stemmer.
    """
    if q in sent_terms:
        return 1.0
    if len(q) < _MIN_PREFIX:
        return 0.0
    for s in sent_terms:
        if len(s) >= _MIN_PREFIX and _common_prefix_len(q, s) >= _MIN_PREFIX:
            return 0.7
    return 0.0


def clean_text(text: str) -> str:
    text = _MD_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if len(s.strip()) > 5]


def compress_context(query: str, document_text: str, max_words: int = 220) -> str:
    """Compress ``document_text`` to the most query-relevant sentences.

    Returns a string of at most ~``max_words`` words, sentences in original order.
    """
    sentences = split_sentences(clean_text(document_text))
    if not sentences:
        return ""

    query_terms = {t for t in _WORD_RE.findall(query.lower()) if len(t) > 1 and t not in STOPWORDS}
    if not query_terms:
        # No query signal: fall back to the leading sentences under budget.
        return _take_in_order(sentences, list(range(len(sentences))), max_words)

    # Local IDF: how many sentences each query term appears in (prefix-aware).
    sent_terms = [set(_WORD_RE.findall(s.lower())) for s in sentences]
    df: Counter[str] = Counter()
    for terms in sent_terms:
        for t in query_terms:
            if _match_weight(t, terms) > 0:
                df[t] += 1
    n = len(sentences)
    term_weight = {t: math.log(1.0 + n / (df[t] + 0.5)) for t in query_terms}

    scores: list[float] = []
    for idx, terms in enumerate(sent_terms):
        overlap = sum(term_weight[t] * _match_weight(t, terms) for t in query_terms)
        length_norm = math.sqrt(len(terms) + 1)
        position_bias = 1.0 / (idx + 2.0)
        scores.append(overlap / length_norm + 0.15 * position_bias)

    ranked = sorted(range(n), key=lambda i: (scores[i], -i), reverse=True)

    chosen: list[int] = []
    words = 0
    for i in ranked:
        if scores[i] <= 0:
            continue
        wc = len(sentences[i].split())
        if words + wc > max_words and chosen:
            continue
        chosen.append(i)
        words += wc
        if words >= max_words:
            break

    if not chosen:  # nothing overlapped: fall back to leading sentences
        return _take_in_order(sentences, list(range(len(sentences))), max_words)

    return _take_in_order(sentences, chosen, max_words)


def _take_in_order(sentences: list[str], indices: Iterable[int], max_words: int) -> str:
    ordered = sorted(set(indices))
    out: list[str] = []
    words = 0
    for i in ordered:
        wc = len(sentences[i].split())
        if words + wc > max_words and out:
            break
        out.append(sentences[i])
        words += wc
    return " ".join(out)


def compress_documents(
    query: str, docs: list[dict[str, Any]], max_words: int = 220
) -> str:
    """Compress a list of retrieved docs into one query-focused context block."""
    joined = "\n".join(str(d.get("text", "")) for d in docs)
    return compress_context(query, joined, max_words=max_words)
