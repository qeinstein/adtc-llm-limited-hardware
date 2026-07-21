"""Offline sparse retrieval (BM25) over the bilingual clinical corpus.

Pure standard library — no numpy, no network, no model. Chosen deliberately:
BM25 over a curated corpus is fast, transparent, and RAM-negligible on the
target hardware, and it works for both English and Kiswahili because both are
Latin-script (a shared ``\\w+`` tokenizer covers both). This keeps the retrieval
layer dependency-free so it runs anywhere and is trivially unit-testable without
model weights.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Bilingual function words. Kiswahili is agglutinative and its connectors (na, wa,
# ya, za, kwa...) appear in almost every sentence — without removing them, a query
# like "...na... wa... za..." matches on grammar instead of clinical content
# (this genuinely mis-ranked our flagship pediatric prompt before it was added).
STOPWORDS: frozenset[str] = frozenset(
    # English
    "a an the and or of to in is are am was were be been for on with as at by it its "
    "this that these those i you he she they we do does did what which how when where "
    "who whom should could would can may might will my me your his her their our if "
    "then than so but also not no yes into from about over under out up down".split()
    # Kiswahili (function/grammatical words, question words, common connectors)
    + (
        "na wa ya za la kwa ni katika kama ana au ili tu pia kila huu hii huyu huyo "
        "hilo haya hizo hizi huo hivi hivyo wako wangu yake zake wake gani nini je "
        "ndani juu chini sana kwenye kuwa cha vya pa mwa wenye lakini ambaye ambao "
        "ambayo ambacho hapa huku pale yeye wao mimi sisi nyinyi kwenda vile hivi "
        "nifanye nini bila baada kabla".split()
    )
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer (handles English + Kiswahili, both Latin script)."""
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokenize and drop bilingual stopwords / 1-char tokens (keeps numbers like '39')."""
    return [t for t in tokenize(text) if len(t) > 1 and t not in STOPWORDS]


class BM25Retriever:
    """Okapi BM25 retriever with title boosting.

    Documents are dicts; ``text`` is required, ``title`` is optional and boosted
    (its terms count ``title_weight`` times) because titles are dense with the
    condition name in both languages.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, title_weight: int = 2):
        self.k1 = k1
        self.b = b
        self.title_weight = title_weight
        self.documents: list[dict[str, Any]] = []
        self._doc_tf: list[Counter[str]] = []
        self._doc_len: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    # -- indexing -------------------------------------------------------------
    def _doc_tokens(self, doc: dict[str, Any]) -> list[str]:
        title = str(doc.get("title", ""))
        body = str(doc.get("text", ""))
        return content_tokens(title) * self.title_weight + content_tokens(body)

    def fit(self, documents: Iterable[dict[str, Any]]) -> "BM25Retriever":
        self.documents = list(documents)
        self._doc_tf = []
        self._doc_len = []
        df: Counter[str] = Counter()

        for doc in self.documents:
            tokens = self._doc_tokens(doc)
            tf = Counter(tokens)
            self._doc_tf.append(tf)
            self._doc_len.append(len(tokens))
            df.update(tf.keys())

        n = len(self.documents)
        self._avgdl = (sum(self._doc_len) / n) if n else 0.0
        # BM25 idf with +1 smoothing so it is always positive.
        self._idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        return self

    @classmethod
    def from_json(cls, path: Path | str, **kwargs: Any) -> "BM25Retriever":
        with open(path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        return cls(**kwargs).fit(docs)

    # -- scoring --------------------------------------------------------------
    def _score(self, query_terms: list[str], idx: int) -> float:
        tf = self._doc_tf[idx]
        dl = self._doc_len[idx]
        if dl == 0:
            return 0.0
        denom_norm = self.k1 * (1.0 - self.b + self.b * dl / (self._avgdl or 1.0))
        score = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if not f:
                continue
            idf = self._idf.get(term, 0.0)
            score += idf * (f * (self.k1 + 1.0)) / (f + denom_norm)
        return score

    def retrieve(
        self, query: str, top_n: int = 3, min_score: float = 1e-9
    ) -> list[dict[str, Any]]:
        """Return the ``top_n`` best-matching documents (with a ``score`` field)."""
        if not self.documents:
            return []
        query_terms = content_tokens(query)
        if not query_terms:
            return []
        scored = [
            (self._score(query_terms, i), i) for i in range(len(self.documents))
        ]
        scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)
        results: list[dict[str, Any]] = []
        for score, i in scored[:top_n]:
            if score <= min_score:
                break
            results.append({**self.documents[i], "score": round(score, 4)})
        return results

    def __len__(self) -> int:
        return len(self.documents)
