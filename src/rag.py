"""Retrieval-augmented generation pipeline for the clinical advisor.

Wires the offline BM25 retriever and the query-focused compressor into a single
prompt-assembly step. RAG is the highest-ROI accuracy lever for a small model:
it grounds answers in curated WHO/IMCI-style guidance instead of relying on the
1.7B model's parametric memory. (It does NOT affect the profiler's automated
lm-eval score, which runs the raw model — RAG is for real answers + judges.)

Prompt layout is deliberately ``[stable system+few-shot] -> [RAG context] ->
[query]`` so the fixed prefix can be KV-cached across queries on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.compressor import compress_documents
from src.config import GUIDELINES_PATH, SYSTEM_PROMPT
from src.retriever import BM25Retriever, content_tokens

# Two short bilingual exemplars. Few-shot markedly lifts small-model quality and
# pins the expected answer shape (assessment -> action -> danger signs -> refer).
FEWSHOT = (
    "\n\nExample (English):\n"
    "Q: A child has watery diarrhoea and is very thirsty. What do I do?\n"
    "A: Assess dehydration (sunken eyes, slow skin pinch, restlessness). Start ORS "
    "after each loose stool and give zinc 20 mg daily for 10-14 days; continue "
    "feeding/breastfeeding. DANGER SIGNS (unable to drink, blood in stool, "
    "lethargy) -> refer urgently. This is decision support; confirm with a clinician.\n"
    "\nMfano (Kiswahili):\n"
    "S: Mtoto ana homa na anapumua haraka. Nifanye nini?\n"
    "J: Hesabu mipumuo kwa dakika moja (kupumua haraka ni ishara ya nimonia kwa "
    "watoto). Anza matibabu kwa mujibu wa mwongozo wa IMCI na hakikisha maji ya "
    "kutosha. ISHARA ZA HATARI (kushindwa kunyonya, degedege, kifua kinachozama) "
    "-> peleka haraka kituo cha rufaa. Huu ni ushauri wa kusaidia, si mbadala wa daktari."
)

SAFE_UNGROUNDED_EN = (
    "I don't have verified guidance for this specific question in the offline "
    "clinical corpus. Please consult a clinician or follow the national treatment "
    "guideline, especially if there are danger signs."
)
SAFE_UNGROUNDED_SW = (
    "Sina mwongozo uliohakikiwa kwa swali hili katika maktaba ya kliniki ya nje ya "
    "mtandao. Tafadhali wasiliana na daktari au fuata mwongozo wa kitaifa wa "
    "matibabu, hasa ikiwa kuna ishara za hatari."
)


def query_language(query: str) -> str:
    """Choose the language for a fixed, non-clinical safety response."""
    sw_markers = {
        "mtoto", "mgonjwa", "mjamzito", "homa", "kikohozi", "maumivu",
        "nifanye", "tafadhali", "dalili", "ishara", "hatari", "dawa",
    }
    terms = set(query.lower().replace("?", " ").replace(",", " ").split())
    return "sw" if terms & sw_markers else "en"


def ungrounded_response(query: str) -> str:
    """Return a safe response when the reviewed corpus has no relevant match."""
    return SAFE_UNGROUNDED_SW if query_language(query) == "sw" else SAFE_UNGROUNDED_EN



@dataclass
class RAGResult:
    query: str
    retrieved: list[dict[str, Any]]
    context: str
    user_content: str
    is_grounded: bool


def _has_sufficient_lexical_support(query: str, document: dict[str, Any]) -> bool:
    """Reject a weak partial BM25 hit before it reaches clinical generation.

    BM25 can return a plausible-looking document from one generic word (such as
    ``mtoto``/child) even when the question names no condition. A reviewed
    source must cover more than half of the query's content terms to count as
    grounding. This deliberately prefers a safe referral for underspecified,
    fabricated, or heavily misspelled questions.
    """
    terms = content_tokens(query)
    if not terms:
        return False
    source = f"{document.get('title', '')} {document.get('text', '')}"
    source_terms = set(content_tokens(source))
    matched = sum(term in source_terms for term in terms)
    coverage = matched / len(terms)
    if coverage > 0.5:
        return True
    # A short, bilingual query can legitimately contain two exact clinical
    # terms plus an inflected synonym absent from the source (for example,
    # ``mtoto ana homa kali na kikohozi``). Permit an exactly-half match only
    # when BM25's absolute evidence is strong; weak half-matches are the shape
    # seen for fabricated-drug prompts such as "dose of Zaptomycin".
    return coverage == 0.5 and float(document.get("score", 0.0)) >= 4.0


class RAGPipeline:
    def __init__(
        self,
        retriever: Optional[BM25Retriever] = None,
        guidelines_path: Path | str = GUIDELINES_PATH,
        use_fewshot: bool = True,
    ):
        if retriever is not None:
            self.retriever = retriever
        elif Path(guidelines_path).exists():
            self.retriever = BM25Retriever.from_json(guidelines_path)
        else:
            self.retriever = BM25Retriever()
        self.use_fewshot = use_fewshot

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT + (FEWSHOT if self.use_fewshot else "")

    def system_prompt_for(self, result: "RAGResult") -> str:
        """Same system prompt, minus the few-shot exemplars when nothing was retrieved.

        Why (found by real testing, not theory): FEWSHOT embeds two COMPLETE
        worked clinical answers. Given input with no clinical signal — a
        greeting, a thank-you, a typo — a 0.6B model has nothing to anchor on
        and just copies the nearest in-context example verbatim; a bare "hi"
        came back as the full ORS/zinc diarrhoea answer. Dropping the examples
        when there is no retrieved context removes the thing being copied and
        lets the model answer normally, rather than us enumerating greetings to
        intercept (no word list survives two languages plus typos).
        """
        return self.system_prompt if result.is_grounded else SYSTEM_PROMPT

    def build(
        self, query: str, top_n: int = 3, max_context_words: int = 220
    ) -> RAGResult:
        retrieved = self.retriever.retrieve(query, top_n=top_n)
        is_grounded = bool(retrieved) and _has_sufficient_lexical_support(query, retrieved[0])
        context = (
            compress_documents(query, retrieved, max_words=max_context_words)
            if is_grounded
            else ""
        )
        if context:
            user_content = (
                f"Reference guidance (retrieved from the offline clinical corpus):\n"
                f"{context}\n\n"
                f"Question:\n{query}"
            )
        else:
            user_content = query
        return RAGResult(
            query=query,
            retrieved=retrieved,
            context=context,
            user_content=user_content,
            is_grounded=is_grounded,
        )
