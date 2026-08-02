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
from src.retriever import BM25Retriever

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



@dataclass
class RAGResult:
    query: str
    retrieved: list[dict[str, Any]]
    context: str
    user_content: str


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
        return self.system_prompt if result.retrieved else SYSTEM_PROMPT

    def build(
        self, query: str, top_n: int = 3, max_context_words: int = 220
    ) -> RAGResult:
        retrieved = self.retriever.retrieve(query, top_n=top_n)
        context = (
            compress_documents(query, retrieved, max_words=max_context_words)
            if retrieved
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
            query=query, retrieved=retrieved, context=context, user_content=user_content
        )
