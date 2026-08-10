"""Tests for the RAG assembly pipeline (no model weights required)."""

from src.rag import RAGPipeline, SAFE_UNGROUNDED_EN, SAFE_UNGROUNDED_SW, ungrounded_response
from src.retriever import BM25Retriever

DOCS = [
    {"id": "med_x", "title": "Malaria", "text": "Confirm malaria with an mRDT before giving ACT. High fever and chills are typical."},
    {"id": "med_y", "title": "Diarrhoea", "text": "Give ORS and zinc for watery diarrhoea; watch for dehydration danger signs."},
]


def _pipeline():
    return RAGPipeline(retriever=BM25Retriever().fit(DOCS))


def test_build_grounds_with_context():
    rag = _pipeline()
    res = rag.build("how do I confirm malaria", top_n=1)
    assert res.retrieved and res.retrieved[0]["id"] == "med_x"
    assert res.is_grounded
    assert res.context
    assert "Reference guidance" in res.user_content
    assert "Question:" in res.user_content


def test_build_without_match_falls_back_to_query():
    rag = _pipeline()
    res = rag.build("totally unrelated astrophysics query", top_n=2)
    assert res.context == ""
    assert res.user_content == "totally unrelated astrophysics query"
    assert not res.is_grounded


def test_weak_partial_match_is_not_treated_as_grounding():
    rag = _pipeline()
    res = rag.build("malaria zaptomycin", top_n=1)
    assert res.retrieved  # BM25 finds the generic malaria token.
    assert not res.is_grounded
    assert res.context == ""


def test_ungrounded_response_is_fixed_and_bilingual():
    assert ungrounded_response("unrelated astrophysics query") == SAFE_UNGROUNDED_EN
    assert ungrounded_response("Mtoto ana tatizo lisilojulikana") == SAFE_UNGROUNDED_SW


def test_no_rag_path_top_n_zero():
    rag = _pipeline()
    res = rag.build("malaria", top_n=0)
    assert res.retrieved == []
    assert res.user_content == "malaria"


def test_system_prompt_contains_safety_and_fewshot():
    rag = _pipeline()
    sp = rag.system_prompt
    assert "decision support" in sp.lower()
    assert "danger" in sp.lower()
    assert "Kiswahili" in sp  # bilingual few-shot present

    rag_no_fs = RAGPipeline(retriever=BM25Retriever().fit(DOCS), use_fewshot=False)
    assert "Example (English)" not in rag_no_fs.system_prompt
