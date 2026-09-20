"""Tests for the RAG assembly pipeline (no model weights required)."""

from src.rag import RAGPipeline
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


def test_no_rag_path_top_n_zero():
    rag = _pipeline()
    res = rag.build("malaria", top_n=0)
    assert res.retrieved == []
    assert res.user_content == "malaria"


def test_system_prompt_is_the_only_instruction_layer():
    rag = _pipeline()
    sp = rag.system_prompt
    assert "MEDICATIONS" in sp
    assert "explicitly asked" in sp
    assert rag.system_prompt_for(_pipeline().build("hello")) == sp
