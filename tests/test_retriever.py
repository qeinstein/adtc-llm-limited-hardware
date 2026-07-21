"""Tests for the stdlib BM25 retriever (no model weights required)."""

from src.retriever import BM25Retriever, tokenize

DOCS = [
    {"id": "d1", "title": "Malaria", "text": "High fever (homa kali) and chills; confirm with mRDT before ACT treatment."},
    {"id": "d2", "title": "Diarrhoea", "text": "Watery diarrhoea causes dehydration; give ORS and zinc."},
    {"id": "d3", "title": "Tuberculosis", "text": "Persistent cough over two weeks; night sweats and weight loss."},
]


def test_tokenize_lowercases_and_splits():
    assert tokenize("Homa Kali, mRDT!") == ["homa", "kali", "mrdt"]


def test_relevant_doc_ranks_first_english():
    r = BM25Retriever().fit(DOCS)
    top = r.retrieve("what test confirms malaria fever", top_n=1)
    assert top and top[0]["id"] == "d1"
    assert top[0]["score"] > 0


def test_relevant_doc_ranks_first_swahili():
    r = BM25Retriever().fit(DOCS)
    top = r.retrieve("mgonjwa ana homa kali", top_n=1)
    assert top and top[0]["id"] == "d1"


def test_diarrhoea_query_retrieves_ors_doc():
    r = BM25Retriever().fit(DOCS)
    top = r.retrieve("child dehydration ORS zinc", top_n=1)
    assert top[0]["id"] == "d2"


def test_top_n_and_score_filtering():
    r = BM25Retriever().fit(DOCS)
    res = r.retrieve("malaria", top_n=3)
    assert 1 <= len(res) <= 3
    # A query term absent from the corpus returns nothing (all scores 0).
    assert r.retrieve("cardiology angioplasty stent", top_n=3) == []


def test_empty_corpus_and_empty_query():
    assert BM25Retriever().retrieve("anything") == []
    r = BM25Retriever().fit(DOCS)
    assert r.retrieve("") == []


def test_title_boost_helps_ranking():
    r = BM25Retriever(title_weight=3).fit(DOCS)
    top = r.retrieve("tuberculosis", top_n=1)
    assert top[0]["id"] == "d3"
