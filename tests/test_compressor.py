"""Tests for query-focused extractive compression (no model weights required)."""

from src.compressor import clean_text, compress_context, compress_documents, split_sentences

DOC = (
    "Malaria is caused by Plasmodium parasites. Primary symptoms include high fever, "
    "chills, and headache. Diagnosis must be confirmed with an mRDT or blood smear. "
    "First-line treatment for uncomplicated cases is artemisinin-based combination "
    "therapy. Prevention uses insecticide-treated bed nets and indoor spraying. "
    "Danger signs include convulsions, inability to drink, and lethargy."
)


def test_clean_and_split():
    assert "##" not in clean_text("## Heading\n\nBody text here.")
    assert len(split_sentences(DOC)) >= 5


def test_compression_respects_word_budget():
    out = compress_context("how is malaria diagnosed", DOC, max_words=15)
    assert 0 < len(out.split()) <= 20  # small overshoot tolerance for one sentence


def test_compression_selects_relevant_sentence():
    out = compress_context("how is malaria diagnosed", DOC, max_words=25)
    assert "mrdt" in out.lower() or "smear" in out.lower()


def test_compression_preserves_original_order():
    out = compress_context("treatment and diagnosis", DOC, max_words=60)
    # 'Diagnosis' sentence precedes 'treatment' sentence in the source order.
    assert out.lower().index("diagnosis") < out.lower().index("treatment")


def test_empty_inputs():
    assert compress_context("q", "") == ""
    assert compress_context("", DOC, max_words=20)  # no query -> leading fallback


def test_compress_documents_joins_docs():
    docs = [{"text": "ORS treats dehydration."}, {"text": "Zinc reduces diarrhoea duration."}]
    out = compress_documents("dehydration treatment", docs, max_words=30)
    assert "ors" in out.lower()
