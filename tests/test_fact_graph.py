"""Ensure the fact-graph builder enforces its advertised exact-source gate."""

from scripts.build_fact_graph import _verified


def test_verification_accepts_normalized_exact_source_substring():
    assert _verified("Give zinc 20 mg daily", "Give zinc: 20 mg daily for 10 days.")


def test_verification_rejects_partial_paraphrase():
    source = "Refer urgently if the child is unable to drink or breastfeed."
    assert not _verified("Refer urgently if the child cannot drink and give ORS.", source)
