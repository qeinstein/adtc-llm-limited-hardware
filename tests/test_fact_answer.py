"""Unit tests for the constrained, verified-fact renderer.

No model is needed here: the model-scoring boundary is replaced with fixed
scores, allowing selection and safe fallback policy to be tested deterministically.
"""

from src import fact_answer


FACTS = [
    {
        "guideline_id": "malaria", "type": "assess",
        "text_en": "Check with an mRDT.", "text_sw": "Pima kwa mRDT.",
    },
    {
        "guideline_id": "malaria", "type": "action",
        "text_en": "Give treatment according to the result.", "text_sw": "Toa matibabu kulingana na majibu.",
    },
    {
        "guideline_id": "malaria", "type": "danger_sign",
        "text_en": "Refer urgently if unable to drink.", "text_sw": "Peleka haraka akishindwa kunywa.",
    },
]


def test_answer_renders_only_selected_verified_facts(monkeypatch):
    monkeypatch.setattr(fact_answer, "_load_graph", lambda: FACTS)
    scores = {f["text_en"]: -0.20 for f in FACTS}
    monkeypatch.setattr(fact_answer, "_score_fact", lambda _llm, _q, text: scores[text])

    out = fact_answer.answer(object(), "How do I assess malaria?", guideline_ids=["malaria"])

    assert not out.used_fallback
    assert out.guideline_ids == ["malaria"]
    assert "Check for: Check with an mRDT." in out.text
    assert "DANGER SIGNS" in out.text
    assert len(out.selected) == 3


def test_answer_refuses_when_no_fact_meets_absolute_threshold(monkeypatch):
    monkeypatch.setattr(fact_answer, "_load_graph", lambda: FACTS)
    monkeypatch.setattr(fact_answer, "_score_fact", lambda *_args: -0.50)

    out = fact_answer.answer(object(), "Dose of invented drug")

    assert out.used_fallback
    assert out.text == fact_answer.SAFE_FALLBACK_EN
    assert out.selected == []


def test_answer_uses_swahili_verified_text(monkeypatch):
    monkeypatch.setattr(fact_answer, "_load_graph", lambda: FACTS)
    monkeypatch.setattr(fact_answer, "_score_fact", lambda *_args: -0.20)

    out = fact_answer.answer(object(), "nifanye nini", lang="sw")

    assert not out.used_fallback
    assert "Angalia: Pima kwa mRDT." in out.text
    assert "ISHARA ZA HATARI" in out.text
