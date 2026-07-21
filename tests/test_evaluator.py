"""Tests for the offline concept-recall evaluator (no model weights required)."""

from src.evaluator import ConceptRecallEvaluator, keyword_matches, score_answer


def test_exact_and_prefix_token_match():
    ans = "Refer the patient urgently and start ACT after a positive mRDT."
    assert keyword_matches("referral", ans)  # refer~referral (prefix)
    assert keyword_matches("act", ans)
    assert keyword_matches("mrdt", ans)
    assert not keyword_matches("insulin", ans)


def test_phrase_match_is_substring():
    ans = "Sponge the child with maji ya uvuguvugu and give paracetamol."
    assert keyword_matches("maji ya uvuguvugu", ans)
    assert not keyword_matches("maji ya moto", ans)


def test_short_keyword_needs_exact():
    # 'tb' is < min prefix length, so only an exact token hit counts.
    assert keyword_matches("tb", "screen for tb with genexpert")
    assert not keyword_matches("tb", "the table is set")  # no false prefix hit


def test_score_answer_fraction():
    acc, matched, missed = score_answer("give ors and zinc", ["ors", "zinc", "referral"])
    assert matched == ["ors", "zinc"]
    assert missed == ["referral"]
    assert abs(acc - 2 / 3) < 1e-9


def test_evaluator_with_stub_answer_fn():
    cases = [
        {"query": "malaria test?", "gold_keywords": ["mrdt", "act"], "description": "malaria"},
        {"query": "diarrhoea?", "gold_keywords": ["ors", "zinc"], "description": "diarrhoea"},
    ]
    ev = ConceptRecallEvaluator(cases)

    def stub(q: str) -> str:
        return "confirm with mRDT then give ACT" if "malaria" in q else "give ORS only"

    report = ev.evaluate(stub)
    assert report["n_cases"] == 2
    # case1 = 2/2, case2 = 1/2 -> mean 0.75
    assert abs(report["mean_accuracy"] - 0.75) < 1e-9
