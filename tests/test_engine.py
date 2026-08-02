"""Tests for the repetition-guard used to catch degenerate chat generation
(found via manual qualitative testing on real Kiswahili prompts -- the
fine-tuned model reliably loops on longer, complex Kiswahili input even with
a raised repeat_penalty, so this is a real generation-time safety net, not a
hypothetical edge case)."""

from src.engine import SAFETY_FALLBACK, _guard_repetition


def test_normal_text_passes_through_unchanged():
    text = "Assess the child for danger signs. Refer urgently if needed. This is decision support."
    assert _guard_repetition(text) == text


def test_pure_repetition_falls_back_to_safety_message():
    looped = (
        "Kipindupisha rufaa cha kutumia kutambu au kupima chuma haraka safi huu anapompezuka "
        * 5
    )
    assert _guard_repetition(looped) == SAFETY_FALLBACK


def test_repetition_after_clean_sentence_truncates_to_that_sentence():
    mixed = "Assess for danger signs immediately. " + "loop chunk words here again and again " * 4
    assert _guard_repetition(mixed) == "Assess for danger signs immediately."


def test_real_swahili_loop_from_stress_test_log_is_truncated():
    prefix = (
        "Dalili ya salama na mazoezi kwa dakika 10 tu wakati wa kunyonya, "
        "ni dalila za daktari. "
    )
    phrase = (
        "Kipindupisha rufaa cha kutumia (kutambu) au kupima chuma haraka safi; "
        "huu anapompezuka na mafuta kwa dunawitana ya hatari zaidesha kunyonya, "
        "ni dalili za daktori. "
    )
    real_case = prefix + phrase * 6
    out = _guard_repetition(real_case)
    # The sentence-level pass keeps the FIRST occurrence of the repeated sentence
    # (that one is legitimate content) and cuts from the second onward. An earlier
    # version discarded the first occurrence too, which threw away a usable answer.
    assert out == (prefix + phrase).strip()
    assert out.count("Kipindupisha") == 1


def test_repetition_starting_mid_clause_falls_back():
    midclause = "The patient has fever and " + "chunk repeats here now yes indeed " * 4
    assert _guard_repetition(midclause) == SAFETY_FALLBACK
