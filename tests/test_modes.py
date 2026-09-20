"""Unit tests for Fast / Medium / High reasoning modes (src/modes.py)."""
import pytest

from src.modes import (
    DEFAULT_MODE,
    MODES,
    budgets,
    describe,
    needs_answer_regen,
    normalize_mode,
    phase1_max_tokens,
    phase2_messages,
    steering,
)


def test_default_is_medium():
    assert DEFAULT_MODE == "medium"
    assert normalize_mode(None) == "medium"
    assert normalize_mode("") == "medium"
    assert normalize_mode("   ") == "medium"


def test_normalize_case_and_whitespace():
    assert normalize_mode("HIGH") == "high"
    assert normalize_mode(" Fast ") == "fast"
    assert normalize_mode("Medium") == "medium"


def test_normalize_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_mode("turbo")
    with pytest.raises(ValueError):
        normalize_mode("med")
    # surrounding whitespace is fine (stripped) — must NOT raise:
    assert normalize_mode("medium ") == "medium"


def test_exactly_three_modes():
    assert sorted(MODES) == ["fast", "high", "medium"]


def test_budgets_split_thinking_and_answer():
    assert budgets("fast") == (128, 256)
    assert budgets("medium") == (512, 512)
    assert budgets("high") == (1536, 768)
    for mode in MODES:
        think, answer = budgets(mode)
        assert think > 0 and answer > 0  # answer always reserved


def test_high_is_bounded():
    # Worst single-phase completion must leave room for the ~1400-token
    # prompt inside the 4096 serving context.
    assert phase1_max_tokens("high") == 2304
    assert phase1_max_tokens("high") + 1400 < 4096
    # Budgets strictly increase with effort.
    assert phase1_max_tokens("fast") < phase1_max_tokens("medium")
    assert phase1_max_tokens("medium") < phase1_max_tokens("high")


def test_steering_medium_is_neutral():
    assert steering("medium") == ""
    assert steering("fast")
    assert steering("high")


def test_needs_answer_regen_matrix():
    assert needs_answer_regen("some thinking", "") is True
    assert needs_answer_regen("some thinking", "   ") is True
    assert needs_answer_regen("thinking", "answer") is False
    assert needs_answer_regen("", "") is False  # nothing at all: not regen's job
    assert needs_answer_regen("", "answer") is False


def test_phase2_appends_thinking_and_instruction():
    base = [{"role": "system", "content": "S"},
            {"role": "user", "content": "Q"}]
    out = phase2_messages(base, "my Reasoning")
    assert out[:2] == base
    assert out[2] == {"role": "assistant", "content": "my Reasoning"}
    assert out[3]["role"] == "user"
    assert "final answer" in out[3]["content"].lower()


def test_phase2_truncates_huge_thinking():
    base = [{"role": "user", "content": "Q"}]
    out = phase2_messages(base, "x" * 20000, max_thinking_chars=6000)
    assert len(out[1]["content"]) == 6000


def test_describe():
    d = describe("high")
    assert d == {"mode": "high", "thinking_tokens": 1536,
                 "answer_tokens": 768, "max_tokens": 2304}
