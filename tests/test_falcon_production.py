from pathlib import Path

import pytest

from scripts.build_falcon_dataset import canonical, near_holdout
from scripts.train_falcon_production import FalconDataset, latest_checkpoint


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking=False):
        assert tokenize is False
        assert add_generation_prompt is True
        del enable_thinking
        return " | ".join(m["content"] for m in messages) + " | assistant:"

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": list(range(1, len(text.split()) + 1))}


def test_sft_masks_prompt_and_keeps_eos():
    rows = [{
        "format": "sft", "example_id": "s1", "source": "test",
        "instruction": "What is urgent?", "input": "", "output": "Refer now.",
    }]
    item = FalconDataset(rows, FakeTokenizer(), 32, "System")[0]
    prompt_len = item["labels"].index(next(x for x in item["labels"] if x != -100))
    assert prompt_len > 0
    assert all(x == -100 for x in item["labels"][:prompt_len])
    assert item["labels"][-1] == 99
    assert item["tokens"] == len(item["labels"]) - prompt_len


def test_sft_rejects_target_that_cannot_fit():
    rows = [{
        "format": "sft", "example_id": "long", "source": "test",
        "instruction": "Question", "input": "", "output": "one two three four five",
    }]
    with pytest.raises(ValueError, match="failed training tokenization"):
        FalconDataset(rows, FakeTokenizer(), 3, "System")


def test_mcqa_rejects_choice_truncation_instead_of_training_partial_answer():
    rows = [{
        "format": "mcqa", "example_id": "m1", "source": "test",
        "context": "one two three", "choices": ["red", "blue"], "gold": 0,
    }]
    with pytest.raises(ValueError, match="failed training tokenization"):
        FalconDataset(rows, FakeTokenizer(), 3, "System")


def test_holdout_gate_catches_exact_and_near_duplicates():
    assert near_holdout("A child has severe chest indrawing and needs urgent referral", [canonical("A child has severe chest indrawing and needs urgent referral")]) == "exact"
    assert near_holdout("A child has severe chest indrawing and needs urgent referral today", [canonical("A child has severe chest indrawing and needs urgent referral")]) == "near"
    assert near_holdout("What is two plus two?", [canonical("A child has severe chest indrawing and needs urgent referral")]) is None


def test_latest_checkpoint_ignores_incomplete_directories(tmp_path: Path):
    complete = tmp_path / "checkpoint-25"
    complete.mkdir()
    (complete / "trainer_state.json").write_text("{}")
    incomplete = tmp_path / "checkpoint-50"
    incomplete.mkdir()
    assert latest_checkpoint(tmp_path) == complete
