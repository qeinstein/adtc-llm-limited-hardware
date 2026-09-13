from pathlib import Path

import pytest

from scripts.build_falcon_dataset import canonical, distribution, near_holdout
from scripts.train_falcon_production import FalconDataset, checkpoint_is_complete, latest_checkpoint, select_checkpoint, stable_eval_subset, token_share_sampling_weights


class FakeTokenizer:
    eos_token_id = 99
    im_end_token_id = 98
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking=False):
        assert tokenize is False
        del enable_thinking
        rendered = " | ".join(m["content"] for m in messages[:2]) + " | assistant:"
        if not add_generation_prompt:
            rendered += " " + messages[-1]["content"] + " <im_end>"
        return rendered

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = list(range(1, len(text.split()) + 1))
        if "<im_end>" in text:
            ids[-1] = self.im_end_token_id
        return {"input_ids": ids}


def test_sft_masks_prompt_and_keeps_chat_turn_terminator():
    rows = [{
        "format": "sft", "example_id": "s1", "source": "test",
        "instruction": "What is urgent?", "input": "", "output": "Refer now.",
    }]
    item = FalconDataset(rows, FakeTokenizer(), 32, "System")[0]
    prompt_len = item["labels"].index(next(x for x in item["labels"] if x != -100))
    assert prompt_len > 0
    assert all(x == -100 for x in item["labels"][:prompt_len])
    assert item["labels"][-1] == 98
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
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
        (complete / name).write_text('{"global_step": 25}' if name == "trainer_state.json" else "{}")
    incomplete = tmp_path / "checkpoint-50"
    incomplete.mkdir()
    (incomplete / "trainer_state.json").write_text('{"global_step": 50}')
    assert latest_checkpoint(tmp_path) == complete


def test_checkpoint_selection_uses_eval_loss_not_last_step(tmp_path: Path):
    for step in (10, 20):
        complete = tmp_path / f"checkpoint-{step}"
        complete.mkdir()
        for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
            (complete / name).write_text('{"global_step": %d}' % step if name == "trainer_state.json" else "{}")
        (complete / "checkpoint_manifest.json").write_text('{"complete": true, "global_step": %d}' % step)
    result = select_checkpoint(tmp_path, [{"step": 10, "eval_loss": 2.0}, {"step": 20, "eval_loss": 3.0}])
    assert result["status"] == "selected"
    assert result["selected_step"] == 10


def test_checkpoint_selection_refuses_to_infer_without_eval(tmp_path: Path):
    assert select_checkpoint(tmp_path, [{"step": 10, "loss": 1.0}])["status"] == "no_eval"


def test_token_share_weights_match_expected_loss_token_mass():
    items = [
        {"kind": "sft", "tokens": 10},
        {"kind": "sft", "tokens": 30},
        {"kind": "mcqa", "tokens": 5},
    ]
    weights = token_share_sampling_weights(items, {"sft": 0.75, "mcqa": 0.25})
    sft_mass = sum(w * item["tokens"] for w, item in zip(weights, items) if item["kind"] == "sft")
    mcqa_mass = sum(w * item["tokens"] for w, item in zip(weights, items) if item["kind"] == "mcqa")
    assert sft_mass == pytest.approx(0.75)
    assert mcqa_mass == pytest.approx(0.25)


def test_fast_eval_subset_is_deterministic_and_bounded():
    rows = [{"example_id": f"row-{i}"} for i in range(20)]
    selected = stable_eval_subset(rows, 5)
    assert len(selected) == 5
    assert selected == stable_eval_subset(list(reversed(rows)), 5)


def test_exact_length_distribution_reports_response_shape():
    result = distribution([1, 2, 10, 20])
    assert result["count"] == 4
    assert result["min"] == 1
    assert result["p50"] == 2
    assert result["p95"] == 10
    assert result["max"] == 20


def test_checkpoint_manifest_can_require_fp16_scaler(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
        (checkpoint / name).write_text("{}")
    (checkpoint / "trainer_state.json").write_text('{"global_step": 1}')
    (checkpoint / "checkpoint_manifest.json").write_text('{"complete": true, "global_step": 1, "scaler_required": true}')
    assert not checkpoint_is_complete(checkpoint)
    (checkpoint / "scaler.pt").write_text("{}")
    assert checkpoint_is_complete(checkpoint)
