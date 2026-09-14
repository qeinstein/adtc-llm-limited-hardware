import json
from pathlib import Path

import pytest

from scripts.build_falcon_dataset import assign_prompt_group_splits, canonical, distribution, near_holdout
from scripts.falcon_format import generation_stop_ids
from scripts.train_falcon_production import DeterministicTokenShareSampler, FalconDataset, checkpoint_is_complete, enforce_objective_token_policy, latest_checkpoint, normalize_mcqa_scores, objective_loss_token_summary, select_checkpoint, stable_eval_subset, token_share_sampling_weights, truncate_mcqa_context
from scripts.select_falcon_candidate import frozen_gate_passes, select_dev_validation_candidate
from scripts.verify_falcon_promotion import build_promotion_manifest, verify_frozen_quality_report, verify_promoted_adapter


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

    def get_vocab(self):
        return {"<im_end>": self.im_end_token_id}


class StopTokenizer:
    eos_token_id = 11
    unk_token_id = 0
    pad_token_id = 0

    def get_vocab(self):
        return {"<|im_end|>": 228}


def test_generation_stop_ids_use_vocab_and_never_pad_or_unknown():
    class Config:
        eos_token_id = [11, 228]

    assert generation_stop_ids(StopTokenizer(), Config()) == [11, 228]


def test_generation_stop_ids_do_not_fallback_to_convert_tokens_unknown():
    class BadTokenizer(StopTokenizer):
        def get_vocab(self):
            return {}

        def convert_tokens_to_ids(self, token):
            assert token == "<|im_end|>"
            return 0

    class Config:
        eos_token_id = 11

    assert generation_stop_ids(BadTokenizer(), Config()) == [11]


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


def test_mcqa_context_window_is_head_tail_and_records_truncation():
    values, truncated = truncate_mcqa_context(list(range(10)), 6)
    assert truncated is True
    assert values == [0, 1, 2, 3, 8, 9]


def test_mcqa_context_under_window_is_unchanged():
    original = list(range(6))
    values, truncated = truncate_mcqa_context(original, 6)
    assert truncated is False
    assert values == original


def test_mcqa_dataset_records_bounded_context_truncation():
    rows = [{
        "format": "mcqa", "example_id": "m-long", "source": "test",
        "context": "one two three four five six seven eight nine ten",
        "choices": ["red", "blue"], "gold": 0,
    }]
    dataset = FalconDataset(rows, FakeTokenizer(), 32, "System", mcqa_context_max_tokens=6)
    assert dataset.mcqa_context_truncated == 1
    assert dataset[0]["context_truncated"] is True
    assert len(dataset[0]["context_ids"]) == 6


def test_mcqa_length_normalization_uses_token_lengths_not_characters():
    assert normalize_mcqa_scores([10.0, 20.0], [2, 4]) == [5.0, 5.0]
    with pytest.raises(ValueError, match="equal length"):
        normalize_mcqa_scores([1.0], [1, 2])


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


def test_materialize_selected_adapter_promotes_selected_weights_only(tmp_path: Path):
    from scripts.train_falcon_production import materialize_selected_adapter

    selected = tmp_path / "checkpoint-10"
    selected.mkdir()
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (selected / name).write_text("{}")
    (selected / "trainer_state.json").write_text('{"global_step": 10}')
    (selected / "checkpoint_manifest.json").write_text('{"complete": true, "global_step": 10}')
    (selected / "adapter_config.json").write_text('{"r": 16}')
    (selected / "adapter_model.safetensors").write_bytes(b"selected")

    class Tokenizer:
        def save_pretrained(self, path):
            Path(path, "tokenizer_config.json").write_text("{}")

    destination = tmp_path / "final-adapter"
    materialize_selected_adapter(selected, destination, Tokenizer())
    assert (destination / "adapter_model.safetensors").read_bytes() == b"selected"
    assert (destination / "tokenizer_config.json").exists()
    assert not (destination / "optimizer.pt").exists()


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


def test_deterministic_sampler_replays_exact_sequence_for_resume():
    first = list(DeterministicTokenShareSampler([0.2, 0.8], 32, seed=3407))
    second = list(DeterministicTokenShareSampler([0.2, 0.8], 32, seed=3407))
    different = list(DeterministicTokenShareSampler([0.2, 0.8], 32, seed=3408))
    assert first == second
    assert first != different
    state = DeterministicTokenShareSampler([0.2, 0.8], 32, seed=3407).state_dict()
    assert state["algorithm"] == "torch.multinomial_private_generator_v1"
    assert state["resume_semantics"].startswith("replay full deterministic sequence")


def test_objective_token_summary_and_fail_closed_policy():
    items = [
        {"kind": "sft", "tokens": 30},
        {"kind": "mcqa", "tokens": 20},
    ]
    assert objective_loss_token_summary(items) == {
        "loss_token_totals": {"mcqa": 20, "sft": 30},
        "loss_token_shares_percent": {"mcqa": 40.0, "sft": 60.0},
    }
    assert enforce_objective_token_policy(
        items,
        {"minimum_raw_loss_token_share": {"sft": 0.5}, "maximum_raw_loss_token_share": {"mcqa": 0.5}},
    )["loss_token_totals"] == {"mcqa": 20, "sft": 30}
    with pytest.raises(RuntimeError, match="violates fail-closed policy"):
        enforce_objective_token_policy(items, {"minimum_raw_loss_token_share": {"sft": 0.7}})


def test_fast_eval_subset_is_deterministic_and_bounded():
    rows = [{"example_id": f"row-{i}"} for i in range(20)]
    selected = stable_eval_subset(rows, 5)
    assert len(selected) == 5
    assert selected == stable_eval_subset(list(reversed(rows)), 5)


def test_fast_eval_subset_preserves_small_objective_group():
    rows = [
        *({"example_id": f"sft-{i}", "format": "sft"} for i in range(4)),
        *({"example_id": f"mcqa-{i}", "format": "mcqa"} for i in range(20)),
    ]
    selected = stable_eval_subset(rows, 8)
    assert len(selected) == 8
    assert sum(row["format"] == "sft" for row in selected) == 4
    assert selected == stable_eval_subset(list(reversed(rows)), 8)


def test_split_keeps_mcqa_prompt_permutations_together():
    rows = [
        {"identity": "a", "prompt_identity": "same question", "example_id": "a"},
        {"identity": "b", "prompt_identity": "same question", "example_id": "b"},
    ]
    assign_prompt_group_splits(rows, "3407", 0.5)
    assert {row["split"] for row in rows} == {"dev"} or {row["split"] for row in rows} == {"train"}


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


def test_checkpoint_manifest_can_require_sampler_state(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
        (checkpoint / name).write_text("{}")
    (checkpoint / "trainer_state.json").write_text('{"global_step": 1}')
    (checkpoint / "checkpoint_manifest.json").write_text('{"complete": true, "global_step": 1, "sampler_required": true}')
    assert not checkpoint_is_complete(checkpoint)
    (checkpoint / "sampler_state.json").write_text("{}")
    assert checkpoint_is_complete(checkpoint)


def _quality(passed, *, critical=None, missing=0):
    return {
        "prompt_count": 8,
        "passed_count": passed,
        "critical_failures": critical or [],
        "missing_count": missing,
    }


def test_dev_validation_selection_never_uses_frozen_report():
    candidates = [
        {"step": 4, "checkpoint": "/ckpt-4", "eval_loss": 2.0},
        {"step": 8, "checkpoint": "/ckpt-8", "eval_loss": 2.2},
    ]
    reports = {
        4: {"dev": _quality(8), "validation": _quality(6)},
        8: {"dev": _quality(6), "validation": _quality(8)},
    }
    result = select_dev_validation_candidate(candidates, reports, minimum_pass_rate=75)
    assert result["status"] == "selected_for_frozen_gate"
    assert result["selected_step"] == 4
    assert "frozen" not in result["selection_criterion"]


def test_dev_validation_selection_rejects_critical_failure_even_at_high_pass_rate():
    result = select_dev_validation_candidate(
        [{"step": 4, "checkpoint": "/ckpt-4", "eval_loss": 1.0}],
        {4: {"dev": _quality(8, critical=["d01"]), "validation": _quality(8)}},
        minimum_pass_rate=75,
    )
    assert result["status"] == "no_candidate_passed_dev_validation"
    assert result["selected_checkpoint"] is None


def test_frozen_gate_requires_complete_quality_by_default():
    assert frozen_gate_passes(_quality(8))
    assert not frozen_gate_passes(_quality(7))
    assert not frozen_gate_passes(_quality(8, critical=["h01"]))


def test_kaggle_stage_selection_does_not_rank_on_frozen_battery():
    import json

    notebook = Path("kaggle/phase04-falcon-production/phase04_falcon_production.ipynb")
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    select_at = source.index("select_dev_validation_candidate")
    frozen_eval_at = source.index("frozen_eval_cmd")
    assert select_at < frozen_eval_at
    assert "highest_frozen_generation_pass_rate" not in source


def _promotable_quality():
    return {
        "battery": "docs/research/falcon_probe_heldout.json",
        "prompt_count": 8,
        "scored_count": 8,
        "passed_count": 8,
        "failed_count": 0,
        "missing_count": 0,
        "pass_rate_percent": 100.0,
        "critical_failures": [],
    }


def test_promotion_manifest_is_required_and_binds_adapter_payload(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    quality = _promotable_quality()
    selection = {
        "status": "selected_and_frozen_gate_passed",
        "selected_step": 8,
        "selected_eval_loss": 2.0,
        "selection_criterion": "dev_validation",
        "minimum_pass_rate_percent": 75.0,
        "selected_dev_validation_pass_rate_percent": 100.0,
        "frozen_gate": {"status": "passed"},
    }
    manifest = build_promotion_manifest(
        adapter,
        quality_selection=selection,
        frozen_report=quality,
        experiment_id="test-exp",
        stage="stage_a_capability_preserving",
        repo_sha="deadbeef",
    )
    manifest_path = adapter / "promotion_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    checked = verify_promoted_adapter(adapter)
    assert checked["status"] == "selected_and_frozen_gate_passed"
    assert checked["selected_step"] == 8
    (adapter / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_promoted_adapter(adapter)


def test_promotion_manifest_rejects_failed_frozen_quality(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    selection = {"status": "selected_and_frozen_gate_passed", "selected_step": 8, "frozen_gate": {"status": "passed"}}
    with pytest.raises(ValueError, match="frozen quality report"):
        build_promotion_manifest(
            adapter,
            quality_selection=selection,
            frozen_report={**_promotable_quality(), "passed_count": 7, "pass_rate_percent": 87.5},
            experiment_id="test-exp",
            stage="stage_a_capability_preserving",
            repo_sha="deadbeef",
        )


def test_export_requires_promoted_input_and_both_frozen_reports():
    import json

    notebook = json.loads(Path("kaggle/phase04-falcon-production/phase04_falcon_production.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "verify_promoted_adapter" in source
    assert "verify_frozen_quality_report(merged_quality_path" in source
    assert "verify_frozen_quality_report(quantized_quality_path" in source
    assert "--report-only" in source
    assert "export must include the frozen clinical/safety battery" in source
    assert "export_manifest['deployment_model']" in source
    assert "Falcon-H1-1.5B-Deep-Instruct-Q4_K_M.gguf" not in source
    assert "exported_and_frozen_gate_passed" in source
    trainer = Path("scripts/train_falcon_production.py").read_text(encoding="utf-8")
    assert "training_complete_quality_gate_pending" in trainer
    assert "promotion_status': 'promoted_after_frozen_gate'" in source
