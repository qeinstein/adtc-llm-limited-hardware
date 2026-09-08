import json
from pathlib import Path

from src.data_audit import audit_manifest, measure_row, summarize_rows


class FakeTokenizer:
    eos_token_id = 99
    eos_token = "<eos>"
    pad_token = "<eos>"

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": list(range(len(text.split())))}

    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, enable_thinking
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return " ".join(message["content"] for message in messages) + " assistant"


def test_mcqa_counts_prompt_once_per_choice_and_applies_repeat():
    spec = {
        "name": "arc_easy",
        "format": "mcqa",
        "repeat": 2,
        "stream": "mcqa",
        "capability": "reasoning_science",
        "domain": "general",
        "provenance": "sourced",
    }
    row = {"context": "one two three", "choices": ["red", "blue sky"], "gold": 1}
    metric = measure_row(FakeTokenizer(), spec, row, max_len=16, eos_id=99)
    assert metric is not None
    assert metric.prompt_trained == 6
    assert metric.target_trained == 3
    assert metric.answer_length == 2

    summary = summarize_rows([metric], raw_rows=1, invalid_rows=0)
    assert summary["effective_examples"] == 2
    assert summary["prompt_tokens"] == 12
    assert summary["target_tokens"] == 6


def test_corpus_truncation_matches_loader():
    spec = {
        "name": "corpus",
        "format": "corpus",
        "repeat": 1,
        "stream": "causal_lm",
        "capability": "healthcare_continued_pretraining",
        "domain": "medical",
        "provenance": "sourced",
    }
    row = {"text": " ".join(f"t{i}" for i in range(20))}
    metric = measure_row(FakeTokenizer(), spec, row, max_len=12, eos_id=99)
    assert metric is not None
    assert metric.prompt_trained == 3
    assert metric.target_trained == 9
    assert metric.tokens_lost == 8
    assert metric.truncated


def test_manifest_reports_duplicates_missing_inputs_and_facets(tmp_path: Path):
    rows = [
        {
            "instruction": "Habari yako rafiki",
            "input": "",
            "output": "Niko vizuri sana asante",
        },
        {
            "instruction": "Habari yako rafiki",
            "input": "",
            "output": "Niko vizuri sana asante",
        },
    ]
    (tmp_path / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    manifest = {
        "audit_id": "test",
        "repository_root_parents": 0,
        "tokenizer": "fake",
        "max_len": 64,
        "sources": [
            {
                "name": "chat",
                "path": "rows.json",
                "format": "alpaca",
                "repeat": 3,
                "stream": "generation",
                "capability": "conversation",
                "domain": "general",
                "provenance": "curated",
                "language": "sw",
            },
            {
                "name": "missing",
                "path": "absent.jsonl",
                "format": "mcqa",
                "repeat": 1,
                "stream": "mcqa",
                "capability": "mcqa",
                "domain": "mixed",
                "provenance": "sourced",
                "required": False,
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = audit_manifest(manifest_path, FakeTokenizer())
    assert report["per_source"]["chat"]["exact_duplicates"] == 1
    assert report["per_source"]["chat"]["effective_examples"] == 6
    assert report["facets"]["language"]["sw"]["token_share_percent"] == 100.0
    assert report["missing_sources"][0]["name"] == "missing"
