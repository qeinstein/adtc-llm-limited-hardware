"""Regression tests for scripts/validate_battery.py.

Guards the CI battery schema: the Falcon baseline run once burned a full
build+bench cycle on a prompts file whose first entry had no "text" key.
"""

from pathlib import Path

from scripts.validate_battery import validate

BATTERY = Path(__file__).resolve().parent.parent / "docs" / "research" / "falcon_baseline_prompts.json"
SWAHILI = Path(__file__).resolve().parent.parent / "data" / "swahili_eval_set.json"


def _write(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "battery.json"
    p.write_text(__import__("json").dumps(payload))
    return p


def test_shipped_falcon_battery_is_valid():
    assert validate(BATTERY) == []


def test_shipped_swahili_query_battery_is_valid():
    assert validate(SWAHILI) == []


def test_missing_text_key_fails(tmp_path):
    p = _write(tmp_path, {"prompts": [{"id": "p01", "section": "A", "max_tokens": 10}]})
    errors = validate(p)
    assert any("missing key 'text'" in e for e in errors)


def test_duplicate_id_fails(tmp_path):
    p = _write(tmp_path, {"prompts": [
        {"id": "p01", "section": "A", "text": "a", "max_tokens": 10},
        {"id": "p01", "section": "B", "text": "b", "max_tokens": 10},
    ]})
    assert any("duplicate id" in e for e in validate(p))


def test_bad_max_tokens_fails(tmp_path):
    p = _write(tmp_path, {"prompts": [
        {"id": "p01", "section": "A", "text": "a", "max_tokens": 0},
    ]})
    assert any("max_tokens" in e for e in validate(p))
