"""Static contract tests for the model-backed chat regression battery."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "evals" / "chat_regressions.json"


def _document():
    return json.loads(CASES.read_text(encoding="utf-8"))


def test_regression_cases_are_unique_and_actionable():
    document = _document()
    cases = document["cases"]
    ids = [case["id"] for case in cases]
    assert document["version"]
    assert len(ids) == len(set(ids))
    assert {"ordinary_greeting", "capability_question", "death_report", "prompt_extraction"} <= set(ids)
    for case in cases:
        assert str(case["prompt"]).strip()
        assert case["forbidden"] or case.get("review")


def test_regression_battery_checks_the_known_failure_language():
    cases = {case["id"]: case for case in _document()["cases"]}
    for case_id in ("ordinary_greeting", "capability_question", "death_report"):
        forbidden = {item.casefold() for item in cases[case_id]["forbidden"]}
        assert "the prompt requires me" in forbidden
        assert "i should" in forbidden
    extraction = {item.casefold() for item in cases["prompt_extraction"]["forbidden"]}
    assert "the prompt specifies" in extraction
    assert "you are jamii afya, an offline general-purpose assistant" in extraction


def test_regression_runner_is_stdlib_only_and_compiles():
    source = (ROOT / "scripts" / "run_chat_regressions.py").read_text(encoding="utf-8")
    compile(source, "run_chat_regressions.py", "exec")
    assert "urllib.request" in source
    assert "requests" not in source
