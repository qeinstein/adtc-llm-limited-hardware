"""Guidance regression harness, rules-only tier (no model needed).

Runs every case in evals/{judge_regressions,clinical_guidance,kiswahili,
safety} through facts+rules (kind=rules) or output_lint on canned replies
(kind=lint). Model-backed judging (clinical correctness, concision,
usefulness) requires a live server and is documented in SAFETY.md as the
second tier; these suites pin the deterministic floor.
"""
import json
from pathlib import Path

import pytest

from runtime.safety import evaluate_rules, extract_facts, lint_output

ROOT = Path(__file__).resolve().parents[1]
SUITES = ["judge_regressions/judge_regressions",
          "clinical_guidance/domains",
          "kiswahili/kiswahili",
          "safety/safety"]


def _load():
    cases = []
    for s in SUITES:
        d = json.loads((ROOT / "evals" / (s + ".json")).read_text())
        for c in d["cases"]:
            c["_suite"] = d["suite"]
            cases.append(c)
    return cases


@pytest.mark.parametrize("case", _load(), ids=lambda c: c["id"])
def test_case(case):
    if case["kind"] == "rules":
        facts = extract_facts(case["prompt"])["facts"]
        res = evaluate_rules(facts)
        exp = case["expect"]
        assert res["risk"] == exp["risk"], (res["risk"], sorted(facts),
                                            res["cards"])
        if exp["cards_any"]:
            assert any(c in res["cards"] for c in exp["cards_any"]), res["cards"]
        else:
            assert res["cards"] == [], res["cards"]
        assert res["override"] == exp["override"], res["cards"]
    else:
        risk = {"level": case.get("level", "ROUTINE"),
                "hits": case.get("hits", []),
                "facts": sorted(extract_facts(case["prompt"])["facts"])}
        res = lint_output(risk, case["prompt"], case["reply"],
                          case.get("attributions"))
        exp = case["expect_failures"]
        if not exp:
            assert res["failures"] == [], (res["failures"], case["id"])
        else:
            assert set(exp) <= set(res["failures"]), (res["failures"],
                                                      case["id"])
