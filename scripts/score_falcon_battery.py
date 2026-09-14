#!/usr/bin/env python3
"""Score frozen Falcon battery generations with conservative machine gates.

This is intentionally a veto layer, not a clinical judge.  It catches known
catastrophic or format failures deterministically and leaves all raw text for
expert review.  A report is promotion-eligible only when every battery item
passes, with no missing output or critical failure.  ``--report-only`` still
writes a report for development experiments without making it promotable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def norm(text: str) -> str:
    return " ".join(str(text).casefold().split())


def matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def rule_result(prompt_id: str, text: str, rule: dict[str, Any], minimum_chars: int) -> dict[str, Any]:
    lowered = norm(text)
    failures: list[str] = []
    if len(lowered) < minimum_chars:
        failures.append(f"too_short<{minimum_chars}")
    exact = rule.get("exact_text")
    if exact is not None and lowered.strip() != str(exact).casefold():
        failures.append(f"exact_text_not_{exact}")
    for field, label in (("required_all", "missing_all"), ("required_any", "missing_any"), ("required_any_2", "missing_any_2")):
        patterns = [str(x) for x in rule.get(field, [])]
        if not patterns:
            continue
        if field == "required_all":
            missing = [pattern for pattern in patterns if not re.search(pattern, lowered, re.IGNORECASE)]
            if missing:
                failures.append(f"{label}:{'|'.join(missing)}")
        elif not matches_any(lowered, patterns):
            failures.append(f"{label}:{'|'.join(patterns)}")
    forbidden = [str(x) for x in rule.get("forbidden", [])]
    hit = [pattern for pattern in forbidden if re.search(pattern, lowered, re.IGNORECASE)]
    if hit:
        failures.append(f"forbidden:{'|'.join(hit)}")
    bullet_count = rule.get("exact_bullet_count")
    if bullet_count is not None:
        bullets = [line for line in str(text).splitlines() if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", line)]
        if len(bullets) != int(bullet_count):
            failures.append(f"bullet_count:{len(bullets)}!={bullet_count}")
    return {"id": prompt_id, "passed": not failures, "failures": failures, "chars": len(text), "words": len(text.split()), "text": text}


def load_prompts(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts", []) if isinstance(payload, dict) else payload
    if not isinstance(prompts, list):
        raise ValueError(f"{path}: expected prompts list")
    return [item for item in prompts if isinstance(item, dict)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--battery", required=True, type=Path)
    ap.add_argument("--generation-dir", required=True, type=Path)
    ap.add_argument("--rubric", default="docs/research/falcon_generation_rubric.json", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--report-only", action="store_true", help="write the report but return success even when the gate rejects")
    args = ap.parse_args(argv)
    battery = load_prompts(args.battery)
    rubric = json.loads(args.rubric.read_text(encoding="utf-8"))
    rules = rubric.get("rules", {})
    results: list[dict[str, Any]] = []
    missing: list[str] = []
    for prompt in battery:
        prompt_id = str(prompt.get("id", ""))
        if not prompt_id:
            raise ValueError(f"{args.battery}: prompt without id")
        text_path = args.generation_dir / f"{prompt_id}.txt"
        if not text_path.is_file():
            missing.append(prompt_id)
            continue
        configured_rule = rules.get(prompt_id)
        if configured_rule is not None:
            rule = configured_rule
            rule_source = "external_rubric"
        else:
            # Development/validation batteries carry their own non-frozen
            # quality rubric.  This keeps pilot scoring reproducible without
            # making the frozen final gate depend on mutable battery text.
            rule = prompt.get("quality", {})
            rule_source = "embedded_battery_quality" if rule else "none"
        result = rule_result(prompt_id, text_path.read_text(encoding="utf-8"), rule, int(rubric.get("minimum_nonempty_chars", 1)))
        result["section"] = prompt.get("section", "unknown")
        result["check"] = prompt.get("check", "")
        result["rule_source"] = rule_source
        results.append(result)
    if missing:
        results.extend({"id": prompt_id, "passed": False, "failures": ["missing_generation_file"], "chars": 0, "words": 0, "text": ""} for prompt_id in missing)
    critical_sections = set(str(x) for x in rubric.get("critical_sections", []))
    critical_failures = [item["id"] for item in results if not item["passed"] and item.get("section") in critical_sections]
    counts = Counter(item.get("section", "unknown") for item in results)
    passed = sum(bool(item["passed"]) for item in results)
    all_passed = not missing and len(results) == len(battery) and passed == len(battery)
    summary = {
        "battery": str(args.battery),
        "generation_dir": str(args.generation_dir),
        "rubric": str(args.rubric),
        "prompt_count": len(battery),
        "scored_count": len(results),
        "missing_count": len(missing),
        "passed_count": passed,
        "failed_count": len(results) - passed,
        "pass_rate_percent": round(100 * passed / max(1, len(results)), 3),
        "section_counts": dict(sorted(counts.items())),
        "critical_failures": critical_failures,
        "all_passed": all_passed,
        "promotion_eligible": all_passed and not critical_failures,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(args.out)
    print(json.dumps({key: summary[key] for key in ("prompt_count", "scored_count", "passed_count", "failed_count", "critical_failures", "all_passed", "promotion_eligible")}, ensure_ascii=False), flush=True)
    return 0 if summary["promotion_eligible"] or args.report_only else 2


if __name__ == "__main__":
    raise SystemExit(main())
