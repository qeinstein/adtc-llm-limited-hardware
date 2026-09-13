"""Validate a prompt-battery JSON file (schema gate for CI batteries).

Catches malformed batteries before runner hours are burned: every prompt needs
a unique id, a section, a positive max_tokens budget, and non-empty text.
Usage: python scripts/validate_battery.py <battery.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_KEYS = ("id", "section", "text", "max_tokens")


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"{path}: invalid JSON: {e}"]
    # The original Falcon batteries are wrapped in {"prompts": [...]}; the
    # Swahili battery is intentionally a plain list with query/gold-keyword
    # records.  Validate both schemas instead of letting CI reject a holdout
    # that the evaluator itself can run.
    wrapped = isinstance(data, dict)
    prompts = data.get("prompts") if wrapped else data
    if not isinstance(prompts, list) or not prompts:
        expected = "'prompts' must be a non-empty list" if wrapped else "top-level value must be a non-empty list"
        return [f"{path}: {expected}"]
    seen: set[str] = set()
    for i, pr in enumerate(prompts):
        where = f"{path}[{i}]"
        if not isinstance(pr, dict):
            errors.append(f"{where}: not an object")
            continue
        if wrapped:
            for key in REQUIRED_KEYS:
                if key not in pr:
                    errors.append(f"{where}: missing key {key!r}")
        else:
            if not any(key in pr for key in ("text", "query", "instruction")):
                errors.append(f"{where}: missing prompt text field ('text', 'query', or 'instruction')")
        pid = pr.get("id")
        if isinstance(pid, str):
            if pid in seen:
                errors.append(f"{where}: duplicate id {pid!r}")
            seen.add(pid)
        text_value = pr.get("text", pr.get("query", pr.get("instruction", "")))
        if not str(text_value).strip():
            errors.append(f"{where}: empty prompt text")
        mt = pr.get("max_tokens")
        if not wrapped and mt is None:
            mt = 256
        if mt is not None and (not isinstance(mt, int) or mt <= 0):
            errors.append(f"{where}: max_tokens must be a positive int")
    return errors


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <battery.json>")
        return 2
    errors = validate(Path(argv[1]))
    for e in errors:
        print(f"[battery] ERROR: {e}")
    if errors:
        return 1
    print("[battery] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
