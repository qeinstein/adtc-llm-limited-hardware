"""Run the model-backed chat regression cases against a running Jamii Afya UI.

This is an evaluation harness, not a response filter. It reports prompt
mirroring and obvious extraction regressions but never edits, truncates, or
rewrites a model answer. It uses only the Python standard library so it works
on Windows, macOS, and Linux.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = ROOT / "evals" / "chat_regressions.json"


def _request(url: str, prompt: str, timeout: float) -> dict[str, Any]:
    payload = json.dumps({"message": prompt, "history": []}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise TypeError("chat endpoint returned a non-object JSON response")
    return parsed


def _violations(case: dict[str, Any], reply: str) -> list[str]:
    lowered = reply.casefold()
    failures: list[str] = []
    if not reply.strip():
        failures.append("empty reply")
    for phrase in case.get("forbidden", []):
        if str(phrase).casefold() in lowered:
            failures.append(f"forbidden phrase: {phrase}")
    if "<think>" in reply or "</think>" in reply:
        failures.append("reasoning marker reached visible reply")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8420/api/chat")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    document = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = document.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("regression file must contain a non-empty cases list")

    failed = 0
    for case in cases:
        case_id = str(case["id"])
        try:
            response = _request(args.url, str(case["prompt"]), args.timeout)
            reply = str(response.get("reply") or "")
            problems = _violations(case, reply)
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            reply = ""
            problems = [f"request/evaluation error: {exc}"]
        if problems:
            failed += 1
            print(f"FAIL {case_id}: {'; '.join(problems)}")
        else:
            print(f"PASS {case_id}")
        print(f"  reply: {reply[:240].replace(chr(10), ' ')}")

    print(f"\n{len(cases) - failed}/{len(cases)} cases passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
