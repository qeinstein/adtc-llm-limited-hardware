#!/usr/bin/env python3
"""Populate the active Falcon model card from completed validation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def value(block: dict[str, Any], *keys: str) -> Any:
    current: Any = block
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", default="MODEL_CARD.md", type=Path)
    parser.add_argument("--evaluation", default="artifacts/falcon-final-eval.json", type=Path)
    parser.add_argument("--profiler", default="artifacts/adtc-submission.json", type=Path)
    parser.add_argument("--export", default=None, type=Path)
    args = parser.parse_args(argv)
    evaluation = read_json(args.evaluation.resolve())
    profiler = read_json(args.profiler.resolve())
    export = read_json(args.export.resolve()) if args.export else {}
    model = evaluation.get("model") or {}
    summary = evaluation.get("summary") or {}
    no_system = evaluation.get("no_system_safety_summary") or {}
    throughput = profiler.get("throughput") or {}
    memory = profiler.get("memory") or {}
    accuracy = profiler.get("accuracy") or []
    thermal = profiler.get("cpu_thermal") or {}
    sha = str(model.get("sha256") or export.get("deployment_sha256") or "unknown")
    bytes_value = model.get("bytes") or export.get("deployment_bytes") or "unknown"
    failed = [str(item["id"]) for item in evaluation.get("questions", []) if not item.get("pass")]
    accuracy_lines = []
    for item in accuracy:
        accuracy_lines.append(f"  - `{item.get('task', item.get('metric', 'accuracy'))}`: `{item}`")
    if not accuracy_lines:
        accuracy_lines.append("  - No accuracy row was emitted (the full profiler run must be treated as failed).")
    base = args.card.read_text(encoding="utf-8")
    marker = "\n## Final benchmark results\n"
    if marker in base:
        base = base.split(marker, 1)[0].rstrip() + "\n"
    section = f"""
## Final benchmark results

These results were measured after export on the exact quantized GGUF, not on
the adapter or an FP16 model.

- GGUF SHA256: `{sha}`
- GGUF size: `{bytes_value}` bytes
- 48-question evaluation: `{summary.get('passed', 0)}/{summary.get('total', 0)}` (`{summary.get('score_percent', 0)}%`)
- 48Q failed IDs: `{', '.join(failed) if failed else 'none'}`
- Critical D01-D12: `{summary.get('critical_safety_passed', 0)}/{summary.get('critical_safety_total', 0)}`
- No-system critical safety: `{no_system.get('passed', 0)}/{no_system.get('total', 0)}`
- Official profiler accuracy:
{chr(10).join(accuracy_lines)}
- Official profiler generation tok/s: `{throughput.get('tokens_per_second_generation', 'unknown')}`
- Official profiler prompt tok/s: `{throughput.get('tokens_per_second_prompt', 'unknown')}`
- Official profiler first-token latency: `{throughput.get('first_token_latency_ms', 'unknown')} ms`
- Official profiler peak RSS: `{memory.get('peak_rss_mb', 'unknown')} MB`
- Official profiler steady RSS: `{memory.get('steady_state_rss_mb', 'unknown')} MB`
- CPU utilization p99: `{value(thermal, 'cpu_percent_p99')}`
- Thermal peak/throttled: `{value(thermal, 'core_temp_c_peak')}` / `{value(thermal, 'throttled')}`
- Measurement label: **Official ADTC profiler software, GitHub Actions hardware self-check**

"""
    args.card.write_text(base.rstrip() + "\n" + section.lstrip(), encoding="utf-8")
    print(json.dumps({"card": str(args.card), "sha256": sha, "bytes": bytes_value}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
