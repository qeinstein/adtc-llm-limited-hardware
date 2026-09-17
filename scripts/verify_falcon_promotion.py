#!/usr/bin/env python3
"""Fail-closed checks for Falcon checkpoint promotion and export.

Training checkpoints are useful for resume, but they are not submission
artifacts until the development/validation selector and the one-shot frozen
clinical gate have passed.  This module keeps that distinction explicit and
portable between the Kaggle notebook, export shell, and tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.select_falcon_candidate import frozen_gate_passes
except ModuleNotFoundError:  # direct ``python scripts/<tool>.py`` invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.select_falcon_candidate import frozen_gate_passes


PROMOTION_SCHEMA = "falcon-promotion-v1"
PROMOTION_STATUS = "selected_and_frozen_gate_passed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_payload_files(adapter: Path) -> list[Path]:
    """Return only files whose bytes define the PEFT adapter payload."""
    files = [
        path for path in adapter.iterdir()
        if path.is_file() and (
            path.name == "adapter_config.json" or path.name.startswith("adapter_model.")
        )
    ]
    return sorted(files, key=lambda path: path.name)


def _require_frozen_report(report: dict[str, Any], minimum_pass_rate: float) -> None:
    if not frozen_gate_passes(report, minimum_pass_rate):
        raise ValueError(
            "frozen quality report is not promotable: "
            f"pass_rate={report.get('pass_rate_percent')!r}, "
            f"passed={report.get('passed_count')!r}/{report.get('prompt_count')!r}, "
            f"missing={report.get('missing_count')!r}, "
            f"critical={report.get('critical_failures')!r}"
        )


def build_promotion_manifest(
    adapter: Path,
    *,
    quality_selection: dict[str, Any],
    frozen_report: dict[str, Any],
    experiment_id: str,
    stage: str,
    repo_sha: str,
    config_sha256: str | None = None,
    data_manifest_sha256: str | None = None,
    minimum_pass_rate: float = 100.0,
) -> dict[str, Any]:
    """Build an immutable promotion record after the frozen gate passes."""
    adapter = adapter.resolve()
    if not adapter.is_dir():
        raise ValueError(f"adapter directory does not exist: {adapter}")
    payload = adapter_payload_files(adapter)
    if not any(path.name == "adapter_config.json" for path in payload):
        raise ValueError(f"adapter_config.json missing: {adapter}")
    if not any(path.name.startswith("adapter_model.") for path in payload):
        raise ValueError(f"adapter_model.* missing: {adapter}")
    if quality_selection.get("status") != PROMOTION_STATUS:
        raise ValueError("quality selection is not marked selected_and_frozen_gate_passed")
    _require_frozen_report(frozen_report, minimum_pass_rate)
    frozen_gate = quality_selection.get("frozen_gate") or {}
    if frozen_gate.get("status") != "passed":
        raise ValueError("quality selection frozen_gate is not passed")

    manifest: dict[str, Any] = {
        "schema": PROMOTION_SCHEMA,
        "status": PROMOTION_STATUS,
        "experiment_id": experiment_id,
        "stage": stage,
        "repo_sha": repo_sha,
        "selected_step": int(quality_selection["selected_step"]),
        "selected_eval_loss": quality_selection.get("selected_eval_loss"),
        "minimum_frozen_pass_rate_percent": float(minimum_pass_rate),
        "frozen_gate": {
            "status": "passed",
            "battery": frozen_report.get("battery"),
            "report_sha256": frozen_report.get("report_sha256"),
            "prompt_count": int(frozen_report.get("prompt_count", 0)),
            "passed_count": int(frozen_report.get("passed_count", 0)),
            "pass_rate_percent": float(frozen_report.get("pass_rate_percent", 0.0)),
            "missing_count": int(frozen_report.get("missing_count", 0)),
            "critical_failures": list(frozen_report.get("critical_failures", [])),
        },
        "quality_selection": {
            "selection_criterion": quality_selection.get("selection_criterion"),
            "minimum_dev_validation_pass_rate_percent": quality_selection.get(
                "minimum_pass_rate_percent"
            ),
            "selected_dev_validation_pass_rate_percent": quality_selection.get(
                "selected_dev_validation_pass_rate_percent"
            ),
        },
        "adapter_payload_sha256": {
            path.name: sha256_file(path) for path in payload
        },
    }
    if config_sha256:
        manifest["config_sha256"] = config_sha256
    if data_manifest_sha256:
        manifest["data_manifest_sha256"] = data_manifest_sha256
    return manifest


def verify_promoted_adapter(
    adapter: Path,
    *,
    manifest_path: Path | None = None,
    minimum_pass_rate: float = 100.0,
) -> dict[str, Any]:
    """Validate a retrieved adapter before any base-model merge/export."""
    adapter = adapter.resolve()
    if not adapter.is_dir():
        raise ValueError(f"adapter directory does not exist: {adapter}")
    payload = adapter_payload_files(adapter)
    if not any(path.name == "adapter_config.json" for path in payload):
        raise ValueError("promoted adapter has no adapter_config.json")
    if not any(path.name.startswith("adapter_model.") for path in payload):
        raise ValueError("promoted adapter has no adapter_model.*")
    manifest_path = (manifest_path or (adapter / "promotion_manifest.json")).resolve()
    if not manifest_path.is_file():
        raise ValueError(
            f"promotion_manifest.json is required; refusing un-gated adapter: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid promotion manifest: {exc}") from exc
    if manifest.get("schema") != PROMOTION_SCHEMA:
        raise ValueError(f"unsupported promotion manifest schema: {manifest.get('schema')!r}")
    if manifest.get("status") != PROMOTION_STATUS:
        raise ValueError(f"adapter is not promoted: status={manifest.get('status')!r}")
    frozen = manifest.get("frozen_gate") or {}
    if frozen.get("status") != "passed":
        raise ValueError("promotion manifest frozen gate is not passed")
    if float(manifest.get("minimum_frozen_pass_rate_percent", 0.0)) < minimum_pass_rate:
        raise ValueError("promotion manifest was created with a weaker frozen threshold")
    if int(frozen.get("missing_count", 1)) != 0 or frozen.get("critical_failures"):
        raise ValueError("promotion manifest contains missing or critical frozen failures")
    if float(frozen.get("pass_rate_percent", 0.0)) < minimum_pass_rate:
        raise ValueError("promotion manifest frozen pass rate is below export threshold")
    expected = manifest.get("adapter_payload_sha256") or {}
    actual = {path.name: sha256_file(path) for path in payload}
    if expected != actual:
        raise ValueError("adapter payload hash mismatch; manifest and adapter do not match")
    return {
        "adapter": str(adapter),
        "manifest": str(manifest_path),
        "status": manifest["status"],
        "selected_step": manifest.get("selected_step"),
        "frozen_pass_rate_percent": frozen.get("pass_rate_percent"),
        "adapter_payload_sha256": actual,
    }


def verify_frozen_quality_report(
    report_path: Path,
    *,
    minimum_pass_rate: float = 100.0,
    expected_battery: str | None = None,
) -> dict[str, Any]:
    """Validate a merged or quantized frozen report before export completes."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid quality report {report_path}: {exc}") from exc
    if expected_battery is not None and Path(str(report.get("battery", ""))).name != Path(expected_battery).name:
        raise ValueError(
            f"quality report battery mismatch: {report.get('battery')!r} != {expected_battery!r}"
        )
    _require_frozen_report(report, minimum_pass_rate)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--quality-report", type=Path, action="append", default=[])
    parser.add_argument("--minimum-pass-rate", type=float, default=100.0)
    args = parser.parse_args(argv)
    result = verify_promoted_adapter(
        args.adapter,
        manifest_path=args.manifest,
        minimum_pass_rate=args.minimum_pass_rate,
    )
    for report in args.quality_report:
        quality = verify_frozen_quality_report(report, minimum_pass_rate=args.minimum_pass_rate)
        result.setdefault("quality_reports", []).append({
            "path": str(report),
            "pass_rate_percent": quality.get("pass_rate_percent"),
        })
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
