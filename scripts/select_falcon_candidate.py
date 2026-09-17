#!/usr/bin/env python3
"""Select a Falcon checkpoint using development/validation reports only.

The final frozen battery is a one-shot promotion gate and must never be used
to choose among checkpoints.  This module keeps that rule explicit and
testable so the Kaggle notebook cannot accidentally optimize on the holdout.
"""

from __future__ import annotations

from typing import Any


def report_pass_rate(report: dict[str, Any]) -> float:
    """Return a bounded pass rate from a scored battery report."""
    count = max(1, int(report.get("prompt_count", 0)))
    passed = max(0, int(report.get("passed_count", 0)))
    return 100.0 * min(passed, count) / count


def report_is_eligible(report: dict[str, Any], minimum_pass_rate: float) -> bool:
    """Apply the non-frozen quality gate for a dev/validation report."""
    if int(report.get("missing_count", 0)) != 0:
        return False
    if report.get("critical_failures"):
        return False
    return report_pass_rate(report) >= float(minimum_pass_rate)


def frozen_gate_passes(report: dict[str, Any], minimum_pass_rate: float = 100.0) -> bool:
    """Apply the one-shot final gate to a selected candidate only."""
    return report_is_eligible(report, minimum_pass_rate)


def select_dev_validation_candidate(
    candidates: list[dict[str, Any]],
    reports: dict[int, dict[str, dict[str, Any]]],
    *,
    minimum_pass_rate: float = 75.0,
) -> dict[str, Any]:
    """Choose the strongest eligible loss-ranked candidate without holdout use.

    ``reports[step]`` must contain ``dev`` and ``validation`` reports.  The
    returned record is suitable for writing to ``quality_selection.json``;
    it includes every considered candidate so a failed selection is auditable.
    """
    considered: list[dict[str, Any]] = []
    for candidate in candidates:
        step = int(candidate["step"])
        pair = reports.get(step, {})
        dev = pair.get("dev", {})
        validation = pair.get("validation", {})
        dev_rate = report_pass_rate(dev)
        validation_rate = report_pass_rate(validation)
        eligible = report_is_eligible(dev, minimum_pass_rate) and report_is_eligible(validation, minimum_pass_rate)
        considered.append({
            "step": step,
            "checkpoint": str(candidate["checkpoint"]),
            "eval_loss": float(candidate.get("eval_loss", float("inf"))),
            "dev_pass_rate_percent": round(dev_rate, 3),
            "validation_pass_rate_percent": round(validation_rate, 3),
            "mean_pass_rate_percent": round((dev_rate + validation_rate) / 2.0, 3),
            "worst_pass_rate_percent": round(min(dev_rate, validation_rate), 3),
            "dev_critical_failures": list(dev.get("critical_failures", [])),
            "validation_critical_failures": list(validation.get("critical_failures", [])),
            "dev_validation_eligible": eligible,
        })
    eligible = [item for item in considered if item["dev_validation_eligible"]]
    if not eligible:
        return {
            "status": "no_candidate_passed_dev_validation",
            "selection_criterion": "highest_mean_dev_validation_pass_rate_then_worst_rate_then_eval_loss",
            "minimum_pass_rate_percent": float(minimum_pass_rate),
            "candidates_considered": considered,
            "selected_checkpoint": None,
        }
    selected = max(
        eligible,
        key=lambda item: (
            float(item["mean_pass_rate_percent"]),
            float(item["worst_pass_rate_percent"]),
            -float(item["eval_loss"]),
            -int(item["step"]),
        ),
    )
    return {
        "status": "selected_for_frozen_gate",
        "selection_criterion": "highest_mean_dev_validation_pass_rate_then_worst_rate_then_eval_loss",
        "minimum_pass_rate_percent": float(minimum_pass_rate),
        "candidates_considered": considered,
        "selected_checkpoint": selected["checkpoint"],
        "selected_step": selected["step"],
        "selected_eval_loss": selected["eval_loss"],
        "selected_dev_validation_pass_rate_percent": selected["mean_pass_rate_percent"],
    }

